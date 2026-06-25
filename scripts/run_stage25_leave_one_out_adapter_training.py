import argparse
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments/stage6_real_anchor_dataset/stage6_real_anchor_dataset_manifest.csv"
DEFAULT_HEAVY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage25_leave_one_out_adapter_training")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage25_leave_one_out_adapter_training"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.anchor_predictor import GaussianAnchorDynamicPredictor  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import (  # noqa: E402
    average,
    evaluate,
    psnr_from_mse,
    render_prediction,
    save_model,
    write_train_csv,
)
from scripts.run_stage21b_residual_zero_anchor_adapter_training import write_eval_csv  # noqa: E402
from scripts.run_stage21c_medium_anchor_adapter_training import write_gap_eval_csv  # noqa: E402
from scripts.run_stage21d_validated_anchor_adapter_training import (  # noqa: E402
    build_tasks_for_gaps,
    eval_summary,
    write_validation_csv,
)


def train_fold(base_args, heldout_sample, device, opt, background):
    fold_heavy_root = base_args.heavy_root / heldout_sample
    fold_summary_root = base_args.summary_root / heldout_sample
    fold_heavy_root.mkdir(parents=True, exist_ok=True)
    fold_summary_root.mkdir(parents=True, exist_ok=True)

    fold_args = SimpleNamespace(**vars(base_args))
    fold_args.samples = list(base_args.samples)
    fold_args.eval_samples = [heldout_sample]
    fold_args.heavy_root = fold_heavy_root
    fold_args.summary_root = fold_summary_root
    rng = random.Random(base_args.seed + sum(ord(ch) for ch in heldout_sample))
    torch.manual_seed(base_args.seed + len(heldout_sample))

    train_tasks, eval_tasks, selected_train_rows, selected_eval_rows, rows_by_gap = build_tasks_for_gaps(
        fold_args, opt.image_height, opt.image_width, device
    )
    train_order = list(range(len(train_tasks)))
    rng.shuffle(train_order)

    model = GaussianAnchorDynamicPredictor(
        hidden_dim=base_args.hidden_dim,
        apply_output_constraints=False,
        zero_init_residual=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_args.lr, weight_decay=1e-4)

    initial_train_eval = evaluate(model, train_tasks, background, opt)
    initial_eval = evaluate(model, eval_tasks, background, opt)
    best_eval = eval_summary(initial_eval)
    best_eval_rows = initial_eval
    best_step = 0
    best_checkpoint = save_model(model, fold_heavy_root / "stage25_best_anchor_adapter.safetensors")
    validation_log = [{"step": 0, **best_eval, "best_so_far": True}]

    train_log = []
    model.train()
    for step in range(1, base_args.steps + 1):
        if (step - 1) % len(train_order) == 0:
            rng.shuffle(train_order)
        task = train_tasks[train_order[(step - 1) % len(train_order)]]
        pred_rgb = render_prediction(model, task, background, opt).clamp(0.0, 1.0)
        loss = F.mse_loss(pred_rgb, task["target"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        mse = float(loss.detach().item())
        train_log.append({
            "step": step,
            "sample": task["sample"],
            "frame_gap": task["frame_gap"],
            "target_frame_index": task["target_frame_index"],
            "normalized_time": task["normalized_time"],
            "rgb_mse": mse,
            "rgb_psnr": psnr_from_mse(mse),
        })

        if step % base_args.eval_interval == 0 or step == base_args.steps:
            current_eval_rows = evaluate(model, eval_tasks, background, opt)
            current_eval = eval_summary(current_eval_rows)
            is_best = current_eval["margin_over_linear_psnr"] > best_eval["margin_over_linear_psnr"]
            if is_best:
                best_eval = current_eval
                best_eval_rows = current_eval_rows
                best_step = step
                best_checkpoint = save_model(model, fold_heavy_root / "stage25_best_anchor_adapter.safetensors")
            validation_row = {"step": step, **current_eval, "best_so_far": is_best}
            validation_log.append(validation_row)
            print(json.dumps({"heldout": heldout_sample, **validation_row}), flush=True)
            model.train()

    final_train_eval = evaluate(model, train_tasks, background, opt)
    final_eval = evaluate(model, eval_tasks, background, opt)
    final_checkpoint = save_model(model, fold_heavy_root / "stage25_final_anchor_adapter.safetensors")

    train_csv = fold_summary_root / "stage25_train_rgb_losses.csv"
    validation_csv = fold_summary_root / "stage25_validation_log.csv"
    initial_eval_csv = fold_summary_root / "stage25_initial_eval.csv"
    final_eval_csv = fold_summary_root / "stage25_final_eval.csv"
    best_eval_csv = fold_summary_root / "stage25_best_eval.csv"
    gap_eval_csv = fold_summary_root / "stage25_best_gap_eval_summary.csv"
    write_train_csv(train_log, train_csv)
    write_validation_csv(validation_log, validation_csv)
    write_eval_csv(initial_eval, initial_eval_csv)
    write_eval_csv(final_eval, final_eval_csv)
    write_eval_csv(best_eval_rows, best_eval_csv)
    write_gap_eval_csv(initial_eval, best_eval_rows, gap_eval_csv)

    final_eval_summary = eval_summary(final_eval)
    fold_summary = {
        "heldout_sample": heldout_sample,
        "train_samples": sorted(set(base_args.samples) - {heldout_sample}),
        "eval_samples": [heldout_sample],
        "frame_gaps": base_args.frame_gaps,
        "selected_train_row_count": len(selected_train_rows),
        "selected_eval_row_count": len(selected_eval_rows),
        "selected_train_samples": sorted({row["sample"] for row in selected_train_rows}),
        "selected_eval_samples": sorted({row["sample"] for row in selected_eval_rows}),
        "rows_by_gap": rows_by_gap,
        "train_task_count": len(train_tasks),
        "eval_task_count": len(eval_tasks),
        "steps": base_args.steps,
        "eval_interval": base_args.eval_interval,
        "hidden_dim": base_args.hidden_dim,
        "lr": base_args.lr,
        "quant_bits": base_args.quant_bits,
        "initial_train_model_psnr_avg": average(initial_train_eval, "model_psnr"),
        "final_train_model_psnr_avg": average(final_train_eval, "model_psnr"),
        "initial_eval": eval_summary(initial_eval),
        "final_eval": final_eval_summary,
        "best_step": best_step,
        "best_eval": best_eval,
        "train_losses_csv": str(train_csv),
        "validation_csv": str(validation_csv),
        "initial_eval_csv": str(initial_eval_csv),
        "final_eval_csv": str(final_eval_csv),
        "best_eval_csv": str(best_eval_csv),
        "best_gap_eval_csv": str(gap_eval_csv),
        "external_best_checkpoint": best_checkpoint,
        "external_final_checkpoint": final_checkpoint,
    }
    fold_summary_path = fold_summary_root / "stage25_fold_summary.json"
    fold_summary_path.write_text(json.dumps(fold_summary, indent=2), encoding="utf-8")
    return fold_summary


def write_aggregate_csv(rows, path):
    fields = [
        "heldout_sample",
        "best_step",
        "initial_margin_over_linear_psnr",
        "best_margin_over_linear_psnr",
        "final_margin_over_linear_psnr",
        "train_task_count",
        "eval_task_count",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--frame_gaps", nargs="*", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--max_train_rows_per_gap", type=int, default=12)
    parser.add_argument("--max_eval_rows_per_gap", type=int, default=5)
    parser.add_argument("--targets_per_row", type=int, default=2)
    parser.add_argument("--steps", type=int, default=384)
    parser.add_argument("--eval_interval", type=int, default=96)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)

    fold_summaries = []
    aggregate_rows = []
    for heldout in args.samples:
        print(f"=== Stage25 heldout={heldout} ===", flush=True)
        fold = train_fold(args, heldout, device, opt, background)
        fold_summaries.append(fold)
        aggregate_rows.append({
            "heldout_sample": heldout,
            "best_step": fold["best_step"],
            "initial_margin_over_linear_psnr": fold["initial_eval"]["margin_over_linear_psnr"],
            "best_margin_over_linear_psnr": fold["best_eval"]["margin_over_linear_psnr"],
            "final_margin_over_linear_psnr": fold["final_eval"]["margin_over_linear_psnr"],
            "train_task_count": fold["train_task_count"],
            "eval_task_count": fold["eval_task_count"],
        })

    aggregate_csv = args.summary_root / "stage25_leave_one_out_adapter_training_aggregate.csv"
    write_aggregate_csv(aggregate_rows, aggregate_csv)
    mean_best_margin = sum(row["best_margin_over_linear_psnr"] for row in aggregate_rows) / len(aggregate_rows)
    min_best_margin = min(row["best_margin_over_linear_psnr"] for row in aggregate_rows)
    summary = {
        "stage": 25,
        "mode": "leave-one-sample-out validated anchor adapter training",
        "samples": args.samples,
        "frame_gaps": args.frame_gaps,
        "max_train_rows_per_gap": args.max_train_rows_per_gap,
        "max_eval_rows_per_gap": args.max_eval_rows_per_gap,
        "targets_per_row": args.targets_per_row,
        "steps": args.steps,
        "eval_interval": args.eval_interval,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "quant_bits": args.quant_bits,
        "aggregate_csv": str(aggregate_csv),
        "aggregate": aggregate_rows,
        "folds": fold_summaries,
        "mean_best_margin_over_linear_psnr": mean_best_margin,
        "min_best_margin_over_linear_psnr": min_best_margin,
        "notes": "Each fold trains on three samples and validates on the held-out sample. Inputs remain q8 keyframe anchors plus timestamp only.",
    }
    summary_path = args.summary_root / "stage25_leave_one_out_adapter_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "aggregate_csv": str(aggregate_csv),
        "mean_best_margin_over_linear_psnr": mean_best_margin,
        "min_best_margin_over_linear_psnr": min_best_margin,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
