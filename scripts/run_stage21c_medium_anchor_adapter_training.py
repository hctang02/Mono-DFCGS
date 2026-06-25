import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments/stage6_real_anchor_dataset/stage6_real_anchor_dataset_manifest.csv"
DEFAULT_HEAVY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage21c_medium_anchor_adapter_training")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage21c_medium_anchor_adapter_training"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.anchor_predictor import GaussianAnchorDynamicPredictor  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import (  # noqa: E402
    average,
    evaluate,
    make_tasks,
    psnr_from_mse,
    read_manifest,
    render_prediction,
    save_model,
    split_rows,
    write_train_csv,
)
from scripts.run_stage21b_residual_zero_anchor_adapter_training import (  # noqa: E402
    select_rows_balanced,
    write_eval_csv,
)


def build_tasks_for_gaps(args, height, width, device):
    train_tasks = []
    eval_tasks = []
    selected_train_rows = []
    selected_eval_rows = []
    rows_by_gap = {}
    for gap in args.frame_gaps:
        rows = read_manifest(args.manifest, gap, args.samples)
        if not rows:
            raise RuntimeError(f"No Stage6 rows found for samples={args.samples} frame_gap={gap}")
        train_rows, eval_rows = split_rows(rows, args.eval_samples)
        train_sel = select_rows_balanced(train_rows, args.max_train_rows_per_gap)
        eval_sel = select_rows_balanced(eval_rows, args.max_eval_rows_per_gap)
        selected_train_rows.extend(train_sel)
        selected_eval_rows.extend(eval_sel)
        train_tasks.extend(make_tasks(train_sel, len(train_sel), args.targets_per_row, height, width, device, args.quant_bits))
        eval_tasks.extend(make_tasks(eval_sel, len(eval_sel), args.targets_per_row, height, width, device, args.quant_bits))
        rows_by_gap[str(gap)] = {
            "available_rows": len(rows),
            "selected_train_rows": len(train_sel),
            "selected_eval_rows": len(eval_sel),
            "selected_train_samples": sorted({row["sample"] for row in train_sel}),
            "selected_eval_samples": sorted({row["sample"] for row in eval_sel}),
        }
    return train_tasks, eval_tasks, selected_train_rows, selected_eval_rows, rows_by_gap


def group_average(rows, group_key, metric_key):
    groups = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row[metric_key])
    return {str(key): sum(values) / max(len(values), 1) for key, values in sorted(groups.items())}


def write_gap_eval_csv(initial_rows, final_rows, path):
    gaps = sorted({row["frame_gap"] for row in initial_rows} | {row["frame_gap"] for row in final_rows})
    initial_by_gap = {int(k): v for k, v in group_average(initial_rows, "frame_gap", "model_psnr").items()}
    final_by_gap = {int(k): v for k, v in group_average(final_rows, "frame_gap", "model_psnr").items()}
    linear_by_gap = {int(k): v for k, v in group_average(final_rows, "frame_gap", "linear_psnr").items()}
    fields = ["frame_gap", "initial_model_psnr", "final_model_psnr", "linear_psnr", "delta_psnr", "margin_over_linear_psnr"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for gap in gaps:
            initial = initial_by_gap[gap]
            final = final_by_gap[gap]
            linear = linear_by_gap[gap]
            writer.writerow({
                "frame_gap": gap,
                "initial_model_psnr": initial,
                "final_model_psnr": final,
                "linear_psnr": linear,
                "delta_psnr": final - initial,
                "margin_over_linear_psnr": final - linear,
            })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--eval_samples", nargs="*", default=["robot"])
    parser.add_argument("--frame_gaps", nargs="*", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--max_train_rows_per_gap", type=int, default=6)
    parser.add_argument("--max_eval_rows_per_gap", type=int, default=3)
    parser.add_argument("--targets_per_row", type=int, default=2)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
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
    height, width = opt.image_height, opt.image_width
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)

    train_tasks, eval_tasks, selected_train_rows, selected_eval_rows, rows_by_gap = build_tasks_for_gaps(args, height, width, device)
    model = GaussianAnchorDynamicPredictor(
        hidden_dim=args.hidden_dim,
        apply_output_constraints=False,
        zero_init_residual=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    initial_train_eval = evaluate(model, train_tasks, background, opt)
    initial_eval = evaluate(model, eval_tasks, background, opt)

    train_log = []
    model.train()
    for step in range(1, args.steps + 1):
        task = train_tasks[(step - 1) % len(train_tasks)]
        pred_rgb = render_prediction(model, task, background, opt).clamp(0.0, 1.0)
        loss = F.mse_loss(pred_rgb, task["target"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        mse = float(loss.detach().item())
        row = {
            "step": step,
            "sample": task["sample"],
            "frame_gap": task["frame_gap"],
            "target_frame_index": task["target_frame_index"],
            "normalized_time": task["normalized_time"],
            "rgb_mse": mse,
            "rgb_psnr": psnr_from_mse(mse),
        }
        train_log.append(row)
        print(json.dumps(row), flush=True)

    final_train_eval = evaluate(model, train_tasks, background, opt)
    final_eval = evaluate(model, eval_tasks, background, opt)
    train_csv = args.summary_root / "stage21c_train_rgb_losses.csv"
    initial_eval_csv = args.summary_root / "stage21c_initial_eval.csv"
    final_eval_csv = args.summary_root / "stage21c_final_eval.csv"
    gap_eval_csv = args.summary_root / "stage21c_gap_eval_summary.csv"
    write_train_csv(train_log, train_csv)
    write_eval_csv(initial_eval, initial_eval_csv)
    write_eval_csv(final_eval, final_eval_csv)
    write_gap_eval_csv(initial_eval, final_eval, gap_eval_csv)
    checkpoint_info = save_model(model, args.heavy_root / "stage21c_medium_anchor_adapter.safetensors")

    initial_eval_model_psnr = average(initial_eval, "model_psnr")
    final_eval_model_psnr = average(final_eval, "model_psnr")
    final_eval_linear_psnr = average(final_eval, "linear_psnr")
    summary = {
        "stage": "21c",
        "mode": "medium residual-zero Gaussian-anchor-only adapter training",
        "manifest": str(args.manifest),
        "samples": args.samples,
        "eval_samples": args.eval_samples,
        "frame_gaps": args.frame_gaps,
        "max_train_rows_per_gap": args.max_train_rows_per_gap,
        "max_eval_rows_per_gap": args.max_eval_rows_per_gap,
        "targets_per_row": args.targets_per_row,
        "selected_train_row_count": len(selected_train_rows),
        "selected_eval_row_count": len(selected_eval_rows),
        "selected_train_samples": sorted({row["sample"] for row in selected_train_rows}),
        "selected_eval_samples": sorted({row["sample"] for row in selected_eval_rows}),
        "rows_by_gap": rows_by_gap,
        "train_task_count": len(train_tasks),
        "eval_task_count": len(eval_tasks),
        "steps": args.steps,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "quant_bits": args.quant_bits,
        "zero_init_residual": True,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_train_model_psnr_avg": average(initial_train_eval, "model_psnr"),
        "final_train_model_psnr_avg": average(final_train_eval, "model_psnr"),
        "initial_train_linear_psnr_avg": average(initial_train_eval, "linear_psnr"),
        "final_train_linear_psnr_avg": average(final_train_eval, "linear_psnr"),
        "initial_eval_model_psnr_avg": initial_eval_model_psnr,
        "final_eval_model_psnr_avg": final_eval_model_psnr,
        "initial_eval_linear_psnr_avg": average(initial_eval, "linear_psnr"),
        "final_eval_linear_psnr_avg": final_eval_linear_psnr,
        "eval_model_psnr_delta": final_eval_model_psnr - initial_eval_model_psnr,
        "eval_margin_over_linear_psnr": final_eval_model_psnr - final_eval_linear_psnr,
        "train_losses_csv": str(train_csv),
        "initial_eval_csv": str(initial_eval_csv),
        "final_eval_csv": str(final_eval_csv),
        "gap_eval_csv": str(gap_eval_csv),
        "train_log": train_log,
        "external_adapter_checkpoint": checkpoint_info,
        "notes": "Medium-size anchor-only training across multiple GOP gaps. Inputs remain q8 keyframe anchors plus timestamp only.",
    }
    summary_path = args.summary_root / "stage21c_medium_anchor_adapter_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "train_losses_csv": str(train_csv),
        "initial_eval_csv": str(initial_eval_csv),
        "final_eval_csv": str(final_eval_csv),
        "gap_eval_csv": str(gap_eval_csv),
        "external_adapter_checkpoint": checkpoint_info["path"],
        "initial_eval_model_psnr_avg": initial_eval_model_psnr,
        "final_eval_model_psnr_avg": final_eval_model_psnr,
        "final_eval_linear_psnr_avg": final_eval_linear_psnr,
        "eval_margin_over_linear_psnr": summary["eval_margin_over_linear_psnr"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
