import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments/stage61_davis_anchor_export_data_full/stage61_davis_anchor_export_manifest.csv"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage62_adapter_training_infra_v2")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage62_adapter_training_infra_v2"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.anchor_predictor import GaussianAnchorDynamicPredictor  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import (  # noqa: E402
    average,
    evaluate,
    make_tasks,
    psnr_from_mse,
    render_prediction,
    save_model,
)
from scripts.run_stage21b_residual_zero_anchor_adapter_training import (  # noqa: E402
    select_rows_balanced,
)
from scripts.run_stage21c_medium_anchor_adapter_training import (  # noqa: E402
    group_average,
)


SELECTED_ROW_FIELDS = [
    "selection", "sample", "dataset", "split", "sequence", "frame_gap", "left_index", "right_index",
    "segment_length", "middle_frame_count", "dataset_item", "anchor_mib", "gaussians_per_anchor",
]


def normalize_manifest_row(row):
    out = dict(row)
    if "sample" not in out or not out["sample"]:
        dataset = out.get("dataset", "unknown")
        split = out.get("split", "unknown")
        sequence = out.get("sequence", out.get("sample", "unknown"))
        out["sample"] = f"{dataset}/{split}/{sequence}"
    out.setdefault("dataset", out["sample"].split("/", 1)[0])
    out.setdefault("split", "unknown")
    out.setdefault("sequence", out["sample"])
    out["frame_gap"] = int(out["frame_gap"])
    out["left_index"] = int(out["left_index"])
    out["right_index"] = int(out["right_index"])
    out["segment_length"] = int(out.get("segment_length", out["right_index"] - out["left_index"]))
    out["middle_frame_count"] = int(out["middle_frame_count"])
    out["anchor_mib"] = float(out.get("anchor_mib", 0.0) or 0.0)
    out["gaussians_per_anchor"] = int(out.get("gaussians_per_anchor", 0) or 0)
    return out


def read_stage62_manifest(path, frame_gaps, splits, samples):
    gap_set = set(frame_gaps or [])
    split_set = set(splits or [])
    sample_set = set(samples or [])
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = normalize_manifest_row(raw)
            if gap_set and row["frame_gap"] not in gap_set:
                continue
            if split_set and row["split"] not in split_set:
                continue
            if sample_set and row["sample"] not in sample_set and row["sequence"] not in sample_set:
                continue
            if row["middle_frame_count"] <= 0:
                continue
            if not Path(row["dataset_item"]).exists():
                continue
            rows.append(row)
    return rows


def build_tasks(args, height, width, device):
    train_rows_all = read_stage62_manifest(args.manifest, args.frame_gaps, args.train_splits, args.train_samples)
    eval_rows_all = read_stage62_manifest(args.manifest, args.frame_gaps, args.eval_splits, args.eval_samples)
    if not train_rows_all or not eval_rows_all:
        raise RuntimeError(f"Need non-empty train/eval rows, got train={len(train_rows_all)} eval={len(eval_rows_all)}")

    selected_train_rows = []
    selected_eval_rows = []
    rows_by_gap = {}
    for gap in args.frame_gaps:
        train_gap_rows = [row for row in train_rows_all if row["frame_gap"] == gap]
        eval_gap_rows = [row for row in eval_rows_all if row["frame_gap"] == gap]
        train_sel = select_rows_balanced(train_gap_rows, args.max_train_rows_per_gap)
        eval_sel = select_rows_balanced(eval_gap_rows, args.max_eval_rows_per_gap)
        selected_train_rows.extend(train_sel)
        selected_eval_rows.extend(eval_sel)
        rows_by_gap[str(gap)] = {
            "available_train_rows": len(train_gap_rows),
            "available_eval_rows": len(eval_gap_rows),
            "selected_train_rows": len(train_sel),
            "selected_eval_rows": len(eval_sel),
            "selected_train_sequences": sorted({row["sequence"] for row in train_sel}),
            "selected_eval_sequences": sorted({row["sequence"] for row in eval_sel}),
        }

    train_tasks = make_tasks(selected_train_rows, len(selected_train_rows), args.targets_per_row, height, width, device, args.quant_bits)
    eval_tasks = make_tasks(selected_eval_rows, len(selected_eval_rows), args.targets_per_row, height, width, device, args.quant_bits)
    if not train_tasks or not eval_tasks:
        raise RuntimeError(f"Need non-empty train/eval tasks, got train={len(train_tasks)} eval={len(eval_tasks)}")
    return train_tasks, eval_tasks, selected_train_rows, selected_eval_rows, rows_by_gap, train_rows_all, eval_rows_all


def eval_summary(rows):
    model_psnr = average(rows, "model_psnr")
    linear_psnr = average(rows, "linear_psnr")
    return {
        "model_psnr_avg": model_psnr,
        "linear_psnr_avg": linear_psnr,
        "margin_over_linear_psnr": model_psnr - linear_psnr,
        "model_psnr_by_gap": group_average(rows, "frame_gap", "model_psnr"),
        "linear_psnr_by_gap": group_average(rows, "frame_gap", "linear_psnr"),
    }


def write_validation_csv(rows, path):
    fields = ["step", "model_psnr_avg", "linear_psnr_avg", "margin_over_linear_psnr", "best_so_far"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_stage62_train_csv(rows, path):
    fields = ["step", "sample", "frame_gap", "target_frame_index", "normalized_time", "rgb_mse", "rgb_psnr"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_stage62_eval_csv(rows, path):
    fields = [
        "sample", "frame_gap", "target_frame_index", "normalized_time",
        "model_mse", "model_psnr", "linear_mse", "linear_psnr",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_stage62_gap_eval_csv(initial_rows, best_rows, path):
    gaps = sorted({row["frame_gap"] for row in initial_rows} | {row["frame_gap"] for row in best_rows})
    initial_by_gap = {int(k): v for k, v in group_average(initial_rows, "frame_gap", "model_psnr").items()}
    best_by_gap = {int(k): v for k, v in group_average(best_rows, "frame_gap", "model_psnr").items()}
    linear_by_gap = {int(k): v for k, v in group_average(best_rows, "frame_gap", "linear_psnr").items()}
    fields = ["frame_gap", "initial_model_psnr", "best_model_psnr", "linear_psnr", "delta_psnr", "margin_over_linear_psnr"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for gap in gaps:
            initial = initial_by_gap[gap]
            best = best_by_gap[gap]
            linear = linear_by_gap[gap]
            writer.writerow({
                "frame_gap": gap,
                "initial_model_psnr": initial,
                "best_model_psnr": best,
                "linear_psnr": linear,
                "delta_psnr": best - initial,
                "margin_over_linear_psnr": best - linear,
            })


def write_selected_rows_csv(train_rows, eval_rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SELECTED_ROW_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for selection, rows in (("train", train_rows), ("eval", eval_rows)):
            for row in rows:
                writer.writerow({"selection": selection, **row})


def load_model_safetensors(model, path):
    state = load_file(str(path), device="cpu")
    model.load_state_dict(state, strict=True)


def save_training_state(path, model, optimizer, step, best_step, best_eval):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "best_step": best_step,
        "best_eval": best_eval,
    }, path)
    return {"path": str(path), "step": step, "best_step": best_step}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train_splits", nargs="*", default=["train"])
    parser.add_argument("--eval_splits", nargs="*", default=["val"])
    parser.add_argument("--train_samples", nargs="*", default=[])
    parser.add_argument("--eval_samples", nargs="*", default=[])
    parser.add_argument("--frame_gaps", nargs="*", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--max_train_rows_per_gap", type=int, default=2)
    parser.add_argument("--max_eval_rows_per_gap", type=int, default=1)
    parser.add_argument("--targets_per_row", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--eval_interval", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--resume_model", type=Path, default=None)
    parser.add_argument("--resume_state", type=Path, default=None)
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
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    height, width = opt.image_height, opt.image_width
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)

    train_tasks, eval_tasks, selected_train_rows, selected_eval_rows, rows_by_gap, train_rows_all, eval_rows_all = build_tasks(args, height, width, device)
    train_order = list(range(len(train_tasks)))
    rng.shuffle(train_order)

    model = GaussianAnchorDynamicPredictor(
        hidden_dim=args.hidden_dim,
        apply_output_constraints=False,
        zero_init_residual=True,
    ).to(device)
    if args.resume_model is not None:
        load_model_safetensors(model, args.resume_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    start_step = 0
    best_step = 0
    if args.resume_state is not None:
        state = torch.load(args.resume_state, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_step = int(state.get("step", 0))
        best_step = int(state.get("best_step", 0))

    initial_train_eval = evaluate(model, train_tasks, background, opt)
    initial_eval = evaluate(model, eval_tasks, background, opt)
    best_eval_rows = initial_eval
    best_eval = eval_summary(initial_eval)
    best_checkpoint = save_model(model, args.heavy_root / "stage62_best_anchor_adapter.safetensors")
    validation_log = [{"step": start_step, **best_eval, "best_so_far": True}]
    train_log = []

    model.train()
    for step in range(start_step + 1, args.steps + 1):
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

        if step % args.eval_interval == 0 or step == args.steps:
            current_eval_rows = evaluate(model, eval_tasks, background, opt)
            current_eval = eval_summary(current_eval_rows)
            is_best = current_eval["margin_over_linear_psnr"] > best_eval["margin_over_linear_psnr"]
            if is_best:
                best_eval = current_eval
                best_eval_rows = current_eval_rows
                best_step = step
                best_checkpoint = save_model(model, args.heavy_root / "stage62_best_anchor_adapter.safetensors")
            validation_row = {"step": step, **current_eval, "best_so_far": is_best}
            validation_log.append(validation_row)
            save_training_state(args.heavy_root / "stage62_latest_training_state.pt", model, optimizer, step, best_step, best_eval)
            print(json.dumps(validation_row), flush=True)
            model.train()

    final_train_eval = evaluate(model, train_tasks, background, opt)
    final_eval = evaluate(model, eval_tasks, background, opt)
    final_eval_summary = eval_summary(final_eval)
    final_checkpoint = save_model(model, args.heavy_root / "stage62_final_anchor_adapter.safetensors")
    latest_state = save_training_state(args.heavy_root / "stage62_latest_training_state.pt", model, optimizer, args.steps, best_step, best_eval)

    train_csv = args.summary_root / "stage62_train_rgb_losses.csv"
    validation_csv = args.summary_root / "stage62_validation_log.csv"
    selected_rows_csv = args.summary_root / "stage62_selected_rows.csv"
    initial_eval_csv = args.summary_root / "stage62_initial_eval.csv"
    final_eval_csv = args.summary_root / "stage62_final_eval.csv"
    best_eval_csv = args.summary_root / "stage62_best_eval.csv"
    gap_eval_csv = args.summary_root / "stage62_best_gap_eval_summary.csv"
    write_stage62_train_csv(train_log, train_csv)
    write_validation_csv(validation_log, validation_csv)
    write_selected_rows_csv(selected_train_rows, selected_eval_rows, selected_rows_csv)
    write_stage62_eval_csv(initial_eval, initial_eval_csv)
    write_stage62_eval_csv(final_eval, final_eval_csv)
    write_stage62_eval_csv(best_eval_rows, best_eval_csv)
    write_stage62_gap_eval_csv(initial_eval, best_eval_rows, gap_eval_csv)

    summary = {
        "stage": 62,
        "mode": "adapter training infra v2 on DAVIS anchors",
        "manifest": str(args.manifest),
        "train_splits": args.train_splits,
        "eval_splits": args.eval_splits,
        "frame_gaps": args.frame_gaps,
        "available_train_rows": len(train_rows_all),
        "available_eval_rows": len(eval_rows_all),
        "selected_train_row_count": len(selected_train_rows),
        "selected_eval_row_count": len(selected_eval_rows),
        "train_task_count": len(train_tasks),
        "eval_task_count": len(eval_tasks),
        "rows_by_gap": rows_by_gap,
        "steps": args.steps,
        "eval_interval": args.eval_interval,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "quant_bits": args.quant_bits,
        "targets_per_row": args.targets_per_row,
        "zero_init_residual": True,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "start_step": start_step,
        "best_step": best_step,
        "initial_train": eval_summary(initial_train_eval),
        "initial_eval": eval_summary(initial_eval),
        "final_train": eval_summary(final_train_eval),
        "final_eval": final_eval_summary,
        "best_eval": best_eval,
        "train_losses_csv": str(train_csv),
        "validation_csv": str(validation_csv),
        "selected_rows_csv": str(selected_rows_csv),
        "initial_eval_csv": str(initial_eval_csv),
        "final_eval_csv": str(final_eval_csv),
        "best_eval_csv": str(best_eval_csv),
        "best_gap_eval_csv": str(gap_eval_csv),
        "external_best_checkpoint": best_checkpoint,
        "external_final_checkpoint": final_checkpoint,
        "external_latest_training_state": latest_state,
        "notes": "Stage62 is infrastructure validation, not a medium/long training result. Inputs remain quantized keyframe Gaussian anchors plus timestamp only.",
    }
    summary_path = args.summary_root / "stage62_adapter_training_infra_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "best_step": best_step,
        "best_margin_over_linear_psnr": best_eval["margin_over_linear_psnr"],
        "final_margin_over_linear_psnr": final_eval_summary["margin_over_linear_psnr"],
        "best_checkpoint": best_checkpoint["path"],
        "final_checkpoint": final_checkpoint["path"],
        "latest_training_state": latest_state["path"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
