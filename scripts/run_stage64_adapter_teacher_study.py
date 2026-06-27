import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments/stage61_davis_anchor_export_data_full/stage61_davis_anchor_export_manifest.csv"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage64_adapter_teacher_study")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage64_adapter_teacher_study"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.anchor_predictor import GaussianAnchorDynamicPredictor  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import (  # noqa: E402
    anchor_to_device,
    average,
    evaluate,
    linear_anchor,
    load_rgb,
    maybe_quantize_anchor,
    psnr_from_mse,
    render_prediction,
    save_model,
)
from scripts.run_stage21b_residual_zero_anchor_adapter_training import select_rows_balanced  # noqa: E402
from scripts.run_stage21c_medium_anchor_adapter_training import group_average  # noqa: E402
from scripts.run_stage62_adapter_training_infra_v2 import normalize_manifest_row, read_stage62_manifest  # noqa: E402


VARIANTS = [
    {"name": "rgb_h128", "loss": "rgb", "hidden_dim": 128, "lr": 8e-6},
    {"name": "rgb_h256", "loss": "rgb", "hidden_dim": 256, "lr": 8e-6},
    {"name": "teacher_h128", "loss": "teacher", "hidden_dim": 128, "lr": 2e-4},
    {"name": "teacher_h256", "loss": "teacher", "hidden_dim": 256, "lr": 2e-4},
]


def build_dense_index(rows):
    dense = {}
    for row in rows:
        if int(row["frame_gap"]) != 1:
            continue
        key_base = (row["split"], row["sequence"])
        dense[(key_base[0], key_base[1], int(row["left_index"]))] = (row["dataset_item"], "left_anchor")
        dense[(key_base[0], key_base[1], int(row["right_index"]))] = (row["dataset_item"], "right_anchor")
    return dense


def read_dense_manifest(path, splits, samples):
    split_set = set(splits or [])
    sample_set = set(samples or [])
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = normalize_manifest_row(raw)
            if row["frame_gap"] != 1:
                continue
            if split_set and row["split"] not in split_set:
                continue
            if sample_set and row["sample"] not in sample_set and row["sequence"] not in sample_set:
                continue
            if not Path(row["dataset_item"]).exists():
                continue
            rows.append(row)
    return rows


def select_rows(args):
    train_rows_all = read_stage62_manifest(args.manifest, args.frame_gaps, args.train_splits, args.train_samples)
    eval_rows_all = read_stage62_manifest(args.manifest, args.frame_gaps, args.eval_splits, args.eval_samples)
    selected_train_rows = []
    selected_eval_rows = []
    rows_by_gap = {}
    for gap in args.frame_gaps:
        train_gap = [row for row in train_rows_all if row["frame_gap"] == gap]
        eval_gap = [row for row in eval_rows_all if row["frame_gap"] == gap]
        train_sel = select_rows_balanced(train_gap, args.max_train_rows_per_gap)
        eval_sel = select_rows_balanced(eval_gap, args.max_eval_rows_per_gap)
        selected_train_rows.extend(train_sel)
        selected_eval_rows.extend(eval_sel)
        rows_by_gap[str(gap)] = {
            "available_train_rows": len(train_gap),
            "available_eval_rows": len(eval_gap),
            "selected_train_rows": len(train_sel),
            "selected_eval_rows": len(eval_sel),
        }
    return train_rows_all, eval_rows_all, selected_train_rows, selected_eval_rows, rows_by_gap


def selected_mids(item, targets_per_row):
    mids = item["intermediate_frames"]
    if targets_per_row >= len(mids):
        return mids
    if targets_per_row == 1:
        return [mids[len(mids) // 2]]
    idxs = np.linspace(0, len(mids) - 1, targets_per_row).round().astype(int).tolist()
    return [mids[idx] for idx in idxs]


def load_teacher_anchor(dense_index, split, sequence, frame_index, device):
    key = (split, sequence, frame_index)
    if key not in dense_index:
        raise KeyError(f"Missing dense teacher anchor for {key}")
    item_path, anchor_key = dense_index[key]
    item = torch.load(item_path, map_location="cpu", weights_only=True)
    return anchor_to_device(item[anchor_key], device)


def make_teacher_tasks(rows, dense_index, targets_per_row, height, width, device, quant_bits):
    tasks = []
    for row in rows:
        item = torch.load(row["dataset_item"], map_location="cpu", weights_only=True)
        left = maybe_quantize_anchor(anchor_to_device(item["left_anchor"], device), quant_bits)
        right = maybe_quantize_anchor(anchor_to_device(item["right_anchor"], device), quant_bits)
        for mid in selected_mids(item, targets_per_row):
            teacher = load_teacher_anchor(dense_index, row["split"], row["sequence"], int(mid["frame_index"]), device)
            tasks.append({
                "sample": row["sample"],
                "dataset": row["dataset"],
                "split": row["split"],
                "sequence": row["sequence"],
                "frame_gap": item["frame_gap"],
                "left_index": item["left_index"],
                "right_index": item["right_index"],
                "target_frame_index": int(mid["frame_index"]),
                "normalized_time": float(mid["normalized_time"]),
                "target_rgb_path": mid["rgb_path"],
                "left": left,
                "right": right,
                "teacher": teacher,
                "target": load_rgb(mid["rgb_path"], height, width, device),
            })
    return tasks


def anchor_mse(model, task):
    t = torch.tensor([task["normalized_time"]], dtype=torch.float32, device=next(model.parameters()).device)
    pred_anchor = model(task["left"], task["right"], t, apply_output_constraints=False)
    return F.mse_loss(flatten_static_anchor(pred_anchor), flatten_static_anchor(task["teacher"]))


def evaluate_teacher(model, tasks, background, opt):
    model.eval()
    rows = evaluate(model, tasks, background, opt)
    with torch.no_grad():
        for row, task in zip(rows, tasks):
            teacher_mse = float(anchor_mse(model, task).item())
            linear_mse = float(F.mse_loss(flatten_static_anchor(linear_anchor(task["left"], task["right"], task["normalized_time"])), flatten_static_anchor(task["teacher"])).item())
            row["teacher_mse"] = teacher_mse
            row["linear_teacher_mse"] = linear_mse
            row["teacher_mse_delta_vs_linear"] = teacher_mse - linear_mse
    return rows


def eval_summary(rows):
    model_psnr = average(rows, "model_psnr")
    linear_psnr = average(rows, "linear_psnr")
    teacher_mse = average(rows, "teacher_mse") if "teacher_mse" in rows[0] else 0.0
    linear_teacher_mse = average(rows, "linear_teacher_mse") if "linear_teacher_mse" in rows[0] else 0.0
    return {
        "model_psnr_avg": model_psnr,
        "linear_psnr_avg": linear_psnr,
        "margin_over_linear_psnr": model_psnr - linear_psnr,
        "teacher_mse_avg": teacher_mse,
        "linear_teacher_mse_avg": linear_teacher_mse,
        "teacher_mse_delta_vs_linear": teacher_mse - linear_teacher_mse,
        "model_psnr_by_gap": group_average(rows, "frame_gap", "model_psnr"),
        "linear_psnr_by_gap": group_average(rows, "frame_gap", "linear_psnr"),
    }


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def train_variant(variant, args, train_tasks, eval_tasks, background, opt, device):
    variant_root = args.heavy_root / variant["name"]
    variant_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    order = list(range(len(train_tasks)))
    rng.shuffle(order)
    model = GaussianAnchorDynamicPredictor(
        hidden_dim=variant["hidden_dim"],
        apply_output_constraints=False,
        zero_init_residual=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=variant["lr"], weight_decay=1e-4)

    initial_eval_rows = evaluate_teacher(model, eval_tasks, background, opt)
    best_eval = eval_summary(initial_eval_rows)
    best_rows = initial_eval_rows
    best_step = 0
    best_checkpoint = save_model(model, variant_root / "best_adapter.safetensors")
    train_log = []
    validation_log = [{"variant": variant["name"], "step": 0, **best_eval, "best_so_far": True}]
    model.train()
    for step in range(1, args.steps + 1):
        if (step - 1) % len(order) == 0:
            rng.shuffle(order)
        task = train_tasks[order[(step - 1) % len(order)]]
        if variant["loss"] == "teacher":
            loss = anchor_mse(model, task)
        else:
            pred_rgb = render_prediction(model, task, background, opt).clamp(0.0, 1.0)
            loss = F.mse_loss(pred_rgb, task["target"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_log.append({
            "variant": variant["name"],
            "step": step,
            "loss_type": variant["loss"],
            "loss": float(loss.detach().item()),
            "sample": task["sample"],
            "frame_gap": task["frame_gap"],
            "target_frame_index": task["target_frame_index"],
            "normalized_time": task["normalized_time"],
        })
        if step % args.eval_interval == 0 or step == args.steps:
            current_rows = evaluate_teacher(model, eval_tasks, background, opt)
            current_eval = eval_summary(current_rows)
            if variant["loss"] == "teacher":
                is_best = current_eval["teacher_mse_avg"] < best_eval["teacher_mse_avg"]
            else:
                is_best = current_eval["margin_over_linear_psnr"] > best_eval["margin_over_linear_psnr"]
            if is_best:
                best_eval = current_eval
                best_rows = current_rows
                best_step = step
                best_checkpoint = save_model(model, variant_root / "best_adapter.safetensors")
            validation_log.append({"variant": variant["name"], "step": step, **current_eval, "best_so_far": is_best})
            print(json.dumps(validation_log[-1]), flush=True)
            model.train()
    final_rows = evaluate_teacher(model, eval_tasks, background, opt)
    final_eval = eval_summary(final_rows)
    final_checkpoint = save_model(model, variant_root / "final_adapter.safetensors")
    return {
        "variant": variant,
        "initial_eval": eval_summary(initial_eval_rows),
        "final_eval": final_eval,
        "best_eval": best_eval,
        "best_step": best_step,
        "best_checkpoint": best_checkpoint,
        "final_checkpoint": final_checkpoint,
        "train_log": train_log,
        "validation_log": validation_log,
        "best_rows": best_rows,
        "final_rows": final_rows,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train_splits", nargs="*", default=["train"])
    parser.add_argument("--eval_splits", nargs="*", default=["val"])
    parser.add_argument("--train_samples", nargs="*", default=[])
    parser.add_argument("--eval_samples", nargs="*", default=[])
    parser.add_argument("--frame_gaps", nargs="*", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--max_train_rows_per_gap", type=int, default=4)
    parser.add_argument("--max_eval_rows_per_gap", type=int, default=2)
    parser.add_argument("--targets_per_row", type=int, default=1)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--eval_interval", type=int, default=24)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--variants", nargs="*", default=[variant["name"] for variant in VARIANTS])
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)

    train_rows_all, eval_rows_all, train_rows, eval_rows, rows_by_gap = select_rows(args)
    dense_rows = read_dense_manifest(args.manifest, args.train_splits + args.eval_splits, args.train_samples + args.eval_samples)
    dense_index = build_dense_index(dense_rows)
    train_tasks = make_teacher_tasks(train_rows, dense_index, args.targets_per_row, opt.image_height, opt.image_width, device, args.quant_bits)
    eval_tasks = make_teacher_tasks(eval_rows, dense_index, args.targets_per_row, opt.image_height, opt.image_width, device, args.quant_bits)
    if not train_tasks or not eval_tasks:
        raise RuntimeError(f"Need non-empty train/eval tasks, got train={len(train_tasks)} eval={len(eval_tasks)}")

    variants = [variant for variant in VARIANTS if variant["name"] in set(args.variants)]
    all_train_logs = []
    all_validation_logs = []
    variant_rows = []
    best_eval_rows = []
    results = []
    for variant in variants:
        result = train_variant(variant, args, train_tasks, eval_tasks, background, opt, device)
        results.append(result)
        all_train_logs.extend(result["train_log"])
        all_validation_logs.extend(result["validation_log"])
        best_eval_rows.extend({"variant": variant["name"], **row} for row in result["best_rows"])
        variant_rows.append({
            "variant": variant["name"],
            "loss": variant["loss"],
            "hidden_dim": variant["hidden_dim"],
            "lr": variant["lr"],
            "parameter_count": result["parameter_count"],
            "best_step": result["best_step"],
            "initial_margin_over_linear_psnr": result["initial_eval"]["margin_over_linear_psnr"],
            "final_margin_over_linear_psnr": result["final_eval"]["margin_over_linear_psnr"],
            "best_margin_over_linear_psnr": result["best_eval"]["margin_over_linear_psnr"],
            "initial_teacher_mse": result["initial_eval"]["teacher_mse_avg"],
            "final_teacher_mse": result["final_eval"]["teacher_mse_avg"],
            "best_teacher_mse": result["best_eval"]["teacher_mse_avg"],
            "linear_teacher_mse": result["best_eval"]["linear_teacher_mse_avg"],
            "best_checkpoint": result["best_checkpoint"]["path"],
            "final_checkpoint": result["final_checkpoint"]["path"],
        })

    variant_csv = args.summary_root / "stage64_variant_summary.csv"
    train_csv = args.summary_root / "stage64_train_log.csv"
    validation_csv = args.summary_root / "stage64_validation_log.csv"
    best_eval_csv = args.summary_root / "stage64_best_eval_rows.csv"
    write_csv(variant_rows, variant_csv, list(variant_rows[0].keys()))
    write_csv(all_train_logs, train_csv, ["variant", "step", "loss_type", "loss", "sample", "frame_gap", "target_frame_index", "normalized_time"])
    write_csv(all_validation_logs, validation_csv, list(all_validation_logs[0].keys()))
    write_csv(best_eval_rows, best_eval_csv, list(best_eval_rows[0].keys()))

    summary = {
        "stage": 64,
        "mode": "adapter architecture and dense-anchor teacher study",
        "manifest": str(args.manifest),
        "train_splits": args.train_splits,
        "eval_splits": args.eval_splits,
        "frame_gaps": args.frame_gaps,
        "available_train_rows": len(train_rows_all),
        "available_eval_rows": len(eval_rows_all),
        "selected_train_rows": len(train_rows),
        "selected_eval_rows": len(eval_rows),
        "train_tasks": len(train_tasks),
        "eval_tasks": len(eval_tasks),
        "steps": args.steps,
        "eval_interval": args.eval_interval,
        "quant_bits": args.quant_bits,
        "targets_per_row": args.targets_per_row,
        "rows_by_gap": rows_by_gap,
        "variant_summary_csv": str(variant_csv),
        "train_log_csv": str(train_csv),
        "validation_log_csv": str(validation_csv),
        "best_eval_csv": str(best_eval_csv),
        "variant_rows": variant_rows,
        "notes": "Teacher variants use dense gap1 anchors as offline training targets only. Test-time inputs remain q8 endpoint Gaussian anchors plus timestamp.",
    }
    summary_path = args.summary_root / "stage64_adapter_teacher_study_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    best_by_psnr = max(variant_rows, key=lambda row: row["best_margin_over_linear_psnr"])
    best_by_teacher = min(variant_rows, key=lambda row: row["best_teacher_mse"])
    print(json.dumps({
        "summary": str(summary_path),
        "variant_csv": str(variant_csv),
        "best_by_psnr": best_by_psnr,
        "best_by_teacher": best_by_teacher,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
