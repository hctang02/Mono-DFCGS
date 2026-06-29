import argparse
import csv
import json
import random
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_MANIFEST = REPO_ROOT / "experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_tasks.csv"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage145_large_scale_adapter_training_launch")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage145_large_scale_adapter_training_launch"
DEFAULT_INIT_CHECKPOINT = Path(
    "/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors"
)


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.anchor_predictor import GaussianAnchorDynamicPredictor  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import (  # noqa: E402
    anchor_to_device,
    linear_anchor,
    load_rgb,
    maybe_quantize_anchor,
    psnr_from_mse,
    render_anchor,
    render_prediction,
    save_model,
)
from scripts.run_stage80_adapter_training_smoke import (  # noqa: E402
    EVAL_FIELDS,
    make_tasks,
    parse_gap_weights,
    read_task_rows,
    safetensors_device,
    select_balanced,
    selection_score,
    summarize_eval,
    write_csv,
)


TRAIN_FIELDS = [
    "step",
    "task_id",
    "sequence",
    "codec",
    "reference_gap",
    "target_index",
    "normalized_time",
    "loss",
    "loss_weight",
    "weighted_loss",
    "loss_psnr",
]

VALIDATION_FIELDS = [
    "step",
    "model_psnr_avg",
    "linear_psnr_avg",
    "margin_over_linear_psnr",
    "min_gap_margin_over_linear_psnr",
    "selection_score",
    "best_so_far",
    "gap4_model_psnr",
    "gap4_linear_psnr",
    "gap4_margin_over_linear_psnr",
    "gap8_model_psnr",
    "gap8_linear_psnr",
    "gap8_margin_over_linear_psnr",
]

SELECTED_ROW_FIELDS = [
    "selection",
    "task_id",
    "task_split",
    "sequence",
    "codec",
    "bits",
    "reference_gap",
    "left_index",
    "right_index",
    "target_index",
    "normalized_time",
    "target_rgb_path",
    "left_anchor_source_item",
    "left_anchor_source_side",
    "right_anchor_source_item",
    "right_anchor_source_side",
]


def lru_get(cache, key):
    if cache is None or key not in cache:
        return None
    value = cache.pop(key)
    cache[key] = value
    return value


def lru_put(cache, key, value, max_items):
    if cache is None:
        return
    cache[key] = value
    while len(cache) > max_items:
        cache.popitem(last=False)


def make_lru_cache(max_items):
    if max_items <= 0:
        return None
    return OrderedDict()


def load_source_anchor_lru(row, prefix, device, cache, max_items):
    source_item = row[f"{prefix}_anchor_source_item"]
    source_side = row[f"{prefix}_anchor_source_side"]
    key = (source_item, source_side, row["bits"], str(device))
    cached = lru_get(cache, key)
    if cached is not None:
        return cached
    item = torch.load(source_item, map_location="cpu", weights_only=True)
    if source_side not in item:
        raise KeyError(f"Missing {source_side} in {source_item}")
    anchor = maybe_quantize_anchor(anchor_to_device(item[source_side], device), row["bits"])
    lru_put(cache, key, anchor, max_items)
    return anchor


def load_target_rgb_lru(path, height, width, device, cache, max_items):
    key = (path, height, width, str(device))
    cached = lru_get(cache, key)
    if cached is not None:
        return cached
    rgb = load_rgb(path, height, width, device)
    lru_put(cache, key, rgb, max_items)
    return rgb


def make_lazy_task(row, height, width, device, anchor_cache, rgb_cache, max_anchor_cache, max_rgb_cache):
    return {
        "task_id": row["task_id"],
        "task_split": row["task_split"],
        "sequence": row["sequence"],
        "codec": row["codec"],
        "bits": row["bits"],
        "frame_gap": row["reference_gap"],
        "reference_gap": row["reference_gap"],
        "left_index": row["left_index"],
        "right_index": row["right_index"],
        "target_frame_index": row["target_index"],
        "target_index": row["target_index"],
        "normalized_time": row["normalized_time"],
        "target_rgb_path": row["target_rgb_path"],
        "left": load_source_anchor_lru(row, "left", device, anchor_cache, max_anchor_cache),
        "right": load_source_anchor_lru(row, "right", device, anchor_cache, max_anchor_cache),
        "target": load_target_rgb_lru(row["target_rgb_path"], height, width, device, rgb_cache, max_rgb_cache),
    }


def group_count(rows, key):
    counts = defaultdict(int)
    for row in rows:
        counts[str(row[key])] += 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def row_group_count(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[f"{row['codec']}_gap{row['reference_gap']}"] += 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def flatten_validation_row(step, eval_summary, score, is_best):
    model_by_gap = eval_summary.get("model_psnr_by_gap", {})
    linear_by_gap = eval_summary.get("linear_psnr_by_gap", {})
    margin_by_gap = eval_summary.get("margin_over_linear_psnr_by_gap", {})
    return {
        "step": step,
        "model_psnr_avg": eval_summary["model_psnr_avg"],
        "linear_psnr_avg": eval_summary["linear_psnr_avg"],
        "margin_over_linear_psnr": eval_summary["margin_over_linear_psnr"],
        "min_gap_margin_over_linear_psnr": eval_summary["min_gap_margin_over_linear_psnr"],
        "selection_score": score,
        "best_so_far": is_best,
        "gap4_model_psnr": model_by_gap.get("4"),
        "gap4_linear_psnr": linear_by_gap.get("4"),
        "gap4_margin_over_linear_psnr": margin_by_gap.get("4"),
        "gap8_model_psnr": model_by_gap.get("8"),
        "gap8_linear_psnr": linear_by_gap.get("8"),
        "gap8_margin_over_linear_psnr": margin_by_gap.get("8"),
    }


def evaluate_model(model, tasks, background, opt, step):
    model.eval()
    rows = []
    with torch.no_grad():
        for task in tasks:
            pred_rgb = render_prediction(model, task, background, opt).clamp(0.0, 1.0)
            model_mse = float(F.mse_loss(pred_rgb, task["target"]).item())
            linear_rgb = render_anchor(linear_anchor(task["left"], task["right"], task["normalized_time"]), background, opt).clamp(0.0, 1.0)
            linear_mse = float(F.mse_loss(linear_rgb, task["target"]).item())
            model_psnr = psnr_from_mse(model_mse)
            linear_psnr = psnr_from_mse(linear_mse)
            rows.append({
                "step": step,
                "task_id": task["task_id"],
                "task_split": task["task_split"],
                "sequence": task["sequence"],
                "codec": task["codec"],
                "reference_gap": task["reference_gap"],
                "left_index": task["left_index"],
                "right_index": task["right_index"],
                "target_index": task["target_index"],
                "normalized_time": task["normalized_time"],
                "model_mse": model_mse,
                "model_psnr": model_psnr,
                "linear_mse": linear_mse,
                "linear_psnr": linear_psnr,
                "margin_over_linear_psnr": model_psnr - linear_psnr,
            })
    return rows


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


def load_model_state(model, path, device):
    state = load_file(str(path), device=safetensors_device(device))
    model.load_state_dict(state, strict=True)
    return {"path": str(path), "tensor_count": len(state), "parameter_count": sum(t.numel() for t in state.values())}


def write_selected_rows(rows, selection, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SELECTED_ROW_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"selection": selection, **row})


def write_report(summary, path):
    initial = summary["initial_eval"]
    best = summary["best_eval"]
    final = summary["final_eval"]
    lines = [
        "# Stage145 Large-Scale Adapter Training Launch",
        "",
        "## Configuration",
        "",
        f"- task manifest: `{summary['task_manifest']}`",
        f"- codecs: `{summary['codecs']}`",
        f"- gaps: `{summary['gaps']}`",
        f"- available train rows: `{summary['available_train_rows']}`",
        f"- selected train rows: `{summary['selected_train_rows']}`",
        f"- selected eval rows: `{summary['selected_eval_rows']}`",
        f"- steps: `{summary['steps']}`",
        f"- eval interval: `{summary['eval_interval']}`",
        f"- init checkpoint: `{summary['init_checkpoint']['path'] if summary.get('init_checkpoint') else None}`",
        f"- heavy root: `{summary['heavy_root']}`",
        f"- gap loss weights: `{summary['gap_loss_weights']}`",
        f"- best metric: `{summary['best_metric']}`",
        "",
        "## Evaluation",
        "",
        "| checkpoint | step | model PSNR | linear PSNR | margin | min gap margin |",
        "|---|---:|---:|---:|---:|---:|",
        f"| initial | 0 | {initial['model_psnr_avg']} | {initial['linear_psnr_avg']} | {initial['margin_over_linear_psnr']} | {initial['min_gap_margin_over_linear_psnr']} |",
        f"| best | {summary['best_step']} | {best['model_psnr_avg']} | {best['linear_psnr_avg']} | {best['margin_over_linear_psnr']} | {best['min_gap_margin_over_linear_psnr']} |",
        f"| final | {summary['steps']} | {final['model_psnr_avg']} | {final['linear_psnr_avg']} | {final['margin_over_linear_psnr']} | {final['min_gap_margin_over_linear_psnr']} |",
        "",
        "## Gap Evaluation",
        "",
        "| gap | initial model | best model | final model | best margin |",
        "|---:|---:|---:|---:|---:|",
    ]
    gap_keys = sorted(best.get("model_psnr_by_gap", {}), key=lambda item: int(item))
    for gap in gap_keys:
        lines.append(
            f"| {gap} | {initial['model_psnr_by_gap'].get(gap)} | {best['model_psnr_by_gap'].get(gap)} | {final['model_psnr_by_gap'].get(gap)} | {best['margin_over_linear_psnr_by_gap'].get(gap)} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- This stage starts model-side quality rescue; it is not a q-bit tuning stage.",
        "- Target RGB is used only for offline training supervision.",
        "- Decoder-side inputs remain endpoint Gaussian anchors plus normalized time.",
        "- Heavy checkpoints and optimizer state are outside git.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=145)
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--codecs", nargs="+", default=["q12"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--train_sequences", nargs="*", default=[])
    parser.add_argument("--eval_sequences", nargs="*", default=[])
    parser.add_argument("--max_train_tasks", type=int, default=0)
    parser.add_argument("--max_eval_tasks", type=int, default=96)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=4e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--gap_loss_weights", nargs="*", default=["4:2.0", "8:1.5"])
    parser.add_argument("--best_metric", choices=["mean_margin", "min_gap_margin", "protected_gap4_margin"], default="min_gap_margin")
    parser.add_argument("--gap4_penalty", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--init_checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--resume_state", type=Path, default=None)
    parser.add_argument("--train_anchor_cache", type=int, default=96)
    parser.add_argument("--train_rgb_cache", type=int, default=32)
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
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)

    train_rows_all = read_task_rows(args.task_manifest, "train", args.codecs, args.gaps, args.train_sequences)
    eval_rows_all = read_task_rows(args.task_manifest, "eval", args.codecs, args.gaps, args.eval_sequences)
    train_rows = select_balanced(train_rows_all, args.max_train_tasks, args.seed)
    eval_rows = select_balanced(eval_rows_all, args.max_eval_tasks, args.seed + 1)
    if not train_rows or not eval_rows:
        raise RuntimeError(f"Need non-empty train/eval rows, got train={len(train_rows)} eval={len(eval_rows)}")

    eval_tasks = make_tasks(eval_rows, opt.image_height, opt.image_width, device, cache_anchors=True, cache_rgb=True)
    train_order = list(range(len(train_rows)))
    rng.shuffle(train_order)
    gap_loss_weights = parse_gap_weights(args.gap_loss_weights)

    model = GaussianAnchorDynamicPredictor(
        hidden_dim=args.hidden_dim,
        apply_output_constraints=False,
        zero_init_residual=True,
    ).to(device)
    init_checkpoint = None
    if args.init_checkpoint is not None:
        init_checkpoint = load_model_state(model, args.init_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    best_step = 0
    best_eval = None
    if args.resume_state is not None:
        state = torch.load(args.resume_state, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_step = int(state.get("step", 0))
        best_step = int(state.get("best_step", 0))
        best_eval = state.get("best_eval")

    initial_rows = evaluate_model(model, eval_tasks, background, opt, step=start_step)
    initial_eval = summarize_eval(initial_rows)
    if best_eval is None:
        best_eval = initial_eval
    best_rows = initial_rows
    best_score = selection_score(best_eval, args)
    best_checkpoint = save_model(model, args.heavy_root / "best_adapter.safetensors")
    validation_log = [flatten_validation_row(start_step, initial_eval, selection_score(initial_eval, args), True)]
    train_log = []

    anchor_cache = make_lru_cache(args.train_anchor_cache)
    rgb_cache = make_lru_cache(args.train_rgb_cache)
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        if (step - 1) % len(train_order) == 0:
            rng.shuffle(train_order)
        row = train_rows[train_order[(step - 1) % len(train_order)]]
        task = make_lazy_task(
            row,
            opt.image_height,
            opt.image_width,
            device,
            anchor_cache,
            rgb_cache,
            args.train_anchor_cache,
            args.train_rgb_cache,
        )
        pred_rgb = render_prediction(model, task, background, opt).clamp(0.0, 1.0)
        raw_loss = F.mse_loss(pred_rgb, task["target"])
        loss_weight = gap_loss_weights.get(int(task["reference_gap"]), 1.0)
        loss = raw_loss * loss_weight
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        raw_loss_value = float(raw_loss.detach().item())
        weighted_loss_value = float(loss.detach().item())
        train_row = {
            "step": step,
            "task_id": task["task_id"],
            "sequence": task["sequence"],
            "codec": task["codec"],
            "reference_gap": task["reference_gap"],
            "target_index": task["target_index"],
            "normalized_time": task["normalized_time"],
            "loss": raw_loss_value,
            "loss_weight": loss_weight,
            "weighted_loss": weighted_loss_value,
            "loss_psnr": psnr_from_mse(raw_loss_value),
        }
        train_log.append(train_row)
        if step == 1 or step % args.log_interval == 0:
            print(json.dumps(train_row), flush=True)

        if step % args.eval_interval == 0 or step == args.steps:
            current_rows = evaluate_model(model, eval_tasks, background, opt, step=step)
            current_eval = summarize_eval(current_rows)
            current_score = selection_score(current_eval, args)
            is_best = current_score > best_score
            if is_best:
                best_rows = current_rows
                best_eval = current_eval
                best_score = current_score
                best_step = step
                best_checkpoint = save_model(model, args.heavy_root / "best_adapter.safetensors")
            validation_row = flatten_validation_row(step, current_eval, current_score, is_best)
            validation_log.append(validation_row)
            save_training_state(args.heavy_root / "latest_training_state.pt", model, optimizer, step, best_step, best_eval)
            print(json.dumps(validation_row), flush=True)
            model.train()

    final_rows = evaluate_model(model, eval_tasks, background, opt, step=args.steps)
    final_eval = summarize_eval(final_rows)
    final_checkpoint = save_model(model, args.heavy_root / "final_adapter.safetensors")
    latest_state = save_training_state(args.heavy_root / "latest_training_state.pt", model, optimizer, args.steps, best_step, best_eval)

    train_csv = args.summary_root / "stage145_train_log.csv"
    validation_csv = args.summary_root / "stage145_validation_log.csv"
    initial_eval_csv = args.summary_root / "stage145_initial_eval_rows.csv"
    best_eval_csv = args.summary_root / "stage145_best_eval_rows.csv"
    final_eval_csv = args.summary_root / "stage145_final_eval_rows.csv"
    selected_train_csv = args.summary_root / "stage145_selected_train_rows.csv"
    selected_eval_csv = args.summary_root / "stage145_selected_eval_rows.csv"
    write_csv(train_log, train_csv, TRAIN_FIELDS)
    write_csv(validation_log, validation_csv, VALIDATION_FIELDS)
    write_csv(initial_rows, initial_eval_csv, EVAL_FIELDS)
    write_csv(best_rows, best_eval_csv, EVAL_FIELDS)
    write_csv(final_rows, final_eval_csv, EVAL_FIELDS)
    write_selected_rows(train_rows, "train", selected_train_csv)
    write_selected_rows(eval_rows, "eval", selected_eval_csv)

    summary = {
        "stage": args.stage,
        "mode": "large-scale lazy-load adapter training launch",
        "task_manifest": str(args.task_manifest),
        "codecs": args.codecs,
        "gaps": args.gaps,
        "available_train_rows": len(train_rows_all),
        "available_eval_rows": len(eval_rows_all),
        "selected_train_rows": len(train_rows),
        "selected_eval_rows": len(eval_rows),
        "available_train_rows_by_group": row_group_count(train_rows_all),
        "selected_train_rows_by_group": row_group_count(train_rows),
        "selected_eval_rows_by_group": row_group_count(eval_rows),
        "selected_train_sequences": len(set(row["sequence"] for row in train_rows)),
        "selected_eval_sequences": len(set(row["sequence"] for row in eval_rows)),
        "selected_train_rows_by_gap": group_count(train_rows, "reference_gap"),
        "steps": args.steps,
        "eval_interval": args.eval_interval,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "gap_loss_weights": gap_loss_weights,
        "best_metric": args.best_metric,
        "gap4_penalty": args.gap4_penalty,
        "seed": args.seed,
        "heavy_root": str(args.heavy_root),
        "summary_root": str(args.summary_root),
        "init_checkpoint": init_checkpoint,
        "resume_state": str(args.resume_state) if args.resume_state is not None else None,
        "best_step": best_step,
        "best_selection_score": best_score,
        "initial_eval": initial_eval,
        "best_eval": best_eval,
        "final_eval": final_eval,
        "best_checkpoint": best_checkpoint,
        "final_checkpoint": final_checkpoint,
        "latest_state": latest_state,
        "train_anchor_cache": args.train_anchor_cache,
        "train_rgb_cache": args.train_rgb_cache,
        "final_anchor_cache_items": len(anchor_cache) if anchor_cache is not None else 0,
        "final_rgb_cache_items": len(rgb_cache) if rgb_cache is not None else 0,
        "train_log_csv": str(train_csv),
        "validation_log_csv": str(validation_csv),
        "initial_eval_csv": str(initial_eval_csv),
        "best_eval_csv": str(best_eval_csv),
        "final_eval_csv": str(final_eval_csv),
        "selected_train_csv": str(selected_train_csv),
        "selected_eval_csv": str(selected_eval_csv),
        "notes": "Offline RGB targets supervise training only; deployable decoder inputs remain endpoint Gaussian anchors plus normalized time.",
    }
    summary_path = args.summary_root / "stage145_large_scale_adapter_training_launch_summary.json"
    report_path = args.summary_root / "stage145_large_scale_adapter_training_launch_report.md"
    package_path = args.summary_root / "stage145_large_scale_adapter_training_launch_package.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    package = {
        "stage": args.stage,
        "mode": summary["mode"],
        "summary_json": str(summary_path),
        "report_md": str(report_path),
        "best_checkpoint": best_checkpoint["path"],
        "final_checkpoint": final_checkpoint["path"],
        "latest_state": latest_state["path"],
        "selected_train_rows": len(train_rows),
        "selected_eval_rows": len(eval_rows),
        "best_step": best_step,
        "best_eval": best_eval,
        "final_eval": final_eval,
    }
    package_path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary, report_path)
    print(json.dumps({
        "summary": str(summary_path),
        "report": str(report_path),
        "package": str(package_path),
        "best_checkpoint": best_checkpoint["path"],
        "final_checkpoint": final_checkpoint["path"],
        "best_step": best_step,
        "best_eval": best_eval,
        "final_eval": final_eval,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
