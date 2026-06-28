import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_MANIFEST = REPO_ROOT / "experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_tasks.csv"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage80_adapter_training_smoke")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage80_adapter_training_smoke"
DEFAULT_REFERENCE_CHECKPOINT = Path(
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


EVAL_FIELDS = [
    "step",
    "task_id",
    "task_split",
    "sequence",
    "codec",
    "reference_gap",
    "left_index",
    "right_index",
    "target_index",
    "normalized_time",
    "model_mse",
    "model_psnr",
    "linear_mse",
    "linear_psnr",
    "margin_over_linear_psnr",
]

TRAIN_FIELDS = [
    "step",
    "task_id",
    "sequence",
    "codec",
    "reference_gap",
    "target_index",
    "normalized_time",
    "loss",
    "loss_psnr",
]

VALIDATION_FIELDS = [
    "step",
    "model_psnr_avg",
    "linear_psnr_avg",
    "margin_over_linear_psnr",
    "best_so_far",
]


def read_task_rows(path, task_split, codecs, gaps, sequences):
    codec_set = set(codecs)
    gap_set = set(gaps)
    sequence_set = set(sequences or [])
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["task_split"] != task_split:
                continue
            if codec_set and row["codec"] not in codec_set:
                continue
            row["reference_gap"] = int(row["reference_gap"])
            if gap_set and row["reference_gap"] not in gap_set:
                continue
            if sequence_set and row["sequence"] not in sequence_set:
                continue
            row["bits"] = int(row["bits"])
            row["left_index"] = int(row["left_index"])
            row["right_index"] = int(row["right_index"])
            row["target_index"] = int(row["target_index"])
            row["segment_length"] = int(row["segment_length"])
            row["normalized_time"] = float(row["normalized_time"])
            if not Path(row["left_anchor_source_item"]).exists():
                continue
            if not Path(row["right_anchor_source_item"]).exists():
                continue
            if not Path(row["target_rgb_path"]).exists():
                continue
            rows.append(row)
    return rows


def select_balanced(rows, max_tasks, seed):
    if max_tasks <= 0 or len(rows) <= max_tasks:
        return rows
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["sequence"], row["codec"], row["reference_gap"])].append(row)
    for items in groups.values():
        rng.shuffle(items)
    keys = sorted(groups)
    rng.shuffle(keys)
    selected = []
    offset = 0
    while len(selected) < max_tasks:
        progressed = False
        for key in keys:
            items = groups[key]
            if offset >= len(items):
                continue
            selected.append(items[offset])
            progressed = True
            if len(selected) >= max_tasks:
                break
        if not progressed:
            break
        offset += 1
    return selected


def load_source_anchor(row, prefix, device, cache):
    source_item = row[f"{prefix}_anchor_source_item"]
    source_side = row[f"{prefix}_anchor_source_side"]
    key = (source_item, source_side, row["bits"], str(device))
    if cache is not None and key in cache:
        return cache[key]
    item = torch.load(source_item, map_location="cpu", weights_only=True)
    if source_side not in item:
        raise KeyError(f"Missing {source_side} in {source_item}")
    anchor = maybe_quantize_anchor(anchor_to_device(item[source_side], device), row["bits"])
    if cache is not None:
        cache[key] = anchor
    return anchor


def load_target_rgb(path, height, width, device, cache):
    key = (path, height, width, str(device))
    if cache is not None and key in cache:
        return cache[key]
    rgb = load_rgb(path, height, width, device)
    if cache is not None:
        cache[key] = rgb
    return rgb


def make_tasks(rows, height, width, device, cache_anchors=True, cache_rgb=True):
    anchor_cache = {} if cache_anchors else None
    rgb_cache = {} if cache_rgb else None
    tasks = []
    for row in rows:
        tasks.append({
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
            "left": load_source_anchor(row, "left", device, anchor_cache),
            "right": load_source_anchor(row, "right", device, anchor_cache),
            "target": load_target_rgb(row["target_rgb_path"], height, width, device, rgb_cache),
        })
    return tasks


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


def average(rows, key):
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def group_average(rows, group_key, value_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(float(row[value_key]))
    return {key: sum(values) / len(values) for key, values in sorted(grouped.items())}


def summarize_eval(rows):
    return {
        "task_count": len(rows),
        "model_psnr_avg": average(rows, "model_psnr"),
        "linear_psnr_avg": average(rows, "linear_psnr"),
        "margin_over_linear_psnr": average(rows, "margin_over_linear_psnr"),
        "model_psnr_by_gap": group_average(rows, "reference_gap", "model_psnr"),
        "linear_psnr_by_gap": group_average(rows, "reference_gap", "linear_psnr"),
        "margin_over_linear_psnr_by_gap": group_average(rows, "reference_gap", "margin_over_linear_psnr"),
    }


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def safetensors_device(device):
    if device.type == "cuda" and device.index is None:
        return "cuda:0"
    return str(device)


def load_reference_model(path, hidden_dim, device):
    model = GaussianAnchorDynamicPredictor(
        hidden_dim=hidden_dim,
        apply_output_constraints=False,
        zero_init_residual=True,
    ).to(device)
    state = load_file(str(path), device=safetensors_device(device))
    model.load_state_dict(state, strict=True)
    return model


def train_adapter(args, train_tasks, eval_tasks, background, opt, device):
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    model = GaussianAnchorDynamicPredictor(
        hidden_dim=args.hidden_dim,
        apply_output_constraints=False,
        zero_init_residual=True,
    ).to(device)
    init_checkpoint = None
    if args.init_checkpoint is not None:
        state = load_file(str(args.init_checkpoint), device=safetensors_device(device))
        model.load_state_dict(state, strict=True)
        init_checkpoint = {
            "path": str(args.init_checkpoint),
            "tensor_count": len(state),
            "parameter_count": sum(t.numel() for t in state.values()),
        }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    order = list(range(len(train_tasks)))
    rng.shuffle(order)

    initial_rows = evaluate_model(model, eval_tasks, background, opt, step=0)
    best_rows = initial_rows
    best_eval = summarize_eval(initial_rows)
    best_step = 0
    best_checkpoint = save_model(model, args.heavy_root / "best_adapter.safetensors")
    validation_log = [{"step": 0, **best_eval, "best_so_far": True}]
    train_log = []

    model.train()
    for step in range(1, args.steps + 1):
        if (step - 1) % len(order) == 0:
            rng.shuffle(order)
        task = train_tasks[order[(step - 1) % len(order)]]
        pred_rgb = render_prediction(model, task, background, opt).clamp(0.0, 1.0)
        loss = F.mse_loss(pred_rgb, task["target"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        loss_value = float(loss.detach().item())
        row = {
            "step": step,
            "task_id": task["task_id"],
            "sequence": task["sequence"],
            "codec": task["codec"],
            "reference_gap": task["reference_gap"],
            "target_index": task["target_index"],
            "normalized_time": task["normalized_time"],
            "loss": loss_value,
            "loss_psnr": psnr_from_mse(loss_value),
        }
        train_log.append(row)
        print(json.dumps(row), flush=True)

        if step % args.eval_interval == 0 or step == args.steps:
            current_rows = evaluate_model(model, eval_tasks, background, opt, step=step)
            current_eval = summarize_eval(current_rows)
            is_best = current_eval["margin_over_linear_psnr"] > best_eval["margin_over_linear_psnr"]
            if is_best:
                best_rows = current_rows
                best_eval = current_eval
                best_step = step
                best_checkpoint = save_model(model, args.heavy_root / "best_adapter.safetensors")
            validation_row = {"step": step, **current_eval, "best_so_far": is_best}
            validation_log.append(validation_row)
            print(json.dumps(validation_row), flush=True)
            model.train()

    final_rows = evaluate_model(model, eval_tasks, background, opt, step=args.steps)
    final_eval = summarize_eval(final_rows)
    final_checkpoint = save_model(model, args.heavy_root / "final_adapter.safetensors")
    return {
        "model": model,
        "train_log": train_log,
        "validation_log": validation_log,
        "initial_rows": initial_rows,
        "best_rows": best_rows,
        "final_rows": final_rows,
        "initial_eval": summarize_eval(initial_rows),
        "best_eval": best_eval,
        "final_eval": final_eval,
        "best_step": best_step,
        "best_checkpoint": best_checkpoint,
        "final_checkpoint": final_checkpoint,
        "init_checkpoint": init_checkpoint,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def write_report(summary, path):
    lines = [
        f"# {summary['stage_label']}",
        "",
        "## Configuration",
        "",
        f"- task manifest: `{summary['task_manifest']}`",
        f"- codecs: `{summary['codecs']}`",
        f"- gaps: `{summary['gaps']}`",
        f"- train tasks: `{summary['train_task_count']}`",
        f"- eval tasks: `{summary['eval_task_count']}`",
        f"- steps: `{summary['steps']}`",
        f"- heavy root: `{summary['heavy_root']}`",
        f"- init checkpoint: `{summary['init_checkpoint']['path'] if summary.get('init_checkpoint') else None}`",
        "",
        "## Evaluation",
        "",
        "| checkpoint | step | model PSNR | linear PSNR | margin |",
        "|---|---:|---:|---:|---:|",
        f"| initial | 0 | {summary['initial_eval']['model_psnr_avg']} | {summary['initial_eval']['linear_psnr_avg']} | {summary['initial_eval']['margin_over_linear_psnr']} |",
        f"| best | {summary['best_step']} | {summary['best_eval']['model_psnr_avg']} | {summary['best_eval']['linear_psnr_avg']} | {summary['best_eval']['margin_over_linear_psnr']} |",
        f"| final | {summary['steps']} | {summary['final_eval']['model_psnr_avg']} | {summary['final_eval']['linear_psnr_avg']} | {summary['final_eval']['margin_over_linear_psnr']} |",
    ]
    if summary.get("reference_eval") is not None:
        ref = summary["reference_eval"]
        lines.append(f"| reference adapter | -1 | {ref['model_psnr_avg']} | {ref['linear_psnr_avg']} | {ref['margin_over_linear_psnr']} |")
    lines.extend([
        "",
        "## Notes",
        "",
        f"- {summary['run_note']}",
        "- Training uses target RGB only as offline supervision; transmitted test-time inputs remain endpoint Gaussian anchors plus normalized time.",
        "- Checkpoints are stored outside git under the heavy root.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=80)
    parser.add_argument("--stage_label", default="Stage80 Adapter Training Smoke")
    parser.add_argument("--mode", default="adapter task-manifest RGB training smoke")
    parser.add_argument("--output_prefix", default="stage80")
    parser.add_argument("--summary_name", default="stage80_adapter_training_smoke")
    parser.add_argument("--run_note", default="This is a smoke run, not a final long-training result.")
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--codecs", nargs="+", default=["q10"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4])
    parser.add_argument("--train_sequences", nargs="*", default=[])
    parser.add_argument("--eval_sequences", nargs="*", default=[])
    parser.add_argument("--max_train_tasks", type=int, default=4)
    parser.add_argument("--max_eval_tasks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--init_checkpoint", type=Path, default=None)
    parser.add_argument("--reference_checkpoint", type=Path, default=DEFAULT_REFERENCE_CHECKPOINT)
    parser.add_argument("--reference_hidden_dim", type=int, default=256)
    parser.add_argument("--no_cache_anchors", action="store_true")
    parser.add_argument("--no_cache_rgb", action="store_true")
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

    train_rows_all = read_task_rows(args.task_manifest, "train", args.codecs, args.gaps, args.train_sequences)
    eval_rows_all = read_task_rows(args.task_manifest, "eval", args.codecs, args.gaps, args.eval_sequences)
    train_rows = select_balanced(train_rows_all, args.max_train_tasks, args.seed)
    eval_rows = select_balanced(eval_rows_all, args.max_eval_tasks, args.seed + 1)
    if not train_rows or not eval_rows:
        raise RuntimeError(f"Need non-empty train/eval tasks, got train={len(train_rows)} eval={len(eval_rows)}")

    train_tasks = make_tasks(
        train_rows,
        opt.image_height,
        opt.image_width,
        device,
        cache_anchors=not args.no_cache_anchors,
        cache_rgb=not args.no_cache_rgb,
    )
    eval_tasks = make_tasks(
        eval_rows,
        opt.image_height,
        opt.image_width,
        device,
        cache_anchors=not args.no_cache_anchors,
        cache_rgb=not args.no_cache_rgb,
    )

    reference_rows = []
    reference_eval = None
    if args.reference_checkpoint.exists():
        reference_model = load_reference_model(args.reference_checkpoint, args.reference_hidden_dim, device)
        reference_rows = evaluate_model(reference_model, eval_tasks, background, opt, step=-1)
        reference_eval = summarize_eval(reference_rows)

    result = train_adapter(args, train_tasks, eval_tasks, background, opt, device)

    train_csv = args.summary_root / f"{args.output_prefix}_train_log.csv"
    validation_csv = args.summary_root / f"{args.output_prefix}_validation_log.csv"
    best_eval_csv = args.summary_root / f"{args.output_prefix}_best_eval_rows.csv"
    final_eval_csv = args.summary_root / f"{args.output_prefix}_final_eval_rows.csv"
    reference_eval_csv = args.summary_root / f"{args.output_prefix}_reference_eval_rows.csv"
    write_csv(result["train_log"], train_csv, TRAIN_FIELDS)
    write_csv(result["validation_log"], validation_csv, VALIDATION_FIELDS)
    write_csv(result["best_rows"], best_eval_csv, EVAL_FIELDS)
    write_csv(result["final_rows"], final_eval_csv, EVAL_FIELDS)
    write_csv(reference_rows, reference_eval_csv, EVAL_FIELDS)

    summary = {
        "stage": args.stage,
        "stage_label": args.stage_label,
        "mode": args.mode,
        "run_note": args.run_note,
        "task_manifest": str(args.task_manifest),
        "codecs": args.codecs,
        "gaps": args.gaps,
        "available_train_rows": len(train_rows_all),
        "available_eval_rows": len(eval_rows_all),
        "train_task_count": len(train_tasks),
        "eval_task_count": len(eval_tasks),
        "steps": args.steps,
        "eval_interval": args.eval_interval,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "parameter_count": result["parameter_count"],
        "heavy_root": str(args.heavy_root),
        "summary_root": str(args.summary_root),
        "initial_eval": result["initial_eval"],
        "best_eval": result["best_eval"],
        "final_eval": result["final_eval"],
        "best_step": result["best_step"],
        "init_checkpoint": result["init_checkpoint"],
        "best_checkpoint": result["best_checkpoint"],
        "final_checkpoint": result["final_checkpoint"],
        "reference_checkpoint": str(args.reference_checkpoint),
        "reference_checkpoint_exists": args.reference_checkpoint.exists(),
        "reference_eval": reference_eval,
        "train_log_csv": str(train_csv),
        "validation_log_csv": str(validation_csv),
        "best_eval_csv": str(best_eval_csv),
        "final_eval_csv": str(final_eval_csv),
        "reference_eval_csv": str(reference_eval_csv),
        "notes": f"{args.run_note} Offline RGB targets are used for training supervision; transmitted test-time inputs remain endpoint static Gaussian anchors plus normalized time.",
    }
    summary_path = args.summary_root / f"{args.summary_name}_summary.json"
    report_path = args.summary_root / f"{args.summary_name}_report.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, report_path)
    print(json.dumps({
        "summary": str(summary_path),
        "report": str(report_path),
        "best_checkpoint": result["best_checkpoint"]["path"],
        "final_checkpoint": result["final_checkpoint"]["path"],
        "best_eval": result["best_eval"],
        "final_eval": result["final_eval"],
        "reference_eval": reference_eval,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
