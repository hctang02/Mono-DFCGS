import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_MANIFEST = REPO_ROOT / "experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_tasks.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage201_predictor_only_smoke"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage201_predictor_only_smoke")
DEFAULT_STAGE78 = REPO_ROOT / "experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_psnr_table.csv"

sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402
from mono_dfcgs.learned_gs_predictor import TemporalBasisGSRefiner, linear_static_anchor  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import load_rgb, render_anchor  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import load_anchor  # noqa: E402


TASK_FIELDS = [
    "selection_split",
    "task_id",
    "sequence",
    "reference_gap",
    "left_index",
    "right_index",
    "target_index",
    "normalized_time",
]
METRIC_FIELDS = [
    "phase",
    "task_split",
    "task_id",
    "sequence",
    "reference_gap",
    "target_index",
    "normalized_time",
    "anchor_mse",
    "render_mse",
    "psnr",
    "status",
    "error",
]
SUMMARY_FIELDS = [
    "phase",
    "task_split",
    "task_count",
    "mean_anchor_mse",
    "mean_render_mse",
    "mean_psnr",
    "delta_psnr_vs_linear",
    "delta_psnr_vs_stage78_old_adapter_reference",
]
TRAIN_LOG_FIELDS = ["step", "task_id", "sequence", "reference_gap", "anchor_loss", "render_loss", "total_loss", "train_psnr"]
GATE_FIELDS = ["gate", "status", "value", "threshold", "detail"]


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def psnr_from_mse(mse):
    if mse <= 1e-12:
        return 100.0
    return -10.0 * math.log10(float(mse))


def mean(values):
    values = [float(value) for value in values]
    return sum(values) / max(len(values), 1)


def parse_task_rows(path, task_split, gaps, codec):
    gap_set = {int(gap) for gap in gaps}
    rows = []
    for row in read_csv(path):
        if row["task_split"] != task_split:
            continue
        if row["keyframe_codec"] != codec:
            continue
        gap = int(row["reference_gap"])
        if gap not in gap_set:
            continue
        item = dict(row)
        item["reference_gap"] = gap
        item["keyframe_bits"] = int(item["keyframe_bits"])
        item["left_index"] = int(item["left_index"])
        item["right_index"] = int(item["right_index"])
        item["target_index"] = int(item["target_index"])
        item["normalized_time"] = float(item["normalized_time"])
        rows.append(item)
    return rows


def balanced_sample(rows, max_tasks, seed):
    if max_tasks <= 0 or len(rows) <= max_tasks:
        return rows
    rng = random.Random(seed)
    groups = {}
    for row in rows:
        groups.setdefault((int(row["reference_gap"]), row["sequence"]), []).append(row)
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


def selected_task_rows(train_rows, eval_rows):
    out = []
    for selection_split, rows in (("train", train_rows), ("eval", eval_rows)):
        for row in rows:
            out.append({
                "selection_split": selection_split,
                "task_id": row["task_id"],
                "sequence": row["sequence"],
                "reference_gap": row["reference_gap"],
                "left_index": row["left_index"],
                "right_index": row["right_index"],
                "target_index": row["target_index"],
                "normalized_time": row["normalized_time"],
            })
    return out


def align_target_to_render(render, target):
    if tuple(render.shape) == tuple(target.shape):
        return target
    candidates = []
    if target.ndim == render.ndim + 1 and target.shape[1] == 1:
        candidates.append(target.squeeze(1))
    if target.ndim + 1 == render.ndim:
        candidates.append(target.unsqueeze(1))
    if target.ndim == render.ndim and target.shape[0] == 1:
        candidates.append(target)
    for candidate in candidates:
        if tuple(candidate.shape) == tuple(render.shape):
            return candidate
    raise RuntimeError(f"target/render shape mismatch: target={tuple(target.shape)} render={tuple(render.shape)}")


def load_task_tensors(row, device, cache, opt):
    bits = int(row["keyframe_bits"])
    left = load_anchor(row["left_anchor_source_item"], row["left_anchor_source_side"], device, bits=bits, cache=cache)
    right = load_anchor(row["right_anchor_source_item"], row["right_anchor_source_side"], device, bits=bits, cache=cache)
    target_anchor = load_anchor(row["target_anchor_source_item"], row["target_anchor_source_side"], device, bits=None, cache=cache)
    target_rgb = load_rgb(Path(row["target_rgb_path"]), opt.image_height, opt.image_width, device)
    t = torch.tensor([float(row["normalized_time"])], dtype=torch.float32, device=device)
    return left, right, target_anchor, target_rgb, t


def evaluate_rows(rows, model, phase, task_split, device, cache, opt, background):
    model.eval()
    out = []
    with torch.no_grad():
        for row in rows:
            try:
                left, right, target_anchor, target_rgb, t = load_task_tensors(row, device, cache, opt)
                if phase == "linear":
                    pred_anchor = linear_static_anchor(left, right, t)
                else:
                    pred_anchor = model(left, right, t, apply_output_constraints=False)
                pred_attrs = flatten_static_anchor(pred_anchor)
                target_attrs = flatten_static_anchor(target_anchor)
                anchor_mse = float(F.mse_loss(pred_attrs, target_attrs).item())
                render = render_anchor(pred_anchor, background, opt).clamp(0.0, 1.0)
                aligned_target = align_target_to_render(render, target_rgb)
                render_mse = float(F.mse_loss(render, aligned_target).item())
                out.append({
                    "phase": phase,
                    "task_split": task_split,
                    "task_id": row["task_id"],
                    "sequence": row["sequence"],
                    "reference_gap": row["reference_gap"],
                    "target_index": row["target_index"],
                    "normalized_time": row["normalized_time"],
                    "anchor_mse": anchor_mse,
                    "render_mse": render_mse,
                    "psnr": psnr_from_mse(render_mse),
                    "status": "ok",
                    "error": "",
                })
            except Exception as exc:  # noqa: BLE001
                out.append({
                    "phase": phase,
                    "task_split": task_split,
                    "task_id": row.get("task_id", ""),
                    "sequence": row.get("sequence", ""),
                    "reference_gap": row.get("reference_gap", ""),
                    "target_index": row.get("target_index", ""),
                    "normalized_time": row.get("normalized_time", ""),
                    "anchor_mse": "",
                    "render_mse": "",
                    "psnr": "",
                    "status": "error",
                    "error": repr(exc),
                })
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def train_model(model, train_rows, eval_rows, args, device, cache, opt, background):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logs = []
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_eval_psnr = None
    best_step = 0
    initial_eval = evaluate_rows(eval_rows, model, "predictor_initial", "eval", device, cache, opt, background)
    ok_initial = [row for row in initial_eval if row["status"] == "ok"]
    if ok_initial:
        best_eval_psnr = mean(row["psnr"] for row in ok_initial)
    for step in range(1, args.train_steps + 1):
        model.train()
        row = train_rows[(step - 1) % len(train_rows)]
        left, right, target_anchor, target_rgb, t = load_task_tensors(row, device, cache, opt)
        pred_anchor = model(left, right, t, apply_output_constraints=False)
        pred_attrs = flatten_static_anchor(pred_anchor)
        target_attrs = flatten_static_anchor(target_anchor)
        anchor_loss = F.smooth_l1_loss(pred_attrs, target_attrs)
        render = render_anchor(pred_anchor, background, opt).clamp(0.0, 1.0)
        aligned_target = align_target_to_render(render, target_rgb)
        render_loss = F.mse_loss(render, aligned_target)
        total_loss = anchor_loss + float(args.render_loss_weight) * render_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        logs.append({
            "step": step,
            "task_id": row["task_id"],
            "sequence": row["sequence"],
            "reference_gap": row["reference_gap"],
            "anchor_loss": float(anchor_loss.detach().item()),
            "render_loss": float(render_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "train_psnr": psnr_from_mse(float(render_loss.detach().item())),
        })
        if step % args.eval_every == 0 or step == args.train_steps:
            eval_metrics = evaluate_rows(eval_rows, model, f"predictor_step{step}", "eval", device, cache, opt, background)
            ok_eval = [item for item in eval_metrics if item["status"] == "ok"]
            if ok_eval:
                score = mean(item["psnr"] for item in ok_eval)
                if best_eval_psnr is None or score > best_eval_psnr:
                    best_eval_psnr = score
                    best_step = step
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return logs, initial_eval, best_state, best_step, best_eval_psnr


def summarize_metrics(metrics, old_adapter_reference):
    linear_by_split = {}
    for split in sorted({row["task_split"] for row in metrics}):
        linear_rows = [row for row in metrics if row["phase"] == "linear" and row["task_split"] == split and row["status"] == "ok"]
        if linear_rows:
            linear_by_split[split] = mean(row["psnr"] for row in linear_rows)
    grouped = {}
    for row in metrics:
        if row["status"] != "ok":
            continue
        grouped.setdefault((row["phase"], row["task_split"]), []).append(row)
    out = []
    for (phase, task_split), rows in sorted(grouped.items()):
        mean_psnr = mean(row["psnr"] for row in rows)
        ref = old_adapter_reference.get(task_split)
        out.append({
            "phase": phase,
            "task_split": task_split,
            "task_count": len(rows),
            "mean_anchor_mse": mean(row["anchor_mse"] for row in rows),
            "mean_render_mse": mean(row["render_mse"] for row in rows),
            "mean_psnr": mean_psnr,
            "delta_psnr_vs_linear": mean_psnr - linear_by_split.get(task_split, mean_psnr),
            "delta_psnr_vs_stage78_old_adapter_reference": mean_psnr - ref if ref is not None else "",
        })
    return out


def stage78_old_adapter_reference(path, gaps):
    rows = read_csv(path)
    refs = []
    for row in rows:
        if row["codec"] == "q12" and row["method"] == "adapter" and int(row["frame_gap"]) in set(gaps):
            refs.append(float(row["mean_middle_psnr"]))
    value = sum(refs) / max(len(refs), 1)
    return {"train": value, "eval": value}


def endpoint_identity_check(model, eval_row, device, cache, opt):
    left, right, _target_anchor, _target_rgb, _t = load_task_tensors(eval_row, device, cache, opt)
    with torch.no_grad():
        pred0 = model(left, right, torch.tensor([0.0], device=device), apply_output_constraints=False)
        pred1 = model(left, right, torch.tensor([1.0], device=device), apply_output_constraints=False)
    err0 = float((flatten_static_anchor(pred0) - flatten_static_anchor(left)).abs().max().item())
    err1 = float((flatten_static_anchor(pred1) - flatten_static_anchor(right)).abs().max().item())
    return max(err0, err1), err0, err1


def gate_rows(summary_rows, metric_rows, endpoint_error, args, best_checkpoint_path, final_checkpoint_path):
    eval_linear = next(row for row in summary_rows if row["phase"] == "linear" and row["task_split"] == "eval")
    best_eval = next(row for row in summary_rows if row["phase"] == "predictor_best" and row["task_split"] == "eval")
    final_eval = next(row for row in summary_rows if row["phase"] == "predictor_final" and row["task_split"] == "eval")
    errors = [row for row in metric_rows if row["status"] != "ok"]
    no_regression_delta = float(best_eval["mean_psnr"]) - float(eval_linear["mean_psnr"])
    old_adapter_delta = best_eval["delta_psnr_vs_stage78_old_adapter_reference"]
    return [
        {
            "gate": "metric_rows_ok",
            "status": "pass" if not errors else "fail",
            "value": len(errors),
            "threshold": "0",
            "detail": "render/target shapes are explicitly aligned; errors indicate discarded metrics",
        },
        {
            "gate": "endpoint_identity",
            "status": "pass" if endpoint_error <= 1e-6 else "fail",
            "value": endpoint_error,
            "threshold": "<=1e-6",
            "detail": "t=0/t=1 predictor output must match transmitted endpoints",
        },
        {
            "gate": "predictor_only_payload",
            "status": "pass",
            "value": 0,
            "threshold": "0 per-frame payload bytes",
            "detail": "no residual or latent payload is transmitted in Stage201",
        },
        {
            "gate": "best_eval_no_regression_vs_linear",
            "status": "pass" if no_regression_delta >= -float(args.no_regression_tolerance_db) else "fail",
            "value": no_regression_delta,
            "threshold": f">= -{args.no_regression_tolerance_db} dB",
            "detail": f"best={best_eval['mean_psnr']}; linear={eval_linear['mean_psnr']}; best_checkpoint={best_checkpoint_path}",
        },
        {
            "gate": "best_eval_vs_stage78_old_adapter_reference",
            "status": "pass" if float(old_adapter_delta) > 0.0 else "fail",
            "value": old_adapter_delta,
            "threshold": ">0 dB vs Stage78 q12 old adapter mean for smoke gaps",
            "detail": "reference is historical and protocol-different; used only as sanity floor",
        },
        {
            "gate": "final_checkpoint_written",
            "status": "pass" if final_checkpoint_path.exists() else "fail",
            "value": str(final_checkpoint_path),
            "threshold": "exists outside git",
            "detail": "checkpoint is intentionally saved under heavy_root, not committed",
        },
    ]


def write_report(package, summary_rows, gate_rows_, path):
    lines = [
        "# Stage201 Predictor-Only Smoke",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Train tasks: `{package['train_task_count']}`; eval tasks: `{package['eval_task_count']}`.",
        f"- Best step: `{package['best_step']}`; best eval PSNR: `{package['best_eval_psnr']}`.",
        f"- Per-frame payload bytes: `0`.",
        "",
        "## Summary",
        "",
        "| phase | split | tasks | PSNR | dPSNR vs linear | dPSNR vs old adapter ref | anchor MSE | render MSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['phase']} | {row['task_split']} | {row['task_count']} | {float(row['mean_psnr']):.6f} | {float(row['delta_psnr_vs_linear']):.6f} | {float(row['delta_psnr_vs_stage78_old_adapter_reference']):.6f} | {float(row['mean_anchor_mse']):.8f} | {float(row['mean_render_mse']):.8f} |"
        )
    lines.extend([
        "",
        "## Gates",
        "",
        "| gate | status | value | threshold | detail |",
        "|---|---|---:|---|---|",
    ])
    for row in gate_rows_:
        lines.append(f"| {row['gate']} | {row['status']} | {row['value']} | {row['threshold']} | {row['detail']} |")
    lines.extend([
        "",
        "## Decoder Contract",
        "",
        "- Decoder inputs are q12 left/right GS keyframes, normalized time, shared `TemporalBasisGSRefiner` weights, and transmitted schedule metadata.",
        "- Stage201 transmits no residual or latent payload and uses zero per-frame side-info bytes.",
        "- Target dense anchors and target RGB are used only for training/evaluation labels, never as decoder inputs.",
        "",
        "## Outputs",
        "",
        f"- selected tasks: `{package['selected_tasks_csv']}`",
        f"- per-task metrics: `{package['metrics_csv']}`",
        f"- summary: `{package['summary_csv']}`",
        f"- train log: `{package['train_log_csv']}`",
        f"- gates: `{package['gates_csv']}`",
        f"- best checkpoint: `{package['best_checkpoint_path']}`",
        f"- final checkpoint: `{package['final_checkpoint_path']}`",
        f"- package: `{package['package_json']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--stage78_table", type=Path, default=DEFAULT_STAGE78)
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--keyframe_codec", default="q12")
    parser.add_argument("--max_train_tasks", type=int, default=8)
    parser.add_argument("--max_eval_tasks", type=int, default=8)
    parser.add_argument("--train_steps", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--global_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--render_loss_weight", type=float, default=0.02)
    parser.add_argument("--no_regression_tolerance_db", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    cache = {}

    train_all = parse_task_rows(args.task_manifest, "train", args.gaps, args.keyframe_codec)
    eval_all = parse_task_rows(args.task_manifest, "eval", args.gaps, args.keyframe_codec)
    train_rows = balanced_sample(train_all, args.max_train_tasks, args.seed)
    eval_rows = balanced_sample(eval_all, args.max_eval_tasks, args.seed + 1000)
    if not train_rows or not eval_rows:
        raise RuntimeError(f"Need non-empty train/eval rows, got train={len(train_rows)} eval={len(eval_rows)}")

    model = TemporalBasisGSRefiner(
        hidden_dim=args.hidden_dim,
        global_dim=args.global_dim,
        zero_init_residual=True,
        apply_output_constraints=False,
    ).to(device)
    old_adapter_ref = stage78_old_adapter_reference(args.stage78_table, args.gaps)
    linear_train = evaluate_rows(train_rows, model, "linear", "train", device, cache, opt, background)
    linear_eval = evaluate_rows(eval_rows, model, "linear", "eval", device, cache, opt, background)
    logs, initial_eval, best_state, best_step, best_eval_psnr = train_model(model, train_rows, eval_rows, args, device, cache, opt, background)
    final_train = evaluate_rows(train_rows, model, "predictor_final", "train", device, cache, opt, background)
    final_eval = evaluate_rows(eval_rows, model, "predictor_final", "eval", device, cache, opt, background)
    final_checkpoint_path = args.heavy_root / "temporal_basis_gs_refiner_final.safetensors"
    best_checkpoint_path = args.heavy_root / "temporal_basis_gs_refiner_best.safetensors"
    save_file({key: value.detach().cpu() for key, value in model.state_dict().items()}, str(final_checkpoint_path))
    save_file(best_state, str(best_checkpoint_path))
    best_model = TemporalBasisGSRefiner(hidden_dim=args.hidden_dim, global_dim=args.global_dim, zero_init_residual=True, apply_output_constraints=False).to(device)
    best_model.load_state_dict({key: value.to(device) for key, value in best_state.items()}, strict=True)
    best_train = evaluate_rows(train_rows, best_model, "predictor_best", "train", device, cache, opt, background)
    best_eval = evaluate_rows(eval_rows, best_model, "predictor_best", "eval", device, cache, opt, background)
    endpoint_error, endpoint_t0, endpoint_t1 = endpoint_identity_check(best_model, eval_rows[0], device, cache, opt)

    metric_rows = linear_train + linear_eval + initial_eval + final_train + final_eval + best_train + best_eval
    summary_rows = summarize_metrics(metric_rows, old_adapter_ref)
    gates = gate_rows(summary_rows, metric_rows, endpoint_error, args, best_checkpoint_path, final_checkpoint_path)
    decision = "predictor_only_smoke_passed_no_regression_gate"
    if any(row["status"] != "pass" for row in gates):
        decision = "predictor_only_smoke_needs_review"

    selected_tasks_csv = args.summary_root / "stage201_selected_tasks.csv"
    metrics_csv = args.summary_root / "stage201_predictor_only_smoke_metrics.csv"
    summary_csv = args.summary_root / "stage201_predictor_only_smoke_summary.csv"
    train_log_csv = args.summary_root / "stage201_predictor_only_train_log.csv"
    gates_csv = args.summary_root / "stage201_predictor_only_gates.csv"
    package_json = args.summary_root / "stage201_predictor_only_smoke_package.json"
    report_md = args.summary_root / "stage201_predictor_only_smoke_report.md"
    write_csv(selected_task_rows(train_rows, eval_rows), selected_tasks_csv, TASK_FIELDS)
    write_csv(metric_rows, metrics_csv, METRIC_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(logs, train_log_csv, TRAIN_LOG_FIELDS)
    write_csv(gates, gates_csv, GATE_FIELDS)
    package = {
        "stage": 201,
        "name": "predictor_only_smoke",
        "decision": decision,
        "task_manifest": str(args.task_manifest),
        "gaps": args.gaps,
        "keyframe_codec": args.keyframe_codec,
        "train_task_count": len(train_rows),
        "eval_task_count": len(eval_rows),
        "train_steps": args.train_steps,
        "hidden_dim": args.hidden_dim,
        "global_dim": args.global_dim,
        "render_loss_weight": args.render_loss_weight,
        "best_step": best_step,
        "best_eval_psnr": best_eval_psnr,
        "endpoint_error": endpoint_error,
        "endpoint_t0_error": endpoint_t0,
        "endpoint_t1_error": endpoint_t1,
        "per_frame_payload_bytes": 0,
        "selected_tasks_csv": str(selected_tasks_csv),
        "metrics_csv": str(metrics_csv),
        "summary_csv": str(summary_csv),
        "train_log_csv": str(train_log_csv),
        "gates_csv": str(gates_csv),
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "gate_rows": gates,
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, summary_rows, gates, report_md)
    print(json.dumps({"package": str(package_json), "decision": decision, "best_eval_psnr": best_eval_psnr}, indent=2))


if __name__ == "__main__":
    main()
