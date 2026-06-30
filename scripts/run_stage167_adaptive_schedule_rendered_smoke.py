import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE166_ROWS = REPO_ROOT / "experiments/stage166_adaptive_schedule_label_rd_comparison/stage166_sampled_row_consequences.csv"
DEFAULT_STAGE166_SMOKE = REPO_ROOT / "experiments/stage166_adaptive_schedule_label_rd_comparison/stage166_smoke_candidates.csv"
DEFAULT_STAGE165_SCHEDULE_ROWS = REPO_ROOT / "experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_adaptive_schedule_rows.csv"
DEFAULT_CHECKPOINT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/streamsplat.safetensors")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage167_adaptive_schedule_rendered_smoke"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage167_adaptive_schedule_rendered_smoke")


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, unflatten_static_anchor  # noqa: E402
from mono_dfcgs.residual_sideinfo_codec import decode_residual_sideinfo_entropy, encode_topk_residual_sideinfo_entropy  # noqa: E402
from scripts.run_stage153_middle_multimetric_badcase_eval import format_optional, load_metric_modules, mean, percentile, tensor_to_rgb8  # noqa: E402
from scripts.run_stage155_streamsplat_base_sideinfo_upper_bound import compute_metrics, render_batch_with_gs, render_static_anchor, stream_gaussians_at_time  # noqa: E402
from scripts.run_stage156_streamsplat_half_anchor_gaussian_residual import split_half_anchor  # noqa: E402
from scripts.run_stage157_selected_half_anchor_broader_validation import side_mib  # noqa: E402
from scripts.run_stage6_export_real_anchor_dataset import load_model  # noqa: E402
from scripts.run_stage72_original_davis_baseline import DEFAULT_DAVIS_ROOT  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_DENSE_MANIFEST, build_dense_index, load_anchor  # noqa: E402


SCHEDULES = ["uniform_gap8", "stage165_adaptive", "uniform_gap4"]

TARGET_FIELDS = [
    "rank", "target_key", "source_task_id", "sequence", "target_index", "source_gap", "hard_quality_label",
    "high_payload_label", "stage166_false_negative_hard", "stage166_payload_bytes", "selection_score", "selection_reason",
]

ROW_FIELDS = [
    "target_key", "source_task_id", "sequence", "target_index", "source_gap", "hard_quality_label", "high_payload_label",
    "schedule", "left_index", "right_index", "segment_length", "normalized_time", "status", "selected_half",
    "psnr", "ssim", "ms_ssim", "lpips", "original_psnr", "original_ssim", "original_ms_ssim", "original_lpips",
    "payload_bytes", "side_mib_per_intermediate", "delta_psnr_vs_uniform_gap8", "delta_lpips_vs_uniform_gap8", "render_note",
]

SUMMARY_FIELDS = [
    "schedule", "target_count", "rendered_count", "target_keyframe_count", "mean_psnr", "p10_psnr", "mean_ssim",
    "mean_ms_ssim", "mean_lpips", "p90_lpips", "mean_original_psnr", "mean_original_lpips", "mean_payload_bytes",
    "mean_side_mib_per_intermediate", "mean_delta_psnr_vs_uniform_gap8", "mean_delta_lpips_vs_uniform_gap8",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_int_set(value):
    value = str(value).strip()
    if not value:
        return []
    return sorted({int(part) for part in value.split()})


def frame_rgb_path(davis_root, sequence, index):
    return davis_root / "JPEGImages" / "Full-Resolution" / sequence / f"{int(index):05d}.jpg"


def put_label(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def prepare_schedule_maps(schedule_rows):
    out = {name: {} for name in SCHEDULES}
    for row in schedule_rows:
        seq = row["sequence"]
        out["uniform_gap8"][seq] = parse_int_set(row["uniform_gap8_keyframes"])
        out["stage165_adaptive"][seq] = parse_int_set(row["adaptive_keyframes"])
        out["uniform_gap4"][seq] = parse_int_set(row["uniform_gap4_keyframes"])
    return out


def load_smoke_ranks(path):
    ranks = {}
    for row in read_csv(path):
        ranks[row["sequence"]] = int(row["rank"])
    return ranks


def target_key(sequence, target_index):
    return f"{sequence}:{int(target_index):05d}"


def score_target(row, smoke_ranks):
    hard = int(row["hard_quality_label"])
    high_payload = int(row["high_payload_label"])
    false_hard = int(row["stage165_adaptive_false_negative_hard"])
    seq_rank = smoke_ranks.get(row["sequence"], 99)
    smoke_bonus = max(0, 24 - seq_rank) if seq_rank != 99 else 0
    return 100 * false_hard + 40 * hard + 20 * high_payload + smoke_bonus + float(row["payload_bytes"]) / 10000.0


def selection_reason(row, smoke_ranks):
    reasons = []
    if int(row["stage165_adaptive_false_negative_hard"]):
        reasons.append("adaptive_hard_false_negative")
    if int(row["hard_quality_label"]):
        reasons.append("hard_quality_label")
    if int(row["high_payload_label"]):
        reasons.append("high_payload_label")
    if row["sequence"] in smoke_ranks:
        reasons.append("stage166_smoke_sequence")
    return ";".join(reasons) if reasons else "residual_control"


def select_targets(stage166_rows, smoke_ranks, max_targets):
    best = {}
    for row in stage166_rows:
        if int(row["stage165_adaptive_target_is_keyframe"]):
            continue
        key = target_key(row["sequence"], row["target_index"])
        score = score_target(row, smoke_ranks)
        item = dict(row)
        item["selection_score"] = score
        item["selection_reason"] = selection_reason(row, smoke_ranks)
        if key not in best or score > float(best[key]["selection_score"]):
            best[key] = item
    selected = sorted(best.values(), key=lambda row: (float(row["selection_score"]), float(row["payload_bytes"])), reverse=True)[:max_targets]
    out = []
    for rank, row in enumerate(selected, 1):
        out.append({
            "rank": rank,
            "target_key": target_key(row["sequence"], row["target_index"]),
            "source_task_id": row["task_id"],
            "sequence": row["sequence"],
            "target_index": int(row["target_index"]),
            "source_gap": int(row["gap"]),
            "hard_quality_label": int(row["hard_quality_label"]),
            "high_payload_label": int(row["high_payload_label"]),
            "stage166_false_negative_hard": int(row["stage165_adaptive_false_negative_hard"]),
            "stage166_payload_bytes": float(row["payload_bytes"]),
            "selection_score": float(row["selection_score"]),
            "selection_reason": row["selection_reason"],
        })
    return out


def adjacent_keyframes(keyframes, target):
    target = int(target)
    if target in keyframes:
        return target, target
    left = None
    right = None
    for idx in keyframes:
        if idx < target:
            left = idx
        elif idx > target:
            right = idx
            break
    if left is None or right is None:
        raise ValueError(f"No adjacent keyframes for target {target} in {keyframes}")
    return left, right


def build_task(target_row, schedule, schedule_maps, davis_root):
    sequence = target_row["sequence"]
    target = int(target_row["target_index"])
    left, right = adjacent_keyframes(schedule_maps[schedule][sequence], target)
    if left == right:
        return None, left, right
    segment = right - left
    normalized = (target - left) / float(segment)
    task = {
        "task_id": f"stage167_{target_row['source_task_id']}_{schedule}",
        "dataset": "DAVIS",
        "split": "val",
        "sequence": sequence,
        "codec": "q12",
        "reference_gap": segment,
        "left_index": left,
        "right_index": right,
        "target_index": target,
        "segment_length": segment,
        "normalized_time": normalized,
        "left_rgb_path": str(frame_rgb_path(davis_root, sequence, left)),
        "right_rgb_path": str(frame_rgb_path(davis_root, sequence, right)),
        "target_rgb_path": str(frame_rgb_path(davis_root, sequence, target)),
    }
    return task, left, right


def render_stage158(task, model, opt, device, dense_index, lpips_model, ms_ssim_module, keep_fraction, side_bits, selector_payload_bytes, zlib_level, image_cache):
    pred_gs, original_render, target = render_batch_with_gs([task], model, opt, device)
    original_metrics = compute_metrics(original_render, target, lpips_model, ms_ssim_module)
    stream_anchor = stream_gaussians_at_time(pred_gs, float(task["normalized_time"]), opt, sample_idx=0)
    dense_key = (task["dataset"], task["split"], task["sequence"], int(task["target_index"]))
    target_item, target_side = dense_index[dense_key]
    target_anchor = load_anchor(target_item, target_side, device, bits=None, cache=None)
    target_attrs = flatten_static_anchor(target_anchor)
    candidates = []
    for half in ["left", "right"]:
        half_anchor = split_half_anchor(stream_anchor, half)
        base_attrs = flatten_static_anchor(half_anchor)
        payload, info = encode_topk_residual_sideinfo_entropy(
            base_attrs,
            target_attrs,
            keep_fraction,
            side_bits,
            zlib_level=zlib_level,
        )
        corrected_anchor = unflatten_static_anchor(decode_residual_sideinfo_entropy(base_attrs, payload))
        corrected_render = render_static_anchor(corrected_anchor, torch.tensor(opt.background_color, dtype=torch.float32, device=device), opt)
        metrics = compute_metrics(corrected_render, target, lpips_model, ms_ssim_module)
        candidates.append((half, corrected_render, metrics, int(info["payload_bytes"])))
    selected_half, selected_render, selected_metrics, payload_bytes = sorted(candidates, key=lambda item: float(item[2]["psnr"]), reverse=True)[0]
    payload_bytes += int(selector_payload_bytes)
    image_cache[(task["sequence"], int(task["target_index"]), task["schedule"])] = tensor_to_rgb8(selected_render.detach().cpu())
    image_cache[(task["sequence"], int(task["target_index"]), "target")] = tensor_to_rgb8(target.detach().cpu())
    del pred_gs, original_render, target, target_anchor, target_attrs, candidates
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return selected_half, selected_metrics, original_metrics, payload_bytes


def empty_metric_row(target_row, schedule, left, right, note):
    return {
        "target_key": target_row["target_key"],
        "source_task_id": target_row["source_task_id"],
        "sequence": target_row["sequence"],
        "target_index": int(target_row["target_index"]),
        "source_gap": int(target_row["source_gap"]),
        "hard_quality_label": int(target_row["hard_quality_label"]),
        "high_payload_label": int(target_row["high_payload_label"]),
        "schedule": schedule,
        "left_index": left,
        "right_index": right,
        "segment_length": 0,
        "normalized_time": 0.0,
        "status": "target_keyframe_no_middle_render",
        "selected_half": "NA",
        "psnr": None,
        "ssim": None,
        "ms_ssim": None,
        "lpips": None,
        "original_psnr": None,
        "original_ssim": None,
        "original_ms_ssim": None,
        "original_lpips": None,
        "payload_bytes": 0,
        "side_mib_per_intermediate": 0.0,
        "delta_psnr_vs_uniform_gap8": None,
        "delta_lpips_vs_uniform_gap8": None,
        "render_note": note,
    }


def rendered_metric_row(target_row, schedule, task, selected_half, selected_metrics, original_metrics, payload_bytes):
    return {
        "target_key": target_row["target_key"],
        "source_task_id": target_row["source_task_id"],
        "sequence": target_row["sequence"],
        "target_index": int(target_row["target_index"]),
        "source_gap": int(target_row["source_gap"]),
        "hard_quality_label": int(target_row["hard_quality_label"]),
        "high_payload_label": int(target_row["high_payload_label"]),
        "schedule": schedule,
        "left_index": int(task["left_index"]),
        "right_index": int(task["right_index"]),
        "segment_length": int(task["segment_length"]),
        "normalized_time": float(task["normalized_time"]),
        "status": "rendered_middle_recovery",
        "selected_half": selected_half,
        "psnr": selected_metrics["psnr"],
        "ssim": selected_metrics["ssim"],
        "ms_ssim": selected_metrics["ms_ssim"],
        "lpips": selected_metrics["lpips"],
        "original_psnr": original_metrics["psnr"],
        "original_ssim": original_metrics["ssim"],
        "original_ms_ssim": original_metrics["ms_ssim"],
        "original_lpips": original_metrics["lpips"],
        "payload_bytes": int(payload_bytes),
        "side_mib_per_intermediate": side_mib(payload_bytes),
        "delta_psnr_vs_uniform_gap8": None,
        "delta_lpips_vs_uniform_gap8": None,
        "render_note": "actual_adjacent_schedule_segment",
    }


def add_baseline_deltas(rows):
    baseline = {}
    for row in rows:
        if row["schedule"] == "uniform_gap8" and row["status"] == "rendered_middle_recovery":
            baseline[row["target_key"]] = row
    for row in rows:
        base = baseline.get(row["target_key"])
        if base and row["status"] == "rendered_middle_recovery":
            row["delta_psnr_vs_uniform_gap8"] = float(row["psnr"]) - float(base["psnr"])
            if row["lpips"] is not None and base["lpips"] is not None:
                row["delta_lpips_vs_uniform_gap8"] = float(row["lpips"]) - float(base["lpips"])
    return rows


def summarize(rows):
    out = []
    groups = defaultdict(list)
    for row in rows:
        groups[row["schedule"]].append(row)
    for schedule in SCHEDULES:
        group = groups.get(schedule, [])
        rendered = [row for row in group if row["status"] == "rendered_middle_recovery"]
        out.append({
            "schedule": schedule,
            "target_count": len(group),
            "rendered_count": len(rendered),
            "target_keyframe_count": len(group) - len(rendered),
            "mean_psnr": mean(row["psnr"] for row in rendered),
            "p10_psnr": percentile((row["psnr"] for row in rendered), 10) if rendered else None,
            "mean_ssim": mean(row["ssim"] for row in rendered),
            "mean_ms_ssim": mean(row["ms_ssim"] for row in rendered),
            "mean_lpips": mean(row["lpips"] for row in rendered),
            "p90_lpips": percentile((row["lpips"] for row in rendered), 90) if rendered else None,
            "mean_original_psnr": mean(row["original_psnr"] for row in rendered),
            "mean_original_lpips": mean(row["original_lpips"] for row in rendered),
            "mean_payload_bytes": mean(row["payload_bytes"] for row in rendered),
            "mean_side_mib_per_intermediate": mean(row["side_mib_per_intermediate"] for row in rendered),
            "mean_delta_psnr_vs_uniform_gap8": mean(row["delta_psnr_vs_uniform_gap8"] for row in rendered),
            "mean_delta_lpips_vs_uniform_gap8": mean(row["delta_lpips_vs_uniform_gap8"] for row in rendered),
        })
    return out


def save_contact_sheet(targets, rows, image_cache, heavy_root):
    heavy_root.mkdir(parents=True, exist_ok=True)
    by_target = defaultdict(dict)
    for row in rows:
        by_target[row["target_key"]][row["schedule"]] = row
    cells = []
    for target in targets:
        sequence = target["sequence"]
        target_index = int(target["target_index"])
        key = target["target_key"]
        target_img = image_cache.get((sequence, target_index, "target"))
        if target_img is None:
            continue
        row_cells = []
        for label in ["target", "uniform_gap8", "stage165_adaptive", "uniform_gap4"]:
            if label == "target":
                img = target_img.copy()
                text = f"target {sequence} {target_index}"
            else:
                img = image_cache.get((sequence, target_index, label), target_img).copy()
                metric = by_target[key].get(label, {})
                if metric.get("status") == "target_keyframe_no_middle_render":
                    text = f"{label}: keyframe"
                else:
                    text = f"{label}: {format_optional(metric.get('psnr'))} dB"
            put_label(img, text, 8, 22)
            row_cells.append(img)
        cells.append(np.concatenate(row_cells, axis=1))
    if not cells:
        return None
    sheet = np.concatenate(cells, axis=0)
    path = heavy_root / "stage167_adaptive_schedule_rendered_smoke_contact_sheet.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return str(path)


def write_report(summary_rows, targets, package, path):
    lines = [
        "# Stage167 Adaptive Schedule Rendered Smoke",
        "",
        "## Scope",
        "",
        "This is a small rendered smoke on sampled targets that remain middle-recovery targets under the Stage165 adaptive schedule.",
        "The target set is intentionally biased toward Stage166 hard false negatives, so the mean metrics are stress-case indicators rather than representative sequence averages.",
        "It is not a full sequence-level RD validation.",
        "",
        "## Summary",
        "",
        "| schedule | targets | rendered | keyframes | PSNR | SSIM | MS-SSIM | LPIPS | payload bytes | delta PSNR vs gap8 | delta LPIPS vs gap8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['schedule']} | {row['target_count']} | {row['rendered_count']} | {row['target_keyframe_count']} | "
            f"{format_optional(row['mean_psnr'])} | {format_optional(row['mean_ssim'])} | {format_optional(row['mean_ms_ssim'])} | "
            f"{format_optional(row['mean_lpips'])} | {format_optional(row['mean_payload_bytes'])} | "
            f"{format_optional(row['mean_delta_psnr_vs_uniform_gap8'])} | {format_optional(row['mean_delta_lpips_vs_uniform_gap8'])} |"
        )
    lines.extend([
        "",
        "## Targets",
        "",
        "| rank | sequence | target | reason | score |",
        "|---:|---|---:|---|---:|",
    ])
    for row in targets:
        lines.append(f"| {row['rank']} | {row['sequence']} | {row['target_index']} | {row['selection_reason']} | {float(row['selection_score']):.3f} |")
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Heavy contact sheet: `{package['contact_sheet']}`.",
        "- Keep this as smoke evidence only; broader adaptive rendered validation is still required before final claims.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage166_rows", type=Path, default=DEFAULT_STAGE166_ROWS)
    parser.add_argument("--stage166_smoke", type=Path, default=DEFAULT_STAGE166_SMOKE)
    parser.add_argument("--stage165_schedule_rows", type=Path, default=DEFAULT_STAGE165_SCHEDULE_ROWS)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--max_targets", type=int, default=8)
    parser.add_argument("--keep_fraction", type=float, default=1.0)
    parser.add_argument("--side_bits", type=int, default=6)
    parser.add_argument("--selector_payload_bytes", type=int, default=1)
    parser.add_argument("--zlib_level", type=int, default=9)
    parser.add_argument("--disable_lpips", action="store_true")
    parser.add_argument("--disable_ms_ssim", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    smoke_ranks = load_smoke_ranks(args.stage166_smoke)
    targets = select_targets(read_csv(args.stage166_rows), smoke_ranks, args.max_targets)
    schedule_maps = prepare_schedule_maps(read_csv(args.stage165_schedule_rows))
    dense_index = build_dense_index(args.dense_manifest, ["val"])
    device = torch.device(args.device)
    opt = Options()
    opt.resume = str(args.checkpoint)
    opt.compile = False
    opt.input_frames = 1
    opt.output_frames = 3
    opt.epoch = 0
    model = load_model(deepcopy(opt), args.checkpoint, str(device))
    lpips_model, lpips_error, ms_ssim_module, ms_ssim_error = load_metric_modules(device, args.disable_lpips, args.disable_ms_ssim)
    rows = []
    image_cache = {}
    for target_row in targets:
        for schedule in SCHEDULES:
            task, left, right = build_task(target_row, schedule, schedule_maps, args.davis_root)
            if task is None:
                rows.append(empty_metric_row(target_row, schedule, left, right, "target is transmitted keyframe under this schedule"))
                continue
            task["schedule"] = schedule
            selected_half, selected_metrics, original_metrics, payload_bytes = render_stage158(
                task,
                model,
                opt,
                device,
                dense_index,
                lpips_model,
                ms_ssim_module,
                args.keep_fraction,
                args.side_bits,
                args.selector_payload_bytes,
                args.zlib_level,
                image_cache,
            )
            rows.append(rendered_metric_row(target_row, schedule, task, selected_half, selected_metrics, original_metrics, payload_bytes))
            print(json.dumps({"rendered": len(rows), "target": target_row["target_key"], "schedule": schedule}), flush=True)
    rows = add_baseline_deltas(rows)
    summary_rows = summarize(rows)
    contact_sheet = save_contact_sheet(targets, rows, image_cache, args.heavy_root)
    rows_csv = args.summary_root / "stage167_rendered_smoke_rows.csv"
    targets_csv = args.summary_root / "stage167_rendered_smoke_targets.csv"
    summary_csv = args.summary_root / "stage167_rendered_smoke_summary.csv"
    package_json = args.summary_root / "stage167_adaptive_schedule_rendered_smoke_package.json"
    report_md = args.summary_root / "stage167_adaptive_schedule_rendered_smoke_report.md"
    write_csv(targets, targets_csv, TARGET_FIELDS)
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    adaptive = next(row for row in summary_rows if row["schedule"] == "stage165_adaptive")
    decision = "scale_to_broader_rendered_validation" if adaptive["mean_psnr"] is not None and float(adaptive["mean_psnr"]) >= 28.0 else "inspect_smoke_before_scaling"
    package = {
        "stage": 167,
        "status": "adaptive_schedule_rendered_smoke_packaged",
        "decision": decision,
        "policy": "streamsplat_guided_half_anchor_entropy_residual_v1",
        "target_count": len(targets),
        "schedule_count": len(SCHEDULES),
        "summary_rows": summary_rows,
        "lpips_available": lpips_model is not None,
        "lpips_error": lpips_error,
        "ms_ssim_available": ms_ssim_module is not None,
        "ms_ssim_error": ms_ssim_error,
        "contact_sheet": contact_sheet,
        "targets_csv": str(targets_csv),
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "heavy_root": str(args.heavy_root),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary_rows, targets, package, report_md)
    print(json.dumps({"package": str(package_json), "report": str(report_md), "decision": decision, "contact_sheet": contact_sheet}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
