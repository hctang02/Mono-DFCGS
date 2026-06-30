import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE169_TARGETS = REPO_ROOT / "experiments/stage169_combined_adaptive_rendered_validation_protocol/stage169_combined_validation_targets.csv"
DEFAULT_STAGE169_SCHEDULE_ROWS = REPO_ROOT / "experiments/stage169_combined_adaptive_rendered_validation_protocol/stage169_combined_validation_schedule_rows.csv"
DEFAULT_STAGE167_ROWS = REPO_ROOT / "experiments/stage167_adaptive_schedule_rendered_smoke/stage167_rendered_smoke_rows.csv"
DEFAULT_STAGE168_ROWS = REPO_ROOT / "experiments/stage168_positive_coverage_rendered_smoke/stage168_positive_coverage_rows.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage170_combined_adaptive_rendered_validation_execution"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage170_combined_adaptive_rendered_validation_execution")


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage153_middle_multimetric_badcase_eval import format_optional, load_metric_modules, mean, percentile  # noqa: E402
from scripts.run_stage167_adaptive_schedule_rendered_smoke import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_STAGE165_SCHEDULE_ROWS,
    build_task,
    empty_metric_row,
    prepare_schedule_maps,
    put_label,
    read_csv,
    rendered_metric_row,
    render_stage158,
    write_csv,
)
from scripts.run_stage6_export_real_anchor_dataset import load_model  # noqa: E402
from scripts.run_stage72_original_davis_baseline import DEFAULT_DAVIS_ROOT  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_DENSE_MANIFEST, build_dense_index  # noqa: E402


ROW_FIELDS = [
    "category", "target_key", "source_task_id", "sequence", "target_index", "source_gap", "hard_quality_label", "high_payload_label",
    "schedule", "left_index", "right_index", "segment_length", "normalized_time", "status", "row_source", "selected_half",
    "psnr", "ssim", "ms_ssim", "lpips", "original_psnr", "original_ssim", "original_ms_ssim", "original_lpips",
    "payload_bytes", "side_mib_per_intermediate", "delta_psnr_vs_uniform_gap8", "delta_lpips_vs_uniform_gap8", "render_note",
]
SUMMARY_FIELDS = [
    "category", "schedule", "row_count", "rendered_count", "keyframe_marker_count", "reused_count", "new_render_count",
    "mean_psnr", "p10_psnr", "mean_ssim", "mean_ms_ssim", "mean_lpips", "p90_lpips", "mean_original_psnr",
    "mean_original_lpips", "mean_payload_bytes", "mean_side_mib_per_intermediate", "mean_delta_psnr_vs_uniform_gap8",
    "mean_delta_lpips_vs_uniform_gap8",
]
SOURCE_SUMMARY_FIELDS = ["row_source", "row_count", "rendered_count", "keyframe_marker_count"]


def load_existing_rows(stage167_rows, stage168_rows):
    out = {}
    for source, path in [("stage167", stage167_rows), ("stage168", stage168_rows)]:
        if not path.exists():
            continue
        for row in read_csv(path):
            item = dict(row)
            item["row_source"] = source
            out[(item["target_key"], item["schedule"])] = item
    return out


def target_map(rows):
    return {row["target_key"]: row for row in rows}


def add_context(row, target, category, row_source):
    out = dict(row)
    out["category"] = category
    out["source_task_id"] = target["source_task_id"]
    out["source_gap"] = int(target["source_gap"])
    out["hard_quality_label"] = int(target["hard_quality_label"])
    out["high_payload_label"] = int(target["high_payload_label"])
    out["row_source"] = row_source
    return out


def add_baseline_deltas(rows):
    baseline = {}
    for row in rows:
        if row["schedule"] == "uniform_gap8" and row["status"] == "rendered_middle_recovery" and row.get("psnr") not in (None, ""):
            baseline[row["target_key"]] = row
    for row in rows:
        base = baseline.get(row["target_key"])
        if base and row["status"] == "rendered_middle_recovery" and row.get("psnr") not in (None, ""):
            row["delta_psnr_vs_uniform_gap8"] = float(row["psnr"]) - float(base["psnr"])
            if row.get("lpips") not in (None, "") and base.get("lpips") not in (None, ""):
                row["delta_lpips_vs_uniform_gap8"] = float(row["lpips"]) - float(base["lpips"])
    return rows


def numeric(row, key):
    value = row.get(key)
    if value in (None, "", "NA"):
        return None
    return float(value)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["category"], row["schedule"])].append(row)
    out = []
    for (category, schedule), group in sorted(groups.items()):
        rendered = [row for row in group if row["status"] == "rendered_middle_recovery"]
        out.append({
            "category": category,
            "schedule": schedule,
            "row_count": len(group),
            "rendered_count": len(rendered),
            "keyframe_marker_count": sum(1 for row in group if row["status"] == "target_keyframe_no_middle_render"),
            "reused_count": sum(1 for row in group if str(row["row_source"]).startswith("stage16")),
            "new_render_count": sum(1 for row in group if row["row_source"] == "stage170_rendered"),
            "mean_psnr": mean(numeric(row, "psnr") for row in rendered),
            "p10_psnr": percentile((numeric(row, "psnr") for row in rendered), 10) if rendered else None,
            "mean_ssim": mean(numeric(row, "ssim") for row in rendered),
            "mean_ms_ssim": mean(numeric(row, "ms_ssim") for row in rendered),
            "mean_lpips": mean(numeric(row, "lpips") for row in rendered),
            "p90_lpips": percentile((numeric(row, "lpips") for row in rendered), 90) if rendered else None,
            "mean_original_psnr": mean(numeric(row, "original_psnr") for row in rendered),
            "mean_original_lpips": mean(numeric(row, "original_lpips") for row in rendered),
            "mean_payload_bytes": mean(numeric(row, "payload_bytes") for row in rendered),
            "mean_side_mib_per_intermediate": mean(numeric(row, "side_mib_per_intermediate") for row in rendered),
            "mean_delta_psnr_vs_uniform_gap8": mean(numeric(row, "delta_psnr_vs_uniform_gap8") for row in rendered),
            "mean_delta_lpips_vs_uniform_gap8": mean(numeric(row, "delta_lpips_vs_uniform_gap8") for row in rendered),
        })
    return out


def source_summary(rows):
    out = []
    counts = Counter(row["row_source"] for row in rows)
    for source, count in sorted(counts.items()):
        group = [row for row in rows if row["row_source"] == source]
        out.append({
            "row_source": source,
            "row_count": count,
            "rendered_count": sum(1 for row in group if row["status"] == "rendered_middle_recovery"),
            "keyframe_marker_count": sum(1 for row in group if row["status"] == "target_keyframe_no_middle_render"),
        })
    return out


def save_contact_sheet(rows, image_cache, heavy_root):
    heavy_root.mkdir(parents=True, exist_ok=True)
    rendered_new = [row for row in rows if row["row_source"] == "stage170_rendered"]
    if not rendered_new:
        return None
    cells = []
    for row in rendered_new:
        key = (row["sequence"], int(row["target_index"]), row["schedule"])
        target_key = (row["sequence"], int(row["target_index"]), "target")
        pred = image_cache.get(key)
        target = image_cache.get(target_key)
        if pred is None or target is None:
            continue
        target_img = target.copy()
        pred_img = pred.copy()
        put_label(target_img, f"target {row['target_key']}", 8, 22)
        put_label(pred_img, f"{row['schedule']} {format_optional(row.get('psnr'))} dB", 8, 22)
        cells.append(np.concatenate([target_img, pred_img], axis=1))
    if not cells:
        return None
    sheet = np.concatenate(cells, axis=0)
    path = heavy_root / "stage170_combined_adaptive_rendered_validation_contact_sheet.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return str(path)


def write_report(summary_rows, source_rows, package, path):
    lines = [
        "# Stage170 Combined Adaptive Rendered Validation Execution",
        "",
        "## Scope",
        "",
        "This executes the Stage169 protocol by reusing Stage167/168 rows, rendering only missing rows, and marking adaptive/keyframe rows without claiming middle-render metrics.",
        "",
        "## Source Coverage",
        "",
        "| source | rows | rendered | keyframe markers |",
        "|---|---:|---:|---:|",
    ]
    for row in source_rows:
        lines.append(f"| {row['row_source']} | {row['row_count']} | {row['rendered_count']} | {row['keyframe_marker_count']} |")
    lines.extend([
        "",
        "## Category/Schedule Summary",
        "",
        "| category | schedule | rows | rendered | keyframes | reused | new renders | PSNR | SSIM | MS-SSIM | LPIPS | payload bytes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        lines.append(
            f"| {row['category']} | {row['schedule']} | {row['row_count']} | {row['rendered_count']} | {row['keyframe_marker_count']} | "
            f"{row['reused_count']} | {row['new_render_count']} | {format_optional(row['mean_psnr'])} | {format_optional(row['mean_ssim'])} | "
            f"{format_optional(row['mean_ms_ssim'])} | {format_optional(row['mean_lpips'])} | {format_optional(row['mean_payload_bytes'])} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- New renders completed: `{package['new_render_count']}`.",
        f"- Reused rows: `{package['reused_row_count']}`.",
        f"- Keyframe marker rows: `{package['keyframe_marker_count']}`.",
        f"- Heavy contact sheet for new renders: `{package['contact_sheet']}`.",
        "- This remains sampled validation; full-sequence RD should only follow if this combined evidence is acceptable.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage169_targets", type=Path, default=DEFAULT_STAGE169_TARGETS)
    parser.add_argument("--stage169_schedule_rows", type=Path, default=DEFAULT_STAGE169_SCHEDULE_ROWS)
    parser.add_argument("--stage167_rows", type=Path, default=DEFAULT_STAGE167_ROWS)
    parser.add_argument("--stage168_rows", type=Path, default=DEFAULT_STAGE168_ROWS)
    parser.add_argument("--stage165_schedule_rows", type=Path, default=DEFAULT_STAGE165_SCHEDULE_ROWS)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--max_new_renders", type=int, default=64)
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
    targets = target_map(read_csv(args.stage169_targets))
    protocol_rows = read_csv(args.stage169_schedule_rows)
    existing_rows = load_existing_rows(args.stage167_rows, args.stage168_rows)
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
    new_renders = 0
    for protocol in protocol_rows:
        key = (protocol["target_key"], protocol["schedule"])
        target = targets[protocol["target_key"]]
        if key in existing_rows:
            rows.append(add_context(existing_rows[key], target, protocol["category"], existing_rows[key]["row_source"]))
            continue
        task, left, right = build_task(target, protocol["schedule"], schedule_maps, args.davis_root)
        if protocol["expected_status"] == "target_keyframe_no_middle_render" or task is None:
            rows.append(add_context(empty_metric_row(target, protocol["schedule"], left, right, "target is transmitted keyframe under this schedule"), target, protocol["category"], "stage170_keyframe_marker"))
            continue
        if new_renders >= int(args.max_new_renders):
            raise RuntimeError(f"max_new_renders reached before completing protocol: {args.max_new_renders}")
        task["schedule"] = protocol["schedule"]
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
        rows.append(add_context(rendered_metric_row(target, protocol["schedule"], task, selected_half, selected_metrics, original_metrics, payload_bytes), target, protocol["category"], "stage170_rendered"))
        new_renders += 1
        print(json.dumps({"new_renders": new_renders, "target": protocol["target_key"], "schedule": protocol["schedule"]}), flush=True)
    rows = add_baseline_deltas(rows)
    summary_rows = summarize(rows)
    source_rows = source_summary(rows)
    contact_sheet = save_contact_sheet(rows, image_cache, args.heavy_root)
    rows_csv = args.summary_root / "stage170_combined_validation_rows.csv"
    summary_csv = args.summary_root / "stage170_combined_validation_summary.csv"
    source_csv = args.summary_root / "stage170_combined_validation_source_summary.csv"
    package_json = args.summary_root / "stage170_combined_adaptive_rendered_validation_execution_package.json"
    report_md = args.summary_root / "stage170_combined_adaptive_rendered_validation_execution_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(source_rows, source_csv, SOURCE_SUMMARY_FIELDS)
    protocol_complete = len(rows) == len(protocol_rows)
    decision = "combined_validation_ready_for_review" if protocol_complete and new_renders == sum(int(row["requires_stage170_render"]) for row in protocol_rows) else "combined_validation_incomplete"
    package = {
        "stage": 170,
        "status": "combined_adaptive_rendered_validation_execution_packaged",
        "decision": decision,
        "policy": "streamsplat_guided_half_anchor_entropy_residual_v1",
        "protocol_row_count": len(protocol_rows),
        "output_row_count": len(rows),
        "new_render_count": new_renders,
        "reused_row_count": sum(1 for row in rows if row["row_source"] in ("stage167", "stage168")),
        "keyframe_marker_count": sum(1 for row in rows if row["row_source"] == "stage170_keyframe_marker"),
        "summary_rows": summary_rows,
        "source_summary_rows": source_rows,
        "lpips_available": lpips_model is not None,
        "lpips_error": lpips_error,
        "ms_ssim_available": ms_ssim_module is not None,
        "ms_ssim_error": ms_ssim_error,
        "contact_sheet": contact_sheet,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "source_summary_csv": str(source_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "heavy_root": str(args.heavy_root),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary_rows, source_rows, package, report_md)
    print(json.dumps({"package": str(package_json), "report": str(report_md), "decision": decision, "new_render_count": new_renders, "contact_sheet": contact_sheet}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
