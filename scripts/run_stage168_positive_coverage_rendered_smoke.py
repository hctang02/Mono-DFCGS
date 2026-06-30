import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage168_positive_coverage_rendered_smoke"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage168_positive_coverage_rendered_smoke")


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage153_middle_multimetric_badcase_eval import format_optional, load_metric_modules  # noqa: E402
from scripts.run_stage167_adaptive_schedule_rendered_smoke import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_STAGE165_SCHEDULE_ROWS,
    DEFAULT_STAGE166_ROWS,
    DEFAULT_STAGE166_SMOKE,
    ROW_FIELDS,
    SCHEDULES,
    SUMMARY_FIELDS,
    TARGET_FIELDS,
    add_baseline_deltas,
    build_task,
    empty_metric_row,
    load_smoke_ranks,
    prepare_schedule_maps,
    put_label,
    read_csv,
    rendered_metric_row,
    render_stage158,
    summarize,
    target_key,
    write_csv,
)
from scripts.run_stage6_export_real_anchor_dataset import load_model  # noqa: E402
from scripts.run_stage72_original_davis_baseline import DEFAULT_DAVIS_ROOT  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_DENSE_MANIFEST, build_dense_index  # noqa: E402


POSITIVE_SEQUENCES = ["motocross-jump", "cows", "camel", "scooter-black", "india", "shooting", "car-roundabout"]


def select_promoted_targets(stage166_rows, smoke_ranks, max_targets):
    best = {}
    for row in stage166_rows:
        if not int(row["stage165_adaptive_target_is_keyframe"]):
            continue
        hard = int(row["hard_quality_label"])
        high_payload = int(row["high_payload_label"])
        if not hard and not high_payload:
            continue
        sequence = row["sequence"]
        seq_priority = 40 if sequence in POSITIVE_SEQUENCES else 0
        smoke_rank = smoke_ranks.get(sequence, 99)
        smoke_bonus = max(0, 24 - smoke_rank) if smoke_rank != 99 else 0
        score = seq_priority + 100 * hard + 60 * high_payload + smoke_bonus + float(row["payload_bytes"]) / 10000.0
        key = target_key(sequence, row["target_index"])
        item = dict(row)
        item["selection_score"] = score
        reasons = []
        if sequence in POSITIVE_SEQUENCES:
            reasons.append("positive_sequence")
        if hard:
            reasons.append("hard_quality_promoted")
        if high_payload:
            reasons.append("high_payload_promoted")
        if sequence in smoke_ranks:
            reasons.append("stage166_smoke_sequence")
        item["selection_reason"] = ";".join(reasons)
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


def save_contact_sheet(targets, rows, image_cache, heavy_root):
    heavy_root.mkdir(parents=True, exist_ok=True)
    by_target = {}
    for row in rows:
        by_target.setdefault(row["target_key"], {})[row["schedule"]] = row
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
                metric = by_target[key].get(label, {})
                img = image_cache.get((sequence, target_index, label), target_img).copy()
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
    path = heavy_root / "stage168_positive_coverage_rendered_smoke_contact_sheet.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return str(path)


def write_report(summary_rows, targets, package, path):
    promoted = next(row for row in summary_rows if row["schedule"] == "stage165_adaptive")
    gap8 = next(row for row in summary_rows if row["schedule"] == "uniform_gap8")
    lines = [
        "# Stage168 Positive-Coverage Rendered Smoke",
        "",
        "## Scope",
        "",
        "This smoke targets Stage165 adaptive rows promoted to keyframes, so adaptive rows are intentionally marked as no-middle-render keyframes rather than rendered middle predictions.",
        "Uniform gap8 recovery is rendered to measure the middle-frame payload/quality that adaptive promotion avoids.",
        "",
        "## Summary",
        "",
        "| schedule | targets | rendered | keyframes | PSNR | SSIM | MS-SSIM | LPIPS | payload bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['schedule']} | {row['target_count']} | {row['rendered_count']} | {row['target_keyframe_count']} | "
            f"{format_optional(row['mean_psnr'])} | {format_optional(row['mean_ssim'])} | {format_optional(row['mean_ms_ssim'])} | "
            f"{format_optional(row['mean_lpips'])} | {format_optional(row['mean_payload_bytes'])} |"
        )
    lines.extend([
        "",
        "## Promotion Takeaway",
        "",
        f"- Adaptive promoted targets: `{promoted['target_keyframe_count']}` / `{promoted['target_count']}`.",
        f"- Uniform gap8 rendered PSNR/LPIPS on the same targets: `{format_optional(gap8['mean_psnr'])}` / `{format_optional(gap8['mean_lpips'])}`.",
        f"- Uniform gap8 mean middle residual payload avoided by promotion: `{format_optional(gap8['mean_payload_bytes'])}` bytes.",
        "",
        "## Targets",
        "",
        "| rank | sequence | target | hard | high payload | reason | score |",
        "|---:|---|---:|---:|---:|---|---:|",
    ])
    for row in targets:
        lines.append(
            f"| {row['rank']} | {row['sequence']} | {row['target_index']} | {row['hard_quality_label']} | "
            f"{row['high_payload_label']} | {row['selection_reason']} | {float(row['selection_score']):.3f} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Heavy contact sheet: `{package['contact_sheet']}`.",
        "- This complements Stage167; a broader validation should combine positive promotions and remaining false negatives.",
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
    targets = select_promoted_targets(read_csv(args.stage166_rows), smoke_ranks, args.max_targets)
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
            print(json.dumps({"rendered_or_marked": len(rows), "target": target_row["target_key"], "schedule": schedule}), flush=True)
    rows = add_baseline_deltas(rows)
    summary_rows = summarize(rows)
    contact_sheet = save_contact_sheet(targets, rows, image_cache, args.heavy_root)
    targets_csv = args.summary_root / "stage168_positive_coverage_targets.csv"
    rows_csv = args.summary_root / "stage168_positive_coverage_rows.csv"
    summary_csv = args.summary_root / "stage168_positive_coverage_summary.csv"
    package_json = args.summary_root / "stage168_positive_coverage_rendered_smoke_package.json"
    report_md = args.summary_root / "stage168_positive_coverage_rendered_smoke_report.md"
    write_csv(targets, targets_csv, TARGET_FIELDS)
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    adaptive = next(row for row in summary_rows if row["schedule"] == "stage165_adaptive")
    gap8 = next(row for row in summary_rows if row["schedule"] == "uniform_gap8")
    decision = "positive_promotions_confirmed_for_broader_validation" if int(adaptive["target_keyframe_count"]) == len(targets) and gap8["mean_payload_bytes"] is not None and float(gap8["mean_payload_bytes"]) > 200000.0 else "positive_smoke_needs_inspection"
    package = {
        "stage": 168,
        "status": "positive_coverage_rendered_smoke_packaged",
        "decision": decision,
        "policy": "streamsplat_guided_half_anchor_entropy_residual_v1",
        "target_count": len(targets),
        "positive_sequences": POSITIVE_SEQUENCES,
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
