import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE179_TARGETS = REPO_ROOT / "experiments/stage179_broader_sampled_adaptive_validation_protocol/stage179_broader_validation_targets.csv"
DEFAULT_STAGE179_SCHEDULE_ROWS = REPO_ROOT / "experiments/stage179_broader_sampled_adaptive_validation_protocol/stage179_broader_validation_schedule_rows.csv"
DEFAULT_STAGE174_ROWS = REPO_ROOT / "experiments/stage174_medium_rendered_validation_execution/stage174_medium_validation_rows.csv"
DEFAULT_STAGE177_QUALITY_ROWS = REPO_ROOT / "experiments/stage177_selector_fixed_gap_psnr_comparison/stage177_final_quality_by_schedule.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage180_broader_sampled_adaptive_validation_execution"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage180_broader_sampled_adaptive_validation_execution")

sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage153_middle_multimetric_badcase_eval import load_metric_modules, mean, percentile  # noqa: E402
from scripts.run_stage167_adaptive_schedule_rendered_smoke import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_STAGE165_SCHEDULE_ROWS,
    build_task,
    empty_metric_row,
    prepare_schedule_maps,
    read_csv,
    rendered_metric_row,
    render_stage158,
    write_csv,
)
from scripts.run_stage174_medium_rendered_validation_execution import ROW_FIELDS as RENDER_ROW_FIELDS, add_baseline_deltas  # noqa: E402
from scripts.run_stage177_selector_fixed_gap_psnr_comparison import (  # noqa: E402
    DELTA_FIELDS,
    QUALITY_FIELDS,
    SUMMARY_FIELDS as FINAL_SUMMARY_FIELDS,
    TARGET_FIELDS,
    make_target_rows,
    render_keyframe_metrics,
    summarize_deltas,
    summarize_quality,
)
from scripts.run_stage6_export_real_anchor_dataset import load_model  # noqa: E402
from scripts.run_stage72_original_davis_baseline import DEFAULT_DAVIS_ROOT  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_DENSE_MANIFEST, build_dense_index  # noqa: E402


SOURCE_SUMMARY_FIELDS = ["row_source", "row_count", "rendered_count", "keyframe_marker_count"]


def read_plain_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def target_map(rows):
    return {row["target_key"]: row for row in rows}


def existing_render_map(rows):
    return {(row["target_key"], row["schedule"]): row for row in rows}


def existing_quality_map(rows):
    return {(row["target_key"], row["schedule"]): row for row in rows}


def add_context(row, target, category, row_source):
    out = dict(row)
    out["category"] = category
    out["source_task_id"] = target["source_task_id"]
    out["sequence"] = target["sequence"]
    out["target_index"] = int(target["target_index"])
    out["source_gap"] = int(float(target["source_gap"]))
    out["hard_quality_label"] = int(float(target["hard_quality_label"]))
    out["high_payload_label"] = int(float(target["high_payload_label"]))
    out["row_source"] = row_source
    return out


def numeric(row, key):
    value = row.get(key)
    if value in (None, "", "NA"):
        return None
    return float(value)


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


def render_summary(rows):
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
            "reused_count": sum(1 for row in group if str(row["row_source"]).startswith("stage174")),
            "new_render_count": sum(1 for row in group if row["row_source"] == "stage180_rendered"),
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


def existing_final_quality_row(row, target):
    out = dict(row)
    out["category"] = target["category"]
    out["sequence"] = target["sequence"]
    out["target_index"] = int(target["target_index"])
    for key in ("psnr", "ssim", "ms_ssim", "lpips", "payload_bytes"):
        out[key] = numeric(out, key) or 0.0
    return out


def final_quality_from_render(row):
    return {
        "target_key": row["target_key"],
        "category": row["category"],
        "sequence": row["sequence"],
        "target_index": int(row["target_index"]),
        "schedule": row["schedule"],
        "final_type": "middle_recovery",
        "row_source": row["row_source"],
        "psnr": numeric(row, "psnr"),
        "ssim": numeric(row, "ssim"),
        "ms_ssim": numeric(row, "ms_ssim"),
        "lpips": numeric(row, "lpips"),
        "payload_bytes": numeric(row, "payload_bytes") or 0.0,
        "status_note": "Stage180 rendered/reused middle recovery",
    }


def final_quality_from_keyframe(target, schedule, metrics, row_source):
    return {
        "target_key": target["target_key"],
        "category": target["category"],
        "sequence": target["sequence"],
        "target_index": int(target["target_index"]),
        "schedule": schedule,
        "final_type": "q12_target_keyframe",
        "row_source": row_source,
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "ms_ssim": metrics["ms_ssim"],
        "lpips": metrics["lpips"],
        "payload_bytes": 0.0,
        "status_note": "Target is transmitted as q12 keyframe; PSNR is keyframe render quality",
    }


def build_final_quality_rows(render_rows, targets, protocol_rows, existing_quality_rows, dense_index, device, opt, background, lpips_model, ms_ssim_module, davis_root, max_new_keyframe_metrics):
    render_by_key = {(row["target_key"], row["schedule"]): row for row in render_rows}
    quality_rows = []
    keyframe_cache = {}
    new_keyframe_metrics = 0
    for protocol in protocol_rows:
        key = (protocol["target_key"], protocol["schedule"])
        target = targets[protocol["target_key"]]
        render_row = render_by_key[key]
        if render_row["status"] == "rendered_middle_recovery":
            quality_rows.append(final_quality_from_render(render_row))
            continue
        existing = existing_quality_rows.get(key)
        if existing and existing.get("final_type") == "q12_target_keyframe":
            quality_rows.append(existing_final_quality_row(existing, target))
            continue
        if new_keyframe_metrics >= int(max_new_keyframe_metrics):
            raise RuntimeError(f"max_new_keyframe_metrics reached before completing protocol: {max_new_keyframe_metrics}")
        metrics = render_keyframe_metrics(target, dense_index, device, opt, background, lpips_model, ms_ssim_module, keyframe_cache, davis_root)
        quality_rows.append(final_quality_from_keyframe(target, protocol["schedule"], metrics, "stage180_q12_keyframe_metric"))
        new_keyframe_metrics += 1
        print(json.dumps({"new_keyframe_metrics": new_keyframe_metrics, "target": protocol["target_key"], "schedule": protocol["schedule"]}), flush=True)
    return quality_rows, new_keyframe_metrics


def write_report(render_summary_rows, source_rows, final_summary_rows, delta_rows, package, path):
    overall = next(row for row in delta_rows if row["group"] == "overall")
    lines = [
        "# Stage180 Broader Sampled Adaptive Validation Execution",
        "",
        "## Scope",
        "",
        "This executes the Stage179 90-target broader sampled protocol by reusing Stage174/177 rows and rendering only missing middle/keyframe metrics.",
        "",
        "## Execution Summary",
        "",
        f"- Protocol rows covered: `{package['output_row_count']} / {package['protocol_row_count']}`.",
        f"- New middle renders: `{package['new_render_count']}` / expected `{package['expected_new_render_count']}`.",
        f"- New q12 keyframe metrics: `{package['new_keyframe_metric_count']}` / expected `{package['expected_new_keyframe_metric_count']}`.",
        f"- Reused Stage174 rows: `{package['reused_stage174_row_count']}`.",
        f"- Keyframe marker rows: `{package['keyframe_marker_count']}`.",
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
        "## Overall Final Quality",
        "",
        "| schedule | targets | keyframes | middle recovery | mean PSNR | p10 PSNR | mean LPIPS | mean payload bytes/target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in final_summary_rows:
        if row["group"] != "overall":
            continue
        lines.append(
            f"| {row['schedule']} | {row['target_count']} | {row['keyframe_count']} | {row['middle_recovery_count']} | "
            f"{float(row['mean_psnr']):.6f} | {float(row['p10_psnr']):.6f} | {float(row['mean_lpips']):.6f} | {float(row['mean_payload_bytes_per_target']):.3f} |"
        )
    lines.extend([
        "",
        "## Paired Adaptive Delta",
        "",
        f"- Adaptive minus uniform gap8 PSNR: `{overall['mean_delta_psnr_vs_gap8']}` dB.",
        f"- Adaptive minus uniform gap4 PSNR: `{overall['mean_delta_psnr_vs_gap4']}` dB.",
        f"- Adaptive minus uniform gap8 LPIPS: `{overall['mean_delta_lpips_vs_gap8']}`.",
        f"- Adaptive minus uniform gap4 LPIPS: `{overall['mean_delta_lpips_vs_gap4']}`.",
        "",
        "## Category Delta",
        "",
        "| category | targets | adaptive keyframes | gap8 PSNR | adaptive PSNR | gap4 PSNR | delta vs gap8 | delta vs gap4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in delta_rows:
        if row["group"] == "overall":
            continue
        lines.append(
            f"| {row['group']} | {row['target_count']} | {row['adaptive_keyframe_count']} | "
            f"{float(row['mean_uniform_gap8_psnr']):.6f} | {float(row['mean_stage165_adaptive_psnr']):.6f} | {float(row['mean_uniform_gap4_psnr']):.6f} | "
            f"{float(row['mean_delta_psnr_vs_gap8']):.6f} | {float(row['mean_delta_psnr_vs_gap4']):.6f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- This is still sampled broader validation, not final full-sequence RD.",
        "- Adaptive keyframe rows use q12 keyframe reconstruction quality, not Stage158 middle-recovery quality.",
        "- Stage158 middle recovery remains fixed as `streamsplat_guided_half_anchor_entropy_residual_v1`.",
        "",
        "## Outputs",
        "",
        f"- Render rows CSV: `{package['render_rows_csv']}`",
        f"- Final quality CSV: `{package['final_quality_csv']}`",
        f"- Target delta CSV: `{package['target_delta_csv']}`",
        f"- Final summary CSV: `{package['final_summary_csv']}`",
        f"- Category delta CSV: `{package['category_delta_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage179_targets", type=Path, default=DEFAULT_STAGE179_TARGETS)
    parser.add_argument("--stage179_schedule_rows", type=Path, default=DEFAULT_STAGE179_SCHEDULE_ROWS)
    parser.add_argument("--stage174_rows", type=Path, default=DEFAULT_STAGE174_ROWS)
    parser.add_argument("--stage177_quality_rows", type=Path, default=DEFAULT_STAGE177_QUALITY_ROWS)
    parser.add_argument("--stage165_schedule_rows", type=Path, default=DEFAULT_STAGE165_SCHEDULE_ROWS)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--max_new_renders", type=int, default=120)
    parser.add_argument("--max_new_keyframe_metrics", type=int, default=120)
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
    targets = target_map(read_plain_csv(args.stage179_targets))
    protocol_rows = read_plain_csv(args.stage179_schedule_rows)
    existing_render_rows = existing_render_map(read_plain_csv(args.stage174_rows))
    existing_quality_rows = existing_quality_map(read_plain_csv(args.stage177_quality_rows))
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
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    lpips_model, lpips_error, ms_ssim_module, ms_ssim_error = load_metric_modules(device, args.disable_lpips, args.disable_ms_ssim)
    rows = []
    image_cache = {}
    new_renders = 0
    for protocol in protocol_rows:
        key = (protocol["target_key"], protocol["schedule"])
        target = targets[protocol["target_key"]]
        if key in existing_render_rows:
            source = f"stage174:{existing_render_rows[key].get('row_source', '')}"
            rows.append(add_context(existing_render_rows[key], target, protocol["category"], source))
            continue
        task, left, right = build_task(target, protocol["schedule"], schedule_maps, args.davis_root)
        if protocol["expected_status"] == "target_keyframe_no_middle_render" or task is None:
            rows.append(add_context(empty_metric_row(target, protocol["schedule"], left, right, "target is transmitted keyframe under this schedule"), target, protocol["category"], "stage180_keyframe_marker"))
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
        rows.append(add_context(rendered_metric_row(target, protocol["schedule"], task, selected_half, selected_metrics, original_metrics, payload_bytes), target, protocol["category"], "stage180_rendered"))
        new_renders += 1
        print(json.dumps({"new_renders": new_renders, "target": protocol["target_key"], "schedule": protocol["schedule"]}), flush=True)
    rows = add_baseline_deltas(rows)
    quality_rows, new_keyframe_metrics = build_final_quality_rows(
        rows,
        targets,
        protocol_rows,
        existing_quality_rows,
        dense_index,
        device,
        opt,
        background,
        lpips_model,
        ms_ssim_module,
        args.davis_root,
        args.max_new_keyframe_metrics,
    )
    target_rows = make_target_rows(quality_rows)
    final_summary_rows = summarize_quality(quality_rows)
    delta_rows = summarize_deltas(target_rows)
    render_summary_rows = render_summary(rows)
    source_rows = source_summary(rows)
    render_rows_csv = args.summary_root / "stage180_broader_validation_rows.csv"
    render_summary_csv = args.summary_root / "stage180_broader_validation_render_summary.csv"
    source_summary_csv = args.summary_root / "stage180_broader_validation_source_summary.csv"
    final_quality_csv = args.summary_root / "stage180_final_quality_by_schedule.csv"
    target_delta_csv = args.summary_root / "stage180_adaptive_vs_fixed_gap_target_deltas.csv"
    final_summary_csv = args.summary_root / "stage180_final_quality_summary.csv"
    category_delta_csv = args.summary_root / "stage180_category_delta_summary.csv"
    package_json = args.summary_root / "stage180_broader_sampled_adaptive_validation_execution_package.json"
    report_md = args.summary_root / "stage180_broader_sampled_adaptive_validation_execution_report.md"
    write_csv(rows, render_rows_csv, RENDER_ROW_FIELDS)
    write_csv(render_summary_rows, render_summary_csv, [
        "category", "schedule", "row_count", "rendered_count", "keyframe_marker_count", "reused_count", "new_render_count",
        "mean_psnr", "p10_psnr", "mean_ssim", "mean_ms_ssim", "mean_lpips", "p90_lpips", "mean_original_psnr",
        "mean_original_lpips", "mean_payload_bytes", "mean_side_mib_per_intermediate", "mean_delta_psnr_vs_uniform_gap8",
        "mean_delta_lpips_vs_uniform_gap8",
    ])
    write_csv(source_rows, source_summary_csv, SOURCE_SUMMARY_FIELDS)
    write_csv(quality_rows, final_quality_csv, QUALITY_FIELDS)
    write_csv(target_rows, target_delta_csv, TARGET_FIELDS)
    write_csv(final_summary_rows, final_summary_csv, FINAL_SUMMARY_FIELDS)
    write_csv(delta_rows, category_delta_csv, DELTA_FIELDS)
    overall_delta = next(row for row in delta_rows if row["group"] == "overall")
    expected_new_renders = sum(int(row["requires_stage180_render"]) for row in protocol_rows)
    expected_new_keyframe_metrics = sum(int(row["requires_stage180_keyframe_metric"]) for row in protocol_rows)
    complete = len(rows) == len(protocol_rows) and new_renders == expected_new_renders and new_keyframe_metrics == expected_new_keyframe_metrics
    decision = "broader_validation_ready_for_review" if complete else "broader_validation_incomplete"
    package = {
        "stage": 180,
        "status": "broader_sampled_adaptive_validation_execution_packaged",
        "decision": decision,
        "policy": "streamsplat_guided_half_anchor_entropy_residual_v1_plus_rgb_motion_adaptive_schedule",
        "protocol_row_count": len(protocol_rows),
        "output_row_count": len(rows),
        "target_count": len(targets),
        "final_quality_row_count": len(quality_rows),
        "new_render_count": new_renders,
        "expected_new_render_count": expected_new_renders,
        "new_keyframe_metric_count": new_keyframe_metrics,
        "expected_new_keyframe_metric_count": expected_new_keyframe_metrics,
        "reused_stage174_row_count": sum(1 for row in rows if str(row["row_source"]).startswith("stage174")),
        "keyframe_marker_count": sum(1 for row in rows if row["status"] == "target_keyframe_no_middle_render"),
        "mean_delta_psnr_vs_uniform_gap8": overall_delta["mean_delta_psnr_vs_gap8"],
        "mean_delta_psnr_vs_uniform_gap4": overall_delta["mean_delta_psnr_vs_gap4"],
        "mean_delta_lpips_vs_uniform_gap8": overall_delta["mean_delta_lpips_vs_gap8"],
        "mean_delta_lpips_vs_uniform_gap4": overall_delta["mean_delta_lpips_vs_gap4"],
        "render_summary_rows": render_summary_rows,
        "source_summary_rows": source_rows,
        "final_summary_rows": final_summary_rows,
        "category_delta_rows": delta_rows,
        "lpips_available": lpips_model is not None,
        "lpips_error": lpips_error,
        "ms_ssim_available": ms_ssim_module is not None,
        "ms_ssim_error": ms_ssim_error,
        "render_rows_csv": str(render_rows_csv),
        "render_summary_csv": str(render_summary_csv),
        "source_summary_csv": str(source_summary_csv),
        "final_quality_csv": str(final_quality_csv),
        "target_delta_csv": str(target_delta_csv),
        "final_summary_csv": str(final_summary_csv),
        "category_delta_csv": str(category_delta_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "heavy_root": str(args.heavy_root),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(render_summary_rows, source_rows, final_summary_rows, delta_rows, package, report_md)
    print(json.dumps({
        "package": str(package_json),
        "decision": decision,
        "new_render_count": new_renders,
        "new_keyframe_metric_count": new_keyframe_metrics,
        "mean_delta_psnr_vs_uniform_gap8": overall_delta["mean_delta_psnr_vs_gap8"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
