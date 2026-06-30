import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE174_ROWS = REPO_ROOT / "experiments/stage174_medium_rendered_validation_execution/stage174_medium_validation_rows.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage177_selector_fixed_gap_psnr_comparison"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage153_middle_multimetric_badcase_eval import load_metric_modules, mean, percentile  # noqa: E402
from scripts.run_stage155_streamsplat_base_sideinfo_upper_bound import compute_metrics, render_static_anchor  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import load_rgb  # noqa: E402
from scripts.run_stage72_original_davis_baseline import DEFAULT_DAVIS_ROOT  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_DENSE_MANIFEST, build_dense_index, load_anchor  # noqa: E402


SCHEDULES = ["uniform_gap8", "stage165_adaptive", "uniform_gap4"]
QUALITY_FIELDS = [
    "target_key", "category", "sequence", "target_index", "schedule", "final_type", "row_source",
    "psnr", "ssim", "ms_ssim", "lpips", "payload_bytes", "status_note",
]
TARGET_FIELDS = [
    "target_key", "category", "sequence", "target_index", "adaptive_final_type",
    "uniform_gap8_psnr", "stage165_adaptive_psnr", "uniform_gap4_psnr",
    "adaptive_delta_psnr_vs_uniform_gap8", "adaptive_delta_psnr_vs_uniform_gap4",
    "uniform_gap8_lpips", "stage165_adaptive_lpips", "uniform_gap4_lpips",
    "adaptive_delta_lpips_vs_uniform_gap8", "adaptive_delta_lpips_vs_uniform_gap4",
    "uniform_gap8_payload_bytes", "stage165_adaptive_payload_bytes", "uniform_gap4_payload_bytes",
]
SUMMARY_FIELDS = [
    "group", "schedule", "target_count", "keyframe_count", "middle_recovery_count", "mean_psnr", "p10_psnr",
    "mean_ssim", "mean_ms_ssim", "mean_lpips", "p90_lpips", "mean_payload_bytes_per_target",
]
DELTA_FIELDS = [
    "group", "target_count", "adaptive_keyframe_count", "mean_uniform_gap8_psnr", "mean_stage165_adaptive_psnr",
    "mean_uniform_gap4_psnr", "mean_delta_psnr_vs_gap8", "mean_delta_psnr_vs_gap4", "mean_delta_lpips_vs_gap8",
    "mean_delta_lpips_vs_gap4",
]


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def frame_rgb_path(davis_root, sequence, index):
    return davis_root / "JPEGImages" / "Full-Resolution" / sequence / f"{int(index):05d}.jpg"


def numeric(row, key):
    value = row.get(key)
    if value in (None, "", "NA"):
        return None
    return float(value)


def group_stage174_rows(rows):
    out = {}
    for row in rows:
        out[(row["target_key"], row["schedule"])] = row
    return out


def target_records(rows):
    out = {}
    for row in rows:
        key = row["target_key"]
        if key not in out:
            out[key] = {
                "target_key": key,
                "category": row["category"],
                "sequence": row["sequence"],
                "target_index": int(row["target_index"]),
            }
    return out


def render_keyframe_metrics(target, dense_index, device, opt, background, lpips_model, ms_ssim_module, cache, davis_root):
    key = (target["sequence"], int(target["target_index"]))
    if key in cache:
        return cache[key]
    dense_key = ("DAVIS", "val", target["sequence"], int(target["target_index"]))
    target_item, target_side = dense_index[dense_key]
    anchor = load_anchor(target_item, target_side, device, bits=12, cache=None)
    render = render_static_anchor(anchor, background, opt)
    target_rgb = load_rgb(frame_rgb_path(davis_root, target["sequence"], target["target_index"]), opt.image_height, opt.image_width, device)
    metrics = compute_metrics(render, target_rgb, lpips_model, ms_ssim_module)
    cache[key] = metrics
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def final_quality_row(row, target, dense_index, device, opt, background, lpips_model, ms_ssim_module, keyframe_cache, davis_root):
    if row["status"] == "rendered_middle_recovery":
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
            "status_note": "Stage174 rendered/reused middle recovery",
        }
    metrics = render_keyframe_metrics(target, dense_index, device, opt, background, lpips_model, ms_ssim_module, keyframe_cache, davis_root)
    return {
        "target_key": row["target_key"],
        "category": row["category"],
        "sequence": row["sequence"],
        "target_index": int(row["target_index"]),
        "schedule": row["schedule"],
        "final_type": "q12_target_keyframe",
        "row_source": row["row_source"],
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "ms_ssim": metrics["ms_ssim"],
        "lpips": metrics["lpips"],
        "payload_bytes": 0.0,
        "status_note": "Target is transmitted as q12 keyframe; PSNR is keyframe render quality",
    }


def make_target_rows(quality_rows):
    by_target = defaultdict(dict)
    meta = {}
    for row in quality_rows:
        by_target[row["target_key"]][row["schedule"]] = row
        meta[row["target_key"]] = row
    out = []
    for key in sorted(by_target):
        sched = by_target[key]
        gap8 = sched["uniform_gap8"]
        adaptive = sched["stage165_adaptive"]
        gap4 = sched["uniform_gap4"]
        out.append({
            "target_key": key,
            "category": meta[key]["category"],
            "sequence": meta[key]["sequence"],
            "target_index": int(meta[key]["target_index"]),
            "adaptive_final_type": adaptive["final_type"],
            "uniform_gap8_psnr": gap8["psnr"],
            "stage165_adaptive_psnr": adaptive["psnr"],
            "uniform_gap4_psnr": gap4["psnr"],
            "adaptive_delta_psnr_vs_uniform_gap8": adaptive["psnr"] - gap8["psnr"],
            "adaptive_delta_psnr_vs_uniform_gap4": adaptive["psnr"] - gap4["psnr"],
            "uniform_gap8_lpips": gap8["lpips"],
            "stage165_adaptive_lpips": adaptive["lpips"],
            "uniform_gap4_lpips": gap4["lpips"],
            "adaptive_delta_lpips_vs_uniform_gap8": adaptive["lpips"] - gap8["lpips"] if adaptive["lpips"] is not None and gap8["lpips"] is not None else None,
            "adaptive_delta_lpips_vs_uniform_gap4": adaptive["lpips"] - gap4["lpips"] if adaptive["lpips"] is not None and gap4["lpips"] is not None else None,
            "uniform_gap8_payload_bytes": gap8["payload_bytes"],
            "stage165_adaptive_payload_bytes": adaptive["payload_bytes"],
            "uniform_gap4_payload_bytes": gap4["payload_bytes"],
        })
    return out


def summarize_quality(quality_rows):
    out = []
    groups = {("overall", schedule): [row for row in quality_rows if row["schedule"] == schedule] for schedule in SCHEDULES}
    for row in quality_rows:
        groups.setdefault((row["category"], row["schedule"]), []).append(row)
    for (group, schedule), rows in sorted(groups.items()):
        out.append({
            "group": group,
            "schedule": schedule,
            "target_count": len(rows),
            "keyframe_count": sum(1 for row in rows if row["final_type"] == "q12_target_keyframe"),
            "middle_recovery_count": sum(1 for row in rows if row["final_type"] == "middle_recovery"),
            "mean_psnr": mean(row["psnr"] for row in rows),
            "p10_psnr": percentile((row["psnr"] for row in rows), 10),
            "mean_ssim": mean(row["ssim"] for row in rows),
            "mean_ms_ssim": mean(row["ms_ssim"] for row in rows),
            "mean_lpips": mean(row["lpips"] for row in rows),
            "p90_lpips": percentile((row["lpips"] for row in rows), 90),
            "mean_payload_bytes_per_target": mean(row["payload_bytes"] for row in rows),
        })
    return out


def summarize_deltas(target_rows):
    groups = {"overall": target_rows}
    for row in target_rows:
        groups.setdefault(row["category"], []).append(row)
    out = []
    for group, rows in sorted(groups.items()):
        out.append({
            "group": group,
            "target_count": len(rows),
            "adaptive_keyframe_count": sum(1 for row in rows if row["adaptive_final_type"] == "q12_target_keyframe"),
            "mean_uniform_gap8_psnr": mean(row["uniform_gap8_psnr"] for row in rows),
            "mean_stage165_adaptive_psnr": mean(row["stage165_adaptive_psnr"] for row in rows),
            "mean_uniform_gap4_psnr": mean(row["uniform_gap4_psnr"] for row in rows),
            "mean_delta_psnr_vs_gap8": mean(row["adaptive_delta_psnr_vs_uniform_gap8"] for row in rows),
            "mean_delta_psnr_vs_gap4": mean(row["adaptive_delta_psnr_vs_uniform_gap4"] for row in rows),
            "mean_delta_lpips_vs_gap8": mean(row["adaptive_delta_lpips_vs_uniform_gap8"] for row in rows),
            "mean_delta_lpips_vs_gap4": mean(row["adaptive_delta_lpips_vs_uniform_gap4"] for row in rows),
        })
    return out


def row_for(summary_rows, group, schedule):
    for row in summary_rows:
        if row["group"] == group and row["schedule"] == schedule:
            return row
    raise KeyError((group, schedule))


def write_report(summary_rows, delta_rows, package, path):
    overall_delta = next(row for row in delta_rows if row["group"] == "overall")
    lines = [
        "# Stage177 Selector Fixed-Gap PSNR Comparison",
        "",
        "## Scope",
        "",
        "This compares final target quality for fixed uniform gap8, Stage165 adaptive, and fixed uniform gap4 on the Stage174 medium target set.",
        "Rendered middle-recovery rows reuse Stage174 metrics. Target-keyframe rows are evaluated by rendering the target q12 keyframe anchor.",
        "",
        "## Overall PSNR",
        "",
        "| schedule | targets | keyframes | middle recovery | mean PSNR | p10 PSNR | mean LPIPS | mean payload bytes/target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for schedule in SCHEDULES:
        row = row_for(summary_rows, "overall", schedule)
        lines.append(
            f"| {schedule} | {row['target_count']} | {row['keyframe_count']} | {row['middle_recovery_count']} | "
            f"{row['mean_psnr']:.6f} | {row['p10_psnr']:.6f} | {row['mean_lpips']:.6f} | {row['mean_payload_bytes_per_target']:.3f} |"
        )
    lines.extend([
        "",
        "## Paired Adaptive Delta",
        "",
        f"- Adaptive minus uniform gap8 PSNR: `{overall_delta['mean_delta_psnr_vs_gap8']:.6f}` dB.",
        f"- Adaptive minus uniform gap4 PSNR: `{overall_delta['mean_delta_psnr_vs_gap4']:.6f}` dB.",
        f"- Adaptive keyframe targets: `{overall_delta['adaptive_keyframe_count']}` / `{overall_delta['target_count']}`.",
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
            f"{row['mean_uniform_gap8_psnr']:.6f} | {row['mean_stage165_adaptive_psnr']:.6f} | {row['mean_uniform_gap4_psnr']:.6f} | "
            f"{row['mean_delta_psnr_vs_gap8']:.6f} | {row['mean_delta_psnr_vs_gap4']:.6f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- This table answers the fixed-gap versus selector question in PSNR terms on the Stage174 medium set.",
        "- Adaptive keyframe improvements are q12 keyframe reconstruction quality, not Stage158 middle recovery quality.",
        "- Residual rows where adaptive keeps the same segment as gap8 should match gap8 by construction.",
        "- This remains a sampled medium-set comparison, not final full-sequence RD.",
        "",
        "## Outputs",
        "",
        f"- Per-schedule quality CSV: `{package['quality_csv']}`",
        f"- Per-target delta CSV: `{package['target_delta_csv']}`",
        f"- Schedule summary CSV: `{package['summary_csv']}`",
        f"- Category delta CSV: `{package['category_delta_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage174_rows", type=Path, default=DEFAULT_STAGE174_ROWS)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--disable_lpips", action="store_true")
    parser.add_argument("--disable_ms_ssim", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.stage174_rows)
    grouped = group_stage174_rows(rows)
    targets = target_records(rows)
    dense_index = build_dense_index(args.dense_manifest, ["val"])
    device = torch.device(args.device)
    opt = Options()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    lpips_model, lpips_error, ms_ssim_module, ms_ssim_error = load_metric_modules(device, args.disable_lpips, args.disable_ms_ssim)
    keyframe_cache = {}
    quality_rows = []
    for key in sorted(targets):
        target = targets[key]
        for schedule in SCHEDULES:
            quality_rows.append(final_quality_row(grouped[(key, schedule)], target, dense_index, device, opt, background, lpips_model, ms_ssim_module, keyframe_cache, args.davis_root))
    target_rows = make_target_rows(quality_rows)
    summary_rows = summarize_quality(quality_rows)
    delta_rows = summarize_deltas(target_rows)
    quality_csv = args.output_root / "stage177_final_quality_by_schedule.csv"
    target_delta_csv = args.output_root / "stage177_adaptive_vs_fixed_gap_target_deltas.csv"
    summary_csv = args.output_root / "stage177_final_quality_summary.csv"
    category_delta_csv = args.output_root / "stage177_category_delta_summary.csv"
    package_json = args.output_root / "stage177_selector_fixed_gap_psnr_comparison_package.json"
    report_md = args.output_root / "stage177_selector_fixed_gap_psnr_comparison_report.md"
    write_csv(quality_rows, quality_csv, QUALITY_FIELDS)
    write_csv(target_rows, target_delta_csv, TARGET_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(delta_rows, category_delta_csv, DELTA_FIELDS)
    overall_delta = next(row for row in delta_rows if row["group"] == "overall")
    package = {
        "stage": 177,
        "status": "selector_fixed_gap_psnr_comparison_packaged",
        "comparison_scope": "stage174_medium_targets",
        "target_count": len(targets),
        "quality_row_count": len(quality_rows),
        "adaptive_keyframe_count": overall_delta["adaptive_keyframe_count"],
        "mean_delta_psnr_vs_uniform_gap8": overall_delta["mean_delta_psnr_vs_gap8"],
        "mean_delta_psnr_vs_uniform_gap4": overall_delta["mean_delta_psnr_vs_gap4"],
        "lpips_available": lpips_model is not None,
        "lpips_error": lpips_error,
        "ms_ssim_available": ms_ssim_module is not None,
        "ms_ssim_error": ms_ssim_error,
        "quality_csv": str(quality_csv),
        "target_delta_csv": str(target_delta_csv),
        "summary_csv": str(summary_csv),
        "category_delta_csv": str(category_delta_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary_rows, delta_rows, package, report_md)
    print(json.dumps({"package": str(package_json), "mean_delta_psnr_vs_uniform_gap8": overall_delta["mean_delta_psnr_vs_gap8"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
