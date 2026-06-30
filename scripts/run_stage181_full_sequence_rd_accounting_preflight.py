import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE165_SCHEDULE_ROWS = REPO_ROOT / "experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_adaptive_schedule_rows.csv"
DEFAULT_STAGE172_PACKAGE = REPO_ROOT / "experiments/stage172_keyframe_rate_accounting_audit/stage172_keyframe_rate_accounting_audit_package.json"
DEFAULT_STAGE180_FINAL_SUMMARY = REPO_ROOT / "experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_final_quality_summary.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage181_full_sequence_rd_accounting_preflight"

SCHEDULES = ["uniform_gap8", "stage165_adaptive", "uniform_gap4"]
MIB = 1024.0 * 1024.0

KEYFRAME_FIELDS = [
    "schedule",
    "sequence_count",
    "total_frames",
    "keyframe_count",
    "keyframe_ratio",
    "delta_keyframes_vs_gap8",
    "delta_keyframes_vs_gap4",
    "main_anchor_mib_per_frame_proxy",
    "main_anchor_total_mib_proxy",
    "metadata_bits",
    "metadata_bytes",
    "metadata_mib_total",
    "metadata_mib_per_frame",
    "keyframe_metadata_scope",
]
RESIDUAL_FIELDS = [
    "schedule",
    "validation_scope",
    "target_count",
    "keyframe_count",
    "middle_recovery_count",
    "mean_payload_bytes_per_target",
    "sampled_residual_mib_per_target",
    "mean_psnr",
    "mean_lpips",
    "residual_scope",
]
TOTAL_FIELDS = [
    "schedule",
    "main_anchor_mib_per_frame_proxy",
    "metadata_mib_per_frame_exact",
    "stage180_residual_mib_per_target_proxy",
    "stage180_broader_total_proxy_mib_per_frame",
    "delta_total_proxy_vs_gap8",
    "delta_total_proxy_vs_gap4",
    "mean_psnr_stage180",
    "mean_lpips_stage180",
    "rate_scope",
]
REQUIREMENT_FIELDS = ["item", "status", "next_required_work"]


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def f(row, key):
    return float(row[key])


def i(row, key):
    return int(float(row[key]))


def ceil_bytes(bits):
    return (int(bits) + 7) // 8


def stage172_components(package):
    return {row["schedule"]: row for row in package["component_rows"]}


def stage180_overall(summary_rows):
    return {row["schedule"]: row for row in summary_rows if row["group"] == "overall"}


def sum_keyframes(schedule_rows, schedule):
    key = "adaptive_keyframe_count" if schedule == "stage165_adaptive" else f"{schedule}_keyframe_count"
    return sum(i(row, key) for row in schedule_rows)


def sum_metadata_bits(schedule_rows, schedule):
    if schedule == "stage165_adaptive":
        return sum(i(row, "metadata_bits") for row in schedule_rows)
    return 8


def build_keyframe_rows(stage165_rows, stage172):
    components = stage172_components(stage172)
    total_frames = sum(i(row, "total_frames") for row in stage165_rows)
    sequence_count = len(stage165_rows)
    gap8_keyframes = sum_keyframes(stage165_rows, "uniform_gap8")
    gap4_keyframes = sum_keyframes(stage165_rows, "uniform_gap4")
    out = []
    for schedule in SCHEDULES:
        keyframes = sum_keyframes(stage165_rows, schedule)
        metadata_bits = sum_metadata_bits(stage165_rows, schedule)
        metadata_bytes = ceil_bytes(metadata_bits)
        metadata_mib_total = metadata_bytes / MIB
        main_anchor = float(components[schedule]["main_anchor_mib_per_frame_proxy"])
        out.append({
            "schedule": schedule,
            "sequence_count": sequence_count,
            "total_frames": total_frames,
            "keyframe_count": keyframes,
            "keyframe_ratio": keyframes / float(total_frames),
            "delta_keyframes_vs_gap8": keyframes - gap8_keyframes,
            "delta_keyframes_vs_gap4": keyframes - gap4_keyframes,
            "main_anchor_mib_per_frame_proxy": main_anchor,
            "main_anchor_total_mib_proxy": main_anchor * total_frames,
            "metadata_bits": metadata_bits,
            "metadata_bytes": metadata_bytes,
            "metadata_mib_total": metadata_mib_total,
            "metadata_mib_per_frame": metadata_mib_total / float(total_frames),
            "keyframe_metadata_scope": "exact_schedule_counts_main_anchor_proxy_payload",
        })
    return out


def build_residual_rows(stage180_rows):
    overall = stage180_overall(stage180_rows)
    out = []
    for schedule in SCHEDULES:
        row = overall[schedule]
        mean_payload = f(row, "mean_payload_bytes_per_target")
        out.append({
            "schedule": schedule,
            "validation_scope": "stage180_broader_sampled_90_targets",
            "target_count": i(row, "target_count"),
            "keyframe_count": i(row, "keyframe_count"),
            "middle_recovery_count": i(row, "middle_recovery_count"),
            "mean_payload_bytes_per_target": mean_payload,
            "sampled_residual_mib_per_target": mean_payload / MIB,
            "mean_psnr": f(row, "mean_psnr"),
            "mean_lpips": f(row, "mean_lpips"),
            "residual_scope": "sampled_estimate_not_full_sequence_payload_measurement",
        })
    return out


def build_total_rows(keyframe_rows, residual_rows):
    keyframes = {row["schedule"]: row for row in keyframe_rows}
    residuals = {row["schedule"]: row for row in residual_rows}
    totals = {}
    out = []
    for schedule in SCHEDULES:
        key = keyframes[schedule]
        residual = residuals[schedule]
        total = float(key["main_anchor_mib_per_frame_proxy"]) + float(key["metadata_mib_per_frame"]) + float(residual["sampled_residual_mib_per_target"])
        totals[schedule] = total
        out.append({
            "schedule": schedule,
            "main_anchor_mib_per_frame_proxy": key["main_anchor_mib_per_frame_proxy"],
            "metadata_mib_per_frame_exact": key["metadata_mib_per_frame"],
            "stage180_residual_mib_per_target_proxy": residual["sampled_residual_mib_per_target"],
            "stage180_broader_total_proxy_mib_per_frame": total,
            "delta_total_proxy_vs_gap8": None,
            "delta_total_proxy_vs_gap4": None,
            "mean_psnr_stage180": residual["mean_psnr"],
            "mean_lpips_stage180": residual["mean_lpips"],
            "rate_scope": "full_sequence_keyframe_metadata_plus_stage180_sampled_residual_proxy",
        })
    for row in out:
        row["delta_total_proxy_vs_gap8"] = row["stage180_broader_total_proxy_mib_per_frame"] - totals["uniform_gap8"]
        row["delta_total_proxy_vs_gap4"] = row["stage180_broader_total_proxy_mib_per_frame"] - totals["uniform_gap4"]
    return out


def build_requirements():
    return [
        {
            "item": "full_sequence_keyframe_indices",
            "status": "counted_exact_from_stage165_schedule_rows",
            "next_required_work": "none_for_schedule_counting",
        },
        {
            "item": "schedule_metadata",
            "status": "counted_exact_as_327_bytes_for_adaptive_and_1_byte_mode_for_uniforms",
            "next_required_work": "replace with actual bitstream syntax if final codec changes",
        },
        {
            "item": "main_anchor_payload",
            "status": "proxy_from_stage172_interpolated_anchor_accounting",
            "next_required_work": "measure actual q12 keyframe bitstreams for every transmitted keyframe before paper-level RD",
        },
        {
            "item": "stage158_residual_payload",
            "status": "stage180_broader_sampled_estimate",
            "next_required_work": "run all-frame/full-sequence residual payload encode for every non-keyframe recovered frame",
        },
        {
            "item": "all_frame_quality",
            "status": "not_final_full_sequence_quality",
            "next_required_work": "evaluate all frames or declared sampled protocol with sequence-level reporting",
        },
        {
            "item": "decoder_contract",
            "status": "schedule_and_payload_transmitted_no_rgb_motion_features_at_decoder",
            "next_required_work": "keep in final package and count any additional side-info",
        },
    ]


def write_report(keyframe_rows, residual_rows, total_rows, requirements, package, path):
    adaptive = next(row for row in total_rows if row["schedule"] == "stage165_adaptive")
    lines = [
        "# Stage181 Full-Sequence RD Accounting Preflight",
        "",
        "## Scope",
        "",
        "This stage separates exact full-sequence keyframe/metadata counting from sampled-estimated residual payload accounting.",
        "It is a preflight, not final full-sequence RD.",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Adaptive total proxy delta vs gap8: `{adaptive['delta_total_proxy_vs_gap8']}` MiB/frame.",
        f"- Adaptive total proxy delta vs gap4: `{adaptive['delta_total_proxy_vs_gap4']}` MiB/frame.",
        "",
        "## Full-Sequence Keyframe And Metadata Counts",
        "",
        "| schedule | frames | keyframes | keyframe ratio | main anchor MiB/frame proxy | metadata bytes | metadata MiB/frame |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in keyframe_rows:
        lines.append(
            f"| {row['schedule']} | {row['total_frames']} | {row['keyframe_count']} | {float(row['keyframe_ratio']):.6f} | "
            f"{float(row['main_anchor_mib_per_frame_proxy']):.12f} | {row['metadata_bytes']} | {float(row['metadata_mib_per_frame']):.12f} |"
        )
    lines.extend([
        "",
        "## Stage180 Broader Residual Proxy",
        "",
        "| schedule | targets | keyframes | middle recovery | mean payload bytes/target | residual MiB/target | PSNR | LPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in residual_rows:
        lines.append(
            f"| {row['schedule']} | {row['target_count']} | {row['keyframe_count']} | {row['middle_recovery_count']} | "
            f"{float(row['mean_payload_bytes_per_target']):.3f} | {float(row['sampled_residual_mib_per_target']):.12f} | "
            f"{float(row['mean_psnr']):.6f} | {float(row['mean_lpips']):.6f} |"
        )
    lines.extend([
        "",
        "## Combined Proxy",
        "",
        "| schedule | main anchor | metadata | residual proxy | total proxy | delta vs gap8 | delta vs gap4 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in total_rows:
        lines.append(
            f"| {row['schedule']} | {float(row['main_anchor_mib_per_frame_proxy']):.12f} | {float(row['metadata_mib_per_frame_exact']):.12f} | "
            f"{float(row['stage180_residual_mib_per_target_proxy']):.12f} | {float(row['stage180_broader_total_proxy_mib_per_frame']):.12f} | "
            f"{float(row['delta_total_proxy_vs_gap8']):.12f} | {float(row['delta_total_proxy_vs_gap4']):.12f} |"
        )
    lines.extend([
        "",
        "## Requirements Before Final RD",
        "",
        "| item | status | next required work |",
        "|---|---|---|",
    ])
    for row in requirements:
        lines.append(f"| {row['item']} | {row['status']} | {row['next_required_work']} |")
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Keyframe/metadata CSV: `{package['keyframe_csv']}`",
        f"- Residual proxy CSV: `{package['residual_proxy_csv']}`",
        f"- Total proxy CSV: `{package['total_proxy_csv']}`",
        f"- Requirements CSV: `{package['requirements_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage165_schedule_rows", type=Path, default=DEFAULT_STAGE165_SCHEDULE_ROWS)
    parser.add_argument("--stage172_package", type=Path, default=DEFAULT_STAGE172_PACKAGE)
    parser.add_argument("--stage180_final_summary", type=Path, default=DEFAULT_STAGE180_FINAL_SUMMARY)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage165_rows = read_csv(args.stage165_schedule_rows)
    stage172 = read_json(args.stage172_package)
    stage180_rows = read_csv(args.stage180_final_summary)
    keyframe_rows = build_keyframe_rows(stage165_rows, stage172)
    residual_rows = build_residual_rows(stage180_rows)
    total_rows = build_total_rows(keyframe_rows, residual_rows)
    requirements = build_requirements()
    adaptive = next(row for row in total_rows if row["schedule"] == "stage165_adaptive")
    decision = "adaptive_rate_promising_under_broader_sampled_proxy" if adaptive["delta_total_proxy_vs_gap8"] < 0 and adaptive["delta_total_proxy_vs_gap4"] < 0 else "needs_full_sequence_payload_measurement_before_decision"
    keyframe_csv = args.output_root / "stage181_full_sequence_keyframe_metadata_accounting.csv"
    residual_proxy_csv = args.output_root / "stage181_stage180_residual_payload_proxy.csv"
    total_proxy_csv = args.output_root / "stage181_total_rate_proxy_comparison.csv"
    requirements_csv = args.output_root / "stage181_final_rd_requirements.csv"
    package_json = args.output_root / "stage181_full_sequence_rd_accounting_preflight_package.json"
    report_md = args.output_root / "stage181_full_sequence_rd_accounting_preflight_report.md"
    write_csv(keyframe_rows, keyframe_csv, KEYFRAME_FIELDS)
    write_csv(residual_rows, residual_proxy_csv, RESIDUAL_FIELDS)
    write_csv(total_rows, total_proxy_csv, TOTAL_FIELDS)
    write_csv(requirements, requirements_csv, REQUIREMENT_FIELDS)
    package = {
        "stage": 181,
        "status": "full_sequence_rd_accounting_preflight_packaged",
        "decision": decision,
        "accounting_scope": "full_sequence_keyframe_metadata_plus_stage180_broader_sampled_residual_proxy",
        "not_final_full_sequence_rd": True,
        "total_frames": keyframe_rows[0]["total_frames"],
        "sequence_count": keyframe_rows[0]["sequence_count"],
        "adaptive_total_proxy_mib_per_frame": adaptive["stage180_broader_total_proxy_mib_per_frame"],
        "adaptive_delta_total_proxy_vs_gap8": adaptive["delta_total_proxy_vs_gap8"],
        "adaptive_delta_total_proxy_vs_gap4": adaptive["delta_total_proxy_vs_gap4"],
        "keyframe_rows": keyframe_rows,
        "residual_proxy_rows": residual_rows,
        "total_proxy_rows": total_rows,
        "requirements": requirements,
        "keyframe_csv": str(keyframe_csv),
        "residual_proxy_csv": str(residual_proxy_csv),
        "total_proxy_csv": str(total_proxy_csv),
        "requirements_csv": str(requirements_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(keyframe_rows, residual_rows, total_rows, requirements, package, report_md)
    print(json.dumps({
        "package": str(package_json),
        "decision": decision,
        "adaptive_total_proxy_mib_per_frame": adaptive["stage180_broader_total_proxy_mib_per_frame"],
        "adaptive_delta_total_proxy_vs_gap8": adaptive["delta_total_proxy_vs_gap8"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
