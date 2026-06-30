import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE183_ROOT = REPO_ROOT / "experiments/stage183_full_sequence_payload_measurement_protocol"
DEFAULT_STAGE184_ROOT = REPO_ROOT / "experiments/stage184_full_sequence_payload_measurement_execution"
DEFAULT_STAGE181_ROOT = REPO_ROOT / "experiments/stage181_full_sequence_rd_accounting_preflight"
DEFAULT_STAGE180_ROOT = REPO_ROOT / "experiments/stage180_broader_sampled_adaptive_validation_execution"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage185_measured_full_sequence_rd_aggregation"

DEFAULT_FRAME_SCHEDULE_ROWS = DEFAULT_STAGE183_ROOT / "stage183_full_sequence_frame_schedule_payload_rows.csv"
DEFAULT_RESIDUAL_MEASUREMENTS = DEFAULT_STAGE184_ROOT / "stage184_unique_stage158_residual_payload_measurements.csv"
DEFAULT_SCHEDULE_KEYFRAME_MEASUREMENTS = DEFAULT_STAGE184_ROOT / "stage184_schedule_packed_keyframe_payload_measurements.csv"
DEFAULT_STAGE181_KEYFRAME_METADATA = DEFAULT_STAGE181_ROOT / "stage181_full_sequence_keyframe_metadata_accounting.csv"
DEFAULT_STAGE181_TOTAL_PROXY = DEFAULT_STAGE181_ROOT / "stage181_total_rate_proxy_comparison.csv"
DEFAULT_STAGE180_FINAL_SUMMARY = DEFAULT_STAGE180_ROOT / "stage180_final_quality_summary.csv"

SCHEDULES = ["uniform_gap8", "stage165_adaptive", "uniform_gap4"]
MIB = 1024.0 * 1024.0

TOTAL_FIELDS = [
    "schedule",
    "sequence_count",
    "total_frames",
    "keyframe_count",
    "residual_count",
    "keyframe_bitstream_bytes",
    "keyframe_mib_per_frame",
    "residual_payload_bytes",
    "residual_mib_per_frame",
    "metadata_bytes",
    "metadata_mib_per_frame",
    "total_payload_bytes",
    "total_mib_per_frame",
    "stage181_proxy_mib_per_frame",
    "measured_minus_stage181_proxy_mib_per_frame",
    "delta_total_mib_per_frame_vs_gap8",
    "delta_total_mib_per_frame_vs_gap4",
    "stage180_mean_psnr",
    "stage180_mean_lpips",
    "rate_scope",
]

SEQUENCE_FIELDS = [
    "schedule",
    "sequence",
    "total_frames",
    "keyframe_count",
    "residual_count",
    "keyframe_bitstream_bytes",
    "keyframe_mib_per_frame",
    "residual_payload_bytes",
    "residual_mib_per_frame",
    "total_payload_bytes_without_global_metadata",
    "total_mib_per_frame_without_global_metadata",
]

COMPONENT_FIELDS = ["schedule", "component", "bytes", "mib", "mib_per_frame", "fraction_of_total"]

VALIDATION_FIELDS = ["item", "expected", "actual", "status"]


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def numeric(row, key, default=0.0):
    value = row.get(key)
    if value in (None, "", "NA"):
        return default
    return float(value)


def read_stage180_quality(path):
    out = {}
    if not path.exists():
        return out
    for row in read_csv(path):
        if row["group"] == "overall":
            out[row["schedule"]] = row
    return out


def sum_metadata_rows(path):
    out = {}
    for row in read_csv(path):
        out[row["schedule"]] = row
    return out


def stage181_proxy_rows(path):
    out = {}
    for row in read_csv(path):
        out[row["schedule"]] = row
    return out


def aggregate(frame_rows, residual_rows, schedule_keyframe_rows, metadata_rows, proxy_rows, quality_rows):
    residual_by_key = {row["measurement_key"]: row for row in residual_rows if row.get("status") == "ok"}
    keyframe_by_schedule_sequence = {
        (row["schedule"], row["sequence"]): row
        for row in schedule_keyframe_rows
        if row.get("status") == "ok"
    }
    rows_by_schedule = defaultdict(list)
    rows_by_schedule_sequence = defaultdict(list)
    for row in frame_rows:
        rows_by_schedule[row["schedule"]].append(row)
        rows_by_schedule_sequence[(row["schedule"], row["sequence"])].append(row)

    total_rows = []
    sequence_rows = []
    component_rows = []
    validation_rows = []
    for schedule in SCHEDULES:
        group = rows_by_schedule[schedule]
        sequences = sorted({row["sequence"] for row in group})
        total_frames = len(group)
        keyframe_count = sum(1 for row in group if row["measurement_type"] == "q12_keyframe_payload")
        residual_count = sum(1 for row in group if row["measurement_type"] == "stage158_residual_payload")
        residual_payload_bytes = 0.0
        missing_residuals = []
        for row in group:
            if row["measurement_type"] != "stage158_residual_payload":
                continue
            measured = residual_by_key.get(row["measurement_key"])
            if measured is None:
                missing_residuals.append(row["measurement_key"])
            else:
                residual_payload_bytes += numeric(measured, "payload_bytes")
        keyframe_bitstream_bytes = 0.0
        missing_keyframes = []
        for sequence in sequences:
            measured = keyframe_by_schedule_sequence.get((schedule, sequence))
            if measured is None:
                missing_keyframes.append(sequence)
            else:
                keyframe_bitstream_bytes += numeric(measured, "bitstream_bytes")
        metadata_bytes = numeric(metadata_rows[schedule], "metadata_bytes")
        total_payload_bytes = keyframe_bitstream_bytes + residual_payload_bytes + metadata_bytes
        total_mib_per_frame = total_payload_bytes / MIB / float(total_frames)
        proxy = proxy_rows.get(schedule, {})
        quality = quality_rows.get(schedule, {})
        total_rows.append({
            "schedule": schedule,
            "sequence_count": len(sequences),
            "total_frames": total_frames,
            "keyframe_count": keyframe_count,
            "residual_count": residual_count,
            "keyframe_bitstream_bytes": keyframe_bitstream_bytes,
            "keyframe_mib_per_frame": keyframe_bitstream_bytes / MIB / float(total_frames),
            "residual_payload_bytes": residual_payload_bytes,
            "residual_mib_per_frame": residual_payload_bytes / MIB / float(total_frames),
            "metadata_bytes": metadata_bytes,
            "metadata_mib_per_frame": metadata_bytes / MIB / float(total_frames),
            "total_payload_bytes": total_payload_bytes,
            "total_mib_per_frame": total_mib_per_frame,
            "stage181_proxy_mib_per_frame": numeric(proxy, "stage180_broader_total_proxy_mib_per_frame"),
            "measured_minus_stage181_proxy_mib_per_frame": total_mib_per_frame - numeric(proxy, "stage180_broader_total_proxy_mib_per_frame"),
            "delta_total_mib_per_frame_vs_gap8": None,
            "delta_total_mib_per_frame_vs_gap4": None,
            "stage180_mean_psnr": numeric(quality, "mean_psnr"),
            "stage180_mean_lpips": numeric(quality, "mean_lpips"),
            "rate_scope": "measured_schedule_packed_q12_keyframes_plus_measured_stage158_residual_payloads_plus_exact_metadata",
        })
        validation_rows.append({
            "item": f"{schedule}_missing_residual_measurements",
            "expected": 0,
            "actual": len(missing_residuals),
            "status": "ok" if not missing_residuals else "error",
        })
        validation_rows.append({
            "item": f"{schedule}_missing_schedule_keyframe_measurements",
            "expected": 0,
            "actual": len(missing_keyframes),
            "status": "ok" if not missing_keyframes else "error",
        })

    totals = {row["schedule"]: row["total_mib_per_frame"] for row in total_rows}
    for row in total_rows:
        row["delta_total_mib_per_frame_vs_gap8"] = row["total_mib_per_frame"] - totals["uniform_gap8"]
        row["delta_total_mib_per_frame_vs_gap4"] = row["total_mib_per_frame"] - totals["uniform_gap4"]
        total_bytes = float(row["total_payload_bytes"])
        for component, byte_key in [
            ("q12_schedule_packed_keyframes", "keyframe_bitstream_bytes"),
            ("stage158_residual_payloads", "residual_payload_bytes"),
            ("schedule_metadata", "metadata_bytes"),
        ]:
            bytes_value = float(row[byte_key])
            component_rows.append({
                "schedule": row["schedule"],
                "component": component,
                "bytes": bytes_value,
                "mib": bytes_value / MIB,
                "mib_per_frame": bytes_value / MIB / float(row["total_frames"]),
                "fraction_of_total": bytes_value / total_bytes if total_bytes else 0.0,
            })

    for (schedule, sequence), group in sorted(rows_by_schedule_sequence.items()):
        total_frames = len(group)
        keyframe_count = sum(1 for row in group if row["measurement_type"] == "q12_keyframe_payload")
        residual_group = [row for row in group if row["measurement_type"] == "stage158_residual_payload"]
        residual_payload_bytes = sum(numeric(residual_by_key[row["measurement_key"]], "payload_bytes") for row in residual_group)
        keyframe_bytes = numeric(keyframe_by_schedule_sequence[(schedule, sequence)], "bitstream_bytes")
        total_payload = keyframe_bytes + residual_payload_bytes
        sequence_rows.append({
            "schedule": schedule,
            "sequence": sequence,
            "total_frames": total_frames,
            "keyframe_count": keyframe_count,
            "residual_count": len(residual_group),
            "keyframe_bitstream_bytes": keyframe_bytes,
            "keyframe_mib_per_frame": keyframe_bytes / MIB / float(total_frames),
            "residual_payload_bytes": residual_payload_bytes,
            "residual_mib_per_frame": residual_payload_bytes / MIB / float(total_frames),
            "total_payload_bytes_without_global_metadata": total_payload,
            "total_mib_per_frame_without_global_metadata": total_payload / MIB / float(total_frames),
        })

    validation_rows.append({
        "item": "frame_schedule_rows_covered",
        "expected": len(frame_rows),
        "actual": sum(int(row["total_frames"]) for row in total_rows),
        "status": "ok" if len(frame_rows) == sum(int(row["total_frames"]) for row in total_rows) else "error",
    })
    return total_rows, sequence_rows, component_rows, validation_rows


def decision(total_rows):
    rows = {row["schedule"]: row for row in total_rows}
    adaptive = rows["stage165_adaptive"]
    if adaptive["delta_total_mib_per_frame_vs_gap8"] < 0.0 and adaptive["delta_total_mib_per_frame_vs_gap4"] < 0.0:
        return "adaptive_measured_rate_lower_than_gap8_and_gap4"
    if adaptive["delta_total_mib_per_frame_vs_gap8"] < 0.0:
        return "adaptive_measured_rate_lower_than_gap8_only"
    return "adaptive_measured_rate_not_lower_than_gap8"


def write_report(total_rows, component_rows, validation_rows, package, path):
    adaptive = next(row for row in total_rows if row["schedule"] == "stage165_adaptive")
    lines = [
        "# Stage185 Measured Full-Sequence RD Aggregation",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Adaptive measured total delta vs gap8: `{adaptive['delta_total_mib_per_frame_vs_gap8']}` MiB/frame.",
        f"- Adaptive measured total delta vs gap4: `{adaptive['delta_total_mib_per_frame_vs_gap4']}` MiB/frame.",
        "",
        "## Measured Full-Sequence Rate",
        "",
        "| schedule | frames | keyframes | residuals | keyframe MiB/frame | residual MiB/frame | metadata MiB/frame | total MiB/frame | delta vs gap8 | delta vs gap4 | Stage180 PSNR | Stage180 LPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in total_rows:
        lines.append(
            f"| {row['schedule']} | {row['total_frames']} | {row['keyframe_count']} | {row['residual_count']} | "
            f"{float(row['keyframe_mib_per_frame']):.12f} | {float(row['residual_mib_per_frame']):.12f} | {float(row['metadata_mib_per_frame']):.12f} | "
            f"{float(row['total_mib_per_frame']):.12f} | {float(row['delta_total_mib_per_frame_vs_gap8']):.12f} | {float(row['delta_total_mib_per_frame_vs_gap4']):.12f} | "
            f"{float(row['stage180_mean_psnr']):.6f} | {float(row['stage180_mean_lpips']):.6f} |"
        )
    lines.extend([
        "",
        "## Measured Versus Stage181 Proxy",
        "",
        "| schedule | measured total | Stage181 proxy | measured - proxy |",
        "|---|---:|---:|---:|",
    ])
    for row in total_rows:
        lines.append(
            f"| {row['schedule']} | {float(row['total_mib_per_frame']):.12f} | {float(row['stage181_proxy_mib_per_frame']):.12f} | {float(row['measured_minus_stage181_proxy_mib_per_frame']):.12f} |"
        )
    lines.extend([
        "",
        "## Component Fractions",
        "",
        "| schedule | component | MiB/frame | fraction |",
        "|---|---|---:|---:|",
    ])
    for row in component_rows:
        lines.append(f"| {row['schedule']} | {row['component']} | {float(row['mib_per_frame']):.12f} | {float(row['fraction_of_total']):.6f} |")
    lines.extend([
        "",
        "## Validation",
        "",
        "| item | expected | actual | status |",
        "|---|---:|---:|---|",
    ])
    for row in validation_rows:
        lines.append(f"| {row['item']} | {row['expected']} | {row['actual']} | {row['status']} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- This is the first measured full-sequence payload aggregation for the frozen adaptive schedule candidate.",
        "- Stage180 quality values remain sampled broader quality evidence; Stage186 should expand quality reporting before final paper claims.",
        "- Keyframe rate uses schedule/sequence-packed q12 bitstreams, avoiding per-keyframe container overcount.",
        "",
        "## Outputs",
        "",
        f"- Total RD CSV: `{package['total_csv']}`",
        f"- Sequence RD CSV: `{package['sequence_csv']}`",
        f"- Component CSV: `{package['component_csv']}`",
        f"- Validation CSV: `{package['validation_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_schedule_rows", type=Path, default=DEFAULT_FRAME_SCHEDULE_ROWS)
    parser.add_argument("--residual_measurements", type=Path, default=DEFAULT_RESIDUAL_MEASUREMENTS)
    parser.add_argument("--schedule_keyframe_measurements", type=Path, default=DEFAULT_SCHEDULE_KEYFRAME_MEASUREMENTS)
    parser.add_argument("--stage181_keyframe_metadata", type=Path, default=DEFAULT_STAGE181_KEYFRAME_METADATA)
    parser.add_argument("--stage181_total_proxy", type=Path, default=DEFAULT_STAGE181_TOTAL_PROXY)
    parser.add_argument("--stage180_final_summary", type=Path, default=DEFAULT_STAGE180_FINAL_SUMMARY)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    total_rows, sequence_rows, component_rows, validation_rows = aggregate(
        read_csv(args.frame_schedule_rows),
        read_csv(args.residual_measurements),
        read_csv(args.schedule_keyframe_measurements),
        sum_metadata_rows(args.stage181_keyframe_metadata),
        stage181_proxy_rows(args.stage181_total_proxy),
        read_stage180_quality(args.stage180_final_summary),
    )
    total_csv = args.output_root / "stage185_measured_full_sequence_total_rd.csv"
    sequence_csv = args.output_root / "stage185_measured_sequence_rd_breakdown.csv"
    component_csv = args.output_root / "stage185_measured_component_breakdown.csv"
    validation_csv = args.output_root / "stage185_aggregation_validation.csv"
    package_json = args.output_root / "stage185_measured_full_sequence_rd_aggregation_package.json"
    report_md = args.output_root / "stage185_measured_full_sequence_rd_aggregation_report.md"
    write_csv(total_rows, total_csv, TOTAL_FIELDS)
    write_csv(sequence_rows, sequence_csv, SEQUENCE_FIELDS)
    write_csv(component_rows, component_csv, COMPONENT_FIELDS)
    write_csv(validation_rows, validation_csv, VALIDATION_FIELDS)
    package = {
        "stage": 185,
        "status": "measured_full_sequence_rd_aggregation_packaged",
        "decision": decision(total_rows),
        "total_rows": total_rows,
        "component_rows": component_rows,
        "validation_rows": validation_rows,
        "total_csv": str(total_csv),
        "sequence_csv": str(sequence_csv),
        "component_csv": str(component_csv),
        "validation_csv": str(validation_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(total_rows, component_rows, validation_rows, package, report_md)
    adaptive = next(row for row in total_rows if row["schedule"] == "stage165_adaptive")
    print(json.dumps({
        "package": str(package_json),
        "decision": package["decision"],
        "adaptive_total_mib_per_frame": adaptive["total_mib_per_frame"],
        "adaptive_delta_vs_gap8": adaptive["delta_total_mib_per_frame_vs_gap8"],
        "adaptive_delta_vs_gap4": adaptive["delta_total_mib_per_frame_vs_gap4"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
