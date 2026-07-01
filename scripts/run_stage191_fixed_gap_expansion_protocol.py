import argparse
import csv
import json
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE165_ROWS = REPO_ROOT / "experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_adaptive_schedule_rows.csv"
DEFAULT_STAGE184_ROOT = REPO_ROOT / "experiments/stage184_full_sequence_payload_measurement_execution"
DEFAULT_STAGE186_ROOT = REPO_ROOT / "experiments/stage186_full_sequence_quality_validation"
DEFAULT_STAGE181_METADATA = REPO_ROOT / "experiments/stage181_full_sequence_rd_accounting_preflight/stage181_full_sequence_keyframe_metadata_accounting.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage191_fixed_gap_expansion_protocol"

FIXED_GAPS = [2, 4, 6, 8, 16]
ADAPTIVE_SCHEDULE = "stage165_adaptive"
SCHEDULES = [f"uniform_gap{gap}" for gap in FIXED_GAPS] + [ADAPTIVE_SCHEDULE]
FIXED_METADATA_BYTES = 1

FRAME_FIELDS = [
    "sequence",
    "total_frames",
    "schedule",
    "frame_index",
    "measurement_type",
    "measurement_key",
    "left_index",
    "right_index",
    "segment_length",
    "normalized_time",
    "is_keyframe",
    "notes",
]

KEYFRAME_FIELDS = ["measurement_key", "sequence", "frame_index", "used_by_schedules", "schedule_count"]
RESIDUAL_FIELDS = [
    "measurement_key",
    "sequence",
    "target_index",
    "left_index",
    "right_index",
    "segment_length",
    "normalized_time",
    "used_by_schedules",
    "schedule_count",
]
SUMMARY_FIELDS = [
    "schedule",
    "schedule_family",
    "gap",
    "sequence_count",
    "total_frames",
    "keyframe_rows",
    "residual_rows",
    "keyframe_ratio",
    "unique_keyframe_measurements",
    "unique_residual_measurements",
    "metadata_bytes",
]
COVERAGE_FIELDS = [
    "measurement_scope",
    "expected_count",
    "existing_ok_count",
    "missing_count",
    "reuse_fraction",
    "existing_source",
]
MISSING_FIELDS = ["measurement_scope", "measurement_key", "schedule", "sequence", "frame_index", "left_index", "right_index", "segment_length"]


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_keyframes(value):
    return [int(item) for item in str(value).split() if item != ""]


def uniform_keyframes(total_frames, gap):
    if total_frames <= 0:
        return []
    frames = list(range(0, total_frames, gap))
    last = total_frames - 1
    if frames[-1] != last:
        frames.append(last)
    return frames


def adjacent_keyframes(keyframes, frame_index):
    pos = bisect_left(keyframes, frame_index)
    if pos < len(keyframes) and keyframes[pos] == frame_index:
        return frame_index, frame_index
    if pos == 0 or pos == len(keyframes):
        raise ValueError(f"frame {frame_index} is outside schedule keyframe coverage")
    return keyframes[pos - 1], keyframes[pos]


def keyframe_measurement_key(sequence, frame_index):
    return f"keyframe::{sequence}:{int(frame_index):05d}"


def residual_measurement_key(sequence, target_index, left_index, right_index):
    return f"residual::{sequence}:{int(target_index):05d}:{int(left_index):05d}:{int(right_index):05d}"


def schedule_keyframes_for_row(row, schedule):
    if schedule == ADAPTIVE_SCHEDULE:
        return parse_keyframes(row["adaptive_keyframes"])
    gap = int(schedule.removeprefix("uniform_gap"))
    return uniform_keyframes(int(row["total_frames"]), gap)


def schedule_metadata_bytes(stage165_rows, metadata_by_schedule, schedule):
    if schedule in metadata_by_schedule:
        return int(float(metadata_by_schedule[schedule]["metadata_bytes"]))
    if schedule == ADAPTIVE_SCHEDULE:
        return sum(int(row["metadata_bytes"]) for row in stage165_rows)
    return FIXED_METADATA_BYTES


def build_protocol(stage165_rows, metadata_by_schedule):
    frame_rows = []
    keyframe_uses = defaultdict(set)
    residual_uses = defaultdict(set)
    keyframe_meta = {}
    residual_meta = {}
    expected_counts = defaultdict(lambda: {"keyframe_rows": 0, "residual_rows": 0, "total_frames": 0})

    for seq_row in stage165_rows:
        sequence = seq_row["sequence"]
        total_frames = int(seq_row["total_frames"])
        for schedule in SCHEDULES:
            keyframes = schedule_keyframes_for_row(seq_row, schedule)
            if keyframes[0] != 0 or keyframes[-1] != total_frames - 1:
                raise ValueError(f"schedule coverage mismatch for {sequence} {schedule}")
            keyframe_set = set(keyframes)
            expected_counts[schedule]["total_frames"] += total_frames
            for frame_index in range(total_frames):
                if frame_index in keyframe_set:
                    key = keyframe_measurement_key(sequence, frame_index)
                    keyframe_uses[key].add(schedule)
                    keyframe_meta[key] = {"measurement_key": key, "sequence": sequence, "frame_index": frame_index}
                    row = {
                        "sequence": sequence,
                        "total_frames": total_frames,
                        "schedule": schedule,
                        "frame_index": frame_index,
                        "measurement_type": "q12_keyframe_payload",
                        "measurement_key": key,
                        "left_index": frame_index,
                        "right_index": frame_index,
                        "segment_length": 0,
                        "normalized_time": 0.0,
                        "is_keyframe": 1,
                        "notes": "measure transmitted q12 keyframe anchor payload",
                    }
                    expected_counts[schedule]["keyframe_rows"] += 1
                else:
                    left, right = adjacent_keyframes(keyframes, frame_index)
                    segment_length = right - left
                    normalized_time = (frame_index - left) / float(segment_length)
                    key = residual_measurement_key(sequence, frame_index, left, right)
                    residual_uses[key].add(schedule)
                    residual_meta[key] = {
                        "measurement_key": key,
                        "sequence": sequence,
                        "target_index": frame_index,
                        "left_index": left,
                        "right_index": right,
                        "segment_length": segment_length,
                        "normalized_time": normalized_time,
                    }
                    row = {
                        "sequence": sequence,
                        "total_frames": total_frames,
                        "schedule": schedule,
                        "frame_index": frame_index,
                        "measurement_type": "stage158_residual_payload",
                        "measurement_key": key,
                        "left_index": left,
                        "right_index": right,
                        "segment_length": segment_length,
                        "normalized_time": normalized_time,
                        "is_keyframe": 0,
                        "notes": "measure Stage158 entropy residual payload for recovered non-keyframe",
                    }
                    expected_counts[schedule]["residual_rows"] += 1
                frame_rows.append(row)

    keyframe_rows = []
    for key, schedules in sorted(keyframe_uses.items()):
        row = dict(keyframe_meta[key])
        row["used_by_schedules"] = " ".join(sorted(schedules))
        row["schedule_count"] = len(schedules)
        keyframe_rows.append(row)

    residual_rows = []
    for key, schedules in sorted(residual_uses.items()):
        row = dict(residual_meta[key])
        row["used_by_schedules"] = " ".join(sorted(schedules))
        row["schedule_count"] = len(schedules)
        residual_rows.append(row)

    summary_rows = []
    for schedule in SCHEDULES:
        group = [row for row in frame_rows if row["schedule"] == schedule]
        schedule_keyframes = {row["measurement_key"] for row in group if row["measurement_type"] == "q12_keyframe_payload"}
        schedule_residuals = {row["measurement_key"] for row in group if row["measurement_type"] == "stage158_residual_payload"}
        counts = expected_counts[schedule]
        gap = "" if schedule == ADAPTIVE_SCHEDULE else int(schedule.removeprefix("uniform_gap"))
        summary_rows.append(
            {
                "schedule": schedule,
                "schedule_family": "adaptive" if schedule == ADAPTIVE_SCHEDULE else "fixed_gap",
                "gap": gap,
                "sequence_count": len(stage165_rows),
                "total_frames": counts["total_frames"],
                "keyframe_rows": counts["keyframe_rows"],
                "residual_rows": counts["residual_rows"],
                "keyframe_ratio": counts["keyframe_rows"] / float(counts["total_frames"]),
                "unique_keyframe_measurements": len(schedule_keyframes),
                "unique_residual_measurements": len(schedule_residuals),
                "metadata_bytes": schedule_metadata_bytes(stage165_rows, metadata_by_schedule, schedule),
            }
        )
    return frame_rows, keyframe_rows, residual_rows, summary_rows


def ok_keys(rows, key="measurement_key"):
    return {row[key] for row in rows if row.get("status") == "ok"}


def ok_schedule_groups(rows):
    return {(row["schedule"], row["sequence"]) for row in rows if row.get("status") == "ok"}


def schedule_groups(frame_rows):
    groups = set()
    for row in frame_rows:
        if row["measurement_type"] == "q12_keyframe_payload":
            groups.add((row["schedule"], row["sequence"]))
    return groups


def build_coverage_and_missing(frame_rows, keyframe_rows, residual_rows, args):
    stage184_keyframes = ok_keys(read_csv(args.stage184_root / "stage184_unique_keyframe_payload_measurements.csv"))
    stage184_residuals = ok_keys(read_csv(args.stage184_root / "stage184_unique_stage158_residual_payload_measurements.csv"))
    stage184_schedule_groups = ok_schedule_groups(read_csv(args.stage184_root / "stage184_schedule_packed_keyframe_payload_measurements.csv"))
    stage186_keyframes = ok_keys(read_csv(args.stage186_root / "stage186_unique_keyframe_quality_metrics.csv"))
    stage186_residuals = ok_keys(read_csv(args.stage186_root / "stage186_unique_stage158_residual_quality_metrics.csv"))

    expected_keyframes = {row["measurement_key"] for row in keyframe_rows}
    expected_residuals = {row["measurement_key"] for row in residual_rows}
    expected_schedule_groups = schedule_groups(frame_rows)

    coverage_specs = [
        ("payload_single_keyframe", expected_keyframes, stage184_keyframes, "stage184_unique_keyframe_payload_measurements.csv"),
        ("payload_residual", expected_residuals, stage184_residuals, "stage184_unique_stage158_residual_payload_measurements.csv"),
        ("payload_schedule_packed_keyframe_group", expected_schedule_groups, stage184_schedule_groups, "stage184_schedule_packed_keyframe_payload_measurements.csv"),
        ("quality_single_keyframe", expected_keyframes, stage186_keyframes, "stage186_unique_keyframe_quality_metrics.csv"),
        ("quality_residual", expected_residuals, stage186_residuals, "stage186_unique_stage158_residual_quality_metrics.csv"),
    ]
    coverage_rows = []
    for scope, expected, existing, source in coverage_specs:
        reuse = expected & existing
        coverage_rows.append(
            {
                "measurement_scope": scope,
                "expected_count": len(expected),
                "existing_ok_count": len(reuse),
                "missing_count": len(expected - existing),
                "reuse_fraction": len(reuse) / float(len(expected)) if expected else 1.0,
                "existing_source": source,
            }
        )

    first_frame_by_key = {row["measurement_key"]: row for row in frame_rows}
    missing_rows = []
    for scope, expected, existing, _source in coverage_specs:
        if scope == "payload_schedule_packed_keyframe_group":
            for schedule, sequence in sorted(expected - existing):
                missing_rows.append(
                    {
                        "measurement_scope": scope,
                        "measurement_key": f"schedule_keyframes::{schedule}:{sequence}",
                        "schedule": schedule,
                        "sequence": sequence,
                        "frame_index": "",
                        "left_index": "",
                        "right_index": "",
                        "segment_length": "",
                    }
                )
            continue
        for key in sorted(expected - existing):
            row = first_frame_by_key.get(key, {})
            missing_rows.append(
                {
                    "measurement_scope": scope,
                    "measurement_key": key,
                    "schedule": row.get("schedule", ""),
                    "sequence": row.get("sequence", ""),
                    "frame_index": row.get("frame_index", ""),
                    "left_index": row.get("left_index", ""),
                    "right_index": row.get("right_index", ""),
                    "segment_length": row.get("segment_length", ""),
                }
            )
    return coverage_rows, missing_rows


def write_report(summary_rows, coverage_rows, package, path):
    lines = [
        "# Stage191 Fixed-Gap Expansion Protocol",
        "",
        "## Scope",
        "",
        "This stage builds a protocol for expanded fixed-gap full-sequence measurement. It does not measure payloads or quality.",
        "",
        "## Schedules",
        "",
        "| schedule | frames | keyframes | residual rows | keyframe ratio | metadata bytes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['schedule']} | {row['total_frames']} | {row['keyframe_rows']} | {row['residual_rows']} | "
            f"{float(row['keyframe_ratio']):.6f} | {row['metadata_bytes']} |"
        )
    lines.extend([
        "",
        "## Reuse Coverage For Stage192",
        "",
        "| scope | expected | existing ok | missing | reuse fraction |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in coverage_rows:
        lines.append(
            f"| {row['measurement_scope']} | {row['expected_count']} | {row['existing_ok_count']} | {row['missing_count']} | {float(row['reuse_fraction']):.6f} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        "- Stage192 should measure the missing gap2/gap6/gap16 payload and quality rows, reusing existing gap4/gap8/adaptive rows.",
        "- This expanded baseline set is required before claiming the selector beats fixed-gap schedules.",
        "",
        "## Outputs",
        "",
        f"- Frame/schedule rows: `{package['frame_schedule_csv']}`",
        f"- Unique keyframe rows: `{package['unique_keyframe_csv']}`",
        f"- Unique residual rows: `{package['unique_residual_csv']}`",
        f"- Summary CSV: `{package['summary_csv']}`",
        f"- Reuse coverage CSV: `{package['reuse_coverage_csv']}`",
        f"- Missing measurements CSV: `{package['missing_measurements_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage165_rows", type=Path, default=DEFAULT_STAGE165_ROWS)
    parser.add_argument("--stage184_root", type=Path, default=DEFAULT_STAGE184_ROOT)
    parser.add_argument("--stage186_root", type=Path, default=DEFAULT_STAGE186_ROOT)
    parser.add_argument("--stage181_metadata", type=Path, default=DEFAULT_STAGE181_METADATA)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage165_rows = read_csv(args.stage165_rows)
    metadata_by_schedule = {row["schedule"]: row for row in read_csv(args.stage181_metadata)}
    frame_rows, keyframe_rows, residual_rows, summary_rows = build_protocol(stage165_rows, metadata_by_schedule)
    coverage_rows, missing_rows = build_coverage_and_missing(frame_rows, keyframe_rows, residual_rows, args)

    frame_schedule_csv = args.output_root / "stage191_expanded_fixed_gap_frame_schedule_rows.csv"
    unique_keyframe_csv = args.output_root / "stage191_unique_keyframe_measurement_rows.csv"
    unique_residual_csv = args.output_root / "stage191_unique_stage158_residual_measurement_rows.csv"
    summary_csv = args.output_root / "stage191_expanded_fixed_gap_schedule_summary.csv"
    reuse_coverage_csv = args.output_root / "stage191_existing_measurement_reuse_coverage.csv"
    missing_csv = args.output_root / "stage191_missing_measurements_for_stage192.csv"
    package_json = args.output_root / "stage191_fixed_gap_expansion_protocol_package.json"
    report_md = args.output_root / "stage191_fixed_gap_expansion_protocol_report.md"

    write_csv(frame_rows, frame_schedule_csv, FRAME_FIELDS)
    write_csv(keyframe_rows, unique_keyframe_csv, KEYFRAME_FIELDS)
    write_csv(residual_rows, unique_residual_csv, RESIDUAL_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(coverage_rows, reuse_coverage_csv, COVERAGE_FIELDS)
    write_csv(missing_rows, missing_csv, MISSING_FIELDS)

    package = {
        "stage": 191,
        "status": "fixed_gap_expansion_protocol_packaged",
        "decision": "measure_expanded_fixed_gap_baselines_next",
        "schedules": SCHEDULES,
        "fixed_gaps": FIXED_GAPS,
        "sequence_count": len(stage165_rows),
        "frame_schedule_row_count": len(frame_rows),
        "unique_keyframe_measurement_count": len(keyframe_rows),
        "unique_residual_measurement_count": len(residual_rows),
        "summary_rows": summary_rows,
        "reuse_coverage_rows": coverage_rows,
        "frame_schedule_csv": str(frame_schedule_csv.relative_to(REPO_ROOT)),
        "unique_keyframe_csv": str(unique_keyframe_csv.relative_to(REPO_ROOT)),
        "unique_residual_csv": str(unique_residual_csv.relative_to(REPO_ROOT)),
        "summary_csv": str(summary_csv.relative_to(REPO_ROOT)),
        "reuse_coverage_csv": str(reuse_coverage_csv.relative_to(REPO_ROOT)),
        "missing_measurements_csv": str(missing_csv.relative_to(REPO_ROOT)),
        "package_json": str(package_json.relative_to(REPO_ROOT)),
        "report_md": str(report_md.relative_to(REPO_ROOT)),
        "next": "Run Stage192 to measure missing payload/quality rows and aggregate expanded fixed-gap RD-quality.",
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary_rows, coverage_rows, package, report_md)
    print(json.dumps({"package": str(package_json), "decision": package["decision"], "coverage": coverage_rows}, indent=2))


if __name__ == "__main__":
    main()
