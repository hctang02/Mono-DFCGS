import argparse
import csv
import json
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE165_SCHEDULE_ROWS = REPO_ROOT / "experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_adaptive_schedule_rows.csv"
DEFAULT_STAGE182_PACKAGE = REPO_ROOT / "experiments/stage182_selector_refinement_or_freeze_decision/stage182_selector_refinement_or_freeze_decision_package.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage183_full_sequence_payload_measurement_protocol"

SCHEDULES = ["uniform_gap8", "stage165_adaptive", "uniform_gap4"]
SCHEDULE_KEYFRAME_COL = {
    "uniform_gap8": "uniform_gap8_keyframes",
    "stage165_adaptive": "adaptive_keyframes",
    "uniform_gap4": "uniform_gap4_keyframes",
}
SCHEDULE_COUNT_COL = {
    "uniform_gap8": "uniform_gap8_keyframe_count",
    "stage165_adaptive": "adaptive_keyframe_count",
    "uniform_gap4": "uniform_gap4_keyframe_count",
}

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
    "sequence_count",
    "total_frames",
    "keyframe_rows",
    "residual_rows",
    "keyframe_ratio",
    "unique_keyframe_measurements",
    "unique_residual_measurements",
]


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


def parse_keyframes(value):
    return [int(item) for item in str(value).split() if item != ""]


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


def build_protocol_rows(stage165_rows):
    frame_rows = []
    keyframe_uses = defaultdict(set)
    residual_uses = defaultdict(set)
    residual_meta = {}
    keyframe_meta = {}
    expected_counts = defaultdict(lambda: {"keyframe_rows": 0, "residual_rows": 0, "total_frames": 0})
    for seq_row in stage165_rows:
        sequence = seq_row["sequence"]
        total_frames = int(seq_row["total_frames"])
        for schedule in SCHEDULES:
            keyframes = parse_keyframes(seq_row[SCHEDULE_KEYFRAME_COL[schedule]])
            keyframe_set = set(keyframes)
            if len(keyframes) != int(seq_row[SCHEDULE_COUNT_COL[schedule]]):
                raise ValueError(f"keyframe count mismatch for {sequence} {schedule}")
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
        counts = expected_counts[schedule]
        schedule_keyframes = {row["measurement_key"] for row in frame_rows if row["schedule"] == schedule and row["measurement_type"] == "q12_keyframe_payload"}
        schedule_residuals = {row["measurement_key"] for row in frame_rows if row["schedule"] == schedule and row["measurement_type"] == "stage158_residual_payload"}
        summary_rows.append({
            "schedule": schedule,
            "sequence_count": len(stage165_rows),
            "total_frames": counts["total_frames"],
            "keyframe_rows": counts["keyframe_rows"],
            "residual_rows": counts["residual_rows"],
            "keyframe_ratio": counts["keyframe_rows"] / float(counts["total_frames"]),
            "unique_keyframe_measurements": len(schedule_keyframes),
            "unique_residual_measurements": len(schedule_residuals),
        })
    return frame_rows, keyframe_rows, residual_rows, summary_rows


def write_report(summary_rows, package, path):
    lines = [
        "# Stage183 Full-Sequence Payload Measurement Protocol",
        "",
        "## Scope",
        "",
        "This is a protocol-only stage for full-sequence payload measurement. It does not run bitstream measurement or rendering.",
        "",
        "## Summary",
        "",
        f"- Frame/schedule rows: `{package['frame_schedule_row_count']}`.",
        f"- Unique q12 keyframe payload measurements: `{package['unique_keyframe_measurement_count']}`.",
        f"- Unique Stage158 residual payload measurements: `{package['unique_residual_measurement_count']}`.",
        f"- Frozen selector policy: `{package['frozen_policy_name']}`.",
        "",
        "## Per-Schedule Counts",
        "",
        "| schedule | frames | keyframes | residual rows | keyframe ratio | unique keyframes | unique residuals |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['schedule']} | {row['total_frames']} | {row['keyframe_rows']} | {row['residual_rows']} | "
            f"{float(row['keyframe_ratio']):.6f} | {row['unique_keyframe_measurements']} | {row['unique_residual_measurements']} |"
        )
    lines.extend([
        "",
        "## Execution Contract For Next Stage",
        "",
        "- Measure actual q12 keyframe anchor payload for rows in the unique keyframe table.",
        "- Measure Stage158 q6/keep1.0 entropy residual payload plus counted selector byte for rows in the unique residual table.",
        "- Reuse identical measurement keys across schedules where listed by `used_by_schedules`.",
        "- Do not use target dense anchors, target RGB, rendered metrics, or oracle labels as decoder-side inputs.",
        "- Keep any heavy intermediate bitstreams outside git unless they are small CSV/JSON summaries.",
        "",
        "## Outputs",
        "",
        f"- Frame/schedule protocol CSV: `{package['frame_schedule_csv']}`",
        f"- Unique keyframe measurement CSV: `{package['unique_keyframe_csv']}`",
        f"- Unique residual measurement CSV: `{package['unique_residual_csv']}`",
        f"- Summary CSV: `{package['summary_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage165_schedule_rows", type=Path, default=DEFAULT_STAGE165_SCHEDULE_ROWS)
    parser.add_argument("--stage182_package", type=Path, default=DEFAULT_STAGE182_PACKAGE)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage165_rows = read_csv(args.stage165_schedule_rows)
    stage182 = read_json(args.stage182_package)
    frame_rows, keyframe_rows, residual_rows, summary_rows = build_protocol_rows(stage165_rows)
    frame_schedule_csv = args.output_root / "stage183_full_sequence_frame_schedule_payload_rows.csv"
    unique_keyframe_csv = args.output_root / "stage183_unique_keyframe_payload_measurement_rows.csv"
    unique_residual_csv = args.output_root / "stage183_unique_stage158_residual_payload_measurement_rows.csv"
    summary_csv = args.output_root / "stage183_full_sequence_payload_measurement_summary.csv"
    package_json = args.output_root / "stage183_full_sequence_payload_measurement_protocol_package.json"
    report_md = args.output_root / "stage183_full_sequence_payload_measurement_protocol_report.md"
    write_csv(frame_rows, frame_schedule_csv, FRAME_FIELDS)
    write_csv(keyframe_rows, unique_keyframe_csv, KEYFRAME_FIELDS)
    write_csv(residual_rows, unique_residual_csv, RESIDUAL_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    package = {
        "stage": 183,
        "status": "full_sequence_payload_measurement_protocol_packaged",
        "stage182_decision": stage182["decision"],
        "frozen_policy_name": stage182["frozen_policy_name"],
        "sequence_count": len(stage165_rows),
        "total_frames_per_schedule": sum(int(row["total_frames"]) for row in stage165_rows),
        "schedule_count": len(SCHEDULES),
        "frame_schedule_row_count": len(frame_rows),
        "unique_keyframe_measurement_count": len(keyframe_rows),
        "unique_residual_measurement_count": len(residual_rows),
        "summary_rows": summary_rows,
        "frame_schedule_csv": str(frame_schedule_csv),
        "unique_keyframe_csv": str(unique_keyframe_csv),
        "unique_residual_csv": str(unique_residual_csv),
        "summary_csv": str(summary_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary_rows, package, report_md)
    print(json.dumps({
        "package": str(package_json),
        "frame_schedule_row_count": len(frame_rows),
        "unique_keyframe_measurement_count": len(keyframe_rows),
        "unique_residual_measurement_count": len(residual_rows),
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
