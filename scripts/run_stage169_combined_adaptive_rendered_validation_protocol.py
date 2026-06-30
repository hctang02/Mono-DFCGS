import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE166_ROWS = REPO_ROOT / "experiments/stage166_adaptive_schedule_label_rd_comparison/stage166_sampled_row_consequences.csv"
DEFAULT_STAGE167_TARGETS = REPO_ROOT / "experiments/stage167_adaptive_schedule_rendered_smoke/stage167_rendered_smoke_targets.csv"
DEFAULT_STAGE167_ROWS = REPO_ROOT / "experiments/stage167_adaptive_schedule_rendered_smoke/stage167_rendered_smoke_rows.csv"
DEFAULT_STAGE168_TARGETS = REPO_ROOT / "experiments/stage168_positive_coverage_rendered_smoke/stage168_positive_coverage_targets.csv"
DEFAULT_STAGE168_ROWS = REPO_ROOT / "experiments/stage168_positive_coverage_rendered_smoke/stage168_positive_coverage_rows.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage169_combined_adaptive_rendered_validation_protocol"

SCHEDULES = ["uniform_gap8", "stage165_adaptive", "uniform_gap4"]
POSITIVE_SEQUENCES = ["motocross-jump", "cows", "camel", "scooter-black", "india", "shooting", "car-roundabout"]

TARGET_FIELDS = [
    "rank", "target_key", "category", "source_task_id", "sequence", "target_index", "source_gap",
    "hard_quality_label", "high_payload_label", "adaptive_target_is_keyframe", "uniform_gap4_target_is_keyframe",
    "stage166_payload_bytes", "stage166_psnr", "stage166_lpips", "priority_score", "existing_stage167_target",
    "existing_stage168_target", "selection_reason",
]
SCHEDULE_ROW_FIELDS = [
    "target_key", "category", "sequence", "target_index", "schedule", "expected_status", "existing_source",
    "requires_stage170_render", "priority_score", "notes",
]
SUMMARY_FIELDS = [
    "category", "target_count", "schedule_row_count", "existing_schedule_rows", "new_render_schedule_rows",
    "keyframe_marker_rows", "mean_payload_bytes", "hard_count", "high_payload_count",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    vals = [float(v) for v in values if v not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def target_key(sequence, target_index):
    return f"{sequence}:{int(target_index):05d}"


def target_keys_from_targets(path):
    if not path.exists():
        return set()
    return {row["target_key"] for row in read_csv(path)}


def schedule_sources(*row_paths):
    out = {}
    for stage_name, path in row_paths:
        if not path.exists():
            continue
        for row in read_csv(path):
            out[(row["target_key"], row["schedule"])] = stage_name
    return out


def base_target(row, category, priority_score, existing167, existing168, reason):
    return {
        "target_key": target_key(row["sequence"], row["target_index"]),
        "category": category,
        "source_task_id": row["task_id"],
        "sequence": row["sequence"],
        "target_index": int(row["target_index"]),
        "source_gap": int(row["gap"]),
        "hard_quality_label": int(row["hard_quality_label"]),
        "high_payload_label": int(row["high_payload_label"]),
        "adaptive_target_is_keyframe": int(row["stage165_adaptive_target_is_keyframe"]),
        "uniform_gap4_target_is_keyframe": int(row["uniform_gap4_target_is_keyframe"]),
        "stage166_payload_bytes": float(row["payload_bytes"]),
        "stage166_psnr": float(row["stage158_psnr"]),
        "stage166_lpips": float(row["stage158_lpips"]),
        "priority_score": float(priority_score),
        "existing_stage167_target": int(target_key(row["sequence"], row["target_index"]) in existing167),
        "existing_stage168_target": int(target_key(row["sequence"], row["target_index"]) in existing168),
        "selection_reason": reason,
    }


def select_false_negative(rows, existing167, existing168):
    selected = []
    for row in rows:
        if int(row["stage165_adaptive_false_negative_hard"]):
            score = 200.0 + (26.0 - float(row["stage158_psnr"])) * 10.0 + float(row["stage158_lpips"]) * 20.0 + float(row["payload_bytes"]) / 10000.0
            selected.append(base_target(row, "false_negative_residual", score, existing167, existing168, "hard row missed by adaptive selector"))
    selected.sort(key=lambda item: (float(item["priority_score"]), float(item["stage166_payload_bytes"])), reverse=True)
    return selected


def select_positive_promoted(rows, existing167, existing168, per_sequence):
    selected = []
    by_sequence = defaultdict(list)
    for row in rows:
        if not int(row["stage165_adaptive_target_is_keyframe"]):
            continue
        if not int(row["hard_quality_label"]) and not int(row["high_payload_label"]):
            continue
        seq = row["sequence"]
        seq_bonus = 50.0 if seq in POSITIVE_SEQUENCES else 0.0
        score = seq_bonus + 100.0 * int(row["hard_quality_label"]) + 60.0 * int(row["high_payload_label"]) + float(row["payload_bytes"]) / 10000.0
        by_sequence[seq].append(base_target(row, "positive_promoted", score, existing167, existing168, "adaptive promotes hard/high-payload target"))
    for seq in POSITIVE_SEQUENCES:
        group = sorted(by_sequence.get(seq, []), key=lambda item: (float(item["priority_score"]), float(item["stage166_payload_bytes"])), reverse=True)
        selected.extend(group[:per_sequence])
    selected.sort(key=lambda item: (POSITIVE_SEQUENCES.index(item["sequence"]) if item["sequence"] in POSITIVE_SEQUENCES else 99, -float(item["priority_score"])))
    return selected


def select_residual_payload_controls(rows, existing167, existing168, max_targets):
    selected = []
    for row in rows:
        if int(row["stage165_adaptive_target_is_keyframe"]):
            continue
        if int(row["hard_quality_label"]):
            continue
        if not int(row["high_payload_label"]):
            continue
        score = 100.0 + float(row["payload_bytes"]) / 1000.0 + float(row["stage158_lpips"]) * 10.0
        selected.append(base_target(row, "high_payload_residual_control", score, existing167, existing168, "high-payload target not promoted and not hard-quality"))
    selected.sort(key=lambda item: (float(item["priority_score"]), float(item["stage166_payload_bytes"])), reverse=True)
    return selected[:max_targets]


def dedupe_targets(groups):
    out = []
    seen = set()
    for group in groups:
        for row in group:
            if row["target_key"] in seen:
                continue
            seen.add(row["target_key"])
            row = dict(row)
            row["rank"] = len(out) + 1
            out.append(row)
    return out


def expected_status(target, schedule):
    if schedule == "stage165_adaptive" and int(target["adaptive_target_is_keyframe"]):
        return "target_keyframe_no_middle_render"
    if schedule == "uniform_gap4" and int(target["uniform_gap4_target_is_keyframe"]):
        return "target_keyframe_no_middle_render"
    return "rendered_middle_recovery"


def build_schedule_protocol(targets, existing_schedule_sources):
    rows = []
    for target in targets:
        for schedule in SCHEDULES:
            existing = existing_schedule_sources.get((target["target_key"], schedule), "")
            status = expected_status(target, schedule)
            requires = int(status == "rendered_middle_recovery" and existing == "")
            rows.append({
                "target_key": target["target_key"],
                "category": target["category"],
                "sequence": target["sequence"],
                "target_index": int(target["target_index"]),
                "schedule": schedule,
                "expected_status": status,
                "existing_source": existing,
                "requires_stage170_render": requires,
                "priority_score": float(target["priority_score"]),
                "notes": "reuse existing smoke row" if existing else ("metadata/keyframe marker only" if status != "rendered_middle_recovery" else "render in Stage170"),
            })
    return rows


def summarize(targets, schedule_rows):
    out = []
    by_category = defaultdict(list)
    sched_by_category = defaultdict(list)
    for row in targets:
        by_category[row["category"]].append(row)
    for row in schedule_rows:
        sched_by_category[row["category"]].append(row)
    for category, group in sorted(by_category.items()):
        sched = sched_by_category[category]
        out.append({
            "category": category,
            "target_count": len(group),
            "schedule_row_count": len(sched),
            "existing_schedule_rows": sum(1 for row in sched if row["existing_source"]),
            "new_render_schedule_rows": sum(int(row["requires_stage170_render"]) for row in sched),
            "keyframe_marker_rows": sum(1 for row in sched if row["expected_status"] == "target_keyframe_no_middle_render" and not row["existing_source"]),
            "mean_payload_bytes": mean(row["stage166_payload_bytes"] for row in group),
            "hard_count": sum(int(row["hard_quality_label"]) for row in group),
            "high_payload_count": sum(int(row["high_payload_label"]) for row in group),
        })
    return out


def write_report(targets, schedule_rows, summary_rows, package, path):
    counts = Counter(row["category"] for row in targets)
    lines = [
        "# Stage169 Combined Adaptive Rendered Validation Protocol",
        "",
        "## Scope",
        "",
        "This is a protocol-only stage. It selects targets and schedule rows for Stage170 without running any rendering.",
        "Stage170 should reuse existing Stage167/168 rows and render only missing schedule rows.",
        "",
        "## Target Counts",
        "",
    ]
    for category, count in sorted(counts.items()):
        lines.append(f"- `{category}`: `{count}` targets")
    lines.extend([
        "",
        "## Summary",
        "",
        "| category | targets | schedule rows | existing | new renders | keyframe markers | mean payload | hard | high payload |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        lines.append(
            f"| {row['category']} | {row['target_count']} | {row['schedule_row_count']} | {row['existing_schedule_rows']} | "
            f"{row['new_render_schedule_rows']} | {row['keyframe_marker_rows']} | {float(row['mean_payload_bytes']):.3f} | "
            f"{row['hard_count']} | {row['high_payload_count']} |"
        )
    lines.extend([
        "",
        "## Stage170 Contract",
        "",
        "- Render only rows with `requires_stage170_render=1`.",
        "- For adaptive promoted targets, record `target_keyframe_no_middle_render`; do not claim middle-render metrics.",
        "- Reuse Stage167/168 smoke rows when `existing_source` is present.",
        "- Count adaptive schedule metadata and residual payloads consistently in the Stage170 report.",
        "- Decoder receives transmitted schedule/keyframes only; RGB/motion selector features remain encoder-side only.",
        "",
        "## Outputs",
        "",
        f"- Targets CSV: `{package['targets_csv']}`",
        f"- Schedule row CSV: `{package['schedule_rows_csv']}`",
        f"- Summary CSV: `{package['summary_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage166_rows", type=Path, default=DEFAULT_STAGE166_ROWS)
    parser.add_argument("--stage167_targets", type=Path, default=DEFAULT_STAGE167_TARGETS)
    parser.add_argument("--stage167_rows", type=Path, default=DEFAULT_STAGE167_ROWS)
    parser.add_argument("--stage168_targets", type=Path, default=DEFAULT_STAGE168_TARGETS)
    parser.add_argument("--stage168_rows", type=Path, default=DEFAULT_STAGE168_ROWS)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--positive_per_sequence", type=int, default=2)
    parser.add_argument("--payload_control_count", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.stage166_rows)
    existing167 = target_keys_from_targets(args.stage167_targets)
    existing168 = target_keys_from_targets(args.stage168_targets)
    existing_schedule_sources = schedule_sources(("stage167", args.stage167_rows), ("stage168", args.stage168_rows))
    targets = dedupe_targets([
        select_false_negative(rows, existing167, existing168),
        select_positive_promoted(rows, existing167, existing168, args.positive_per_sequence),
        select_residual_payload_controls(rows, existing167, existing168, args.payload_control_count),
    ])
    schedule_rows = build_schedule_protocol(targets, existing_schedule_sources)
    summary_rows = summarize(targets, schedule_rows)
    targets_csv = args.summary_root / "stage169_combined_validation_targets.csv"
    schedule_rows_csv = args.summary_root / "stage169_combined_validation_schedule_rows.csv"
    summary_csv = args.summary_root / "stage169_combined_validation_summary.csv"
    package_json = args.summary_root / "stage169_combined_adaptive_rendered_validation_protocol_package.json"
    report_md = args.summary_root / "stage169_combined_adaptive_rendered_validation_protocol_report.md"
    write_csv(targets, targets_csv, TARGET_FIELDS)
    write_csv(schedule_rows, schedule_rows_csv, SCHEDULE_ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    package = {
        "stage": 169,
        "status": "combined_adaptive_rendered_validation_protocol_packaged",
        "target_count": len(targets),
        "schedule_row_count": len(schedule_rows),
        "new_render_schedule_rows": sum(int(row["requires_stage170_render"]) for row in schedule_rows),
        "existing_schedule_rows": sum(1 for row in schedule_rows if row["existing_source"]),
        "keyframe_marker_rows": sum(1 for row in schedule_rows if row["expected_status"] == "target_keyframe_no_middle_render" and not row["existing_source"]),
        "positive_sequences": POSITIVE_SEQUENCES,
        "summary_rows": summary_rows,
        "targets_csv": str(targets_csv),
        "schedule_rows_csv": str(schedule_rows_csv),
        "summary_csv": str(summary_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(targets, schedule_rows, summary_rows, package, report_md)
    print(json.dumps({"package": str(package_json), "report": str(report_md), "target_count": len(targets), "new_render_schedule_rows": package["new_render_schedule_rows"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
