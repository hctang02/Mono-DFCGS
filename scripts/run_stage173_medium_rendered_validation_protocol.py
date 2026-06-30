import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE166_ROWS = REPO_ROOT / "experiments/stage166_adaptive_schedule_label_rd_comparison/stage166_sampled_row_consequences.csv"
DEFAULT_STAGE169_TARGETS = REPO_ROOT / "experiments/stage169_combined_adaptive_rendered_validation_protocol/stage169_combined_validation_targets.csv"
DEFAULT_STAGE167_ROWS = REPO_ROOT / "experiments/stage167_adaptive_schedule_rendered_smoke/stage167_rendered_smoke_rows.csv"
DEFAULT_STAGE168_ROWS = REPO_ROOT / "experiments/stage168_positive_coverage_rendered_smoke/stage168_positive_coverage_rows.csv"
DEFAULT_STAGE170_ROWS = REPO_ROOT / "experiments/stage170_combined_adaptive_rendered_validation_execution/stage170_combined_validation_rows.csv"
DEFAULT_STAGE172_PACKAGE = REPO_ROOT / "experiments/stage172_keyframe_rate_accounting_audit/stage172_keyframe_rate_accounting_audit_package.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage173_medium_rendered_validation_protocol"

SCHEDULES = ["uniform_gap8", "stage165_adaptive", "uniform_gap4"]

TARGET_FIELDS = [
    "rank",
    "target_key",
    "category",
    "source_task_id",
    "sequence",
    "target_index",
    "source_gap",
    "hard_quality_label",
    "high_payload_label",
    "adaptive_target_is_keyframe",
    "uniform_gap8_target_is_keyframe",
    "uniform_gap4_target_is_keyframe",
    "stage166_payload_bytes",
    "stage166_psnr",
    "stage166_lpips",
    "priority_score",
    "stage170_core_target",
    "selection_reason",
]
SCHEDULE_ROW_FIELDS = [
    "target_key",
    "category",
    "sequence",
    "target_index",
    "schedule",
    "expected_status",
    "existing_source",
    "requires_stage174_render",
    "priority_score",
    "notes",
]
SUMMARY_FIELDS = [
    "category",
    "target_count",
    "schedule_row_count",
    "existing_schedule_rows",
    "new_render_schedule_rows",
    "keyframe_marker_rows",
    "mean_payload_bytes",
    "mean_psnr",
    "mean_lpips",
    "hard_count",
    "high_payload_count",
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


def mean(values):
    vals = [float(v) for v in values if v not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def target_key(sequence, target_index):
    return f"{sequence}:{int(target_index):05d}"


def stage166_map(rows):
    return {target_key(row["sequence"], row["target_index"]): row for row in rows}


def schedule_sources(*paths):
    out = {}
    for source, path in paths:
        if not path.exists():
            continue
        for row in read_csv(path):
            out[(row["target_key"], row["schedule"])] = source
    return out


def base_target(row, category, priority_score, reason, stage170_core):
    return {
        "rank": 0,
        "target_key": target_key(row["sequence"], row["target_index"]),
        "category": category,
        "source_task_id": row["task_id"],
        "sequence": row["sequence"],
        "target_index": int(row["target_index"]),
        "source_gap": int(row["gap"]),
        "hard_quality_label": int(row["hard_quality_label"]),
        "high_payload_label": int(row["high_payload_label"]),
        "adaptive_target_is_keyframe": int(row["stage165_adaptive_target_is_keyframe"]),
        "uniform_gap8_target_is_keyframe": int(row["uniform_gap8_target_is_keyframe"]),
        "uniform_gap4_target_is_keyframe": int(row["uniform_gap4_target_is_keyframe"]),
        "stage166_payload_bytes": float(row["payload_bytes"]),
        "stage166_psnr": float(row["stage158_psnr"]),
        "stage166_lpips": float(row["stage158_lpips"]),
        "priority_score": float(priority_score),
        "stage170_core_target": int(stage170_core),
        "selection_reason": reason,
    }


def score_positive(row):
    return 200.0 * int(row["hard_quality_label"]) + 120.0 * int(row["high_payload_label"]) + float(row["payload_bytes"]) / 1000.0 + float(row["stage158_lpips"]) * 20.0


def score_payload_control(row):
    return float(row["payload_bytes"]) / 1000.0 + float(row["stage158_lpips"]) * 10.0


def score_false_positive(row):
    return float(row["payload_bytes"]) / 1000.0 + float(row["stage158_lpips"]) * 30.0


def score_easy(row):
    return float(row["stage158_psnr"]) * 10.0 - float(row["stage158_lpips"]) * 20.0 - float(row["payload_bytes"]) / 10000.0


def take_candidates(rows, used, predicate, category, score_fn, limit, reason):
    candidates = []
    for row in rows:
        key = target_key(row["sequence"], row["target_index"])
        if key in used or not predicate(row):
            continue
        candidates.append(base_target(row, category, score_fn(row), reason, 0))
    candidates.sort(key=lambda item: (float(item["priority_score"]), float(item["stage166_payload_bytes"])), reverse=True)
    out = []
    for item in candidates:
        if item["target_key"] in used:
            continue
        used.add(item["target_key"])
        out.append(item)
        if len(out) >= limit:
            break
    return out


def build_targets(stage166_rows, stage169_targets, max_targets):
    by_key = stage166_map(stage166_rows)
    targets = []
    used = set()
    for core in stage169_targets:
        key = core["target_key"]
        row = by_key[key]
        score = 1000.0 + float(core.get("priority_score", 0.0))
        item = base_target(row, core["category"], score, "Stage170 combined-validation core target", 1)
        used.add(key)
        targets.append(item)

    targets.extend(take_candidates(
        stage166_rows,
        used,
        lambda row: int(row["stage165_adaptive_target_is_keyframe"]) and (int(row["hard_quality_label"]) or int(row["high_payload_label"])),
        "positive_promoted_extension",
        score_positive,
        8,
        "additional adaptive-promoted hard/high-payload target",
    ))
    targets.extend(take_candidates(
        stage166_rows,
        used,
        lambda row: (not int(row["stage165_adaptive_target_is_keyframe"])) and int(row["high_payload_label"]) and not int(row["hard_quality_label"]),
        "high_payload_residual_control_extension",
        score_payload_control,
        8,
        "additional high-payload residual target not promoted by adaptive schedule",
    ))
    targets.extend(take_candidates(
        stage166_rows,
        used,
        lambda row: int(row["stage165_adaptive_target_is_keyframe"]) and not int(row["hard_quality_label"]) and not int(row["high_payload_label"]),
        "selector_false_positive_keyframe_control",
        score_false_positive,
        4,
        "adaptive keyframe selection without hard/high-payload label",
    ))
    remaining = max(0, max_targets - len(targets))
    targets.extend(take_candidates(
        stage166_rows,
        used,
        lambda row: (not int(row["stage165_adaptive_target_is_keyframe"])) and not int(row["hard_quality_label"]) and not int(row["high_payload_label"]),
        "normal_residual_control",
        score_easy,
        remaining,
        "normal/easy residual control not promoted by adaptive schedule",
    ))
    targets = targets[:max_targets]
    for rank, row in enumerate(targets, 1):
        row["rank"] = rank
    return targets


def expected_status(target, schedule):
    if schedule == "uniform_gap8" and int(target["uniform_gap8_target_is_keyframe"]):
        return "target_keyframe_no_middle_render"
    if schedule == "stage165_adaptive" and int(target["adaptive_target_is_keyframe"]):
        return "target_keyframe_no_middle_render"
    if schedule == "uniform_gap4" and int(target["uniform_gap4_target_is_keyframe"]):
        return "target_keyframe_no_middle_render"
    return "rendered_middle_recovery"


def build_schedule_rows(targets, existing_sources):
    rows = []
    for target in targets:
        for schedule in SCHEDULES:
            existing = existing_sources.get((target["target_key"], schedule), "")
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
                "requires_stage174_render": requires,
                "priority_score": float(target["priority_score"]),
                "notes": "reuse existing rendered/keyframe row" if existing else ("metadata/keyframe marker only" if status != "rendered_middle_recovery" else "render in Stage174"),
            })
    return rows


def summarize(targets, schedule_rows):
    by_category = defaultdict(list)
    sched_by_category = defaultdict(list)
    for row in targets:
        by_category[row["category"]].append(row)
    for row in schedule_rows:
        sched_by_category[row["category"]].append(row)
    out = []
    for category, group in sorted(by_category.items()):
        sched = sched_by_category[category]
        out.append({
            "category": category,
            "target_count": len(group),
            "schedule_row_count": len(sched),
            "existing_schedule_rows": sum(1 for row in sched if row["existing_source"]),
            "new_render_schedule_rows": sum(int(row["requires_stage174_render"]) for row in sched),
            "keyframe_marker_rows": sum(1 for row in sched if row["expected_status"] == "target_keyframe_no_middle_render" and not row["existing_source"]),
            "mean_payload_bytes": mean(row["stage166_payload_bytes"] for row in group),
            "mean_psnr": mean(row["stage166_psnr"] for row in group),
            "mean_lpips": mean(row["stage166_lpips"] for row in group),
            "hard_count": sum(int(row["hard_quality_label"]) for row in group),
            "high_payload_count": sum(int(row["high_payload_label"]) for row in group),
        })
    return out


def write_report(targets, schedule_rows, summary_rows, package, path):
    counts = Counter(row["category"] for row in targets)
    lines = [
        "# Stage173 Medium Rendered Validation Protocol",
        "",
        "## Scope",
        "",
        "This is a protocol-only stage. It selects medium-scale targets and schedule rows for Stage174 without rendering.",
        "Stage174 should reuse Stage167/168/170 rows and render only rows with `requires_stage174_render=1`.",
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
        "| category | targets | schedule rows | existing | new renders | keyframe markers | mean payload | mean PSNR | mean LPIPS | hard | high payload |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        lines.append(
            f"| {row['category']} | {row['target_count']} | {row['schedule_row_count']} | {row['existing_schedule_rows']} | "
            f"{row['new_render_schedule_rows']} | {row['keyframe_marker_rows']} | {float(row['mean_payload_bytes']):.3f} | "
            f"{float(row['mean_psnr']):.6f} | {float(row['mean_lpips']):.6f} | {row['hard_count']} | {row['high_payload_count']} |"
        )
    lines.extend([
        "",
        "## Stage174 Contract",
        "",
        "- Render only rows with `requires_stage174_render=1`.",
        "- Reuse rows marked by `existing_source` from Stage167/168/170.",
        "- For target keyframes, record `target_keyframe_no_middle_render`; do not claim middle-render metrics.",
        "- Keep Stage158 `streamsplat_guided_half_anchor_entropy_residual_v1` fixed.",
        "- Keep heavy contact sheets outside repo.",
        "",
        "## Outputs",
        "",
        f"- Targets CSV: `{package['targets_csv']}`",
        f"- Schedule rows CSV: `{package['schedule_rows_csv']}`",
        f"- Summary CSV: `{package['summary_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage166_rows", type=Path, default=DEFAULT_STAGE166_ROWS)
    parser.add_argument("--stage169_targets", type=Path, default=DEFAULT_STAGE169_TARGETS)
    parser.add_argument("--stage167_rows", type=Path, default=DEFAULT_STAGE167_ROWS)
    parser.add_argument("--stage168_rows", type=Path, default=DEFAULT_STAGE168_ROWS)
    parser.add_argument("--stage170_rows", type=Path, default=DEFAULT_STAGE170_ROWS)
    parser.add_argument("--stage172_package", type=Path, default=DEFAULT_STAGE172_PACKAGE)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max_targets", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage166_rows = read_csv(args.stage166_rows)
    stage169_targets = read_csv(args.stage169_targets)
    stage172 = read_json(args.stage172_package)
    existing_sources = schedule_sources(
        ("stage167", args.stage167_rows),
        ("stage168", args.stage168_rows),
        ("stage170", args.stage170_rows),
    )
    targets = build_targets(stage166_rows, stage169_targets, args.max_targets)
    schedule_rows = build_schedule_rows(targets, existing_sources)
    summary_rows = summarize(targets, schedule_rows)
    targets_csv = args.output_root / "stage173_medium_validation_targets.csv"
    schedule_rows_csv = args.output_root / "stage173_medium_validation_schedule_rows.csv"
    summary_csv = args.output_root / "stage173_medium_validation_summary.csv"
    package_json = args.output_root / "stage173_medium_rendered_validation_protocol_package.json"
    report_md = args.output_root / "stage173_medium_rendered_validation_protocol_report.md"
    write_csv(targets, targets_csv, TARGET_FIELDS)
    write_csv(schedule_rows, schedule_rows_csv, SCHEDULE_ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    package = {
        "stage": 173,
        "status": "medium_rendered_validation_protocol_packaged",
        "stage172_decision": stage172["decision"],
        "target_count": len(targets),
        "schedule_row_count": len(schedule_rows),
        "existing_schedule_rows": sum(1 for row in schedule_rows if row["existing_source"]),
        "new_render_schedule_rows": sum(int(row["requires_stage174_render"]) for row in schedule_rows),
        "keyframe_marker_rows": sum(1 for row in schedule_rows if row["expected_status"] == "target_keyframe_no_middle_render" and not row["existing_source"]),
        "stage170_core_target_count": sum(int(row["stage170_core_target"]) for row in targets),
        "summary_rows": summary_rows,
        "targets_csv": str(targets_csv),
        "schedule_rows_csv": str(schedule_rows_csv),
        "summary_csv": str(summary_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(targets, schedule_rows, summary_rows, package, report_md)
    print(json.dumps({"package": str(package_json), "target_count": len(targets), "new_render_schedule_rows": package["new_render_schedule_rows"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
