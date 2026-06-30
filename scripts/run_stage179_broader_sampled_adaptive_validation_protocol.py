import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE166_ROWS = REPO_ROOT / "experiments/stage166_adaptive_schedule_label_rd_comparison/stage166_sampled_row_consequences.csv"
DEFAULT_STAGE173_TARGETS = REPO_ROOT / "experiments/stage173_medium_rendered_validation_protocol/stage173_medium_validation_targets.csv"
DEFAULT_STAGE174_ROWS = REPO_ROOT / "experiments/stage174_medium_rendered_validation_execution/stage174_medium_validation_rows.csv"
DEFAULT_STAGE177_QUALITY_ROWS = REPO_ROOT / "experiments/stage177_selector_fixed_gap_psnr_comparison/stage177_final_quality_by_schedule.csv"
DEFAULT_STAGE176_PACKAGE = REPO_ROOT / "experiments/stage176_adaptive_schedule_candidate_package/stage176_adaptive_schedule_candidate_package.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage179_broader_sampled_adaptive_validation_protocol"

SCHEDULES = ["uniform_gap8", "stage165_adaptive", "uniform_gap4"]
WEAK_SUBJECTIVE_SEQUENCES = {
    "bike-packing",
    "breakdance",
    "camel",
    "cows",
    "dance-twirl",
    "goat",
    "motocross-jump",
    "scooter-black",
    "shooting",
    "soapbox",
}

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
    "stage174_core_target",
    "selection_reason",
]
SCHEDULE_ROW_FIELDS = [
    "target_key",
    "category",
    "sequence",
    "target_index",
    "schedule",
    "expected_status",
    "existing_schedule_source",
    "existing_middle_metric_source",
    "existing_keyframe_metric_source",
    "requires_stage180_render",
    "requires_stage180_keyframe_metric",
    "priority_score",
    "stage174_core_target",
    "selection_reason",
    "notes",
]
SUMMARY_FIELDS = [
    "category",
    "target_count",
    "stage174_core_targets",
    "schedule_row_count",
    "existing_schedule_rows",
    "existing_middle_metric_rows",
    "existing_keyframe_metric_rows",
    "new_render_rows",
    "new_keyframe_metric_rows",
    "keyframe_schedule_rows",
    "mean_payload_bytes",
    "mean_psnr",
    "mean_lpips",
    "hard_count",
    "high_payload_count",
]
SOURCE_FIELDS = [
    "source_type",
    "source",
    "row_count",
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


def as_int(value):
    return int(float(value))


def target_key(sequence, target_index):
    return f"{sequence}:{int(target_index):05d}"


def stage166_map(rows):
    return {target_key(row["sequence"], row["target_index"]): row for row in rows}


def schedule_maps(stage174_rows, stage177_rows):
    schedule = {}
    middle_metrics = {}
    keyframe_metrics = {}
    for row in stage174_rows:
        key = (row["target_key"], row["schedule"])
        source = f"stage174:{row.get('row_source', '')}"
        schedule[key] = source
        if row.get("status") == "rendered_middle_recovery":
            middle_metrics[key] = source
    for row in stage177_rows:
        key = (row["target_key"], row["schedule"])
        if row.get("final_type") == "q12_target_keyframe":
            keyframe_metrics[key] = f"stage177:{row.get('row_source', '')}"
        elif row.get("final_type") == "middle_recovery" and key not in middle_metrics:
            middle_metrics[key] = f"stage177:{row.get('row_source', '')}"
    return schedule, middle_metrics, keyframe_metrics


def base_target(row, category, priority_score, reason, stage174_core):
    return {
        "rank": 0,
        "target_key": target_key(row["sequence"], row["target_index"]),
        "category": category,
        "source_task_id": row["task_id"],
        "sequence": row["sequence"],
        "target_index": int(row["target_index"]),
        "source_gap": int(row["gap"]),
        "hard_quality_label": as_int(row["hard_quality_label"]),
        "high_payload_label": as_int(row["high_payload_label"]),
        "adaptive_target_is_keyframe": as_int(row["stage165_adaptive_target_is_keyframe"]),
        "uniform_gap8_target_is_keyframe": as_int(row["uniform_gap8_target_is_keyframe"]),
        "uniform_gap4_target_is_keyframe": as_int(row["uniform_gap4_target_is_keyframe"]),
        "stage166_payload_bytes": float(row["payload_bytes"]),
        "stage166_psnr": float(row["stage158_psnr"]),
        "stage166_lpips": float(row["stage158_lpips"]),
        "priority_score": float(priority_score),
        "stage174_core_target": int(stage174_core),
        "selection_reason": reason,
    }


def score_difficulty(row):
    hard = as_int(row["hard_quality_label"])
    high = as_int(row["high_payload_label"])
    payload = float(row["payload_bytes"])
    lpips = float(row["stage158_lpips"])
    psnr = float(row["stage158_psnr"])
    return hard * 300.0 + high * 220.0 + payload / 1000.0 + lpips * 80.0 + max(0.0, 30.0 - psnr) * 12.0


def score_positive(row):
    return score_difficulty(row) + 200.0 * as_int(row["stage165_adaptive_target_is_keyframe"])


def score_false_positive(row):
    return float(row["payload_bytes"]) / 1000.0 + float(row["stage158_lpips"]) * 60.0 + float(row["stage158_psnr"]) * 0.5


def score_normal(row):
    return float(row["stage158_psnr"]) * 4.0 - float(row["stage158_lpips"]) * 20.0 + float(row["payload_bytes"]) / 20000.0


def add_items(rows, used, targets, predicate, category, score_fn, limit, reason, max_targets):
    remaining = max_targets - len(targets)
    if remaining <= 0 or limit <= 0:
        return
    take_limit = min(limit, remaining)
    candidates = []
    for row in rows:
        key = target_key(row["sequence"], row["target_index"])
        if key in used or not predicate(row):
            continue
        candidates.append(base_target(row, category, score_fn(row), reason, 0))
    candidates.sort(key=lambda item: (float(item["priority_score"]), item["sequence"], int(item["target_index"])), reverse=True)
    for item in candidates:
        if item["target_key"] in used:
            continue
        used.add(item["target_key"])
        targets.append(item)
        if sum(1 for row in targets if row["category"] == category and row["selection_reason"] == reason) >= take_limit:
            break


def add_sequence_coverage(rows, used, targets, max_targets):
    represented = {row["sequence"] for row in targets}
    by_sequence = defaultdict(list)
    for row in rows:
        key = target_key(row["sequence"], row["target_index"])
        if key in used or row["sequence"] in represented:
            continue
        by_sequence[row["sequence"]].append(row)
    for sequence in sorted(by_sequence):
        if len(targets) >= max_targets:
            break
        candidates = sorted(by_sequence[sequence], key=lambda row: (score_difficulty(row), int(row["target_index"])), reverse=True)
        row = candidates[0]
        item = base_target(row, "broader_sequence_coverage_probe", score_difficulty(row), "one representative target for sequence-level coverage", 0)
        used.add(item["target_key"])
        targets.append(item)


def add_stage174_core(stage166_rows, stage173_targets, max_targets):
    by_key = stage166_map(stage166_rows)
    targets = []
    used = set()
    for core in stage173_targets:
        if len(targets) >= max_targets:
            break
        key = core["target_key"]
        if key not in by_key or key in used:
            continue
        row = by_key[key]
        score = 10000.0 + float(core.get("priority_score", 0.0) or 0.0)
        item = base_target(row, core["category"], score, "Stage174 medium-validation core target", 1)
        used.add(key)
        targets.append(item)
    return targets, used


def build_targets(stage166_rows, stage173_targets, max_targets):
    targets, used = add_stage174_core(stage166_rows, stage173_targets, max_targets)
    add_sequence_coverage(stage166_rows, used, targets, max_targets)
    add_items(
        stage166_rows,
        used,
        targets,
        lambda row: (not as_int(row["stage165_adaptive_target_is_keyframe"])) and (as_int(row["hard_quality_label"]) or as_int(row["high_payload_label"])),
        "broader_false_negative_residual",
        score_difficulty,
        12,
        "broader residual target missed by adaptive schedule",
        max_targets,
    )
    add_items(
        stage166_rows,
        used,
        targets,
        lambda row: as_int(row["stage165_adaptive_target_is_keyframe"]) and (as_int(row["hard_quality_label"]) or as_int(row["high_payload_label"])),
        "broader_positive_promoted",
        score_positive,
        18,
        "broader adaptive-promoted hard/high-payload target",
        max_targets,
    )
    add_items(
        stage166_rows,
        used,
        targets,
        lambda row: as_int(row["stage165_adaptive_target_is_keyframe"]) and not as_int(row["hard_quality_label"]) and not as_int(row["high_payload_label"]),
        "broader_selector_false_positive_keyframe_control",
        score_false_positive,
        8,
        "broader adaptive keyframe selection without hard/high-payload label",
        max_targets,
    )
    add_items(
        stage166_rows,
        used,
        targets,
        lambda row: (not as_int(row["stage165_adaptive_target_is_keyframe"])) and as_int(row["high_payload_label"]),
        "broader_high_payload_residual_control",
        score_difficulty,
        8,
        "broader high-payload residual target not promoted by adaptive schedule",
        max_targets,
    )
    add_items(
        stage166_rows,
        used,
        targets,
        lambda row: row["sequence"] in WEAK_SUBJECTIVE_SEQUENCES,
        "broader_weak_sequence_probe",
        score_difficulty,
        8,
        "additional target from weak subjective sequence set",
        max_targets,
    )
    add_items(
        stage166_rows,
        used,
        targets,
        lambda row: not as_int(row["hard_quality_label"]) and not as_int(row["high_payload_label"]),
        "broader_normal_residual_control",
        score_normal,
        max_targets,
        "broader normal/easy residual control",
        max_targets,
    )
    for rank, row in enumerate(targets[:max_targets], 1):
        row["rank"] = rank
    return targets[:max_targets]


def expected_status(target, schedule):
    if schedule == "uniform_gap8" and int(target["uniform_gap8_target_is_keyframe"]):
        return "target_keyframe_no_middle_render"
    if schedule == "stage165_adaptive" and int(target["adaptive_target_is_keyframe"]):
        return "target_keyframe_no_middle_render"
    if schedule == "uniform_gap4" and int(target["uniform_gap4_target_is_keyframe"]):
        return "target_keyframe_no_middle_render"
    return "rendered_middle_recovery"


def build_schedule_rows(targets, existing_schedule, middle_metrics, keyframe_metrics):
    rows = []
    for target in targets:
        for schedule in SCHEDULES:
            key = (target["target_key"], schedule)
            status = expected_status(target, schedule)
            existing_source = existing_schedule.get(key, "")
            middle_source = middle_metrics.get(key, "")
            keyframe_source = keyframe_metrics.get(key, "")
            requires_render = int(status == "rendered_middle_recovery" and middle_source == "")
            requires_keyframe_metric = int(status == "target_keyframe_no_middle_render" and keyframe_source == "")
            if middle_source:
                notes = "reuse existing middle metric"
            elif keyframe_source:
                notes = "reuse existing q12 keyframe final-quality metric"
            elif requires_render:
                notes = "render Stage158 middle recovery in Stage180"
            elif requires_keyframe_metric:
                notes = "render q12 target keyframe metric in Stage180"
            else:
                notes = "metadata-only schedule marker"
            rows.append({
                "target_key": target["target_key"],
                "category": target["category"],
                "sequence": target["sequence"],
                "target_index": int(target["target_index"]),
                "schedule": schedule,
                "expected_status": status,
                "existing_schedule_source": existing_source,
                "existing_middle_metric_source": middle_source,
                "existing_keyframe_metric_source": keyframe_source,
                "requires_stage180_render": requires_render,
                "requires_stage180_keyframe_metric": requires_keyframe_metric,
                "priority_score": float(target["priority_score"]),
                "stage174_core_target": int(target["stage174_core_target"]),
                "selection_reason": target["selection_reason"],
                "notes": notes,
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
            "stage174_core_targets": sum(int(row["stage174_core_target"]) for row in group),
            "schedule_row_count": len(sched),
            "existing_schedule_rows": sum(1 for row in sched if row["existing_schedule_source"]),
            "existing_middle_metric_rows": sum(1 for row in sched if row["existing_middle_metric_source"]),
            "existing_keyframe_metric_rows": sum(1 for row in sched if row["existing_keyframe_metric_source"]),
            "new_render_rows": sum(int(row["requires_stage180_render"]) for row in sched),
            "new_keyframe_metric_rows": sum(int(row["requires_stage180_keyframe_metric"]) for row in sched),
            "keyframe_schedule_rows": sum(1 for row in sched if row["expected_status"] == "target_keyframe_no_middle_render"),
            "mean_payload_bytes": mean(row["stage166_payload_bytes"] for row in group),
            "mean_psnr": mean(row["stage166_psnr"] for row in group),
            "mean_lpips": mean(row["stage166_lpips"] for row in group),
            "hard_count": sum(int(row["hard_quality_label"]) for row in group),
            "high_payload_count": sum(int(row["high_payload_label"]) for row in group),
        })
    return out


def source_summary(schedule_rows):
    counts = Counter()
    for row in schedule_rows:
        if row["existing_middle_metric_source"]:
            counts[("existing_middle_metric", row["existing_middle_metric_source"])] += 1
        if row["existing_keyframe_metric_source"]:
            counts[("existing_keyframe_metric", row["existing_keyframe_metric_source"])] += 1
        if row["requires_stage180_render"]:
            counts[("new_work", "stage180_middle_render")] += 1
        if row["requires_stage180_keyframe_metric"]:
            counts[("new_work", "stage180_q12_keyframe_metric")] += 1
    return [{"source_type": kind, "source": source, "row_count": count} for (kind, source), count in sorted(counts.items())]


def write_report(targets, schedule_rows, summary_rows, source_rows, package, path):
    counts = Counter(row["category"] for row in targets)
    lines = [
        "# Stage179 Broader Sampled Adaptive Validation Protocol",
        "",
        "## Scope",
        "",
        "This is a protocol-only stage. It expands the Stage174 medium validation target set and prepares Stage180 execution inputs.",
        "Stage179 performs no rendering and writes no heavy media.",
        "",
        "## Target Counts",
        "",
    ]
    for category, count in sorted(counts.items()):
        lines.append(f"- `{category}`: `{count}` targets")
    lines.extend([
        "",
        "## Work Summary",
        "",
        f"- Targets: `{package['target_count']}`.",
        f"- Schedule rows: `{package['schedule_row_count']}`.",
        f"- Stage174 core targets retained: `{package['stage174_core_target_count']}`.",
        f"- New targets beyond Stage174: `{package['new_target_count']}`.",
        f"- Existing middle metric rows: `{package['existing_middle_metric_rows']}`.",
        f"- Existing keyframe metric rows: `{package['existing_keyframe_metric_rows']}`.",
        f"- Stage180 middle renders required: `{package['new_render_rows']}`.",
        f"- Stage180 q12 keyframe metrics required: `{package['new_keyframe_metric_rows']}`.",
        "",
        "## Category Summary",
        "",
        "| category | targets | core | schedule rows | existing middle | existing keyframe | new renders | new keyframes | keyframe rows | mean payload | mean PSNR | mean LPIPS | hard | high payload |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        lines.append(
            f"| {row['category']} | {row['target_count']} | {row['stage174_core_targets']} | {row['schedule_row_count']} | "
            f"{row['existing_middle_metric_rows']} | {row['existing_keyframe_metric_rows']} | {row['new_render_rows']} | "
            f"{row['new_keyframe_metric_rows']} | {row['keyframe_schedule_rows']} | {float(row['mean_payload_bytes']):.3f} | "
            f"{float(row['mean_psnr']):.6f} | {float(row['mean_lpips']):.6f} | {row['hard_count']} | {row['high_payload_count']} |"
        )
    lines.extend([
        "",
        "## Reuse And New Work",
        "",
        "| source type | source | rows |",
        "|---|---|---:|",
    ])
    for row in source_rows:
        lines.append(f"| {row['source_type']} | {row['source']} | {row['row_count']} |")
    lines.extend([
        "",
        "## Stage180 Contract",
        "",
        "- Render only middle rows with `requires_stage180_render=1`.",
        "- Render q12 target keyframe metrics only for keyframe rows with `requires_stage180_keyframe_metric=1`.",
        "- Reuse Stage174 middle metrics and Stage177 final-quality keyframe metrics where available.",
        "- Keep Stage158 `streamsplat_guided_half_anchor_entropy_residual_v1` fixed.",
        "- Keep heavy contact sheets outside git if Stage180 exports visuals.",
        "",
        "## Outputs",
        "",
        f"- Targets CSV: `{package['targets_csv']}`",
        f"- Schedule rows CSV: `{package['schedule_rows_csv']}`",
        f"- Summary CSV: `{package['summary_csv']}`",
        f"- Source summary CSV: `{package['source_summary_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage166_rows", type=Path, default=DEFAULT_STAGE166_ROWS)
    parser.add_argument("--stage173_targets", type=Path, default=DEFAULT_STAGE173_TARGETS)
    parser.add_argument("--stage174_rows", type=Path, default=DEFAULT_STAGE174_ROWS)
    parser.add_argument("--stage177_quality_rows", type=Path, default=DEFAULT_STAGE177_QUALITY_ROWS)
    parser.add_argument("--stage176_package", type=Path, default=DEFAULT_STAGE176_PACKAGE)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max_targets", type=int, default=90)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage166_rows = read_csv(args.stage166_rows)
    stage173_targets = read_csv(args.stage173_targets)
    stage174_rows = read_csv(args.stage174_rows)
    stage177_rows = read_csv(args.stage177_quality_rows)
    stage176 = read_json(args.stage176_package)
    existing_schedule, middle_metrics, keyframe_metrics = schedule_maps(stage174_rows, stage177_rows)
    targets = build_targets(stage166_rows, stage173_targets, args.max_targets)
    schedule_rows = build_schedule_rows(targets, existing_schedule, middle_metrics, keyframe_metrics)
    summary_rows = summarize(targets, schedule_rows)
    source_rows = source_summary(schedule_rows)
    targets_csv = args.output_root / "stage179_broader_validation_targets.csv"
    schedule_rows_csv = args.output_root / "stage179_broader_validation_schedule_rows.csv"
    summary_csv = args.output_root / "stage179_broader_validation_summary.csv"
    source_summary_csv = args.output_root / "stage179_broader_validation_source_summary.csv"
    package_json = args.output_root / "stage179_broader_sampled_adaptive_validation_protocol_package.json"
    report_md = args.output_root / "stage179_broader_sampled_adaptive_validation_protocol_report.md"
    write_csv(targets, targets_csv, TARGET_FIELDS)
    write_csv(schedule_rows, schedule_rows_csv, SCHEDULE_ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(source_rows, source_summary_csv, SOURCE_FIELDS)
    package = {
        "stage": 179,
        "status": "broader_sampled_adaptive_validation_protocol_packaged",
        "policy": stage176.get("policy", "rgb_motion_rank_gate_gap8_plus_extra_targets_v1_sampled_candidate"),
        "candidate_status": stage176.get("candidate_status", "sampled_validated_candidate_not_final_full_sequence_rd"),
        "target_count": len(targets),
        "schedule_row_count": len(schedule_rows),
        "stage174_core_target_count": sum(int(row["stage174_core_target"]) for row in targets),
        "new_target_count": sum(1 for row in targets if not int(row["stage174_core_target"])),
        "existing_schedule_rows": sum(1 for row in schedule_rows if row["existing_schedule_source"]),
        "existing_middle_metric_rows": sum(1 for row in schedule_rows if row["existing_middle_metric_source"]),
        "existing_keyframe_metric_rows": sum(1 for row in schedule_rows if row["existing_keyframe_metric_source"]),
        "new_render_rows": sum(int(row["requires_stage180_render"]) for row in schedule_rows),
        "new_keyframe_metric_rows": sum(int(row["requires_stage180_keyframe_metric"]) for row in schedule_rows),
        "keyframe_schedule_rows": sum(1 for row in schedule_rows if row["expected_status"] == "target_keyframe_no_middle_render"),
        "category_counts": dict(sorted(Counter(row["category"] for row in targets).items())),
        "summary_rows": summary_rows,
        "source_summary_rows": source_rows,
        "targets_csv": str(targets_csv),
        "schedule_rows_csv": str(schedule_rows_csv),
        "summary_csv": str(summary_csv),
        "source_summary_csv": str(source_summary_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(targets, schedule_rows, summary_rows, source_rows, package, report_md)
    print(json.dumps({
        "package": str(package_json),
        "target_count": len(targets),
        "new_render_rows": package["new_render_rows"],
        "new_keyframe_metric_rows": package["new_keyframe_metric_rows"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
