import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE163_ROWS = REPO_ROOT / "experiments/stage163_davis_rgb_motion_selector_data/stage163_davis_rgb_motion_selector_feature_rows.csv"
DEFAULT_STAGE165_SELECTED_ROWS = REPO_ROOT / "experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_multifeature_selected_rows.csv"
DEFAULT_STAGE165_SCHEDULE_ROWS = REPO_ROOT / "experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_adaptive_schedule_rows.csv"
DEFAULT_STAGE165_PACKAGE = REPO_ROOT / "experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_multifeature_keyframe_schedule_preflight_package.json"
DEFAULT_STAGE147_ROWS = REPO_ROOT / "experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_rows.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage166_adaptive_schedule_label_rd_comparison"

SCHEDULES = ["uniform_gap8", "stage165_adaptive", "uniform_gap4"]

SCHEDULE_COMPARISON_FIELDS = [
    "schedule", "total_frames", "total_keyframe_count", "keyframe_ratio", "keyframe_delta_vs_uniform_gap8",
    "keyframe_delta_vs_uniform_gap4", "metadata_bits", "metadata_bytes", "metadata_mib", "metadata_mib_per_frame",
    "main_anchor_mib_per_frame_proxy", "anchor_rate_interpolation_weight", "sampled_row_count", "promoted_row_count",
    "residual_row_count", "promoted_row_fraction", "hard_quality_total", "promoted_hard_quality_count",
    "hard_quality_coverage", "residual_hard_quality_count", "high_payload_total", "promoted_high_payload_count",
    "high_payload_coverage", "residual_high_payload_count", "residual_payload_bytes", "avoided_payload_bytes",
    "avoided_payload_fraction", "residual_side_mib", "avoided_side_mib", "mean_residual_side_mib_per_sample",
    "mean_remaining_middle_psnr", "mean_remaining_middle_lpips", "total_proxy_mib_per_frame",
]

ROW_CONSEQUENCE_FIELDS = [
    "task_id", "sequence", "gap", "left_index", "target_index", "right_index", "normalized_time",
    "stage165_selected_for_extra_keyframe", "hard_quality_label", "high_payload_label", "stage158_psnr", "stage158_lpips",
    "payload_bytes", "side_mib_per_intermediate", "q12_main_anchor_mib_per_frame_ref",
    "uniform_gap8_target_is_keyframe", "stage165_adaptive_target_is_keyframe", "uniform_gap4_target_is_keyframe",
    "uniform_gap8_residual_payload_bytes", "stage165_adaptive_residual_payload_bytes", "uniform_gap4_residual_payload_bytes",
    "stage165_adaptive_avoided_payload_bytes_vs_gap8", "stage165_adaptive_false_negative_hard", "stage165_adaptive_false_negative_high_payload",
]

SEQUENCE_COVERAGE_FIELDS = [
    "sequence", "total_frames", "uniform_gap8_keyframe_count", "stage165_adaptive_keyframe_count", "uniform_gap4_keyframe_count",
    "extra_keyframes_vs_gap8", "sampled_rows", "hard_quality_rows", "stage165_adaptive_promoted_rows",
    "stage165_adaptive_promoted_hard_rows", "stage165_adaptive_false_negative_hard_rows", "high_payload_rows",
    "stage165_adaptive_promoted_high_payload_rows", "stage165_adaptive_false_negative_high_payload_rows",
    "stage165_adaptive_avoided_payload_bytes", "stage165_adaptive_residual_payload_bytes",
    "stage165_adaptive_mean_remaining_psnr", "stage165_adaptive_mean_remaining_lpips", "smoke_score",
]

SMOKE_FIELDS = [
    "rank", "sequence", "reason", "smoke_score", "hard_quality_rows", "stage165_adaptive_promoted_hard_rows",
    "stage165_adaptive_false_negative_hard_rows", "high_payload_rows", "stage165_adaptive_promoted_high_payload_rows",
    "extra_keyframes_vs_gap8", "stage165_adaptive_avoided_payload_bytes", "stage165_adaptive_residual_payload_bytes",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def parse_int_set(value):
    value = str(value).strip()
    if not value:
        return set()
    return {int(part) for part in value.split()}


def load_stage147_anchor_refs(path):
    refs = {}
    for row in read_csv(path):
        refs[int(row["reference_gap"])] = float(row["q12_main_anchor_mib_per_frame"])
    return refs


def prepare_stage163_rows(rows):
    out = []
    for row in rows:
        item = dict(row)
        item["gap"] = int(item["gap"])
        item["left_index"] = int(item["left_index"])
        item["target_index"] = int(item["target_index"])
        item["right_index"] = int(item["right_index"])
        item["normalized_time"] = float(item["normalized_time"])
        item["stage158_psnr"] = float(item["stage158_psnr"])
        item["stage158_lpips"] = float(item["stage158_lpips"])
        item["payload_bytes"] = float(item["payload_bytes"])
        item["side_mib_per_intermediate"] = float(item["side_mib_per_intermediate"])
        item["direct_total_mib_per_frame_ref"] = float(item["direct_total_mib_per_frame_ref"])
        item["q12_main_anchor_mib_per_frame_ref"] = item["direct_total_mib_per_frame_ref"] - item["side_mib_per_intermediate"]
        item["hard_quality_label"] = int(float(item.get("label_low_psnr_lt26", 0)) or float(item.get("label_high_lpips_gt022", 0)))
        item["high_payload_label"] = int(float(item.get("label_high_payload_gt220k", 0)))
        out.append(item)
    return out


def prepare_selected_map(rows):
    out = {}
    for row in rows:
        out[row["task_id"]] = int(float(row["selected_for_extra_keyframe"]))
    return out


def prepare_schedule_maps(schedule_rows):
    maps = {name: {} for name in SCHEDULES}
    info = {}
    adaptive_metadata_bits = 0
    for row in schedule_rows:
        sequence = row["sequence"]
        total_frames = int(row["total_frames"])
        maps["uniform_gap8"][sequence] = parse_int_set(row["uniform_gap8_keyframes"])
        maps["stage165_adaptive"][sequence] = parse_int_set(row["adaptive_keyframes"])
        maps["uniform_gap4"][sequence] = parse_int_set(row["uniform_gap4_keyframes"])
        adaptive_metadata_bits += int(row["metadata_bits"])
        info[sequence] = {
            "total_frames": total_frames,
            "uniform_gap8_keyframe_count": int(row["uniform_gap8_keyframe_count"]),
            "stage165_adaptive_keyframe_count": int(row["adaptive_keyframe_count"]),
            "uniform_gap4_keyframe_count": int(row["uniform_gap4_keyframe_count"]),
            "extra_keyframes_vs_gap8": int(row["extra_keyframes_vs_gap8"]),
        }
    metadata_bits = {
        "uniform_gap8": 8,
        "stage165_adaptive": adaptive_metadata_bits,
        "uniform_gap4": 8,
    }
    return maps, info, metadata_bits


def total_keyframes(schedule_maps, schedule):
    return sum(len(keyframes) for keyframes in schedule_maps[schedule].values())


def total_frames(sequence_info):
    return sum(int(item["total_frames"]) for item in sequence_info.values())


def main_anchor_proxy(schedule, keyframe_counts, refs):
    gap8_ref = float(refs[8])
    gap4_ref = float(refs[4])
    if schedule == "uniform_gap8":
        return gap8_ref, 0.0
    if schedule == "uniform_gap4":
        return gap4_ref, 1.0
    denom = keyframe_counts["uniform_gap4"] - keyframe_counts["uniform_gap8"]
    weight = 0.0 if denom == 0 else (keyframe_counts["stage165_adaptive"] - keyframe_counts["uniform_gap8"]) / denom
    proxy = gap8_ref + weight * (gap4_ref - gap8_ref)
    return proxy, weight


def build_row_consequences(rows, selected_map, schedule_maps):
    out = []
    for row in rows:
        sequence = row["sequence"]
        target = int(row["target_index"])
        promoted = {schedule: int(target in schedule_maps[schedule][sequence]) for schedule in SCHEDULES}
        payload = float(row["payload_bytes"])
        item = {
            "task_id": row["task_id"],
            "sequence": sequence,
            "gap": int(row["gap"]),
            "left_index": int(row["left_index"]),
            "target_index": target,
            "right_index": int(row["right_index"]),
            "normalized_time": float(row["normalized_time"]),
            "stage165_selected_for_extra_keyframe": int(selected_map.get(row["task_id"], 0)),
            "hard_quality_label": int(row["hard_quality_label"]),
            "high_payload_label": int(row["high_payload_label"]),
            "stage158_psnr": float(row["stage158_psnr"]),
            "stage158_lpips": float(row["stage158_lpips"]),
            "payload_bytes": payload,
            "side_mib_per_intermediate": float(row["side_mib_per_intermediate"]),
            "q12_main_anchor_mib_per_frame_ref": float(row["q12_main_anchor_mib_per_frame_ref"]),
            "uniform_gap8_target_is_keyframe": promoted["uniform_gap8"],
            "stage165_adaptive_target_is_keyframe": promoted["stage165_adaptive"],
            "uniform_gap4_target_is_keyframe": promoted["uniform_gap4"],
            "uniform_gap8_residual_payload_bytes": 0.0 if promoted["uniform_gap8"] else payload,
            "stage165_adaptive_residual_payload_bytes": 0.0 if promoted["stage165_adaptive"] else payload,
            "uniform_gap4_residual_payload_bytes": 0.0 if promoted["uniform_gap4"] else payload,
            "stage165_adaptive_avoided_payload_bytes_vs_gap8": payload if promoted["stage165_adaptive"] and not promoted["uniform_gap8"] else 0.0,
            "stage165_adaptive_false_negative_hard": int(int(row["hard_quality_label"]) and not promoted["stage165_adaptive"]),
            "stage165_adaptive_false_negative_high_payload": int(int(row["high_payload_label"]) and not promoted["stage165_adaptive"]),
        }
        out.append(item)
    return out


def summarize_schedule(schedule, rows, schedule_maps, sequence_info, metadata_bits, keyframe_counts, anchor_refs):
    frames = total_frames(sequence_info)
    keyframes = keyframe_counts[schedule]
    promoted = [row for row in rows if int(row[f"{schedule}_target_is_keyframe"])]
    residual = [row for row in rows if not int(row[f"{schedule}_target_is_keyframe"])]
    hard_total = sum(int(row["hard_quality_label"]) for row in rows)
    payload_total = sum(int(row["high_payload_label"]) for row in rows)
    hard_promoted = sum(int(row["hard_quality_label"]) for row in promoted)
    payload_promoted = sum(int(row["high_payload_label"]) for row in promoted)
    total_payload = sum(float(row["payload_bytes"]) for row in rows)
    residual_payload = sum(float(row["payload_bytes"]) for row in residual)
    avoided_payload = total_payload - residual_payload
    residual_side = sum(float(row["side_mib_per_intermediate"]) for row in residual)
    total_side = sum(float(row["side_mib_per_intermediate"]) for row in rows)
    avoided_side = total_side - residual_side
    metadata_mib = int(metadata_bits[schedule]) / 8.0 / (1024.0 * 1024.0)
    metadata_per_frame = metadata_mib / frames if frames else 0.0
    anchor_proxy, anchor_weight = main_anchor_proxy(schedule, keyframe_counts, anchor_refs)
    mean_residual_side = residual_side / len(rows) if rows else 0.0
    total_proxy = anchor_proxy + mean_residual_side + metadata_per_frame
    return {
        "schedule": schedule,
        "total_frames": frames,
        "total_keyframe_count": keyframes,
        "keyframe_ratio": keyframes / frames if frames else 0.0,
        "keyframe_delta_vs_uniform_gap8": keyframes - keyframe_counts["uniform_gap8"],
        "keyframe_delta_vs_uniform_gap4": keyframes - keyframe_counts["uniform_gap4"],
        "metadata_bits": int(metadata_bits[schedule]),
        "metadata_bytes": math.ceil(int(metadata_bits[schedule]) / 8.0),
        "metadata_mib": metadata_mib,
        "metadata_mib_per_frame": metadata_per_frame,
        "main_anchor_mib_per_frame_proxy": anchor_proxy,
        "anchor_rate_interpolation_weight": anchor_weight,
        "sampled_row_count": len(rows),
        "promoted_row_count": len(promoted),
        "residual_row_count": len(residual),
        "promoted_row_fraction": len(promoted) / len(rows) if rows else 0.0,
        "hard_quality_total": hard_total,
        "promoted_hard_quality_count": hard_promoted,
        "hard_quality_coverage": hard_promoted / hard_total if hard_total else 0.0,
        "residual_hard_quality_count": hard_total - hard_promoted,
        "high_payload_total": payload_total,
        "promoted_high_payload_count": payload_promoted,
        "high_payload_coverage": payload_promoted / payload_total if payload_total else 0.0,
        "residual_high_payload_count": payload_total - payload_promoted,
        "residual_payload_bytes": residual_payload,
        "avoided_payload_bytes": avoided_payload,
        "avoided_payload_fraction": avoided_payload / total_payload if total_payload else 0.0,
        "residual_side_mib": residual_side,
        "avoided_side_mib": avoided_side,
        "mean_residual_side_mib_per_sample": mean_residual_side,
        "mean_remaining_middle_psnr": mean(row["stage158_psnr"] for row in residual),
        "mean_remaining_middle_lpips": mean(row["stage158_lpips"] for row in residual),
        "total_proxy_mib_per_frame": total_proxy,
    }


def build_sequence_coverage(rows, sequence_info):
    groups = defaultdict(list)
    for row in rows:
        groups[row["sequence"]].append(row)
    out = []
    for sequence, group in sorted(groups.items()):
        promoted = [row for row in group if int(row["stage165_adaptive_target_is_keyframe"])]
        residual = [row for row in group if not int(row["stage165_adaptive_target_is_keyframe"])]
        hard_total = sum(int(row["hard_quality_label"]) for row in group)
        hard_promoted = sum(int(row["hard_quality_label"]) for row in promoted)
        payload_total = sum(int(row["high_payload_label"]) for row in group)
        payload_promoted = sum(int(row["high_payload_label"]) for row in promoted)
        avoided_payload = sum(float(row["payload_bytes"]) for row in promoted)
        residual_payload = sum(float(row["payload_bytes"]) for row in residual)
        info = sequence_info[sequence]
        false_hard = hard_total - hard_promoted
        false_payload = payload_total - payload_promoted
        smoke_score = 5 * false_hard + 4 * hard_promoted + 2 * payload_promoted + int(info["extra_keyframes_vs_gap8"])
        out.append({
            "sequence": sequence,
            "total_frames": int(info["total_frames"]),
            "uniform_gap8_keyframe_count": int(info["uniform_gap8_keyframe_count"]),
            "stage165_adaptive_keyframe_count": int(info["stage165_adaptive_keyframe_count"]),
            "uniform_gap4_keyframe_count": int(info["uniform_gap4_keyframe_count"]),
            "extra_keyframes_vs_gap8": int(info["extra_keyframes_vs_gap8"]),
            "sampled_rows": len(group),
            "hard_quality_rows": hard_total,
            "stage165_adaptive_promoted_rows": len(promoted),
            "stage165_adaptive_promoted_hard_rows": hard_promoted,
            "stage165_adaptive_false_negative_hard_rows": false_hard,
            "high_payload_rows": payload_total,
            "stage165_adaptive_promoted_high_payload_rows": payload_promoted,
            "stage165_adaptive_false_negative_high_payload_rows": false_payload,
            "stage165_adaptive_avoided_payload_bytes": avoided_payload,
            "stage165_adaptive_residual_payload_bytes": residual_payload,
            "stage165_adaptive_mean_remaining_psnr": mean(row["stage158_psnr"] for row in residual),
            "stage165_adaptive_mean_remaining_lpips": mean(row["stage158_lpips"] for row in residual),
            "smoke_score": smoke_score,
        })
    return out


def smoke_reason(row):
    reasons = []
    if int(row["stage165_adaptive_false_negative_hard_rows"]):
        reasons.append("hard_false_negative")
    if int(row["stage165_adaptive_promoted_hard_rows"]):
        reasons.append("hard_promoted")
    if int(row["stage165_adaptive_promoted_high_payload_rows"]) >= 3:
        reasons.append("payload_heavy_promoted")
    if int(row["extra_keyframes_vs_gap8"]) >= 3:
        reasons.append("many_extra_keyframes")
    return ";".join(reasons) if reasons else "coverage_control"


def choose_smoke_candidates(sequence_rows, max_sequences):
    candidates = [row for row in sequence_rows if int(row["hard_quality_rows"]) > 0 or int(row["stage165_adaptive_promoted_high_payload_rows"]) >= 3]
    candidates.sort(
        key=lambda row: (
            int(row["smoke_score"]),
            int(row["stage165_adaptive_false_negative_hard_rows"]),
            int(row["stage165_adaptive_promoted_hard_rows"]),
            float(row["stage165_adaptive_avoided_payload_bytes"]),
        ),
        reverse=True,
    )
    out = []
    for rank, row in enumerate(candidates[:max_sequences], 1):
        item = dict(row)
        item["rank"] = rank
        item["reason"] = smoke_reason(row)
        out.append(item)
    return out


def write_report(package, comparison, sequence_rows, smoke_rows, path):
    lines = [
        "# Stage166 Adaptive Schedule Label/RD Comparison",
        "",
        "## Scope",
        "",
        "This is a pre-render label/RD proxy. It compares schedule metadata, keyframe count, and sampled Stage158 residual labels; it does not rerender the adaptive schedules.",
        "Promoted rows only mean that the sampled target index becomes a keyframe under that schedule; this is not a substitute for rendered uniform-gap quality evaluation.",
        "",
        "## Schedule Comparison",
        "",
        "| schedule | keys | key ratio | metadata bytes | main anchor proxy | promoted rows | hard coverage | payload coverage | avoided payload | total proxy MiB/frame |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['schedule']} | {row['total_keyframe_count']} | {float(row['keyframe_ratio']):.6f} | "
            f"{row['metadata_bytes']} | {float(row['main_anchor_mib_per_frame_proxy']):.6f} | "
            f"{row['promoted_row_count']}/{row['sampled_row_count']} | {float(row['hard_quality_coverage']):.6f} | "
            f"{float(row['high_payload_coverage']):.6f} | {float(row['avoided_payload_bytes']):.0f} | "
            f"{float(row['total_proxy_mib_per_frame']):.6f} |"
        )
    adaptive = next(row for row in comparison if row["schedule"] == "stage165_adaptive")
    gap8 = next(row for row in comparison if row["schedule"] == "uniform_gap8")
    gap4 = next(row for row in comparison if row["schedule"] == "uniform_gap4")
    lines.extend([
        "",
        "## Adaptive Takeaway",
        "",
        f"- Adaptive keyframes: `{adaptive['total_keyframe_count']}`, between uniform gap8 `{gap8['total_keyframe_count']}` and uniform gap4 `{gap4['total_keyframe_count']}`.",
        f"- Adaptive metadata: `{adaptive['metadata_bits']}` bits / `{adaptive['metadata_bytes']}` bytes.",
        f"- Sampled hard-row coverage: `{adaptive['promoted_hard_quality_count']}` / `{adaptive['hard_quality_total']}`.",
        f"- Sampled high-payload coverage: `{adaptive['promoted_high_payload_count']}` / `{adaptive['high_payload_total']}`.",
        f"- Sampled residual payload avoided versus treating all sampled targets as middle frames: `{adaptive['avoided_payload_bytes']:.0f}` bytes.",
        "",
        "## Smoke Candidates",
        "",
        "| rank | sequence | reason | score | hard promoted/missed | payload promoted | extra keys | avoided payload |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in smoke_rows:
        lines.append(
            f"| {row['rank']} | {row['sequence']} | {row['reason']} | {row['smoke_score']} | "
            f"{row['stage165_adaptive_promoted_hard_rows']}/{row['stage165_adaptive_false_negative_hard_rows']} | "
            f"{row['stage165_adaptive_promoted_high_payload_rows']} | {row['extra_keyframes_vs_gap8']} | "
            f"{float(row['stage165_adaptive_avoided_payload_bytes']):.0f} |"
        )
    top_false_negative = [row for row in sequence_rows if int(row["stage165_adaptive_false_negative_hard_rows"]) > 0]
    top_false_negative.sort(key=lambda row: int(row["stage165_adaptive_false_negative_hard_rows"]), reverse=True)
    lines.extend([
        "",
        "## Remaining Risk",
        "",
    ])
    if top_false_negative:
        lines.append("Hard-label false negatives remain in:")
        for row in top_false_negative[:10]:
            lines.append(f"- `{row['sequence']}`: `{row['stage165_adaptive_false_negative_hard_rows']}` hard rows not promoted")
    else:
        lines.append("No sampled hard-label false negatives remain under the adaptive keyframe proxy.")
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        "- Run a small rendered smoke before claiming final RD, because inserted keyframes change subsequent interpolation intervals and this proxy does not rerender them.",
        "- Decoder-side contract is unchanged: transmit the adaptive schedule metadata; do not require RGB/motion feature extraction at the decoder.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage163_rows", type=Path, default=DEFAULT_STAGE163_ROWS)
    parser.add_argument("--stage165_selected_rows", type=Path, default=DEFAULT_STAGE165_SELECTED_ROWS)
    parser.add_argument("--stage165_schedule_rows", type=Path, default=DEFAULT_STAGE165_SCHEDULE_ROWS)
    parser.add_argument("--stage165_package", type=Path, default=DEFAULT_STAGE165_PACKAGE)
    parser.add_argument("--stage147_rows", type=Path, default=DEFAULT_STAGE147_ROWS)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--max_smoke_sequences", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage165_package = read_json(args.stage165_package)
    stage163_rows = prepare_stage163_rows(read_csv(args.stage163_rows))
    selected_map = prepare_selected_map(read_csv(args.stage165_selected_rows))
    schedule_rows = read_csv(args.stage165_schedule_rows)
    schedule_maps, sequence_info, metadata_bits = prepare_schedule_maps(schedule_rows)
    anchor_refs = load_stage147_anchor_refs(args.stage147_rows)
    row_consequences = build_row_consequences(stage163_rows, selected_map, schedule_maps)
    keyframe_counts = {schedule: total_keyframes(schedule_maps, schedule) for schedule in SCHEDULES}
    comparison = [
        summarize_schedule(schedule, row_consequences, schedule_maps, sequence_info, metadata_bits, keyframe_counts, anchor_refs)
        for schedule in SCHEDULES
    ]
    sequence_coverage = build_sequence_coverage(row_consequences, sequence_info)
    smoke_candidates = choose_smoke_candidates(sequence_coverage, args.max_smoke_sequences)
    comparison_csv = args.summary_root / "stage166_schedule_comparison.csv"
    rows_csv = args.summary_root / "stage166_sampled_row_consequences.csv"
    sequence_csv = args.summary_root / "stage166_sequence_label_coverage.csv"
    smoke_csv = args.summary_root / "stage166_smoke_candidates.csv"
    package_json = args.summary_root / "stage166_adaptive_schedule_label_rd_comparison_package.json"
    report_md = args.summary_root / "stage166_adaptive_schedule_label_rd_comparison_report.md"
    write_csv(comparison, comparison_csv, SCHEDULE_COMPARISON_FIELDS)
    write_csv(row_consequences, rows_csv, ROW_CONSEQUENCE_FIELDS)
    write_csv(sequence_coverage, sequence_csv, SEQUENCE_COVERAGE_FIELDS)
    write_csv(smoke_candidates, smoke_csv, SMOKE_FIELDS)
    adaptive = next(row for row in comparison if row["schedule"] == "stage165_adaptive")
    decision = "promising_for_small_rendered_smoke" if float(adaptive["hard_quality_coverage"]) >= 0.70 and float(adaptive["high_payload_coverage"]) >= 0.75 else "needs_selector_refinement_before_rendered_smoke"
    package = {
        "stage": 166,
        "status": "adaptive_schedule_label_rd_comparison_packaged",
        "decision": decision,
        "source_stage163_rows": str(args.stage163_rows),
        "source_stage165_package": str(args.stage165_package),
        "source_stage147_rows": str(args.stage147_rows),
        "stage165_selected_policy": stage165_package["selected_policy"],
        "metadata_policy": "uniform schedules count one mode byte; adaptive schedule counts one mode byte plus transmitted keyframe indices per sequence",
        "rate_proxy_note": "main-anchor MiB/frame is interpolated between Stage147 uniform gap8 and gap4 by keyframe count; sampled residual side payload is counted only for the 120 Stage163 rows",
        "schedule_comparison": comparison,
        "smoke_sequences": [row["sequence"] for row in smoke_candidates],
        "comparison_csv": str(comparison_csv),
        "row_consequences_csv": str(rows_csv),
        "sequence_coverage_csv": str(sequence_csv),
        "smoke_candidates_csv": str(smoke_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, comparison, sequence_coverage, smoke_candidates, report_md)
    print(json.dumps({"package": str(package_json), "report": str(report_md), "decision": decision, "adaptive": adaptive, "smoke_sequences": package["smoke_sequences"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
