import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE163_ROWS = REPO_ROOT / "experiments/stage163_davis_rgb_motion_selector_data/stage163_davis_rgb_motion_selector_feature_rows.csv"
DEFAULT_TASK_MANIFEST = REPO_ROOT / "experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_tasks.csv"
DEFAULT_STAGE162_PACKAGE = REPO_ROOT / "experiments/stage162_keyframe_selector_protocol_source_audit/stage162_keyframe_selector_protocol_source_audit_package.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage165_multifeature_keyframe_schedule_preflight"

FEATURE_COLUMNS = [
    "rgb_motion_proxy_score",
    "rgb_mad_linear_interp_target",
    "rgb_mad_left_right",
    "edge_mad_left_right",
    "hist_chi_left_right",
]

SWEEP_FIELDS = [
    "candidate", "rank_threshold", "min_votes", "selected_count", "hard_quality_count", "high_payload_count",
    "true_hard_selected", "true_payload_selected", "precision_hard", "recall_hard", "f1_hard", "precision_payload", "recall_payload",
    "mean_selected_psnr", "mean_unselected_psnr", "mean_selected_lpips", "mean_unselected_lpips", "mean_selected_payload", "mean_unselected_payload",
]
ROW_FIELDS = [
    "task_id", "sequence", "gap", "left_index", "target_index", "right_index", "normalized_time", "vote_count", "selector_score", "selected_for_extra_keyframe",
    "hard_quality_label", "high_payload_label", "stage158_psnr", "stage158_lpips", "payload_bytes",
    "rgb_motion_proxy_score", "rgb_mad_linear_interp_target", "rgb_mad_left_right", "edge_mad_left_right", "hist_chi_left_right",
]
SCHEDULE_FIELDS = [
    "sequence", "total_frames", "base_gap", "uniform_gap8_keyframes", "uniform_gap4_keyframes", "adaptive_keyframes",
    "uniform_gap8_keyframe_count", "uniform_gap4_keyframe_count", "adaptive_keyframe_count", "extra_keyframes_vs_gap8",
    "uniform_gap8_keyframe_ratio", "uniform_gap4_keyframe_ratio", "adaptive_keyframe_ratio",
    "metadata_bits", "metadata_bytes", "metadata_mib", "selected_task_count", "hard_task_count", "selected_hard_count",
    "high_payload_task_count", "selected_high_payload_count", "mean_selected_payload_bytes", "mean_selected_psnr", "mean_selected_lpips",
]
SUMMARY_FIELDS = [
    "schedule", "sequence_count", "total_frame_count", "total_keyframe_count", "mean_keyframe_ratio", "metadata_bits", "metadata_bytes",
    "metadata_mib", "selected_task_count", "hard_task_count", "selected_hard_count", "high_payload_task_count", "selected_high_payload_count",
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


def parse_tasks(path):
    by_id = {}
    by_sequence = {}
    for row in read_csv(path):
        by_id[row["task_id"]] = row
        if row["task_split"] == "eval" and row["codec"] == "q12":
            by_sequence[row["sequence"]] = int(row["total_frames"])
    return by_id, by_sequence


def percentile_ranks(rows, col):
    indexed = sorted((float(row[col]), idx) for idx, row in enumerate(rows))
    n = len(indexed)
    ranks = [0.0] * n
    if n <= 1:
        return ranks
    for rank, (_, idx) in enumerate(indexed):
        ranks[idx] = rank / (n - 1)
    return ranks


def prepare_rows(raw_rows):
    rows = []
    for row in raw_rows:
        item = dict(row)
        for col in FEATURE_COLUMNS:
            item[col] = float(item[col])
        item["gap"] = int(item["gap"])
        item["left_index"] = int(item["left_index"])
        item["target_index"] = int(item["target_index"])
        item["right_index"] = int(item["right_index"])
        item["normalized_time"] = float(item["normalized_time"])
        item["stage158_psnr"] = float(item["stage158_psnr"])
        item["stage158_lpips"] = float(item["stage158_lpips"])
        item["payload_bytes"] = float(item["payload_bytes"])
        item["hard_quality_label"] = int(float(item["stage158_psnr"]) < 26.0 or float(item["stage158_lpips"]) > 0.22)
        item["high_payload_label"] = int(float(item["payload_bytes"]) > 220000.0)
        rows.append(item)
    rank_by_col = {col: percentile_ranks(rows, col) for col in FEATURE_COLUMNS}
    for idx, row in enumerate(rows):
        for col in FEATURE_COLUMNS:
            row[f"{col}_rank"] = rank_by_col[col][idx]
    return rows


def evaluate_gate(rows, rank_threshold, min_votes):
    selected = []
    unselected = []
    for row in rows:
        votes = sum(int(float(row[f"{col}_rank"]) >= float(rank_threshold)) for col in FEATURE_COLUMNS)
        row_selected = votes >= int(min_votes)
        item = dict(row)
        item["vote_count"] = votes
        item["selector_score"] = votes + mean(row[f"{col}_rank"] for col in FEATURE_COLUMNS)
        if row_selected:
            selected.append(item)
        else:
            unselected.append(item)
    hard_total = sum(row["hard_quality_label"] for row in rows)
    payload_total = sum(row["high_payload_label"] for row in rows)
    hard_selected = sum(row["hard_quality_label"] for row in selected)
    payload_selected = sum(row["high_payload_label"] for row in selected)
    precision_hard = hard_selected / len(selected) if selected else 0.0
    recall_hard = hard_selected / hard_total if hard_total else 0.0
    f1_hard = 2.0 * precision_hard * recall_hard / (precision_hard + recall_hard) if precision_hard + recall_hard > 0.0 else 0.0
    precision_payload = payload_selected / len(selected) if selected else 0.0
    recall_payload = payload_selected / payload_total if payload_total else 0.0
    return {
        "candidate": f"rank_gate_t{rank_threshold}_votes{min_votes}",
        "rank_threshold": float(rank_threshold),
        "min_votes": int(min_votes),
        "selected_count": len(selected),
        "hard_quality_count": hard_total,
        "high_payload_count": payload_total,
        "true_hard_selected": hard_selected,
        "true_payload_selected": payload_selected,
        "precision_hard": precision_hard,
        "recall_hard": recall_hard,
        "f1_hard": f1_hard,
        "precision_payload": precision_payload,
        "recall_payload": recall_payload,
        "mean_selected_psnr": mean(row["stage158_psnr"] for row in selected),
        "mean_unselected_psnr": mean(row["stage158_psnr"] for row in unselected),
        "mean_selected_lpips": mean(row["stage158_lpips"] for row in selected),
        "mean_unselected_lpips": mean(row["stage158_lpips"] for row in unselected),
        "mean_selected_payload": mean(row["payload_bytes"] for row in selected),
        "mean_unselected_payload": mean(row["payload_bytes"] for row in unselected),
    }


def sweep(rows):
    out = []
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        for votes in [1, 2, 3, 4]:
            result = evaluate_gate(rows, threshold, votes)
            # Avoid degenerate near-all selectors unless they clearly dominate.
            out.append(result)
    out.sort(key=lambda row: (float(row["f1_hard"]), float(row["recall_payload"]), -abs(float(row["selected_count"]) - 48)), reverse=True)
    return out


def apply_policy(rows, policy):
    out = []
    threshold = float(policy["rank_threshold"])
    min_votes = int(policy["min_votes"])
    for row in rows:
        votes = sum(int(float(row[f"{col}_rank"]) >= threshold) for col in FEATURE_COLUMNS)
        score = votes + mean(row[f"{col}_rank"] for col in FEATURE_COLUMNS)
        selected = int(votes >= min_votes)
        out.append({
            "task_id": row["task_id"],
            "sequence": row["sequence"],
            "gap": row["gap"],
            "left_index": row["left_index"],
            "target_index": row["target_index"],
            "right_index": row["right_index"],
            "normalized_time": row["normalized_time"],
            "vote_count": votes,
            "selector_score": score,
            "selected_for_extra_keyframe": selected,
            "hard_quality_label": row["hard_quality_label"],
            "high_payload_label": row["high_payload_label"],
            "stage158_psnr": row["stage158_psnr"],
            "stage158_lpips": row["stage158_lpips"],
            "payload_bytes": row["payload_bytes"],
            "rgb_motion_proxy_score": row["rgb_motion_proxy_score"],
            "rgb_mad_linear_interp_target": row["rgb_mad_linear_interp_target"],
            "rgb_mad_left_right": row["rgb_mad_left_right"],
            "edge_mad_left_right": row["edge_mad_left_right"],
            "hist_chi_left_right": row["hist_chi_left_right"],
        })
    return out


def uniform_indices(total_frames, gap):
    indices = list(range(0, int(total_frames), int(gap)))
    last = int(total_frames) - 1
    if indices[-1] != last:
        indices.append(last)
    return sorted(set(indices))


def schedule_metadata_bits(total_frames, keyframes, mode_bytes):
    bits_per_index = math.ceil(math.log2(max(int(total_frames), 2)))
    return int(mode_bytes) * 8 + len(keyframes) * bits_per_index


def build_schedules(selected_rows, total_frames_by_sequence, mode_bytes):
    groups = defaultdict(list)
    for row in selected_rows:
        groups[row["sequence"]].append(row)
    schedules = []
    for sequence, group in sorted(groups.items()):
        total_frames = int(total_frames_by_sequence[sequence])
        gap8 = uniform_indices(total_frames, 8)
        gap4 = uniform_indices(total_frames, 4)
        extra_targets = sorted({int(row["target_index"]) for row in group if int(row["selected_for_extra_keyframe"])})
        adaptive = sorted(set(gap8 + extra_targets))
        metadata_bits = schedule_metadata_bits(total_frames, adaptive, mode_bytes)
        selected_group = [row for row in group if int(row["selected_for_extra_keyframe"])]
        schedules.append({
            "sequence": sequence,
            "total_frames": total_frames,
            "base_gap": 8,
            "uniform_gap8_keyframes": " ".join(str(x) for x in gap8),
            "uniform_gap4_keyframes": " ".join(str(x) for x in gap4),
            "adaptive_keyframes": " ".join(str(x) for x in adaptive),
            "uniform_gap8_keyframe_count": len(gap8),
            "uniform_gap4_keyframe_count": len(gap4),
            "adaptive_keyframe_count": len(adaptive),
            "extra_keyframes_vs_gap8": len(adaptive) - len(gap8),
            "uniform_gap8_keyframe_ratio": len(gap8) / total_frames,
            "uniform_gap4_keyframe_ratio": len(gap4) / total_frames,
            "adaptive_keyframe_ratio": len(adaptive) / total_frames,
            "metadata_bits": metadata_bits,
            "metadata_bytes": math.ceil(metadata_bits / 8.0),
            "metadata_mib": metadata_bits / 8.0 / (1024.0 * 1024.0),
            "selected_task_count": len(selected_group),
            "hard_task_count": sum(int(row["hard_quality_label"]) for row in group),
            "selected_hard_count": sum(int(row["selected_for_extra_keyframe"]) and int(row["hard_quality_label"]) for row in group),
            "high_payload_task_count": sum(int(row["high_payload_label"]) for row in group),
            "selected_high_payload_count": sum(int(row["selected_for_extra_keyframe"]) and int(row["high_payload_label"]) for row in group),
            "mean_selected_payload_bytes": mean(row["payload_bytes"] for row in selected_group),
            "mean_selected_psnr": mean(row["stage158_psnr"] for row in selected_group),
            "mean_selected_lpips": mean(row["stage158_lpips"] for row in selected_group),
        })
    return schedules


def schedule_summary(schedules):
    total_frames = sum(int(row["total_frames"]) for row in schedules)
    total_keyframes = sum(int(row["adaptive_keyframe_count"]) for row in schedules)
    total_bits = sum(int(row["metadata_bits"]) for row in schedules)
    return [
        {
            "schedule": "rgb_motion_rank_gate_gap8_plus_extra_targets_v1",
            "sequence_count": len(schedules),
            "total_frame_count": total_frames,
            "total_keyframe_count": total_keyframes,
            "mean_keyframe_ratio": total_keyframes / total_frames if total_frames else 0.0,
            "metadata_bits": total_bits,
            "metadata_bytes": math.ceil(total_bits / 8.0),
            "metadata_mib": total_bits / 8.0 / (1024.0 * 1024.0),
            "selected_task_count": sum(int(row["selected_task_count"]) for row in schedules),
            "hard_task_count": sum(int(row["hard_task_count"]) for row in schedules),
            "selected_hard_count": sum(int(row["selected_hard_count"]) for row in schedules),
            "high_payload_task_count": sum(int(row["high_payload_task_count"]) for row in schedules),
            "selected_high_payload_count": sum(int(row["selected_high_payload_count"]) for row in schedules),
        }
    ]


def write_report(package, sweep_rows, schedules, summary, path):
    best = package["selected_policy"]
    sched = summary[0]
    lines = [
        "# Stage165 Multi-Feature Keyframe Schedule Preflight",
        "",
        "## Scope",
        "",
        "This is a metadata/label preflight. It converts row-level RGB/motion hard-window signals into adaptive keyframe schedules but does not run heavy rendered RD yet.",
        "",
        "## Selected Gate",
        "",
        f"- Rank threshold: `{best['rank_threshold']}`",
        f"- Minimum feature votes: `{best['min_votes']}`",
        f"- Selected rows: `{best['selected_count']}` / `{package['row_count']}`",
        f"- Hard-quality precision/recall/F1: `{best['precision_hard']:.6f}` / `{best['recall_hard']:.6f}` / `{best['f1_hard']:.6f}`",
        f"- Payload precision/recall: `{best['precision_payload']:.6f}` / `{best['recall_payload']:.6f}`",
        "",
        "## Schedule Summary",
        "",
        f"- Schedule: `{sched['schedule']}`",
        f"- Sequences: `{sched['sequence_count']}`",
        f"- Total frames: `{sched['total_frame_count']}`",
        f"- Total keyframes: `{sched['total_keyframe_count']}`",
        f"- Mean keyframe ratio: `{sched['mean_keyframe_ratio']:.6f}`",
        f"- Metadata: `{sched['metadata_bits']}` bits / `{sched['metadata_bytes']}` bytes / `{sched['metadata_mib']:.9f}` MiB",
        f"- Selected hard-quality rows: `{sched['selected_hard_count']}` / `{sched['hard_task_count']}`",
        f"- Selected high-payload rows: `{sched['selected_high_payload_count']}` / `{sched['high_payload_task_count']}`",
        "",
        "## Top Gate Sweep Rows",
        "",
        "| candidate | threshold | votes | selected | precision hard | recall hard | F1 hard | recall payload | selected payload | unselected payload |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sweep_rows[:10]:
        lines.append(
            f"| {row['candidate']} | {float(row['rank_threshold']):.2f} | {row['min_votes']} | {row['selected_count']} | "
            f"{float(row['precision_hard']):.6f} | {float(row['recall_hard']):.6f} | {float(row['f1_hard']):.6f} | {float(row['recall_payload']):.6f} | "
            f"{float(row['mean_selected_payload']):.3f} | {float(row['mean_unselected_payload']):.3f} |"
        )
    lines.extend([
        "",
        "## Schedule Rows",
        "",
        "| sequence | frames | gap8 keys | gap4 keys | adaptive keys | extra vs gap8 | metadata bytes | selected/hard | selected/high payload |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in schedules:
        lines.append(
            f"| {row['sequence']} | {row['total_frames']} | {row['uniform_gap8_keyframe_count']} | {row['uniform_gap4_keyframe_count']} | "
            f"{row['adaptive_keyframe_count']} | {row['extra_keyframes_vs_gap8']} | {row['metadata_bytes']} | "
            f"{row['selected_hard_count']}/{row['hard_task_count']} | {row['selected_high_payload_count']}/{row['high_payload_task_count']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The candidate starts from uniform gap8 and adds selected target frames as extra keyframes.",
        "- Keyframe indices are transmitted and counted as metadata; decoder does not reproduce RGB/motion feature extraction.",
        "- This remains a pre-render schedule candidate. Stage166 should evaluate rendered/label RD for the proposed schedules and compare against uniform gap4/gap8.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage163_rows", type=Path, default=DEFAULT_STAGE163_ROWS)
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--stage162_package", type=Path, default=DEFAULT_STAGE162_PACKAGE)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--mode_bytes", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage162 = read_json(args.stage162_package)
    _, total_frames_by_sequence = parse_tasks(args.task_manifest)
    rows = prepare_rows(read_csv(args.stage163_rows))
    sweep_rows = sweep(rows)
    selected_policy = sweep_rows[0]
    selected_rows = apply_policy(rows, selected_policy)
    schedules = build_schedules(selected_rows, total_frames_by_sequence, args.mode_bytes)
    summary = schedule_summary(schedules)
    sweep_csv = args.summary_root / "stage165_multifeature_gate_sweep.csv"
    selected_csv = args.summary_root / "stage165_multifeature_selected_rows.csv"
    schedules_csv = args.summary_root / "stage165_adaptive_schedule_rows.csv"
    summary_csv = args.summary_root / "stage165_adaptive_schedule_summary.csv"
    package_json = args.summary_root / "stage165_multifeature_keyframe_schedule_preflight_package.json"
    report_md = args.summary_root / "stage165_multifeature_keyframe_schedule_preflight_report.md"
    write_csv(sweep_rows, sweep_csv, SWEEP_FIELDS)
    write_csv(selected_rows, selected_csv, ROW_FIELDS)
    write_csv(schedules, schedules_csv, SCHEDULE_FIELDS)
    write_csv(summary, summary_csv, SUMMARY_FIELDS)
    package = {
        "stage": 165,
        "status": "multifeature_keyframe_schedule_preflight_packaged",
        "source_stage163_rows": str(args.stage163_rows),
        "source_stage162_protocol": str(args.stage162_package),
        "fixed_middle_recovery_policy": stage162["fixed_middle_recovery_policy"],
        "row_count": len(rows),
        "selected_policy": selected_policy,
        "schedule_summary": summary[0],
        "mode_bytes": int(args.mode_bytes),
        "schedule_logic": "start uniform gap8, add selected target indices as extra keyframes",
        "inference_feature_scope": "Stage163 RGB/motion feature ranks only",
        "rendered_rd_status": "not_run_preflight_only",
        "sweep_csv": str(sweep_csv),
        "selected_csv": str(selected_csv),
        "schedules_csv": str(schedules_csv),
        "summary_csv": str(summary_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, sweep_rows, schedules, summary, report_md)
    print(json.dumps({"package": str(package_json), "report": str(report_md), "selected_policy": selected_policy, "schedule_summary": summary[0]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
