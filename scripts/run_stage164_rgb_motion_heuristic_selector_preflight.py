import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE163_ROWS = REPO_ROOT / "experiments/stage163_davis_rgb_motion_selector_data/stage163_davis_rgb_motion_selector_feature_rows.csv"
DEFAULT_STAGE162_PACKAGE = REPO_ROOT / "experiments/stage162_keyframe_selector_protocol_source_audit/stage162_keyframe_selector_protocol_source_audit_package.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage164_rgb_motion_heuristic_selector_preflight"

SWEEP_FIELDS = [
    "candidate", "score_column", "top_fraction", "threshold", "selected_count", "hard_quality_count", "high_payload_count",
    "true_hard_selected", "true_payload_selected", "precision_hard", "recall_hard", "f1_hard", "precision_payload", "recall_payload",
    "mean_selected_psnr", "mean_unselected_psnr", "mean_selected_lpips", "mean_unselected_lpips", "mean_selected_payload", "mean_unselected_payload",
]
SELECTED_FIELDS = [
    "task_id", "sequence", "gap", "left_index", "target_index", "right_index", "normalized_time", "selector_score", "selected_by_heuristic",
    "hard_quality_label", "high_payload_label", "stage158_psnr", "stage158_lpips", "payload_bytes", "rgb_motion_proxy_score",
    "rgb_mad_linear_interp_target", "rgb_mad_left_right", "edge_mad_left_right", "hist_chi_left_right",
]
SEQUENCE_FIELDS = [
    "sequence", "gap", "task_count", "selected_count", "hard_quality_count", "selected_hard_count", "high_payload_count", "selected_high_payload_count",
    "mean_selector_score", "mean_stage158_psnr", "mean_stage158_lpips", "mean_payload_bytes",
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


def percentile_threshold(values, top_fraction):
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    keep = max(1, int(round(len(vals) * float(top_fraction))))
    idx = max(0, len(vals) - keep)
    return vals[idx]


def percentile_ranks(values):
    indexed = sorted((float(v), idx) for idx, v in enumerate(values))
    n = len(indexed)
    ranks = [0.0] * n
    if n <= 1:
        return ranks
    for rank, (_, idx) in enumerate(indexed):
        ranks[idx] = rank / (n - 1)
    return ranks


def prepare_rows(rows):
    numeric_cols = [
        "rgb_motion_proxy_score", "rgb_mad_linear_interp_target", "rgb_mad_left_right", "edge_mad_left_right", "hist_chi_left_right",
        "stage158_psnr", "stage158_lpips", "payload_bytes",
    ]
    out = []
    for row in rows:
        item = dict(row)
        for col in numeric_cols:
            item[col] = float(item[col])
        item["gap"] = int(item["gap"])
        item["hard_quality_label"] = int(float(item["stage158_psnr"]) < 26.0 or float(item["stage158_lpips"]) > 0.22)
        item["high_payload_label"] = int(float(item["payload_bytes"]) > 220000.0)
        out.append(item)
    rank_cols = ["rgb_motion_proxy_score", "rgb_mad_linear_interp_target", "rgb_mad_left_right", "edge_mad_left_right", "hist_chi_left_right"]
    ranks = {col: percentile_ranks([row[col] for row in out]) for col in rank_cols}
    for idx, row in enumerate(out):
        row["combined_percentile_score"] = sum(ranks[col][idx] for col in rank_cols) / len(rank_cols)
    return out


def evaluate_candidate(rows, candidate, score_column, top_fraction):
    threshold = percentile_threshold([row[score_column] for row in rows], top_fraction)
    selected = [row for row in rows if float(row[score_column]) >= threshold]
    unselected = [row for row in rows if float(row[score_column]) < threshold]
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
        "candidate": candidate,
        "score_column": score_column,
        "top_fraction": float(top_fraction),
        "threshold": threshold,
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
    candidates = [
        ("proxy_score", "rgb_motion_proxy_score"),
        ("interp_target_mad", "rgb_mad_linear_interp_target"),
        ("left_right_mad", "rgb_mad_left_right"),
        ("edge_left_right", "edge_mad_left_right"),
        ("combined_percentile", "combined_percentile_score"),
    ]
    top_fractions = [0.2, 0.3, 0.4, 0.5]
    out = []
    for name, col in candidates:
        for frac in top_fractions:
            out.append(evaluate_candidate(rows, name, col, frac))
    out.sort(key=lambda row: (float(row["f1_hard"]), float(row["recall_payload"]), -float(row["top_fraction"])), reverse=True)
    return out


def selected_rows(rows, selected_policy):
    score_col = selected_policy["score_column"]
    threshold = float(selected_policy["threshold"])
    out = []
    for row in rows:
        selected = int(float(row[score_col]) >= threshold)
        out.append({
            "task_id": row["task_id"],
            "sequence": row["sequence"],
            "gap": row["gap"],
            "left_index": row["left_index"],
            "target_index": row["target_index"],
            "right_index": row["right_index"],
            "normalized_time": row["normalized_time"],
            "selector_score": float(row[score_col]),
            "selected_by_heuristic": selected,
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


def summarize_selected(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["sequence"], int(row["gap"]))].append(row)
    out = []
    for (sequence, gap), group in sorted(groups.items()):
        out.append({
            "sequence": sequence,
            "gap": gap,
            "task_count": len(group),
            "selected_count": sum(int(row["selected_by_heuristic"]) for row in group),
            "hard_quality_count": sum(int(row["hard_quality_label"]) for row in group),
            "selected_hard_count": sum(int(row["selected_by_heuristic"]) and int(row["hard_quality_label"]) for row in group),
            "high_payload_count": sum(int(row["high_payload_label"]) for row in group),
            "selected_high_payload_count": sum(int(row["selected_by_heuristic"]) and int(row["high_payload_label"]) for row in group),
            "mean_selector_score": mean(row["selector_score"] for row in group),
            "mean_stage158_psnr": mean(row["stage158_psnr"] for row in group),
            "mean_stage158_lpips": mean(row["stage158_lpips"] for row in group),
            "mean_payload_bytes": mean(row["payload_bytes"] for row in group),
        })
    return out


def write_report(package, sweep_rows, selected, sequence_summary, path):
    best = package["selected_policy"]
    lines = [
        "# Stage164 RGB/Motion Heuristic Selector Preflight",
        "",
        "## Scope",
        "",
        "This stage evaluates row-level hard-segment selection using only Stage163 deployable RGB/motion features.",
        "It does not instantiate full adaptive GOP schedules yet; Stage165 should convert selected hard windows into schedule metadata.",
        "",
        "## Selected Heuristic",
        "",
        f"- Candidate: `{best['candidate']}`",
        f"- Score column: `{best['score_column']}`",
        f"- Top fraction: `{best['top_fraction']}`",
        f"- Threshold: `{best['threshold']}`",
        f"- Precision/recall/F1 for hard quality: `{best['precision_hard']:.6f}` / `{best['recall_hard']:.6f}` / `{best['f1_hard']:.6f}`",
        f"- Payload recall: `{best['recall_payload']:.6f}`",
        "",
        "## Top Sweep Rows",
        "",
        "| candidate | score | top frac | selected | precision hard | recall hard | F1 hard | recall payload | selected PSNR | unselected PSNR | selected payload | unselected payload |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sweep_rows[:10]:
        lines.append(
            f"| {row['candidate']} | {row['score_column']} | {float(row['top_fraction']):.2f} | {row['selected_count']} | "
            f"{float(row['precision_hard']):.6f} | {float(row['recall_hard']):.6f} | {float(row['f1_hard']):.6f} | {float(row['recall_payload']):.6f} | "
            f"{float(row['mean_selected_psnr']):.6f} | {float(row['mean_unselected_psnr']):.6f} | {float(row['mean_selected_payload']):.3f} | {float(row['mean_unselected_payload']):.3f} |"
        )
    lines.extend([
        "",
        "## Sequence/Gap Summary For Selected Heuristic",
        "",
        "| sequence | gap | tasks | selected | hard | selected hard | high payload | selected high payload | mean score | PSNR | LPIPS | payload |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sequence_summary:
        if int(row["selected_count"]) == 0 and int(row["hard_quality_count"]) == 0 and int(row["high_payload_count"]) == 0:
            continue
        lines.append(
            f"| {row['sequence']} | {row['gap']} | {row['task_count']} | {row['selected_count']} | {row['hard_quality_count']} | {row['selected_hard_count']} | "
            f"{row['high_payload_count']} | {row['selected_high_payload_count']} | {float(row['mean_selector_score']):.6f} | {float(row['mean_stage158_psnr']):.6f} | "
            f"{float(row['mean_stage158_lpips']):.6f} | {float(row['mean_payload_bytes']):.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- A selected hard-segment row means a future adaptive scheduler should consider shortening the GOP or placing an extra keyframe near that window.",
        "- False negatives remain important because Stage163 showed some low-PSNR cases are not captured by simple RGB motion alone.",
        "- This heuristic uses no target dense anchors, rendered metrics, or labels at inference; labels are used only for this preflight evaluation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage163_rows", type=Path, default=DEFAULT_STAGE163_ROWS)
    parser.add_argument("--stage162_package", type=Path, default=DEFAULT_STAGE162_PACKAGE)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage162 = read_json(args.stage162_package)
    rows = prepare_rows(read_csv(args.stage163_rows))
    sweep_rows = sweep(rows)
    best = sweep_rows[0]
    selected = selected_rows(rows, best)
    sequence_summary = summarize_selected(selected)
    sweep_csv = args.summary_root / "stage164_rgb_motion_heuristic_sweep.csv"
    selected_csv = args.summary_root / "stage164_rgb_motion_heuristic_selected_rows.csv"
    sequence_csv = args.summary_root / "stage164_rgb_motion_heuristic_sequence_summary.csv"
    package_json = args.summary_root / "stage164_rgb_motion_heuristic_selector_preflight_package.json"
    report_md = args.summary_root / "stage164_rgb_motion_heuristic_selector_preflight_report.md"
    write_csv(sweep_rows, sweep_csv, SWEEP_FIELDS)
    write_csv(selected, selected_csv, SELECTED_FIELDS)
    write_csv(sequence_summary, sequence_csv, SEQUENCE_FIELDS)
    package = {
        "stage": 164,
        "status": "rgb_motion_heuristic_selector_preflight_packaged",
        "source_stage163_rows": str(args.stage163_rows),
        "source_stage162_protocol": str(args.stage162_package),
        "fixed_middle_recovery_policy": stage162["fixed_middle_recovery_policy"],
        "row_count": len(rows),
        "hard_quality_count": sum(row["hard_quality_label"] for row in rows),
        "high_payload_count": sum(row["high_payload_label"] for row in rows),
        "selected_policy": best,
        "inference_feature_scope": "RGB/motion proxy features from input frames only",
        "schedule_status": "not_instantiated_row_level_preflight_only",
        "sweep_csv": str(sweep_csv),
        "selected_csv": str(selected_csv),
        "sequence_csv": str(sequence_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, sweep_rows, selected, sequence_summary, report_md)
    print(json.dumps({"package": str(package_json), "report": str(report_md), "selected_policy": best}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
