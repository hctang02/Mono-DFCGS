import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_CSV = REPO_ROOT / "experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_comparison.csv"
DEFAULT_DECISION_CSV = REPO_ROOT / "experiments/stage69_selector_fallback_calibration/stage69_selector_decision_records.csv"
DEFAULT_POLICY_CSV = REPO_ROOT / "experiments/stage69_selector_fallback_calibration/stage69_selector_policy_summary.csv"
DEFAULT_CHOICES_CSV = REPO_ROOT / "experiments/stage69_selector_fallback_calibration/stage69_selector_policy_choices.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage84_rendered_selector_label_package"


LABEL_FIELDS = [
    "sample",
    "dataset",
    "split",
    "sequence",
    "reference_gap",
    "rate_mib_per_frame",
    "uniform_adapter_all_psnr",
    "predicted_adapter_all_psnr",
    "delta_all_psnr",
    "rendered_accept_predicted",
    "oracle_choice",
    "uniform_linear_all_psnr",
    "predicted_linear_all_psnr",
    "delta_linear_all_psnr",
    "selector_cost",
    "selector_cost_per_keyframe",
    "max_segment_ratio",
    "mean_abs_len_error_ratio",
    "segment_std_ratio",
    "jaccard_to_uniform",
    "position_mae_to_uniform",
    "exact_uniform",
    "uniform_indices",
    "predicted_indices",
    "label_source",
    "deployable_at_test_time",
]

GAP_FIELDS = [
    "reference_gap",
    "count",
    "positive_count",
    "mean_delta_all_psnr",
    "min_delta_all_psnr",
    "max_delta_all_psnr",
    "mean_rate_mib_per_frame",
]

POLICY_FIELDS = [
    "policy",
    "count",
    "accepted_predicted_points",
    "positive_all_points",
    "mean_delta_all_psnr",
    "min_delta_all_psnr",
    "max_delta_all_psnr",
    "deployable_category",
    "train_policy",
    "notes",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_sample(sample):
    parts = sample.split("/")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return "", "", sample


def f(row, key):
    value = row.get(key, "")
    return float(value) if value not in {"", None} else 0.0


def mean(values):
    values = [float(value) for value in values]
    if not values:
        return None
    return sum(values) / len(values)


def build_labels(comparison_rows, decision_rows):
    decisions = {(row["sample"], int(row["reference_gap"])): row for row in decision_rows}
    labels = []
    for row in comparison_rows:
        sample = row["sample"]
        gap = int(row["reference_gap"])
        decision = decisions[(sample, gap)]
        dataset, split, sequence = parse_sample(sample)
        delta = f(row, "selector_delta_adapter_all_psnr")
        labels.append({
            "sample": sample,
            "dataset": dataset,
            "split": split,
            "sequence": sequence,
            "reference_gap": gap,
            "rate_mib_per_frame": f(row, "rate_mib_per_frame"),
            "uniform_adapter_all_psnr": f(row, "uniform_adapter_all_psnr"),
            "predicted_adapter_all_psnr": f(row, "predicted_adapter_all_psnr"),
            "delta_all_psnr": delta,
            "rendered_accept_predicted": "true" if delta > 0.0 else "false",
            "oracle_choice": "predicted_full_feature_dp" if delta > 0.0 else "uniform",
            "uniform_linear_all_psnr": f(row, "uniform_linear_all_psnr"),
            "predicted_linear_all_psnr": f(row, "predicted_linear_all_psnr"),
            "delta_linear_all_psnr": f(row, "selector_delta_linear_all_psnr"),
            "selector_cost": f(decision, "selector_cost"),
            "selector_cost_per_keyframe": f(decision, "selector_cost_per_keyframe"),
            "max_segment_ratio": f(decision, "max_segment_ratio"),
            "mean_abs_len_error_ratio": f(decision, "mean_abs_len_error_ratio"),
            "segment_std_ratio": f(decision, "segment_std_ratio"),
            "jaccard_to_uniform": f(decision, "jaccard_to_uniform"),
            "position_mae_to_uniform": f(decision, "position_mae_to_uniform"),
            "exact_uniform": decision["exact_uniform"],
            "uniform_indices": row["uniform_indices"],
            "predicted_indices": row["predicted_indices"],
            "label_source": "Stage68 rendered predicted-vs-uniform adapter all-frame PSNR delta",
            "deployable_at_test_time": "false",
        })
    return sorted(labels, key=lambda item: (item["sample"], int(item["reference_gap"])))


def gap_summary(labels):
    grouped = defaultdict(list)
    for row in labels:
        grouped[int(row["reference_gap"])].append(row)
    out = []
    for gap, rows in sorted(grouped.items()):
        deltas = [row["delta_all_psnr"] for row in rows]
        rates = [row["rate_mib_per_frame"] for row in rows]
        out.append({
            "reference_gap": gap,
            "count": len(rows),
            "positive_count": sum(1 for value in deltas if value > 0.0),
            "mean_delta_all_psnr": mean(deltas),
            "min_delta_all_psnr": min(deltas) if deltas else None,
            "max_delta_all_psnr": max(deltas) if deltas else None,
            "mean_rate_mib_per_frame": mean(rates),
        })
    return out


def policy_category(policy):
    if policy == "uniform":
        return "safe_deployable_baseline"
    if policy == "fixed_predicted":
        return "deployable_candidate_unstable"
    if policy == "loocv_threshold_fallback":
        return "deployable_style_small_sample"
    return "analysis_oracle_not_deployable"


def policy_rows(rows):
    out = []
    for row in rows:
        out.append({
            **row,
            "deployable_category": policy_category(row["policy"]),
        })
    return out


def best_policy(rows, allowed_categories):
    candidates = [row for row in rows if row["deployable_category"] in allowed_categories]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (f(row, "mean_delta_all_psnr"), f(row, "min_delta_all_psnr")))


def best_safe_policy(rows, allowed_categories):
    candidates = [row for row in rows if row["deployable_category"] in allowed_categories and f(row, "min_delta_all_psnr") >= 0.0]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (f(row, "mean_delta_all_psnr"), f(row, "accepted_predicted_points")))


def write_report(summary, gap_rows, policy_summary_rows, path):
    lines = [
        "# Stage84 Rendered Selector Label Package",
        "",
        "## Rendered Labels",
        "",
        f"- label count: `{summary['label_count']}`",
        f"- positive predicted selections: `{summary['positive_label_count']}`",
        f"- mean rendered delta: `{summary['mean_delta_all_psnr']}`",
        f"- minimum rendered delta: `{summary['min_delta_all_psnr']}`",
        "",
        "| gap | count | positives | mean delta | min delta | max delta | mean rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gap_rows:
        lines.append(
            f"| {row['reference_gap']} | {row['count']} | {row['positive_count']} | {row['mean_delta_all_psnr']} | {row['min_delta_all_psnr']} | {row['max_delta_all_psnr']} | {row['mean_rate_mib_per_frame']} |"
        )
    lines.extend([
        "",
        "## Policy Guardrails",
        "",
        "| policy | category | accepted | mean delta | min delta | notes |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in policy_summary_rows:
        lines.append(
            f"| {row['policy']} | {row['deployable_category']} | {row['accepted_predicted_points']} | {row['mean_delta_all_psnr']} | {row['min_delta_all_psnr']} | {row['notes']} |"
        )
    lines.extend([
        "",
        "## Conclusion",
        "",
        f"- best deployable policy: `{summary['best_deployable_policy']['policy']}`",
        f"- best deployable mean delta: `{summary['best_deployable_policy']['mean_delta_all_psnr']}`",
        f"- best safe deployable policy: `{summary['best_safe_deployable_policy']['policy']}`",
        f"- best safe deployable min delta: `{summary['best_safe_deployable_policy']['min_delta_all_psnr']}`",
        "- Rendered labels are offline supervision only; a final selector must use feed-forward features plus deterministic DP at test time.",
        "- Current rendered label set is small, so oracle-positive and same-data threshold rows are analysis upper bounds, not deployable claims.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison_csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--decision_csv", type=Path, default=DEFAULT_DECISION_CSV)
    parser.add_argument("--policy_csv", type=Path, default=DEFAULT_POLICY_CSV)
    parser.add_argument("--choices_csv", type=Path, default=DEFAULT_CHOICES_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    labels = build_labels(read_csv(args.comparison_csv), read_csv(args.decision_csv))
    gap_rows = gap_summary(labels)
    policy_summary_rows = policy_rows(read_csv(args.policy_csv))
    deployable_categories = {"safe_deployable_baseline", "deployable_candidate_unstable", "deployable_style_small_sample"}
    deployable = best_policy(policy_summary_rows, deployable_categories)
    safe_deployable = best_safe_policy(policy_summary_rows, deployable_categories)
    analysis_best = best_policy(policy_summary_rows, deployable_categories | {"analysis_oracle_not_deployable"})
    deltas = [row["delta_all_psnr"] for row in labels]

    label_csv = args.summary_root / "stage84_rendered_selector_labels.csv"
    gap_csv = args.summary_root / "stage84_rendered_selector_gap_summary.csv"
    policy_csv = args.summary_root / "stage84_selector_policy_guardrails.csv"
    choices_copy_csv = args.summary_root / "stage84_selector_policy_choices.csv"
    summary_json = args.summary_root / "stage84_rendered_selector_label_package_summary.json"
    report_md = args.summary_root / "stage84_rendered_selector_label_package_report.md"

    write_csv(labels, label_csv, LABEL_FIELDS)
    write_csv(gap_rows, gap_csv, GAP_FIELDS)
    write_csv(policy_summary_rows, policy_csv, POLICY_FIELDS)
    write_csv(read_csv(args.choices_csv), choices_copy_csv, list(read_csv(args.choices_csv)[0].keys()))

    summary = {
        "stage": 84,
        "mode": "rendered selector label package",
        "comparison_csv": str(args.comparison_csv),
        "decision_csv": str(args.decision_csv),
        "policy_csv": str(args.policy_csv),
        "choices_csv": str(args.choices_csv),
        "label_csv": str(label_csv),
        "gap_summary_csv": str(gap_csv),
        "policy_guardrails_csv": str(policy_csv),
        "policy_choices_csv": str(choices_copy_csv),
        "report_md": str(report_md),
        "label_count": len(labels),
        "positive_label_count": sum(1 for value in deltas if value > 0.0),
        "mean_delta_all_psnr": mean(deltas),
        "min_delta_all_psnr": min(deltas) if deltas else None,
        "max_delta_all_psnr": max(deltas) if deltas else None,
        "gap_summary": gap_rows,
        "policy_guardrails": policy_summary_rows,
        "best_deployable_policy": deployable,
        "best_safe_deployable_policy": safe_deployable,
        "best_analysis_policy": analysis_best,
        "notes": [
            "Rendered labels are offline labels and are not available at test time.",
            "A deployable selector must use feed-forward predicted costs and deterministic DP only.",
            "Stage68/69 rendered label count is small, so policy results are guardrails rather than final selector claims.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, gap_rows, policy_summary_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "label_count": summary["label_count"],
        "positive_label_count": summary["positive_label_count"],
        "best_deployable_policy": deployable,
        "best_safe_deployable_policy": safe_deployable,
        "best_analysis_policy": analysis_best,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
