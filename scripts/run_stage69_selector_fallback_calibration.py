import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_CSV = REPO_ROOT / "experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_comparison.csv"
DEFAULT_SELECTION_CSV = REPO_ROOT / "experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_selections.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage69_selector_fallback_calibration"


DECISION_FIELDS = [
    "sample", "reference_gap", "delta_all_psnr", "rate_mib_per_frame", "selector_cost", "selector_cost_per_keyframe",
    "max_segment_ratio", "mean_abs_len_error_ratio", "segment_std_ratio", "jaccard_to_uniform", "position_mae_to_uniform",
    "exact_uniform", "uniform_indices", "predicted_indices",
]

CHOICE_FIELDS = [
    "policy", "sample", "reference_gap", "chosen_method", "delta_all_psnr", "chosen_reason", "train_policy",
    "indices",
]

SUMMARY_FIELDS = [
    "policy", "count", "accepted_predicted_points", "positive_all_points", "mean_delta_all_psnr",
    "min_delta_all_psnr", "max_delta_all_psnr", "train_policy", "notes",
]

THRESHOLD_FEATURES = [
    "selector_cost",
    "selector_cost_per_keyframe",
    "max_segment_ratio",
    "mean_abs_len_error_ratio",
    "segment_std_ratio",
    "jaccard_to_uniform",
    "position_mae_to_uniform",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_indices(text):
    if text in {None, ""}:
        return []
    return [int(value) for value in str(text).split()]


def parse_segments(text):
    if text in {None, ""}:
        return []
    return [int(value) for value in str(text).split()]


def jaccard(a, b):
    a = set(a)
    b = set(b)
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def position_mae(a, b):
    if not a or not b:
        return 0.0
    count = min(len(a), len(b))
    return float(np.mean([abs(a[i] - b[i]) for i in range(count)]))


def mean(values):
    values = [float(value) for value in values]
    if not values:
        return None
    return float(np.mean(values))


def build_decision_records(comparison_csv, selection_csv):
    comparisons = read_csv(comparison_csv)
    selections = {}
    for row in read_csv(selection_csv):
        selections[(row["sample"], int(row["reference_gap"]), row["method"])] = row
    records = []
    for row in comparisons:
        sample = row["sample"]
        gap = int(row["reference_gap"])
        pred = selections[(sample, gap, "predicted_full_feature_dp")]
        uniform = selections[(sample, gap, "uniform")]
        predicted_indices = parse_indices(row["predicted_indices"])
        uniform_indices = parse_indices(row["uniform_indices"])
        segments = parse_segments(pred["segment_lengths"])
        lengths = np.asarray(segments, dtype=np.float64)
        selector_cost = float(pred["selector_cost"])
        keyframe_count = int(pred["keyframe_count"])
        records.append({
            "sample": sample,
            "reference_gap": gap,
            "delta_all_psnr": float(row["selector_delta_adapter_all_psnr"]),
            "rate_mib_per_frame": float(row["rate_mib_per_frame"]),
            "selector_cost": selector_cost,
            "selector_cost_per_keyframe": selector_cost / max(keyframe_count, 1),
            "max_segment_ratio": float(pred["max_segment_length"]) / max(gap, 1),
            "mean_abs_len_error_ratio": float(np.mean(np.abs(lengths - gap))) / max(gap, 1),
            "segment_std_ratio": float(np.std(lengths)) / max(gap, 1),
            "jaccard_to_uniform": jaccard(predicted_indices, uniform_indices),
            "position_mae_to_uniform": position_mae(predicted_indices, uniform_indices),
            "exact_uniform": "true" if predicted_indices == uniform_indices else "false",
            "uniform_indices": row["uniform_indices"],
            "predicted_indices": row["predicted_indices"],
        })
        del uniform
    return records


def choice(policy, rec, accept, reason, train_policy):
    return {
        "policy": policy,
        "sample": rec["sample"],
        "reference_gap": rec["reference_gap"],
        "chosen_method": "predicted_full_feature_dp" if accept else "uniform",
        "delta_all_psnr": rec["delta_all_psnr"] if accept else 0.0,
        "chosen_reason": reason,
        "train_policy": train_policy,
        "indices": rec["predicted_indices"] if accept else rec["uniform_indices"],
    }


def summarize(policy, choices, train_policy, notes):
    values = [row["delta_all_psnr"] for row in choices]
    return {
        "policy": policy,
        "count": len(choices),
        "accepted_predicted_points": sum(1 for row in choices if row["chosen_method"] != "uniform"),
        "positive_all_points": sum(1 for value in values if value > 0.0),
        "mean_delta_all_psnr": mean(values),
        "min_delta_all_psnr": float(np.min(values)) if values else None,
        "max_delta_all_psnr": float(np.max(values)) if values else None,
        "train_policy": train_policy,
        "notes": notes,
    }


def uniform_policy(records):
    return [choice("uniform", rec, False, "always uniform", "always uniform") for rec in records]


def fixed_predicted_policy(records):
    return [choice("fixed_predicted", rec, True, "always predicted", "always predicted") for rec in records]


def oracle_positive_policy(records):
    return [
        choice("oracle_positive_fallback", rec, rec["delta_all_psnr"] > 0.0, "actual rendered delta sign", "uses actual rendered outcome")
        for rec in records
    ]


def threshold_accept(rec, feature, direction, threshold):
    value = float(rec[feature])
    if direction == "le":
        return value <= threshold
    return value >= threshold


def threshold_choices(policy, records, feature, direction, threshold, train_policy):
    return [
        choice(policy, rec, threshold_accept(rec, feature, direction, threshold), f"{feature} {direction} {threshold:.6g}", train_policy)
        for rec in records
    ]


def score_choices(choices):
    values = [row["delta_all_psnr"] for row in choices]
    accepted = sum(1 for row in choices if row["chosen_method"] != "uniform")
    positives = sum(1 for value in values if value > 0.0)
    min_delta = float(np.min(values)) if values else 0.0
    return (float(np.mean(values)) if values else 0.0, min_delta, positives, -accepted)


def train_threshold(train_records):
    best_spec = ("uniform", "le", 0.0, "always uniform")
    best_choices = uniform_policy(train_records)
    best_score = score_choices(best_choices)
    for feature in THRESHOLD_FEATURES:
        thresholds = sorted({float(rec[feature]) for rec in train_records})
        for threshold in thresholds:
            for direction in ["le", "ge"]:
                train_policy = f"{feature} {direction} {threshold:.6g}"
                choices = threshold_choices("train_threshold", train_records, feature, direction, threshold, train_policy)
                score = score_choices(choices)
                if score > best_score:
                    best_score = score
                    best_spec = (feature, direction, threshold, train_policy)
                    best_choices = choices
    return best_spec, best_score, best_choices


def loocv_threshold_policy(records):
    choices = []
    for heldout in sorted({rec["sample"] for rec in records}):
        train_records = [rec for rec in records if rec["sample"] != heldout]
        eval_records = [rec for rec in records if rec["sample"] == heldout]
        feature, direction, threshold, train_policy = train_threshold(train_records)[0]
        if feature == "uniform":
            choices.extend(choice("loocv_threshold_fallback", rec, False, "trained fallback chose uniform", train_policy) for rec in eval_records)
        else:
            choices.extend(threshold_choices("loocv_threshold_fallback", eval_records, feature, direction, threshold, train_policy))
    return sorted(choices, key=lambda row: (row["sample"], row["reference_gap"]))


def same_data_threshold_policy(records):
    feature, direction, threshold, train_policy = train_threshold(records)[0]
    if feature == "uniform":
        return uniform_policy(records)
    return threshold_choices("same_data_threshold_fallback", records, feature, direction, threshold, train_policy)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison_csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--selection_csv", type=Path, default=DEFAULT_SELECTION_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    records = build_decision_records(args.comparison_csv, args.selection_csv)
    policies = [
        ("uniform", uniform_policy(records), "always uniform", "deployable baseline"),
        ("fixed_predicted", fixed_predicted_policy(records), "always predicted", "deployable but unstable Stage68 selector"),
        ("oracle_positive_fallback", oracle_positive_policy(records), "uses actual rendered outcome", "oracle upper bound; not deployable"),
        ("same_data_threshold_fallback", same_data_threshold_policy(records), "trained and evaluated on Stage68", "analysis upper bound; not deployable"),
        ("loocv_threshold_fallback", loocv_threshold_policy(records), "leave-one-sequence-out threshold", "deployable-style small-sample calibration"),
    ]
    choices = []
    summaries = []
    for policy, policy_choices, train_policy, notes in policies:
        choices.extend(policy_choices)
        summaries.append(summarize(policy, policy_choices, train_policy, notes))

    decision_csv = args.summary_root / "stage69_selector_decision_records.csv"
    choices_csv = args.summary_root / "stage69_selector_policy_choices.csv"
    policy_csv = args.summary_root / "stage69_selector_policy_summary.csv"
    summary_json = args.summary_root / "stage69_selector_fallback_calibration_summary.json"
    write_csv(records, decision_csv, DECISION_FIELDS)
    write_csv(choices, choices_csv, CHOICE_FIELDS)
    write_csv(summaries, policy_csv, SUMMARY_FIELDS)
    best_deployable = max([row for row in summaries if row["policy"] in {"uniform", "fixed_predicted", "loocv_threshold_fallback"}], key=lambda row: (row["mean_delta_all_psnr"], row["min_delta_all_psnr"]))
    summary = {
        "stage": 69,
        "mode": "selector fallback calibration analysis",
        "comparison_csv": str(args.comparison_csv),
        "selection_csv": str(args.selection_csv),
        "decision_csv": str(decision_csv),
        "choices_csv": str(choices_csv),
        "policy_csv": str(policy_csv),
        "policy_summaries": summaries,
        "best_deployable_style_policy": best_deployable,
        "notes": [
            "This stage reuses Stage68 rendered outcomes and does not rerender.",
            "oracle_positive_fallback and same_data_threshold_fallback are analysis upper bounds, not deployable claims.",
            "loocv_threshold_fallback is deployable-style but trained from only the Stage68 rendered outcome set.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_json),
        "policy_csv": str(policy_csv),
        "best_deployable_style_policy": best_deployable,
        "policy_summaries": summaries,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
