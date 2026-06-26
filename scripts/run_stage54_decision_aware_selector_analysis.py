import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE48_CSV = REPO_ROOT / "experiments/stage48_predicted_adaptive_selector_rd/stage48_predicted_adaptive_selector_rd.csv"
DEFAULT_STAGE48_COMPARISON_CSV = REPO_ROOT / "experiments/stage48_predicted_adaptive_selector_rd/stage48_selector_comparison.csv"
DEFAULT_STAGE49_CSV = REPO_ROOT / "experiments/stage49_extended_adaptive_rd/stage49_extended_adaptive_rd.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage54_decision_aware_selector_analysis"


CANDIDATE_METHODS = [
    "length_raw_log_prior_0p1",
    "full_raw_log_prior_0p1",
    "length_sample_z_rank_prior_0p1",
    "full_sample_z_rank",
    "full_sample_z_rank_prior_0p1",
    "full_sample_z_rank_prior_0p3",
]


DECISION_FIELDS = [
    "sample",
    "reference_gap",
    "method",
    "delta_all_psnr",
    "delta_middle_psnr",
    "candidate_all_psnr",
    "uniform_all_psnr",
    "oracle_all_psnr",
    "oracle_delta_all_psnr",
    "keyframe_count",
    "exact_uniform",
    "exact_oracle",
    "jaccard_to_uniform",
    "jaccard_to_oracle",
    "position_mae_to_uniform",
    "position_mae_to_oracle",
    "max_segment_ratio",
    "mean_abs_len_error_ratio",
    "segment_std_ratio",
    "selector_cost",
    "candidate_indices",
    "uniform_indices",
    "oracle_indices",
]


CHOICE_FIELDS = [
    "policy",
    "sample",
    "reference_gap",
    "chosen_method",
    "delta_all_psnr",
    "delta_middle_psnr",
    "chosen_reason",
    "train_policy",
    "jaccard_to_oracle",
    "max_segment_ratio",
    "mean_abs_len_error_ratio",
    "indices",
]


POLICY_FIELDS = [
    "policy",
    "count",
    "accepted_adaptive_points",
    "exact_uniform_points",
    "positive_all_points",
    "positive_middle_points",
    "mean_delta_all_psnr",
    "mean_delta_middle_psnr",
    "min_delta_all_psnr",
    "min_delta_middle_psnr",
    "notes",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, fields, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
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


def maybe_float(row, key):
    value = row.get(key)
    if value in {None, ""}:
        return None
    return float(value)


def maybe_int(row, key):
    value = row.get(key)
    if value in {None, ""}:
        return None
    return int(float(value))


def bool_text(value):
    return "true" if value else "false"


def mean(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def jaccard(a, b):
    a = set(a)
    b = set(b)
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def position_mae(a, b):
    if not a or not b:
        return None
    count = min(len(a), len(b))
    return float(np.mean([abs(a[i] - b[i]) for i in range(count)]))


def load_stage_rows(stage48_csv, stage49_csv):
    stage48 = {}
    for row in read_csv(stage48_csv):
        key = (row["sample"], int(row["reference_gap"]), row["method"])
        stage48[key] = row
    stage49 = {}
    for row in read_csv(stage49_csv):
        if int(row["reference_gap"]) not in {4, 8, 16}:
            continue
        key = (row["sample"], int(row["reference_gap"]), row["method"])
        stage49[key] = row
    return stage48, stage49


def load_comparisons(path):
    out = {}
    for row in read_csv(path):
        key = (row["sample"], int(row["reference_gap"]), row["method"])
        out[key] = row
    return out


def segment_features(indices, gap):
    lengths = [b - a for a, b in zip(indices[:-1], indices[1:])]
    if not lengths:
        return 0.0, 0.0, 0.0
    max_ratio = max(lengths) / gap
    mean_abs = float(np.mean([abs(length - gap) for length in lengths])) / gap
    std = float(np.std(lengths)) / gap
    return max_ratio, mean_abs, std


def build_decision_records(stage48, stage49, comparisons):
    records = []
    points = sorted({(sample, gap) for sample, gap, method in stage48 if method == "uniform"})
    for sample, gap in points:
        uniform = stage48[(sample, gap, "uniform")]
        oracle = stage49.get((sample, gap, "rendered_prior_0p1"))
        stage49_uniform = stage49.get((sample, gap, "uniform"))
        if oracle is None or stage49_uniform is None:
            continue
        uniform_indices = parse_indices(uniform["indices"])
        oracle_indices = parse_indices(oracle["indices"])
        uniform_all = maybe_float(uniform, "adapter_all_psnr")
        oracle_all = maybe_float(oracle, "adapter_all_psnr")
        oracle_delta = oracle_all - maybe_float(stage49_uniform, "adapter_all_psnr")
        for method in CANDIDATE_METHODS:
            candidate = stage48.get((sample, gap, method))
            if candidate is None:
                continue
            indices = parse_indices(candidate["indices"])
            max_ratio, len_err, seg_std = segment_features(indices, gap)
            comparison = comparisons.get((sample, gap, method), {})
            delta_all = maybe_float(comparison, "selector_delta_adapter_all_psnr")
            delta_middle = maybe_float(comparison, "selector_delta_adapter_middle_psnr")
            if delta_all is None:
                delta_all = maybe_float(candidate, "adapter_all_psnr") - uniform_all
            records.append({
                "sample": sample,
                "reference_gap": gap,
                "method": method,
                "delta_all_psnr": delta_all,
                "delta_middle_psnr": delta_middle,
                "candidate_all_psnr": maybe_float(candidate, "adapter_all_psnr"),
                "uniform_all_psnr": uniform_all,
                "oracle_all_psnr": oracle_all,
                "oracle_delta_all_psnr": oracle_delta,
                "keyframe_count": maybe_int(candidate, "keyframe_count"),
                "exact_uniform": bool_text(indices == uniform_indices),
                "exact_oracle": bool_text(indices == oracle_indices),
                "jaccard_to_uniform": jaccard(indices, uniform_indices),
                "jaccard_to_oracle": jaccard(indices, oracle_indices),
                "position_mae_to_uniform": position_mae(indices, uniform_indices),
                "position_mae_to_oracle": position_mae(indices, oracle_indices),
                "max_segment_ratio": max_ratio,
                "mean_abs_len_error_ratio": len_err,
                "segment_std_ratio": seg_std,
                "selector_cost": maybe_float(candidate, "selector_cost"),
                "candidate_indices": " ".join(str(v) for v in indices),
                "uniform_indices": " ".join(str(v) for v in uniform_indices),
                "oracle_indices": " ".join(str(v) for v in oracle_indices),
            })
    return records


def uniform_choices(points):
    return [{
        "policy": "uniform",
        "sample": sample,
        "reference_gap": gap,
        "chosen_method": "uniform",
        "delta_all_psnr": 0.0,
        "delta_middle_psnr": 0.0,
        "chosen_reason": "fallback baseline",
        "train_policy": "always uniform",
        "jaccard_to_oracle": None,
        "max_segment_ratio": 1.0,
        "mean_abs_len_error_ratio": 0.0,
        "indices": "",
    } for sample, gap in points]


def fixed_method_choices(records, method):
    out = []
    for rec in records:
        if rec["method"] != method:
            continue
        out.append(choice_from_record(f"fixed_{method}", rec, f"fixed method={method}", "fixed method"))
    return out


def choice_from_record(policy, rec, reason, train_policy):
    return {
        "policy": policy,
        "sample": rec["sample"],
        "reference_gap": rec["reference_gap"],
        "chosen_method": rec["method"],
        "delta_all_psnr": rec["delta_all_psnr"],
        "delta_middle_psnr": rec["delta_middle_psnr"],
        "chosen_reason": reason,
        "train_policy": train_policy,
        "jaccard_to_oracle": rec["jaccard_to_oracle"],
        "max_segment_ratio": rec["max_segment_ratio"],
        "mean_abs_len_error_ratio": rec["mean_abs_len_error_ratio"],
        "indices": rec["candidate_indices"],
    }


def uniform_choice(policy, sample, gap, reason, train_policy):
    return {
        "policy": policy,
        "sample": sample,
        "reference_gap": gap,
        "chosen_method": "uniform",
        "delta_all_psnr": 0.0,
        "delta_middle_psnr": 0.0,
        "chosen_reason": reason,
        "train_policy": train_policy,
        "jaccard_to_oracle": None,
        "max_segment_ratio": 1.0,
        "mean_abs_len_error_ratio": 0.0,
        "indices": "",
    }


def oracle_best_candidate_pool(records, points):
    by_point = defaultdict(list)
    for rec in records:
        by_point[(rec["sample"], rec["reference_gap"])].append(rec)
    choices = []
    for sample, gap in points:
        candidates = by_point[(sample, gap)]
        best = max(candidates, key=lambda row: row["delta_all_psnr"])
        if best["delta_all_psnr"] <= 0.0:
            choices.append(uniform_choice("oracle_best_candidate_pool", sample, gap, "oracle fallback chose uniform", "uses actual RD outcome"))
        else:
            choices.append(choice_from_record("oracle_best_candidate_pool", best, "oracle chose best actual candidate", "uses actual RD outcome"))
    return choices


def oracle_layout_imitation(records, points):
    by_point = defaultdict(list)
    for rec in records:
        by_point[(rec["sample"], rec["reference_gap"])].append(rec)
    choices = []
    for sample, gap in points:
        candidates = by_point[(sample, gap)]
        best = max(candidates, key=lambda row: (row["jaccard_to_oracle"], -row["position_mae_to_oracle"], row["exact_oracle"] == "true"))
        uniform_jaccard = jaccard(parse_indices(best["uniform_indices"]), parse_indices(best["oracle_indices"]))
        if uniform_jaccard >= best["jaccard_to_oracle"]:
            choices.append(uniform_choice("oracle_layout_imitation", sample, gap, "uniform is at least as oracle-like by Jaccard", "uses oracle layout"))
        else:
            choices.append(choice_from_record("oracle_layout_imitation", best, "candidate closest to oracle layout", "uses oracle layout"))
    return choices


def evaluate_threshold_policy(records, method, max_threshold, err_threshold, eval_points, policy_name, train_policy):
    rec_map = {(rec["sample"], rec["reference_gap"], rec["method"]): rec for rec in records}
    choices = []
    for sample, gap in eval_points:
        rec = rec_map.get((sample, gap, method))
        if rec and rec["max_segment_ratio"] <= max_threshold and rec["mean_abs_len_error_ratio"] <= err_threshold:
            choices.append(choice_from_record(policy_name, rec, "layout threshold accepted candidate", train_policy))
        else:
            choices.append(uniform_choice(policy_name, sample, gap, "layout threshold fallback to uniform", train_policy))
    return choices


def mean_delta(choices):
    return mean(choice["delta_all_psnr"] for choice in choices)


def train_layout_threshold_policy(records, heldout_sample, points):
    train_points = [point for point in points if point[0] != heldout_sample]
    eval_points = [point for point in points if point[0] == heldout_sample]
    max_values = sorted({rec["max_segment_ratio"] for rec in records if rec["sample"] != heldout_sample}) + [float("inf")]
    err_values = sorted({rec["mean_abs_len_error_ratio"] for rec in records if rec["sample"] != heldout_sample}) + [float("inf")]
    best_spec = ("uniform", 0.0, 0.0, 0.0, 0, "always uniform")
    best_choices = uniform_choices(train_points)
    for method in CANDIDATE_METHODS:
        for max_threshold in max_values:
            for err_threshold in err_values:
                train_policy = f"method={method}; max_segment_ratio<={max_threshold:.6g}; mean_abs_len_error_ratio<={err_threshold:.6g}"
                choices = evaluate_threshold_policy(records, method, max_threshold, err_threshold, train_points, "train", train_policy)
                score = mean_delta(choices)
                positives = sum(1 for choice in choices if choice["delta_all_psnr"] > 0.0)
                accepted = sum(1 for choice in choices if choice["chosen_method"] != "uniform")
                spec = (method, max_threshold, err_threshold, score, positives, train_policy)
                if (score, positives, -accepted) > (best_spec[3], best_spec[4], -sum(1 for choice in best_choices if choice["chosen_method"] != "uniform")):
                    best_spec = spec
                    best_choices = choices
    method, max_threshold, err_threshold, _score, _positives, train_policy = best_spec
    if method == "uniform":
        return [uniform_choice("loocv_layout_threshold", sample, gap, "trained policy chose uniform", train_policy) for sample, gap in eval_points]
    return evaluate_threshold_policy(records, method, max_threshold, err_threshold, eval_points, "loocv_layout_threshold", train_policy)


def loocv_layout_threshold(records, points):
    choices = []
    for heldout in sorted({sample for sample, _gap in points}):
        choices.extend(train_layout_threshold_policy(records, heldout, points))
    return sorted(choices, key=lambda row: (row["sample"], row["reference_gap"]))


def summarize_policy(policy, choices, notes):
    all_values = [choice["delta_all_psnr"] for choice in choices]
    middle_values = [choice["delta_middle_psnr"] for choice in choices if choice["delta_middle_psnr"] is not None]
    return {
        "policy": policy,
        "count": len(choices),
        "accepted_adaptive_points": sum(1 for choice in choices if choice["chosen_method"] != "uniform"),
        "exact_uniform_points": sum(1 for choice in choices if abs(choice["delta_all_psnr"]) < 1e-12),
        "positive_all_points": sum(1 for value in all_values if value > 0.0),
        "positive_middle_points": sum(1 for value in middle_values if value > 0.0),
        "mean_delta_all_psnr": mean(all_values),
        "mean_delta_middle_psnr": mean(middle_values),
        "min_delta_all_psnr": min(all_values) if all_values else None,
        "min_delta_middle_psnr": min(middle_values) if middle_values else None,
        "notes": notes,
    }


def write_report(summary, path):
    lines = [
        "# Stage54 Decision-Aware Selector Analysis",
        "",
        "## Outputs",
        f"- Decision records CSV: `{summary['decision_records_csv']}`",
        f"- Policy choices CSV: `{summary['policy_choices_csv']}`",
        f"- Policy summary CSV: `{summary['policy_summary_csv']}`",
        f"- Summary JSON: `{summary['summary_json']}`",
        "",
        "## Policy Summary",
        "| Policy | Mean all delta | Positive all | Min all delta | Accepted adaptive | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["policy_summaries"]:
        lines.append(
            f"| {row['policy']} | {row['mean_delta_all_psnr']:.6f} | {row['positive_all_points']}/{row['count']} | "
            f"{row['min_delta_all_psnr']:.6f} | {row['accepted_adaptive_points']} | {row['notes']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "- `oracle_best_candidate_pool` is an upper bound that uses actual RD outcomes to choose among Stage48 predicted layouts and uniform.",
        "- `oracle_layout_imitation` uses the Stage49 rendered-oracle layout only for analysis; it is not deployable.",
        "- `loocv_layout_threshold` is a deployable-style fallback rule trained leave-one-sample-out from layout features only, but it reuses Stage48 rendered outcomes as labels for this analysis.",
        "- If the oracle candidate pool is much better than fixed methods, the next step is a decision-aware selector objective/fallback classifier rather than only segment-cost correlation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage48_csv", type=Path, default=DEFAULT_STAGE48_CSV)
    parser.add_argument("--stage48_comparison_csv", type=Path, default=DEFAULT_STAGE48_COMPARISON_CSV)
    parser.add_argument("--stage49_csv", type=Path, default=DEFAULT_STAGE49_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage48, stage49 = load_stage_rows(args.stage48_csv, args.stage49_csv)
    comparisons = load_comparisons(args.stage48_comparison_csv)
    decision_records = build_decision_records(stage48, stage49, comparisons)
    points = sorted({(rec["sample"], rec["reference_gap"]) for rec in decision_records})

    policy_choices = []
    policy_summaries = []
    policies = [
        ("uniform", uniform_choices(points), "baseline fallback"),
        ("oracle_best_candidate_pool", oracle_best_candidate_pool(decision_records, points), "upper bound over Stage48 candidate layouts plus uniform"),
        ("oracle_layout_imitation", oracle_layout_imitation(decision_records, points), "upper bound using Stage49 oracle layout similarity"),
        ("loocv_layout_threshold", loocv_layout_threshold(decision_records, points), "leave-one-sample-out layout-threshold fallback"),
    ]
    for method in CANDIDATE_METHODS:
        policies.append((f"fixed_{method}", fixed_method_choices(decision_records, method), "fixed Stage48 predicted method"))
    for policy, choices, notes in policies:
        policy_choices.extend(choices)
        policy_summaries.append(summarize_policy(policy, choices, notes))

    decision_csv = args.summary_root / "stage54_decision_records.csv"
    choices_csv = args.summary_root / "stage54_policy_choices.csv"
    policy_csv = args.summary_root / "stage54_policy_summary.csv"
    summary_json = args.summary_root / "stage54_decision_aware_selector_analysis_summary.json"
    report_md = args.summary_root / "stage54_decision_aware_selector_analysis_report.md"
    write_csv(decision_records, DECISION_FIELDS, decision_csv)
    write_csv(policy_choices, CHOICE_FIELDS, choices_csv)
    write_csv(policy_summaries, POLICY_FIELDS, policy_csv)
    summary = {
        "stage": 54,
        "mode": "decision-aware selector analysis from existing Stage48/49 RD",
        "stage48_csv": str(args.stage48_csv),
        "stage48_comparison_csv": str(args.stage48_comparison_csv),
        "stage49_csv": str(args.stage49_csv),
        "common_points": len(points),
        "candidate_records": len(decision_records),
        "decision_records_csv": str(decision_csv),
        "policy_choices_csv": str(choices_csv),
        "policy_summary_csv": str(policy_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "policy_summaries": policy_summaries,
        "notes": [
            "This stage does not rerender; it reuses Stage48 predicted-selector RD and Stage49 rendered-oracle RD.",
            "oracle_best_candidate_pool and oracle_layout_imitation are analysis upper bounds, not deployable selectors.",
            "loocv_layout_threshold is a deployable-style fallback rule using layout features, but its labels come from existing rendered RD outcomes.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "common_points": summary["common_points"],
        "candidate_records": summary["candidate_records"],
        "policy_summaries": policy_summaries,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
