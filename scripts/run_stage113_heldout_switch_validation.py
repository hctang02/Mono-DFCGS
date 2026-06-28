import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE111_ROWS = REPO_ROOT / "experiments/stage111_broader_switch_predictor/stage111_broader_switch_predictor_rows.csv"
DEFAULT_STAGE112_POLICY = REPO_ROOT / "experiments/stage112_broader_group_switch_policy_package/stage112_broader_group_switch_policy.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage113_heldout_switch_validation"

POLICY_ALIASES = {
    "endpoint_only": "endpoint_only",
    "stage106_fixed_group_policy": "stage106_fixed_group_policy",
    "stage110_group_best_policy": "stage112_group_switch_v2",
    "train_fold_group_policy": "train_fold_group_policy",
    "score_stat_mlp_cv": "score_stat_mlp_cv",
    "oracle_task_best": "oracle_task_best",
}

ROW_FIELDS = [
    "fold",
    "policy",
    "stage97_task_id",
    "source_task_id",
    "sequence",
    "base_method",
    "reference_gap",
    "target_index",
    "oracle_candidate",
    "selected_candidate",
    "selected_sideinfo_psnr",
    "endpoint_sideinfo_psnr",
    "oracle_task_best_sideinfo_psnr",
    "teacher_oracle_sideinfo_psnr",
    "delta_psnr_vs_endpoint",
    "gap_to_oracle_task_best",
    "gap_to_teacher_oracle",
    "selection_correct",
]

SUMMARY_FIELDS = [
    "policy",
    "task_count",
    "mean_selected_sideinfo_psnr",
    "mean_endpoint_sideinfo_psnr",
    "mean_oracle_task_best_sideinfo_psnr",
    "mean_teacher_oracle_sideinfo_psnr",
    "mean_delta_psnr_vs_endpoint",
    "mean_gap_to_oracle_task_best",
    "mean_gap_to_teacher_oracle",
    "selection_accuracy",
    "min_delta_psnr_vs_endpoint",
    "negative_task_count",
    "selected_candidate_counts",
]

GROUP_FIELDS = [
    "policy",
    "base_method",
    "reference_gap",
    *SUMMARY_FIELDS[1:],
]

FOLD_FIELDS = [
    "fold",
    *SUMMARY_FIELDS,
]

FOLD_GROUP_FIELDS = [
    "fold",
    "policy",
    "base_method",
    "reference_gap",
    *SUMMARY_FIELDS[1:],
]

SEQUENCE_FIELDS = [
    "policy",
    "sequence",
    *SUMMARY_FIELDS[1:],
]

SAFETY_FIELDS = [
    "policy",
    "task_count",
    "mean_delta_psnr_vs_endpoint",
    "min_fold_gain_vs_endpoint",
    "negative_fold_count",
    "min_group_gain_vs_endpoint",
    "negative_group_count",
    "min_fold_group_gain_vs_endpoint",
    "negative_fold_group_count",
    "min_sequence_gain_vs_endpoint",
    "negative_sequence_count",
    "stage65_adapter_gap4_gain_vs_endpoint",
    "aggregate_safe_no_group_regression",
    "fold_group_safe_no_regression",
    "selected_candidate_counts",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def f(row, key):
    return float(row[key])


def candidate_counts(rows):
    counts = Counter(row["selected_candidate"] for row in rows)
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def summarize_items(policy, rows):
    def avg(key):
        return sum(f(row, key) for row in rows) / len(rows)

    deltas = [f(row, "delta_psnr_vs_endpoint") for row in rows]
    return {
        "policy": policy,
        "task_count": len(rows),
        "mean_selected_sideinfo_psnr": avg("selected_sideinfo_psnr"),
        "mean_endpoint_sideinfo_psnr": avg("endpoint_sideinfo_psnr"),
        "mean_oracle_task_best_sideinfo_psnr": avg("oracle_task_best_sideinfo_psnr"),
        "mean_teacher_oracle_sideinfo_psnr": avg("teacher_oracle_sideinfo_psnr"),
        "mean_delta_psnr_vs_endpoint": avg("delta_psnr_vs_endpoint"),
        "mean_gap_to_oracle_task_best": avg("gap_to_oracle_task_best"),
        "mean_gap_to_teacher_oracle": avg("gap_to_teacher_oracle"),
        "selection_accuracy": avg("selection_correct"),
        "min_delta_psnr_vs_endpoint": min(deltas),
        "negative_task_count": sum(1 for value in deltas if value < 0.0),
        "selected_candidate_counts": candidate_counts(rows),
    }


def summarize_by(rows, key_fn, decorate_fn, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    out = []
    for key, items in sorted(grouped.items(), key=lambda item: item[0]):
        summary = summarize_items(items[0]["policy"], items)
        summary.update(decorate_fn(key))
        out.append({field: summary[field] for field in fields})
    return out


def clone_filtered_rows(rows):
    out = []
    for row in rows:
        if row["policy"] not in POLICY_ALIASES:
            continue
        cloned = {field: row[field] for field in ROW_FIELDS}
        cloned["policy"] = POLICY_ALIASES[row["policy"]]
        out.append(cloned)
    return out


def verify_stage112_rows(rows, package):
    table = package["selection_table"]
    mismatches = []
    for row in rows:
        if row["policy"] != "stage112_group_switch_v2":
            continue
        expected = table.get(row["base_method"], {}).get(str(int(row["reference_gap"])), package["fallback_candidate"])
        if row["selected_candidate"] != expected:
            mismatches.append({
                "stage97_task_id": row["stage97_task_id"],
                "base_method": row["base_method"],
                "reference_gap": row["reference_gap"],
                "selected_candidate": row["selected_candidate"],
                "expected_candidate": expected,
            })
    return mismatches


def build_summaries(rows):
    overall = summarize_by(
        rows,
        lambda row: row["policy"],
        lambda key: {"policy": key},
        SUMMARY_FIELDS,
    )
    group = summarize_by(
        rows,
        lambda row: (row["policy"], row["base_method"], int(row["reference_gap"])),
        lambda key: {"policy": key[0], "base_method": key[1], "reference_gap": key[2]},
        GROUP_FIELDS,
    )
    fold = summarize_by(
        rows,
        lambda row: (int(row["fold"]), row["policy"]),
        lambda key: {"fold": key[0], "policy": key[1]},
        FOLD_FIELDS,
    )
    fold_group = summarize_by(
        rows,
        lambda row: (int(row["fold"]), row["policy"], row["base_method"], int(row["reference_gap"])),
        lambda key: {"fold": key[0], "policy": key[1], "base_method": key[2], "reference_gap": key[3]},
        FOLD_GROUP_FIELDS,
    )
    sequence = summarize_by(
        rows,
        lambda row: (row["policy"], row["sequence"]),
        lambda key: {"policy": key[0], "sequence": key[1]},
        SEQUENCE_FIELDS,
    )
    return overall, group, fold, fold_group, sequence


def min_gain(rows, policy, key="mean_delta_psnr_vs_endpoint"):
    values = [float(row[key]) for row in rows if row["policy"] == policy]
    return min(values) if values else 0.0


def negative_count(rows, policy, key="mean_delta_psnr_vs_endpoint"):
    return sum(1 for row in rows if row["policy"] == policy and float(row[key]) < 0.0)


def find_group_gain(group_rows, policy, base_method, reference_gap):
    for row in group_rows:
        if row["policy"] == policy and row["base_method"] == base_method and int(row["reference_gap"]) == reference_gap:
            return float(row["mean_delta_psnr_vs_endpoint"])
    return 0.0


def build_safety_summary(overall_rows, group_rows, fold_rows, fold_group_rows, sequence_rows):
    out = []
    for row in overall_rows:
        policy = row["policy"]
        min_group = min_gain(group_rows, policy)
        min_fold_group = min_gain(fold_group_rows, policy)
        out.append({
            "policy": policy,
            "task_count": row["task_count"],
            "mean_delta_psnr_vs_endpoint": row["mean_delta_psnr_vs_endpoint"],
            "min_fold_gain_vs_endpoint": min_gain(fold_rows, policy),
            "negative_fold_count": negative_count(fold_rows, policy),
            "min_group_gain_vs_endpoint": min_group,
            "negative_group_count": negative_count(group_rows, policy),
            "min_fold_group_gain_vs_endpoint": min_fold_group,
            "negative_fold_group_count": negative_count(fold_group_rows, policy),
            "min_sequence_gain_vs_endpoint": min_gain(sequence_rows, policy),
            "negative_sequence_count": negative_count(sequence_rows, policy),
            "stage65_adapter_gap4_gain_vs_endpoint": find_group_gain(group_rows, policy, "stage65_adapter", 4),
            "aggregate_safe_no_group_regression": int(min_group >= 0.0),
            "fold_group_safe_no_regression": int(min_fold_group >= 0.0),
            "selected_candidate_counts": row["selected_candidate_counts"],
        })
    return out


def write_report(summary, overall_rows, safety_rows, group_rows, fold_rows, path):
    focus_policies = [
        "endpoint_only",
        "stage106_fixed_group_policy",
        "stage112_group_switch_v2",
        "train_fold_group_policy",
        "score_stat_mlp_cv",
        "oracle_task_best",
    ]
    lines = [
        "# Stage113 Held-Out Switch Validation",
        "",
        "## Configuration",
        "",
        f"- input rows: `{summary['input_rows']}`",
        f"- held-out unit: `{summary['held_out_unit']}`",
        f"- fold count: `{summary['fold_count']}`",
        f"- Stage112 policy: `{summary['stage112_policy_name']}`",
        f"- Stage112 alias mismatch count: `{summary['stage112_alias_mismatch_count']}`",
        "- no rerendering, no training, no checkpoint, no heavy tensor output",
        "",
        "## Overall Summary",
        "",
        "| policy | tasks | selected PSNR | gain vs endpoint | oracle task gap | accuracy | selections |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in overall_rows:
        if row["policy"] not in focus_policies:
            continue
        lines.append(
            f"| {row['policy']} | {row['task_count']} | {row['mean_selected_sideinfo_psnr']} | {row['mean_delta_psnr_vs_endpoint']} | {row['mean_gap_to_oracle_task_best']} | {row['selection_accuracy']} | {row['selected_candidate_counts']} |"
        )
    lines.extend([
        "",
        "## Safety Summary",
        "",
        "| policy | mean gain | min fold gain | neg folds | min group gain | neg groups | min fold-group gain | neg fold-groups | Stage65 gap4 gain | aggregate safe | fold-group safe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in safety_rows:
        if row["policy"] not in focus_policies:
            continue
        lines.append(
            f"| {row['policy']} | {row['mean_delta_psnr_vs_endpoint']} | {row['min_fold_gain_vs_endpoint']} | {row['negative_fold_count']} | {row['min_group_gain_vs_endpoint']} | {row['negative_group_count']} | {row['min_fold_group_gain_vs_endpoint']} | {row['negative_fold_group_count']} | {row['stage65_adapter_gap4_gain_vs_endpoint']} | {row['aggregate_safe_no_group_regression']} | {row['fold_group_safe_no_regression']} |"
        )
    lines.extend([
        "",
        "## Group Summary",
        "",
        "| policy | base | gap | tasks | gain vs endpoint | selections |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in group_rows:
        if row["policy"] not in ["stage106_fixed_group_policy", "stage112_group_switch_v2", "train_fold_group_policy", "score_stat_mlp_cv"]:
            continue
        lines.append(
            f"| {row['policy']} | {row['base_method']} | {row['reference_gap']} | {row['task_count']} | {row['mean_delta_psnr_vs_endpoint']} | {row['selected_candidate_counts']} |"
        )
    lines.extend([
        "",
        "## Fold Summary",
        "",
        "| fold | policy | tasks | gain vs endpoint | selections |",
        "|---:|---|---:|---:|---|",
    ])
    for row in fold_rows:
        if row["policy"] not in ["stage112_group_switch_v2", "train_fold_group_policy", "score_stat_mlp_cv"]:
            continue
        lines.append(
            f"| {row['fold']} | {row['policy']} | {row['task_count']} | {row['mean_delta_psnr_vs_endpoint']} | {row['selected_candidate_counts']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- The Stage111 fold split is by sorted `stage97_task_id` modulo fold count; this is task-id held-out CV, not a fresh sequence-heldout render run.",
        "- `stage112_group_switch_v2` is evaluated by aliasing Stage111 `stage110_group_best_policy` rows and verifying the Stage112 package selection table.",
        "- Stage110/111 rows still use teacher residual values at selected indices, so this validates selector switching, not a complete residual-value codec.",
        "- If fold-group stability is required as a hard safety condition, any negative fold-group cell should be treated as a blocker before residual value prediction.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage111_rows", type=Path, default=DEFAULT_STAGE111_ROWS)
    parser.add_argument("--stage112_policy", type=Path, default=DEFAULT_STAGE112_POLICY)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    package = json.loads(args.stage112_policy.read_text(encoding="utf-8"))
    rows = clone_filtered_rows(read_csv(args.stage111_rows))
    mismatches = verify_stage112_rows(rows, package)
    overall_rows, group_rows, fold_rows, fold_group_rows, sequence_rows = build_summaries(rows)
    safety_rows = build_safety_summary(overall_rows, group_rows, fold_rows, fold_group_rows, sequence_rows)
    fold_count = len({int(row["fold"]) for row in rows})
    summary = {
        "stage": 113,
        "mode": "held-out switch validation",
        "input_rows": str(args.stage111_rows),
        "stage112_policy": str(args.stage112_policy),
        "stage112_policy_name": package["policy_name"],
        "held_out_unit": "stage97_task_id modulo fold from Stage111 rows",
        "fold_count": fold_count,
        "task_policy_rows": len(rows),
        "unique_tasks_per_policy": len(rows) // len(set(row["policy"] for row in rows)),
        "policies": sorted({row["policy"] for row in rows}),
        "stage112_alias_source_policy": "stage110_group_best_policy",
        "stage112_alias_mismatch_count": len(mismatches),
        "stage112_alias_mismatches": mismatches[:20],
        "overall_summary": overall_rows,
        "safety_summary": safety_rows,
        "limitations": [
            "This is a CPU diagnostic over existing rendered rows, not a new render run.",
            "The fold split is task-id based and not a strict sequence-heldout split.",
            "Residual values remain teacher values at selected indices.",
        ],
    }
    rows_csv = args.summary_root / "stage113_heldout_switch_validation_rows.csv"
    overall_csv = args.summary_root / "stage113_heldout_switch_validation_overall_summary.csv"
    group_csv = args.summary_root / "stage113_heldout_switch_validation_group_summary.csv"
    fold_csv = args.summary_root / "stage113_heldout_switch_validation_fold_summary.csv"
    fold_group_csv = args.summary_root / "stage113_heldout_switch_validation_fold_group_summary.csv"
    sequence_csv = args.summary_root / "stage113_heldout_switch_validation_sequence_summary.csv"
    safety_csv = args.summary_root / "stage113_heldout_switch_validation_safety_summary.csv"
    summary_json = args.summary_root / "stage113_heldout_switch_validation_summary.json"
    report_md = args.summary_root / "stage113_heldout_switch_validation_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(overall_rows, overall_csv, SUMMARY_FIELDS)
    write_csv(group_rows, group_csv, GROUP_FIELDS)
    write_csv(fold_rows, fold_csv, FOLD_FIELDS)
    write_csv(fold_group_rows, fold_group_csv, FOLD_GROUP_FIELDS)
    write_csv(sequence_rows, sequence_csv, SEQUENCE_FIELDS)
    write_csv(safety_rows, safety_csv, SAFETY_FIELDS)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, overall_rows, safety_rows, group_rows, fold_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "task_policy_rows": len(rows),
        "stage112_alias_mismatch_count": len(mismatches),
    }, indent=2))


if __name__ == "__main__":
    main()
