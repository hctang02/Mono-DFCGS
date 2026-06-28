import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE110_ROOT = REPO_ROOT / "experiments/stage110_broader_rendered_selector_labels"
DEFAULT_STAGE111_ROOT = REPO_ROOT / "experiments/stage111_broader_switch_predictor"
DEFAULT_STAGE106_POLICY = REPO_ROOT / "experiments/stage106_render_aware_group_policy_package/stage106_render_aware_group_policy.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage112_broader_group_switch_policy_package"


SUMMARY_FIELDS = [
    "policy_name",
    "policy_type",
    "task_count",
    "mean_selected_sideinfo_psnr",
    "mean_endpoint_sideinfo_psnr",
    "mean_gain_vs_endpoint",
    "mean_teacher_oracle_sideinfo_psnr",
    "mean_gap_to_teacher_oracle_psnr",
    "selected_candidate_counts",
]

GROUP_FIELDS = [
    "base_method",
    "reference_gap",
    "selected_candidate",
    "mean_sideinfo_psnr",
    "endpoint_mean_sideinfo_psnr",
    "mean_gain_vs_endpoint",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def find_row(rows, key, value):
    for row in rows:
        if row[key] == value:
            return row
    raise ValueError(f"Missing row where {key}={value}")


def build_selection_table(group_choice_rows):
    table = {}
    for row in group_choice_rows:
        table.setdefault(row["base_method"], {})[str(int(row["reference_gap"]))] = row["selected_candidate"]
    return table


def policy_summary(row):
    return {
        "task_count": int(row["task_count"]),
        "mean_base_psnr": float(row.get("mean_base_psnr", 0.0)),
        "mean_endpoint_sideinfo_psnr": float(row["mean_endpoint_sideinfo_psnr"]),
        "mean_selected_sideinfo_psnr": float(row["mean_selected_sideinfo_psnr"]),
        "mean_gain_vs_endpoint": float(row.get("mean_delta_psnr_vs_endpoint", row.get("gain_vs_endpoint", 0.0))),
        "mean_teacher_oracle_sideinfo_psnr": float(row["mean_teacher_oracle_sideinfo_psnr"]),
        "mean_gap_to_teacher_oracle_psnr": float(row.get("mean_gap_to_teacher_oracle_psnr", row.get("mean_gap_to_teacher_oracle", 0.0))),
        "selected_candidate_counts": row["selected_candidate_counts"],
    }


def stage111_summary(row):
    return {
        "task_count": int(row["task_count"]),
        "mean_endpoint_sideinfo_psnr": float(row["mean_endpoint_sideinfo_psnr"]),
        "mean_selected_sideinfo_psnr": float(row["mean_selected_sideinfo_psnr"]),
        "mean_gain_vs_endpoint": float(row["mean_delta_psnr_vs_endpoint"]),
        "mean_oracle_task_best_sideinfo_psnr": float(row["mean_oracle_task_best_sideinfo_psnr"]),
        "mean_gap_to_oracle_task_best": float(row["mean_gap_to_oracle_task_best"]),
        "selection_accuracy": float(row["selection_accuracy"]),
        "selected_candidate_counts": row["selected_candidate_counts"],
    }


def write_report(package, group_choice_rows, comparison_rows, path):
    validation = package["validation_summary"]
    lines = [
        "# Stage112 Broader Group Switch Policy Package",
        "",
        "## Policy",
        "",
        f"- name: `{package['policy_name']}`",
        f"- type: `{package['policy_type']}`",
        "- decoder inputs: `base_method`, `reference_gap`",
        f"- fallback: `{package['fallback_candidate']}`",
        "- no target dense anchor, target residual, rendered PSNR, target RGB, or oracle labels are decoder inputs",
        "",
        "## Selection Table",
        "",
        "| base | gap | selected candidate | validation gain vs endpoint |",
        "|---|---:|---|---:|",
    ]
    for row in group_choice_rows:
        lines.append(
            f"| {row['base_method']} | {row['reference_gap']} | {row['selected_candidate']} | {row['mean_gain_vs_endpoint']} |"
        )
    lines.extend([
        "",
        "## Validation Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| task count | {validation['task_count']} |",
        f"| endpoint PSNR | {validation['mean_endpoint_sideinfo_psnr']} |",
        f"| policy PSNR | {validation['mean_selected_sideinfo_psnr']} |",
        f"| gain vs endpoint | {validation['mean_gain_vs_endpoint']} |",
        f"| teacher oracle PSNR | {validation['mean_teacher_oracle_sideinfo_psnr']} |",
        f"| gap to teacher | {validation['mean_gap_to_teacher_oracle_psnr']} |",
        "",
        "## Policy Comparison",
        "",
        "| policy | selected PSNR | gain vs endpoint | note |",
        "|---|---:|---:|---|",
    ])
    for row in comparison_rows:
        lines.append(f"| {row['policy']} | {row['selected_psnr']} | {row['gain_vs_endpoint']} | {row['note']} |")
    lines.extend([
        "",
        "## Notes",
        "",
        "- This package only selects the index-selection candidate; it does not predict residual values.",
        "- Stage110/111 validation candidates still used teacher residual values at selected indices.",
        "- Side-info rate is unchanged across candidates because all use the same q6 top10 residual payload shape.",
        "- The learned Stage111 switch is not packaged because it regresses Stage65 adapter gap4.",
        "- This policy should be validated in Stage113 before being treated as final.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage110_root", type=Path, default=DEFAULT_STAGE110_ROOT)
    parser.add_argument("--stage111_root", type=Path, default=DEFAULT_STAGE111_ROOT)
    parser.add_argument("--stage106_policy", type=Path, default=DEFAULT_STAGE106_POLICY)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--policy_name", default="render_aware_group_switch_v2")
    parser.add_argument("--fallback_candidate", default="endpoint_diff_baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage106 = json.loads(args.stage106_policy.read_text(encoding="utf-8"))
    group_choice_rows = read_csv(args.stage110_root / "stage110_broader_render_aware_policy_group_choices.csv")
    stage110_overall = read_csv(args.stage110_root / "stage110_broader_render_aware_policy_overall_summary.csv")
    stage111_summary_rows = read_csv(args.stage111_root / "stage111_broader_switch_predictor_summary.csv")
    policy_row = find_row(stage110_overall, "policy", "group_best_mean_psnr")
    endpoint_row = find_row(stage110_overall, "policy", "endpoint_only")
    stage106_row = find_row(stage110_overall, "policy", "stage106_fixed_group_policy")
    oracle_row = find_row(stage110_overall, "policy", "oracle_task_best")
    learned_row = find_row(stage111_summary_rows, "policy", "score_stat_mlp_cv")
    selection_table = build_selection_table(group_choice_rows)
    validation_summary = policy_summary(policy_row)
    comparison_rows = [
        {
            "policy": "endpoint_only",
            "selected_psnr": endpoint_row["mean_selected_sideinfo_psnr"],
            "gain_vs_endpoint": endpoint_row["mean_delta_psnr_vs_endpoint"],
            "note": "baseline",
        },
        {
            "policy": "stage106_fixed_group_policy",
            "selected_psnr": stage106_row["mean_selected_sideinfo_psnr"],
            "gain_vs_endpoint": stage106_row["mean_delta_psnr_vs_endpoint"],
            "note": "previous packaged policy",
        },
        {
            "policy": args.policy_name,
            "selected_psnr": policy_row["mean_selected_sideinfo_psnr"],
            "gain_vs_endpoint": policy_row["mean_delta_psnr_vs_endpoint"],
            "note": "packaged conservative policy",
        },
        {
            "policy": "score_stat_mlp_cv",
            "selected_psnr": learned_row["mean_selected_sideinfo_psnr"],
            "gain_vs_endpoint": learned_row["mean_delta_psnr_vs_endpoint"],
            "note": "not packaged; adapter gap4 regression",
        },
        {
            "policy": "oracle_task_best",
            "selected_psnr": oracle_row["mean_selected_sideinfo_psnr"],
            "gain_vs_endpoint": oracle_row["mean_delta_psnr_vs_endpoint"],
            "note": "upper bound, not deployable",
        },
    ]
    package = {
        "stage": 112,
        "policy_name": args.policy_name,
        "policy_type": "metadata_group_switch",
        "source_policy": "stage110_group_best_mean_psnr",
        "previous_policy": stage106.get("policy_name", "render_aware_group_switch_v1"),
        "decoder_inputs": ["base_method", "reference_gap"],
        "forbidden_decoder_inputs": [
            "target_dense_anchor",
            "target_residual",
            "rendered_psnr",
            "oracle_task_label",
            "target_rgb",
        ],
        "fallback_candidate": args.fallback_candidate,
        "selection_table": selection_table,
        "validation_summary": validation_summary,
        "stage111_best_learned_summary": stage111_summary(learned_row),
        "rate_note": "Policy switches index-selection candidate only; q6 top10 residual side-info payload shape is unchanged.",
        "selection_rationale": [
            "Stage111 learned switch has higher overall PSNR but regresses Stage65 adapter gap4.",
            "Stage110 group policy has no per-group regression relative to endpoint on Stage110 rows.",
            "The policy uses only decoder-known metadata and is feed-forward.",
        ],
        "limitations": [
            "Residual values are still teacher values in Stage110/111 validation.",
            "This package is not a complete deployable residual-value codec.",
            "The policy was selected from Stage110 rendered validation rows and needs Stage113 held-out validation.",
        ],
    }

    package_json = args.summary_root / "stage112_broader_group_switch_policy.json"
    summary_csv = args.summary_root / "stage112_broader_group_switch_policy_summary.csv"
    table_csv = args.summary_root / "stage112_broader_group_switch_policy_table.csv"
    comparison_csv = args.summary_root / "stage112_broader_group_switch_policy_comparison.csv"
    report_md = args.summary_root / "stage112_broader_group_switch_policy_report.md"
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_csv([
        {
            "policy_name": args.policy_name,
            "policy_type": package["policy_type"],
            "task_count": validation_summary["task_count"],
            "mean_selected_sideinfo_psnr": validation_summary["mean_selected_sideinfo_psnr"],
            "mean_endpoint_sideinfo_psnr": validation_summary["mean_endpoint_sideinfo_psnr"],
            "mean_gain_vs_endpoint": validation_summary["mean_gain_vs_endpoint"],
            "mean_teacher_oracle_sideinfo_psnr": validation_summary["mean_teacher_oracle_sideinfo_psnr"],
            "mean_gap_to_teacher_oracle_psnr": validation_summary["mean_gap_to_teacher_oracle_psnr"],
            "selected_candidate_counts": validation_summary["selected_candidate_counts"],
        }
    ], summary_csv, SUMMARY_FIELDS)
    write_csv(group_choice_rows, table_csv, GROUP_FIELDS)
    write_csv(comparison_rows, comparison_csv, ["policy", "selected_psnr", "gain_vs_endpoint", "note"])
    write_report(package, group_choice_rows, comparison_rows, report_md)
    print(json.dumps({
        "package": str(package_json),
        "report": str(report_md),
        "policy_name": args.policy_name,
        "mean_gain_vs_endpoint": validation_summary["mean_gain_vs_endpoint"],
    }, indent=2))


if __name__ == "__main__":
    main()
