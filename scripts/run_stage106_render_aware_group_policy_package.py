import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE105_ROOT = REPO_ROOT / "experiments/stage105_render_aware_selector_policy_preflight"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage106_render_aware_group_policy_package"


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


def find_policy_row(rows, policy):
    for row in rows:
        if row["policy"] == policy:
            return row
    raise ValueError(f"Missing policy row: {policy}")


def build_selection_table(group_choice_rows):
    table = {}
    for row in group_choice_rows:
        table.setdefault(row["base_method"], {})[str(int(row["reference_gap"]))] = row["selected_candidate"]
    return table


def write_report(package, group_choice_rows, path):
    validation = package["validation_summary"]
    lines = [
        "# Stage106 Render-Aware Group Policy Package",
        "",
        "## Policy",
        "",
        f"- name: `{package['policy_name']}`",
        f"- type: `{package['policy_type']}`",
        "- decoder inputs: `base_method`, `reference_gap`",
        f"- fallback: `{package['fallback_candidate']}`",
        "- no target residual, rendered PSNR, or oracle task labels are decoder inputs",
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
        "## Notes",
        "",
        "- This package only selects the index-selection candidate; it does not predict residual values.",
        "- All Stage105 validation candidates still used teacher residual values at selected indices.",
        "- Side-info rate is unchanged across candidates because all use the same q6 top10 residual payload shape.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage105_root", type=Path, default=DEFAULT_STAGE105_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--policy", default="group_best_mean_psnr")
    parser.add_argument("--policy_name", default="render_aware_group_switch_v1")
    parser.add_argument("--fallback_candidate", default="endpoint_diff_baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    group_choices_csv = args.stage105_root / "stage105_render_aware_policy_group_choices.csv"
    overall_summary_csv = args.stage105_root / "stage105_render_aware_policy_overall_summary.csv"
    group_choice_rows = read_csv(group_choices_csv)
    overall_rows = read_csv(overall_summary_csv)
    policy_row = find_policy_row(overall_rows, args.policy)
    selection_table = build_selection_table(group_choice_rows)
    validation_summary = {
        "task_count": int(policy_row["task_count"]),
        "mean_base_psnr": float(policy_row["mean_base_psnr"]),
        "mean_endpoint_sideinfo_psnr": float(policy_row["mean_endpoint_sideinfo_psnr"]),
        "mean_selected_sideinfo_psnr": float(policy_row["mean_selected_sideinfo_psnr"]),
        "mean_gain_vs_endpoint": float(policy_row["mean_delta_psnr_vs_endpoint"]),
        "mean_teacher_oracle_sideinfo_psnr": float(policy_row["mean_teacher_oracle_sideinfo_psnr"]),
        "mean_gap_to_teacher_oracle_psnr": float(policy_row["mean_gap_to_teacher_oracle_psnr"]),
        "mean_precision_at_keep": float(policy_row["mean_precision_at_keep"]),
        "mean_energy_recall_total": float(policy_row["mean_energy_recall_total"]),
        "mean_relative_energy_recall_vs_oracle": float(policy_row["mean_relative_energy_recall_vs_oracle"]),
        "selected_candidate_counts": policy_row["selected_candidate_counts"],
    }
    package = {
        "stage": 106,
        "policy_name": args.policy_name,
        "policy_type": "metadata_group_switch",
        "source_policy": args.policy,
        "decoder_inputs": ["base_method", "reference_gap"],
        "forbidden_decoder_inputs": ["target_dense_anchor", "target_residual", "rendered_psnr", "oracle_task_label"],
        "fallback_candidate": args.fallback_candidate,
        "selection_table": selection_table,
        "validation_summary": validation_summary,
        "rate_note": "Policy switches index-selection candidate only; q6 top10 residual side-info payload shape is unchanged.",
        "limitations": [
            "Residual values are still teacher values in Stage105 validation.",
            "This package is not a complete deployable residual-value codec.",
            "The policy was selected from rendered validation rows and should be revalidated on held-out data.",
        ],
    }

    package_json = args.summary_root / "stage106_render_aware_group_policy.json"
    summary_csv = args.summary_root / "stage106_render_aware_group_policy_summary.csv"
    group_csv = args.summary_root / "stage106_render_aware_group_policy_table.csv"
    report_md = args.summary_root / "stage106_render_aware_group_policy_report.md"
    package_json.write_text(json.dumps(package, indent=2), encoding="utf-8")
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
    write_csv(group_choice_rows, group_csv, GROUP_FIELDS)
    write_report(package, group_choice_rows, report_md)
    print(json.dumps({
        "package": str(package_json),
        "report": str(report_md),
        "mean_gain_vs_endpoint": validation_summary["mean_gain_vs_endpoint"],
    }, indent=2))


if __name__ == "__main__":
    main()
