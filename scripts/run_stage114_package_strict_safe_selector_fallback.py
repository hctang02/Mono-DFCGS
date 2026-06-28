import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE112_POLICY = REPO_ROOT / "experiments/stage112_broader_group_switch_policy_package/stage112_broader_group_switch_policy.json"
DEFAULT_STAGE113_ROOT = REPO_ROOT / "experiments/stage113_heldout_switch_validation"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage114_strict_safe_selector_fallback_package"

SUMMARY_FIELDS = [
    "policy_name",
    "policy_type",
    "selected_candidate",
    "task_count",
    "mean_selected_sideinfo_psnr",
    "mean_endpoint_sideinfo_psnr",
    "mean_gain_vs_endpoint",
    "min_fold_gain_vs_endpoint",
    "min_group_gain_vs_endpoint",
    "min_fold_group_gain_vs_endpoint",
    "aggregate_safe_no_group_regression",
    "fold_group_safe_no_regression",
]

COMPARISON_FIELDS = [
    "policy",
    "selected_psnr",
    "gain_vs_endpoint",
    "min_group_gain_vs_endpoint",
    "min_fold_group_gain_vs_endpoint",
    "aggregate_safe",
    "fold_group_safe",
    "decision",
    "reason",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def find_row(rows, policy):
    for row in rows:
        if row["policy"] == policy:
            return row
    raise ValueError(f"Missing policy row: {policy}")


def strict_endpoint_table(source_table):
    return {
        base_method: {str(gap): "endpoint_diff_baseline" for gap in sorted(gaps, key=lambda item: int(item))}
        for base_method, gaps in sorted(source_table.items())
    }


def comparison_row(policy, overall, safety, decision, reason):
    return {
        "policy": policy,
        "selected_psnr": float(overall["mean_selected_sideinfo_psnr"]),
        "gain_vs_endpoint": float(overall["mean_delta_psnr_vs_endpoint"]),
        "min_group_gain_vs_endpoint": float(safety["min_group_gain_vs_endpoint"]),
        "min_fold_group_gain_vs_endpoint": float(safety["min_fold_group_gain_vs_endpoint"]),
        "aggregate_safe": int(safety["aggregate_safe_no_group_regression"]),
        "fold_group_safe": int(safety["fold_group_safe_no_regression"]),
        "decision": decision,
        "reason": reason,
    }


def write_report(package, summary_row, comparison_rows, path):
    lines = [
        "# Stage114 Strict-Safe Selector Fallback Package",
        "",
        "## Policy",
        "",
        f"- name: `{package['policy_name']}`",
        f"- type: `{package['policy_type']}`",
        "- fixed selected candidate: `endpoint_diff_baseline`",
        "- decoder inputs: none beyond normal endpoint anchors already available to the decoder",
        "- no target dense anchor, target residual, rendered PSNR, target RGB, or oracle labels are decoder inputs",
        "",
        "## Validation Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| task count | {summary_row['task_count']} |",
        f"| selected PSNR | {summary_row['mean_selected_sideinfo_psnr']} |",
        f"| endpoint PSNR | {summary_row['mean_endpoint_sideinfo_psnr']} |",
        f"| gain vs endpoint | {summary_row['mean_gain_vs_endpoint']} |",
        f"| min fold gain | {summary_row['min_fold_gain_vs_endpoint']} |",
        f"| min group gain | {summary_row['min_group_gain_vs_endpoint']} |",
        f"| min fold-group gain | {summary_row['min_fold_group_gain_vs_endpoint']} |",
        f"| aggregate safe | {summary_row['aggregate_safe_no_group_regression']} |",
        f"| fold-group safe | {summary_row['fold_group_safe_no_regression']} |",
        "",
        "## Policy Comparison",
        "",
        "| policy | selected PSNR | gain | min group gain | min fold-group gain | aggregate safe | fold-group safe | decision | reason |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['policy']} | {row['selected_psnr']} | {row['gain_vs_endpoint']} | {row['min_group_gain_vs_endpoint']} | {row['min_fold_group_gain_vs_endpoint']} | {row['aggregate_safe']} | {row['fold_group_safe']} | {row['decision']} | {row['reason']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- This package intentionally gives up Stage112's average gain to guarantee strict no-regression under the Stage113 diagnostic.",
        "- It is only an index-selection policy; residual values in Stage110/111/113 diagnostics are still teacher values.",
        "- It should be used as the default safe selector for deterministic-index side-info codec work unless broader rendered validation re-qualifies Stage112 v2.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage112_policy", type=Path, default=DEFAULT_STAGE112_POLICY)
    parser.add_argument("--stage113_root", type=Path, default=DEFAULT_STAGE113_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--policy_name", default="strict_safe_endpoint_selector_v1")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage112 = json.loads(args.stage112_policy.read_text(encoding="utf-8"))
    overall_rows = read_csv(args.stage113_root / "stage113_heldout_switch_validation_overall_summary.csv")
    safety_rows = read_csv(args.stage113_root / "stage113_heldout_switch_validation_safety_summary.csv")
    endpoint_overall = find_row(overall_rows, "endpoint_only")
    endpoint_safety = find_row(safety_rows, "endpoint_only")
    stage106_overall = find_row(overall_rows, "stage106_fixed_group_policy")
    stage106_safety = find_row(safety_rows, "stage106_fixed_group_policy")
    stage112_overall = find_row(overall_rows, "stage112_group_switch_v2")
    stage112_safety = find_row(safety_rows, "stage112_group_switch_v2")
    learned_overall = find_row(overall_rows, "score_stat_mlp_cv")
    learned_safety = find_row(safety_rows, "score_stat_mlp_cv")
    selection_table = strict_endpoint_table(stage112["selection_table"])
    summary_row = {
        "policy_name": args.policy_name,
        "policy_type": "fixed_candidate_selector",
        "selected_candidate": "endpoint_diff_baseline",
        "task_count": int(endpoint_overall["task_count"]),
        "mean_selected_sideinfo_psnr": float(endpoint_overall["mean_selected_sideinfo_psnr"]),
        "mean_endpoint_sideinfo_psnr": float(endpoint_overall["mean_endpoint_sideinfo_psnr"]),
        "mean_gain_vs_endpoint": float(endpoint_overall["mean_delta_psnr_vs_endpoint"]),
        "min_fold_gain_vs_endpoint": float(endpoint_safety["min_fold_gain_vs_endpoint"]),
        "min_group_gain_vs_endpoint": float(endpoint_safety["min_group_gain_vs_endpoint"]),
        "min_fold_group_gain_vs_endpoint": float(endpoint_safety["min_fold_group_gain_vs_endpoint"]),
        "aggregate_safe_no_group_regression": int(endpoint_safety["aggregate_safe_no_group_regression"]),
        "fold_group_safe_no_regression": int(endpoint_safety["fold_group_safe_no_regression"]),
    }
    comparison_rows = [
        comparison_row("endpoint_only", endpoint_overall, endpoint_safety, "package", "strict-safe fallback selected by user"),
        comparison_row("stage106_fixed_group_policy", stage106_overall, stage106_safety, "reject", "linear gap4 aggregate regression"),
        comparison_row("stage112_group_switch_v2", stage112_overall, stage112_safety, "reject_as_final", "fold-group regression under Stage113 strict criterion"),
        comparison_row("score_stat_mlp_cv", learned_overall, learned_safety, "reject", "Stage65 adapter gap4 aggregate regression"),
    ]
    package = {
        "stage": 114,
        "policy_name": args.policy_name,
        "policy_type": "fixed_candidate_selector",
        "selected_candidate": "endpoint_diff_baseline",
        "source_decision": "user_selected_strict_safe_fallback_after_stage113",
        "decoder_inputs": [],
        "decoder_available_prerequisites": ["left_anchor", "right_anchor"],
        "forbidden_decoder_inputs": [
            "target_dense_anchor",
            "target_residual",
            "rendered_psnr",
            "oracle_task_label",
            "target_rgb",
        ],
        "fallback_candidate": "endpoint_diff_baseline",
        "selection_table": selection_table,
        "validation_summary": summary_row,
        "comparison_rows": comparison_rows,
        "previous_aggregate_safe_candidate": {
            "policy_name": stage112["policy_name"],
            "rejected_as_final_reason": "Stage113 fold-group safe flag is 0 with min fold-group gain -0.03366017781158855.",
        },
        "rate_note": "Fixed endpoint selector transmits no selector switch side-info and keeps the q6 top10 residual payload shape unchanged.",
        "limitations": [
            "This package selects residual indices only; it does not predict residual values.",
            "Stage110/111/113 validation residual values are still teacher values at selected indices.",
            "Stage113 is a CPU diagnostic over existing rendered rows, not a fresh sequence-heldout render run.",
        ],
    }
    package_json = args.summary_root / "stage114_strict_safe_selector_fallback_policy.json"
    summary_csv = args.summary_root / "stage114_strict_safe_selector_fallback_summary.csv"
    comparison_csv = args.summary_root / "stage114_strict_safe_selector_fallback_comparison.csv"
    report_md = args.summary_root / "stage114_strict_safe_selector_fallback_report.md"
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_csv([summary_row], summary_csv, SUMMARY_FIELDS)
    write_csv(comparison_rows, comparison_csv, COMPARISON_FIELDS)
    write_report(package, summary_row, comparison_rows, report_md)
    print(json.dumps({
        "package": str(package_json),
        "report": str(report_md),
        "policy_name": args.policy_name,
        "fold_group_safe_no_regression": summary_row["fold_group_safe_no_regression"],
    }, indent=2))


if __name__ == "__main__":
    main()
