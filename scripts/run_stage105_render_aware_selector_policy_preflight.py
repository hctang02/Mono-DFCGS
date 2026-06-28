import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE103_ROWS = REPO_ROOT / "experiments/stage103_broader_rendered_selector_validation/stage103_broader_rendered_selector_rows.csv"
DEFAULT_STAGE106_POLICY = REPO_ROOT / "experiments/stage106_render_aware_group_policy_package/stage106_render_aware_group_policy.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage105_render_aware_selector_policy_preflight"
DEPLOYABLE_CANDIDATES = ["endpoint_diff_baseline", "shared_energy_regression", "shared_topk_bce"]


ROW_FIELDS = [
    "policy",
    "stage97_task_id",
    "source_task_id",
    "sequence",
    "reference_gap",
    "target_index",
    "base_method",
    "selected_candidate",
    "base_psnr",
    "endpoint_sideinfo_psnr",
    "selected_sideinfo_psnr",
    "teacher_oracle_sideinfo_psnr",
    "delta_psnr_vs_endpoint",
    "delta_psnr_vs_base",
    "gap_to_teacher_oracle_psnr",
    "precision_at_keep",
    "energy_recall_total",
    "relative_energy_recall_vs_oracle",
]

GROUP_SUMMARY_FIELDS = [
    "policy",
    "base_method",
    "reference_gap",
    "task_count",
    "mean_base_psnr",
    "mean_endpoint_sideinfo_psnr",
    "mean_selected_sideinfo_psnr",
    "mean_teacher_oracle_sideinfo_psnr",
    "mean_delta_psnr_vs_endpoint",
    "mean_delta_psnr_vs_base",
    "mean_gap_to_teacher_oracle_psnr",
    "mean_precision_at_keep",
    "mean_energy_recall_total",
    "mean_relative_energy_recall_vs_oracle",
    "selected_candidate_counts",
]

OVERALL_SUMMARY_FIELDS = [
    "policy",
    "task_count",
    "mean_base_psnr",
    "mean_endpoint_sideinfo_psnr",
    "mean_selected_sideinfo_psnr",
    "mean_teacher_oracle_sideinfo_psnr",
    "mean_delta_psnr_vs_endpoint",
    "mean_delta_psnr_vs_base",
    "mean_gap_to_teacher_oracle_psnr",
    "mean_precision_at_keep",
    "mean_energy_recall_total",
    "mean_relative_energy_recall_vs_oracle",
    "selected_candidate_counts",
]

GROUP_CHOICE_FIELDS = [
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


def f(row, key):
    return float(row[key])


def task_key(row):
    return (
        row["stage97_task_id"],
        row["source_task_id"],
        row["base_method"],
        row["reference_gap"],
        row["target_index"],
    )


def candidate_counts(items):
    counts = Counter(row["selected_candidate"] for row in items)
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def group_rows_by_task(stage103_rows):
    tasks = defaultdict(dict)
    for row in stage103_rows:
        tasks[task_key(row)][row["candidate"]] = row
    return {key: candidates for key, candidates in tasks.items() if all(candidate in candidates for candidate in DEPLOYABLE_CANDIDATES)}


def build_group_choices(tasks):
    grouped = defaultdict(list)
    for candidates in tasks.values():
        endpoint = candidates["endpoint_diff_baseline"]
        key = (endpoint["base_method"], int(endpoint["reference_gap"]))
        grouped[key].append(candidates)
    choices = {}
    rows = []
    for key, items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        endpoint_mean = sum(f(candidates["endpoint_diff_baseline"], "sideinfo_psnr") for candidates in items) / len(items)
        best_candidate = None
        best_mean = None
        for candidate in DEPLOYABLE_CANDIDATES:
            mean_psnr = sum(f(candidates[candidate], "sideinfo_psnr") for candidates in items) / len(items)
            if best_mean is None or mean_psnr > best_mean:
                best_candidate = candidate
                best_mean = mean_psnr
        choices[key] = best_candidate
        rows.append({
            "base_method": key[0],
            "reference_gap": key[1],
            "selected_candidate": best_candidate,
            "mean_sideinfo_psnr": best_mean,
            "endpoint_mean_sideinfo_psnr": endpoint_mean,
            "mean_gain_vs_endpoint": best_mean - endpoint_mean,
        })
    return choices, rows


def choose_candidate(policy, candidates, group_choices, stage106_policy):
    endpoint = candidates["endpoint_diff_baseline"]
    group_key = (endpoint["base_method"], int(endpoint["reference_gap"]))
    if policy == "endpoint_only":
        return "endpoint_diff_baseline"
    if policy == "always_shared_energy_regression":
        return "shared_energy_regression"
    if policy == "always_shared_topk_bce":
        return "shared_topk_bce"
    if policy == "group_best_mean_psnr":
        return group_choices[group_key]
    if policy == "stage106_fixed_group_policy":
        return stage106_policy.get(endpoint["base_method"], {}).get(str(int(endpoint["reference_gap"])), "endpoint_diff_baseline")
    if policy == "oracle_task_best":
        return max(DEPLOYABLE_CANDIDATES, key=lambda candidate: f(candidates[candidate], "sideinfo_psnr"))
    raise ValueError(f"Unknown policy: {policy}")


def build_policy_rows(tasks, group_choices, policies, stage106_policy):
    rows = []
    for _, candidates in sorted(tasks.items()):
        endpoint = candidates["endpoint_diff_baseline"]
        for policy in policies:
            selected_candidate = choose_candidate(policy, candidates, group_choices, stage106_policy)
            selected = candidates[selected_candidate]
            rows.append({
                "policy": policy,
                "stage97_task_id": selected["stage97_task_id"],
                "source_task_id": selected["source_task_id"],
                "sequence": selected["sequence"],
                "reference_gap": int(selected["reference_gap"]),
                "target_index": int(selected["target_index"]),
                "base_method": selected["base_method"],
                "selected_candidate": selected_candidate,
                "base_psnr": f(selected, "base_psnr"),
                "endpoint_sideinfo_psnr": f(endpoint, "sideinfo_psnr"),
                "selected_sideinfo_psnr": f(selected, "sideinfo_psnr"),
                "teacher_oracle_sideinfo_psnr": f(selected, "teacher_oracle_sideinfo_psnr"),
                "delta_psnr_vs_endpoint": f(selected, "sideinfo_psnr") - f(endpoint, "sideinfo_psnr"),
                "delta_psnr_vs_base": f(selected, "delta_psnr_vs_base"),
                "gap_to_teacher_oracle_psnr": f(selected, "gap_to_teacher_oracle_psnr"),
                "precision_at_keep": f(selected, "precision_at_keep"),
                "energy_recall_total": f(selected, "energy_recall_total"),
                "relative_energy_recall_vs_oracle": f(selected, "relative_energy_recall_vs_oracle"),
            })
    return rows


def summarize_items(policy, items):
    def avg(key):
        return sum(float(row[key]) for row in items) / len(items)
    return {
        "policy": policy,
        "task_count": len(items),
        "mean_base_psnr": avg("base_psnr"),
        "mean_endpoint_sideinfo_psnr": avg("endpoint_sideinfo_psnr"),
        "mean_selected_sideinfo_psnr": avg("selected_sideinfo_psnr"),
        "mean_teacher_oracle_sideinfo_psnr": avg("teacher_oracle_sideinfo_psnr"),
        "mean_delta_psnr_vs_endpoint": avg("delta_psnr_vs_endpoint"),
        "mean_delta_psnr_vs_base": avg("delta_psnr_vs_base"),
        "mean_gap_to_teacher_oracle_psnr": avg("gap_to_teacher_oracle_psnr"),
        "mean_precision_at_keep": avg("precision_at_keep"),
        "mean_energy_recall_total": avg("energy_recall_total"),
        "mean_relative_energy_recall_vs_oracle": avg("relative_energy_recall_vs_oracle"),
        "selected_candidate_counts": candidate_counts(items),
    }


def summarize_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row["base_method"], row["reference_gap"])].append(row)
    out = []
    for (policy, base_method, gap), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], int(item[0][2]))):
        summary = summarize_items(policy, items)
        summary.update({"base_method": base_method, "reference_gap": int(gap)})
        out.append(summary)
    return out


def summarize_overall(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    return [summarize_items(policy, items) for policy, items in sorted(grouped.items())]


def write_report(summary, group_choice_rows, group_summary_rows, overall_summary_rows, path):
    lines = [
        f"# {summary.get('report_title', 'Stage105 Render-Aware Selector Policy Preflight')}",
        "",
        "## Configuration",
        "",
        f"- input rows: `{summary['input_rows']}`",
        f"- task count: `{summary['task_count']}`",
        f"- policies: `{', '.join(summary['policies'])}`",
        "- no rendering or heavy tensor output",
        "",
        "## Overall Summary",
        "",
        "| policy | tasks | selected PSNR | endpoint PSNR | gain vs endpoint | teacher PSNR | gap to teacher | selections |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in overall_summary_rows:
        lines.append(
            f"| {row['policy']} | {row['task_count']} | {row['mean_selected_sideinfo_psnr']} | {row['mean_endpoint_sideinfo_psnr']} | {row['mean_delta_psnr_vs_endpoint']} | {row['mean_teacher_oracle_sideinfo_psnr']} | {row['mean_gap_to_teacher_oracle_psnr']} | {row['selected_candidate_counts']} |"
        )
    lines.extend([
        "",
        "## Group Policy Choices",
        "",
        "| base | gap | selected candidate | mean PSNR | endpoint PSNR | gain |",
        "|---|---:|---|---:|---:|---:|",
    ])
    for row in group_choice_rows:
        lines.append(
            f"| {row['base_method']} | {row['reference_gap']} | {row['selected_candidate']} | {row['mean_sideinfo_psnr']} | {row['endpoint_mean_sideinfo_psnr']} | {row['mean_gain_vs_endpoint']} |"
        )
    lines.extend([
        "",
        "## Group Summary",
        "",
        "| policy | base | gap | tasks | selected PSNR | gain vs endpoint | gap to teacher | selections |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in group_summary_rows:
        lines.append(
            f"| {row['policy']} | {row['base_method']} | {row['reference_gap']} | {row['task_count']} | {row['mean_selected_sideinfo_psnr']} | {row['mean_delta_psnr_vs_endpoint']} | {row['mean_gap_to_teacher_oracle_psnr']} | {row['selected_candidate_counts']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `group_best_mean_psnr` is a fixed group policy selected from rendered validation rows.",
        "- `oracle_task_best` uses per-task rendered PSNR and is only an upper bound, not deployable.",
        "- All candidates still use teacher residual values at selected indices; residual value prediction remains unresolved.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage103_rows", type=Path, default=DEFAULT_STAGE103_ROWS)
    parser.add_argument("--stage106_policy", type=Path, default=DEFAULT_STAGE106_POLICY)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--stage", type=int, default=105)
    parser.add_argument("--mode", default="render-aware selector policy preflight")
    parser.add_argument("--output_prefix", default="stage105_render_aware_policy")
    parser.add_argument("--report_title", default="Stage105 Render-Aware Selector Policy Preflight")
    parser.add_argument("--policies", nargs="+", default=[
        "endpoint_only",
        "always_shared_energy_regression",
        "always_shared_topk_bce",
        "group_best_mean_psnr",
        "oracle_task_best",
    ])
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage103_rows = read_csv(args.stage103_rows)
    stage106_policy = {}
    if args.stage106_policy and args.stage106_policy.exists():
        stage106_policy = json.loads(args.stage106_policy.read_text(encoding="utf-8")).get("selection_table", {})
    tasks = group_rows_by_task(stage103_rows)
    group_choices, group_choice_rows = build_group_choices(tasks)
    policy_rows = build_policy_rows(tasks, group_choices, args.policies, stage106_policy)
    group_summary_rows = summarize_groups(policy_rows)
    overall_summary_rows = summarize_overall(policy_rows)

    rows_csv = args.summary_root / f"{args.output_prefix}_rows.csv"
    group_summary_csv = args.summary_root / f"{args.output_prefix}_group_summary.csv"
    overall_summary_csv = args.summary_root / f"{args.output_prefix}_overall_summary.csv"
    group_choices_csv = args.summary_root / f"{args.output_prefix}_group_choices.csv"
    summary_json = args.summary_root / f"{args.output_prefix}_summary.json"
    report_md = args.summary_root / f"{args.output_prefix}_report.md"

    write_csv(policy_rows, rows_csv, ROW_FIELDS)
    write_csv(group_summary_rows, group_summary_csv, GROUP_SUMMARY_FIELDS)
    write_csv(overall_summary_rows, overall_summary_csv, OVERALL_SUMMARY_FIELDS)
    write_csv(group_choice_rows, group_choices_csv, GROUP_CHOICE_FIELDS)
    summary = {
        "stage": args.stage,
        "mode": args.mode,
        "report_title": args.report_title,
        "input_rows": str(args.stage103_rows),
        "stage106_policy": str(args.stage106_policy),
        "task_count": len(tasks),
        "policies": args.policies,
        "deployable_candidates": DEPLOYABLE_CANDIDATES,
        "rows_csv": str(rows_csv),
        "group_summary_csv": str(group_summary_csv),
        "overall_summary_csv": str(overall_summary_csv),
        "group_choices_csv": str(group_choices_csv),
        "report_md": str(report_md),
        "group_choice_rows": group_choice_rows,
        "overall_summary_rows": overall_summary_rows,
        "notes": [
            "No rendering or heavy tensor output is produced.",
            "Oracle task policy uses per-task rendered PSNR and is not deployable.",
            "Candidates still use teacher residual values at selected indices.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, group_choice_rows, group_summary_rows, overall_summary_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "task_count": len(tasks),
    }, indent=2))


if __name__ == "__main__":
    main()
