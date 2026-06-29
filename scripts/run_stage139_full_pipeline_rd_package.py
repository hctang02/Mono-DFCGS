import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE78_RATE_TABLE = REPO_ROOT / "experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_rate_table.csv"
DEFAULT_STAGE137_ROWS = REPO_ROOT / "experiments/stage137_broader_render_aware_scale_validation/stage137_broader_render_aware_scale_validation_rows.csv"
DEFAULT_STAGE138_POLICY = REPO_ROOT / "experiments/stage138_render_aware_deployable_policy_package/stage138_render_aware_deployable_policy.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage139_full_pipeline_rd_package"

ROW_FIELDS = [
    "role",
    "setting_label",
    "keep_fraction",
    "adapter_delta_scale",
    "reference_gap",
    "task_count",
    "q12_main_anchor_mib_per_frame",
    "residual_payload_mib_per_frame",
    "selected_index_payload_mib_per_frame",
    "policy_scale_payload_mib_per_frame",
    "direct_total_mib_per_frame",
    "amortized_total_mib_per_frame",
    "mean_base_psnr",
    "mean_full_adapter_psnr",
    "mean_final_psnr",
    "mean_delta_psnr_vs_base",
    "mean_delta_psnr_vs_stage132_scale1",
    "transmitted_residual_payload_bytes",
    "transmitted_selected_index_bytes",
    "deployable_no_teacher",
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


def average(rows, key):
    values = [float(row[key]) for row in rows]
    return sum(values) / max(len(values), 1)


def build_rate_lookup(rate_rows):
    out = {}
    for row in rate_rows:
        if row["codec"] != "q12" or row["method"] != "linear":
            continue
        out[int(row["frame_gap"])] = float(row["mean_static_anchor_mib_per_frame_with_metadata"])
    return out


def selected_rows(stage137_rows, setting_label, scale):
    return [
        row for row in stage137_rows
        if row["setting_label"] == setting_label and abs(float(row["adapter_delta_scale"]) - float(scale)) < 1e-12
    ]


def group_by_gap(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["reference_gap"])].append(row)
    return grouped


def build_metric_row(role, setting, rows, previous_rows, reference_gap):
    if not rows:
        raise RuntimeError(f"no rows for {role} {setting['setting_label']} gap {reference_gap}")
    previous_psnr = average(previous_rows, "selected_predicted_psnr") if previous_rows else average(rows, "selected_predicted_psnr")
    final_psnr = average(rows, "selected_predicted_psnr")
    q12_main_rate = average(rows, "q12_main_anchor_mib_per_frame")
    return {
        "role": role,
        "setting_label": setting["setting_label"],
        "keep_fraction": setting["keep_fraction"],
        "adapter_delta_scale": setting["adapter_delta_scale"],
        "reference_gap": reference_gap,
        "task_count": len(rows),
        "q12_main_anchor_mib_per_frame": q12_main_rate,
        "residual_payload_mib_per_frame": 0.0,
        "selected_index_payload_mib_per_frame": 0.0,
        "policy_scale_payload_mib_per_frame": 0.0,
        "direct_total_mib_per_frame": q12_main_rate,
        "amortized_total_mib_per_frame": q12_main_rate,
        "mean_base_psnr": average(rows, "base_psnr"),
        "mean_full_adapter_psnr": average(rows, "full_predictor_psnr"),
        "mean_final_psnr": final_psnr,
        "mean_delta_psnr_vs_base": average(rows, "delta_psnr_vs_base"),
        "mean_delta_psnr_vs_stage132_scale1": final_psnr - previous_psnr,
        "transmitted_residual_payload_bytes": 0,
        "transmitted_selected_index_bytes": 0,
        "deployable_no_teacher": 1,
    }


def build_rows(policy, stage137_rows):
    out = []
    for setting in policy["settings"]:
        rows = selected_rows(stage137_rows, setting["setting_label"], setting["adapter_delta_scale"])
        previous_rows = selected_rows(stage137_rows, setting["setting_label"], 1.0)
        out.append(build_metric_row(setting["role"], setting, rows, previous_rows, "all"))
        selected_by_gap = group_by_gap(rows)
        previous_by_gap = group_by_gap(previous_rows)
        for gap in sorted(selected_by_gap):
            out.append(build_metric_row(setting["role"], setting, selected_by_gap[gap], previous_by_gap.get(gap, []), gap))
    return out


def format_float(value):
    return f"{float(value):.6f}"


def write_report(policy, rows, package, path):
    aggregate_rows = [row for row in rows if row["reference_gap"] == "all"]
    gap_rows = [row for row in rows if row["reference_gap"] != "all"]
    lines = [
        "# Stage139 Full-Pipeline RD Package",
        "",
        "## Scope",
        "",
        "- Packages full-pipeline rate accounting for the Stage138 deployable no-teacher predictor policy.",
        "- Main stream rate is q12 linear-anchor rate from Stage78/Stage137.",
        "- Residual payload, selected-index payload, and policy-scale payload are zero per frame.",
        "- Teacher residual side-info is not part of the deployable pipeline.",
        "",
        "## Policy",
        "",
        f"- policy: `{policy['policy_name']}`",
        f"- primary: `{policy['selected_primary_setting']}` scale `{policy['selected_primary_adapter_delta_scale']}`",
        f"- low-rate: `{policy['optional_low_rate_setting']}` scale `{policy['optional_low_rate_adapter_delta_scale']}`",
        "",
        "## Aggregate RD",
        "",
        "| role | setting | keep | scale | tasks | rate | final PSNR | delta base | delta Stage132 scale1 | residual bytes | index bytes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['role']} | {row['setting_label']} | {row['keep_fraction']} | {row['adapter_delta_scale']} | {row['task_count']} | {format_float(row['direct_total_mib_per_frame'])} | {format_float(row['mean_final_psnr'])} | {format_float(row['mean_delta_psnr_vs_base'])} | {format_float(row['mean_delta_psnr_vs_stage132_scale1'])} | {row['transmitted_residual_payload_bytes']} | {row['transmitted_selected_index_bytes']} |"
        )
    lines.extend([
        "",
        "## Gap Breakdown",
        "",
        "| role | setting | gap | tasks | q12 anchor rate | final PSNR | delta base | delta Stage132 scale1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in gap_rows:
        lines.append(
            f"| {row['role']} | {row['setting_label']} | {row['reference_gap']} | {row['task_count']} | {format_float(row['q12_main_anchor_mib_per_frame'])} | {format_float(row['mean_final_psnr'])} | {format_float(row['mean_delta_psnr_vs_base'])} | {format_float(row['mean_delta_psnr_vs_stage132_scale1'])} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- rows CSV: `{package['rows_csv']}`",
        f"- summary JSON: `{package['summary_json']}`",
        f"- package JSON: `{package['package_json']}`",
        f"- report Markdown: `{package['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage78_rate_table", type=Path, default=DEFAULT_STAGE78_RATE_TABLE)
    parser.add_argument("--stage137_rows", type=Path, default=DEFAULT_STAGE137_ROWS)
    parser.add_argument("--stage138_policy", type=Path, default=DEFAULT_STAGE138_POLICY)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    policy = read_json(args.stage138_policy)
    rate_lookup = build_rate_lookup(read_csv(args.stage78_rate_table))
    rows = build_rows(policy, read_csv(args.stage137_rows))
    for row in rows:
        gap = row["reference_gap"]
        if gap != "all" and abs(float(row["q12_main_anchor_mib_per_frame"]) - rate_lookup[int(gap)]) > 1e-12:
            raise RuntimeError(f"q12 rate mismatch for gap {gap}")
    rows_csv = args.summary_root / "stage139_full_pipeline_rd_rows.csv"
    summary_json = args.summary_root / "stage139_full_pipeline_rd_summary.json"
    package_json = args.summary_root / "stage139_full_pipeline_rd_package.json"
    report_md = args.summary_root / "stage139_full_pipeline_rd_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    aggregate_rows = [row for row in rows if row["reference_gap"] == "all"]
    summary = {
        "stage": 139,
        "mode": "full-pipeline RD package",
        "stage78_rate_table": str(args.stage78_rate_table),
        "stage137_rows": str(args.stage137_rows),
        "stage138_policy": str(args.stage138_policy),
        "policy_name": policy["policy_name"],
        "row_count": len(rows),
        "aggregate_rows": aggregate_rows,
        "q12_rate_lookup": rate_lookup,
        "rate_accounting": {
            "q12_main_anchor_mib_per_frame": "from Stage78 rate table and Stage137 rows",
            "residual_payload_mib_per_frame": 0.0,
            "selected_index_payload_mib_per_frame": 0.0,
            "policy_scale_payload_mib_per_frame": 0.0,
            "teacher_sideinfo_included": 0,
        },
        "decoder_side_only": 1,
        "forbidden_decoder_inputs": policy["decoder_contract"]["forbidden_inputs"],
        "rows_csv": str(rows_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package = {
        "stage": 139,
        "mode": "full-pipeline RD package",
        "policy_name": policy["policy_name"],
        "stage138_policy": str(args.stage138_policy),
        "rows_csv": str(rows_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "row_count": len(rows),
        "aggregate_rows": aggregate_rows,
        "notes": [
            "Main stream rate is q12 linear-anchor rate from Stage78/Stage137.",
            "No residual payload bytes are transmitted.",
            "No selected-index bytes are transmitted.",
            "Policy adapter-delta scale is a fixed policy constant and has no per-frame side-info bytes.",
            "Teacher residual side-info is not part of this deployable pipeline package.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(policy, rows, package, report_md)
    print(json.dumps({"package": str(package_json), "row_count": len(rows), "aggregate_rows": aggregate_rows}, indent=2))


if __name__ == "__main__":
    main()
