import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE129_SUMMARY = REPO_ROOT / "experiments/stage129_broader_predictor_codec_rendered_validation/stage129_broader_predictor_codec_rendered_validation_summary.csv"
DEFAULT_STAGE134_PACKAGE = REPO_ROOT / "experiments/stage134_mlp_render_regression_diagnostic/stage134_mlp_render_regression_diagnostic_package.json"
DEFAULT_STAGE137_SUMMARY = REPO_ROOT / "experiments/stage137_broader_render_aware_scale_validation/stage137_broader_render_aware_scale_validation_summary.json"
DEFAULT_STAGE138_POLICY = REPO_ROOT / "experiments/stage138_render_aware_deployable_policy_package/stage138_render_aware_deployable_policy.json"
DEFAULT_STAGE139_PACKAGE = REPO_ROOT / "experiments/stage139_full_pipeline_rd_package/stage139_full_pipeline_rd_package.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage140_multi_setting_ablation_package"

ABLATION_FIELDS = [
    "method_family",
    "setting_label",
    "setting_role",
    "keep_fraction",
    "adapter_delta_scale",
    "task_count",
    "direct_total_mib_per_frame",
    "psnr",
    "delta_psnr_vs_base",
    "delta_psnr_vs_stage132_scale1",
    "deployable_no_teacher",
    "final_selected",
    "rejected_final",
    "rejection_reason",
    "transmitted_residual_payload_bytes",
    "transmitted_selected_index_bytes",
    "source_stage",
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


def format_float(value):
    return f"{float(value):.6f}"


def stage132_lookup(summary_rows):
    return {
        row["setting_label"]: row
        for row in summary_rows
        if abs(float(row["adapter_delta_scale"]) - 1.0) < 1e-12
    }


def selected_lookup(policy):
    return {
        (row["setting_label"], float(row["adapter_delta_scale"]))
        for row in policy["settings"]
    }


def add_reference_rows(rows, stage137_summary):
    first_by_setting = {}
    for row in stage137_summary["summary_rows"]:
        first_by_setting.setdefault(row["setting_label"], row)
    for setting_label, row in sorted(first_by_setting.items()):
        rows.append({
            "method_family": "linear_base_reference",
            "setting_label": setting_label,
            "setting_role": row["setting_role"],
            "keep_fraction": row["keep_fraction"],
            "adapter_delta_scale": 0.0,
            "task_count": row["task_count"],
            "direct_total_mib_per_frame": row["mean_direct_total_mib_per_frame"],
            "psnr": row["mean_base_psnr"],
            "delta_psnr_vs_base": 0.0,
            "delta_psnr_vs_stage132_scale1": "",
            "deployable_no_teacher": 1,
            "final_selected": 0,
            "rejected_final": 0,
            "rejection_reason": "reference only",
            "transmitted_residual_payload_bytes": 0,
            "transmitted_selected_index_bytes": 0,
            "source_stage": 137,
        })
        rows.append({
            "method_family": "full_stage65_adapter_reference",
            "setting_label": setting_label,
            "setting_role": row["setting_role"],
            "keep_fraction": row["keep_fraction"],
            "adapter_delta_scale": "full",
            "task_count": row["task_count"],
            "direct_total_mib_per_frame": row["mean_direct_total_mib_per_frame"],
            "psnr": row["mean_full_predictor_psnr"],
            "delta_psnr_vs_base": float(row["mean_full_predictor_psnr"]) - float(row["mean_base_psnr"]),
            "delta_psnr_vs_stage132_scale1": "",
            "deployable_no_teacher": 1,
            "final_selected": 0,
            "rejected_final": 0,
            "rejection_reason": "reference only; not the selected sparse residual codec",
            "transmitted_residual_payload_bytes": 0,
            "transmitted_selected_index_bytes": 0,
            "source_stage": 137,
        })


def add_scale_rows(rows, stage137_summary, policy):
    unscaled = stage132_lookup(stage137_summary["summary_rows"])
    selected = selected_lookup(policy)
    for row in stage137_summary["summary_rows"]:
        previous = unscaled[row["setting_label"]]
        scale = float(row["adapter_delta_scale"])
        final_selected = int((row["setting_label"], scale) in selected)
        rows.append({
            "method_family": "adapter_delta_selected_scaled",
            "setting_label": row["setting_label"],
            "setting_role": row["setting_role"],
            "keep_fraction": row["keep_fraction"],
            "adapter_delta_scale": row["adapter_delta_scale"],
            "task_count": row["task_count"],
            "direct_total_mib_per_frame": row["mean_direct_total_mib_per_frame"],
            "psnr": row["mean_selected_predicted_psnr"],
            "delta_psnr_vs_base": row["mean_delta_psnr_vs_base"],
            "delta_psnr_vs_stage132_scale1": float(row["mean_selected_predicted_psnr"]) - float(previous["mean_selected_predicted_psnr"]),
            "deployable_no_teacher": 1,
            "final_selected": final_selected,
            "rejected_final": 0,
            "rejection_reason": "" if final_selected else "not selected by Stage138 policy",
            "transmitted_residual_payload_bytes": 0,
            "transmitted_selected_index_bytes": 0,
            "source_stage": 137,
        })


def add_mlp_rows(rows, stage129_rows):
    for row in stage129_rows:
        rows.append({
            "method_family": "dedicated_mlp_selected_predictor",
            "setting_label": row["setting_label"],
            "setting_role": row["setting_role"],
            "keep_fraction": row["keep_fraction"],
            "adapter_delta_scale": "n/a",
            "task_count": row["task_count"],
            "direct_total_mib_per_frame": row["mean_direct_total_mib_per_frame"],
            "psnr": row["mean_predictor_psnr"],
            "delta_psnr_vs_base": row["mean_delta_psnr_vs_base"],
            "delta_psnr_vs_stage132_scale1": "",
            "deployable_no_teacher": 1,
            "final_selected": 0,
            "rejected_final": 1,
            "rejection_reason": "rendered PSNR regression in Stage129 and Stage134",
            "transmitted_residual_payload_bytes": 0,
            "transmitted_selected_index_bytes": 0,
            "source_stage": 129,
        })


def build_rows(stage137_summary, policy, stage129_rows):
    rows = []
    add_reference_rows(rows, stage137_summary)
    add_scale_rows(rows, stage137_summary, policy)
    add_mlp_rows(rows, stage129_rows)
    return sorted(rows, key=lambda row: (row["setting_label"], row["method_family"], str(row["adapter_delta_scale"])))


def best_selected(policy):
    return next(row for row in policy["settings"] if row["role"] == "primary")


def write_report(rows, package, summary, path):
    selected = [row for row in rows if int(row["final_selected"]) == 1]
    rejected = [row for row in rows if int(row["rejected_final"]) == 1]
    lines = [
        "# Stage140 Multi-Setting Ablation Package",
        "",
        "## Conclusion",
        "",
        f"- Final primary: `{summary['final_primary']['setting_label']}` scale `{summary['final_primary']['adapter_delta_scale']}`.",
        f"- Final primary PSNR: `{format_float(summary['final_primary']['psnr'])}` at rate `{format_float(summary['final_primary']['direct_total_mib_per_frame'])}`.",
        f"- Improvement over Stage132 q4/top20 scale1: `{format_float(summary['final_primary']['delta_psnr_vs_stage132_policy'])}` dB.",
        "- Dedicated MLP remains rejected because rendered PSNR regresses despite residual-MSE improvements.",
        "- Teacher residual side-info is not used as a deployable or optimization target in this package.",
        "",
        "## Selected Rows",
        "",
        "| setting | role | keep | scale | rate | PSNR | delta base | delta Stage132 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['setting_label']} | {row['setting_role']} | {row['keep_fraction']} | {row['adapter_delta_scale']} | {format_float(row['direct_total_mib_per_frame'])} | {format_float(row['psnr'])} | {format_float(row['delta_psnr_vs_base'])} | {format_float(row['delta_psnr_vs_stage132_scale1'])} |"
        )
    lines.extend([
        "",
        "## Rejected MLP Rows",
        "",
        "| setting | role | rate | PSNR | delta base | reason |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in rejected:
        lines.append(
            f"| {row['setting_label']} | {row['setting_role']} | {format_float(row['direct_total_mib_per_frame'])} | {format_float(row['psnr'])} | {format_float(row['delta_psnr_vs_base'])} | {row['rejection_reason']} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- ablation CSV: `{package['ablation_csv']}`",
        f"- summary JSON: `{package['summary_json']}`",
        f"- package JSON: `{package['package_json']}`",
        f"- report Markdown: `{package['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage129_summary", type=Path, default=DEFAULT_STAGE129_SUMMARY)
    parser.add_argument("--stage134_package", type=Path, default=DEFAULT_STAGE134_PACKAGE)
    parser.add_argument("--stage137_summary", type=Path, default=DEFAULT_STAGE137_SUMMARY)
    parser.add_argument("--stage138_policy", type=Path, default=DEFAULT_STAGE138_POLICY)
    parser.add_argument("--stage139_package", type=Path, default=DEFAULT_STAGE139_PACKAGE)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage137_summary = read_json(args.stage137_summary)
    stage138_policy = read_json(args.stage138_policy)
    stage134_package = read_json(args.stage134_package)
    stage139_package = read_json(args.stage139_package)
    rows = build_rows(stage137_summary, stage138_policy, read_csv(args.stage129_summary))
    final_primary = best_selected(stage138_policy)
    ablation_csv = args.summary_root / "stage140_multi_setting_ablation_rows.csv"
    summary_json = args.summary_root / "stage140_multi_setting_ablation_summary.json"
    package_json = args.summary_root / "stage140_multi_setting_ablation_package.json"
    report_md = args.summary_root / "stage140_multi_setting_ablation_report.md"
    write_csv(rows, ablation_csv, ABLATION_FIELDS)
    summary = {
        "stage": 140,
        "mode": "multi-setting ablation package",
        "stage129_summary": str(args.stage129_summary),
        "stage134_package": str(args.stage134_package),
        "stage137_summary": str(args.stage137_summary),
        "stage138_policy": str(args.stage138_policy),
        "stage139_package": str(args.stage139_package),
        "row_count": len(rows),
        "final_policy": stage138_policy["policy_name"],
        "final_primary": final_primary,
        "stage134_row_count": stage134_package["row_count"],
        "stage139_row_count": stage139_package["row_count"],
        "ablation_csv": str(ablation_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "notes": [
            "Stage137 scale sweep supports the Stage138 scale 0.75 policy.",
            "Stage129/134 MLP is rejected due to rendered PSNR regression.",
            "Teacher residual side-info is not optimized or included as deployable side-info.",
        ],
    }
    package = {
        "stage": 140,
        "mode": "multi-setting ablation package",
        "final_policy": stage138_policy["policy_name"],
        "row_count": len(rows),
        "ablation_csv": str(ablation_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "notes": summary["notes"],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(rows, package, summary, report_md)
    print(json.dumps({"package": str(package_json), "row_count": len(rows), "final_primary": final_primary}, indent=2))


if __name__ == "__main__":
    main()
