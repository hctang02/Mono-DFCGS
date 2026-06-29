import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE124_SUMMARY = REPO_ROOT / "experiments/stage124_feedforward_residual_value_predictor_smoke/stage124_feedforward_residual_value_predictor_smoke_summary.csv"
DEFAULT_STAGE125_SUMMARY = REPO_ROOT / "experiments/stage125_broader_feedforward_residual_value_predictor_validation/stage125_broader_feedforward_residual_value_predictor_validation_summary.csv"
DEFAULT_STAGE127_METRICS = REPO_ROOT / "experiments/stage127_selected_residual_predictor_training_smoke/stage127_selected_residual_predictor_training_smoke_metrics.csv"
DEFAULT_STAGE129_SUMMARY = REPO_ROOT / "experiments/stage129_broader_predictor_codec_rendered_validation/stage129_broader_predictor_codec_rendered_validation_summary.csv"
DEFAULT_STAGE130_ROWS = REPO_ROOT / "experiments/stage130_teacher_sideinfo_vs_predictor_comparison/stage130_teacher_sideinfo_vs_predictor_comparison_rows.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage131_predictor_ablation_package"

ABLATION_FIELDS = [
    "ablation_group",
    "source_stage",
    "method_label",
    "setting_label",
    "task_count",
    "direct_total_mib_per_frame",
    "psnr",
    "delta_psnr_vs_linear_base",
    "delta_psnr_vs_full_adapter",
    "eval_mse_reduction",
    "teacher_required",
    "deployable_no_teacher",
    "takeaway",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def by_setting(rows):
    return {row["setting_label"]: row for row in rows}


def format_float(value):
    if value in (None, ""):
        return "nan"
    return f"{float(value):.6f}"


def build_rows(stage124, stage125, stage127, stage129, stage130):
    rows = []
    stage124_by = by_setting(stage124)
    stage125_by = by_setting(stage125)
    stage127_by = by_setting(stage127)
    stage129_by = by_setting(stage129)
    for setting in ["q4_top10", "q4_top20"]:
        row124 = stage124_by[setting]
        row125 = stage125_by[setting]
        rows.append({
            "ablation_group": "validation_scale_adapter_delta",
            "source_stage": 124,
            "method_label": "adapter_delta_selected_predictor",
            "setting_label": setting,
            "task_count": row124["task_count"],
            "direct_total_mib_per_frame": row124["mean_direct_total_mib_per_frame"],
            "psnr": row124["mean_selected_predicted_psnr"],
            "delta_psnr_vs_linear_base": row124["mean_delta_psnr_vs_base"],
            "delta_psnr_vs_full_adapter": row124["mean_delta_psnr_vs_full_predictor"],
            "eval_mse_reduction": "",
            "teacher_required": 0,
            "deployable_no_teacher": 1,
            "takeaway": "12-task smoke; positive but small no-teacher gain.",
        })
        rows.append({
            "ablation_group": "validation_scale_adapter_delta",
            "source_stage": 125,
            "method_label": "adapter_delta_selected_predictor",
            "setting_label": setting,
            "task_count": row125["task_count"],
            "direct_total_mib_per_frame": row125["mean_direct_total_mib_per_frame"],
            "psnr": row125["mean_selected_predicted_psnr"],
            "delta_psnr_vs_linear_base": row125["mean_delta_psnr_vs_base"],
            "delta_psnr_vs_full_adapter": row125["mean_delta_psnr_vs_full_predictor"],
            "eval_mse_reduction": "",
            "teacher_required": 0,
            "deployable_no_teacher": 1,
            "takeaway": "60-task validation; gain survives and q4_top20 is best.",
        })
        row127 = stage127_by[setting]
        row129 = stage129_by[setting]
        rows.append({
            "ablation_group": "mse_vs_render_dedicated_mlp",
            "source_stage": 127,
            "method_label": "dedicated_mlp_selected_predictor_mse_eval",
            "setting_label": setting,
            "task_count": row127["eval_task_count"],
            "direct_total_mib_per_frame": "",
            "psnr": "",
            "delta_psnr_vs_linear_base": "",
            "delta_psnr_vs_full_adapter": "",
            "eval_mse_reduction": row127["eval_mse_reduction"],
            "teacher_required": 0,
            "deployable_no_teacher": 1,
            "takeaway": "Residual MSE improves, but this does not guarantee rendered PSNR.",
        })
        rows.append({
            "ablation_group": "mse_vs_render_dedicated_mlp",
            "source_stage": 129,
            "method_label": "dedicated_mlp_selected_predictor_rendered",
            "setting_label": setting,
            "task_count": row129["task_count"],
            "direct_total_mib_per_frame": row129["mean_direct_total_mib_per_frame"],
            "psnr": row129["mean_predictor_psnr"],
            "delta_psnr_vs_linear_base": row129["mean_delta_psnr_vs_base"],
            "delta_psnr_vs_full_adapter": row129["mean_delta_psnr_vs_full_adapter"],
            "eval_mse_reduction": "",
            "teacher_required": 0,
            "deployable_no_teacher": 1,
            "takeaway": "Rendered PSNR regresses, so MLP is not render-safe.",
        })
    for row in stage130:
        rows.append({
            "ablation_group": "teacher_vs_predictor_keep_fraction",
            "source_stage": row["source_stage"],
            "method_label": row["method_label"],
            "setting_label": row["setting_label"],
            "task_count": "60",
            "direct_total_mib_per_frame": row["direct_total_mib_per_frame"],
            "psnr": row["psnr"],
            "delta_psnr_vs_linear_base": row["delta_psnr_vs_linear_base"],
            "delta_psnr_vs_full_adapter": row["delta_psnr_vs_full_adapter"],
            "eval_mse_reduction": "",
            "teacher_required": row["teacher_required"],
            "deployable_no_teacher": row["deployable_no_teacher"],
            "takeaway": "Teacher quality is highest; adapter-delta is current best deployable no-teacher point.",
        })
    return rows


def write_report(rows, package, path):
    lines = [
        "# Stage131 Predictor Ablation Package",
        "",
        "## Key Takeaways",
        "",
        "- Adapter-delta selected predictor has small but stable positive rendered gain over linear base.",
        "- Dedicated MLP reduces residual MSE but regresses rendered PSNR, so MSE labels are not enough.",
        "- q4/top20 is the best no-teacher adapter-delta point; q4/top10 is lower keep but slightly lower quality.",
        "- Teacher residual side-info remains a non-deployable upper-quality reference.",
        "",
        "## Rows",
        "",
        "| group | stage | method | setting | tasks | rate | PSNR | delta base | eval MSE reduction | deployable |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['ablation_group']} | {row['source_stage']} | {row['method_label']} | {row['setting_label']} | {row['task_count']} | {format_float(row['direct_total_mib_per_frame'])} | {format_float(row['psnr'])} | {format_float(row['delta_psnr_vs_linear_base'])} | {format_float(row['eval_mse_reduction'])} | {row['deployable_no_teacher']} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- ablation CSV: `{package['ablation_csv']}`",
        f"- package JSON: `{package['package_json']}`",
        f"- report Markdown: `{package['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage124_summary", type=Path, default=DEFAULT_STAGE124_SUMMARY)
    parser.add_argument("--stage125_summary", type=Path, default=DEFAULT_STAGE125_SUMMARY)
    parser.add_argument("--stage127_metrics", type=Path, default=DEFAULT_STAGE127_METRICS)
    parser.add_argument("--stage129_summary", type=Path, default=DEFAULT_STAGE129_SUMMARY)
    parser.add_argument("--stage130_rows", type=Path, default=DEFAULT_STAGE130_ROWS)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    rows = build_rows(
        read_csv(args.stage124_summary),
        read_csv(args.stage125_summary),
        read_csv(args.stage127_metrics),
        read_csv(args.stage129_summary),
        read_csv(args.stage130_rows),
    )
    ablation_csv = args.summary_root / "stage131_predictor_ablation_rows.csv"
    package_json = args.summary_root / "stage131_predictor_ablation_package.json"
    report_md = args.summary_root / "stage131_predictor_ablation_report.md"
    write_csv(rows, ablation_csv, ABLATION_FIELDS)
    package = {
        "stage": 131,
        "mode": "predictor ablation package",
        "ablation_csv": str(ablation_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "row_count": len(rows),
        "recommended_deployable_predictor": "adapter_delta_selected_predictor/q4_top20",
        "rejected_final_predictor": "dedicated_mlp_selected_predictor_v1",
        "rejection_reason": "MSE-trained MLP regresses rendered PSNR in Stage129.",
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(rows, package, report_md)
    print(json.dumps({"package": str(package_json), "row_count": len(rows), "recommended": package["recommended_deployable_predictor"]}, indent=2))


if __name__ == "__main__":
    main()
