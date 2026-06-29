import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE122_SUMMARY = REPO_ROOT / "experiments/stage122_compressed_deterministic_rd_package/stage122_compressed_deterministic_rd_setting_summary.csv"
DEFAULT_STAGE125_SUMMARY = REPO_ROOT / "experiments/stage125_broader_feedforward_residual_value_predictor_validation/stage125_broader_feedforward_residual_value_predictor_validation_summary.csv"
DEFAULT_STAGE129_SUMMARY = REPO_ROOT / "experiments/stage129_broader_predictor_codec_rendered_validation/stage129_broader_predictor_codec_rendered_validation_summary.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage130_teacher_sideinfo_vs_predictor_comparison"

FIELDS = [
    "source_stage",
    "method_label",
    "setting_label",
    "setting_role",
    "teacher_required",
    "deployable_no_teacher",
    "residual_payload_bytes",
    "selected_index_payload_bytes",
    "direct_total_mib_per_frame",
    "amortized_total_mib_per_frame",
    "psnr",
    "delta_psnr_vs_linear_base",
    "delta_psnr_vs_full_adapter",
    "note",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_float(value):
    return f"{float(value):.6f}"


def build_rows(stage122_rows, stage125_rows, stage129_rows):
    out = []
    for row in stage122_rows:
        if row["setting_label"] not in {"q4_top20", "q4_top10"}:
            continue
        out.append({
            "source_stage": 122,
            "method_label": "teacher_compressed_sideinfo",
            "setting_label": row["setting_label"],
            "setting_role": row["role"],
            "teacher_required": 1,
            "deployable_no_teacher": 0,
            "residual_payload_bytes": row["mean_sideinfo_payload_bytes"],
            "selected_index_payload_bytes": 0,
            "direct_total_mib_per_frame": row["mean_direct_total_mib_per_frame"],
            "amortized_total_mib_per_frame": row["mean_amortized_total_mib_per_frame"],
            "psnr": row["mean_psnr"],
            "delta_psnr_vs_linear_base": row["mean_delta_psnr_vs_base"],
            "delta_psnr_vs_full_adapter": "",
            "note": "Teacher residual values from dense target anchors; not deployable without residual value predictor.",
        })
    for row in stage125_rows:
        out.append({
            "source_stage": 125,
            "method_label": "adapter_delta_selected_predictor",
            "setting_label": row["setting_label"],
            "setting_role": row["setting_role"],
            "teacher_required": 0,
            "deployable_no_teacher": 1,
            "residual_payload_bytes": 0,
            "selected_index_payload_bytes": 0,
            "direct_total_mib_per_frame": row["mean_direct_total_mib_per_frame"],
            "amortized_total_mib_per_frame": row["mean_amortized_total_mib_per_frame"],
            "psnr": row["mean_selected_predicted_psnr"],
            "delta_psnr_vs_linear_base": row["mean_delta_psnr_vs_base"],
            "delta_psnr_vs_full_adapter": row["mean_delta_psnr_vs_full_predictor"],
            "note": "No teacher target input; uses Stage65 adapter delta at selected indices.",
        })
    for row in stage129_rows:
        out.append({
            "source_stage": 129,
            "method_label": "dedicated_mlp_selected_predictor",
            "setting_label": row["setting_label"],
            "setting_role": row["setting_role"],
            "teacher_required": 0,
            "deployable_no_teacher": 1,
            "residual_payload_bytes": 0,
            "selected_index_payload_bytes": 0,
            "direct_total_mib_per_frame": row["mean_direct_total_mib_per_frame"],
            "amortized_total_mib_per_frame": row["mean_amortized_total_mib_per_frame"],
            "psnr": row["mean_predictor_psnr"],
            "delta_psnr_vs_linear_base": row["mean_delta_psnr_vs_base"],
            "delta_psnr_vs_full_adapter": row["mean_delta_psnr_vs_full_adapter"],
            "note": "No teacher target input; MSE-trained MLP regresses rendered PSNR in Stage129.",
        })
    return sorted(out, key=lambda row: (row["setting_label"], int(row["source_stage"])))


def write_report(rows, package, path):
    lines = [
        "# Stage130 Teacher Side-Info Vs Predictor Comparison",
        "",
        "## Summary",
        "",
        "| stage | method | setting | deployable | teacher | rate | PSNR | delta base | delta full | residual bytes |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta_full = row["delta_psnr_vs_full_adapter"] if row["delta_psnr_vs_full_adapter"] != "" else "nan"
        lines.append(
            f"| {row['source_stage']} | {row['method_label']} | {row['setting_label']} | {row['deployable_no_teacher']} | {row['teacher_required']} | {format_float(row['direct_total_mib_per_frame'])} | {format_float(row['psnr'])} | {format_float(row['delta_psnr_vs_linear_base'])} | {delta_full} | {row['residual_payload_bytes']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Teacher side-info has the highest quality but requires encoder-side target residual values.",
        "- Adapter-delta selected predictor is the current best no-teacher predictor point.",
        "- Dedicated MLP predictor is deployable in input contract but not render-safe yet.",
        "",
        "## Outputs",
        "",
        f"- comparison CSV: `{package['comparison_csv']}`",
        f"- package JSON: `{package['package_json']}`",
        f"- report Markdown: `{package['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage122_summary", type=Path, default=DEFAULT_STAGE122_SUMMARY)
    parser.add_argument("--stage125_summary", type=Path, default=DEFAULT_STAGE125_SUMMARY)
    parser.add_argument("--stage129_summary", type=Path, default=DEFAULT_STAGE129_SUMMARY)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    rows = build_rows(read_csv(args.stage122_summary), read_csv(args.stage125_summary), read_csv(args.stage129_summary))
    comparison_csv = args.summary_root / "stage130_teacher_sideinfo_vs_predictor_comparison_rows.csv"
    package_json = args.summary_root / "stage130_teacher_sideinfo_vs_predictor_comparison_package.json"
    report_md = args.summary_root / "stage130_teacher_sideinfo_vs_predictor_comparison_report.md"
    write_csv(rows, comparison_csv, FIELDS)
    best_deployable = max((row for row in rows if int(row["deployable_no_teacher"]) == 1), key=lambda row: float(row["psnr"]))
    best_teacher = max((row for row in rows if int(row["teacher_required"]) == 1), key=lambda row: float(row["psnr"]))
    package = {
        "stage": 130,
        "mode": "teacher side-info vs predictor comparison",
        "comparison_csv": str(comparison_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "row_count": len(rows),
        "best_deployable_no_teacher": best_deployable,
        "best_teacher_sideinfo": best_teacher,
        "notes": [
            "Teacher side-info uses target residual values and is not deployable as decoder-only residual prediction.",
            "Adapter-delta selected predictor is currently the best no-teacher predictor among compared points.",
            "Dedicated MLP predictor needs render-aware training before final deployment.",
        ],
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(rows, package, report_md)
    print(json.dumps({"package": str(package_json), "row_count": len(rows), "best_deployable": best_deployable}, indent=2))


if __name__ == "__main__":
    main()
