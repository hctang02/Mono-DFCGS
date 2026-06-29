import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE125_ROWS = REPO_ROOT / "experiments/stage125_broader_feedforward_residual_value_predictor_validation/stage125_broader_feedforward_residual_value_predictor_validation_rows.csv"
DEFAULT_STAGE129_ROWS = REPO_ROOT / "experiments/stage129_broader_predictor_codec_rendered_validation/stage129_broader_predictor_codec_rendered_validation_rows.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage134_mlp_render_regression_diagnostic"

DIAGNOSTIC_FIELDS = [
    "task_id",
    "sequence",
    "reference_gap",
    "setting_label",
    "base_psnr",
    "full_adapter_psnr",
    "adapter_delta_psnr",
    "adapter_delta_vs_base",
    "mlp_psnr",
    "mlp_delta_vs_base",
    "mlp_delta_vs_adapter_delta",
    "mlp_delta_vs_full_adapter",
    "mlp_regresses_base",
    "mlp_regresses_adapter_delta",
]

SUMMARY_FIELDS = [
    "group_key",
    "setting_label",
    "reference_gap",
    "task_count",
    "mean_base_psnr",
    "mean_full_adapter_psnr",
    "mean_adapter_delta_psnr",
    "mean_adapter_delta_vs_base",
    "mean_mlp_psnr",
    "mean_mlp_delta_vs_base",
    "mean_mlp_delta_vs_adapter_delta",
    "mlp_base_regression_count",
    "mlp_adapter_regression_count",
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


def average(rows, key):
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def build_diagnostic_rows(adapter_rows, mlp_rows):
    adapter_lookup = {(row["task_id"], row["setting_label"]): row for row in adapter_rows}
    out = []
    for mlp in mlp_rows:
        key = (mlp["task_id"], mlp["setting_label"])
        adapter = adapter_lookup[key]
        adapter_psnr = f(adapter, "selected_predicted_psnr")
        mlp_psnr = f(mlp, "predictor_psnr")
        base_psnr = f(mlp, "base_psnr")
        full_psnr = f(mlp, "full_adapter_psnr")
        out.append({
            "task_id": mlp["task_id"],
            "sequence": mlp["sequence"],
            "reference_gap": int(float(mlp["reference_gap"])),
            "setting_label": mlp["setting_label"],
            "base_psnr": base_psnr,
            "full_adapter_psnr": full_psnr,
            "adapter_delta_psnr": adapter_psnr,
            "adapter_delta_vs_base": adapter_psnr - base_psnr,
            "mlp_psnr": mlp_psnr,
            "mlp_delta_vs_base": mlp_psnr - base_psnr,
            "mlp_delta_vs_adapter_delta": mlp_psnr - adapter_psnr,
            "mlp_delta_vs_full_adapter": mlp_psnr - full_psnr,
            "mlp_regresses_base": int(mlp_psnr < base_psnr),
            "mlp_regresses_adapter_delta": int(mlp_psnr < adapter_psnr),
        })
    return sorted(out, key=lambda row: (row["setting_label"], row["reference_gap"], row["sequence"], row["task_id"]))


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[("setting", row["setting_label"], "all")].append(row)
        grouped[("gap", row["setting_label"], int(row["reference_gap"]))].append(row)
    out = []
    for (group_key, setting, gap), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], str(item[0][2]))):
        out.append({
            "group_key": group_key,
            "setting_label": setting,
            "reference_gap": gap,
            "task_count": len(items),
            "mean_base_psnr": average(items, "base_psnr"),
            "mean_full_adapter_psnr": average(items, "full_adapter_psnr"),
            "mean_adapter_delta_psnr": average(items, "adapter_delta_psnr"),
            "mean_adapter_delta_vs_base": average(items, "adapter_delta_vs_base"),
            "mean_mlp_psnr": average(items, "mlp_psnr"),
            "mean_mlp_delta_vs_base": average(items, "mlp_delta_vs_base"),
            "mean_mlp_delta_vs_adapter_delta": average(items, "mlp_delta_vs_adapter_delta"),
            "mlp_base_regression_count": sum(int(row["mlp_regresses_base"]) for row in items),
            "mlp_adapter_regression_count": sum(int(row["mlp_regresses_adapter_delta"]) for row in items),
        })
    return out


def format_float(value):
    return f"{float(value):.6f}"


def write_report(summary, summary_rows, worst_rows, path):
    lines = [
        "# Stage134 MLP Render Regression Diagnostic",
        "",
        "## Summary",
        "",
        "| group | setting | gap | tasks | adapter delta | MLP delta base | MLP delta adapter | MLP base regressions | MLP adapter regressions |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['group_key']} | {row['setting_label']} | {row['reference_gap']} | {row['task_count']} | {format_float(row['mean_adapter_delta_vs_base'])} | {format_float(row['mean_mlp_delta_vs_base'])} | {format_float(row['mean_mlp_delta_vs_adapter_delta'])} | {row['mlp_base_regression_count']} | {row['mlp_adapter_regression_count']} |"
        )
    lines.extend([
        "",
        "## Worst MLP Regressions Vs Adapter-Delta",
        "",
        "| task | sequence | gap | setting | base | adapter | MLP | MLP-adapter |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ])
    for row in worst_rows[:12]:
        lines.append(
            f"| {row['task_id']} | {row['sequence']} | {row['reference_gap']} | {row['setting_label']} | {format_float(row['base_psnr'])} | {format_float(row['adapter_delta_psnr'])} | {format_float(row['mlp_psnr'])} | {format_float(row['mlp_delta_vs_adapter_delta'])} |"
        )
    lines.extend([
        "",
        "## Diagnosis",
        "",
        "- MLP residual MSE reduction does not align with rendered PSNR.",
        "- MLP underperforms adapter-delta on most tasks, so the next protocol must use render-aware selection/training criteria.",
        "- Stage135 should formalize render-aware predictor training around rendered PSNR/RGB loss, not attribute MSE alone.",
        "",
        "## Outputs",
        "",
        f"- diagnostic rows: `{summary['diagnostic_rows_csv']}`",
        f"- summary rows: `{summary['summary_csv']}`",
        f"- worst rows: `{summary['worst_csv']}`",
        f"- package JSON: `{summary['package_json']}`",
        f"- report Markdown: `{summary['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage125_rows", type=Path, default=DEFAULT_STAGE125_ROWS)
    parser.add_argument("--stage129_rows", type=Path, default=DEFAULT_STAGE129_ROWS)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    rows = build_diagnostic_rows(read_csv(args.stage125_rows), read_csv(args.stage129_rows))
    summary_rows = summarize(rows)
    worst_rows = sorted(rows, key=lambda row: float(row["mlp_delta_vs_adapter_delta"]))[:20]
    diagnostic_rows_csv = args.summary_root / "stage134_mlp_render_regression_diagnostic_rows.csv"
    summary_csv = args.summary_root / "stage134_mlp_render_regression_diagnostic_summary.csv"
    worst_csv = args.summary_root / "stage134_mlp_render_regression_worst_rows.csv"
    package_json = args.summary_root / "stage134_mlp_render_regression_diagnostic_package.json"
    report_md = args.summary_root / "stage134_mlp_render_regression_diagnostic_report.md"
    write_csv(rows, diagnostic_rows_csv, DIAGNOSTIC_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(worst_rows, worst_csv, DIAGNOSTIC_FIELDS)
    package = {
        "stage": 134,
        "mode": "mlp render regression diagnostic",
        "diagnostic_rows_csv": str(diagnostic_rows_csv),
        "summary_csv": str(summary_csv),
        "worst_csv": str(worst_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "row_count": len(rows),
        "summary_row_count": len(summary_rows),
        "worst_row_count": len(worst_rows),
        "notes": [
            "Teacher side-info is not used in this diagnostic.",
            "Rows compare no-teacher adapter-delta and no-teacher MLP predictors.",
            "The next protocol should be render-aware rather than attribute-MSE-only.",
        ],
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, summary_rows, worst_rows, report_md)
    print(json.dumps({"package": str(package_json), "row_count": len(rows), "worst_delta": worst_rows[0]["mlp_delta_vs_adapter_delta"]}, indent=2))


if __name__ == "__main__":
    main()
