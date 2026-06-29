import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE130_ROWS = REPO_ROOT / "experiments/stage130_teacher_sideinfo_vs_predictor_comparison/stage130_teacher_sideinfo_vs_predictor_comparison_rows.csv"
DEFAULT_STAGE132_POLICY = REPO_ROOT / "experiments/stage132_deployable_predictor_policy_package/stage132_deployable_predictor_policy.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage133_final_predictor_rd_report"

FINAL_FIELDS = [
    "method_label",
    "setting_label",
    "deployable_no_teacher",
    "teacher_required",
    "direct_total_mib_per_frame",
    "psnr",
    "delta_psnr_vs_linear_base",
    "residual_payload_bytes",
    "selected_index_payload_bytes",
    "final_role",
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


def final_role(row, policy):
    if row["method_label"] == "adapter_delta_selected_predictor" and row["setting_label"] == policy["selected_primary_setting"]:
        return "final_deployable_primary"
    if row["method_label"] == "adapter_delta_selected_predictor" and row["setting_label"] == policy["optional_low_rate_setting"]:
        return "final_deployable_low_rate"
    if row["method_label"] == "teacher_compressed_sideinfo":
        return "teacher_reference_only"
    if row["method_label"] == "dedicated_mlp_selected_predictor":
        return "rejected_render_regression"
    return "other"


def build_rows(stage130_rows, policy):
    rows = []
    for row in stage130_rows:
        rows.append({
            "method_label": row["method_label"],
            "setting_label": row["setting_label"],
            "deployable_no_teacher": row["deployable_no_teacher"],
            "teacher_required": row["teacher_required"],
            "direct_total_mib_per_frame": row["direct_total_mib_per_frame"],
            "psnr": row["psnr"],
            "delta_psnr_vs_linear_base": row["delta_psnr_vs_linear_base"],
            "residual_payload_bytes": row["residual_payload_bytes"],
            "selected_index_payload_bytes": row["selected_index_payload_bytes"],
            "final_role": final_role(row, policy),
        })
    return sorted(rows, key=lambda row: (row["final_role"], row["setting_label"], row["method_label"]))


def plot_rd(rows, png_path, pdf_path):
    colors = {
        "teacher_compressed_sideinfo": "#9467bd",
        "adapter_delta_selected_predictor": "#2ca02c",
        "dedicated_mlp_selected_predictor": "#d62728",
    }
    markers = {"1": "o", "0": "X"}
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    for row in rows:
        method = row["method_label"]
        x = float(row["direct_total_mib_per_frame"])
        y = float(row["psnr"])
        deployable = row["deployable_no_teacher"]
        ax.scatter(x, y, s=80, marker=markers.get(deployable, "o"), color=colors.get(method, "#333333"), edgecolor="black", linewidth=0.6, label=method)
        ax.annotate(row["setting_label"], (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(unique.values(), unique.keys(), fontsize=8, loc="best")
    ax.set_xlabel("Direct total rate (MiB/frame)")
    ax.set_ylabel("Rendered PSNR (dB)")
    ax.set_title("Teacher Side-Info Reference vs Predictor-Only Residual Policies")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)


def format_float(value):
    return f"{float(value):.6f}"


def write_report(rows, policy, package, path):
    lines = [
        "# Stage133 Final Predictor RD Report",
        "",
        "## Final Recommendation",
        "",
        f"- Deployable policy: `{policy['policy_name']}`.",
        f"- Primary deployable setting: `{policy['selected_primary_setting']}`.",
        f"- Optional low-rate setting: `{policy['optional_low_rate_setting']}`.",
        "- Teacher residual side-info is retained only as a quality reference.",
        "- Dedicated MLP predictor is rejected until render-aware training fixes PSNR regression.",
        "",
        "## RD Table",
        "",
        "| role | method | setting | deployable | teacher | rate | PSNR | delta base | residual bytes | index bytes |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['final_role']} | {row['method_label']} | {row['setting_label']} | {row['deployable_no_teacher']} | {row['teacher_required']} | {format_float(row['direct_total_mib_per_frame'])} | {format_float(row['psnr'])} | {format_float(row['delta_psnr_vs_linear_base'])} | {row['residual_payload_bytes']} | {row['selected_index_payload_bytes']} |"
        )
    lines.extend([
        "",
        "## Plot",
        "",
        f"- PNG: `{package['rd_plot_png']}`",
        f"- PDF: `{package['rd_plot_pdf']}`",
        "",
        "## Outputs",
        "",
        f"- final rows CSV: `{package['final_rows_csv']}`",
        f"- package JSON: `{package['package_json']}`",
        f"- report Markdown: `{package['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage130_rows", type=Path, default=DEFAULT_STAGE130_ROWS)
    parser.add_argument("--stage132_policy", type=Path, default=DEFAULT_STAGE132_POLICY)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    policy = read_json(args.stage132_policy)
    rows = build_rows(read_csv(args.stage130_rows), policy)
    final_rows_csv = args.summary_root / "stage133_final_predictor_rd_rows.csv"
    package_json = args.summary_root / "stage133_final_predictor_rd_package.json"
    report_md = args.summary_root / "stage133_final_predictor_rd_report.md"
    rd_plot_png = args.summary_root / "stage133_final_predictor_rd_plot.png"
    rd_plot_pdf = args.summary_root / "stage133_final_predictor_rd_plot.pdf"
    write_csv(rows, final_rows_csv, FINAL_FIELDS)
    plot_rd(rows, rd_plot_png, rd_plot_pdf)
    package = {
        "stage": 133,
        "mode": "final predictor RD report",
        "stage130_rows": str(args.stage130_rows),
        "stage132_policy": str(args.stage132_policy),
        "final_rows_csv": str(final_rows_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "rd_plot_png": str(rd_plot_png),
        "rd_plot_pdf": str(rd_plot_pdf),
        "row_count": len(rows),
        "final_deployable_policy": policy["policy_name"],
        "primary_setting": policy["selected_primary_setting"],
        "optional_low_rate_setting": policy["optional_low_rate_setting"],
        "notes": [
            "Teacher residual side-info is reference-only and not decoder-only deployable.",
            "Final deployable policy uses adapter-delta selected residual values with no per-frame residual/index payload.",
            "Dedicated MLP predictor is rejected due to Stage129 rendered regression.",
        ],
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(rows, policy, package, report_md)
    print(json.dumps({"package": str(package_json), "row_count": len(rows), "plot": str(rd_plot_png)}, indent=2))


if __name__ == "__main__":
    main()
