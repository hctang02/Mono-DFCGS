import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE68_RESULT_CSV = REPO_ROOT / "experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_rendered_validation.csv"
DEFAULT_STAGE68_COMPARISON_CSV = REPO_ROOT / "experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_comparison.csv"
DEFAULT_STAGE69_POLICY_CSV = REPO_ROOT / "experiments/stage69_selector_fallback_calibration/stage69_selector_policy_summary.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage70_scoped_davis_rd_package"


RATE_FIELDS = [
    "sample", "reference_gap", "selector", "total_frames", "keyframe_count", "keyframe_ratio",
    "q8_static_anchor_mib_per_frame",
]

PSNR_FIELDS = [
    "sample", "reference_gap", "method", "selector", "q8_static_anchor_mib_per_frame", "all_psnr",
]

SUMMARY_FIELDS = [
    "method", "selector", "reference_gap", "point_count", "mean_rate_mib_per_frame", "mean_all_psnr",
]

BASELINE_FIELDS = ["method", "status", "rate_status", "all_psnr_status", "notes"]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def to_float(row, key):
    return float(row[key])


def to_int(row, key):
    return int(float(row[key]))


def build_rate_rows(result_rows):
    rows = []
    seen = set()
    for row in result_rows:
        key = (row["sample"], int(row["reference_gap"]), row["method"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "sample": row["sample"],
            "reference_gap": int(row["reference_gap"]),
            "selector": row["method"],
            "total_frames": to_int(row, "total_frames"),
            "keyframe_count": to_int(row, "keyframe_count"),
            "keyframe_ratio": to_float(row, "keyframe_ratio"),
            "q8_static_anchor_mib_per_frame": to_float(row, "estimated_q8_static_mib_per_frame"),
        })
    return sorted(rows, key=lambda r: (r["sample"], r["reference_gap"], r["selector"]))


def build_psnr_rows(result_rows):
    out = []
    for row in result_rows:
        selector = row["method"]
        gap = int(row["reference_gap"])
        rate = to_float(row, "estimated_q8_static_mib_per_frame")
        out.append({
            "sample": row["sample"],
            "reference_gap": gap,
            "method": "linear_anchor",
            "selector": selector,
            "q8_static_anchor_mib_per_frame": rate,
            "all_psnr": to_float(row, "linear_all_psnr"),
        })
        out.append({
            "sample": row["sample"],
            "reference_gap": gap,
            "method": "stage65_rgb_h256_adapter",
            "selector": selector,
            "q8_static_anchor_mib_per_frame": rate,
            "all_psnr": to_float(row, "adapter_all_psnr"),
        })
    return sorted(out, key=lambda r: (r["sample"], r["method"], r["selector"], r["reference_gap"]))


def build_method_summary(psnr_rows):
    grouped = defaultdict(list)
    for row in psnr_rows:
        grouped[(row["method"], row["selector"], row["reference_gap"])].append(row)
    out = []
    for (method, selector, gap), rows in sorted(grouped.items()):
        out.append({
            "method": method,
            "selector": selector,
            "reference_gap": gap,
            "point_count": len(rows),
            "mean_rate_mib_per_frame": float(np.mean([row["q8_static_anchor_mib_per_frame"] for row in rows])),
            "mean_all_psnr": float(np.mean([row["all_psnr"] for row in rows])),
        })
    return out


def baseline_status_rows():
    return [
        {
            "method": "FCGS",
            "status": "not_run_locally",
            "rate_status": "missing apples-to-apples transmitted anchor/bitstream rate",
            "all_psnr_status": "missing DAVIS eval-subset rendered all-frame PSNR",
            "notes": "Add local fair baseline before final paper-level RD claims.",
        },
        {
            "method": "D-FCGS",
            "status": "not_run_locally",
            "rate_status": "missing apples-to-apples transmitted anchor/bitstream rate",
            "all_psnr_status": "missing DAVIS eval-subset rendered all-frame PSNR",
            "notes": "Primary baseline still required for final comparison.",
        },
        {
            "method": "CWGS",
            "status": "optional_not_run_locally",
            "rate_status": "missing local rate",
            "all_psnr_status": "missing local all-frame PSNR",
            "notes": "Optional supplemental baseline after FCGS/D-FCGS.",
        },
    ]


def plot_rd(psnr_rows, out_path):
    samples = sorted({row["sample"] for row in psnr_rows})
    plot_specs = [
        ("linear_anchor", "uniform", "Linear + uniform", "#7f7f7f"),
        ("stage65_rgb_h256_adapter", "uniform", "Adapter + uniform", "#1f77b4"),
        ("stage65_rgb_h256_adapter", "predicted_full_feature_dp", "Adapter + predicted selector", "#2ca02c"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        for method, selector, label, color in plot_specs:
            points = [row for row in psnr_rows if row["sample"] == sample and row["method"] == method and row["selector"] == selector]
            points = sorted(points, key=lambda row: row["q8_static_anchor_mib_per_frame"])
            if not points:
                continue
            ax.plot(
                [row["q8_static_anchor_mib_per_frame"] for row in points],
                [row["all_psnr"] for row in points],
                marker="o",
                linewidth=2.0,
                color=color,
                label=label,
            )
        ax.set_title(sample)
        ax.set_xlabel("q8 static anchor MiB/frame")
        ax.set_ylabel("All-frame PSNR (dB)")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage68_result_csv", type=Path, default=DEFAULT_STAGE68_RESULT_CSV)
    parser.add_argument("--stage68_comparison_csv", type=Path, default=DEFAULT_STAGE68_COMPARISON_CSV)
    parser.add_argument("--stage69_policy_csv", type=Path, default=DEFAULT_STAGE69_POLICY_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    result_rows = read_csv(args.stage68_result_csv)
    comparison_rows = read_csv(args.stage68_comparison_csv)
    policy_rows = read_csv(args.stage69_policy_csv)
    rate_rows = build_rate_rows(result_rows)
    psnr_rows = build_psnr_rows(result_rows)
    method_summary = build_method_summary(psnr_rows)
    baseline_rows = baseline_status_rows()

    rate_csv = args.summary_root / "stage70_rate_table.csv"
    psnr_csv = args.summary_root / "stage70_all_psnr_table.csv"
    selector_delta_csv = args.summary_root / "stage70_selector_delta_table.csv"
    method_summary_csv = args.summary_root / "stage70_method_summary.csv"
    baseline_csv = args.summary_root / "stage70_baseline_status.csv"
    rd_plot = args.summary_root / "stage70_scoped_davis_rd_curve.png"
    summary_json = args.summary_root / "stage70_scoped_davis_rd_package_summary.json"
    write_csv(rate_rows, rate_csv, RATE_FIELDS)
    write_csv(psnr_rows, psnr_csv, PSNR_FIELDS)
    write_csv(comparison_rows, selector_delta_csv, list(comparison_rows[0].keys()))
    write_csv(method_summary, method_summary_csv, SUMMARY_FIELDS)
    write_csv(baseline_rows, baseline_csv, BASELINE_FIELDS)
    plot_rd(psnr_rows, rd_plot)

    selector_deltas = [float(row["selector_delta_adapter_all_psnr"]) for row in comparison_rows]
    summary = {
        "stage": 70,
        "mode": "scoped DAVIS RD package from Stage68/69 outputs",
        "stage68_result_csv": str(args.stage68_result_csv),
        "stage68_comparison_csv": str(args.stage68_comparison_csv),
        "stage69_policy_csv": str(args.stage69_policy_csv),
        "rate_csv": str(rate_csv),
        "all_psnr_csv": str(psnr_csv),
        "selector_delta_csv": str(selector_delta_csv),
        "method_summary_csv": str(method_summary_csv),
        "baseline_status_csv": str(baseline_csv),
        "rd_plot": str(rd_plot),
        "samples": sorted({row["sample"] for row in result_rows}),
        "gaps": sorted({int(row["reference_gap"]) for row in result_rows}),
        "method_summary": method_summary,
        "selector_mean_delta_adapter_all_psnr": float(np.mean(selector_deltas)),
        "selector_positive_points": sum(1 for value in selector_deltas if value > 0.0),
        "selector_point_count": len(selector_deltas),
        "policy_summary": policy_rows,
        "baseline_status": baseline_rows,
        "notes": [
            "This is a scoped DAVIS eval-subset package, not the final full benchmark.",
            "Default quality metric is all-frame PSNR.",
            "Rate is q8 static keyframe-anchor MiB/frame; decoder/model weights are not counted per-video.",
            "FCGS/D-FCGS apples-to-apples local baselines are still missing.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_json),
        "rate_csv": str(rate_csv),
        "all_psnr_csv": str(psnr_csv),
        "rd_plot": str(rd_plot),
        "selector_mean_delta_adapter_all_psnr": summary["selector_mean_delta_adapter_all_psnr"],
        "selector_positive_points": summary["selector_positive_points"],
        "selector_point_count": summary["selector_point_count"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
