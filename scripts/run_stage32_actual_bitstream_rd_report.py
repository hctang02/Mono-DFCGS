import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE26_CSV = REPO_ROOT / "experiments/stage26_leave_one_out_full_video_rd/stage26_leave_one_out_full_video_rd.csv"
DEFAULT_STAGE31_CSV = REPO_ROOT / "experiments/stage31_anchor_bitstream_prototype/stage31_anchor_bitstream_prototype.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage32_actual_bitstream_rd_report"


def load_stage26(path):
    rows = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["sample"], int(row["frame_gap"]))
            rows[key] = {
                "sample": row["sample"],
                "frame_gap": int(row["frame_gap"]),
                "estimated_q8_static_mib_per_frame": float(row["estimated_q8_static_mib_per_frame"]),
                "total_frames": int(row["total_frames"]),
                "keyframe_count": int(row["keyframe_count"]),
                "linear_all_psnr": float(row["linear_all_psnr"]),
                "adapter_all_psnr": float(row["adapter_all_psnr"]),
                "delta_all_psnr": float(row["delta_all_psnr"]),
                "linear_middle_psnr": float(row["linear_middle_psnr"]),
                "adapter_middle_psnr": float(row["adapter_middle_psnr"]),
                "delta_middle_psnr": float(row["delta_middle_psnr"]),
                "linear_all_ssim": float(row["linear_all_ssim"]),
                "adapter_all_ssim": float(row["adapter_all_ssim"]),
                "linear_middle_ssim": float(row["linear_middle_ssim"]),
                "adapter_middle_ssim": float(row["adapter_middle_ssim"]),
            }
    return rows


def load_stage31(path):
    rows = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["sample"], int(row["frame_gap"]))
            rows[key] = {
                "raw_bitstream_bytes": int(row["raw_bitstream_bytes"]),
                "zlib_bitstream_bytes": int(row["zlib_bitstream_bytes"]),
                "raw_mib_per_frame": float(row["raw_mib_per_frame"]),
                "zlib_mib_per_frame": float(row["zlib_mib_per_frame"]),
                "raw_overhead_bytes_vs_stage26_anchor": int(row["raw_overhead_bytes_vs_stage26_anchor"]),
                "zlib_savings_percent_vs_raw_bitstream": float(row["zlib_savings_percent_vs_raw_bitstream"]),
                "max_roundtrip_abs_diff": float(row["max_roundtrip_abs_diff"]),
                "mean_roundtrip_mse": float(row["mean_roundtrip_mse"]),
            }
    return rows


def merge_rows(stage26_rows, stage31_rows):
    rows = []
    for key in sorted(stage26_rows):
        if key not in stage31_rows:
            raise RuntimeError(f"Missing Stage31 bitstream row for {key}")
        row = {**stage26_rows[key], **stage31_rows[key]}
        row["raw_rate_over_estimated_q8_ratio"] = row["raw_mib_per_frame"] / row["estimated_q8_static_mib_per_frame"]
        row["zlib_rate_over_estimated_q8_ratio"] = row["zlib_mib_per_frame"] / row["estimated_q8_static_mib_per_frame"]
        rows.append(row)
    return rows


def write_csv(rows, path):
    fields = [
        "sample", "frame_gap", "total_frames", "keyframe_count", "estimated_q8_static_mib_per_frame",
        "raw_mib_per_frame", "zlib_mib_per_frame", "raw_rate_over_estimated_q8_ratio",
        "zlib_rate_over_estimated_q8_ratio", "raw_bitstream_bytes", "zlib_bitstream_bytes",
        "raw_overhead_bytes_vs_stage26_anchor", "zlib_savings_percent_vs_raw_bitstream",
        "linear_all_psnr", "adapter_all_psnr", "delta_all_psnr", "linear_middle_psnr", "adapter_middle_psnr",
        "delta_middle_psnr", "linear_all_ssim", "adapter_all_ssim", "linear_middle_ssim", "adapter_middle_ssim",
        "max_roundtrip_abs_diff", "mean_roundtrip_mse",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by_gap(rows):
    out = []
    for gap in sorted({row["frame_gap"] for row in rows}):
        points = [row for row in rows if row["frame_gap"] == gap]
        out.append({
            "frame_gap": gap,
            "estimated_q8_static_mib_per_frame": float(np.mean([row["estimated_q8_static_mib_per_frame"] for row in points])),
            "raw_mib_per_frame": float(np.mean([row["raw_mib_per_frame"] for row in points])),
            "zlib_mib_per_frame": float(np.mean([row["zlib_mib_per_frame"] for row in points])),
            "linear_all_psnr": float(np.mean([row["linear_all_psnr"] for row in points])),
            "adapter_all_psnr": float(np.mean([row["adapter_all_psnr"] for row in points])),
            "linear_middle_psnr": float(np.mean([row["linear_middle_psnr"] for row in points])),
            "adapter_middle_psnr": float(np.mean([row["adapter_middle_psnr"] for row in points])),
            "linear_all_ssim": float(np.mean([row["linear_all_ssim"] for row in points])),
            "adapter_all_ssim": float(np.mean([row["adapter_all_ssim"] for row in points])),
            "linear_middle_ssim": float(np.mean([row["linear_middle_ssim"] for row in points])),
            "adapter_middle_ssim": float(np.mean([row["adapter_middle_ssim"] for row in points])),
        })
    return out


def write_gap_csv(rows, path):
    fields = [
        "frame_gap", "estimated_q8_static_mib_per_frame", "raw_mib_per_frame", "zlib_mib_per_frame",
        "linear_all_psnr", "adapter_all_psnr", "linear_middle_psnr", "adapter_middle_psnr",
        "linear_all_ssim", "adapter_all_ssim", "linear_middle_ssim", "adapter_middle_ssim",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_per_sample(rows, rate_key, linear_key, adapter_key, ylabel, title, out_path):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sample"]].append(row)
    samples = sorted(grouped)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharey=False)
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        points = sorted(grouped[sample], key=lambda row: row[rate_key])
        xs = [row[rate_key] for row in points]
        linear = [row[linear_key] for row in points]
        adapter = [row[adapter_key] for row in points]
        labels = [f"g{row['frame_gap']}" for row in points]
        ax.plot(xs, linear, marker="o", linewidth=2.0, label="q8 linear", color="#1f77b4")
        ax.plot(xs, adapter, marker="^", linewidth=2.0, label="Stage25 held-out adapter", color="#d62728")
        for x, y, label in zip(xs, adapter, labels):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_title(sample)
        ax.set_xlabel(rate_label(rate_key))
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(frameon=False)
    axes[0].set_ylabel(ylabel)
    axes[2].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_mean(rows, rate_key, linear_key, adapter_key, ylabel, title, out_path):
    points = sorted(rows, key=lambda row: row[rate_key])
    xs = [row[rate_key] for row in points]
    linear = [row[linear_key] for row in points]
    adapter = [row[adapter_key] for row in points]
    labels = [f"g{row['frame_gap']}" for row in points]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(xs, linear, marker="o", linewidth=2.0, label="q8 linear", color="#1f77b4")
    ax.plot(xs, adapter, marker="^", linewidth=2.0, label="Stage25 held-out adapter", color="#d62728")
    for x, y, label in zip(xs, adapter, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel(rate_label(rate_key))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def rate_label(rate_key):
    if rate_key == "raw_mib_per_frame":
        return "Raw q8 bitstream MiB/frame"
    if rate_key == "zlib_mib_per_frame":
        return "Zlib q8 bitstream MiB/frame"
    return "Estimated q8 anchor MiB/frame"


def make_plots(rows, gap_rows, out_dir):
    plots = []
    for rate_key, rate_name in [("raw_mib_per_frame", "raw"), ("zlib_mib_per_frame", "zlib")]:
        specs = [
            ("linear_all_psnr", "adapter_all_psnr", "All-frame PSNR (dB)", f"Stage 32 {rate_name.upper()} Bitstream RD: All PSNR", f"stage32_{rate_name}_all_psnr_rd.png"),
            ("linear_middle_psnr", "adapter_middle_psnr", "Middle-only PSNR (dB)", f"Stage 32 {rate_name.upper()} Bitstream RD: Middle PSNR", f"stage32_{rate_name}_middle_psnr_rd.png"),
            ("linear_all_ssim", "adapter_all_ssim", "All-frame SSIM", f"Stage 32 {rate_name.upper()} Bitstream RD: All SSIM", f"stage32_{rate_name}_all_ssim_rd.png"),
            ("linear_middle_ssim", "adapter_middle_ssim", "Middle-only SSIM", f"Stage 32 {rate_name.upper()} Bitstream RD: Middle SSIM", f"stage32_{rate_name}_middle_ssim_rd.png"),
        ]
        for linear_key, adapter_key, ylabel, title, filename in specs:
            out_path = out_dir / filename
            plot_per_sample(rows, rate_key, linear_key, adapter_key, ylabel, title, out_path)
            plots.append(str(out_path))
        for linear_key, adapter_key, ylabel, title, filename in specs:
            out_path = out_dir / filename.replace("stage32_", "stage32_mean_")
            plot_mean(gap_rows, rate_key, linear_key, adapter_key, ylabel.replace("All-frame", "Mean all-frame").replace("Middle-only", "Mean middle-only"), title.replace("Stage 32", "Stage 32 Mean"), out_path)
            plots.append(str(out_path))
    return plots


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage26_csv", type=Path, default=DEFAULT_STAGE26_CSV)
    parser.add_argument("--stage31_csv", type=Path, default=DEFAULT_STAGE31_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    rows = merge_rows(load_stage26(args.stage26_csv), load_stage31(args.stage31_csv))
    gap_rows = aggregate_by_gap(rows)
    csv_path = args.summary_root / "stage32_actual_bitstream_rd.csv"
    gap_csv = args.summary_root / "stage32_gap_aggregate.csv"
    summary_path = args.summary_root / "stage32_actual_bitstream_rd_summary.json"
    write_csv(rows, csv_path)
    write_gap_csv(gap_rows, gap_csv)
    plots = make_plots(rows, gap_rows, args.summary_root)
    summary = {
        "stage": 32,
        "mode": "actual q8 anchor bitstream RD report",
        "stage26_csv": str(args.stage26_csv),
        "stage31_csv": str(args.stage31_csv),
        "csv": str(csv_path),
        "gap_aggregate_csv": str(gap_csv),
        "plots": plots,
        "rows": rows,
        "gap_aggregate": gap_rows,
        "mean_delta_all_psnr": float(np.mean([row["delta_all_psnr"] for row in rows])),
        "mean_delta_middle_psnr": float(np.mean([row["delta_middle_psnr"] for row in rows])),
        "mean_raw_rate_over_estimated_q8_ratio": float(np.mean([row["raw_rate_over_estimated_q8_ratio"] for row in rows])),
        "mean_zlib_rate_over_estimated_q8_ratio": float(np.mean([row["zlib_rate_over_estimated_q8_ratio"] for row in rows])),
        "mean_zlib_savings_percent_vs_raw_bitstream": float(np.mean([row["zlib_savings_percent_vs_raw_bitstream"] for row in rows])),
        "max_roundtrip_abs_diff": float(max(row["max_roundtrip_abs_diff"] for row in rows)),
        "notes": "Uses Stage31 actual bitstream sizes with Stage26 held-out full-video quality. Zlib is a generic compression baseline.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "gap_aggregate_csv": str(gap_csv),
        "rows": len(rows),
        "mean_delta_all_psnr": summary["mean_delta_all_psnr"],
        "mean_delta_middle_psnr": summary["mean_delta_middle_psnr"],
        "mean_zlib_rate_over_estimated_q8_ratio": summary["mean_zlib_rate_over_estimated_q8_ratio"],
        "mean_zlib_savings_percent_vs_raw_bitstream": summary["mean_zlib_savings_percent_vs_raw_bitstream"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
