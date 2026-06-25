import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE23_CSV = REPO_ROOT / "experiments/stage23_full_video_anchor_only_evaluator/stage23_full_video_anchor_only_evaluator.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage24_full_video_anchor_only_rd_plots"


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "sample": row["sample"],
                "frame_gap": int(row["frame_gap"]),
                "rate": float(row["estimated_q8_static_mib_per_frame"]),
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
            })
    return rows


def plot_metric(rows, linear_key, adapter_key, ylabel, title, out_path):
    samples = sorted({row["sample"] for row in rows})
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sample"]].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharey=False)
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        points = sorted(grouped[sample], key=lambda row: row["rate"])
        xs = [row["rate"] for row in points]
        linear = [row[linear_key] for row in points]
        adapter = [row[adapter_key] for row in points]
        labels = [f"g{row['frame_gap']}" for row in points]
        ax.plot(xs, linear, marker="o", linewidth=2.0, label="q8 linear", color="#1f77b4")
        ax.plot(xs, adapter, marker="^", linewidth=2.0, label="Stage21d adapter", color="#d62728")
        for x, y, label in zip(xs, adapter, labels):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_title(sample)
        ax.set_xlabel("Transmitted q8 static Gaussian MiB/frame")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(frameon=False)
    axes[0].set_ylabel(ylabel)
    axes[2].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_delta(rows, delta_key, ylabel, title, out_path):
    samples = sorted({row["sample"] for row in rows})
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sample"]].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharey=True)
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        points = sorted(grouped[sample], key=lambda row: row["frame_gap"])
        gaps = [str(row["frame_gap"]) for row in points]
        deltas = [row[delta_key] for row in points]
        ax.bar(gaps, deltas, color="#2ca02c")
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(sample)
        ax.set_xlabel("GOP gap")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    axes[0].set_ylabel(ylabel)
    axes[2].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def aggregate(rows):
    samples = sorted({row["sample"] for row in rows})
    out = []
    for sample in samples:
        points = [row for row in rows if row["sample"] == sample]
        out.append({
            "sample": sample,
            "mean_delta_all_psnr": sum(row["delta_all_psnr"] for row in points) / len(points),
            "mean_delta_middle_psnr": sum(row["delta_middle_psnr"] for row in points) / len(points),
            "min_delta_all_psnr": min(row["delta_all_psnr"] for row in points),
            "min_delta_middle_psnr": min(row["delta_middle_psnr"] for row in points),
        })
    return out


def write_aggregate_csv(rows, path):
    fields = ["sample", "mean_delta_all_psnr", "mean_delta_middle_psnr", "min_delta_all_psnr", "min_delta_middle_psnr"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage23_csv", type=Path, default=DEFAULT_STAGE23_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.stage23_csv)
    aggregate_rows = aggregate(rows)
    aggregate_csv = args.summary_root / "stage24_full_video_anchor_only_rd_aggregate.csv"
    write_aggregate_csv(aggregate_rows, aggregate_csv)

    plots = []
    specs = [
        ("linear_all_psnr", "adapter_all_psnr", "All-frame PSNR (dB)", "Stage 24 Full-Video Anchor-Only RD: All PSNR", "stage24_full_video_all_psnr_rd.png"),
        ("linear_middle_psnr", "adapter_middle_psnr", "Middle-only PSNR (dB)", "Stage 24 Full-Video Anchor-Only RD: Middle PSNR", "stage24_full_video_middle_psnr_rd.png"),
        ("linear_all_ssim", "adapter_all_ssim", "All-frame SSIM", "Stage 24 Full-Video Anchor-Only RD: All SSIM", "stage24_full_video_all_ssim_rd.png"),
        ("linear_middle_ssim", "adapter_middle_ssim", "Middle-only SSIM", "Stage 24 Full-Video Anchor-Only RD: Middle SSIM", "stage24_full_video_middle_ssim_rd.png"),
    ]
    for linear_key, adapter_key, ylabel, title, filename in specs:
        out_path = args.summary_root / filename
        plot_metric(rows, linear_key, adapter_key, ylabel, title, out_path)
        plots.append(str(out_path))
    for delta_key, ylabel, title, filename in [
        ("delta_all_psnr", "Adapter - linear all PSNR (dB)", "Stage 24 Full-Video Anchor Adapter Gain: All PSNR", "stage24_full_video_delta_all_psnr.png"),
        ("delta_middle_psnr", "Adapter - linear middle PSNR (dB)", "Stage 24 Full-Video Anchor Adapter Gain: Middle PSNR", "stage24_full_video_delta_middle_psnr.png"),
    ]:
        out_path = args.summary_root / filename
        plot_delta(rows, delta_key, ylabel, title, out_path)
        plots.append(str(out_path))

    summary_path = args.summary_root / "stage24_full_video_anchor_only_rd_plots_summary.json"
    summary = {
        "stage": 24,
        "mode": "full-video anchor-only RD plots",
        "source_stage23_csv": str(args.stage23_csv),
        "rows": len(rows),
        "aggregate_csv": str(aggregate_csv),
        "aggregate": aggregate_rows,
        "plots": plots,
        "mean_delta_all_psnr": sum(row["delta_all_psnr"] for row in rows) / len(rows),
        "mean_delta_middle_psnr": sum(row["delta_middle_psnr"] for row in rows) / len(rows),
        "notes": "Plots Stage23 full-video anchor-only metrics. This is not original StreamSplat RGB/depth-conditioned RD.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "aggregate_csv": str(aggregate_csv),
        "plots": plots,
        "rows": len(rows),
        "mean_delta_all_psnr": summary["mean_delta_all_psnr"],
        "mean_delta_middle_psnr": summary["mean_delta_middle_psnr"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
