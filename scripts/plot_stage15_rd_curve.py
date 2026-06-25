import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "experiments/stage15_selected_keyframe_rd_curve/stage15_selected_keyframe_rd_curve_summary.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/stage15_selected_keyframe_rd_curve"


def read_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in [
                "reference_gap",
                "estimated_q8_static_mib_per_frame",
                "all_psnr_avg",
                "middle_psnr_avg",
                "all_ssim_avg",
                "middle_ssim_avg",
            ]:
                row[key] = float(row[key])
            rows.append(row)
    return rows


def plot_metric(rows, metric_key, ylabel, title, out_path):
    grouped = defaultdict(list)
    samples = sorted({row["sample"] for row in rows})
    methods = ["uniform", "rd_spaced"]
    colors = {"uniform": "#1f77b4", "rd_spaced": "#d62728"}
    markers = {"uniform": "o", "rd_spaced": "s"}

    for row in rows:
        grouped[(row["sample"], row["selection_method"])].append(row)

    fig, axes = plt.subplots(1, len(samples), figsize=(6.2 * len(samples), 4.8), sharey=False)
    if len(samples) == 1:
        axes = [axes]

    for ax, sample in zip(axes, samples):
        for method in methods:
            points = sorted(grouped[(sample, method)], key=lambda r: r["estimated_q8_static_mib_per_frame"])
            if not points:
                continue
            xs = [row["estimated_q8_static_mib_per_frame"] for row in points]
            ys = [row[metric_key] for row in points]
            labels = [f"g{int(row['reference_gap'])}" for row in points]
            ax.plot(xs, ys, marker=markers[method], color=colors[method], linewidth=2.0, label=method)
            for x, y, label in zip(xs, ys, labels):
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_title(sample)
        ax.set_xlabel("Transmitted Gaussian MiB/frame")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(frameon=False)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.csv)
    outputs = []
    specs = [
        ("all_psnr_avg", "All-frame PSNR (dB)", "Stage 15 RD Curve: All-frame PSNR", "stage15_rd_curve_all_psnr.png"),
        ("middle_psnr_avg", "Middle-frame PSNR (dB)", "Stage 15 RD Curve: Middle-only PSNR", "stage15_rd_curve_middle_psnr.png"),
        ("all_ssim_avg", "All-frame SSIM", "Stage 15 RD Curve: All-frame SSIM", "stage15_rd_curve_all_ssim.png"),
        ("middle_ssim_avg", "Middle-frame SSIM", "Stage 15 RD Curve: Middle-only SSIM", "stage15_rd_curve_middle_ssim.png"),
    ]
    for metric_key, ylabel, title, filename in specs:
        out_path = args.out_dir / filename
        plot_metric(rows, metric_key, ylabel, title, out_path)
        outputs.append(str(out_path))
    print("\n".join(outputs))


if __name__ == "__main__":
    raise SystemExit(main())
