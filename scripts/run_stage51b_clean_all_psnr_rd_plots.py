import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE51_CSV = REPO_ROOT / "experiments/stage51_high_rate_multibit_rd/stage51_high_rate_multibit_rd.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage51_high_rate_multibit_rd"

BIT_COLORS = {
    8: "#1f77b4",
    10: "#ff7f0e",
    12: "#2ca02c",
    16: "#d62728",
}

METHOD_LABELS = {
    "uniform": "Uniform",
    "rendered_prior_0p1": "Adaptive rendered_prior_0p1",
}

MEAN_FIELDS = [
    "method",
    "bits",
    "reference_gap",
    "sample_count",
    "mean_zlib_mib_per_frame",
    "mean_all_psnr",
    "min_all_psnr",
    "max_all_psnr",
]

DELTA_FIELDS = [
    "bits",
    "reference_gap",
    "sample_count",
    "mean_rate_delta_mib_per_frame",
    "mean_delta_all_psnr",
    "min_delta_all_psnr",
    "positive_delta_count",
]


def read_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "sample": row["sample"],
                "method": row["method"],
                "reference_gap": int(row["reference_gap"]),
                "bits": int(row["bits"]),
                "zlib_mib_per_frame": float(row["zlib_mib_per_frame"]),
                "adapter_all_psnr": float(row["adapter_all_psnr"]),
            })
    return rows


def write_csv(rows, fields, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_mean_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["bits"], row["reference_gap"])].append(row)
    out = []
    for (method, bits, gap), group in sorted(grouped.items()):
        rates = [row["zlib_mib_per_frame"] for row in group]
        psnrs = [row["adapter_all_psnr"] for row in group]
        out.append({
            "method": method,
            "bits": bits,
            "reference_gap": gap,
            "sample_count": len(group),
            "mean_zlib_mib_per_frame": float(np.mean(rates)),
            "mean_all_psnr": float(np.mean(psnrs)),
            "min_all_psnr": float(np.min(psnrs)),
            "max_all_psnr": float(np.max(psnrs)),
        })
    return out


def aggregate_delta_rows(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["sample"], row["bits"], row["reference_gap"])][row["method"]] = row
    by_point = defaultdict(list)
    for (_sample, bits, gap), methods in grouped.items():
        uniform = methods.get("uniform")
        adaptive = methods.get("rendered_prior_0p1")
        if uniform is None or adaptive is None:
            continue
        by_point[(bits, gap)].append({
            "rate_delta": adaptive["zlib_mib_per_frame"] - uniform["zlib_mib_per_frame"],
            "psnr_delta": adaptive["adapter_all_psnr"] - uniform["adapter_all_psnr"],
        })
    out = []
    for (bits, gap), group in sorted(by_point.items()):
        deltas = [row["psnr_delta"] for row in group]
        rate_deltas = [row["rate_delta"] for row in group]
        out.append({
            "bits": bits,
            "reference_gap": gap,
            "sample_count": len(group),
            "mean_rate_delta_mib_per_frame": float(np.mean(rate_deltas)),
            "mean_delta_all_psnr": float(np.mean(deltas)),
            "min_delta_all_psnr": float(np.min(deltas)),
            "positive_delta_count": sum(1 for value in deltas if value > 0.0),
        })
    return out


def rows_for(rows, sample=None, method=None, bits=None):
    out = rows
    if sample is not None:
        out = [row for row in out if row["sample"] == sample]
    if method is not None:
        out = [row for row in out if row["method"] == method]
    if bits is not None:
        out = [row for row in out if row["bits"] == bits]
    return out


def plot_sample_rd_by_bits(rows, method, out_path):
    samples = sorted({row["sample"] for row in rows})
    bits_values = sorted({row["bits"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), sharex=False, sharey=False)
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        for bits in bits_values:
            points = sorted(rows_for(rows, sample=sample, method=method, bits=bits), key=lambda row: row["zlib_mib_per_frame"])
            if not points:
                continue
            xs = [row["zlib_mib_per_frame"] for row in points]
            ys = [row["adapter_all_psnr"] for row in points]
            color = BIT_COLORS.get(bits)
            ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4.8, color=color, label=f"q{bits}")
            for row in points:
                ax.annotate(
                    f"g{row['reference_gap']}",
                    (row["zlib_mib_per_frame"], row["adapter_all_psnr"]),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=6,
                    color=color,
                )
        ax.set_title(sample)
        ax.set_xlabel("Zlib q-anchor MiB/frame")
        ax.set_ylabel("All-frame PSNR (dB)")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(title="Quant bits", fontsize=7, title_fontsize=8)
    title = METHOD_LABELS.get(method, method)
    fig.suptitle(f"Stage51 Clean RD: {title}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_mean_rd_by_bits(mean_rows, out_path):
    methods = ["uniform", "rendered_prior_0p1"]
    bits_values = sorted({int(row["bits"]) for row in mean_rows})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), sharex=False, sharey=True)
    for ax, method in zip(axes, methods):
        for bits in bits_values:
            points = sorted(
                [row for row in mean_rows if row["method"] == method and int(row["bits"]) == bits],
                key=lambda row: row["mean_zlib_mib_per_frame"],
            )
            if not points:
                continue
            xs = [float(row["mean_zlib_mib_per_frame"]) for row in points]
            ys = [float(row["mean_all_psnr"]) for row in points]
            color = BIT_COLORS.get(bits)
            ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=5.0, color=color, label=f"q{bits}")
            for row in points:
                ax.annotate(
                    f"g{row['reference_gap']}",
                    (float(row["mean_zlib_mib_per_frame"]), float(row["mean_all_psnr"])),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=6,
                    color=color,
                )
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xlabel("Mean zlib q-anchor MiB/frame")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(title="Quant bits", fontsize=7, title_fontsize=8)
    axes[0].set_ylabel("Mean all-frame PSNR (dB)")
    fig.suptitle("Stage51 Clean Mean RD Across Samples", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_delta_heatmap(delta_rows, out_path):
    bits_values = sorted({int(row["bits"]) for row in delta_rows})
    gap_values = sorted({int(row["reference_gap"]) for row in delta_rows})
    matrix = np.full((len(bits_values), len(gap_values)), np.nan, dtype=np.float32)
    value_map = {(int(row["bits"]), int(row["reference_gap"])): float(row["mean_delta_all_psnr"]) for row in delta_rows}
    for i, bits in enumerate(bits_values):
        for j, gap in enumerate(gap_values):
            matrix[i, j] = value_map.get((bits, gap), np.nan)
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    vmax = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix)))) if np.isfinite(matrix).any() else 1.0
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(gap_values)), [f"g{gap}" for gap in gap_values])
    ax.set_yticks(range(len(bits_values)), [f"q{bits}" for bits in bits_values])
    ax.set_xlabel("Reference gap")
    ax.set_ylabel("Quant bits")
    ax.set_title("Adaptive - Uniform mean all-frame PSNR delta (dB)")
    for i, bits in enumerate(bits_values):
        for j, gap in enumerate(gap_values):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Delta PSNR (dB)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage51_csv", type=Path, default=DEFAULT_STAGE51_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.stage51_csv)
    mean_rows = aggregate_mean_rows(rows)
    delta_rows = aggregate_delta_rows(rows)

    mean_csv = args.summary_root / "stage51_clean_mean_all_psnr_rd.csv"
    delta_csv = args.summary_root / "stage51_clean_adaptive_delta_all_psnr.csv"
    adaptive_plot = args.summary_root / "stage51_clean_adaptive_all_psnr_by_bits.png"
    uniform_plot = args.summary_root / "stage51_clean_uniform_all_psnr_by_bits.png"
    mean_plot = args.summary_root / "stage51_clean_mean_all_psnr_by_bits.png"
    delta_plot = args.summary_root / "stage51_clean_adaptive_delta_all_psnr_heatmap.png"
    summary_json = args.summary_root / "stage51_clean_all_psnr_rd_plots_summary.json"

    write_csv(mean_rows, MEAN_FIELDS, mean_csv)
    write_csv(delta_rows, DELTA_FIELDS, delta_csv)
    plot_sample_rd_by_bits(rows, "rendered_prior_0p1", adaptive_plot)
    plot_sample_rd_by_bits(rows, "uniform", uniform_plot)
    plot_mean_rd_by_bits(mean_rows, mean_plot)
    plot_delta_heatmap(delta_rows, delta_plot)

    summary = {
        "stage": "51b",
        "mode": "clean all-frame PSNR RD plots from existing Stage51 CSV",
        "stage51_csv": str(args.stage51_csv),
        "mean_csv": str(mean_csv),
        "delta_csv": str(delta_csv),
        "plots": [str(adaptive_plot), str(uniform_plot), str(mean_plot), str(delta_plot)],
        "notes": [
            "No rendering is rerun; this only replots existing Stage51 all-frame PSNR results.",
            "Colors distinguish quantization bits; point labels show keyframe gap.",
            "The adaptive method rendered_prior_0p1 remains an oracle/calibrated selector, not a fully feed-forward selector.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
