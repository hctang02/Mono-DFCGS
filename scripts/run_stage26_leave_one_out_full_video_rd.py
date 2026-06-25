import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments/stage6_real_anchor_dataset/stage6_real_anchor_dataset_manifest.csv"
DEFAULT_STAGE2_CSV = REPO_ROOT / "experiments/stage2_keyframe_gaussian_anchor/stage2_keyframe_gaussian_anchor_summary.csv"
DEFAULT_STAGE25_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage25_leave_one_out_adapter_training")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage26_leave_one_out_full_video_rd"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage22_anchor_only_rd_curve import load_rates  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import (  # noqa: E402
    evaluate_group,
    group_rows,
    load_adapter,
    read_manifest,
)


def checkpoint_for_sample(stage25_root, sample):
    path = stage25_root / sample / "stage25_best_anchor_adapter.safetensors"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def flatten_row(row, checkpoint_path):
    return {
        "sample": row["sample"],
        "frame_gap": row["frame_gap"],
        "estimated_q8_static_mib_per_frame": row["estimated_q8_static_mib_per_frame"],
        "total_mib": row["total_mib"],
        "total_frames": row["total_frames"],
        "keyframe_count": row["keyframe_count"],
        "keyframe_ratio": row["keyframe_ratio"],
        "middle_count": row["middle_count"],
        "linear_all_psnr": row["linear"]["all"]["psnr_avg"],
        "adapter_all_psnr": row["adapter"]["all"]["psnr_avg"],
        "delta_all_psnr": row["delta_all_psnr"],
        "linear_middle_psnr": row["linear"]["middle_only"]["psnr_avg"],
        "adapter_middle_psnr": row["adapter"]["middle_only"]["psnr_avg"],
        "delta_middle_psnr": row["delta_middle_psnr"],
        "linear_given_psnr": row["linear"]["given_keyframes"]["psnr_avg"],
        "adapter_given_psnr": row["adapter"]["given_keyframes"]["psnr_avg"],
        "delta_given_psnr": row["delta_given_psnr"],
        "linear_all_ssim": row["linear"]["all"]["ssim_avg"],
        "adapter_all_ssim": row["adapter"]["all"]["ssim_avg"],
        "linear_middle_ssim": row["linear"]["middle_only"]["ssim_avg"],
        "adapter_middle_ssim": row["adapter"]["middle_only"]["ssim_avg"],
        "checkpoint": str(checkpoint_path),
    }


def write_csv(rows, path):
    fields = [
        "sample",
        "frame_gap",
        "estimated_q8_static_mib_per_frame",
        "total_mib",
        "total_frames",
        "keyframe_count",
        "keyframe_ratio",
        "middle_count",
        "linear_all_psnr",
        "adapter_all_psnr",
        "delta_all_psnr",
        "linear_middle_psnr",
        "adapter_middle_psnr",
        "delta_middle_psnr",
        "linear_given_psnr",
        "adapter_given_psnr",
        "delta_given_psnr",
        "linear_all_ssim",
        "adapter_all_ssim",
        "linear_middle_ssim",
        "adapter_middle_ssim",
        "checkpoint",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by_sample(rows):
    out = []
    for sample in sorted({row["sample"] for row in rows}):
        points = [row for row in rows if row["sample"] == sample]
        out.append({
            "sample": sample,
            "mean_delta_all_psnr": float(np.mean([row["delta_all_psnr"] for row in points])),
            "mean_delta_middle_psnr": float(np.mean([row["delta_middle_psnr"] for row in points])),
            "min_delta_all_psnr": float(np.min([row["delta_all_psnr"] for row in points])),
            "min_delta_middle_psnr": float(np.min([row["delta_middle_psnr"] for row in points])),
        })
    return out


def aggregate_by_gap(rows):
    out = []
    for gap in sorted({row["frame_gap"] for row in rows}):
        points = [row for row in rows if row["frame_gap"] == gap]
        out.append({
            "frame_gap": gap,
            "estimated_q8_static_mib_per_frame": float(np.mean([row["estimated_q8_static_mib_per_frame"] for row in points])),
            "linear_all_psnr": float(np.mean([row["linear_all_psnr"] for row in points])),
            "adapter_all_psnr": float(np.mean([row["adapter_all_psnr"] for row in points])),
            "delta_all_psnr": float(np.mean([row["delta_all_psnr"] for row in points])),
            "linear_middle_psnr": float(np.mean([row["linear_middle_psnr"] for row in points])),
            "adapter_middle_psnr": float(np.mean([row["adapter_middle_psnr"] for row in points])),
            "delta_middle_psnr": float(np.mean([row["delta_middle_psnr"] for row in points])),
            "linear_all_ssim": float(np.mean([row["linear_all_ssim"] for row in points])),
            "adapter_all_ssim": float(np.mean([row["adapter_all_ssim"] for row in points])),
            "linear_middle_ssim": float(np.mean([row["linear_middle_ssim"] for row in points])),
            "adapter_middle_ssim": float(np.mean([row["adapter_middle_ssim"] for row in points])),
        })
    return out


def write_sample_aggregate_csv(rows, path):
    fields = ["sample", "mean_delta_all_psnr", "mean_delta_middle_psnr", "min_delta_all_psnr", "min_delta_middle_psnr"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_gap_aggregate_csv(rows, path):
    fields = [
        "frame_gap",
        "estimated_q8_static_mib_per_frame",
        "linear_all_psnr",
        "adapter_all_psnr",
        "delta_all_psnr",
        "linear_middle_psnr",
        "adapter_middle_psnr",
        "delta_middle_psnr",
        "linear_all_ssim",
        "adapter_all_ssim",
        "linear_middle_ssim",
        "adapter_middle_ssim",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_per_sample(rows, linear_key, adapter_key, ylabel, title, out_path):
    samples = sorted({row["sample"] for row in rows})
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sample"]].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharey=False)
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        points = sorted(grouped[sample], key=lambda row: row["estimated_q8_static_mib_per_frame"])
        xs = [row["estimated_q8_static_mib_per_frame"] for row in points]
        linear = [row[linear_key] for row in points]
        adapter = [row[adapter_key] for row in points]
        labels = [f"g{row['frame_gap']}" for row in points]
        ax.plot(xs, linear, marker="o", linewidth=2.0, label="q8 linear", color="#1f77b4")
        ax.plot(xs, adapter, marker="^", linewidth=2.0, label="Stage25 held-out adapter", color="#d62728")
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


def plot_average_rd(rows, linear_key, adapter_key, ylabel, title, out_path):
    points = sorted(rows, key=lambda row: row["estimated_q8_static_mib_per_frame"])
    xs = [row["estimated_q8_static_mib_per_frame"] for row in points]
    linear = [row[linear_key] for row in points]
    adapter = [row[adapter_key] for row in points]
    labels = [f"g{row['frame_gap']}" for row in points]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(xs, linear, marker="o", linewidth=2.0, label="q8 linear", color="#1f77b4")
    ax.plot(xs, adapter, marker="^", linewidth=2.0, label="Stage25 held-out adapter", color="#d62728")
    for x, y, label in zip(xs, adapter, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Mean transmitted q8 static Gaussian MiB/frame")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(frameon=False)
    fig.tight_layout()
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


def make_plots(rows, gap_rows, summary_root):
    plots = []
    specs = [
        ("linear_all_psnr", "adapter_all_psnr", "All-frame PSNR (dB)", "Stage 26 Leave-One-Out Full-Video RD: All PSNR", "stage26_per_sample_all_psnr_rd.png"),
        ("linear_middle_psnr", "adapter_middle_psnr", "Middle-only PSNR (dB)", "Stage 26 Leave-One-Out Full-Video RD: Middle PSNR", "stage26_per_sample_middle_psnr_rd.png"),
        ("linear_all_ssim", "adapter_all_ssim", "All-frame SSIM", "Stage 26 Leave-One-Out Full-Video RD: All SSIM", "stage26_per_sample_all_ssim_rd.png"),
        ("linear_middle_ssim", "adapter_middle_ssim", "Middle-only SSIM", "Stage 26 Leave-One-Out Full-Video RD: Middle SSIM", "stage26_per_sample_middle_ssim_rd.png"),
    ]
    for linear_key, adapter_key, ylabel, title, filename in specs:
        out_path = summary_root / filename
        plot_per_sample(rows, linear_key, adapter_key, ylabel, title, out_path)
        plots.append(str(out_path))

    avg_specs = [
        ("linear_all_psnr", "adapter_all_psnr", "Mean all-frame PSNR (dB)", "Stage 26 Mean Full-Video RD: All PSNR", "stage26_mean_all_psnr_rd.png"),
        ("linear_middle_psnr", "adapter_middle_psnr", "Mean middle-only PSNR (dB)", "Stage 26 Mean Full-Video RD: Middle PSNR", "stage26_mean_middle_psnr_rd.png"),
        ("linear_all_ssim", "adapter_all_ssim", "Mean all-frame SSIM", "Stage 26 Mean Full-Video RD: All SSIM", "stage26_mean_all_ssim_rd.png"),
        ("linear_middle_ssim", "adapter_middle_ssim", "Mean middle-only SSIM", "Stage 26 Mean Full-Video RD: Middle SSIM", "stage26_mean_middle_ssim_rd.png"),
    ]
    for linear_key, adapter_key, ylabel, title, filename in avg_specs:
        out_path = summary_root / filename
        plot_average_rd(gap_rows, linear_key, adapter_key, ylabel, title, out_path)
        plots.append(str(out_path))

    for delta_key, ylabel, title, filename in [
        ("delta_all_psnr", "Adapter - linear all PSNR (dB)", "Stage 26 Held-Out Adapter Gain: All PSNR", "stage26_delta_all_psnr.png"),
        ("delta_middle_psnr", "Adapter - linear middle PSNR (dB)", "Stage 26 Held-Out Adapter Gain: Middle PSNR", "stage26_delta_middle_psnr.png"),
    ]:
        out_path = summary_root / filename
        plot_delta(rows, delta_key, ylabel, title, out_path)
        plots.append(str(out_path))
    return plots


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage2_csv", type=Path, default=DEFAULT_STAGE2_CSV)
    parser.add_argument("--stage25_root", type=Path, default=DEFAULT_STAGE25_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--gaps", nargs="*", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)

    manifest_rows = read_manifest(args.manifest, args.samples, args.gaps)
    grouped = group_rows(manifest_rows)
    flat_rows = []
    full_rows = []
    checkpoint_map = {}

    for sample in args.samples:
        checkpoint_path = checkpoint_for_sample(args.stage25_root, sample)
        checkpoint_map[sample] = str(checkpoint_path)
        model = load_adapter(checkpoint_path, args.hidden_dim, device)
        rates = load_rates(args.stage2_csv, sample, "static_anchor", "q8", 0.0)
        for gap in args.gaps:
            key = (sample, gap)
            if key not in grouped:
                raise RuntimeError(f"Missing manifest rows for {sample} gap={gap}")
            print(f"=== Stage26 sample={sample} gap={gap} ===", flush=True)
            summary = evaluate_group(sample, gap, grouped[key], model, opt, background, device, args.quant_bits)
            summary.update(rates[gap])
            full_rows.append(summary)
            flat_rows.append(flatten_row(summary, checkpoint_path))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = args.summary_root / "stage26_leave_one_out_full_video_rd.csv"
    sample_aggregate_csv = args.summary_root / "stage26_sample_aggregate.csv"
    gap_aggregate_csv = args.summary_root / "stage26_gap_aggregate.csv"
    summary_path = args.summary_root / "stage26_leave_one_out_full_video_rd_summary.json"
    write_csv(flat_rows, csv_path)
    sample_aggregate = aggregate_by_sample(flat_rows)
    gap_aggregate = aggregate_by_gap(flat_rows)
    write_sample_aggregate_csv(sample_aggregate, sample_aggregate_csv)
    write_gap_aggregate_csv(gap_aggregate, gap_aggregate_csv)
    plots = make_plots(flat_rows, gap_aggregate, args.summary_root)

    mean_delta_all = float(np.mean([row["delta_all_psnr"] for row in flat_rows]))
    mean_delta_middle = float(np.mean([row["delta_middle_psnr"] for row in flat_rows]))
    negative_all = sum(1 for row in flat_rows if row["delta_all_psnr"] < 0.0)
    negative_middle = sum(1 for row in flat_rows if row["delta_middle_psnr"] < 0.0)
    summary = {
        "stage": 26,
        "mode": "leave-one-sample-out full-video anchor-only RD",
        "samples": args.samples,
        "gaps": args.gaps,
        "checkpoint_map": checkpoint_map,
        "quant_bits": args.quant_bits,
        "rows": flat_rows,
        "csv": str(csv_path),
        "sample_aggregate_csv": str(sample_aggregate_csv),
        "sample_aggregate": sample_aggregate,
        "gap_aggregate_csv": str(gap_aggregate_csv),
        "gap_aggregate": gap_aggregate,
        "plots": plots,
        "mean_delta_all_psnr": mean_delta_all,
        "mean_delta_middle_psnr": mean_delta_middle,
        "negative_all_psnr_points": negative_all,
        "negative_middle_psnr_points": negative_middle,
        "notes": "Each sample is evaluated with its own Stage25 held-out fold checkpoint. Given keyframes are rendered directly from transmitted q8 anchors for both methods.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "rows": len(flat_rows),
        "mean_delta_all_psnr": mean_delta_all,
        "mean_delta_middle_psnr": mean_delta_middle,
        "negative_all_psnr_points": negative_all,
        "negative_middle_psnr_points": negative_middle,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
