import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_ROOT = REPO_ROOT / "experiments/stage1_streamsplat_fair_metrics"
DEFAULT_STAGE2_CSV = REPO_ROOT / "experiments/stage2_keyframe_gaussian_anchor/stage2_keyframe_gaussian_anchor_summary.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage19_original_decoder_variable_gop_baseline"


def load_stage2_rates(path, profile, codec, opacity_threshold):
    rates = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["profile"] != profile or row["codec"] != codec:
                continue
            if float(row["opacity_threshold"]) != float(opacity_threshold):
                continue
            key = (row["sample"], int(row["frame_gap"]))
            rates[key] = {
                "estimated_q8_static_mib_per_frame": float(row["avg_mib_per_video_frame"]),
                "estimated_q8_static_total_mib": float(row["total_mib"]),
                "gaussians_total": int(row["gaussians_total"]),
                "gaussians_kept": int(row["gaussians_kept"]),
                "keep_ratio": float(row["keep_ratio"]),
            }
    return rates


def metric(summary, group, name):
    value = summary[group][name]
    return None if value is None else float(value)


def build_row(summary, rate_info):
    selected = list(summary["selected_keyframes"])
    segment_lengths = [b - a for a, b in zip(selected[:-1], selected[1:])]
    mean_segment_length = sum(segment_lengths) / max(len(segment_lengths), 1)
    row = {
        "stage": 19,
        "mode": "original StreamSplat decoder variable-GOP pre-finetune baseline",
        "sample": summary["sample"],
        "reference_gap": int(summary["frame_gap"]),
        "total_frames": int(summary["total_frames"]),
        "resolution": summary["resolution"],
        "selected_keyframes": selected,
        "keyframe_count": int(summary["keyframe_count"]),
        "keyframe_ratio": float(summary["keyframe_ratio"]),
        "pair_count": int(summary["pair_count"]),
        "segment_lengths": segment_lengths,
        "max_segment_length": max(segment_lengths) if segment_lengths else 0,
        "mean_segment_length": mean_segment_length,
        "complete": bool(summary["complete"]),
        "all_count": int(summary["all"]["count"]),
        "all_psnr_avg": metric(summary, "all", "psnr_avg"),
        "all_psnr_min": metric(summary, "all", "psnr_min"),
        "all_ssim_avg": metric(summary, "all", "ssim_avg"),
        "all_ssim_min": metric(summary, "all", "ssim_min"),
        "middle_count": int(summary["middle_only"]["count"]),
        "middle_psnr_avg": metric(summary, "middle_only", "psnr_avg"),
        "middle_psnr_min": metric(summary, "middle_only", "psnr_min"),
        "middle_ssim_avg": metric(summary, "middle_only", "ssim_avg"),
        "middle_ssim_min": metric(summary, "middle_only", "ssim_min"),
        "given_count": int(summary["given_keyframes"]["count"]),
        "given_psnr_avg": metric(summary, "given_keyframes", "psnr_avg"),
        "given_psnr_min": metric(summary, "given_keyframes", "psnr_min"),
        "given_ssim_avg": metric(summary, "given_keyframes", "ssim_avg"),
        "given_ssim_min": metric(summary, "given_keyframes", "ssim_min"),
        "raw_pred_gs_mib": float(summary["raw_pred_gs_mib"]),
        "raw_pred_gs_mib_per_frame": float(summary["raw_pred_gs_mib_per_frame"]),
        "estimated_q8_static_mib_per_frame": rate_info["estimated_q8_static_mib_per_frame"],
        "estimated_q8_static_total_mib": rate_info["estimated_q8_static_total_mib"],
        "gaussians_total": rate_info["gaussians_total"],
        "gaussians_kept": rate_info["gaussians_kept"],
        "keep_ratio": rate_info["keep_ratio"],
        "rate_note": "q8 static keyframe-anchor payload from Stage2, averaged over total video frames.",
        "quality_note": "Original StreamSplat RGB/depth-conditioned decoder output before any long-GOP fine-tuning.",
    }
    return row


def load_rows(args):
    rates = load_stage2_rates(args.stage2_csv, args.profile, args.codec, args.opacity_threshold)
    rows = []
    for sample in args.samples:
        for gap in args.gaps:
            path = args.stage1_root / f"{sample}_gap{gap}_summary.json"
            if not path.exists():
                raise FileNotFoundError(path)
            key = (sample, gap)
            if key not in rates:
                raise RuntimeError(f"Missing Stage2 rate for sample={sample} gap={gap}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            rows.append(build_row(summary, rates[key]))
    return rows


def average(values):
    values = [value for value in values if value is not None]
    return sum(values) / max(len(values), 1)


def build_gap_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["reference_gap"]].append(row)
    summary_rows = []
    for gap in sorted(grouped):
        points = grouped[gap]
        summary_rows.append({
            "reference_gap": gap,
            "sample_count": len(points),
            "mean_estimated_q8_static_mib_per_frame": average([r["estimated_q8_static_mib_per_frame"] for r in points]),
            "mean_raw_pred_gs_mib_per_frame": average([r["raw_pred_gs_mib_per_frame"] for r in points]),
            "mean_keyframe_ratio": average([r["keyframe_ratio"] for r in points]),
            "mean_all_psnr_avg": average([r["all_psnr_avg"] for r in points]),
            "mean_middle_psnr_avg": average([r["middle_psnr_avg"] for r in points]),
            "mean_given_psnr_avg": average([r["given_psnr_avg"] for r in points]),
            "mean_all_ssim_avg": average([r["all_ssim_avg"] for r in points]),
            "mean_middle_ssim_avg": average([r["middle_ssim_avg"] for r in points]),
            "mean_given_ssim_avg": average([r["given_ssim_avg"] for r in points]),
        })
    return summary_rows


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def plot_metric(rows, metric_key, ylabel, title, out_path):
    samples = sorted({row["sample"] for row in rows})
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sample"]].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharey=False)
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        points = sorted(grouped[sample], key=lambda r: r["estimated_q8_static_mib_per_frame"])
        xs = [row["estimated_q8_static_mib_per_frame"] for row in points]
        ys = [row[metric_key] for row in points]
        labels = [f"g{row['reference_gap']}" for row in points]
        ax.plot(xs, ys, marker="o", linewidth=2.0, color="#1f77b4")
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_title(sample)
        ax.set_xlabel("Transmitted q8 static Gaussian MiB/frame")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    axes[0].set_ylabel(ylabel)
    axes[2].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--gaps", nargs="*", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--stage1_root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--stage2_csv", type=Path, default=DEFAULT_STAGE2_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--profile", default="static_anchor")
    parser.add_argument("--codec", default="q8")
    parser.add_argument("--opacity_threshold", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args)
    gap_summary_rows = build_gap_summary(rows)

    row_fields = [
        "stage",
        "sample",
        "reference_gap",
        "total_frames",
        "keyframe_count",
        "keyframe_ratio",
        "pair_count",
        "max_segment_length",
        "mean_segment_length",
        "estimated_q8_static_mib_per_frame",
        "raw_pred_gs_mib_per_frame",
        "all_count",
        "all_psnr_avg",
        "all_ssim_avg",
        "middle_count",
        "middle_psnr_avg",
        "middle_ssim_avg",
        "given_count",
        "given_psnr_avg",
        "given_ssim_avg",
        "complete",
        "rate_note",
        "quality_note",
    ]
    gap_fields = [
        "reference_gap",
        "sample_count",
        "mean_estimated_q8_static_mib_per_frame",
        "mean_raw_pred_gs_mib_per_frame",
        "mean_keyframe_ratio",
        "mean_all_psnr_avg",
        "mean_middle_psnr_avg",
        "mean_given_psnr_avg",
        "mean_all_ssim_avg",
        "mean_middle_ssim_avg",
        "mean_given_ssim_avg",
    ]

    csv_path = args.summary_root / "stage19_original_decoder_variable_gop_baseline.csv"
    gap_csv_path = args.summary_root / "stage19_original_decoder_variable_gop_gap_averages.csv"
    summary_path = args.summary_root / "stage19_original_decoder_variable_gop_baseline_summary.json"
    write_csv(rows, csv_path, row_fields)
    write_csv(gap_summary_rows, gap_csv_path, gap_fields)

    plots = []
    for metric_key, ylabel, title, filename in [
        ("all_psnr_avg", "All-frame PSNR (dB)", "Stage 19 Original Decoder Baseline: All-frame PSNR", "stage19_original_decoder_all_psnr.png"),
        ("middle_psnr_avg", "Middle-only PSNR (dB)", "Stage 19 Original Decoder Baseline: Middle-only PSNR", "stage19_original_decoder_middle_psnr.png"),
        ("given_psnr_avg", "Given-keyframe PSNR (dB)", "Stage 19 Original Decoder Baseline: Given-keyframe PSNR", "stage19_original_decoder_given_psnr.png"),
    ]:
        out_path = args.summary_root / filename
        plot_metric(rows, metric_key, ylabel, title, out_path)
        plots.append(str(out_path))

    summary = {
        "stage": 19,
        "mode": "original StreamSplat decoder variable-GOP pre-finetune baseline",
        "samples": args.samples,
        "gaps": args.gaps,
        "source_stage1_root": str(args.stage1_root),
        "source_stage2_csv": str(args.stage2_csv),
        "rate_profile": args.profile,
        "rate_codec": args.codec,
        "rate_opacity_threshold": args.opacity_threshold,
        "csv": str(csv_path),
        "gap_average_csv": str(gap_csv_path),
        "plots": plots,
        "rows": rows,
        "gap_averages": gap_summary_rows,
        "notes": "No new inference is run. This aggregates Stage1 pretrained StreamSplat results and Stage2 q8 static-anchor rate estimates as the pre-finetune variable-GOP baseline.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "gap_average_csv": str(gap_csv_path),
        "plots": plots,
        "rows": len(rows),
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
