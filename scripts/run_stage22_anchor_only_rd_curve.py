import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE2_CSV = REPO_ROOT / "experiments/stage2_keyframe_gaussian_anchor/stage2_keyframe_gaussian_anchor_summary.csv"
DEFAULT_STAGE21D_GAP_CSV = REPO_ROOT / "experiments/stage21d_validated_anchor_adapter_training/stage21d_best_gap_eval_summary.csv"
DEFAULT_STAGE21D_SUMMARY = REPO_ROOT / "experiments/stage21d_validated_anchor_adapter_training/stage21d_validated_anchor_adapter_training_summary.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage22_anchor_only_rd_curve"


def load_rates(path, sample, profile, codec, opacity_threshold):
    rates = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sample"] != sample:
                continue
            if row["profile"] != profile or row["codec"] != codec:
                continue
            if float(row["opacity_threshold"]) != float(opacity_threshold):
                continue
            rates[int(row["frame_gap"])] = {
                "estimated_q8_static_mib_per_frame": float(row["avg_mib_per_video_frame"]),
                "keyframe_count": int(row["keyframe_count"]),
                "keyframe_ratio": float(row["keyframe_ratio"]),
                "total_mib": float(row["total_mib"]),
            }
    return rates


def load_gap_eval(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame_gap": int(row["frame_gap"]),
                "linear_psnr": float(row["linear_psnr"]),
                "adapter_psnr": float(row["final_model_psnr"]),
                "delta_psnr": float(row["delta_psnr"]),
                "margin_over_linear_psnr": float(row["margin_over_linear_psnr"]),
            })
    return rows


def build_rows(gap_rows, rates, sample):
    out = []
    for row in gap_rows:
        gap = row["frame_gap"]
        if gap not in rates:
            raise RuntimeError(f"Missing Stage2 rate for sample={sample} gap={gap}")
        out.append({
            "sample": sample,
            "frame_gap": gap,
            **rates[gap],
            **row,
        })
    return sorted(out, key=lambda r: r["estimated_q8_static_mib_per_frame"])


def write_csv(rows, path):
    fields = [
        "sample",
        "frame_gap",
        "estimated_q8_static_mib_per_frame",
        "keyframe_count",
        "keyframe_ratio",
        "linear_psnr",
        "adapter_psnr",
        "delta_psnr",
        "margin_over_linear_psnr",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def plot_rd(rows, out_path):
    xs = [row["estimated_q8_static_mib_per_frame"] for row in rows]
    linear = [row["linear_psnr"] for row in rows]
    adapter = [row["adapter_psnr"] for row in rows]
    labels = [f"g{row['frame_gap']}" for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(xs, linear, marker="o", linewidth=2.0, label="q8 linear anchor", color="#1f77b4")
    ax.plot(xs, adapter, marker="^", linewidth=2.0, label="Stage21d adapter", color="#d62728")
    for x, y, label in zip(xs, adapter, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Transmitted q8 static Gaussian MiB/frame")
    ax.set_ylabel("Robot intermediate RGB PSNR (dB)")
    ax.set_title("Stage 22 Anchor-Only RD Curve")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_delta(rows, out_path):
    gaps = [str(row["frame_gap"]) for row in sorted(rows, key=lambda r: r["frame_gap"])]
    deltas = [row["delta_psnr"] for row in sorted(rows, key=lambda r: r["frame_gap"])]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(gaps, deltas, color="#2ca02c")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("GOP gap")
    ax.set_ylabel("Adapter - linear PSNR (dB)")
    ax.set_title("Stage 22 Anchor Adapter Gain")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="robot")
    parser.add_argument("--stage2_csv", type=Path, default=DEFAULT_STAGE2_CSV)
    parser.add_argument("--stage21d_gap_csv", type=Path, default=DEFAULT_STAGE21D_GAP_CSV)
    parser.add_argument("--stage21d_summary", type=Path, default=DEFAULT_STAGE21D_SUMMARY)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--profile", default="static_anchor")
    parser.add_argument("--codec", default="q8")
    parser.add_argument("--opacity_threshold", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    rates = load_rates(args.stage2_csv, args.sample, args.profile, args.codec, args.opacity_threshold)
    gap_rows = load_gap_eval(args.stage21d_gap_csv)
    rows = build_rows(gap_rows, rates, args.sample)
    stage21d_summary = json.loads(args.stage21d_summary.read_text(encoding="utf-8"))

    csv_path = args.summary_root / "stage22_anchor_only_rd_curve.csv"
    summary_path = args.summary_root / "stage22_anchor_only_rd_curve_summary.json"
    rd_plot = args.summary_root / "stage22_anchor_only_rd_curve.png"
    delta_plot = args.summary_root / "stage22_anchor_only_delta_psnr.png"
    write_csv(rows, csv_path)
    plot_rd(rows, rd_plot)
    plot_delta(rows, delta_plot)

    mean_delta = sum(row["delta_psnr"] for row in rows) / max(len(rows), 1)
    summary = {
        "stage": 22,
        "mode": "anchor-only RD curve from Stage21d validated adapter",
        "sample": args.sample,
        "source_stage2_csv": str(args.stage2_csv),
        "source_stage21d_gap_csv": str(args.stage21d_gap_csv),
        "source_stage21d_summary": str(args.stage21d_summary),
        "stage21d_best_checkpoint": stage21d_summary["external_best_checkpoint"],
        "rate_profile": args.profile,
        "rate_codec": args.codec,
        "rate_opacity_threshold": args.opacity_threshold,
        "rows": rows,
        "mean_delta_psnr": mean_delta,
        "min_delta_psnr": min(row["delta_psnr"] for row in rows),
        "max_delta_psnr": max(row["delta_psnr"] for row in rows),
        "csv": str(csv_path),
        "plots": [str(rd_plot), str(delta_plot)],
        "notes": "Intermediate-target anchor-only RD. Quality is measured on Stage21d robot eval tasks, not full-video all-frame PSNR/SSIM.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "plots": summary["plots"],
        "mean_delta_psnr": mean_delta,
        "rows": len(rows),
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
