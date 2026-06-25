import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_stage12_selected_keyframe_reconstruction import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_STAGE1_CACHE,
    run_selected,
    write_csv,
)


DEFAULT_SELECTION_CSV = REPO_ROOT / "experiments/stage16_segment_error_keyframe_selection/stage16_segment_error_keyframe_selection_summary.csv"
DEFAULT_HEAVY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage17_segment_error_rd_curve")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage17_segment_error_rd_curve"


def plot_metric(rows, metric_key, ylabel, title, out_path):
    samples = sorted({row["sample"] for row in rows})
    methods = ["uniform", "segment_rd"]
    colors = {"uniform": "#1f77b4", "segment_rd": "#2ca02c"}
    markers = {"uniform": "o", "segment_rd": "^"}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["sample"], row["selection_method"])].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharey=False)
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        for method in methods:
            points = sorted(grouped[(sample, method)], key=lambda r: r["estimated_q8_static_mib_per_frame"])
            if not points:
                continue
            xs = [row["estimated_q8_static_mib_per_frame"] for row in points]
            ys = [metric_value(row, metric_key) for row in points]
            labels = [f"g{int(row['reference_gap'])}" for row in points]
            ax.plot(xs, ys, marker=markers[method], color=colors[method], linewidth=2.0, label=method)
            for x, y, label in zip(xs, ys, labels):
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_title(sample)
        ax.set_xlabel("Transmitted Gaussian MiB/frame")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(frameon=False)
    axes[0].set_ylabel(ylabel)
    axes[2].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def metric_value(row, metric_key):
    if metric_key == "all_psnr_avg":
        return row["all"]["psnr_avg"]
    if metric_key == "middle_psnr_avg":
        return row["middle_only"]["psnr_avg"]
    raise KeyError(metric_key)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--methods", nargs="*", default=["uniform", "segment_rd"])
    parser.add_argument("--gaps", nargs="*", type=int, default=[4, 8, 16])
    parser.add_argument("--selection_csv", type=Path, default=DEFAULT_SELECTION_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_segment_length", type=int, default=40)
    parser.add_argument("--save_per_frame", action="store_true")
    parser.add_argument("--reuse_existing", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
    args.summary_root.mkdir(parents=True, exist_ok=True)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for sample in args.samples:
        for method in args.methods:
            for gap in args.gaps:
                print(f"=== Stage17 sample={sample} method={method} gap={gap} ===", flush=True)
                single_path = args.summary_root / f"{sample}_{method}_gap{gap}_summary.json"
                if args.reuse_existing and single_path.exists():
                    rows.append(json.loads(single_path.read_text(encoding="utf-8")))
                    continue
                run_args = SimpleNamespace(
                    sample=sample,
                    method=method,
                    reference_gap=gap,
                    selection_csv=args.selection_csv,
                    checkpoint=args.checkpoint,
                    stage1_cache=args.stage1_cache,
                    heavy_root=args.heavy_root,
                    summary_root=args.summary_root,
                    batch_size=args.batch_size,
                    max_segment_length=args.max_segment_length,
                    save_per_frame=args.save_per_frame,
                )
                summary = run_selected(run_args, device)
                summary["stage"] = 17
                summary["method"] = "segment-error-aware selected-keyframe RD curve"
                rows.append(summary)
                single_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = args.summary_root / "stage17_segment_error_rd_curve_summary.csv"
    summary_path = args.summary_root / "stage17_segment_error_rd_curve_summary.json"
    write_csv(rows, csv_path)
    plot_outputs = []
    for metric_key, ylabel, title, filename in [
        ("all_psnr_avg", "All-frame PSNR (dB)", "Stage 17 RD Curve: All-frame PSNR", "stage17_rd_curve_all_psnr.png"),
        ("middle_psnr_avg", "Middle-frame PSNR (dB)", "Stage 17 RD Curve: Middle-only PSNR", "stage17_rd_curve_middle_psnr.png"),
    ]:
        out_path = args.summary_root / filename
        plot_metric(rows, metric_key, ylabel, title, out_path)
        plot_outputs.append(str(out_path))
    summary = {
        "stage": 17,
        "mode": "segment-error-aware selected-keyframe RD curve",
        "selection_csv": str(args.selection_csv),
        "samples": args.samples,
        "methods": args.methods,
        "gaps": args.gaps,
        "rows": rows,
        "csv": str(csv_path),
        "plots": plot_outputs,
        "notes": "Uses StreamSplat RGB/depth-conditioned selected-pair inference. This evaluates keyframe selection only, not final Gaussian-anchor-only decoding.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "csv": str(csv_path), "plots": plot_outputs, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
