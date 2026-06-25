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
DEFAULT_STAGE1_CACHE = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage1_streamsplat_fair_metrics/cache")
DEFAULT_STAGE16_CSV = REPO_ROOT / "experiments/stage16_segment_error_keyframe_selection/stage16_segment_error_keyframe_selection_summary.csv"
DEFAULT_STAGE25_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage25_leave_one_out_adapter_training")
DEFAULT_STAGE33_MANIFEST = REPO_ROOT / "experiments/stage33_dense_gap1_anchor_export/stage33_dense_gap1_anchor_manifest.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage34_dense_stage16_selector_rd"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import q8_static_mib_per_frame  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import load_adapter  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import (  # noqa: E402
    build_anchor_index,
    flatten_eval,
    read_manifest_rows,
    selected_records,
)


def read_selection_rows(path, samples, methods, gaps):
    sample_set = set(samples)
    method_set = set(methods)
    gap_set = set(gaps)
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gap = int(row["reference_gap"])
            if row["sample"] not in sample_set or row["method"] not in method_set or gap not in gap_set:
                continue
            rows.append({
                "sample": row["sample"],
                "method": row["method"],
                "reference_gap": gap,
                "indices": [int(v) for v in row["indices"].split()],
            })
    return rows


def write_csv(rows, path):
    fields = [
        "sample", "method", "reference_gap", "total_frames", "keyframe_count", "keyframe_ratio",
        "estimated_q8_static_mib_per_frame", "max_segment_length", "mean_segment_length", "segment_lengths", "indices",
        "linear_all_psnr", "adapter_all_psnr", "delta_all_psnr", "linear_middle_psnr", "adapter_middle_psnr",
        "delta_middle_psnr", "linear_given_psnr", "adapter_given_psnr", "delta_given_psnr", "linear_all_ssim",
        "adapter_all_ssim", "linear_middle_ssim", "adapter_middle_ssim", "checkpoint",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selector_comparison(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["sample"], row["reference_gap"])][row["method"]] = row
    out = []
    for (sample, gap), methods in sorted(grouped.items()):
        if "uniform" not in methods or "segment_rd" not in methods:
            continue
        uniform = methods["uniform"]
        selected = methods["segment_rd"]
        out.append({
            "sample": sample,
            "reference_gap": gap,
            "rate_mib_per_frame": selected["estimated_q8_static_mib_per_frame"],
            "selector_delta_adapter_all_psnr": selected["adapter_all_psnr"] - uniform["adapter_all_psnr"],
            "selector_delta_adapter_middle_psnr": selected["adapter_middle_psnr"] - uniform["adapter_middle_psnr"],
            "selector_delta_linear_all_psnr": selected["linear_all_psnr"] - uniform["linear_all_psnr"],
            "selector_delta_linear_middle_psnr": selected["linear_middle_psnr"] - uniform["linear_middle_psnr"],
        })
    return out


def write_comparison_csv(rows, path):
    fields = [
        "sample", "reference_gap", "rate_mib_per_frame", "selector_delta_adapter_all_psnr",
        "selector_delta_adapter_middle_psnr", "selector_delta_linear_all_psnr", "selector_delta_linear_middle_psnr",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_delta(comparisons, key, ylabel, title, out_path):
    grouped = defaultdict(list)
    for row in comparisons:
        grouped[row["sample"]].append(row)
    samples = sorted(grouped)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharey=True)
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        points = sorted(grouped[sample], key=lambda row: row["reference_gap"])
        gaps = [str(row["reference_gap"]) for row in points]
        values = [row[key] for row in points]
        ax.bar(gaps, values, color="#2ca02c")
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(sample)
        ax.set_xlabel("Reference GOP gap")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    axes[0].set_ylabel(ylabel)
    axes[2].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def segment_fields(indices):
    lengths = [b - a for a, b in zip(indices[:-1], indices[1:])]
    return {
        "max_segment_length": int(max(lengths)),
        "mean_segment_length": float(np.mean(lengths)),
        "segment_lengths": " ".join(str(v) for v in lengths),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--stage16_csv", type=Path, default=DEFAULT_STAGE16_CSV)
    parser.add_argument("--stage25_root", type=Path, default=DEFAULT_STAGE25_ROOT)
    parser.add_argument("--stage33_manifest", type=Path, default=DEFAULT_STAGE33_MANIFEST)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--methods", nargs="*", default=["uniform", "segment_rd"])
    parser.add_argument("--gaps", nargs="*", type=int, default=[4, 8, 16])
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
    selection_rows = read_selection_rows(args.stage16_csv, args.samples, args.methods, args.gaps)

    rows = []
    coverage = []
    for sample in args.samples:
        frame_files = sorted((args.stage1_cache / sample / "frames").glob("*.png"))
        if not frame_files:
            raise RuntimeError(f"Missing frame cache for {sample}")
        manifest_rows = read_manifest_rows(args.stage33_manifest, sample)
        anchor_map = build_anchor_index(manifest_rows, device, args.quant_bits)
        missing = sorted(set(range(len(frame_files))) - set(anchor_map))
        if missing:
            raise RuntimeError(f"Stage33 dense anchors missing frames for {sample}: {missing[:10]}")
        coverage.append({"sample": sample, "total_frames": len(frame_files), "anchor_count": len(anchor_map)})
        checkpoint_path = args.stage25_root / sample / "stage25_best_anchor_adapter.safetensors"
        model = load_adapter(checkpoint_path, args.hidden_dim, device)
        for selection in [row for row in selection_rows if row["sample"] == sample]:
            print(f"=== Stage34 sample={sample} method={selection['method']} gap={selection['reference_gap']} ===", flush=True)
            metrics = selected_records(selection["indices"], anchor_map, model, frame_files, opt, background)
            row = flatten_eval(sample, selection["method"], selection["reference_gap"], selection["indices"], metrics, checkpoint_path)
            row.update(segment_fields(selection["indices"]))
            row["estimated_q8_static_mib_per_frame"] = q8_static_mib_per_frame(len(selection["indices"]), len(frame_files))
            rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = args.summary_root / "stage34_dense_stage16_selector_rd.csv"
    comparison_csv = args.summary_root / "stage34_selector_comparison.csv"
    summary_path = args.summary_root / "stage34_dense_stage16_selector_rd_summary.json"
    write_csv(rows, csv_path)
    comparisons = selector_comparison(rows)
    write_comparison_csv(comparisons, comparison_csv)
    plots = []
    for key, ylabel, title, filename in [
        ("selector_delta_adapter_all_psnr", "segment_rd - uniform all PSNR (dB)", "Stage 34 Dense Stage16 segment_rd Gain: All PSNR", "stage34_delta_all_psnr.png"),
        ("selector_delta_adapter_middle_psnr", "segment_rd - uniform middle PSNR (dB)", "Stage 34 Dense Stage16 segment_rd Gain: Middle PSNR", "stage34_delta_middle_psnr.png"),
    ]:
        out_path = args.summary_root / filename
        plot_delta(comparisons, key, ylabel, title, out_path)
        plots.append(str(out_path))
    all_values = [row["selector_delta_adapter_all_psnr"] for row in comparisons]
    middle_values = [row["selector_delta_adapter_middle_psnr"] for row in comparisons]
    summary = {
        "stage": 34,
        "mode": "dense-anchor Stage16 segment_rd selector RD",
        "stage16_csv": str(args.stage16_csv),
        "stage33_manifest": str(args.stage33_manifest),
        "csv": str(csv_path),
        "comparison_csv": str(comparison_csv),
        "plots": plots,
        "coverage": coverage,
        "rows": rows,
        "comparisons": comparisons,
        "mean_selector_delta_adapter_all_psnr": float(np.mean(all_values)),
        "mean_selector_delta_adapter_middle_psnr": float(np.mean(middle_values)),
        "positive_selector_all_points": sum(1 for value in all_values if value > 0.0),
        "positive_selector_middle_points": sum(1 for value in middle_values if value > 0.0),
        "notes": "Uses Stage33 dense gap1 anchors, so Stage16 segment_rd odd-frame keyframes are evaluated without snapping or anchor-availability constraints.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "comparison_csv": str(comparison_csv),
        "rows": len(rows),
        "mean_selector_delta_adapter_all_psnr": summary["mean_selector_delta_adapter_all_psnr"],
        "mean_selector_delta_adapter_middle_psnr": summary["mean_selector_delta_adapter_middle_psnr"],
        "positive_selector_all_points": summary["positive_selector_all_points"],
        "positive_selector_middle_points": summary["positive_selector_middle_points"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
