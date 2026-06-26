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
DEFAULT_STAGE33_MANIFEST = REPO_ROOT / "experiments/stage33_dense_gap1_anchor_export/stage33_dense_gap1_anchor_manifest.csv"
DEFAULT_STAGE45B_CSV = REPO_ROOT / "experiments/stage45b_calibrated_rendered_oracle_selector_rd/stage45b_calibrated_rendered_oracle_selector_rd.csv"
DEFAULT_HEAVY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage46_calibrated_adaptive_actual_bitstream_rd")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage46_calibrated_adaptive_actual_bitstream_rd"


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.anchor_bitstream import decode_anchor_bitstream, encode_anchor_bitstream  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import build_anchor_index, read_manifest_rows  # noqa: E402
from scripts.run_stage36_dense_oracle_actual_bitstream_rd import direct_q8_anchor, roundtrip_error, safe_method_name  # noqa: E402


QUALITY_FIELDS = [
    "linear_all_psnr", "adapter_all_psnr", "delta_all_psnr", "linear_middle_psnr", "adapter_middle_psnr",
    "delta_middle_psnr", "linear_all_ssim", "adapter_all_ssim", "linear_middle_ssim", "adapter_middle_ssim",
]


def load_stage45b(path, adaptive_method):
    rows = []
    wanted = {"uniform", adaptive_method}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["method"] not in wanted:
                continue
            parsed = {
                "sample": row["sample"],
                "method": row["method"],
                "reference_gap": int(row["reference_gap"]),
                "total_frames": int(row["total_frames"]),
                "keyframe_count": int(row["keyframe_count"]),
                "indices": [int(v) for v in row["indices"].split()],
            }
            for key in QUALITY_FIELDS:
                parsed[key] = float(row[key])
            rows.append(parsed)
    return rows


def write_csv(rows, path):
    fields = [
        "sample", "method", "reference_gap", "total_frames", "keyframe_count", "indices", "raw_path", "zlib_path",
        "raw_bitstream_bytes", "zlib_bitstream_bytes", "raw_mib_per_frame", "zlib_mib_per_frame",
        "zlib_savings_percent_vs_raw_bitstream", *QUALITY_FIELDS, "max_roundtrip_abs_diff", "mean_roundtrip_mse",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selector_comparison(rows, adaptive_method, rate_key):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["sample"], row["reference_gap"])][row["method"]] = row
    out = []
    for (sample, gap), methods in sorted(grouped.items()):
        if "uniform" not in methods or adaptive_method not in methods:
            continue
        uniform = methods["uniform"]
        adaptive = methods[adaptive_method]
        out.append({
            "sample": sample,
            "reference_gap": gap,
            "rate_kind": rate_key,
            "uniform_rate_mib_per_frame": uniform[rate_key],
            "adaptive_rate_mib_per_frame": adaptive[rate_key],
            "rate_delta_mib_per_frame": adaptive[rate_key] - uniform[rate_key],
            "selector_delta_adapter_all_psnr": adaptive["adapter_all_psnr"] - uniform["adapter_all_psnr"],
            "selector_delta_adapter_middle_psnr": adaptive["adapter_middle_psnr"] - uniform["adapter_middle_psnr"],
        })
    return out


def write_comparison_csv(rows, path):
    fields = [
        "sample", "reference_gap", "rate_kind", "uniform_rate_mib_per_frame", "adaptive_rate_mib_per_frame",
        "rate_delta_mib_per_frame", "selector_delta_adapter_all_psnr", "selector_delta_adapter_middle_psnr",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_rd(rows, adaptive_method, rate_key, metric_key, ylabel, title, out_path):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["sample"], row["method"])].append(row)
    samples = sorted({row["sample"] for row in rows})
    methods = ["uniform", adaptive_method]
    colors = {"uniform": "#1f77b4", adaptive_method: "#d62728"}
    markers = {"uniform": "o", adaptive_method: "^"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharey=False)
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        for method in methods:
            points = sorted(grouped[(sample, method)], key=lambda row: row[rate_key])
            xs = [row[rate_key] for row in points]
            ys = [row[metric_key] for row in points]
            labels = [f"g{row['reference_gap']}" for row in points]
            ax.plot(xs, ys, marker=markers[method], color=colors[method], linewidth=2.0, label=method)
            for x, y, label in zip(xs, ys, labels):
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_title(sample)
        ax.set_xlabel("Raw bitstream MiB/frame" if rate_key == "raw_mib_per_frame" else "Zlib bitstream MiB/frame")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(frameon=False)
    axes[0].set_ylabel(ylabel)
    axes[2].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage33_manifest", type=Path, default=DEFAULT_STAGE33_MANIFEST)
    parser.add_argument("--stage45b_csv", type=Path, default=DEFAULT_STAGE45B_CSV)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--adaptive_method", default="rendered_prior_0p1")
    return parser.parse_args()


def main():
    args = parse_args()
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    source_rows = load_stage45b(args.stage45b_csv, args.adaptive_method)
    anchor_maps = {}
    for sample in sorted({row["sample"] for row in source_rows}):
        manifest_rows = read_manifest_rows(args.stage33_manifest, sample)
        anchor_maps[sample] = build_anchor_index(manifest_rows, torch.device("cpu"), quant_bits=0)

    rows = []
    for row in source_rows:
        sample = row["sample"]
        method = row["method"]
        gap = row["reference_gap"]
        indices = row["indices"]
        anchors = [anchor_maps[sample][idx] for idx in indices]
        raw_blob = encode_anchor_bitstream(anchors, indices, timestamps=indices, bits=8, compression="none")
        zlib_blob = encode_anchor_bitstream(anchors, indices, timestamps=indices, bits=8, compression="zlib")
        decoded, header = decode_anchor_bitstream(raw_blob)
        if int(header["anchor_count"]) != len(indices):
            raise RuntimeError("Decoded header anchor_count mismatch")
        max_abs, mean_mse = roundtrip_error(anchors, decoded)
        stem = f"{sample}_{safe_method_name(method)}_gap{gap}"
        raw_path = args.heavy_root / f"{stem}_q8_raw.mdfcgs"
        zlib_path = args.heavy_root / f"{stem}_q8_zlib.mdfcgs"
        raw_path.write_bytes(raw_blob)
        zlib_path.write_bytes(zlib_blob)
        raw_size = len(raw_blob)
        zlib_size = len(zlib_blob)
        rows.append({
            **{k: row[k] for k in ["sample", "method", "reference_gap", "total_frames", "keyframe_count", *QUALITY_FIELDS]},
            "indices": " ".join(str(idx) for idx in indices),
            "raw_path": str(raw_path),
            "zlib_path": str(zlib_path),
            "raw_bitstream_bytes": raw_size,
            "zlib_bitstream_bytes": zlib_size,
            "raw_mib_per_frame": raw_size / row["total_frames"] / (1024.0 * 1024.0),
            "zlib_mib_per_frame": zlib_size / row["total_frames"] / (1024.0 * 1024.0),
            "zlib_savings_percent_vs_raw_bitstream": 100.0 * (raw_size - zlib_size) / raw_size,
            "max_roundtrip_abs_diff": max_abs,
            "mean_roundtrip_mse": mean_mse,
        })

    csv_path = args.summary_root / "stage46_calibrated_adaptive_actual_bitstream_rd.csv"
    raw_comparison_csv = args.summary_root / "stage46_raw_selector_comparison.csv"
    zlib_comparison_csv = args.summary_root / "stage46_zlib_selector_comparison.csv"
    summary_path = args.summary_root / "stage46_calibrated_adaptive_actual_bitstream_rd_summary.json"
    write_csv(rows, csv_path)
    raw_comparisons = selector_comparison(rows, args.adaptive_method, "raw_mib_per_frame")
    zlib_comparisons = selector_comparison(rows, args.adaptive_method, "zlib_mib_per_frame")
    write_comparison_csv(raw_comparisons, raw_comparison_csv)
    write_comparison_csv(zlib_comparisons, zlib_comparison_csv)
    plots = []
    for rate_key, rate_name in [("raw_mib_per_frame", "raw"), ("zlib_mib_per_frame", "zlib")]:
        for metric_key, ylabel, filename in [
            ("adapter_all_psnr", "Adapter all-frame PSNR (dB)", f"stage46_{rate_name}_adapter_all_psnr_rd.png"),
            ("adapter_middle_psnr", "Adapter middle-only PSNR (dB)", f"stage46_{rate_name}_adapter_middle_psnr_rd.png"),
        ]:
            out_path = args.summary_root / filename
            plot_rd(rows, args.adaptive_method, rate_key, metric_key, ylabel, f"Stage 46 Calibrated Adaptive {rate_name.upper()} Bitstream RD", out_path)
            plots.append(str(out_path))
    all_values = [row["selector_delta_adapter_all_psnr"] for row in zlib_comparisons]
    middle_values = [row["selector_delta_adapter_middle_psnr"] for row in zlib_comparisons]
    summary = {
        "stage": 46,
        "mode": "calibrated adaptive actual q8 anchor bitstream RD",
        "adaptive_method": args.adaptive_method,
        "stage45b_csv": str(args.stage45b_csv),
        "stage33_manifest": str(args.stage33_manifest),
        "heavy_root": str(args.heavy_root),
        "csv": str(csv_path),
        "raw_comparison_csv": str(raw_comparison_csv),
        "zlib_comparison_csv": str(zlib_comparison_csv),
        "plots": plots,
        "rows": rows,
        "raw_comparisons": raw_comparisons,
        "zlib_comparisons": zlib_comparisons,
        "mean_selector_delta_adapter_all_psnr": float(np.mean(all_values)),
        "mean_selector_delta_adapter_middle_psnr": float(np.mean(middle_values)),
        "positive_selector_all_points": sum(1 for value in all_values if value > 0.0),
        "positive_selector_middle_points": sum(1 for value in middle_values if value > 0.0),
        "mean_zlib_savings_percent_vs_raw_bitstream": float(np.mean([row["zlib_savings_percent_vs_raw_bitstream"] for row in rows])),
        "mean_zlib_rate_delta_mib_per_frame": float(np.mean([row["rate_delta_mib_per_frame"] for row in zlib_comparisons])),
        "max_roundtrip_abs_diff": float(max(row["max_roundtrip_abs_diff"] for row in rows)),
        "notes": "Quality comes from Stage45b; rate is measured from actual Stage46 raw/zlib q8 anchor bitstreams. Selector is still rendered-oracle calibrated, not deployable predicted.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "raw_comparison_csv": str(raw_comparison_csv),
        "zlib_comparison_csv": str(zlib_comparison_csv),
        "plots": plots,
        "mean_selector_delta_adapter_all_psnr": summary["mean_selector_delta_adapter_all_psnr"],
        "mean_selector_delta_adapter_middle_psnr": summary["mean_selector_delta_adapter_middle_psnr"],
        "positive_selector_all_points": summary["positive_selector_all_points"],
        "mean_zlib_rate_delta_mib_per_frame": summary["mean_zlib_rate_delta_mib_per_frame"],
        "mean_zlib_savings_percent_vs_raw_bitstream": summary["mean_zlib_savings_percent_vs_raw_bitstream"],
        "max_roundtrip_abs_diff": summary["max_roundtrip_abs_diff"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
