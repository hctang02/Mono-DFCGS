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
DEFAULT_STAGE25_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage25_leave_one_out_adapter_training")
DEFAULT_STAGE33_MANIFEST = REPO_ROOT / "experiments/stage33_dense_gap1_anchor_export/stage33_dense_gap1_anchor_manifest.csv"
DEFAULT_STAGE49_CSV = REPO_ROOT / "experiments/stage49_extended_adaptive_rd/stage49_extended_adaptive_rd.csv"
DEFAULT_STAGE50_CSV = REPO_ROOT / "experiments/stage50_multibit_anchor_bitstream_prototype/stage50_multibit_anchor_bitstream_prototype.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage51_high_rate_multibit_rd"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import load_adapter  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import build_anchor_index, flatten_eval, read_manifest_rows  # noqa: E402
from scripts.run_stage49_extended_adaptive_rd import selected_records_allow_keyframe_only  # noqa: E402


QUALITY_FIELDS = [
    "linear_all_psnr", "adapter_all_psnr", "delta_all_psnr", "linear_middle_psnr", "adapter_middle_psnr",
    "delta_middle_psnr", "linear_given_psnr", "adapter_given_psnr", "delta_given_psnr", "linear_all_ssim",
    "adapter_all_ssim", "linear_middle_ssim", "adapter_middle_ssim",
]


def read_stage49(path, samples, methods, gaps):
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
                "total_frames": int(row["total_frames"]),
                "keyframe_count": int(row["keyframe_count"]),
                "indices": [int(v) for v in row["indices"].split()],
            })
    return rows


def read_stage50_rates(path):
    rates = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["sample"], row["method"], int(row["reference_gap"]), int(row["bits"]))
            rates[key] = {
                "payload_dtype": row["payload_dtype"],
                "raw_mib_per_frame": float(row["raw_mib_per_frame"]),
                "zlib_mib_per_frame": float(row["zlib_mib_per_frame"]),
                "theoretical_bitpacked_mib_per_frame": float(row["theoretical_bitpacked_mib_per_frame"]),
                "zlib_savings_percent_vs_raw_bitstream": float(row["zlib_savings_percent_vs_raw_bitstream"]),
                "max_roundtrip_abs_diff": float(row["max_roundtrip_abs_diff"]),
            }
    return rates


def write_csv(rows, path):
    fields = [
        "sample", "method", "reference_gap", "bits", "payload_dtype", "total_frames", "keyframe_count", "keyframe_ratio",
        "estimated_q8_static_mib_per_frame", "raw_mib_per_frame", "zlib_mib_per_frame", "theoretical_bitpacked_mib_per_frame",
        "zlib_savings_percent_vs_raw_bitstream", "max_roundtrip_abs_diff", "max_segment_length", "mean_segment_length",
        "segment_lengths", "indices", "checkpoint", *QUALITY_FIELDS,
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selector_comparison(rows, rate_key):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["sample"], row["reference_gap"], row["bits"])][row["method"]] = row
    out = []
    for (sample, gap, bits), methods in sorted(grouped.items()):
        if "uniform" not in methods or "rendered_prior_0p1" not in methods:
            continue
        uniform = methods["uniform"]
        adaptive = methods["rendered_prior_0p1"]
        out.append({
            "sample": sample,
            "reference_gap": gap,
            "bits": bits,
            "rate_kind": rate_key,
            "uniform_rate_mib_per_frame": uniform[rate_key],
            "adaptive_rate_mib_per_frame": adaptive[rate_key],
            "rate_delta_mib_per_frame": adaptive[rate_key] - uniform[rate_key],
            "selector_delta_adapter_all_psnr": adaptive["adapter_all_psnr"] - uniform["adapter_all_psnr"],
            "selector_delta_adapter_middle_psnr": None if adaptive["adapter_middle_psnr"] is None or uniform["adapter_middle_psnr"] is None else adaptive["adapter_middle_psnr"] - uniform["adapter_middle_psnr"],
        })
    return out


def write_comparison_csv(rows, path):
    fields = [
        "sample", "reference_gap", "bits", "rate_kind", "uniform_rate_mib_per_frame", "adaptive_rate_mib_per_frame",
        "rate_delta_mib_per_frame", "selector_delta_adapter_all_psnr", "selector_delta_adapter_middle_psnr",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_rd(rows, method, rate_key, metric_key, ylabel, title, out_path, include_gap1):
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] != method or row[metric_key] is None:
            continue
        if not include_gap1 and row["reference_gap"] == 1:
            continue
        grouped[row["sample"]].append(row)
    samples = sorted({row["sample"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharey=False)
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        points = sorted(grouped[sample], key=lambda row: row[rate_key])
        xs = [row[rate_key] for row in points]
        ys = [row[metric_key] for row in points]
        labels = [f"g{row['reference_gap']}q{row['bits']}" for row in points]
        ax.plot(xs, ys, marker="o", linewidth=1.4, color="#2ca02c")
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=6)
        ax.set_title(sample)
        ax.set_xlabel("Zlib q-anchor MiB/frame" if rate_key == "zlib_mib_per_frame" else "Raw q-anchor MiB/frame")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def aggregate_by_bits(rows, method):
    out = {}
    for bits in sorted({row["bits"] for row in rows}):
        bit_rows = [row for row in rows if row["bits"] == bits and row["method"] == method]
        out[str(bits)] = {
            "rows": len(bit_rows),
            "mean_zlib_mib_per_frame": float(np.mean([row["zlib_mib_per_frame"] for row in bit_rows])),
            "mean_adapter_all_psnr": float(np.mean([row["adapter_all_psnr"] for row in bit_rows])),
            "mean_adapter_middle_psnr": float(np.mean([row["adapter_middle_psnr"] for row in bit_rows if row["adapter_middle_psnr"] is not None])),
        }
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--stage25_root", type=Path, default=DEFAULT_STAGE25_ROOT)
    parser.add_argument("--stage33_manifest", type=Path, default=DEFAULT_STAGE33_MANIFEST)
    parser.add_argument("--stage49_csv", type=Path, default=DEFAULT_STAGE49_CSV)
    parser.add_argument("--stage50_csv", type=Path, default=DEFAULT_STAGE50_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--methods", nargs="*", default=["uniform", "rendered_prior_0p1"])
    parser.add_argument("--gaps", nargs="*", type=int, default=[1, 2, 3, 4, 8, 16])
    parser.add_argument("--bits", nargs="*", type=int, default=[8, 10, 12, 16])
    parser.add_argument("--hidden_dim", type=int, default=128)
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
    source_rows = read_stage49(args.stage49_csv, args.samples, args.methods, args.gaps)
    rates = read_stage50_rates(args.stage50_csv)
    rows = []
    eval_cache = {}
    for sample in args.samples:
        print(f"=== Stage51 sample={sample} ===", flush=True)
        frame_files = sorted((args.stage1_cache / sample / "frames").glob("*.png"))
        manifest_rows = read_manifest_rows(args.stage33_manifest, sample)
        checkpoint_path = args.stage25_root / sample / "stage25_best_anchor_adapter.safetensors"
        model = load_adapter(checkpoint_path, args.hidden_dim, device)
        anchor_maps = {}
        for bits in args.bits:
            anchor_maps[bits] = build_anchor_index(manifest_rows, device, bits)
        for row in [r for r in source_rows if r["sample"] == sample]:
            for bits in args.bits:
                cache_key = (sample, bits, tuple(row["indices"]))
                if cache_key not in eval_cache:
                    print(f"=== Stage51 sample={sample} method={row['method']} gap={row['reference_gap']} bits={bits} ===", flush=True)
                    eval_cache[cache_key] = selected_records_allow_keyframe_only(row["indices"], anchor_maps[bits], model, frame_files, opt, background)
                metrics = eval_cache[cache_key]
                flat = flatten_eval(sample, row["method"], row["reference_gap"], row["indices"], metrics, checkpoint_path)
                rate = rates[(sample, row["method"], row["reference_gap"], bits)]
                flat.update({
                    "bits": bits,
                    "payload_dtype": rate["payload_dtype"],
                    "raw_mib_per_frame": rate["raw_mib_per_frame"],
                    "zlib_mib_per_frame": rate["zlib_mib_per_frame"],
                    "theoretical_bitpacked_mib_per_frame": rate["theoretical_bitpacked_mib_per_frame"],
                    "zlib_savings_percent_vs_raw_bitstream": rate["zlib_savings_percent_vs_raw_bitstream"],
                    "max_roundtrip_abs_diff": rate["max_roundtrip_abs_diff"],
                })
                rows.append(flat)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = args.summary_root / "stage51_high_rate_multibit_rd.csv"
    raw_comparison_csv = args.summary_root / "stage51_raw_selector_comparison.csv"
    zlib_comparison_csv = args.summary_root / "stage51_zlib_selector_comparison.csv"
    summary_path = args.summary_root / "stage51_high_rate_multibit_rd_summary.json"
    write_csv(rows, csv_path)
    raw_comparisons = selector_comparison(rows, "raw_mib_per_frame")
    zlib_comparisons = selector_comparison(rows, "zlib_mib_per_frame")
    write_comparison_csv(raw_comparisons, raw_comparison_csv)
    write_comparison_csv(zlib_comparisons, zlib_comparison_csv)
    plots = []
    for method in args.methods:
        for metric_key, ylabel, middle_only, filename in [
            ("adapter_all_psnr", "Adapter all-frame PSNR (dB)", False, f"stage51_{method}_zlib_all_psnr_rd.png"),
            ("adapter_middle_psnr", "Adapter middle-only PSNR (dB)", True, f"stage51_{method}_zlib_middle_psnr_rd.png"),
        ]:
            out_path = args.summary_root / filename
            plot_rd(rows, method, "zlib_mib_per_frame", metric_key, ylabel, f"Stage51 {method} Zlib Multi-Bit RD", out_path, include_gap1=not middle_only)
            plots.append(str(out_path))
    summary = {
        "stage": 51,
        "mode": "high-rate multi-bit RD",
        "stage49_csv": str(args.stage49_csv),
        "stage50_csv": str(args.stage50_csv),
        "bits": args.bits,
        "gaps": args.gaps,
        "csv": str(csv_path),
        "raw_comparison_csv": str(raw_comparison_csv),
        "zlib_comparison_csv": str(zlib_comparison_csv),
        "plots": plots,
        "rows": rows,
        "raw_comparisons": raw_comparisons,
        "zlib_comparisons": zlib_comparisons,
        "aggregates_by_bits_uniform": aggregate_by_bits(rows, "uniform"),
        "aggregates_by_bits_adaptive": aggregate_by_bits(rows, "rendered_prior_0p1"),
        "max_roundtrip_abs_diff": float(max(row["max_roundtrip_abs_diff"] for row in rows)),
        "notes": "Quality is rendered with anchors dequantized at the specified bit depth. Rates come from Stage50 actual raw/zlib multi-bit containers.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "plots": plots,
        "aggregates_by_bits_uniform": summary["aggregates_by_bits_uniform"],
        "aggregates_by_bits_adaptive": summary["aggregates_by_bits_adaptive"],
        "max_roundtrip_abs_diff": summary["max_roundtrip_abs_diff"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
