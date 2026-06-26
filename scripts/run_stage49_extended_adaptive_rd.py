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
DEFAULT_STAGE44_CSV = REPO_ROOT / "experiments/stage44_rendered_segment_distortion_dataset/stage44_rendered_segment_distortion_dataset.csv"
DEFAULT_HEAVY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage49_extended_adaptive_rd")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage49_extended_adaptive_rd"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.anchor_bitstream import decode_anchor_bitstream, encode_anchor_bitstream  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import render_anchor  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import q8_static_mib_per_frame, uniform_indices  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import (  # noqa: E402
    frame_psnr,
    frame_ssim,
    load_adapter,
    load_rgb_numpy,
    summarize_metrics,
    tensor_to_numpy_rgb,
)
from scripts.run_stage27_anchor_available_selector_rd import build_anchor_index, flatten_eval, read_manifest_rows, selected_records  # noqa: E402
from scripts.run_stage36_dense_oracle_actual_bitstream_rd import roundtrip_error, safe_method_name  # noqa: E402
from scripts.run_stage39_predicted_selector_rd import dp_select_from_costs, segment_fields  # noqa: E402
from scripts.run_stage45_rendered_oracle_adaptive_selector_rd import read_stage44_costs  # noqa: E402


QUALITY_FIELDS = [
    "linear_all_psnr", "adapter_all_psnr", "delta_all_psnr", "linear_middle_psnr", "adapter_middle_psnr",
    "delta_middle_psnr", "linear_given_psnr", "adapter_given_psnr", "delta_given_psnr", "linear_all_ssim",
    "adapter_all_ssim", "linear_middle_ssim", "adapter_middle_ssim",
]


def calibrated_costs(costs, gap, alpha):
    out = {}
    for (a, b), cost in costs.items():
        length = b - a
        prior = ((length - gap) / gap) ** 2
        out[(a, b)] = float(cost + alpha * prior)
    return out


def encode_rates(anchor_map_cpu, sample, method, gap, indices, total_frames, heavy_root):
    anchors = [anchor_map_cpu[idx] for idx in indices]
    raw_blob = encode_anchor_bitstream(anchors, indices, timestamps=indices, bits=8, compression="none")
    zlib_blob = encode_anchor_bitstream(anchors, indices, timestamps=indices, bits=8, compression="zlib")
    decoded, header = decode_anchor_bitstream(raw_blob)
    if int(header["anchor_count"]) != len(indices):
        raise RuntimeError("Decoded header anchor_count mismatch")
    max_abs, mean_mse = roundtrip_error(anchors, decoded)
    stem = f"{sample}_{safe_method_name(method)}_gap{gap}"
    raw_path = heavy_root / f"{stem}_q8_raw.mdfcgs"
    zlib_path = heavy_root / f"{stem}_q8_zlib.mdfcgs"
    raw_path.write_bytes(raw_blob)
    zlib_path.write_bytes(zlib_blob)
    raw_size = len(raw_blob)
    zlib_size = len(zlib_blob)
    return {
        "raw_path": str(raw_path),
        "zlib_path": str(zlib_path),
        "raw_bitstream_bytes": raw_size,
        "zlib_bitstream_bytes": zlib_size,
        "raw_mib_per_frame": raw_size / total_frames / (1024.0 * 1024.0),
        "zlib_mib_per_frame": zlib_size / total_frames / (1024.0 * 1024.0),
        "zlib_savings_percent_vs_raw_bitstream": 100.0 * (raw_size - zlib_size) / raw_size,
        "max_roundtrip_abs_diff": max_abs,
        "mean_roundtrip_mse": mean_mse,
    }


def selected_records_allow_keyframe_only(indices, anchor_map, model, frame_files, opt, background):
    if any(b - a > 1 for a, b in zip(indices[:-1], indices[1:])):
        return selected_records(indices, anchor_map, model, frame_files, opt, background)
    height, width = opt.image_height, opt.image_width
    records = []
    with torch.no_grad():
        for idx in indices:
            ref = load_rgb_numpy(frame_files[idx], height, width)
            pred = tensor_to_numpy_rgb(render_anchor(anchor_map[idx], background, opt))
            records.append({"frame_index": idx, "is_keyframe": True, "psnr": frame_psnr(ref, pred), "ssim": frame_ssim(ref, pred)})
    keyframe_summary = summarize_metrics(records)
    empty = summarize_metrics([])
    return {
        "linear": {"all": keyframe_summary, "middle_only": empty, "given_keyframes": keyframe_summary},
        "adapter": {"all": keyframe_summary, "middle_only": empty, "given_keyframes": keyframe_summary},
        "delta_all_psnr": 0.0,
        "delta_middle_psnr": None,
        "delta_given_psnr": 0.0,
    }


def write_csv(rows, path):
    fields = [
        "sample", "method", "reference_gap", "total_frames", "keyframe_count", "keyframe_ratio",
        "estimated_q8_static_mib_per_frame", "raw_mib_per_frame", "zlib_mib_per_frame", "raw_bitstream_bytes",
        "zlib_bitstream_bytes", "zlib_savings_percent_vs_raw_bitstream", "raw_path", "zlib_path",
        "max_roundtrip_abs_diff", "mean_roundtrip_mse", "max_segment_length", "mean_segment_length", "segment_lengths",
        "indices", "selector_cost", "checkpoint", *QUALITY_FIELDS,
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_optional_float(value):
    if value in {None, ""}:
        return None
    return float(value)


def load_existing_csv(path):
    int_fields = {"reference_gap", "total_frames", "keyframe_count", "raw_bitstream_bytes", "zlib_bitstream_bytes", "max_segment_length"}
    float_fields = {
        "keyframe_ratio", "estimated_q8_static_mib_per_frame", "raw_mib_per_frame", "zlib_mib_per_frame",
        "zlib_savings_percent_vs_raw_bitstream", "max_roundtrip_abs_diff", "mean_roundtrip_mse", "mean_segment_length",
        "selector_cost", *QUALITY_FIELDS,
    }
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            for key in int_fields:
                parsed[key] = int(parsed[key])
            for key in float_fields:
                parsed[key] = _parse_optional_float(parsed[key])
            rows.append(parsed)
    return rows


def actual_selector_comparison(rows, rate_key):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["sample"], row["reference_gap"])][row["method"]] = row
    out = []
    for (sample, gap), methods in sorted(grouped.items()):
        if "uniform" not in methods or "rendered_prior_0p1" not in methods:
            continue
        uniform = methods["uniform"]
        adaptive = methods["rendered_prior_0p1"]
        out.append({
            "sample": sample,
            "reference_gap": gap,
            "rate_kind": rate_key,
            "uniform_rate_mib_per_frame": uniform[rate_key],
            "adaptive_rate_mib_per_frame": adaptive[rate_key],
            "rate_delta_mib_per_frame": adaptive[rate_key] - uniform[rate_key],
            "selector_delta_adapter_all_psnr": adaptive["adapter_all_psnr"] - uniform["adapter_all_psnr"],
            "selector_delta_adapter_middle_psnr": None if adaptive["adapter_middle_psnr"] is None or uniform["adapter_middle_psnr"] is None else adaptive["adapter_middle_psnr"] - uniform["adapter_middle_psnr"],
        })
    return out


def write_actual_comparison_csv(rows, path):
    fields = [
        "sample", "reference_gap", "rate_kind", "uniform_rate_mib_per_frame", "adaptive_rate_mib_per_frame",
        "rate_delta_mib_per_frame", "selector_delta_adapter_all_psnr", "selector_delta_adapter_middle_psnr",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_rd(rows, rate_key, metric_key, ylabel, title, out_path, include_gap1):
    grouped = defaultdict(list)
    for row in rows:
        if row[metric_key] is None:
            continue
        if not include_gap1 and row["reference_gap"] == 1:
            continue
        grouped[(row["sample"], row["method"])].append(row)
    samples = sorted({row["sample"] for row in rows})
    methods = ["uniform", "rendered_prior_0p1"]
    colors = {"uniform": "#1f77b4", "rendered_prior_0p1": "#d62728"}
    markers = {"uniform": "o", "rendered_prior_0p1": "^"}
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
        ax.set_xlabel({
            "estimated_q8_static_mib_per_frame": "Estimated q8 anchor MiB/frame",
            "raw_mib_per_frame": "Raw q8 bitstream MiB/frame",
            "zlib_mib_per_frame": "Zlib q8 bitstream MiB/frame",
        }[rate_key])
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(frameon=False)
    axes[0].set_ylabel(ylabel)
    axes[2].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def aggregate(comparisons):
    all_values = [row["selector_delta_adapter_all_psnr"] for row in comparisons]
    middle_values = [row["selector_delta_adapter_middle_psnr"] for row in comparisons if row["selector_delta_adapter_middle_psnr"] is not None]
    return {
        "count_all": len(all_values),
        "count_middle": len(middle_values),
        "positive_all_points": sum(1 for value in all_values if value > 0.0),
        "positive_middle_points": sum(1 for value in middle_values if value > 0.0),
        "mean_delta_all_psnr": float(np.mean(all_values)),
        "mean_delta_middle_psnr": float(np.mean(middle_values)),
        "min_delta_all_psnr": float(np.min(all_values)),
        "min_delta_middle_psnr": float(np.min(middle_values)),
        "mean_rate_delta_mib_per_frame": float(np.mean([row["rate_delta_mib_per_frame"] for row in comparisons])),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--stage25_root", type=Path, default=DEFAULT_STAGE25_ROOT)
    parser.add_argument("--stage33_manifest", type=Path, default=DEFAULT_STAGE33_MANIFEST)
    parser.add_argument("--stage44_csv", type=Path, default=DEFAULT_STAGE44_CSV)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--gaps", nargs="*", type=int, default=[1, 2, 3, 4, 8, 16])
    parser.add_argument("--adaptive_alpha", type=float, default=0.1)
    parser.add_argument("--max_segment_multiplier", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reuse_existing_csv", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.summary_root / "stage49_extended_adaptive_rd.csv"
    estimated_comparison_csv = args.summary_root / "stage49_estimated_selector_comparison.csv"
    raw_comparison_csv = args.summary_root / "stage49_raw_selector_comparison.csv"
    zlib_comparison_csv = args.summary_root / "stage49_zlib_selector_comparison.csv"
    summary_path = args.summary_root / "stage49_extended_adaptive_rd_summary.json"
    if args.reuse_existing_csv:
        rows = load_existing_csv(csv_path)
        selections = []
    else:
        device = torch.device(args.device)
        opt = Options()
        opt.compile = False
        opt.output_frames = 1
        opt.down_resolution = ()
        background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
        stage44_costs = read_stage44_costs(args.stage44_csv, args.samples)
        rows = []
        selections = []
        eval_cache = {}
        for sample in args.samples:
            print(f"=== Stage49 sample={sample} ===", flush=True)
            frame_files = sorted((args.stage1_cache / sample / "frames").glob("*.png"))
            manifest_rows = read_manifest_rows(args.stage33_manifest, sample)
            anchor_map = build_anchor_index(manifest_rows, device, args.quant_bits)
            anchor_map_cpu = build_anchor_index(manifest_rows, torch.device("cpu"), quant_bits=0)
            checkpoint_path = args.stage25_root / sample / "stage25_best_anchor_adapter.safetensors"
            model = load_adapter(checkpoint_path, args.hidden_dim, device)
            for gap in args.gaps:
                uniform = uniform_indices(len(frame_files), gap)
                budget = len(uniform)
                costs = calibrated_costs(stage44_costs[sample], gap, args.adaptive_alpha)
                selected, selector_cost = dp_select_from_costs(len(frame_files), costs, budget, gap * args.max_segment_multiplier)
                for method, indices, cost in [("uniform", uniform, None), ("rendered_prior_0p1", selected, selector_cost)]:
                    cache_key = (sample, tuple(indices))
                    if cache_key not in eval_cache:
                        print(f"=== Stage49 sample={sample} method={method} gap={gap} ===", flush=True)
                        eval_cache[cache_key] = selected_records_allow_keyframe_only(indices, anchor_map, model, frame_files, opt, background)
                    metrics = eval_cache[cache_key]
                    row = flatten_eval(sample, method, gap, indices, metrics, checkpoint_path)
                    row.update(segment_fields(indices))
                    row["estimated_q8_static_mib_per_frame"] = q8_static_mib_per_frame(len(indices), len(frame_files))
                    row["selector_cost"] = cost
                    row.update(encode_rates(anchor_map_cpu, sample, method, gap, indices, len(frame_files), args.heavy_root))
                    rows.append(row)
                    selections.append({"sample": sample, "method": method, "reference_gap": gap, "indices": indices, "selector_cost": cost})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        write_csv(rows, csv_path)
    estimated_comparisons = actual_selector_comparison(rows, "estimated_q8_static_mib_per_frame")
    raw_comparisons = actual_selector_comparison(rows, "raw_mib_per_frame")
    zlib_comparisons = actual_selector_comparison(rows, "zlib_mib_per_frame")
    write_actual_comparison_csv(estimated_comparisons, estimated_comparison_csv)
    write_actual_comparison_csv(raw_comparisons, raw_comparison_csv)
    write_actual_comparison_csv(zlib_comparisons, zlib_comparison_csv)
    plots = []
    for rate_key, rate_name in [
        ("estimated_q8_static_mib_per_frame", "estimated"),
        ("raw_mib_per_frame", "raw"),
        ("zlib_mib_per_frame", "zlib"),
    ]:
        for metric_key, ylabel, middle_only, filename in [
            ("adapter_all_psnr", "Adapter all-frame PSNR (dB)", False, f"stage49_{rate_name}_adapter_all_psnr_rd.png"),
            ("adapter_middle_psnr", "Adapter middle-only PSNR (dB)", True, f"stage49_{rate_name}_adapter_middle_psnr_rd.png"),
        ]:
            out_path = args.summary_root / filename
            plot_rd(
                rows,
                rate_key,
                metric_key,
                ylabel,
                f"Stage49 Extended Adaptive {rate_name.upper()} RD",
                out_path,
                include_gap1=not middle_only,
            )
            plots.append(str(out_path))
    summary = {
        "stage": 49,
        "mode": "extended adaptive RD with q8 actual bitstreams",
        "gaps": args.gaps,
        "adaptive_method": "rendered_prior_0p1",
        "adaptive_alpha": args.adaptive_alpha,
        "stage44_csv": str(args.stage44_csv),
        "stage33_manifest": str(args.stage33_manifest),
        "heavy_root": str(args.heavy_root),
        "csv": str(csv_path),
        "estimated_comparison_csv": str(estimated_comparison_csv),
        "raw_comparison_csv": str(raw_comparison_csv),
        "zlib_comparison_csv": str(zlib_comparison_csv),
        "plots": plots,
        "rows": rows,
        "selections": selections,
        "estimated_comparisons": estimated_comparisons,
        "raw_comparisons": raw_comparisons,
        "zlib_comparisons": zlib_comparisons,
        "zlib_aggregate": aggregate(zlib_comparisons),
        "mean_zlib_savings_percent_vs_raw_bitstream": float(np.mean([row["zlib_savings_percent_vs_raw_bitstream"] for row in rows])),
        "max_roundtrip_abs_diff": float(max(row["max_roundtrip_abs_diff"] for row in rows)),
        "notes": "All-frame RD includes gap1. Middle-only RD excludes gap1 because it has no non-keyframe middle frames.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "plots": plots,
        "zlib_aggregate": summary["zlib_aggregate"],
        "mean_zlib_savings_percent_vs_raw_bitstream": summary["mean_zlib_savings_percent_vs_raw_bitstream"],
        "max_roundtrip_abs_diff": summary["max_roundtrip_abs_diff"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
