import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments/stage61_davis_anchor_export_data_full/stage61_davis_anchor_export_manifest.csv"
DEFAULT_ADAPTER = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage77_qbit_full_video_anchor_only_rd_sweep"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import uniform_indices  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import load_adapter, maybe_quantize_anchor  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import selected_records  # noqa: E402
from scripts.run_stage66_davis_feedforward_selector_dataset import load_sequence_anchors, read_gap1_manifest  # noqa: E402


ROW_FIELDS = [
    "sample",
    "codec",
    "bits",
    "frame_gap",
    "method",
    "total_frames",
    "keyframe_count",
    "keyframe_ratio",
    "static_anchor_mib",
    "static_anchor_mib_with_metadata",
    "static_anchor_mib_per_frame",
    "static_anchor_mib_per_frame_with_metadata",
    "all_psnr",
    "middle_psnr",
    "given_psnr",
    "all_ssim",
    "middle_ssim",
    "given_ssim",
]

SUMMARY_FIELDS = [
    "codec",
    "bits",
    "frame_gap",
    "method",
    "sequence_count",
    "frame_count",
    "mean_static_anchor_mib_per_frame_with_metadata",
    "mean_all_psnr",
    "mean_middle_psnr",
    "mean_given_psnr",
    "mean_all_ssim",
    "mean_middle_ssim",
    "mean_given_ssim",
    "delta_all_vs_q8",
    "delta_middle_vs_q8",
]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def codec_bits(codec):
    if codec.startswith("q"):
        return int(codec[1:])
    raise ValueError(codec)


def payload_mib(values, bits):
    return values * bits / 8.0 / (1024.0 * 1024.0)


def payload_with_metadata_mib(values, bits):
    metadata_bytes = 13 * 2 * 2
    return (values * bits / 8.0 + metadata_bytes) / (1024.0 * 1024.0)


def metric(metrics, method, scope, key):
    return metrics[method][scope][key]


def quantize_map(anchor_map, bits):
    if bits <= 0:
        return anchor_map
    return {idx: maybe_quantize_anchor(anchor, bits) for idx, anchor in anchor_map.items()}


def evaluate_sequence(sample, anchor_map, rgb_paths, codecs, gaps, model, opt, background):
    indices = sorted(anchor_map)
    frame_files = [rgb_paths[idx] for idx in indices]
    first_attrs = flatten_static_anchor(anchor_map[indices[0]])
    values_per_anchor = int(first_attrs.shape[1] * first_attrs.shape[2])
    out = []
    for codec in codecs:
        bits = codec_bits(codec)
        print(f"=== Stage77 {sample} codec={codec} ===", flush=True)
        qmap = quantize_map(anchor_map, bits)
        anchor_mib = payload_mib(values_per_anchor, bits)
        anchor_mib_meta = payload_with_metadata_mib(values_per_anchor, bits)
        for gap in gaps:
            selected = uniform_indices(len(indices), gap)
            metrics = selected_records(selected, qmap, model, frame_files, opt, background)
            total_frames = len(indices)
            keyframes = len(selected)
            for method in ["linear", "adapter"]:
                out.append({
                    "sample": sample,
                    "codec": codec,
                    "bits": bits,
                    "frame_gap": gap,
                    "method": method,
                    "total_frames": total_frames,
                    "keyframe_count": keyframes,
                    "keyframe_ratio": keyframes / total_frames,
                    "static_anchor_mib": anchor_mib,
                    "static_anchor_mib_with_metadata": anchor_mib_meta,
                    "static_anchor_mib_per_frame": anchor_mib * keyframes / total_frames,
                    "static_anchor_mib_per_frame_with_metadata": anchor_mib_meta * keyframes / total_frames,
                    "all_psnr": metric(metrics, method, "all", "psnr_avg"),
                    "middle_psnr": metric(metrics, method, "middle_only", "psnr_avg"),
                    "given_psnr": metric(metrics, method, "given_keyframes", "psnr_avg"),
                    "all_ssim": metric(metrics, method, "all", "ssim_avg"),
                    "middle_ssim": metric(metrics, method, "middle_only", "ssim_avg"),
                    "given_ssim": metric(metrics, method, "given_keyframes", "ssim_avg"),
                })
        del qmap
        if background.device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def weighted_mean(items, value_key):
    weights = np.array([float(row["total_frames"]) for row in items], dtype=np.float64)
    values = np.array([float(row[value_key]) for row in items], dtype=np.float64)
    return float(np.average(values, weights=weights))


def build_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["codec"], int(row["frame_gap"]), row["method"])].append(row)
    base = {}
    out = []
    for (codec, gap, method), items in sorted(grouped.items(), key=lambda item: (codec_bits(item[0][0]), item[0][1], item[0][2])):
        summary = {
            "codec": codec,
            "bits": codec_bits(codec),
            "frame_gap": gap,
            "method": method,
            "sequence_count": len(items),
            "frame_count": int(sum(int(row["total_frames"]) for row in items)),
            "mean_static_anchor_mib_per_frame_with_metadata": weighted_mean(items, "static_anchor_mib_per_frame_with_metadata"),
            "mean_all_psnr": weighted_mean(items, "all_psnr"),
            "mean_middle_psnr": weighted_mean(items, "middle_psnr"),
            "mean_given_psnr": weighted_mean(items, "given_psnr"),
            "mean_all_ssim": weighted_mean(items, "all_ssim"),
            "mean_middle_ssim": weighted_mean(items, "middle_ssim"),
            "mean_given_ssim": weighted_mean(items, "given_ssim"),
        }
        if codec == "q8":
            base[(gap, method)] = summary
        out.append(summary)
    for row in out:
        ref = base.get((row["frame_gap"], row["method"]))
        row["delta_all_vs_q8"] = row["mean_all_psnr"] - ref["mean_all_psnr"] if ref else None
        row["delta_middle_vs_q8"] = row["mean_middle_psnr"] - ref["mean_middle_psnr"] if ref else None
    return out


def write_report(summary_rows, path):
    lines = [
        "# Stage77 Q-Bit Full-Video Anchor-Only RD Sweep",
        "",
        "## Summary",
        "",
        "| codec | gap | method | MiB/frame | all PSNR | middle PSNR | given PSNR | delta all vs q8 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['codec']} | {row['frame_gap']} | {row['method']} | {row['mean_static_anchor_mib_per_frame_with_metadata']} | {row['mean_all_psnr']} | {row['mean_middle_psnr']} | {row['mean_given_psnr']} | {row['delta_all_vs_q8']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Scope matches Stage70 scoped DAVIS val sequences and uniform gaps.",
        "- q8 rows should reproduce Stage70 uniform anchor-only quality within weighted-average differences from aggregation.",
        "- q10/q12 rows show the quality-rate tradeoff after fixing the aggressive q8 keyframe quantization issue.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sequences", nargs="+", default=["bmx-trees", "car-shadow", "goat", "soapbox"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--codecs", nargs="+", default=["q8", "q10", "q12"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
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
    model = load_adapter(args.adapter, args.hidden_dim, device)
    rows_by_sequence = read_gap1_manifest(args.manifest, [args.split])
    rows = []
    for sequence in args.sequences:
        key = ("DAVIS", args.split, sequence)
        if key not in rows_by_sequence:
            raise RuntimeError(f"Missing gap1 anchors for {key}")
        sample = f"DAVIS/{args.split}/{sequence}"
        print(f"=== Stage77 loading {sample} ===", flush=True)
        indices, anchor_map, _qmap, _attrs_map, rgb_paths = load_sequence_anchors(rows_by_sequence[key], device, quant_bits=0)
        rows.extend(evaluate_sequence(sample, anchor_map, rgb_paths, args.codecs, args.gaps, model, opt, background))
        del anchor_map
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    summary_rows = build_summary(rows)
    rows_csv = args.summary_root / "stage77_qbit_full_video_anchor_only_rd_rows.csv"
    summary_csv = args.summary_root / "stage77_qbit_full_video_anchor_only_rd_summary.csv"
    report_md = args.summary_root / "stage77_qbit_full_video_anchor_only_rd_report.md"
    summary_json = args.summary_root / "stage77_qbit_full_video_anchor_only_rd_summary.json"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_report(summary_rows, report_md)
    summary = {
        "stage": 77,
        "mode": "q-bit full-video anchor-only RD sweep",
        "manifest": str(args.manifest),
        "adapter": str(args.adapter),
        "sequences": args.sequences,
        "gaps": args.gaps,
        "codecs": args.codecs,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "report_md": str(report_md),
        "summary_json": str(summary_json),
        "summary_rows": summary_rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_json),
        "summary_csv": str(summary_csv),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
