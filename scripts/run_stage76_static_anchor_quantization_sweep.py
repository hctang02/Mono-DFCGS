import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments/stage61_davis_anchor_export_data_full/stage61_davis_anchor_export_manifest.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage76_static_anchor_quantization_sweep"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import render_anchor  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import (  # noqa: E402
    frame_psnr,
    load_rgb_numpy,
    maybe_quantize_anchor,
    tensor_to_numpy_rgb,
)
from scripts.run_stage66_davis_feedforward_selector_dataset import load_sequence_anchors, read_gap1_manifest  # noqa: E402


ROW_FIELDS = [
    "sample",
    "codec",
    "bits",
    "frame_count",
    "gaussians_per_anchor",
    "values_per_anchor",
    "payload_mib_no_metadata",
    "payload_mib_with_fp16_min_scale",
    "psnr_512_avg",
    "psnr_512_min",
    "psnr_256_avg",
    "psnr_256_min",
]

SUMMARY_FIELDS = [
    "codec",
    "bits",
    "sequence_count",
    "frame_count",
    "gaussians_per_anchor",
    "values_per_anchor",
    "payload_mib_no_metadata",
    "payload_mib_with_fp16_min_scale",
    "psnr_512_avg",
    "psnr_512_min",
    "psnr_256_avg",
    "psnr_256_min",
    "delta_512_vs_float16",
    "delta_256_vs_float16",
]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def codec_bits(codec):
    if codec == "float16":
        return 16
    if codec.startswith("q"):
        return int(codec[1:])
    raise ValueError(codec)


def payload_mib(values, bits):
    return values * bits / 8.0 / (1024.0 * 1024.0)


def payload_with_metadata_mib(values, bits):
    # Per-channel min and scale for 13 static attributes, stored as fp16.
    metadata_bytes = 13 * 2 * 2
    return (values * bits / 8.0 + metadata_bytes) / (1024.0 * 1024.0)


def render_numpy(anchor, background, opt):
    return tensor_to_numpy_rgb(render_anchor(anchor, background, opt))


def psnr_256(ref, pred):
    ref_small = cv2.resize(ref, (256, 256), interpolation=cv2.INTER_LINEAR)
    pred_small = cv2.resize(pred, (256, 256), interpolation=cv2.INTER_LINEAR)
    return frame_psnr(ref_small, pred_small)


def summarize_values(values):
    return float(np.mean(values)), float(np.min(values))


def evaluate_sequence(sample, rows, codecs, device, opt, background):
    indices, anchor_map, _qmap, _attrs_map, rgb_paths = load_sequence_anchors(rows, device, quant_bits=0)
    if not indices:
        raise RuntimeError(f"No anchors for {sample}")
    first_anchor = anchor_map[indices[0]]
    attrs = flatten_static_anchor(first_anchor)
    gaussians_per_anchor = int(attrs.shape[1])
    values_per_anchor = int(attrs.shape[1] * attrs.shape[2])
    out = []
    for codec in codecs:
        bits = codec_bits(codec)
        psnrs_512 = []
        psnrs_256 = []
        for idx in indices:
            anchor = anchor_map[idx]
            if codec != "float16":
                anchor = maybe_quantize_anchor(anchor, bits)
            ref = load_rgb_numpy(rgb_paths[idx], opt.image_height, opt.image_width)
            pred = render_numpy(anchor, background, opt)
            psnrs_512.append(frame_psnr(ref, pred))
            psnrs_256.append(psnr_256(ref, pred))
        avg512, min512 = summarize_values(psnrs_512)
        avg256, min256 = summarize_values(psnrs_256)
        out.append({
            "sample": sample,
            "codec": codec,
            "bits": bits,
            "frame_count": len(indices),
            "gaussians_per_anchor": gaussians_per_anchor,
            "values_per_anchor": values_per_anchor,
            "payload_mib_no_metadata": payload_mib(values_per_anchor, bits),
            "payload_mib_with_fp16_min_scale": payload_with_metadata_mib(values_per_anchor, bits),
            "psnr_512_avg": avg512,
            "psnr_512_min": min512,
            "psnr_256_avg": avg256,
            "psnr_256_min": min256,
        })
    del anchor_map
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def build_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["codec"]].append(row)
    float512 = None
    float256 = None
    out = []
    for codec in sorted(grouped, key=codec_bits):
        items = grouped[codec]
        weights = np.array([int(row["frame_count"]) for row in items], dtype=np.float64)
        psnr512 = np.array([float(row["psnr_512_avg"]) for row in items], dtype=np.float64)
        psnr256 = np.array([float(row["psnr_256_avg"]) for row in items], dtype=np.float64)
        summary = {
            "codec": codec,
            "bits": codec_bits(codec),
            "sequence_count": len(items),
            "frame_count": int(weights.sum()),
            "gaussians_per_anchor": int(items[0]["gaussians_per_anchor"]),
            "values_per_anchor": int(items[0]["values_per_anchor"]),
            "payload_mib_no_metadata": float(items[0]["payload_mib_no_metadata"]),
            "payload_mib_with_fp16_min_scale": float(items[0]["payload_mib_with_fp16_min_scale"]),
            "psnr_512_avg": float(np.average(psnr512, weights=weights)),
            "psnr_512_min": float(np.min([float(row["psnr_512_min"]) for row in items])),
            "psnr_256_avg": float(np.average(psnr256, weights=weights)),
            "psnr_256_min": float(np.min([float(row["psnr_256_min"]) for row in items])),
        }
        if codec == "float16":
            float512 = summary["psnr_512_avg"]
            float256 = summary["psnr_256_avg"]
        out.append(summary)
    for row in out:
        row["delta_512_vs_float16"] = row["psnr_512_avg"] - float512 if float512 is not None else None
        row["delta_256_vs_float16"] = row["psnr_256_avg"] - float256 if float256 is not None else None
    return out


def write_report(summary_rows, path):
    lines = [
        "# Stage76 Static Anchor Quantization Sweep",
        "",
        "## Summary",
        "",
        "| codec | MiB/anchor | PSNR 512 | delta 512 | PSNR 256 | delta 256 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['codec']} | {row['payload_mib_with_fp16_min_scale']} | {row['psnr_512_avg']} | {row['delta_512_vs_float16']} | {row['psnr_256_avg']} | {row['delta_256_vs_float16']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Rate is a per-keyframe static anchor payload estimate for 13 attributes per Gaussian.",
        "- Metadata includes fp16 per-channel min/scale for q-bit codecs and is negligible at this Gaussian count.",
        "- This stage measures direct keyframe render quality only; middle-frame predictor quality is evaluated separately.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sequences", nargs="+", default=["bmx-trees", "car-shadow", "goat", "soapbox"])
    parser.add_argument("--codecs", nargs="+", default=["float16", "q6", "q8", "q10", "q12", "q16"])
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
    rows_by_sequence = read_gap1_manifest(args.manifest, [args.split])
    rows = []
    for sequence in args.sequences:
        key = ("DAVIS", args.split, sequence)
        if key not in rows_by_sequence:
            raise RuntimeError(f"Missing gap1 anchors for {key}")
        sample = f"DAVIS/{args.split}/{sequence}"
        print(f"=== Stage76 quantization sweep {sample} ===", flush=True)
        rows.extend(evaluate_sequence(sample, rows_by_sequence[key], args.codecs, device, opt, background))
    summary_rows = build_summary(rows)
    rows_csv = args.summary_root / "stage76_static_anchor_quantization_rows.csv"
    summary_csv = args.summary_root / "stage76_static_anchor_quantization_summary.csv"
    report_md = args.summary_root / "stage76_static_anchor_quantization_report.md"
    summary_json = args.summary_root / "stage76_static_anchor_quantization_summary.json"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_report(summary_rows, report_md)
    summary = {
        "stage": 76,
        "mode": "static anchor quantization sweep",
        "manifest": str(args.manifest),
        "sequences": args.sequences,
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
