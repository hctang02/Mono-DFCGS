import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE33_MANIFEST = REPO_ROOT / "experiments/stage33_dense_gap1_anchor_export/stage33_dense_gap1_anchor_manifest.csv"
DEFAULT_STAGE49_CSV = REPO_ROOT / "experiments/stage49_extended_adaptive_rd/stage49_extended_adaptive_rd.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage50_multibit_anchor_bitstream_prototype"


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.anchor_bitstream import decode_anchor_bitstream, encode_anchor_bitstream  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, uniform_dequantize, uniform_quantize  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import build_anchor_index, read_manifest_rows  # noqa: E402


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


def direct_quant_anchor(anchor, bits):
    attrs = flatten_static_anchor(anchor).detach().float().cpu()
    q, mins, scales = uniform_quantize(attrs, bits=bits)
    return uniform_dequantize(q, mins, scales)


def roundtrip_error_bits(source_anchors, decoded_anchors, bits):
    max_abs = 0.0
    mse_values = []
    for source, decoded in zip(source_anchors, decoded_anchors):
        expected = direct_quant_anchor(source, bits)
        actual = flatten_static_anchor(decoded).detach().float().cpu()
        diff = expected - actual
        max_abs = max(max_abs, float(diff.abs().max().item()))
        mse_values.append(float(torch.mean(diff ** 2).item()))
    return max_abs, float(np.mean(mse_values))


def payload_dtype(bits):
    return "uint8" if bits <= 8 else "uint16"


def theoretical_bitpacked_mib_per_frame(keyframe_count, total_frames, bits, gaussians_per_anchor=36864, values_per_gaussian=13):
    byte_count = (keyframe_count * gaussians_per_anchor * values_per_gaussian * bits + 7) // 8
    return byte_count / total_frames / (1024.0 * 1024.0)


def write_csv(rows, path):
    fields = [
        "sample", "method", "reference_gap", "bits", "payload_dtype", "total_frames", "keyframe_count",
        "raw_bitstream_bytes", "zlib_bitstream_bytes", "raw_mib_per_frame", "zlib_mib_per_frame",
        "theoretical_bitpacked_mib_per_frame", "zlib_savings_percent_vs_raw_bitstream", "max_roundtrip_abs_diff",
        "mean_roundtrip_mse",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    out = {}
    for bits in sorted({row["bits"] for row in rows}):
        bit_rows = [row for row in rows if row["bits"] == bits]
        out[str(bits)] = {
            "payload_dtype": payload_dtype(bits),
            "rows": len(bit_rows),
            "mean_raw_mib_per_frame": float(np.mean([row["raw_mib_per_frame"] for row in bit_rows])),
            "mean_zlib_mib_per_frame": float(np.mean([row["zlib_mib_per_frame"] for row in bit_rows])),
            "mean_theoretical_bitpacked_mib_per_frame": float(np.mean([row["theoretical_bitpacked_mib_per_frame"] for row in bit_rows])),
            "mean_zlib_savings_percent_vs_raw_bitstream": float(np.mean([row["zlib_savings_percent_vs_raw_bitstream"] for row in bit_rows])),
            "max_roundtrip_abs_diff": float(max(row["max_roundtrip_abs_diff"] for row in bit_rows)),
        }
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage33_manifest", type=Path, default=DEFAULT_STAGE33_MANIFEST)
    parser.add_argument("--stage49_csv", type=Path, default=DEFAULT_STAGE49_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--methods", nargs="*", default=["uniform", "rendered_prior_0p1"])
    parser.add_argument("--gaps", nargs="*", type=int, default=[1, 2, 3, 4, 8, 16])
    parser.add_argument("--bits", nargs="*", type=int, default=[6, 8, 10, 12, 16])
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    source_rows = read_stage49(args.stage49_csv, args.samples, args.methods, args.gaps)
    anchor_maps = {}
    for sample in sorted({row["sample"] for row in source_rows}):
        manifest_rows = read_manifest_rows(args.stage33_manifest, sample)
        anchor_maps[sample] = build_anchor_index(manifest_rows, torch.device("cpu"), quant_bits=0)

    rows = []
    for row in source_rows:
        anchors = [anchor_maps[row["sample"]][idx] for idx in row["indices"]]
        for bits in args.bits:
            raw_blob = encode_anchor_bitstream(
                anchors, row["indices"], timestamps=row["indices"], bits=bits, compression="none", payload_encoding="dtype"
            )
            zlib_blob = encode_anchor_bitstream(
                anchors, row["indices"], timestamps=row["indices"], bits=bits, compression="zlib", payload_encoding="dtype"
            )
            decoded, header = decode_anchor_bitstream(raw_blob)
            if int(header["bits"]) != bits or int(header["anchor_count"]) != len(row["indices"]):
                raise RuntimeError("Decoded header mismatch")
            max_abs, mean_mse = roundtrip_error_bits(anchors, decoded, bits)
            raw_size = len(raw_blob)
            zlib_size = len(zlib_blob)
            rows.append({
                "sample": row["sample"],
                "method": row["method"],
                "reference_gap": row["reference_gap"],
                "bits": bits,
                "payload_dtype": payload_dtype(bits),
                "total_frames": row["total_frames"],
                "keyframe_count": row["keyframe_count"],
                "raw_bitstream_bytes": raw_size,
                "zlib_bitstream_bytes": zlib_size,
                "raw_mib_per_frame": raw_size / row["total_frames"] / (1024.0 * 1024.0),
                "zlib_mib_per_frame": zlib_size / row["total_frames"] / (1024.0 * 1024.0),
                "theoretical_bitpacked_mib_per_frame": theoretical_bitpacked_mib_per_frame(row["keyframe_count"], row["total_frames"], bits),
                "zlib_savings_percent_vs_raw_bitstream": 100.0 * (raw_size - zlib_size) / raw_size,
                "max_roundtrip_abs_diff": max_abs,
                "mean_roundtrip_mse": mean_mse,
            })

    csv_path = args.summary_root / "stage50_multibit_anchor_bitstream_prototype.csv"
    summary_path = args.summary_root / "stage50_multibit_anchor_bitstream_prototype_summary.json"
    write_csv(rows, csv_path)
    summary = {
        "stage": 50,
        "mode": "multi-bit anchor bitstream prototype",
        "stage49_csv": str(args.stage49_csv),
        "stage33_manifest": str(args.stage33_manifest),
        "bits": args.bits,
        "csv": str(csv_path),
        "rows": rows,
        "aggregates_by_bits": aggregate(rows),
        "max_roundtrip_abs_diff": float(max(row["max_roundtrip_abs_diff"] for row in rows)),
        "notes": "bits<=8 use uint8 payload and bits>8 use uint16 payload. This is not bit-packed; q6 raw size is therefore a storage prototype, while theoretical_bitpacked_mib_per_frame reports the compact payload estimate.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "aggregates_by_bits": summary["aggregates_by_bits"],
        "max_roundtrip_abs_diff": summary["max_roundtrip_abs_diff"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
