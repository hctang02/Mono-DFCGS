import argparse
import csv
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE33_MANIFEST = REPO_ROOT / "experiments/stage33_dense_gap1_anchor_export/stage33_dense_gap1_anchor_manifest.csv"
DEFAULT_STAGE49_CSV = REPO_ROOT / "experiments/stage49_extended_adaptive_rd/stage49_extended_adaptive_rd.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage57_compact_anchor_codec"


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.anchor_bitstream import MAGIC, _compact_json, _split_container, decode_anchor_bitstream, encode_anchor_bitstream  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, uniform_dequantize, uniform_quantize  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import build_anchor_index, read_manifest_rows  # noqa: E402


FIELDS = [
    "sample", "method", "reference_gap", "bits", "total_frames", "keyframe_count",
    "legacy_raw_bitstream_bytes", "legacy_zlib_bitstream_bytes",
    "compact_raw_bitstream_bytes", "compact_zlib_bitstream_bytes",
    "legacy_raw_mib_per_frame", "legacy_zlib_mib_per_frame",
    "compact_raw_mib_per_frame", "compact_zlib_mib_per_frame",
    "legacy_uncompressed_payload_bytes", "compact_uncompressed_payload_bytes",
    "theoretical_compact_payload_bytes", "legacy_raw_header_bytes", "compact_raw_header_bytes",
    "compact_payload_savings_percent_vs_legacy_payload",
    "compact_raw_savings_percent_vs_legacy_raw",
    "compact_zlib_savings_percent_vs_legacy_zlib",
    "max_roundtrip_abs_diff", "mean_roundtrip_mse", "indices",
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


def direct_quant_anchor(anchor, bits):
    attrs = flatten_static_anchor(anchor).detach().float().cpu()
    q, mins, scales = uniform_quantize(attrs, bits=bits)
    return uniform_dequantize(q, mins, scales)


def expected_quant_attrs(source_anchors, bits):
    return [direct_quant_anchor(anchor, bits) for anchor in source_anchors]


def roundtrip_error_attrs(expected_attrs, decoded_anchors):
    max_abs = 0.0
    mse_values = []
    for expected, decoded in zip(expected_attrs, decoded_anchors):
        actual = flatten_static_anchor(decoded).detach().float().cpu()
        diff = expected - actual
        max_abs = max(max_abs, float(diff.abs().max().item()))
        mse_values.append(float(torch.mean(diff ** 2).item()))
    return max_abs, float(np.mean(mse_values))


def mib_per_frame(byte_count, total_frames):
    return byte_count / total_frames / (1024.0 * 1024.0)


def compact_payload_bytes(anchors, bits):
    total = 0
    for anchor in anchors:
        value_count = int(flatten_static_anchor(anchor).numel())
        total += (value_count * bits + 7) // 8
    return total


def decode_and_check(blob, bits, indices):
    decoded, header = decode_anchor_bitstream(blob)
    if int(header["bits"]) != bits or int(header["anchor_count"]) != len(indices):
        raise RuntimeError("Decoded header mismatch")
    return header, decoded


def split_header_and_check(blob, bits, indices):
    header, _ = _split_container(blob)
    if int(header["bits"]) != bits or int(header["anchor_count"]) != len(indices):
        raise RuntimeError("Container header mismatch")
    return header


def encode_raw(anchors, indices, bits, payload_encoding):
    blob = encode_anchor_bitstream(
        anchors,
        indices,
        timestamps=indices,
        bits=bits,
        compression="none",
        payload_encoding=payload_encoding,
    )
    return blob, split_header_and_check(blob, bits, indices)


def encode_and_decode(anchors, indices, bits, payload_encoding):
    blob, _ = encode_raw(anchors, indices, bits, payload_encoding)
    header, decoded = decode_and_check(blob, bits, indices)
    return blob, header, decoded


def zlib_container_from_raw(raw_blob):
    header, payload = _split_container(raw_blob)
    if header.get("compression") != "none":
        raise ValueError("Expected an uncompressed raw container")
    zlib_header = dict(header)
    zlib_header["compression"] = "zlib"
    header_bytes = _compact_json(zlib_header)
    return MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + zlib.compress(payload, level=9)


def percentage_saving(old_value, new_value):
    if old_value == 0:
        return 0.0
    return 100.0 * (old_value - new_value) / old_value


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    out = {}
    for bits in sorted({row["bits"] for row in rows}):
        bit_rows = [row for row in rows if row["bits"] == bits]
        out[str(bits)] = {
            "rows": len(bit_rows),
            "mean_legacy_raw_mib_per_frame": float(np.mean([row["legacy_raw_mib_per_frame"] for row in bit_rows])),
            "mean_legacy_zlib_mib_per_frame": float(np.mean([row["legacy_zlib_mib_per_frame"] for row in bit_rows])),
            "mean_compact_raw_mib_per_frame": float(np.mean([row["compact_raw_mib_per_frame"] for row in bit_rows])),
            "mean_compact_zlib_mib_per_frame": float(np.mean([row["compact_zlib_mib_per_frame"] for row in bit_rows])),
            "mean_compact_payload_savings_percent_vs_legacy_payload": float(np.mean([
                row["compact_payload_savings_percent_vs_legacy_payload"] for row in bit_rows
            ])),
            "mean_compact_raw_savings_percent_vs_legacy_raw": float(np.mean([
                row["compact_raw_savings_percent_vs_legacy_raw"] for row in bit_rows
            ])),
            "mean_compact_zlib_savings_percent_vs_legacy_zlib": float(np.mean([
                row["compact_zlib_savings_percent_vs_legacy_zlib"] for row in bit_rows
            ])),
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
    parser.add_argument("--gaps", nargs="*", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--bits", nargs="*", type=int, default=[1, 2, 4, 6, 8, 10, 12, 16])
    parser.add_argument("--limit_rows", type=int, default=0)
    parser.add_argument("--verify_encodings", choices=["compact", "all"], default="compact")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)

    source_rows = read_stage49(args.stage49_csv, args.samples, args.methods, args.gaps)
    if args.limit_rows > 0:
        source_rows = source_rows[: args.limit_rows]
    anchor_maps = {}
    for sample in sorted({row["sample"] for row in source_rows}):
        manifest_rows = read_manifest_rows(args.stage33_manifest, sample)
        anchor_maps[sample] = build_anchor_index(manifest_rows, torch.device("cpu"), quant_bits=0)

    rows = []
    for row_idx, row in enumerate(source_rows, start=1):
        print(
            f"Stage57 row {row_idx}/{len(source_rows)} sample={row['sample']} method={row['method']} gap={row['reference_gap']}",
            flush=True,
        )
        anchors = [anchor_maps[row["sample"]][idx] for idx in row["indices"]]
        for bits in args.bits:
            print(f"  bits={bits}", flush=True)
            expected_attrs = expected_quant_attrs(anchors, bits)
            legacy_raw, legacy_header = encode_raw(anchors, row["indices"], bits, "dtype")
            legacy_zlib = zlib_container_from_raw(legacy_raw)
            legacy_zlib_header = split_header_and_check(legacy_zlib, bits, row["indices"])
            compact_raw, compact_header, compact_decoded = encode_and_decode(anchors, row["indices"], bits, "bitpack")
            compact_zlib = zlib_container_from_raw(compact_raw)
            compact_zlib_header, compact_zlib_decoded = decode_and_check(compact_zlib, bits, row["indices"])
            theoretical_compact_payload = compact_payload_bytes(anchors, bits)
            if theoretical_compact_payload != int(compact_header["uncompressed_payload_length"]):
                raise RuntimeError("Compact payload length does not match theoretical bitpacked payload")
            roundtrip_stats = [
                roundtrip_error_attrs(expected_attrs, compact_decoded),
                roundtrip_error_attrs(expected_attrs, compact_zlib_decoded),
            ]
            if args.verify_encodings == "all":
                _, legacy_decoded = decode_and_check(legacy_raw, bits, row["indices"])
                _, legacy_zlib_decoded = decode_and_check(legacy_zlib, bits, row["indices"])
                roundtrip_stats.extend([
                    roundtrip_error_attrs(expected_attrs, legacy_decoded),
                    roundtrip_error_attrs(expected_attrs, legacy_zlib_decoded),
                ])
            max_abs = max(stat[0] for stat in roundtrip_stats)
            mean_mse = float(np.mean([stat[1] for stat in roundtrip_stats]))
            legacy_payload = int(legacy_header["uncompressed_payload_length"])
            compact_payload = int(compact_header["uncompressed_payload_length"])
            rows.append({
                "sample": row["sample"],
                "method": row["method"],
                "reference_gap": row["reference_gap"],
                "bits": bits,
                "total_frames": row["total_frames"],
                "keyframe_count": row["keyframe_count"],
                "legacy_raw_bitstream_bytes": len(legacy_raw),
                "legacy_zlib_bitstream_bytes": len(legacy_zlib),
                "compact_raw_bitstream_bytes": len(compact_raw),
                "compact_zlib_bitstream_bytes": len(compact_zlib),
                "legacy_raw_mib_per_frame": mib_per_frame(len(legacy_raw), row["total_frames"]),
                "legacy_zlib_mib_per_frame": mib_per_frame(len(legacy_zlib), row["total_frames"]),
                "compact_raw_mib_per_frame": mib_per_frame(len(compact_raw), row["total_frames"]),
                "compact_zlib_mib_per_frame": mib_per_frame(len(compact_zlib), row["total_frames"]),
                "legacy_uncompressed_payload_bytes": legacy_payload,
                "compact_uncompressed_payload_bytes": compact_payload,
                "theoretical_compact_payload_bytes": theoretical_compact_payload,
                "legacy_raw_header_bytes": len(legacy_raw) - legacy_payload,
                "compact_raw_header_bytes": len(compact_raw) - compact_payload,
                "compact_payload_savings_percent_vs_legacy_payload": percentage_saving(legacy_payload, compact_payload),
                "compact_raw_savings_percent_vs_legacy_raw": percentage_saving(len(legacy_raw), len(compact_raw)),
                "compact_zlib_savings_percent_vs_legacy_zlib": percentage_saving(len(legacy_zlib), len(compact_zlib)),
                "max_roundtrip_abs_diff": max_abs,
                "mean_roundtrip_mse": mean_mse,
                "indices": " ".join(str(v) for v in row["indices"]),
            })

    csv_path = args.summary_root / "stage57_compact_anchor_codec.csv"
    summary_path = args.summary_root / "stage57_compact_anchor_codec_summary.json"
    write_csv(rows, csv_path)
    summary = {
        "stage": 57,
        "mode": "compact anchor codec true bit-packing",
        "stage49_csv": str(args.stage49_csv),
        "stage33_manifest": str(args.stage33_manifest),
        "samples": args.samples,
        "methods": args.methods,
        "gaps": args.gaps,
        "bits": args.bits,
        "verify_encodings": args.verify_encodings,
        "rows": len(rows),
        "csv": str(csv_path),
        "aggregates_by_bits": aggregate(rows),
        "max_roundtrip_abs_diff": float(max(row["max_roundtrip_abs_diff"] for row in rows)) if rows else None,
        "notes": (
            "compact uses payload_encoding=bitpack for q1-q16; legacy uses dtype storage "
            "matching the Stage50 prototype behavior. Roundtrip error is measured against direct quantize-dequantize anchors."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "rows": len(rows),
        "aggregates_by_bits": summary["aggregates_by_bits"],
        "max_roundtrip_abs_diff": summary["max_roundtrip_abs_diff"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
