import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE6_MANIFEST = REPO_ROOT / "experiments/stage6_real_anchor_dataset/stage6_real_anchor_dataset_manifest.csv"
DEFAULT_STAGE26_CSV = REPO_ROOT / "experiments/stage26_leave_one_out_full_video_rd/stage26_leave_one_out_full_video_rd.csv"
DEFAULT_HEAVY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage31_anchor_bitstream_prototype")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage31_anchor_bitstream_prototype"


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.anchor_bitstream import decode_anchor_bitstream, encode_anchor_bitstream  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, uniform_dequantize, uniform_quantize  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import uniform_indices  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import build_anchor_index, read_manifest_rows  # noqa: E402


def load_stage26_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gap = int(row["frame_gap"])
            total_frames = int(row["total_frames"])
            rows.append({
                "sample": row["sample"],
                "frame_gap": gap,
                "total_frames": total_frames,
                "keyframe_count": int(row["keyframe_count"]),
                "indices": uniform_indices(total_frames, gap),
                "stage26_primary_anchor_bytes": int(round(float(row["estimated_q8_static_mib_per_frame"]) * total_frames * 1024.0 * 1024.0)),
            })
    return rows


def direct_q8_anchor(anchor):
    attrs = flatten_static_anchor(anchor).detach().float().cpu()
    q, mins, scales = uniform_quantize(attrs, bits=8)
    return uniform_dequantize(q, mins, scales)


def roundtrip_error(source_anchors, decoded_anchors):
    max_abs = 0.0
    mse_values = []
    for source, decoded in zip(source_anchors, decoded_anchors):
        expected = direct_q8_anchor(source)
        actual = flatten_static_anchor(decoded).detach().float().cpu()
        diff = expected - actual
        max_abs = max(max_abs, float(diff.abs().max().item()))
        mse_values.append(float(torch.mean(diff ** 2).item()))
    return max_abs, float(np.mean(mse_values))


def write_csv(rows, path):
    fields = [
        "sample", "frame_gap", "total_frames", "keyframe_count", "raw_path", "zlib_path",
        "stage26_primary_anchor_bytes", "raw_bitstream_bytes", "zlib_bitstream_bytes",
        "raw_mib_per_frame", "zlib_mib_per_frame", "raw_overhead_bytes_vs_stage26_anchor",
        "zlib_savings_percent_vs_raw_bitstream", "max_roundtrip_abs_diff", "mean_roundtrip_mse",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage6_manifest", type=Path, default=DEFAULT_STAGE6_MANIFEST)
    parser.add_argument("--stage26_csv", type=Path, default=DEFAULT_STAGE26_CSV)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--gaps", nargs="*", type=int, default=[2, 4, 8, 16])
    return parser.parse_args()


def main():
    args = parse_args()
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    source_rows = [row for row in load_stage26_rows(args.stage26_csv) if row["sample"] in args.samples and row["frame_gap"] in args.gaps]

    anchor_maps = {}
    for sample in sorted({row["sample"] for row in source_rows}):
        manifest_rows = read_manifest_rows(args.stage6_manifest, sample)
        anchor_maps[sample] = build_anchor_index(manifest_rows, torch.device("cpu"), quant_bits=0)

    rows = []
    for row in source_rows:
        sample = row["sample"]
        gap = row["frame_gap"]
        indices = row["indices"]
        anchors = [anchor_maps[sample][idx] for idx in indices]
        raw_blob = encode_anchor_bitstream(anchors, indices, timestamps=indices, bits=8, compression="none")
        zlib_blob = encode_anchor_bitstream(anchors, indices, timestamps=indices, bits=8, compression="zlib")
        decoded, header = decode_anchor_bitstream(raw_blob)
        max_abs, mean_mse = roundtrip_error(anchors, decoded)
        if header["anchor_count"] != len(indices):
            raise RuntimeError("Decoded header anchor_count mismatch")

        raw_path = args.heavy_root / f"{sample}_gap{gap}_q8_raw.mdfcgs"
        zlib_path = args.heavy_root / f"{sample}_gap{gap}_q8_zlib.mdfcgs"
        raw_path.write_bytes(raw_blob)
        zlib_path.write_bytes(zlib_blob)
        raw_size = len(raw_blob)
        zlib_size = len(zlib_blob)
        rows.append({
            "sample": sample,
            "frame_gap": gap,
            "total_frames": row["total_frames"],
            "keyframe_count": row["keyframe_count"],
            "raw_path": str(raw_path),
            "zlib_path": str(zlib_path),
            "stage26_primary_anchor_bytes": row["stage26_primary_anchor_bytes"],
            "raw_bitstream_bytes": raw_size,
            "zlib_bitstream_bytes": zlib_size,
            "raw_mib_per_frame": raw_size / row["total_frames"] / (1024.0 * 1024.0),
            "zlib_mib_per_frame": zlib_size / row["total_frames"] / (1024.0 * 1024.0),
            "raw_overhead_bytes_vs_stage26_anchor": raw_size - row["stage26_primary_anchor_bytes"],
            "zlib_savings_percent_vs_raw_bitstream": 100.0 * (raw_size - zlib_size) / raw_size,
            "max_roundtrip_abs_diff": max_abs,
            "mean_roundtrip_mse": mean_mse,
        })

    csv_path = args.summary_root / "stage31_anchor_bitstream_prototype.csv"
    summary_path = args.summary_root / "stage31_anchor_bitstream_prototype_summary.json"
    write_csv(rows, csv_path)
    summary = {
        "stage": 31,
        "mode": "q8 static anchor bitstream prototype",
        "stage26_csv": str(args.stage26_csv),
        "heavy_root": str(args.heavy_root),
        "csv": str(csv_path),
        "rows": rows,
        "mean_raw_overhead_bytes_vs_stage26_anchor": float(np.mean([r["raw_overhead_bytes_vs_stage26_anchor"] for r in rows])),
        "mean_zlib_savings_percent_vs_raw_bitstream": float(np.mean([r["zlib_savings_percent_vs_raw_bitstream"] for r in rows])),
        "max_roundtrip_abs_diff": float(max(r["max_roundtrip_abs_diff"] for r in rows)),
        "mean_roundtrip_mse": float(np.mean([r["mean_roundtrip_mse"] for r in rows])),
        "notes": "Bitstream includes JSON header plus q8 payload. zlib is a generic compressor, not a learned entropy model.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "rows": len(rows),
        "mean_zlib_savings_percent_vs_raw_bitstream": summary["mean_zlib_savings_percent_vs_raw_bitstream"],
        "max_roundtrip_abs_diff": summary["max_roundtrip_abs_diff"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
