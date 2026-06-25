import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE6_MANIFEST = REPO_ROOT / "experiments/stage6_real_anchor_dataset/stage6_real_anchor_dataset_manifest.csv"
DEFAULT_STAGE26_CSV = REPO_ROOT / "experiments/stage26_leave_one_out_full_video_rd/stage26_leave_one_out_full_video_rd.csv"
DEFAULT_STAGE27_CSV = REPO_ROOT / "experiments/stage27_anchor_available_selector_rd/stage27_anchor_available_selector_rd.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage28_enhanced_rate_model_report"
GAUSSIANS_PER_ANCHOR = 36864
VALUES_PER_GAUSSIAN = 13
Q8_BYTES_PER_VALUE = 1
QPARAM_FLOAT_BYTES = 4
INDEX_BYTES = 2
TIMESTAMP_BYTES = 2
HEADER_BYTES = 128
FIELD_TABLE_BYTES = 64


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.gaussian_codec import flatten_static_anchor, uniform_quantize  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import uniform_indices  # noqa: E402


def mib(byte_count):
    return byte_count / (1024.0 * 1024.0)


def read_manifest_rows(path, sample):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sample"] != sample:
                continue
            if not Path(row["dataset_item"]).exists():
                continue
            row["frame_gap"] = int(row["frame_gap"])
            row["left_index"] = int(row["left_index"])
            row["right_index"] = int(row["right_index"])
            rows.append(row)
    return rows


def anchor_to_cpu(anchor):
    return {key: value.unsqueeze(0).float().cpu() for key, value in anchor.items()}


def build_anchor_histograms(manifest_rows, bits):
    histograms = {}
    value_counts = {}
    for row in sorted(manifest_rows, key=lambda r: (r["frame_gap"], r["left_index"])):
        item = torch.load(row["dataset_item"], map_location="cpu", weights_only=True)
        for side in ["left", "right"]:
            idx = int(item[f"{side}_index"])
            if idx in histograms:
                continue
            attrs = flatten_static_anchor(anchor_to_cpu(item[f"{side}_anchor"])).reshape(-1, VALUES_PER_GAUSSIAN)
            q, _, _ = uniform_quantize(attrs.unsqueeze(0), bits=bits)
            flat_q = q.reshape(-1).to(torch.int64)
            histograms[idx] = torch.bincount(flat_q, minlength=1 << bits).cpu().numpy().astype(np.int64)
            value_counts[idx] = int(flat_q.numel())
    return histograms, value_counts


def entropy_bits(hist):
    total = int(hist.sum())
    if total <= 0:
        return 0.0
    probs = hist[hist > 0].astype(np.float64) / total
    return float(-np.sum(probs * np.log2(probs)) * total)


def stage26_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gap = int(row["frame_gap"])
            total_frames = int(row["total_frames"])
            rows.append({
                "source_stage": "stage26",
                "sample": row["sample"],
                "method": "uniform",
                "reference_gap": gap,
                "total_frames": total_frames,
                "keyframe_count": int(row["keyframe_count"]),
                "indices": uniform_indices(total_frames, gap),
                "quality_adapter_all_psnr": float(row["adapter_all_psnr"]),
                "quality_adapter_middle_psnr": float(row["adapter_middle_psnr"]),
            })
    return rows


def stage27_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "source_stage": "stage27",
                "sample": row["sample"],
                "method": row["method"],
                "reference_gap": int(row["reference_gap"]),
                "total_frames": int(row["total_frames"]),
                "keyframe_count": int(row["keyframe_count"]),
                "indices": [int(v) for v in row["indices"].split()],
                "quality_adapter_all_psnr": float(row["adapter_all_psnr"]),
                "quality_adapter_middle_psnr": float(row["adapter_middle_psnr"]),
            })
    return rows


def estimate_row(row, histograms, value_counts):
    keyframe_count = row["keyframe_count"]
    total_frames = row["total_frames"]
    value_count = keyframe_count * GAUSSIANS_PER_ANCHOR * VALUES_PER_GAUSSIAN
    anchor_bytes = value_count * Q8_BYTES_PER_VALUE
    qparam_bytes = keyframe_count * VALUES_PER_GAUSSIAN * 2 * QPARAM_FLOAT_BYTES
    index_bytes = keyframe_count * INDEX_BYTES
    timestamp_bytes = keyframe_count * TIMESTAMP_BYTES
    metadata_bytes = HEADER_BYTES + FIELD_TABLE_BYTES
    container_bytes = anchor_bytes + qparam_bytes + index_bytes + timestamp_bytes + metadata_bytes

    missing = [idx for idx in row["indices"] if idx not in histograms]
    if missing:
        raise RuntimeError(f"Missing anchor histogram for {row['sample']} {row['source_stage']} {row['method']} gap={row['reference_gap']}: {missing[:10]}")
    hist = np.zeros(256, dtype=np.int64)
    entropy_value_count = 0
    for idx in row["indices"]:
        hist += histograms[idx]
        entropy_value_count += value_counts[idx]
    entropy_anchor_bytes = int(math.ceil(entropy_bits(hist) / 8.0))
    entropy_container_bytes = entropy_anchor_bytes + qparam_bytes + index_bytes + timestamp_bytes + metadata_bytes
    bits_per_value = (entropy_anchor_bytes * 8.0) / max(entropy_value_count, 1)

    return {
        **{key: row[key] for key in ["source_stage", "sample", "method", "reference_gap", "total_frames", "keyframe_count"]},
        "indices": " ".join(str(idx) for idx in row["indices"]),
        "quality_adapter_all_psnr": row["quality_adapter_all_psnr"],
        "quality_adapter_middle_psnr": row["quality_adapter_middle_psnr"],
        "primary_anchor_bytes": anchor_bytes,
        "primary_anchor_mib_per_frame": mib(anchor_bytes) / total_frames,
        "qparam_bytes": qparam_bytes,
        "index_bytes": index_bytes,
        "timestamp_bytes": timestamp_bytes,
        "metadata_bytes": metadata_bytes,
        "container_bytes": container_bytes,
        "container_mib_per_frame": mib(container_bytes) / total_frames,
        "container_overhead_bytes": container_bytes - anchor_bytes,
        "container_overhead_percent_vs_anchor": 100.0 * (container_bytes - anchor_bytes) / max(anchor_bytes, 1),
        "entropy_anchor_bytes": entropy_anchor_bytes,
        "entropy_anchor_mib_per_frame": mib(entropy_anchor_bytes) / total_frames,
        "entropy_container_bytes": entropy_container_bytes,
        "entropy_container_mib_per_frame": mib(entropy_container_bytes) / total_frames,
        "q8_zero_order_entropy_bits_per_value": bits_per_value,
        "entropy_savings_percent_vs_raw_q8_anchor": 100.0 * (anchor_bytes - entropy_anchor_bytes) / max(anchor_bytes, 1),
    }


def write_csv(rows, path):
    fields = [
        "source_stage", "sample", "method", "reference_gap", "total_frames", "keyframe_count", "indices",
        "quality_adapter_all_psnr", "quality_adapter_middle_psnr", "primary_anchor_bytes", "primary_anchor_mib_per_frame",
        "qparam_bytes", "index_bytes", "timestamp_bytes", "metadata_bytes", "container_bytes", "container_mib_per_frame",
        "container_overhead_bytes", "container_overhead_percent_vs_anchor", "entropy_anchor_bytes", "entropy_anchor_mib_per_frame",
        "entropy_container_bytes", "entropy_container_mib_per_frame", "q8_zero_order_entropy_bits_per_value",
        "entropy_savings_percent_vs_raw_q8_anchor",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    groups = {}
    for key in sorted({(row["source_stage"], row["method"]) for row in rows}):
        points = [row for row in rows if (row["source_stage"], row["method"]) == key]
        groups["/".join(key)] = {
            "count": len(points),
            "mean_primary_anchor_mib_per_frame": float(np.mean([row["primary_anchor_mib_per_frame"] for row in points])),
            "mean_container_mib_per_frame": float(np.mean([row["container_mib_per_frame"] for row in points])),
            "mean_entropy_container_mib_per_frame": float(np.mean([row["entropy_container_mib_per_frame"] for row in points])),
            "mean_overhead_percent_vs_anchor": float(np.mean([row["container_overhead_percent_vs_anchor"] for row in points])),
            "mean_entropy_bits_per_value": float(np.mean([row["q8_zero_order_entropy_bits_per_value"] for row in points])),
            "mean_entropy_savings_percent_vs_raw_q8_anchor": float(np.mean([row["entropy_savings_percent_vs_raw_q8_anchor"] for row in points])),
        }
    return groups


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage6_manifest", type=Path, default=DEFAULT_STAGE6_MANIFEST)
    parser.add_argument("--stage26_csv", type=Path, default=DEFAULT_STAGE26_CSV)
    parser.add_argument("--stage27_csv", type=Path, default=DEFAULT_STAGE27_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--bits", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    source_rows = stage26_rows(args.stage26_csv) + stage27_rows(args.stage27_csv)
    samples = sorted({row["sample"] for row in source_rows})
    hists_by_sample = {}
    counts_by_sample = {}
    for sample in samples:
        manifest_rows = read_manifest_rows(args.stage6_manifest, sample)
        hists_by_sample[sample], counts_by_sample[sample] = build_anchor_histograms(manifest_rows, args.bits)

    rows = [estimate_row(row, hists_by_sample[row["sample"]], counts_by_sample[row["sample"]]) for row in source_rows]
    csv_path = args.summary_root / "stage28_enhanced_rate_model_report.csv"
    summary_path = args.summary_root / "stage28_enhanced_rate_model_report_summary.json"
    write_csv(rows, csv_path)
    summary = {
        "stage": 28,
        "mode": "enhanced rate model report",
        "stage26_csv": str(args.stage26_csv),
        "stage27_csv": str(args.stage27_csv),
        "bits": args.bits,
        "assumptions": {
            "anchor_payload": "q8 static anchor values only: 36864 Gaussians * 13 values * 8 bits per keyframe",
            "quant_params": "per-keyframe per-field min and scale stored as float32: 13 * 2 * 4 bytes",
            "indices": "uint16 frame index per keyframe",
            "timestamps": "uint16 timestamp per keyframe",
            "metadata": f"fixed header {HEADER_BYTES} bytes + field table {FIELD_TABLE_BYTES} bytes per video",
            "entropy": "zero-order entropy of q8 symbols after per-anchor uniform quantization; no entropy model/header overhead beyond metadata",
        },
        "csv": str(csv_path),
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "csv": str(csv_path), "rows": len(rows), "aggregate": summary["aggregate"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
