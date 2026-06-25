import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_CACHE = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage1_streamsplat_fair_metrics/cache")
DEFAULT_STAGE6_MANIFEST = REPO_ROOT / "experiments/stage6_real_anchor_dataset/stage6_real_anchor_dataset_manifest.csv"
DEFAULT_STAGE16_CSV = REPO_ROOT / "experiments/stage16_segment_error_keyframe_selection/stage16_segment_error_keyframe_selection_summary.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage30_dense_anchor_export_preflight"
SAMPLES = ["n3dv", "meetroom", "driving", "robot"]
GAUSSIANS_PER_ANCHOR = 36864
VALUES_PER_GAUSSIAN = 13
FP16_BYTES_PER_VALUE = 2
Q8_BYTES_PER_VALUE = 1


def mib(byte_count):
    return byte_count / (1024.0 * 1024.0)


def anchor_bytes(bytes_per_value):
    return GAUSSIANS_PER_ANCHOR * VALUES_PER_GAUSSIAN * bytes_per_value


def read_available_anchors(path, sample):
    indices = set()
    row_count = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sample"] != sample or not Path(row["dataset_item"]).exists():
                continue
            indices.add(int(row["left_index"]))
            indices.add(int(row["right_index"]))
            row_count += 1
    return sorted(indices), row_count


def read_stage16_selection_rows(path, samples):
    sample_set = set(samples)
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sample"] not in sample_set:
                continue
            rows.append({
                "sample": row["sample"],
                "method": row["method"],
                "reference_gap": int(row["reference_gap"]),
                "indices": [int(v) for v in row["indices"].split()],
            })
    return rows


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--stage6_manifest", type=Path, default=DEFAULT_STAGE6_MANIFEST)
    parser.add_argument("--stage16_csv", type=Path, default=DEFAULT_STAGE16_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--samples", nargs="*", default=SAMPLES)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    fp16_anchor_bytes = anchor_bytes(FP16_BYTES_PER_VALUE)
    q8_anchor_bytes = anchor_bytes(Q8_BYTES_PER_VALUE)

    coverage_rows = []
    available_by_sample = {}
    for sample in args.samples:
        frame_files = sorted((args.stage1_cache / sample / "frames").glob("*.png"))
        total_frames = len(frame_files)
        if total_frames <= 0:
            raise RuntimeError(f"Missing frame cache for {sample}")
        available, stage6_row_count = read_available_anchors(args.stage6_manifest, sample)
        available_set = set(available)
        missing = [idx for idx in range(total_frames) if idx not in available_set]
        available_by_sample[sample] = available_set
        coverage_rows.append({
            "sample": sample,
            "total_frames": total_frames,
            "stage6_pair_rows": stage6_row_count,
            "current_unique_anchor_count": len(available),
            "current_anchor_ratio": len(available) / total_frames,
            "missing_anchor_count": len(missing),
            "missing_anchor_indices": " ".join(str(idx) for idx in missing),
            "dense_unique_fp16_mib": mib(total_frames * fp16_anchor_bytes),
            "dense_unique_q8_mib": mib(total_frames * q8_anchor_bytes),
            "additional_unique_fp16_mib": mib(len(missing) * fp16_anchor_bytes),
            "additional_unique_q8_mib": mib(len(missing) * q8_anchor_bytes),
            "gap1_pair_count": total_frames - 1,
            "gap1_pair_fp16_mib_if_no_dedup": mib((total_frames - 1) * 2 * fp16_anchor_bytes),
        })

    selection_rows = []
    for row in read_stage16_selection_rows(args.stage16_csv, args.samples):
        available = available_by_sample[row["sample"]]
        missing = [idx for idx in row["indices"] if idx not in available]
        selection_rows.append({
            "sample": row["sample"],
            "method": row["method"],
            "reference_gap": row["reference_gap"],
            "selected_keyframe_count": len(row["indices"]),
            "unavailable_selected_count": len(missing),
            "unavailable_selected_indices": " ".join(str(idx) for idx in missing),
            "is_directly_supported_by_stage6": len(missing) == 0,
        })

    coverage_csv = args.summary_root / "stage30_dense_anchor_coverage.csv"
    selection_csv = args.summary_root / "stage30_stage16_selection_compatibility.csv"
    summary_path = args.summary_root / "stage30_dense_anchor_export_preflight_summary.json"
    write_csv(coverage_rows, coverage_csv, [
        "sample", "total_frames", "stage6_pair_rows", "current_unique_anchor_count", "current_anchor_ratio",
        "missing_anchor_count", "missing_anchor_indices", "dense_unique_fp16_mib", "dense_unique_q8_mib",
        "additional_unique_fp16_mib", "additional_unique_q8_mib", "gap1_pair_count", "gap1_pair_fp16_mib_if_no_dedup",
    ])
    write_csv(selection_rows, selection_csv, [
        "sample", "method", "reference_gap", "selected_keyframe_count", "unavailable_selected_count",
        "unavailable_selected_indices", "is_directly_supported_by_stage6",
    ])

    total_frames = sum(row["total_frames"] for row in coverage_rows)
    total_missing = sum(row["missing_anchor_count"] for row in coverage_rows)
    unsupported = [row for row in selection_rows if row["unavailable_selected_count"] > 0]
    summary = {
        "stage": 30,
        "mode": "dense anchor export preflight",
        "coverage_csv": str(coverage_csv),
        "selection_compatibility_csv": str(selection_csv),
        "samples": args.samples,
        "gaussians_per_anchor": GAUSSIANS_PER_ANCHOR,
        "values_per_gaussian": VALUES_PER_GAUSSIAN,
        "fp16_bytes_per_anchor": fp16_anchor_bytes,
        "q8_bytes_per_anchor": q8_anchor_bytes,
        "total_frames": total_frames,
        "total_current_unique_anchors": sum(row["current_unique_anchor_count"] for row in coverage_rows),
        "total_missing_anchors_for_dense_coverage": total_missing,
        "total_dense_unique_fp16_mib": float(np.sum([row["dense_unique_fp16_mib"] for row in coverage_rows])),
        "total_dense_unique_q8_mib": float(np.sum([row["dense_unique_q8_mib"] for row in coverage_rows])),
        "total_additional_unique_fp16_mib": float(np.sum([row["additional_unique_fp16_mib"] for row in coverage_rows])),
        "total_additional_unique_q8_mib": float(np.sum([row["additional_unique_q8_mib"] for row in coverage_rows])),
        "total_gap1_pair_fp16_mib_if_no_dedup": float(np.sum([row["gap1_pair_fp16_mib_if_no_dedup"] for row in coverage_rows])),
        "stage16_selection_rows": len(selection_rows),
        "stage16_unsupported_rows_with_current_stage6": len(unsupported),
        "coverage_rows": coverage_rows,
        "selection_compatibility_rows": selection_rows,
        "recommended_next_step": "Export adjacent gap1 anchor pairs or unique per-frame anchors to cover odd frames before rerunning unconstrained keyframe selection.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "coverage_csv": str(coverage_csv),
        "selection_compatibility_csv": str(selection_csv),
        "total_missing_anchors_for_dense_coverage": total_missing,
        "stage16_unsupported_rows_with_current_stage6": len(unsupported),
        "total_additional_unique_fp16_mib": summary["total_additional_unique_fp16_mib"],
        "total_gap1_pair_fp16_mib_if_no_dedup": summary["total_gap1_pair_fp16_mib_if_no_dedup"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
