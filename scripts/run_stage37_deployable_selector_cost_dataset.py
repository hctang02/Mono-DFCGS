import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_CACHE = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage1_streamsplat_fair_metrics/cache")
DEFAULT_STAGE25_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage25_leave_one_out_adapter_training")
DEFAULT_STAGE33_MANIFEST = REPO_ROOT / "experiments/stage33_dense_gap1_anchor_export/stage33_dense_gap1_anchor_manifest.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage37_deployable_selector_cost_dataset"


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import load_adapter  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import build_anchor_index, read_manifest_rows  # noqa: E402
from scripts.run_stage29_anchor_attribute_oracle_selector_rd import compute_cost_matrix  # noqa: E402


def load_small_frames(frame_files, size):
    frames = []
    for path in frame_files:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(image.astype(np.float32) / 255.0)
    return frames


def rgb_motion_features(frames, a, b):
    if b <= a:
        return {"rgb_motion_mean": 0.0, "rgb_motion_max": 0.0, "rgb_endpoint_mse": 0.0}
    edge_scores = [float(np.mean(np.abs(frames[i + 1] - frames[i]))) for i in range(a, b)]
    endpoint_mse = float(np.mean((frames[b] - frames[a]) ** 2))
    return {
        "rgb_motion_mean": float(np.mean(edge_scores)),
        "rgb_motion_max": float(np.max(edge_scores)),
        "rgb_endpoint_mse": endpoint_mse,
    }


def anchor_features(attrs_map, a, b):
    left = attrs_map[a]
    right = attrs_map[b]
    diff = right - left
    rgb_diff = diff[..., :3]
    opacity_diff = diff[..., 3:4]
    scale_diff = diff[..., 4:6]
    xyz_diff = diff[..., 6:9]
    rot_diff = diff[..., 9:13]
    return {
        "endpoint_anchor_mse": float(torch.mean(diff ** 2).item()),
        "endpoint_anchor_l1": float(torch.mean(diff.abs()).item()),
        "endpoint_rgb_mse": float(torch.mean(rgb_diff ** 2).item()),
        "endpoint_opacity_mse": float(torch.mean(opacity_diff ** 2).item()),
        "endpoint_scale_mse": float(torch.mean(scale_diff ** 2).item()),
        "endpoint_xyz_mse": float(torch.mean(xyz_diff ** 2).item()),
        "endpoint_rot_mse": float(torch.mean(rot_diff ** 2).item()),
        "left_opacity_mean": float(left[..., 3:4].mean().item()),
        "right_opacity_mean": float(right[..., 3:4].mean().item()),
        "left_scale_mean": float(left[..., 4:6].mean().item()),
        "right_scale_mean": float(right[..., 4:6].mean().item()),
    }


def write_csv(rows, path):
    fields = [
        "sample", "heldout_fold", "split", "left_index", "right_index", "segment_length", "normalized_left",
        "normalized_right", "label_anchor_attr_mse", "log_label_anchor_attr_mse", "endpoint_anchor_mse",
        "endpoint_anchor_l1", "endpoint_rgb_mse", "endpoint_opacity_mse", "endpoint_scale_mse", "endpoint_xyz_mse",
        "endpoint_rot_mse", "left_opacity_mean", "right_opacity_mean", "left_scale_mean", "right_scale_mean",
        "rgb_motion_mean", "rgb_motion_max", "rgb_endpoint_mse",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pearson(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 2 or xs.std() <= 0 or ys.std() <= 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--stage25_root", type=Path, default=DEFAULT_STAGE25_ROOT)
    parser.add_argument("--stage33_manifest", type=Path, default=DEFAULT_STAGE33_MANIFEST)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--max_segment_length", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = []
    sample_summaries = []
    for sample in args.samples:
        print(f"=== Stage37 sample={sample} ===", flush=True)
        frame_files = sorted((args.stage1_cache / sample / "frames").glob("*.png"))
        frames_small = load_small_frames(frame_files, args.image_size)
        manifest_rows = read_manifest_rows(args.stage33_manifest, sample)
        anchor_map = build_anchor_index(manifest_rows, device, args.quant_bits)
        attrs_map = {idx: flatten_static_anchor(anchor).detach().float().cpu() for idx, anchor in anchor_map.items()}
        checkpoint_path = args.stage25_root / sample / "stage25_best_anchor_adapter.safetensors"
        model = load_adapter(checkpoint_path, args.hidden_dim, device)
        indices, costs = compute_cost_matrix(model, anchor_map, args.max_segment_length)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        sample_rows = []
        for (i, j), label in costs.items():
            a = indices[i]
            b = indices[j]
            if b - a <= 1:
                continue
            base = {
                "sample": sample,
                "left_index": a,
                "right_index": b,
                "segment_length": b - a,
                "normalized_left": a / max(len(frame_files) - 1, 1),
                "normalized_right": b / max(len(frame_files) - 1, 1),
                "label_anchor_attr_mse": label,
                "log_label_anchor_attr_mse": float(np.log10(label + 1e-12)),
            }
            base.update(anchor_features(attrs_map, a, b))
            base.update(rgb_motion_features(frames_small, a, b))
            sample_rows.append(base)
        rows.extend(sample_rows)
        labels = [row["label_anchor_attr_mse"] for row in sample_rows]
        sample_summaries.append({
            "sample": sample,
            "row_count": len(sample_rows),
            "label_mean": float(np.mean(labels)),
            "label_min": float(np.min(labels)),
            "label_max": float(np.max(labels)),
            "corr_endpoint_anchor_mse": pearson([row["endpoint_anchor_mse"] for row in sample_rows], labels),
            "corr_rgb_motion_mean": pearson([row["rgb_motion_mean"] for row in sample_rows], labels),
            "corr_segment_length": pearson([row["segment_length"] for row in sample_rows], labels),
        })

    expanded = []
    for heldout in args.samples:
        for row in rows:
            expanded.append({
                "heldout_fold": heldout,
                "split": "eval" if row["sample"] == heldout else "train",
                **row,
            })
    csv_path = args.summary_root / "stage37_deployable_selector_cost_dataset.csv"
    summary_path = args.summary_root / "stage37_deployable_selector_cost_dataset_summary.json"
    write_csv(expanded, csv_path)
    summary = {
        "stage": 37,
        "mode": "deployable selector segment-cost dataset",
        "stage33_manifest": str(args.stage33_manifest),
        "samples": args.samples,
        "max_segment_length": args.max_segment_length,
        "image_size": args.image_size,
        "base_segment_rows": len(rows),
        "expanded_leave_one_out_rows": len(expanded),
        "csv": str(csv_path),
        "sample_summaries": sample_summaries,
        "feature_note": "Features use endpoint anchors, segment length and RGB motion only. Labels use dense intermediate anchors as oracle/proxy targets.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "base_segment_rows": len(rows),
        "expanded_leave_one_out_rows": len(expanded),
        "sample_summaries": sample_summaries,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
