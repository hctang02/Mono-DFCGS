import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments/stage61_davis_anchor_export_data_full/stage61_davis_anchor_export_manifest.csv"
DEFAULT_ADAPTER = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage66_davis_feedforward_selector_dataset"


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import linear_anchor  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import anchor_to_device, load_adapter, maybe_quantize_anchor  # noqa: E402
from scripts.run_stage62_adapter_training_infra_v2 import normalize_manifest_row  # noqa: E402


FEATURE_FIELDS = [
    "segment_length",
    "middle_count",
    "normalized_left",
    "normalized_right",
    "endpoint_anchor_mse",
    "endpoint_anchor_l1",
    "endpoint_rgb_mse",
    "endpoint_opacity_mse",
    "endpoint_scale_mse",
    "endpoint_xyz_mse",
    "endpoint_rot_mse",
    "left_opacity_mean",
    "right_opacity_mean",
    "left_scale_mean",
    "right_scale_mean",
    "rgb_motion_mean",
    "rgb_motion_max",
    "rgb_endpoint_mse",
]

DATASET_FIELDS = [
    "dataset",
    "split",
    "selector_split",
    "sequence",
    "sample",
    "left_index",
    "right_index",
    *FEATURE_FIELDS,
    "label_adapter_mse_sum",
    "label_adapter_mse_mean",
    "label_adapter_mse_max",
    "label_linear_mse_sum",
    "label_linear_mse_mean",
    "label_adapter_delta_vs_linear_sum",
    "label_adapter_delta_vs_linear_mean",
    "label_log_adapter_mse_mean",
    "label_adapter_better_than_linear",
]

SEQUENCE_FIELDS = [
    "dataset",
    "split",
    "selector_split",
    "sequence",
    "sample",
    "frame_count",
    "candidate_segment_count",
    "emitted_rows",
    "label_adapter_mse_mean_avg",
    "label_linear_mse_mean_avg",
    "adapter_better_segment_count",
]

CORRELATION_FIELDS = ["scope", "feature", "pearson_log_adapter_mse_mean", "row_count"]


def read_gap1_manifest(path, splits):
    split_set = set(splits)
    rows_by_sequence = defaultdict(list)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = normalize_manifest_row(raw)
            if row["frame_gap"] != 1:
                continue
            if row["split"] not in split_set:
                continue
            if not Path(row["dataset_item"]).exists():
                continue
            rows_by_sequence[(row["dataset"], row["split"], row["sequence"])].append(row)
    for key in rows_by_sequence:
        rows_by_sequence[key].sort(key=lambda row: row["left_index"])
    return rows_by_sequence


def select_sequence_keys(rows_by_sequence, train_splits, eval_splits, max_train, max_eval, seed):
    rng = random.Random(seed)
    selected = []
    for selector_split, splits, limit in [("train", train_splits, max_train), ("eval", eval_splits, max_eval)]:
        keys = [key for key in rows_by_sequence if key[1] in set(splits)]
        keys.sort()
        rng.shuffle(keys)
        if limit > 0:
            keys = keys[:limit]
        selected.extend((selector_split, key) for key in sorted(keys))
    return selected


def load_sequence_anchors(rows, device, quant_bits):
    anchor_map = {}
    quantized_map = {}
    rgb_paths = {}
    for row in rows:
        item = torch.load(row["dataset_item"], map_location="cpu", weights_only=True)
        for index_key, anchor_key, rgb_key in [
            ("left_index", "left_anchor", "left_rgb_path"),
            ("right_index", "right_anchor", "right_rgb_path"),
        ]:
            frame_index = int(item[index_key])
            if frame_index in anchor_map:
                continue
            anchor = anchor_to_device(item[anchor_key], device)
            anchor_map[frame_index] = anchor
            quantized_map[frame_index] = maybe_quantize_anchor(anchor, quant_bits)
            rgb_paths[frame_index] = item[rgb_key]
    indices = sorted(anchor_map)
    missing = sorted(set(range(indices[0], indices[-1] + 1)) - set(indices)) if indices else []
    if missing:
        raise RuntimeError(f"Dense gap1 anchors are not contiguous, missing={missing[:10]}")
    attrs_map = {idx: flatten_static_anchor(anchor_map[idx]).detach() for idx in indices}
    return indices, anchor_map, quantized_map, attrs_map, rgb_paths


def load_small_frames(rgb_paths, image_size):
    frames = {}
    for idx, path in rgb_paths.items():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frames[idx] = image.astype(np.float32) / 255.0
    return frames


def rgb_motion_features(frames, a, b):
    if b <= a:
        return {"rgb_motion_mean": 0.0, "rgb_motion_max": 0.0, "rgb_endpoint_mse": 0.0}
    edge_scores = [float(np.mean(np.abs(frames[i + 1] - frames[i]))) for i in range(a, b)]
    return {
        "rgb_motion_mean": float(np.mean(edge_scores)),
        "rgb_motion_max": float(np.max(edge_scores)),
        "rgb_endpoint_mse": float(np.mean((frames[b] - frames[a]) ** 2)),
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


def segment_label(model, quantized_map, attrs_map, a, b):
    mids = [idx for idx in range(a + 1, b) if idx in attrs_map]
    if not mids:
        return None
    left = quantized_map[a]
    right = quantized_map[b]
    length = b - a
    adapter_losses = []
    linear_losses = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for idx in mids:
            t = torch.tensor([(idx - a) / length], dtype=torch.float32, device=device)
            target_attrs = attrs_map[idx]
            pred = model(left, right, t, apply_output_constraints=False)
            linear = linear_anchor(left, right, float((idx - a) / length))
            adapter_losses.append(float(torch.mean((flatten_static_anchor(pred) - target_attrs) ** 2).item()))
            linear_losses.append(float(torch.mean((flatten_static_anchor(linear) - target_attrs) ** 2).item()))
    adapter_sum = float(np.sum(adapter_losses))
    linear_sum = float(np.sum(linear_losses))
    adapter_mean = float(np.mean(adapter_losses))
    linear_mean = float(np.mean(linear_losses))
    return {
        "label_adapter_mse_sum": adapter_sum,
        "label_adapter_mse_mean": adapter_mean,
        "label_adapter_mse_max": float(np.max(adapter_losses)),
        "label_linear_mse_sum": linear_sum,
        "label_linear_mse_mean": linear_mean,
        "label_adapter_delta_vs_linear_sum": adapter_sum - linear_sum,
        "label_adapter_delta_vs_linear_mean": adapter_mean - linear_mean,
        "label_log_adapter_mse_mean": float(np.log10(adapter_mean + 1e-12)),
        "label_adapter_better_than_linear": "true" if adapter_mean < linear_mean else "false",
    }


def candidate_segments(indices, max_segment_length, stride, max_segments, rng):
    pairs = []
    index_set = set(indices)
    for a in indices:
        for length in range(2, max_segment_length + 1):
            b = a + length
            if b not in index_set:
                continue
            if stride > 1 and (a % stride) != 0:
                continue
            pairs.append((a, b))
    if max_segments > 0 and len(pairs) > max_segments:
        pairs = sorted(rng.sample(pairs, max_segments))
    return pairs


def pearson(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 2 or xs.std() <= 0 or ys.std() <= 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def feature_correlations(rows):
    out = []
    for scope in ["all", "train", "eval"]:
        scoped = rows if scope == "all" else [row for row in rows if row["selector_split"] == scope]
        labels = [row["label_log_adapter_mse_mean"] for row in scoped]
        for feature in FEATURE_FIELDS:
            out.append({
                "scope": scope,
                "feature": feature,
                "pearson_log_adapter_mse_mean": pearson([row[feature] for row in scoped], labels),
                "row_count": len(scoped),
            })
    return out


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--train_splits", nargs="*", default=["train"])
    parser.add_argument("--eval_splits", nargs="*", default=["val"])
    parser.add_argument("--max_train_sequences", type=int, default=8)
    parser.add_argument("--max_eval_sequences", type=int, default=4)
    parser.add_argument("--max_segment_length", type=int, default=16)
    parser.add_argument("--segment_stride", type=int, default=1)
    parser.add_argument("--max_segments_per_sequence", type=int, default=384)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    splits = list(dict.fromkeys(args.train_splits + args.eval_splits))
    rows_by_sequence = read_gap1_manifest(args.manifest, splits)
    selected = select_sequence_keys(rows_by_sequence, args.train_splits, args.eval_splits, args.max_train_sequences, args.max_eval_sequences, args.seed)
    if not selected:
        raise RuntimeError("No sequences selected for Stage66 dataset")
    model = load_adapter(args.adapter, args.hidden_dim, device)
    all_rows = []
    sequence_rows = []
    for order, (selector_split, key) in enumerate(selected):
        dataset, split, sequence = key
        sample = f"{dataset}/{split}/{sequence}"
        print(f"=== Stage66 {order + 1}/{len(selected)} {sample} selector_split={selector_split} ===", flush=True)
        indices, _anchor_map, quantized_map, attrs_map, rgb_paths = load_sequence_anchors(rows_by_sequence[key], device, args.quant_bits)
        frames = load_small_frames(rgb_paths, args.image_size)
        rng = random.Random(f"{args.seed}:{sample}")
        pairs = candidate_segments(indices, args.max_segment_length, args.segment_stride, args.max_segments_per_sequence, rng)
        emitted = []
        for a, b in pairs:
            label = segment_label(model, quantized_map, attrs_map, a, b)
            if label is None:
                continue
            row = {
                "dataset": dataset,
                "split": split,
                "selector_split": selector_split,
                "sequence": sequence,
                "sample": sample,
                "left_index": a,
                "right_index": b,
                "segment_length": b - a,
                "middle_count": b - a - 1,
                "normalized_left": a / max(indices[-1], 1),
                "normalized_right": b / max(indices[-1], 1),
                **anchor_features(attrs_map, a, b),
                **rgb_motion_features(frames, a, b),
                **label,
            }
            emitted.append(row)
            all_rows.append(row)
        label_means = [row["label_adapter_mse_mean"] for row in emitted]
        linear_means = [row["label_linear_mse_mean"] for row in emitted]
        sequence_rows.append({
            "dataset": dataset,
            "split": split,
            "selector_split": selector_split,
            "sequence": sequence,
            "sample": sample,
            "frame_count": len(indices),
            "candidate_segment_count": len(pairs),
            "emitted_rows": len(emitted),
            "label_adapter_mse_mean_avg": float(np.mean(label_means)) if label_means else None,
            "label_linear_mse_mean_avg": float(np.mean(linear_means)) if linear_means else None,
            "adapter_better_segment_count": sum(1 for row in emitted if row["label_adapter_better_than_linear"] == "true"),
        })
        del quantized_map, attrs_map, frames
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not all_rows:
        raise RuntimeError("Stage66 generated no dataset rows")

    dataset_csv = args.summary_root / "stage66_davis_selector_dataset.csv"
    sequence_csv = args.summary_root / "stage66_davis_selector_sequence_summary.csv"
    correlation_csv = args.summary_root / "stage66_davis_selector_feature_correlations.csv"
    summary_path = args.summary_root / "stage66_davis_selector_dataset_summary.json"
    correlations = feature_correlations(all_rows)
    write_csv(all_rows, dataset_csv, DATASET_FIELDS)
    write_csv(sequence_rows, sequence_csv, SEQUENCE_FIELDS)
    write_csv(correlations, correlation_csv, CORRELATION_FIELDS)
    train_rows = [row for row in all_rows if row["selector_split"] == "train"]
    eval_rows = [row for row in all_rows if row["selector_split"] == "eval"]
    summary = {
        "stage": 66,
        "mode": "DAVIS feed-forward selector segment-cost dataset",
        "manifest": str(args.manifest),
        "adapter": str(args.adapter),
        "train_splits": args.train_splits,
        "eval_splits": args.eval_splits,
        "selected_sequences": [{"selector_split": selector_split, "dataset": key[0], "split": key[1], "sequence": key[2]} for selector_split, key in selected],
        "max_segment_length": args.max_segment_length,
        "segment_stride": args.segment_stride,
        "max_segments_per_sequence": args.max_segments_per_sequence,
        "image_size": args.image_size,
        "hidden_dim": args.hidden_dim,
        "quant_bits": args.quant_bits,
        "row_count": len(all_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "sequence_count": len(sequence_rows),
        "adapter_better_segment_count": sum(1 for row in all_rows if row["label_adapter_better_than_linear"] == "true"),
        "label_adapter_mse_mean_avg": float(np.mean([row["label_adapter_mse_mean"] for row in all_rows])),
        "label_linear_mse_mean_avg": float(np.mean([row["label_linear_mse_mean"] for row in all_rows])),
        "dataset_csv": str(dataset_csv),
        "sequence_csv": str(sequence_csv),
        "correlation_csv": str(correlation_csv),
        "notes": "Features are encoder-side only. Labels use Stage65 adapter errors against dense gap1 anchors as offline supervision and are not test-time inputs.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "dataset_csv": str(dataset_csv),
        "row_count": summary["row_count"],
        "train_rows": summary["train_rows"],
        "eval_rows": summary["eval_rows"],
        "sequence_count": summary["sequence_count"],
        "adapter_better_segment_count": summary["adapter_better_segment_count"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
