import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402


DEFAULT_STAGE1_CACHE = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage1_streamsplat_fair_metrics/cache")
DEFAULT_STAGE6_MANIFEST = REPO_ROOT / "experiments/stage6_real_anchor_dataset/stage6_real_anchor_dataset_manifest.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage16_segment_error_keyframe_selection"
SAMPLES = ["n3dv", "meetroom", "driving", "robot"]


def read_image_small(path, size):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def motion_edge_scores(frame_files, image_size):
    frames = [read_image_small(path, image_size) for path in frame_files]
    return np.asarray([
        float(np.mean(np.abs(frames[i + 1] - frames[i])))
        for i in range(len(frames) - 1)
    ], dtype=np.float64)


def read_stage6_rows(path, sample, frame_gap):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sample"] == sample and int(row["frame_gap"]) == frame_gap and Path(row["dataset_item"]).exists():
                rows.append(row)
    return rows


def gaussian_point_scores(total_frames, rows):
    scores = np.zeros(total_frames, dtype=np.float64)
    counts = np.zeros(total_frames, dtype=np.float64)
    for row in rows:
        item = torch.load(row["dataset_item"], map_location="cpu", weights_only=True)
        left = {key: value.unsqueeze(0).float() for key, value in item["left_anchor"].items()}
        right = {key: value.unsqueeze(0).float() for key, value in item["right_anchor"].items()}
        score = float(torch.mean((flatten_static_anchor(left) - flatten_static_anchor(right)) ** 2).item())
        for idx in (int(row["left_index"]), int(row["right_index"])):
            if 0 <= idx < total_frames:
                scores[idx] += score
                counts[idx] += 1.0
    valid = counts > 0
    scores[valid] /= counts[valid]
    if valid.any() and (~valid).any():
        xs = np.flatnonzero(valid)
        scores = np.interp(np.arange(total_frames), xs, scores[xs])
    return scores


def point_to_edge_scores(point_scores):
    return 0.5 * (point_scores[:-1] + point_scores[1:])


def normalize(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0 or scores.max() <= scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def uniform_indices(total_frames, gap):
    selected = list(range(0, total_frames, gap))
    if selected[-1] != total_frames - 1:
        selected.append(total_frames - 1)
    return selected


def segment_cost(edge_scores, a, b, max_segment_length):
    length = b - a
    if length <= 0:
        return 0.0
    edge_sum = float(edge_scores[a:b].sum())
    cost = edge_sum * length
    if max_segment_length > 0 and length > max_segment_length:
        cost += float((length - max_segment_length) ** 2) * (float(edge_scores.max()) + 1.0)
    return cost


def best_split_for_segment(edge_scores, a, b, max_segment_length):
    base = segment_cost(edge_scores, a, b, max_segment_length)
    best_idx = None
    best_gain = -float("inf")
    for idx in range(a + 1, b):
        gain = base - segment_cost(edge_scores, a, idx, max_segment_length) - segment_cost(edge_scores, idx, b, max_segment_length)
        if gain > best_gain:
            best_gain = gain
            best_idx = idx
    return best_idx, best_gain


def segment_error_greedy_indices(edge_scores, budget, max_segment_length):
    total_frames = len(edge_scores) + 1
    selected = {0, total_frames - 1}
    while len(selected) < budget:
        ordered = sorted(selected)
        best_idx = None
        best_gain = -float("inf")
        best_length = -1
        for a, b in zip(ordered[:-1], ordered[1:]):
            if b - a <= 1:
                continue
            idx, gain = best_split_for_segment(edge_scores, a, b, max_segment_length)
            if idx is None:
                continue
            length = b - a
            if gain > best_gain or (gain == best_gain and length > best_length):
                best_idx = idx
                best_gain = gain
                best_length = length
        if best_idx is None:
            break
        selected.add(best_idx)
    return sorted(selected)


def segment_stats(indices):
    lengths = [b - a for a, b in zip(indices[:-1], indices[1:])]
    return {
        "segment_lengths": " ".join(str(v) for v in lengths),
        "max_segment_length": int(max(lengths)),
        "mean_segment_length": float(np.mean(lengths)),
    }


def q8_static_mib_per_frame(keyframe_count, total_frames, gaussians_per_anchor=36864, values_per_gaussian=13):
    byte_count = keyframe_count * gaussians_per_anchor * values_per_gaussian
    return byte_count / max(total_frames, 1) / (1024.0 * 1024.0)


def run_sample(sample, args):
    frame_dir = args.stage1_cache / sample / "frames"
    frame_files = sorted(frame_dir.glob("*.png"))
    if not frame_files:
        raise RuntimeError(f"Missing frame cache for {sample}: {frame_dir}")
    total_frames = len(frame_files)
    motion_edges = normalize(motion_edge_scores(frame_files, args.score_image_size))
    g_rows = read_stage6_rows(args.stage6_manifest, sample, frame_gap=2)
    gaussian_edges = normalize(point_to_edge_scores(gaussian_point_scores(total_frames, g_rows))) if g_rows else np.zeros(total_frames - 1)
    rd_edges = args.motion_weight * motion_edges + (1.0 - args.motion_weight) * gaussian_edges
    edge_map = {
        "segment_motion": motion_edges,
        "segment_gaussian": gaussian_edges,
        "segment_rd": rd_edges,
    }

    rows = []
    selections = []
    score_summary = {
        "sample": sample,
        "total_frames": total_frames,
        "motion_edge_mean": float(motion_edges.mean()),
        "motion_edge_max": float(motion_edges.max()),
        "gaussian_edge_mean": float(gaussian_edges.mean()),
        "gaussian_edge_max": float(gaussian_edges.max()),
        "gaussian_score_rows": len(g_rows),
    }
    for gap in args.gaps:
        budget = len(uniform_indices(total_frames, gap))
        method_indices = {"uniform": uniform_indices(total_frames, gap)}
        max_segment_length = gap * args.max_segment_multiplier
        for method, edge_scores in edge_map.items():
            method_indices[method] = segment_error_greedy_indices(edge_scores, budget, max_segment_length)
        for method, indices in method_indices.items():
            row = {
                "sample": sample,
                "method": method,
                "reference_gap": gap,
                "total_frames": total_frames,
                "keyframe_count": len(indices),
                "keyframe_ratio": len(indices) / total_frames,
                "estimated_q8_static_mib_per_frame": q8_static_mib_per_frame(len(indices), total_frames),
                "indices": " ".join(str(idx) for idx in indices),
                **segment_stats(indices),
            }
            rows.append(row)
            selections.append({**row, "indices": indices})
    return rows, selections, score_summary


def write_csv(rows, path):
    fields = [
        "sample", "method", "reference_gap", "total_frames", "keyframe_count", "keyframe_ratio",
        "estimated_q8_static_mib_per_frame", "max_segment_length", "mean_segment_length",
        "segment_lengths", "indices",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--stage6_manifest", type=Path, default=DEFAULT_STAGE6_MANIFEST)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--samples", nargs="*", default=SAMPLES)
    parser.add_argument("--gaps", nargs="*", type=int, default=[4, 8, 16])
    parser.add_argument("--score_image_size", type=int, default=128)
    parser.add_argument("--max_segment_multiplier", type=int, default=2)
    parser.add_argument("--motion_weight", type=float, default=0.7)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_selections = []
    score_summaries = []
    for sample in args.samples:
        rows, selections, score_summary = run_sample(sample, args)
        all_rows.extend(rows)
        all_selections.extend(selections)
        score_summaries.append(score_summary)
    csv_path = args.summary_root / "stage16_segment_error_keyframe_selection_summary.csv"
    json_path = args.summary_root / "stage16_segment_error_keyframe_selection_summary.json"
    write_csv(all_rows, csv_path)
    summary = {
        "stage": 16,
        "mode": "segment-error-aware keyframe selection",
        "stage1_cache": str(args.stage1_cache),
        "stage6_manifest": str(args.stage6_manifest),
        "samples": args.samples,
        "gaps": args.gaps,
        "max_segment_multiplier": args.max_segment_multiplier,
        "motion_weight": args.motion_weight,
        "methods": ["uniform", "segment_motion", "segment_gaussian", "segment_rd"],
        "score_summaries": score_summaries,
        "selections": all_selections,
        "csv": str(csv_path),
        "notes": "Greedy segment-splitting selection optimizes estimated segment difficulty instead of frame-wise top-k scores.",
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(json_path), "csv": str(csv_path), "rows": len(all_rows)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
