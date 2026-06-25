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
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage11_keyframe_selection"
SAMPLES = ["n3dv", "meetroom", "driving", "robot"]


def read_image_small(path, size):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def frame_motion_scores(frame_files, image_size):
    frames = [read_image_small(path, image_size) for path in frame_files]
    scores = np.zeros(len(frames), dtype=np.float64)
    for i in range(1, len(frames) - 1):
        prev_diff = np.mean(np.abs(frames[i] - frames[i - 1]))
        next_diff = np.mean(np.abs(frames[i + 1] - frames[i]))
        scores[i] = 0.5 * (prev_diff + next_diff)
    return scores


def read_stage6_rows(path, sample, frame_gap):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sample"] != sample or int(row["frame_gap"]) != frame_gap:
                continue
            if Path(row["dataset_item"]).exists():
                rows.append(row)
    return rows


def gaussian_scores(total_frames, rows):
    scores = np.zeros(total_frames, dtype=np.float64)
    counts = np.zeros(total_frames, dtype=np.float64)
    for row in rows:
        item = torch.load(row["dataset_item"], map_location="cpu", weights_only=True)
        left = {key: value.unsqueeze(0).float() for key, value in item["left_anchor"].items()}
        right = {key: value.unsqueeze(0).float() for key, value in item["right_anchor"].items()}
        score = float(torch.mean((flatten_static_anchor(left) - flatten_static_anchor(right)) ** 2).item())
        a = int(row["left_index"])
        b = int(row["right_index"])
        for idx in (a, b):
            if 0 <= idx < total_frames:
                scores[idx] += score
                counts[idx] += 1.0
    valid = counts > 0
    scores[valid] /= counts[valid]
    return scores


def normalize(scores):
    out = scores.astype(np.float64).copy()
    if out.max() <= out.min():
        return np.zeros_like(out)
    return (out - out.min()) / (out.max() - out.min())


def uniform_indices(total_frames, gap):
    selected = list(range(0, total_frames, gap))
    if selected[-1] != total_frames - 1:
        selected.append(total_frames - 1)
    return selected


def topk_indices(scores, keyframe_count):
    total_frames = len(scores)
    if keyframe_count >= total_frames:
        return list(range(total_frames))
    internal = list(range(1, total_frames - 1))
    ranked = sorted(internal, key=lambda idx: (-scores[idx], idx))
    selected = [0, total_frames - 1] + ranked[:max(keyframe_count - 2, 0)]
    return sorted(set(selected))


def q8_static_mib_per_frame(keyframe_count, total_frames, gaussians_per_anchor=36864, values_per_gaussian=13):
    byte_count = keyframe_count * gaussians_per_anchor * values_per_gaussian
    return byte_count / max(total_frames, 1) / (1024.0 * 1024.0)


def run_sample(sample, args):
    frame_dir = args.stage1_cache / sample / "frames"
    frame_files = sorted(frame_dir.glob("*.png"))
    if not frame_files:
        raise RuntimeError(f"Missing frame cache for {sample}: {frame_dir}")
    total_frames = len(frame_files)
    motion = frame_motion_scores(frame_files, args.score_image_size)
    g_rows = read_stage6_rows(args.stage6_manifest, sample, frame_gap=2)
    gscore = gaussian_scores(total_frames, g_rows) if g_rows else np.zeros(total_frames, dtype=np.float64)
    rd_score = 0.7 * normalize(motion) + 0.3 * normalize(gscore)

    rows = []
    selections = []
    for gap in args.gaps:
        budget = len(uniform_indices(total_frames, gap))
        method_indices = {
            "uniform": uniform_indices(total_frames, gap),
            "motion_aware": topk_indices(motion, budget),
            "gaussian_aware": topk_indices(gscore, budget),
            "rd_aware": topk_indices(rd_score, budget),
        }
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
            }
            rows.append(row)
            selections.append({**row, "indices": indices})
    score_summary = {
        "sample": sample,
        "total_frames": total_frames,
        "motion_score_max": float(motion.max()),
        "motion_score_mean": float(motion.mean()),
        "gaussian_score_max": float(gscore.max()),
        "gaussian_score_mean": float(gscore.mean()),
        "gaussian_score_rows": len(g_rows),
    }
    return rows, selections, score_summary


def write_csv(rows, path):
    fields = [
        "sample", "method", "reference_gap", "total_frames", "keyframe_count",
        "keyframe_ratio", "estimated_q8_static_mib_per_frame", "indices",
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
    csv_path = args.summary_root / "stage11_keyframe_selection_summary.csv"
    json_path = args.summary_root / "stage11_keyframe_selection_summary.json"
    write_csv(all_rows, csv_path)
    summary = {
        "stage": 11,
        "mode": "keyframe selection baseline",
        "stage1_cache": str(args.stage1_cache),
        "stage6_manifest": str(args.stage6_manifest),
        "samples": args.samples,
        "gaps": args.gaps,
        "methods": ["uniform", "motion_aware", "gaussian_aware", "rd_aware"],
        "score_summaries": score_summaries,
        "selections": all_selections,
        "csv": str(csv_path),
        "notes": "Selection baseline only. Quality must be measured by running reconstruction with the selected keyframes in a later stage.",
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(json_path),
        "csv": str(csv_path),
        "rows": len(all_rows),
        "samples": args.samples,
        "gaps": args.gaps,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
