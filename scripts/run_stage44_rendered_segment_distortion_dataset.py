import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_CACHE = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage1_streamsplat_fair_metrics/cache")
DEFAULT_STAGE25_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage25_leave_one_out_adapter_training")
DEFAULT_STAGE33_MANIFEST = REPO_ROOT / "experiments/stage33_dense_gap1_anchor_export/stage33_dense_gap1_anchor_manifest.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage44_rendered_segment_distortion_dataset"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import render_prediction  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import load_adapter  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import build_anchor_index, read_manifest_rows  # noqa: E402
from scripts.run_stage37_deployable_selector_cost_dataset import (  # noqa: E402
    anchor_features,
    load_small_frames,
    pearson,
    rgb_motion_features,
)


def psnr_from_mse(mse):
    if mse <= 1e-12:
        return 100.0
    return -10.0 * math.log10(mse)


def load_rgb_tensor(path, height, width, device):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.0
    return tensor.to(device)


def load_frame_tensors(frame_files, height, width, device):
    return [load_rgb_tensor(path, height, width, device) for path in frame_files]


def sampled_middle_indices(a, b, max_targets):
    middle = list(range(a + 1, b))
    if max_targets <= 0 or len(middle) <= max_targets:
        return middle
    chosen = np.linspace(0, len(middle) - 1, max_targets).round().astype(int).tolist()
    return [middle[idx] for idx in chosen]


def evaluate_segment(model, anchor_map, frame_tensors, a, b, target_indices, background, opt):
    if not target_indices:
        return {
            "sampled_target_count": 0,
            "target_indices": "",
            "adapter_mse_mean_sampled": 0.0,
            "adapter_mse_sum_est": 0.0,
            "adapter_psnr_mean_sampled": 100.0,
            "adapter_psnr_min_sampled": 100.0,
        }
    left = anchor_map[a]
    right = anchor_map[b]
    length = b - a
    mses = []
    psnrs = []
    with torch.no_grad():
        for frame_idx in target_indices:
            t = (frame_idx - a) / length
            pred = render_prediction(model, {"left": left, "right": right, "normalized_time": t}, background, opt).clamp(0.0, 1.0)
            mse = float(F.mse_loss(pred, frame_tensors[frame_idx]).item())
            mses.append(mse)
            psnrs.append(psnr_from_mse(mse))
    mse_mean = float(np.mean(mses))
    return {
        "sampled_target_count": len(target_indices),
        "target_indices": " ".join(str(idx) for idx in target_indices),
        "adapter_mse_mean_sampled": mse_mean,
        "adapter_mse_sum_est": mse_mean * max(b - a - 1, 0),
        "adapter_psnr_mean_sampled": float(np.mean(psnrs)),
        "adapter_psnr_min_sampled": float(np.min(psnrs)),
    }


def write_csv(rows, path):
    fields = [
        "sample", "heldout_fold", "split", "left_index", "right_index", "segment_length", "middle_count",
        "normalized_left", "normalized_right", "sampled_target_count", "target_indices", "adapter_mse_mean_sampled",
        "adapter_mse_sum_est", "log_adapter_mse_sum_est", "adapter_psnr_mean_sampled", "adapter_psnr_min_sampled",
        "endpoint_anchor_mse", "endpoint_anchor_l1", "endpoint_rgb_mse", "endpoint_opacity_mse", "endpoint_scale_mse",
        "endpoint_xyz_mse", "endpoint_rot_mse", "left_opacity_mean", "right_opacity_mean", "left_scale_mean",
        "right_scale_mean", "rgb_motion_mean", "rgb_motion_max", "rgb_endpoint_mse",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--stage25_root", type=Path, default=DEFAULT_STAGE25_ROOT)
    parser.add_argument("--stage33_manifest", type=Path, default=DEFAULT_STAGE33_MANIFEST)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--max_segment_length", type=int, default=32)
    parser.add_argument("--max_targets_per_segment", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    rows = []
    base_rows = []
    sample_summaries = []
    for sample in args.samples:
        print(f"=== Stage44 sample={sample} ===", flush=True)
        frame_files = sorted((args.stage1_cache / sample / "frames").glob("*.png"))
        frame_tensors = load_frame_tensors(frame_files, opt.image_height, opt.image_width, device)
        frames_small = load_small_frames(frame_files, args.image_size)
        manifest_rows = read_manifest_rows(args.stage33_manifest, sample)
        anchor_map = build_anchor_index(manifest_rows, device, args.quant_bits)
        attrs_map = {idx: flatten_static_anchor(anchor).detach().float().cpu() for idx, anchor in anchor_map.items()}
        checkpoint_path = args.stage25_root / sample / "stage25_best_anchor_adapter.safetensors"
        model = load_adapter(checkpoint_path, args.hidden_dim, device)
        sample_rows = []
        total_frames = len(frame_files)
        for a in range(total_frames):
            if a not in anchor_map:
                continue
            max_b = min(total_frames - 1, a + args.max_segment_length)
            for b in range(a + 1, max_b + 1):
                if b not in anchor_map:
                    continue
                targets = sampled_middle_indices(a, b, args.max_targets_per_segment)
                metric = evaluate_segment(model, anchor_map, frame_tensors, a, b, targets, background, opt)
                label = metric["adapter_mse_sum_est"]
                base = {
                    "sample": sample,
                    "left_index": a,
                    "right_index": b,
                    "segment_length": b - a,
                    "middle_count": max(b - a - 1, 0),
                    "normalized_left": a / max(total_frames - 1, 1),
                    "normalized_right": b / max(total_frames - 1, 1),
                    "log_adapter_mse_sum_est": float(np.log10(label + 1e-12)),
                }
                base.update(metric)
                base.update(anchor_features(attrs_map, a, b))
                base.update(rgb_motion_features(frames_small, a, b))
                sample_rows.append(base)
        del model
        del frame_tensors
        if device.type == "cuda":
            torch.cuda.empty_cache()
        base_rows.extend(sample_rows)
        labels = [row["adapter_mse_sum_est"] for row in sample_rows]
        nonzero = [row for row in sample_rows if row["middle_count"] > 0]
        sample_summaries.append({
            "sample": sample,
            "row_count": len(sample_rows),
            "rendered_target_evaluations": int(sum(row["sampled_target_count"] for row in sample_rows)),
            "label_mean": float(np.mean(labels)),
            "label_min": float(np.min(labels)),
            "label_max": float(np.max(labels)),
            "nonzero_segment_rows": len(nonzero),
            "corr_endpoint_anchor_mse": pearson([row["endpoint_anchor_mse"] for row in nonzero], [row["adapter_mse_sum_est"] for row in nonzero]),
            "corr_rgb_motion_mean": pearson([row["rgb_motion_mean"] for row in nonzero], [row["adapter_mse_sum_est"] for row in nonzero]),
            "corr_segment_length": pearson([row["segment_length"] for row in nonzero], [row["adapter_mse_sum_est"] for row in nonzero]),
        })

    for heldout in args.samples:
        for row in base_rows:
            rows.append({
                "heldout_fold": heldout,
                "split": "eval" if row["sample"] == heldout else "train",
                **row,
            })
    csv_path = args.summary_root / "stage44_rendered_segment_distortion_dataset.csv"
    summary_path = args.summary_root / "stage44_rendered_segment_distortion_dataset_summary.json"
    write_csv(rows, csv_path)
    summary = {
        "stage": 44,
        "mode": "rendered segment distortion dataset",
        "stage33_manifest": str(args.stage33_manifest),
        "samples": args.samples,
        "max_segment_length": args.max_segment_length,
        "max_targets_per_segment": args.max_targets_per_segment,
        "base_segment_rows": len(base_rows),
        "expanded_leave_one_out_rows": len(rows),
        "rendered_target_evaluations": int(sum(row["sampled_target_count"] for row in base_rows)),
        "csv": str(csv_path),
        "sample_summaries": sample_summaries,
        "label_note": "adapter_mse_sum_est = mean adapter RGB MSE over sampled middle targets multiplied by full middle-frame count. This approximates rendered segment distortion for fast selector research; rerun with max_targets_per_segment=0 for all middle frames.",
        "feature_note": "Features use only encoder-side endpoint RGB/Gaussian/temporal signals. Rendered distortion is used only as an offline training/oracle label.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "base_segment_rows": len(base_rows),
        "expanded_leave_one_out_rows": len(rows),
        "rendered_target_evaluations": summary["rendered_target_evaluations"],
        "sample_summaries": sample_summaries,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
