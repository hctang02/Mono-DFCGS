import argparse
import csv
import json
import math
import os
import sys
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from skimage.metrics import structural_similarity as ssim


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/streamsplat.safetensors")
DEFAULT_STAGE1_CACHE = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage1_streamsplat_fair_metrics/cache")
DEFAULT_SELECTION_CSV = REPO_ROOT / "experiments/stage11_keyframe_selection/stage11_keyframe_selection_summary.csv"
DEFAULT_HEAVY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage12_selected_keyframe_reconstruction")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage12_selected_keyframe_reconstruction"

sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from model.splat_model_inference import SplatModel  # noqa: E402


def load_model(opt, checkpoint, device):
    model = SplatModel(opt).to(device)
    state = load_file(str(checkpoint), device=device)
    new_state = OrderedDict()
    for key, value in state.items():
        if "_orig_mod." in key and not opt.compile:
            key = key.replace("_orig_mod.", "")
        new_state[key] = value
    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model


def get_image(path, h, w):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def get_depth(path, h, w):
    depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(path)
    return cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def frame_psnr(ref, pred):
    mse = float(np.mean((ref.astype(np.float32) / 255.0 - pred.astype(np.float32) / 255.0) ** 2))
    if mse <= 1e-12:
        return 100.0
    return -10.0 * math.log10(mse)


def compare_indexed_frames(ref_frames, pred_frames, indices):
    if not indices:
        return {"count": 0, "psnr_avg": None, "psnr_min": None, "ssim_avg": None, "ssim_min": None}
    psnrs = []
    ssims = []
    for idx in indices:
        ref = ref_frames[idx]
        pred = pred_frames[idx]
        psnrs.append(frame_psnr(ref, pred))
        ssims.append(float(ssim(ref.astype(np.float32) / 255.0, pred.astype(np.float32) / 255.0, channel_axis=2, data_range=1.0)))
    return {
        "count": len(indices),
        "psnr_avg": float(np.mean(psnrs)),
        "psnr_min": float(np.min(psnrs)),
        "ssim_avg": float(np.mean(ssims)),
        "ssim_min": float(np.min(ssims)),
    }


def tensor_payload_mib(pred_gs):
    total = 0
    shapes = {}
    for key, value in pred_gs.items():
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
            shapes[key] = list(value.shape)
    return total / (1024.0 * 1024.0), shapes


def parse_indices(text):
    return [int(part) for part in text.split() if part.strip()]


def read_selection(selection_csv, sample, method, gap):
    with open(selection_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sample"] == sample and row["method"] == method and int(row["reference_gap"]) == gap:
                row["reference_gap"] = int(row["reference_gap"])
                row["total_frames"] = int(row["total_frames"])
                row["keyframe_count"] = int(row["keyframe_count"])
                row["keyframe_ratio"] = float(row["keyframe_ratio"])
                row["estimated_q8_static_mib_per_frame"] = float(row["estimated_q8_static_mib_per_frame"])
                row["indices"] = parse_indices(row["indices"])
                return row
    raise RuntimeError(f"Selection not found: sample={sample} method={method} gap={gap}")


def load_cached_sample(sample, opt, cache_root):
    frame_dir = cache_root / sample / "frames"
    depth_dir = cache_root / sample / "depths_vitl_u16"
    frame_files = sorted(frame_dir.glob("*.png"))
    depth_files = sorted(depth_dir.glob("*.png"))
    if not frame_files or len(frame_files) != len(depth_files):
        raise RuntimeError(f"Missing cached frames/depths for {sample}: frames={len(frame_files)} depths={len(depth_files)}")
    frames = [get_image(path, opt.image_height, opt.image_width) for path in frame_files]
    depths = [get_depth(path, opt.image_height, opt.image_width) for path in depth_files]
    return frame_files, depth_files, frames, depths


def validate_indices(indices, total_frames, max_segment_length):
    if sorted(indices) != indices or len(set(indices)) != len(indices):
        raise ValueError(f"Indices must be sorted and unique: {indices}")
    if indices[0] != 0 or indices[-1] != total_frames - 1:
        raise ValueError(f"Indices must cover first and last frames: first={indices[0]} last={indices[-1]} total={total_frames}")
    segment_lengths = [b - a for a, b in zip(indices[:-1], indices[1:])]
    max_segment = max(segment_lengths)
    if max_segment_length > 0 and max_segment > max_segment_length:
        raise ValueError(f"Max segment {max_segment} exceeds --max_segment_length {max_segment_length}")
    return segment_lengths


def run_selected(args, device):
    selection = read_selection(args.selection_csv, args.sample, args.method, args.reference_gap)
    opt = Options()
    opt.resume = str(args.checkpoint)
    opt.compile = False
    opt.input_frames = 1
    opt.epoch = 0
    frame_files, depth_files, all_frames, all_depths = load_cached_sample(args.sample, opt, args.stage1_cache)
    total_frames = len(frame_files)
    indices = selection["indices"]
    segment_lengths = validate_indices(indices, total_frames, args.max_segment_length)

    out_dir = args.heavy_root / f"{args.sample}_{args.method}_gap{args.reference_gap}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(deepcopy(opt), args.checkpoint, device)
    pairs = list(zip(indices[:-1], indices[1:]))
    pairs_by_len = defaultdict(list)
    for pair in pairs:
        pairs_by_len[pair[1] - pair[0]].append(pair)

    pred_by_index = {}
    total_gs_mib = 0.0
    gs_shapes = {}
    with torch.no_grad():
        for seg_len, seg_pairs in sorted(pairs_by_len.items()):
            opt.output_frames = seg_len + 1
            model.opt.output_frames = seg_len + 1
            fixed_timestamps = torch.linspace(0.0, 1.0, seg_len + 1, device=device)
            for start in range(0, len(seg_pairs), args.batch_size):
                batch_pairs = seg_pairs[start:start + args.batch_size]
                batch_frames = []
                batch_depths = []
                timestamps = []
                for a, b in batch_pairs:
                    batch_frames.append(np.stack([all_frames[a], all_frames[b]], axis=0))
                    batch_depths.append(np.stack([all_depths[a], all_depths[b]], axis=0))
                    timestamps.append(fixed_timestamps.detach().cpu().numpy())
                frames = torch.from_numpy(np.stack(batch_frames, axis=0)).float().to(device) / 255.0
                frames = frames.permute(0, 1, 4, 2, 3)
                depths = torch.from_numpy(np.stack(batch_depths, axis=0)).float().to(device).unsqueeze(2)
                max_depth = depths.flatten(1).max(dim=1)[0][:, None, None, None, None]
                min_depth = depths.flatten(1).min(dim=1)[0][:, None, None, None, None]
                depths = (depths - min_depth) / (max_depth - min_depth + 1e-8)
                timestamps_t = torch.from_numpy(np.stack(timestamps, axis=0)).float().to(device)
                output = model({"frames": frames, "depths": depths, "timestamps": timestamps_t})
                pred = output["pred_frames"].detach().cpu()
                gs_out = model.forward_gaussians(frames, depths, timestamps_t)
                mib, shapes = tensor_payload_mib(gs_out["pred_gs"])
                total_gs_mib += mib
                gs_shapes = shapes
                for local_idx, (a, _b) in enumerate(batch_pairs):
                    for offset in range(seg_len + 1):
                        global_idx = a + offset
                        arr = pred[local_idx, offset].permute(1, 2, 0).numpy()
                        pred_by_index[global_idx] = (arr.clip(0, 1) * 255).astype(np.uint8)
                del output, gs_out, frames, depths, timestamps_t
                torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()

    if len(pred_by_index) != total_frames:
        missing = [idx for idx in range(total_frames) if idx not in pred_by_index]
        raise RuntimeError(f"Incomplete reconstruction: missing {missing[:10]} total_missing={len(missing)}")
    pred_frames = [pred_by_index[idx] for idx in range(total_frames)]
    all_indices = list(range(total_frames))
    keyframe_set = set(indices)
    middle_indices = [idx for idx in all_indices if idx not in keyframe_set]
    given_indices = [idx for idx in all_indices if idx in keyframe_set]

    per_frame = []
    for idx in all_indices:
        ref = all_frames[idx]
        pred = pred_frames[idx]
        per_frame.append({
            "frame_index": idx,
            "is_keyframe": idx in keyframe_set,
            "psnr": frame_psnr(ref, pred),
            "ssim": float(ssim(ref.astype(np.float32) / 255.0, pred.astype(np.float32) / 255.0, channel_axis=2, data_range=1.0)),
        })
    per_frame_path = out_dir / "per_frame_metrics.json"
    if args.save_per_frame:
        per_frame_path.write_text(json.dumps(per_frame, indent=2), encoding="utf-8")

    return {
        "stage": 12,
        "method": "selected-keyframe StreamSplat reconstruction smoke",
        "sample": args.sample,
        "selection_method": args.method,
        "reference_gap": args.reference_gap,
        "resolution": [opt.image_width, opt.image_height],
        "total_frames": total_frames,
        "selected_keyframes": indices,
        "keyframe_count": len(indices),
        "keyframe_ratio": len(indices) / total_frames,
        "segment_lengths": segment_lengths,
        "max_segment_length": max(segment_lengths),
        "pair_count": len(pairs),
        "complete": len(pred_by_index) == total_frames,
        "all": compare_indexed_frames(all_frames, pred_frames, all_indices),
        "middle_only": compare_indexed_frames(all_frames, pred_frames, middle_indices),
        "given_keyframes": compare_indexed_frames(all_frames, pred_frames, given_indices),
        "estimated_q8_static_mib_per_frame": selection["estimated_q8_static_mib_per_frame"],
        "raw_pred_gs_mib": total_gs_mib,
        "raw_pred_gs_mib_per_frame": total_gs_mib / max(total_frames, 1),
        "pred_gs_shapes_last_batch": gs_shapes,
        "heavy_output_dir": str(out_dir),
        "per_frame_metrics": str(per_frame_path) if args.save_per_frame else None,
        "notes": "Smoke evaluation for selected keyframes. It still uses StreamSplat RGB/depth-conditioned inference for each selected pair, not the final Gaussian-anchor-only decoder.",
    }


def write_csv(rows, path):
    fields = [
        "sample", "selection_method", "reference_gap", "total_frames", "keyframe_count",
        "keyframe_ratio", "max_segment_length", "estimated_q8_static_mib_per_frame",
        "all_psnr_avg", "all_ssim_avg", "middle_psnr_avg", "middle_ssim_avg",
        "given_psnr_avg", "given_ssim_avg", "raw_pred_gs_mib_per_frame",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sample": row["sample"],
                "selection_method": row["selection_method"],
                "reference_gap": row["reference_gap"],
                "total_frames": row["total_frames"],
                "keyframe_count": row["keyframe_count"],
                "keyframe_ratio": row["keyframe_ratio"],
                "max_segment_length": row["max_segment_length"],
                "estimated_q8_static_mib_per_frame": row["estimated_q8_static_mib_per_frame"],
                "all_psnr_avg": row["all"]["psnr_avg"],
                "all_ssim_avg": row["all"]["ssim_avg"],
                "middle_psnr_avg": row["middle_only"]["psnr_avg"],
                "middle_ssim_avg": row["middle_only"]["ssim_avg"],
                "given_psnr_avg": row["given_keyframes"]["psnr_avg"],
                "given_ssim_avg": row["given_keyframes"]["ssim_avg"],
                "raw_pred_gs_mib_per_frame": row["raw_pred_gs_mib_per_frame"],
            })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="robot")
    parser.add_argument("--method", default="rd_aware", choices=["uniform", "motion_aware", "gaussian_aware", "rd_aware"])
    parser.add_argument("--reference_gap", type=int, default=4)
    parser.add_argument("--selection_csv", type=Path, default=DEFAULT_SELECTION_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_segment_length", type=int, default=20)
    parser.add_argument("--save_per_frame", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
    args.summary_root.mkdir(parents=True, exist_ok=True)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = run_selected(args, device)
    stem = f"{args.sample}_{args.method}_gap{args.reference_gap}"
    summary_path = args.summary_root / f"{stem}_summary.json"
    csv_path = args.summary_root / "stage12_selected_keyframe_reconstruction_summary.csv"
    all_summary_path = args.summary_root / "stage12_selected_keyframe_reconstruction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    all_summary_path.write_text(json.dumps({"rows": [summary]}, indent=2), encoding="utf-8")
    write_csv([summary], csv_path)
    print(json.dumps({
        "summary": str(all_summary_path),
        "single_summary": str(summary_path),
        "csv": str(csv_path),
        "sample": args.sample,
        "method": args.method,
        "reference_gap": args.reference_gap,
        "all_psnr_avg": summary["all"]["psnr_avg"],
        "middle_psnr_avg": summary["middle_only"]["psnr_avg"],
        "max_segment_length": summary["max_segment_length"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
