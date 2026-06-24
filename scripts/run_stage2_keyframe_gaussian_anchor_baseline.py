import argparse
import csv
import json
import os
import sys
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/streamsplat.safetensors")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage2_keyframe_gaussian_anchor"
DEFAULT_STAGE1_CACHE = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage1_streamsplat_fair_metrics/cache")

SAMPLES = {
    "n3dv": "/mnt/hdd2tC/tmp/opencode/gt_reference_videos/n3dv_input_81f_560x336.mp4",
    "meetroom": "/mnt/hdd2tC/tmp/opencode/gt_reference_videos/meetroom_input_81f_560x336.mp4",
    "driving": "/mnt/ssd2tB/haocheng/NeoVerse/examples/videos/driving.mp4",
    "robot": "/mnt/ssd2tB/haocheng/NeoVerse/examples/videos/robot.mp4",
}

sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from model.splat_model_inference import SplatModel  # noqa: E402


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


def build_pairs(total_frames, max_gap):
    selected = list(range(0, total_frames, max_gap))
    if selected[-1] != total_frames - 1:
        selected.append(total_frames - 1)
    return selected, [(selected[i], selected[i + 1]) for i in range(len(selected) - 1)]


def normalize_depths(depths):
    max_depth = depths.flatten(1).max(dim=1)[0][:, None, None, None, None]
    min_depth = depths.flatten(1).min(dim=1)[0][:, None, None, None, None]
    return (depths - min_depth) / (max_depth - min_depth + 1e-8)


def split_pair_anchors(pred_gs):
    n = pred_gs["rgb"].shape[1]
    if n % 2 != 0:
        raise ValueError(f"Expected even Gaussian count, got {n}")
    half = n // 2
    anchors = []
    for start, end in [(0, half), (half, n)]:
        anchors.append({
            "rgb": pred_gs["rgb"][:, start:end].detach().cpu(),
            "opacity": pred_gs["opacity"][:, start:end].detach().cpu(),
            "scale": pred_gs["scale"][:, start:end].detach().cpu(),
            "xyz": pred_gs["xyz"][:, start:end].detach().cpu(),
            "rot": pred_gs["rot"][:, start:end].detach().cpu(),
        })
    return anchors


def select_anchor_fields(anchor, batch_index, profile):
    if profile == "static_anchor":
        return {
            "rgb": anchor["rgb"][batch_index],
            "opacity": anchor["opacity"][batch_index, :, :1],
            "scale": anchor["scale"][batch_index],
            "xyz": anchor["xyz"][batch_index, :, 0, :],
            "rot": anchor["rot"][batch_index, :, 0, :],
        }
    if profile == "full_half_anchor":
        return {
            "rgb": anchor["rgb"][batch_index],
            "opacity": anchor["opacity"][batch_index],
            "scale": anchor["scale"][batch_index],
            "xyz": anchor["xyz"][batch_index].reshape(anchor["xyz"].shape[1], -1),
            "rot": anchor["rot"][batch_index].reshape(anchor["rot"].shape[1], -1),
        }
    raise ValueError(profile)


def estimate_payload(fields, opacity_threshold, codec):
    opacity = fields["opacity"][:, 0]
    keep = opacity >= opacity_threshold
    kept = int(keep.sum().item())
    total = int(keep.numel())
    if kept == 0:
        return {
            "gaussians_total": total,
            "gaussians_kept": 0,
            "bytes": 0,
            "mib": 0.0,
        }
    values = {key: value[keep] for key, value in fields.items()}
    num_values = sum(int(value.numel()) for value in values.values())
    if codec == "float32":
        bytes_total = num_values * 4
    elif codec == "float16":
        bytes_total = num_values * 2
    elif codec.startswith("q"):
        bits = int(codec[1:])
        bytes_total = int(np.ceil(num_values * bits / 8.0))
    else:
        raise ValueError(codec)
    return {
        "gaussians_total": total,
        "gaussians_kept": kept,
        "bytes": bytes_total,
        "mib": bytes_total / (1024.0 * 1024.0),
    }


def load_cached_sample(sample, opt, cache_root):
    frame_dir = cache_root / sample / "frames"
    depth_dir = cache_root / sample / "depths_vitl_u16"
    frame_files = sorted(frame_dir.glob("*.png"))
    depth_files = sorted(depth_dir.glob("*.png"))
    if not frame_files or len(frame_files) != len(depth_files):
        raise RuntimeError(
            f"Missing cached frames/depths for {sample}. Run stage1 first. "
            f"frames={len(frame_files)} depths={len(depth_files)}"
        )
    frames = [get_image(path, opt.image_height, opt.image_width) for path in frame_files]
    depths = [get_depth(path, opt.image_height, opt.image_width) for path in depth_files]
    return frames, depths


def collect_unique_anchors(sample, gap, args, model, opt, device):
    frames_all, depths_all = load_cached_sample(sample, opt, args.cache_root)
    selected, pairs = build_pairs(len(frames_all), gap)
    pairs_by_len = defaultdict(list)
    for pair in pairs:
        pairs_by_len[pair[1] - pair[0]].append(pair)

    unique = {}
    with torch.no_grad():
        for _seg_len, seg_pairs in sorted(pairs_by_len.items()):
            for start in range(0, len(seg_pairs), args.batch_size):
                batch_pairs = seg_pairs[start:start + args.batch_size]
                batch_frames = []
                batch_depths = []
                for a, b in batch_pairs:
                    batch_frames.append(np.stack([frames_all[a], frames_all[b]], axis=0))
                    batch_depths.append(np.stack([depths_all[a], depths_all[b]], axis=0))
                frames = torch.from_numpy(np.stack(batch_frames, axis=0)).float().to(device) / 255.0
                frames = frames.permute(0, 1, 4, 2, 3)
                depths = torch.from_numpy(np.stack(batch_depths, axis=0)).float().to(device).unsqueeze(2)
                depths = normalize_depths(depths)
                timestamps = torch.tensor([[0.0, 1.0]] * len(batch_pairs), dtype=torch.float32, device=device)
                gs_out = model.forward_gaussians(frames, depths, timestamps)
                first_half, second_half = split_pair_anchors(gs_out["pred_gs"])
                for local_idx, (a, b) in enumerate(batch_pairs):
                    if a not in unique:
                        unique[a] = {
                            profile: select_anchor_fields(first_half, local_idx, profile)
                            for profile in args.profiles
                        }
                    if b not in unique:
                        unique[b] = {
                            profile: select_anchor_fields(second_half, local_idx, profile)
                            for profile in args.profiles
                        }
                del gs_out, frames, depths, timestamps
                torch.cuda.empty_cache()
    return selected, len(frames_all), unique


def run_sample_gap(sample, gap, args, model, opt, device):
    selected, total_frames, unique = collect_unique_anchors(sample, gap, args, model, opt, device)
    rows = []
    for profile in args.profiles:
        for opacity_threshold in args.opacity_thresholds:
            for codec in args.codecs:
                total_bytes = 0
                kept = 0
                total_gaussians = 0
                for idx in selected:
                    est = estimate_payload(unique[idx][profile], opacity_threshold, codec)
                    total_bytes += est["bytes"]
                    kept += est["gaussians_kept"]
                    total_gaussians += est["gaussians_total"]
                rows.append({
                    "sample": sample,
                    "frame_gap": gap,
                    "total_frames": total_frames,
                    "keyframe_count": len(selected),
                    "keyframe_ratio": len(selected) / max(total_frames, 1),
                    "profile": profile,
                    "codec": codec,
                    "opacity_threshold": opacity_threshold,
                    "gaussians_total": total_gaussians,
                    "gaussians_kept": kept,
                    "keep_ratio": kept / max(total_gaussians, 1),
                    "total_bytes": total_bytes,
                    "total_mib": total_bytes / (1024.0 * 1024.0),
                    "avg_mib_per_video_frame": total_bytes / (1024.0 * 1024.0) / max(total_frames, 1),
                    "note": "Estimated transmitted keyframe Gaussian anchor payload; decoder-generated intermediate Gaussians are excluded.",
                })
    return rows


def write_csv(rows, path):
    fields = [
        "sample",
        "frame_gap",
        "total_frames",
        "keyframe_count",
        "keyframe_ratio",
        "profile",
        "codec",
        "opacity_threshold",
        "gaussians_total",
        "gaussians_kept",
        "keep_ratio",
        "total_bytes",
        "total_mib",
        "avg_mib_per_video_frame",
        "note",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES))
    parser.add_argument("--gaps", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--cache_root", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--profiles", nargs="+", default=["static_anchor", "full_half_anchor"])
    parser.add_argument("--codecs", nargs="+", default=["float32", "float16", "q8", "q6", "q4"])
    parser.add_argument("--opacity_thresholds", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2])
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt = Options()
    opt.resume = str(args.checkpoint)
    opt.compile = False
    opt.input_frames = 1
    opt.output_frames = 2
    opt.epoch = 0
    model = load_model(deepcopy(opt), args.checkpoint, device)

    rows = []
    for gap in args.gaps:
        for sample in args.samples:
            print(f"=== Stage2 {sample} gap={gap} ===", flush=True)
            rows.extend(run_sample_gap(sample, gap, args, model, opt, device))
            torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()

    summary = {"rows": rows}
    json_path = args.summary_root / "stage2_keyframe_gaussian_anchor_summary.json"
    csv_path = args.summary_root / "stage2_keyframe_gaussian_anchor_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    print(json.dumps({"summary": str(json_path), "csv": str(csv_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
