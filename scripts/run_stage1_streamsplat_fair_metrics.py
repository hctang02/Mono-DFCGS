import argparse
import csv
import json
import math
import os
import shutil
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
DEFAULT_DEPTH_CHECKPOINT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/depth_anything_v2_vitl.pth")
DEFAULT_HEAVY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage1_streamsplat_fair_metrics")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage1_streamsplat_fair_metrics"

SAMPLES = {
    "n3dv": "/mnt/hdd2tC/tmp/opencode/gt_reference_videos/n3dv_input_81f_560x336.mp4",
    "meetroom": "/mnt/hdd2tC/tmp/opencode/gt_reference_videos/meetroom_input_81f_560x336.mp4",
    "driving": "/mnt/ssd2tB/haocheng/NeoVerse/examples/videos/driving.mp4",
    "robot": "/mnt/ssd2tB/haocheng/NeoVerse/examples/videos/robot.mp4",
}

sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from model.depth_anything.depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
from model.splat_model_inference import SplatModel  # noqa: E402


def run(cmd):
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    import subprocess

    subprocess.run(cmd, check=True)


def extract_frames(video_path, frame_dir, overwrite=False):
    frame_dir = Path(frame_dir)
    if overwrite and frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frame_dir.glob("*.png"))
    if existing:
        return existing
    run([
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vsync",
        "0",
        str(frame_dir / "%06d.png"),
    ])
    return sorted(frame_dir.glob("*.png"))


def load_depth_model(checkpoint, device):
    model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


def generate_depths(frame_files, depth_dir, checkpoint, device, overwrite=False):
    depth_dir = Path(depth_dir)
    if overwrite and depth_dir.exists():
        shutil.rmtree(depth_dir)
    depth_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(depth_dir.glob("*.png"))
    if len(existing) == len(frame_files):
        return existing
    model = load_depth_model(checkpoint, device)
    for i, path in enumerate(frame_files, 1):
        out_path = depth_dir / path.name
        if out_path.exists() and not overwrite:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        with torch.no_grad():
            depth = model.infer_image(image, 518).astype(np.float32)
        dmin, dmax = float(depth.min()), float(depth.max())
        depth01 = (depth - dmin) / (dmax - dmin + 1e-8)
        cv2.imwrite(str(out_path), np.clip(depth01 * 65535.0, 0, 65535).astype(np.uint16))
        if i % 10 == 0 or i == len(frame_files):
            print(f"depth {i}/{len(frame_files)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return sorted(depth_dir.glob("*.png"))


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


def preprocess(frames, depths, device):
    frames = torch.from_numpy(np.stack(frames)).float().to(device) / 255.0
    frames = frames.permute(0, 3, 1, 2).unsqueeze(0)
    depths = torch.from_numpy(np.stack(depths)).float().to(device)
    depths = depths.unsqueeze(1).unsqueeze(0)
    max_depth = depths.flatten(1).max(dim=1)[0][:, None, None, None, None]
    min_depth = depths.flatten(1).min(dim=1)[0][:, None, None, None, None]
    depths = (depths - min_depth) / (max_depth - min_depth + 1e-8)
    return frames, depths


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


def tensor_payload_mib(pred_gs):
    total = 0
    shapes = {}
    for key, value in pred_gs.items():
        if not torch.is_tensor(value):
            continue
        total += value.numel() * value.element_size()
        shapes[key] = list(value.shape)
    return total / (1024.0 * 1024.0), shapes


def frame_psnr(ref, pred):
    mse = float(np.mean((ref.astype(np.float32) / 255.0 - pred.astype(np.float32) / 255.0) ** 2))
    if mse <= 1e-12:
        return 100.0
    return -10.0 * math.log10(mse)


def compare_indexed_frames(ref_frames, pred_frames, indices):
    psnrs = []
    ssims = []
    for idx in indices:
        ref = ref_frames[idx]
        pred = pred_frames[idx]
        psnrs.append(frame_psnr(ref, pred))
        ssims.append(ssim(ref.astype(np.float32) / 255.0, pred.astype(np.float32) / 255.0, channel_axis=2, data_range=1.0))
    if not indices:
        return {
            "count": 0,
            "psnr_avg": None,
            "psnr_min": None,
            "ssim_avg": None,
            "ssim_min": None,
        }
    return {
        "count": len(indices),
        "psnr_avg": float(np.mean(psnrs)),
        "psnr_min": float(np.min(psnrs)),
        "ssim_avg": float(np.mean(ssims)),
        "ssim_min": float(np.min(ssims)),
    }


def build_pairs(total_frames, max_gap):
    selected = list(range(0, total_frames, max_gap))
    if selected[-1] != total_frames - 1:
        selected.append(total_frames - 1)
    return selected, [(selected[i], selected[i + 1]) for i in range(len(selected) - 1)]


def run_sample_gap(sample, video_path, gap, args, device):
    heavy_root = Path(args.heavy_root)
    sample_root = heavy_root / "cache" / sample
    frame_dir = sample_root / "frames"
    depth_dir = sample_root / "depths_vitl_u16"
    out_dir = heavy_root / f"{sample}_gap{gap}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_files = extract_frames(video_path, frame_dir, overwrite=args.overwrite_frames)
    depth_files = generate_depths(frame_files, depth_dir, args.depth_checkpoint, device, overwrite=args.overwrite_depths)

    opt = Options()
    opt.resume = str(args.checkpoint)
    opt.compile = False
    opt.input_frames = 1
    opt.epoch = 0

    model = load_model(deepcopy(opt), args.checkpoint, device)
    all_frames = [get_image(path, opt.image_height, opt.image_width) for path in frame_files]
    all_depths = [get_depth(path, opt.image_height, opt.image_width) for path in depth_files]
    selected_indices, pairs = build_pairs(len(frame_files), gap)
    keyframe_set = set(selected_indices)

    pred_by_index = {}
    total_gs_mib = 0.0
    gs_shapes = {}
    pairs_by_len = defaultdict(list)
    for pair in pairs:
        pairs_by_len[pair[1] - pair[0]].append(pair)

    with torch.no_grad():
        for seg_len, seg_pairs in sorted(pairs_by_len.items()):
            opt.output_frames = seg_len + 1
            model.opt.output_frames = seg_len + 1
            fixed_timestamps = torch.linspace(0.0, 1.0, seg_len + 1, device=device)
            for start in range(0, len(seg_pairs), args.batch_size):
                batch_pairs = seg_pairs[start:start + args.batch_size]
                input_frames = []
                input_depths = []
                timestamps = []
                for a, b in batch_pairs:
                    input_frames.append(np.stack([all_frames[a], all_frames[b]], axis=0))
                    input_depths.append(np.stack([all_depths[a], all_depths[b]], axis=0))
                    timestamps.append(fixed_timestamps.detach().cpu().numpy())
                frames_np = np.stack(input_frames, axis=0)
                depths_np = np.stack(input_depths, axis=0)
                frames = torch.from_numpy(frames_np).float().to(device) / 255.0
                frames = frames.permute(0, 1, 4, 2, 3)
                depths = torch.from_numpy(depths_np).float().to(device).unsqueeze(2)
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
                for local_idx, (a, b) in enumerate(batch_pairs):
                    for offset in range(seg_len + 1):
                        global_idx = a + offset
                        arr = pred[local_idx, offset].permute(1, 2, 0).numpy()
                        pred_by_index[global_idx] = (arr.clip(0, 1) * 255).astype(np.uint8)
                del output, gs_out, frames, depths, timestamps_t
                torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    pred_frames = [pred_by_index[i] for i in range(len(frame_files))]
    all_indices = list(range(len(frame_files)))
    given_indices = [idx for idx in all_indices if idx in keyframe_set]
    middle_indices = [idx for idx in all_indices if idx not in keyframe_set]
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

    summary = {
        "method": "StreamSplat pretrained fair metrics",
        "sample": sample,
        "input_video": str(video_path),
        "frame_gap": gap,
        "resolution": [opt.image_width, opt.image_height],
        "total_frames": len(frame_files),
        "selected_keyframes": selected_indices,
        "keyframe_count": len(selected_indices),
        "keyframe_ratio": len(selected_indices) / max(len(frame_files), 1),
        "pair_count": len(pairs),
        "complete": len(pred_by_index) == len(frame_files),
        "all": compare_indexed_frames(all_frames, pred_frames, all_indices),
        "middle_only": compare_indexed_frames(all_frames, pred_frames, middle_indices),
        "given_keyframes": compare_indexed_frames(all_frames, pred_frames, given_indices),
        "raw_pred_gs_mib": total_gs_mib,
        "raw_pred_gs_mib_per_frame": total_gs_mib / max(len(frame_files), 1),
        "raw_pred_gs_size_note": "Raw StreamSplat predicted dynamic 3DGS tensors for adjacent keyframe pairs; not a codec bitstream.",
        "pred_gs_shapes_last_batch": gs_shapes,
        "heavy_output_dir": str(out_dir),
    }
    if args.save_per_frame:
        per_frame_path = out_dir / "per_frame_metrics.json"
        per_frame_path.write_text(json.dumps(per_frame, indent=2), encoding="utf-8")
        summary["per_frame_metrics"] = str(per_frame_path)
    return summary


def write_csv(rows, path):
    fields = [
        "sample",
        "frame_gap",
        "total_frames",
        "keyframe_count",
        "keyframe_ratio",
        "raw_pred_gs_mib_per_frame",
        "all_psnr_avg",
        "all_ssim_avg",
        "middle_psnr_avg",
        "middle_ssim_avg",
        "given_psnr_avg",
        "given_ssim_avg",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sample": row["sample"],
                "frame_gap": row["frame_gap"],
                "total_frames": row["total_frames"],
                "keyframe_count": row["keyframe_count"],
                "keyframe_ratio": row["keyframe_ratio"],
                "raw_pred_gs_mib_per_frame": row["raw_pred_gs_mib_per_frame"],
                "all_psnr_avg": row["all"]["psnr_avg"],
                "all_ssim_avg": row["all"]["ssim_avg"],
                "middle_psnr_avg": row["middle_only"]["psnr_avg"],
                "middle_ssim_avg": row["middle_only"]["ssim_avg"],
                "given_psnr_avg": row["given_keyframes"]["psnr_avg"],
                "given_ssim_avg": row["given_keyframes"]["ssim_avg"],
            })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES))
    parser.add_argument("--gaps", nargs="+", type=int, default=[4])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--depth_checkpoint", type=Path, default=DEFAULT_DEPTH_CHECKPOINT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--overwrite_frames", action="store_true")
    parser.add_argument("--overwrite_depths", action="store_true")
    parser.add_argument("--save_per_frame", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.depth_checkpoint.exists():
        raise FileNotFoundError(args.depth_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.summary_root.mkdir(parents=True, exist_ok=True)
    args.heavy_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for gap in args.gaps:
        for sample in args.samples:
            video_path = Path(SAMPLES[sample])
            if not video_path.exists():
                raise FileNotFoundError(video_path)
            print(f"=== Stage1 {sample} gap={gap} ===", flush=True)
            rows.append(run_sample_gap(sample, video_path, gap, args, device))
            out = args.summary_root / f"{sample}_gap{gap}_summary.json"
            out.write_text(json.dumps(rows[-1], indent=2), encoding="utf-8")

    summary = {"rows": rows}
    summary_path = args.summary_root / "stage1_streamsplat_fair_metrics_summary.json"
    csv_path = args.summary_root / "stage1_streamsplat_fair_metrics_summary.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    print(json.dumps({"summary": str(summary_path), "csv": str(csv_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
