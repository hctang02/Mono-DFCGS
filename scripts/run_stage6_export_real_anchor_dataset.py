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
DEFAULT_STAGE1_CACHE = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage1_streamsplat_fair_metrics/cache")
DEFAULT_HEAVY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage6_real_anchor_dataset")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage6_real_anchor_dataset"

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


def normalize_depths(depths):
    max_depth = depths.flatten(1).max(dim=1)[0][:, None, None, None, None]
    min_depth = depths.flatten(1).min(dim=1)[0][:, None, None, None, None]
    return (depths - min_depth) / (max_depth - min_depth + 1e-8)


def build_pairs(total_frames, max_gap):
    selected = list(range(0, total_frames, max_gap))
    if selected[-1] != total_frames - 1:
        selected.append(total_frames - 1)
    return selected, [(selected[i], selected[i + 1]) for i in range(len(selected) - 1)]


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
    return frame_files, depth_files, frames, depths


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


def static_anchor_from_half(anchor, batch_index):
    return {
        "rgb": anchor["rgb"][batch_index].to(torch.float16),
        "opacity": anchor["opacity"][batch_index, :, :1].to(torch.float16),
        "scale": anchor["scale"][batch_index].to(torch.float16),
        "xyz": anchor["xyz"][batch_index, :, 0, :].to(torch.float16),
        "rot": anchor["rot"][batch_index, :, 0, :].to(torch.float16),
    }


def tensor_mib(obj):
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size() / (1024.0 * 1024.0)
    if isinstance(obj, dict):
        return sum(tensor_mib(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_mib(v) for v in obj)
    return 0.0


def make_intermediate_records(a, b, frame_files, depth_files):
    records = []
    seg_len = b - a
    for frame_idx in range(a + 1, b):
        records.append({
            "frame_index": frame_idx,
            "normalized_time": (frame_idx - a) / seg_len,
            "rgb_path": str(frame_files[frame_idx]),
            "depth_path": str(depth_files[frame_idx]),
        })
    return records


def save_pair_dataset_item(out_path, sample, gap, a, b, left_anchor, right_anchor, frame_files, depth_files):
    item = {
        "sample": sample,
        "frame_gap": gap,
        "left_index": a,
        "right_index": b,
        "segment_length": b - a,
        "left_rgb_path": str(frame_files[a]),
        "right_rgb_path": str(frame_files[b]),
        "left_depth_path": str(depth_files[a]),
        "right_depth_path": str(depth_files[b]),
        "intermediate_frames": make_intermediate_records(a, b, frame_files, depth_files),
        "left_anchor": left_anchor,
        "right_anchor": right_anchor,
        "anchor_format": "static_anchor_float16",
        "fields": ["rgb", "opacity", "scale", "xyz", "rot"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(item, out_path)
    return tensor_mib({"left_anchor": left_anchor, "right_anchor": right_anchor})


def run_sample_gap(sample, gap, args, model, opt, device):
    frame_files, depth_files, frames_all, depths_all = load_cached_sample(sample, opt, args.cache_root)
    selected, pairs = build_pairs(len(frame_files), gap)
    if args.max_pairs_per_sample is not None:
        pairs = pairs[:args.max_pairs_per_sample]
    pairs_by_len = defaultdict(list)
    for pair in pairs:
        pairs_by_len[pair[1] - pair[0]].append(pair)

    manifest_rows = []
    total_mib = 0.0
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
                    left_anchor = static_anchor_from_half(first_half, local_idx)
                    right_anchor = static_anchor_from_half(second_half, local_idx)
                    rel_path = Path(sample) / f"gap{gap}" / f"pair_{a:06d}_{b:06d}.pt"
                    out_path = args.heavy_root / rel_path
                    anchor_mib = save_pair_dataset_item(
                        out_path, sample, gap, a, b, left_anchor, right_anchor, frame_files, depth_files
                    )
                    total_mib += anchor_mib
                    manifest_rows.append({
                        "sample": sample,
                        "frame_gap": gap,
                        "left_index": a,
                        "right_index": b,
                        "segment_length": b - a,
                        "middle_frame_count": max(b - a - 1, 0),
                        "dataset_item": str(out_path),
                        "dataset_item_relative": str(rel_path),
                        "anchor_mib": anchor_mib,
                        "gaussians_per_anchor": int(left_anchor["rgb"].shape[0]),
                    })
                del gs_out, frames, depths, timestamps
                torch.cuda.empty_cache()
    return selected, len(frame_files), manifest_rows, total_mib


def write_csv(rows, path):
    fields = [
        "sample",
        "frame_gap",
        "left_index",
        "right_index",
        "segment_length",
        "middle_frame_count",
        "dataset_item",
        "dataset_item_relative",
        "anchor_mib",
        "gaussians_per_anchor",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES))
    parser.add_argument("--gaps", nargs="+", type=int, default=[4])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache_root", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_pairs_per_sample", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt = Options()
    opt.resume = str(args.checkpoint)
    opt.compile = False
    opt.input_frames = 1
    opt.output_frames = 2
    opt.epoch = 0
    model = load_model(deepcopy(opt), args.checkpoint, device)

    all_rows = []
    summaries = []
    for gap in args.gaps:
        for sample in args.samples:
            print(f"=== Stage6 export {sample} gap={gap} ===", flush=True)
            selected, total_frames, rows, total_mib = run_sample_gap(sample, gap, args, model, opt, device)
            all_rows.extend(rows)
            summaries.append({
                "sample": sample,
                "frame_gap": gap,
                "total_frames": total_frames,
                "selected_keyframes": selected,
                "exported_pair_count": len(rows),
                "total_middle_frames": sum(int(row["middle_frame_count"]) for row in rows),
                "total_anchor_mib": total_mib,
                "avg_anchor_mib_per_pair": total_mib / max(len(rows), 1),
                "heavy_root": str(args.heavy_root),
            })
            torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()

    manifest_csv = args.summary_root / "stage6_real_anchor_dataset_manifest.csv"
    summary_json = args.summary_root / "stage6_real_anchor_dataset_summary.json"
    manifest_json = args.summary_root / "stage6_real_anchor_dataset_manifest.json"
    write_csv(all_rows, manifest_csv)
    manifest_json.write_text(json.dumps({"rows": all_rows}, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps({"summaries": summaries}, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_json),
        "manifest_csv": str(manifest_csv),
        "manifest_json": str(manifest_json),
        "rows": len(all_rows),
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
