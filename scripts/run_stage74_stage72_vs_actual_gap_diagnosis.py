import argparse
import csv
import json
import math
import os
import sys
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.amp import autocast


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAVIS_ROOT = Path("/data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS")
DEFAULT_CHECKPOINT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/streamsplat.safetensors")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage74_stage72_vs_actual_gap_diagnosis"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from model.splat_model_inference import SplatModel  # noqa: E402
from scripts.run_stage6_export_real_anchor_dataset import get_depth, get_image  # noqa: E402


SUMMARY_FIELDS = [
    "sample",
    "gap",
    "mode",
    "depth_norm",
    "metric_space",
    "pair_count",
    "all_count",
    "middle_count",
    "given_count",
    "all_psnr",
    "middle_psnr",
    "given_psnr",
]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sorted_images(path):
    return sorted(Path(path).glob("*.jpg"), key=lambda p: p.name)


def depth_path_for_frame(image_path):
    return Path(str(image_path).replace("JPEGImages", "depthImages")).with_name(f"{image_path.stem}_pred.png")


def list_sequence_paths(davis_root, sequence):
    image_dir = Path(davis_root) / "JPEGImages/Full-Resolution" / sequence
    frame_files = sorted_images(image_dir)
    if not frame_files:
        raise FileNotFoundError(f"No DAVIS frames found in {image_dir}")
    depth_files = [depth_path_for_frame(path) for path in frame_files]
    missing = [path for path in depth_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} depth files for {sequence}; first={missing[0]}")
    return frame_files, depth_files


def load_sequence_arrays(davis_root, sequence, opt):
    frame_files, depth_files = list_sequence_paths(davis_root, sequence)
    frames = [get_image(path, opt.image_height, opt.image_width) for path in frame_files]
    depths = [get_depth(path, opt.image_height, opt.image_width) for path in depth_files]
    return frame_files, depth_files, frames, depths


def build_pairs(total_frames, gap, mode):
    if mode == "disjoint_no_tail":
        selected = list(range(0, total_frames, gap))
        return [(selected[i], selected[i + 1]) for i in range(len(selected) - 1)]
    if mode == "disjoint_with_tail":
        selected = list(range(0, total_frames, gap))
        if selected[-1] != total_frames - 1:
            selected.append(total_frames - 1)
        return [(selected[i], selected[i + 1]) for i in range(len(selected) - 1)]
    if mode == "sliding_fixed":
        return [(idx, idx + gap) for idx in range(0, total_frames - gap)]
    raise ValueError(f"Unsupported mode: {mode}")


def normalize_depths(depths, mode):
    if mode == "pair_joint":
        max_depth = depths.flatten(1).max(dim=1)[0][:, None, None, None, None]
        min_depth = depths.flatten(1).min(dim=1)[0][:, None, None, None, None]
        return (depths - min_depth) / (max_depth - min_depth + 1e-8)
    if mode == "per_frame":
        max_depth = depths.flatten(2).max(dim=2)[0][:, :, None, None, None]
        min_depth = depths.flatten(2).min(dim=2)[0][:, :, None, None, None]
        return (depths - min_depth) / (max_depth - min_depth + 1e-8)
    if mode == "none":
        return depths
    raise ValueError(f"Unsupported depth_norm: {mode}")


def load_model_with_audit(opt, checkpoint, device):
    model = SplatModel(opt).to(device)
    state = load_file(str(checkpoint), device=str(device))
    new_state = OrderedDict()
    for key, value in state.items():
        if "_orig_mod." in key and not opt.compile:
            key = key.replace("_orig_mod.", "")
        new_state[key] = value
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    model.eval()
    return model, {
        "checkpoint": str(checkpoint),
        "checkpoint_tensor_count": len(state),
        "checkpoint_value_count": int(sum(value.numel() for value in state.values() if torch.is_tensor(value))),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_keys_sample": list(missing[:20]),
        "unexpected_keys_sample": list(unexpected[:20]),
    }


def psnr_from_mse(mse):
    if mse <= 1e-12:
        return 100.0
    return -10.0 * math.log10(mse)


def metric_mses(pred, refs, metric_space):
    pred = pred.detach().float().cpu().clamp(0.0, 1.0)
    refs = torch.from_numpy(np.stack(refs)).float().permute(0, 3, 1, 2) / 255.0
    if metric_space == "official_256_float":
        pred_eval = F.interpolate(pred, size=(256, 256), mode="bilinear", align_corners=False)
        ref_eval = F.interpolate(refs, size=(256, 256), mode="bilinear", align_corners=False)
    elif metric_space == "stage72_512_float":
        pred_eval = pred
        ref_eval = refs
    elif metric_space == "stage72_512_uint8":
        pred_eval = torch.round(pred * 255.0).clamp(0.0, 255.0) / 255.0
        ref_eval = torch.round(refs * 255.0).clamp(0.0, 255.0) / 255.0
    else:
        raise ValueError(metric_space)
    mse = ((pred_eval - ref_eval) ** 2).flatten(1).mean(dim=1)
    return [float(value) for value in mse]


def summarize_psnrs(mse_values):
    if not mse_values:
        return {"count": 0, "psnr": None}
    psnrs = [psnr_from_mse(value) for value in mse_values]
    return {"count": len(psnrs), "psnr": float(np.mean(psnrs))}


def render_sequence_gap(sample, sequence, gap, mode, depth_norm, args, model, opt, device):
    _frame_files, _depth_files, all_frames, all_depths = load_sequence_arrays(args.davis_root, sequence, opt)
    pairs = build_pairs(len(all_frames), gap, mode)
    metric_records = {
        name: {"all": [], "middle": [], "given": []}
        for name in ["official_256_float", "stage72_512_float", "stage72_512_uint8"]
    }
    if not pairs:
        raise RuntimeError(f"No pairs for {sample} gap={gap} mode={mode}")
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
                batch_frames = []
                batch_depths = []
                batch_timestamps = []
                for a, b in batch_pairs:
                    batch_frames.append(np.stack([all_frames[a], all_frames[b]], axis=0))
                    batch_depths.append(np.stack([all_depths[a], all_depths[b]], axis=0))
                    batch_timestamps.append(fixed_timestamps.detach().cpu().numpy())
                frames = torch.from_numpy(np.stack(batch_frames, axis=0)).float().to(device) / 255.0
                frames = frames.permute(0, 1, 4, 2, 3)
                depths = torch.from_numpy(np.stack(batch_depths, axis=0)).float().to(device).unsqueeze(2)
                depths = normalize_depths(depths, depth_norm)
                timestamps = torch.from_numpy(np.stack(batch_timestamps, axis=0)).float().to(device)
                decoder_out = model.forward_gaussians(frames, depths, timestamps)
                anchor_time = torch.tensor([0.0, 1.0], device=device)
                with autocast("cuda", enabled=False):
                    render_pkg = model.gaussian_renderer(
                        decoder_out["pred_gs"],
                        model.background,
                        opt=opt,
                        timestamps=timestamps,
                        anchor_time=anchor_time,
                        override_opacity=False,
                        training=False,
                    )
                pred = render_pkg["render"]
                for local_idx, (a, _b) in enumerate(batch_pairs):
                    refs_all = [all_frames[a + offset] for offset in range(seg_len + 1)]
                    refs_middle = [all_frames[a + offset] for offset in range(1, seg_len)]
                    refs_given = [all_frames[a], all_frames[a + seg_len]]
                    pred_all = pred[local_idx]
                    pred_middle = pred[local_idx, 1:seg_len]
                    pred_given = torch.stack([pred[local_idx, 0], pred[local_idx, seg_len]], dim=0)
                    for metric_name in metric_records:
                        metric_records[metric_name]["all"].extend(metric_mses(pred_all, refs_all, metric_name))
                        if seg_len > 1:
                            metric_records[metric_name]["middle"].extend(metric_mses(pred_middle, refs_middle, metric_name))
                        metric_records[metric_name]["given"].extend(metric_mses(pred_given, refs_given, metric_name))
                del pred, frames, depths, timestamps, decoder_out, render_pkg
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    rows = []
    for metric_name, scopes in metric_records.items():
        all_summary = summarize_psnrs(scopes["all"])
        middle_summary = summarize_psnrs(scopes["middle"])
        given_summary = summarize_psnrs(scopes["given"])
        rows.append({
            "sample": sample,
            "gap": gap,
            "mode": mode,
            "depth_norm": depth_norm,
            "metric_space": metric_name,
            "pair_count": len(pairs),
            "all_count": all_summary["count"],
            "middle_count": middle_summary["count"],
            "given_count": given_summary["count"],
            "all_psnr": all_summary["psnr"],
            "middle_psnr": middle_summary["psnr"],
            "given_psnr": given_summary["psnr"],
        })
    return rows


def build_aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["gap"], row["mode"], row["depth_norm"], row["metric_space"])].append(row)
    out = []
    for (gap, mode, depth_norm, metric_space), items in sorted(grouped.items()):
        weights_all = np.array([int(row["all_count"]) for row in items], dtype=np.float64)
        weights_middle = np.array([int(row["middle_count"]) for row in items], dtype=np.float64)
        weights_given = np.array([int(row["given_count"]) for row in items], dtype=np.float64)
        out.append({
            "sample": "MEAN_WEIGHTED",
            "gap": gap,
            "mode": mode,
            "depth_norm": depth_norm,
            "metric_space": metric_space,
            "pair_count": int(sum(int(row["pair_count"]) for row in items)),
            "all_count": int(weights_all.sum()),
            "middle_count": int(weights_middle.sum()),
            "given_count": int(weights_given.sum()),
            "all_psnr": float(np.average([row["all_psnr"] for row in items], weights=weights_all)) if weights_all.sum() > 0 else None,
            "middle_psnr": float(np.average([row["middle_psnr"] for row in items], weights=weights_middle)) if weights_middle.sum() > 0 else None,
            "given_psnr": float(np.average([row["given_psnr"] for row in items], weights=weights_given)) if weights_given.sum() > 0 else None,
        })
    return out


def read_val_sequences(davis_root):
    path = Path(davis_root) / "ImageSets/2017/val.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--sequences", nargs="+", default=["bmx-trees", "car-shadow", "goat", "soapbox"])
    parser.add_argument("--all_val", action="store_true")
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 5, 8])
    parser.add_argument("--modes", nargs="+", default=["disjoint_with_tail", "sliding_fixed"])
    parser.add_argument("--depth_norms", nargs="+", default=["pair_joint", "per_frame"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
    args.summary_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    opt = Options()
    opt.resume = str(args.checkpoint)
    opt.compile = False
    opt.input_frames = 1
    opt.output_frames = 2
    opt.epoch = 0
    model, load_audit = load_model_with_audit(deepcopy(opt), args.checkpoint, device)
    sequences = read_val_sequences(args.davis_root) if args.all_val else args.sequences
    rows = []
    for sequence in sequences:
        sample = f"DAVIS/val/{sequence}"
        for gap in args.gaps:
            for mode in args.modes:
                for depth_norm in args.depth_norms:
                    print(f"=== Stage74 {sample} gap={gap} mode={mode} depth_norm={depth_norm} ===", flush=True)
                    rows.extend(render_sequence_gap(sample, sequence, gap, mode, depth_norm, args, model, opt, device))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    aggregate_rows = build_aggregate(rows)
    rows_csv = args.summary_root / "stage74_stage72_vs_actual_gap_diagnosis_rows.csv"
    aggregate_csv = args.summary_root / "stage74_stage72_vs_actual_gap_diagnosis_aggregate.csv"
    summary_json = args.summary_root / "stage74_stage72_vs_actual_gap_diagnosis_summary.json"
    write_csv(rows, rows_csv, SUMMARY_FIELDS)
    write_csv(aggregate_rows, aggregate_csv, SUMMARY_FIELDS)
    summary = {
        "stage": 74,
        "mode": "Stage72 vs actual StreamSplat protocol diagnosis",
        "davis_root": str(args.davis_root),
        "checkpoint": str(args.checkpoint),
        "sequences": sequences,
        "gaps": args.gaps,
        "modes": args.modes,
        "depth_norms": args.depth_norms,
        "metric_spaces": ["official_256_float", "stage72_512_float", "stage72_512_uint8"],
        "load_audit": load_audit,
        "rows_csv": str(rows_csv),
        "aggregate_csv": str(aggregate_csv),
        "aggregate": aggregate_rows,
        "notes": [
            "official_256_float matches the paper text saying metrics are evaluated at 256x256.",
            "sliding_fixed approximates fixed-interval evaluation over all possible windows.",
            "disjoint_with_tail matches the Stage72 continuous sparse-keyframe reconstruction style.",
            "pair_joint depth normalization matches Stage72; per_frame matches training-time normalization in model/splat_model.py.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_json),
        "rows_csv": str(rows_csv),
        "aggregate_csv": str(aggregate_csv),
        "load_audit": load_audit,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
