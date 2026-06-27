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
from skimage.metrics import structural_similarity as ssim
from torch.amp import autocast


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAVIS_ROOT = Path("/data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS")
DEFAULT_CHECKPOINT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/streamsplat.safetensors")
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage72_original_davis_baseline")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage72_original_davis_baseline"
DEFAULT_STAGE70_PSNR = REPO_ROOT / "experiments/stage70_scoped_davis_rd_package/stage70_all_psnr_table.csv"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage6_export_real_anchor_dataset import get_depth, get_image, load_model, normalize_depths  # noqa: E402


ROW_FIELDS = [
    "sample",
    "split",
    "sequence",
    "frame_gap",
    "total_frames",
    "keyframe_count",
    "keyframe_ratio",
    "pair_count",
    "complete",
    "raw_pred_gs_mib",
    "raw_pred_gs_mib_per_frame",
    "all_count",
    "all_psnr_avg",
    "all_psnr_min",
    "all_ssim_avg",
    "middle_count",
    "middle_psnr_avg",
    "middle_psnr_min",
    "middle_ssim_avg",
    "given_count",
    "given_psnr_avg",
    "given_psnr_min",
    "given_ssim_avg",
]

PER_FRAME_FIELDS = ["sample", "frame_gap", "frame_index", "is_keyframe", "psnr", "ssim"]

COMPARE_FIELDS = [
    "sample",
    "frame_gap",
    "original_streamsplat_all_psnr",
    "original_streamsplat_middle_psnr",
    "original_streamsplat_given_psnr",
    "stage70_linear_uniform_all_psnr",
    "stage70_adapter_uniform_all_psnr",
    "stage70_adapter_predicted_all_psnr",
    "delta_original_vs_stage70_adapter_uniform",
    "delta_original_vs_stage70_linear_uniform",
]

GAP_SUMMARY_FIELDS = [
    "frame_gap",
    "sequence_count",
    "mean_original_streamsplat_all_psnr",
    "mean_original_streamsplat_middle_psnr",
    "mean_original_streamsplat_given_psnr",
    "mean_stage70_linear_uniform_all_psnr",
    "mean_stage70_adapter_uniform_all_psnr",
    "mean_stage70_adapter_predicted_all_psnr",
]


def read_csv(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sorted_images(path):
    path = Path(path)
    return sorted(path.glob("*.jpg"), key=lambda p: p.name)


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


def build_pairs(total_frames, gap):
    selected = list(range(0, total_frames, gap))
    if selected[-1] != total_frames - 1:
        selected.append(total_frames - 1)
    return selected, [(selected[i], selected[i + 1]) for i in range(len(selected) - 1)]


def tensor_payload_mib(pred_gs):
    total = 0
    for value in pred_gs.values():
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
    return total / (1024.0 * 1024.0)


def frame_psnr(ref, pred):
    mse = float(np.mean((ref.astype(np.float32) / 255.0 - pred.astype(np.float32) / 255.0) ** 2))
    if mse <= 1e-12:
        return 100.0
    return -10.0 * math.log10(mse)


def summarize_metric(per_frame_rows, indices):
    rows = [per_frame_rows[idx] for idx in indices]
    if not rows:
        return {"count": 0, "psnr_avg": None, "psnr_min": None, "ssim_avg": None, "ssim_min": None}
    psnrs = [row["psnr"] for row in rows]
    ssims = [row["ssim"] for row in rows]
    return {
        "count": len(rows),
        "psnr_avg": float(np.mean(psnrs)),
        "psnr_min": float(np.min(psnrs)),
        "ssim_avg": float(np.mean(ssims)),
        "ssim_min": float(np.min(ssims)),
    }


def load_sequence_arrays(davis_root, sequence, opt):
    frame_files, depth_files = list_sequence_paths(davis_root, sequence)
    frames = [get_image(path, opt.image_height, opt.image_width) for path in frame_files]
    depths = [get_depth(path, opt.image_height, opt.image_width) for path in depth_files]
    return frame_files, depth_files, frames, depths


def render_original_batch(model, opt, frames, depths, timestamps):
    decoder_out = model.forward_gaussians(frames, depths, timestamps)
    anchor_time = torch.tensor([0.0, 1.0], device=frames.device)
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
    return render_pkg["render"], tensor_payload_mib(decoder_out["pred_gs"])


def run_sequence_gap(split, sequence, gap, args, model, opt, device):
    sample = f"DAVIS/{split}/{sequence}"
    out_dir = args.heavy_root / "DAVIS" / split / sequence / f"gap{gap}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    per_frame_path = out_dir / "per_frame_metrics.json"
    if summary_path.exists() and per_frame_path.exists() and not args.overwrite:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        per_frame = json.loads(per_frame_path.read_text(encoding="utf-8"))
        return summary, per_frame

    frame_files, _depth_files, all_frames, all_depths = load_sequence_arrays(args.davis_root, sequence, opt)
    selected_indices, pairs = build_pairs(len(frame_files), gap)
    keyframe_set = set(selected_indices)
    pred_by_index = {}
    total_gs_mib = 0.0
    pairs_by_len = defaultdict(list)
    for pair in pairs:
        pairs_by_len[pair[1] - pair[0]].append(pair)

    with torch.no_grad():
        for seg_len, seg_pairs in sorted(pairs_by_len.items()):
            opt.output_frames = seg_len + 1
            model.opt.output_frames = seg_len + 1
            fixed_timestamps = torch.linspace(0.0, 1.0, seg_len + 1, device=device)
            for start in range(0, len(seg_pairs), args.batch_size):
                batch_pairs = seg_pairs[start : start + args.batch_size]
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
                depths = normalize_depths(depths)
                timestamps = torch.from_numpy(np.stack(batch_timestamps, axis=0)).float().to(device)
                pred, gs_mib = render_original_batch(model, opt, frames, depths, timestamps)
                total_gs_mib += gs_mib
                pred = pred.detach().cpu()
                for local_idx, (a, _b) in enumerate(batch_pairs):
                    for offset in range(seg_len + 1):
                        global_idx = a + offset
                        arr = pred[local_idx, offset].permute(1, 2, 0).numpy()
                        pred_by_index[global_idx] = (arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)
                del pred, frames, depths, timestamps
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if len(pred_by_index) != len(frame_files):
        missing = sorted(set(range(len(frame_files))) - set(pred_by_index))
        raise RuntimeError(f"Incomplete prediction for {sample} gap={gap}: missing {missing[:10]}")

    per_frame = []
    for idx in range(len(frame_files)):
        ref = all_frames[idx]
        pred = pred_by_index[idx]
        per_frame.append({
            "sample": sample,
            "frame_gap": gap,
            "frame_index": idx,
            "is_keyframe": idx in keyframe_set,
            "psnr": frame_psnr(ref, pred),
            "ssim": float(ssim(ref.astype(np.float32) / 255.0, pred.astype(np.float32) / 255.0, channel_axis=2, data_range=1.0)),
        })

    all_indices = list(range(len(frame_files)))
    given_indices = [idx for idx in all_indices if idx in keyframe_set]
    middle_indices = [idx for idx in all_indices if idx not in keyframe_set]
    all_metrics = summarize_metric(per_frame, all_indices)
    middle_metrics = summarize_metric(per_frame, middle_indices)
    given_metrics = summarize_metric(per_frame, given_indices)
    summary = {
        "method": "original_streamsplat_pretrained",
        "sample": sample,
        "split": split,
        "sequence": sequence,
        "frame_gap": gap,
        "resolution": [opt.image_width, opt.image_height],
        "total_frames": len(frame_files),
        "selected_keyframes": selected_indices,
        "keyframe_count": len(selected_indices),
        "keyframe_ratio": len(selected_indices) / len(frame_files),
        "pair_count": len(pairs),
        "complete": len(pred_by_index) == len(frame_files),
        "raw_pred_gs_mib": total_gs_mib,
        "raw_pred_gs_mib_per_frame": total_gs_mib / len(frame_files),
        "all": all_metrics,
        "middle_only": middle_metrics,
        "given_keyframes": given_metrics,
        "heavy_output_dir": str(out_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    per_frame_path.write_text(json.dumps(per_frame, indent=2) + "\n", encoding="utf-8")
    return summary, per_frame


def flatten_summary(row):
    return {
        "sample": row["sample"],
        "split": row["split"],
        "sequence": row["sequence"],
        "frame_gap": row["frame_gap"],
        "total_frames": row["total_frames"],
        "keyframe_count": row["keyframe_count"],
        "keyframe_ratio": row["keyframe_ratio"],
        "pair_count": row["pair_count"],
        "complete": row["complete"],
        "raw_pred_gs_mib": row["raw_pred_gs_mib"],
        "raw_pred_gs_mib_per_frame": row["raw_pred_gs_mib_per_frame"],
        "all_count": row["all"]["count"],
        "all_psnr_avg": row["all"]["psnr_avg"],
        "all_psnr_min": row["all"]["psnr_min"],
        "all_ssim_avg": row["all"]["ssim_avg"],
        "middle_count": row["middle_only"]["count"],
        "middle_psnr_avg": row["middle_only"]["psnr_avg"],
        "middle_psnr_min": row["middle_only"]["psnr_min"],
        "middle_ssim_avg": row["middle_only"]["ssim_avg"],
        "given_count": row["given_keyframes"]["count"],
        "given_psnr_avg": row["given_keyframes"]["psnr_avg"],
        "given_psnr_min": row["given_keyframes"]["psnr_min"],
        "given_ssim_avg": row["given_keyframes"]["ssim_avg"],
    }


def stage70_lookup(path):
    rows = read_csv(path)
    lookup = {}
    for row in rows:
        key = (row["sample"], int(row["reference_gap"]), row["method"], row["selector"])
        lookup[key] = float(row["all_psnr"])
    return lookup


def build_comparison_rows(rows, lookup):
    out = []
    for row in rows:
        sample = row["sample"]
        gap = int(row["frame_gap"])
        linear_uniform = lookup.get((sample, gap, "linear_anchor", "uniform"))
        adapter_uniform = lookup.get((sample, gap, "stage65_rgb_h256_adapter", "uniform"))
        adapter_predicted = lookup.get((sample, gap, "stage65_rgb_h256_adapter", "predicted_full_feature_dp"))
        out.append({
            "sample": sample,
            "frame_gap": gap,
            "original_streamsplat_all_psnr": row["all"]["psnr_avg"],
            "original_streamsplat_middle_psnr": row["middle_only"]["psnr_avg"],
            "original_streamsplat_given_psnr": row["given_keyframes"]["psnr_avg"],
            "stage70_linear_uniform_all_psnr": linear_uniform,
            "stage70_adapter_uniform_all_psnr": adapter_uniform,
            "stage70_adapter_predicted_all_psnr": adapter_predicted,
            "delta_original_vs_stage70_adapter_uniform": row["all"]["psnr_avg"] - adapter_uniform if adapter_uniform is not None else None,
            "delta_original_vs_stage70_linear_uniform": row["all"]["psnr_avg"] - linear_uniform if linear_uniform is not None else None,
        })
    return out


def average(values):
    values = [float(value) for value in values if value not in (None, "")]
    return float(np.mean(values)) if values else None


def build_gap_summary(comparison_rows):
    grouped = defaultdict(list)
    for row in comparison_rows:
        grouped[int(row["frame_gap"])].append(row)
    out = []
    for gap, rows in sorted(grouped.items()):
        out.append({
            "frame_gap": gap,
            "sequence_count": len(rows),
            "mean_original_streamsplat_all_psnr": average(row["original_streamsplat_all_psnr"] for row in rows),
            "mean_original_streamsplat_middle_psnr": average(row["original_streamsplat_middle_psnr"] for row in rows),
            "mean_original_streamsplat_given_psnr": average(row["original_streamsplat_given_psnr"] for row in rows),
            "mean_stage70_linear_uniform_all_psnr": average(row["stage70_linear_uniform_all_psnr"] for row in rows),
            "mean_stage70_adapter_uniform_all_psnr": average(row["stage70_adapter_uniform_all_psnr"] for row in rows),
            "mean_stage70_adapter_predicted_all_psnr": average(row["stage70_adapter_predicted_all_psnr"] for row in rows),
        })
    return out


def write_report(summary, comparison_rows, gap_summary, path):
    lines = [
        "# Stage72 Original DAVIS Baseline",
        "",
        "## Scope",
        "",
        f"- DAVIS root: `{summary['davis_root']}`",
        f"- Sequences: `{', '.join(summary['sequences'])}`",
        f"- Gaps: `{', '.join(str(gap) for gap in summary['gaps'])}`",
        "- Primary metric: all-frame PSNR.",
        "",
        "## Gap Summary",
        "",
        "| gap | original all PSNR | original middle PSNR | original given PSNR | Stage70 linear uniform | Stage70 adapter uniform | Stage70 adapter predicted |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gap_summary:
        lines.append(
            f"| {row['frame_gap']} | {row['mean_original_streamsplat_all_psnr']} | {row['mean_original_streamsplat_middle_psnr']} | {row['mean_original_streamsplat_given_psnr']} | {row['mean_stage70_linear_uniform_all_psnr']} | {row['mean_stage70_adapter_uniform_all_psnr']} | {row['mean_stage70_adapter_predicted_all_psnr']} |"
        )
    lines.extend([
        "",
        "## Per-Sequence Comparison",
        "",
        "| sample | gap | original all PSNR | Stage70 adapter uniform | original - adapter uniform |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in comparison_rows:
        lines.append(
            f"| `{row['sample']}` | {row['frame_gap']} | {row['original_streamsplat_all_psnr']} | {row['stage70_adapter_uniform_all_psnr']} | {row['delta_original_vs_stage70_adapter_uniform']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- This is an original StreamSplat pretrained baseline under the current DAVIS preprocessing and Stage70 subset.",
        "- `raw_pred_gs_mib_per_frame` is diagnostic tensor payload size, not a transmitted codec bitstream.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--stage70_psnr", type=Path, default=DEFAULT_STAGE70_PSNR)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sequences", nargs="+", default=["bmx-trees", "car-shadow", "goat", "soapbox"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.davis_root.exists():
        raise FileNotFoundError(args.davis_root)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    opt = Options()
    opt.resume = str(args.checkpoint)
    opt.compile = False
    opt.input_frames = 1
    opt.output_frames = 2
    opt.epoch = 0
    model = load_model(deepcopy(opt), args.checkpoint, str(device))

    rows = []
    per_frame_rows = []
    for gap in args.gaps:
        for sequence in args.sequences:
            print(f"=== Stage72 original StreamSplat DAVIS/{args.split}/{sequence} gap={gap} ===", flush=True)
            row, per_frame = run_sequence_gap(args.split, sequence, gap, args, model, opt, device)
            rows.append(row)
            per_frame_rows.extend(per_frame)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    lookup = stage70_lookup(args.stage70_psnr)
    comparison_rows = build_comparison_rows(rows, lookup)
    gap_summary = build_gap_summary(comparison_rows)
    flat_rows = [flatten_summary(row) for row in rows]

    rows_csv = args.summary_root / "stage72_original_davis_baseline_rows.csv"
    per_frame_csv = args.summary_root / "stage72_original_davis_baseline_per_frame.csv"
    comparison_csv = args.summary_root / "stage72_original_vs_stage70_comparison.csv"
    gap_summary_csv = args.summary_root / "stage72_original_vs_stage70_gap_summary.csv"
    report_md = args.summary_root / "stage72_original_davis_baseline_report.md"
    summary_json = args.summary_root / "stage72_original_davis_baseline_summary.json"
    write_csv(flat_rows, rows_csv, ROW_FIELDS)
    write_csv(per_frame_rows, per_frame_csv, PER_FRAME_FIELDS)
    write_csv(comparison_rows, comparison_csv, COMPARE_FIELDS)
    write_csv(gap_summary, gap_summary_csv, GAP_SUMMARY_FIELDS)

    summary = {
        "stage": 72,
        "mode": "original StreamSplat DAVIS scoped baseline",
        "davis_root": str(args.davis_root),
        "checkpoint": str(args.checkpoint),
        "heavy_root": str(args.heavy_root),
        "summary_root": str(args.summary_root),
        "split": args.split,
        "sequences": args.sequences,
        "gaps": args.gaps,
        "device": str(device),
        "batch_size": args.batch_size,
        "rows_csv": str(rows_csv),
        "per_frame_csv": str(per_frame_csv),
        "comparison_csv": str(comparison_csv),
        "gap_summary_csv": str(gap_summary_csv),
        "report_md": str(report_md),
        "summary_json": str(summary_json),
        "gap_summary": gap_summary,
        "notes": [
            "Primary metric is all-frame PSNR.",
            "Middle/given PSNR are diagnostic only.",
            "raw_pred_gs_mib_per_frame is raw tensor payload, not a codec bitstream.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, comparison_rows, gap_summary, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "comparison_csv": str(comparison_csv),
        "gap_summary_csv": str(gap_summary_csv),
        "gap_summary": gap_summary,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
