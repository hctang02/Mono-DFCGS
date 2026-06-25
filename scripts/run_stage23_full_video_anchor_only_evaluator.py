import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from skimage.metrics import structural_similarity as ssim


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments/stage6_real_anchor_dataset/stage6_real_anchor_dataset_manifest.csv"
DEFAULT_STAGE2_CSV = REPO_ROOT / "experiments/stage2_keyframe_gaussian_anchor/stage2_keyframe_gaussian_anchor_summary.csv"
DEFAULT_ADAPTER = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage21d_validated_anchor_adapter_training/stage21d_best_anchor_adapter.safetensors")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage23_full_video_anchor_only_evaluator"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.anchor_predictor import GaussianAnchorDynamicPredictor  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, uniform_dequantize, uniform_quantize, unflatten_static_anchor  # noqa: E402
from mono_dfcgs.render_adapter import static_anchor_to_single_frame_gaussians  # noqa: E402
from scripts.run_stage10b_predictor_renderer_rgb_smoke import psnr_from_mse  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import linear_anchor, render_anchor, render_prediction  # noqa: E402
from scripts.run_stage22_anchor_only_rd_curve import load_rates  # noqa: E402


def read_manifest(path, samples, gaps):
    sample_set = set(samples)
    gap_set = set(gaps)
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["frame_gap"] = int(row["frame_gap"])
            row["left_index"] = int(row["left_index"])
            row["right_index"] = int(row["right_index"])
            row["segment_length"] = int(row["segment_length"])
            row["middle_frame_count"] = int(row["middle_frame_count"])
            if row["sample"] not in sample_set or row["frame_gap"] not in gap_set:
                continue
            if Path(row["dataset_item"]).exists():
                rows.append(row)
    return rows


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["sample"], row["frame_gap"])].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: row["left_index"])
    return grouped


def load_rgb_numpy(path, height, width):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image.astype(np.float32) / 255.0


def tensor_to_numpy_rgb(tensor):
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    while tensor.dim() > 3:
        tensor = tensor[0]
    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    return tensor.numpy()


def frame_psnr(ref, pred):
    mse = float(np.mean((ref - pred) ** 2))
    return psnr_from_mse(mse)


def frame_ssim(ref, pred):
    return float(ssim(ref, pred, channel_axis=2, data_range=1.0))


def anchor_to_device(anchor, device):
    return {key: value.unsqueeze(0).float().to(device) for key, value in anchor.items()}


def maybe_quantize_anchor(anchor, bits):
    if bits <= 0:
        return anchor
    attrs = flatten_static_anchor(anchor)
    q, mins, scales = uniform_quantize(attrs, bits=bits)
    return unflatten_static_anchor(uniform_dequantize(q, mins, scales))


def load_adapter(path, hidden_dim, device):
    model = GaussianAnchorDynamicPredictor(
        hidden_dim=hidden_dim,
        apply_output_constraints=False,
        zero_init_residual=True,
    ).to(device)
    state = load_file(str(path), device="cuda:0" if device.type == "cuda" else "cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def empty_metric():
    return {"count": 0, "psnr_avg": None, "psnr_min": None, "ssim_avg": None, "ssim_min": None}


def summarize_metrics(records):
    if not records:
        return empty_metric()
    psnrs = [record["psnr"] for record in records]
    ssims = [record["ssim"] for record in records]
    return {
        "count": len(records),
        "psnr_avg": float(np.mean(psnrs)),
        "psnr_min": float(np.min(psnrs)),
        "ssim_avg": float(np.mean(ssims)),
        "ssim_min": float(np.min(ssims)),
    }


def evaluate_group(sample, gap, rows, model, opt, background, device, quant_bits):
    linear_records = {}
    adapter_records = {}
    keyframes = set()
    total_frames = 0
    height, width = opt.image_height, opt.image_width

    with torch.no_grad():
        for row in rows:
            item = torch.load(row["dataset_item"], map_location="cpu", weights_only=True)
            left = maybe_quantize_anchor(anchor_to_device(item["left_anchor"], device), quant_bits)
            right = maybe_quantize_anchor(anchor_to_device(item["right_anchor"], device), quant_bits)
            a = int(item["left_index"])
            b = int(item["right_index"])
            keyframes.update([a, b])
            total_frames = max(total_frames, b + 1)

            for frame_idx, anchor, rgb_path in [
                (a, left, item["left_rgb_path"]),
                (b, right, item["right_rgb_path"]),
            ]:
                if frame_idx not in linear_records:
                    ref = load_rgb_numpy(rgb_path, height, width)
                    pred = tensor_to_numpy_rgb(render_anchor(anchor, background, opt))
                    metric = {"frame_index": frame_idx, "is_keyframe": True, "psnr": frame_psnr(ref, pred), "ssim": frame_ssim(ref, pred)}
                    linear_records[frame_idx] = metric
                    adapter_records[frame_idx] = dict(metric)

            for mid in item["intermediate_frames"]:
                frame_idx = int(mid["frame_index"])
                t = float(mid["normalized_time"])
                ref = load_rgb_numpy(mid["rgb_path"], height, width)
                linear_pred = tensor_to_numpy_rgb(render_anchor(linear_anchor(left, right, t), background, opt))
                adapter_pred = tensor_to_numpy_rgb(render_prediction(model, {
                    "left": left,
                    "right": right,
                    "normalized_time": t,
                }, background, opt))
                linear_records[frame_idx] = {
                    "frame_index": frame_idx,
                    "is_keyframe": False,
                    "psnr": frame_psnr(ref, linear_pred),
                    "ssim": frame_ssim(ref, linear_pred),
                }
                adapter_records[frame_idx] = {
                    "frame_index": frame_idx,
                    "is_keyframe": False,
                    "psnr": frame_psnr(ref, adapter_pred),
                    "ssim": frame_ssim(ref, adapter_pred),
                }

    all_indices = sorted(linear_records)
    if len(all_indices) != total_frames:
        missing = sorted(set(range(total_frames)) - set(all_indices))
        raise RuntimeError(f"Incomplete prediction for {sample} gap={gap}: total={total_frames} missing={missing[:10]}")

    def method_summary(records):
        all_records = [records[idx] for idx in all_indices]
        middle_records = [records[idx] for idx in all_indices if idx not in keyframes]
        given_records = [records[idx] for idx in all_indices if idx in keyframes]
        return {
            "all": summarize_metrics(all_records),
            "middle_only": summarize_metrics(middle_records),
            "given_keyframes": summarize_metrics(given_records),
        }

    linear_summary = method_summary(linear_records)
    adapter_summary = method_summary(adapter_records)
    return {
        "sample": sample,
        "frame_gap": gap,
        "total_frames": total_frames,
        "keyframe_count": len(keyframes),
        "keyframe_ratio": len(keyframes) / max(total_frames, 1),
        "middle_count": total_frames - len(keyframes),
        "linear": linear_summary,
        "adapter": adapter_summary,
        "delta_all_psnr": adapter_summary["all"]["psnr_avg"] - linear_summary["all"]["psnr_avg"],
        "delta_middle_psnr": adapter_summary["middle_only"]["psnr_avg"] - linear_summary["middle_only"]["psnr_avg"],
        "delta_given_psnr": adapter_summary["given_keyframes"]["psnr_avg"] - linear_summary["given_keyframes"]["psnr_avg"],
    }


def write_csv(rows, path):
    fields = [
        "sample",
        "frame_gap",
        "estimated_q8_static_mib_per_frame",
        "total_frames",
        "keyframe_count",
        "middle_count",
        "linear_all_psnr",
        "adapter_all_psnr",
        "delta_all_psnr",
        "linear_middle_psnr",
        "adapter_middle_psnr",
        "delta_middle_psnr",
        "linear_given_psnr",
        "adapter_given_psnr",
        "delta_given_psnr",
        "linear_all_ssim",
        "adapter_all_ssim",
        "linear_middle_ssim",
        "adapter_middle_ssim",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sample": row["sample"],
                "frame_gap": row["frame_gap"],
                "estimated_q8_static_mib_per_frame": row["estimated_q8_static_mib_per_frame"],
                "total_frames": row["total_frames"],
                "keyframe_count": row["keyframe_count"],
                "middle_count": row["middle_count"],
                "linear_all_psnr": row["linear"]["all"]["psnr_avg"],
                "adapter_all_psnr": row["adapter"]["all"]["psnr_avg"],
                "delta_all_psnr": row["delta_all_psnr"],
                "linear_middle_psnr": row["linear"]["middle_only"]["psnr_avg"],
                "adapter_middle_psnr": row["adapter"]["middle_only"]["psnr_avg"],
                "delta_middle_psnr": row["delta_middle_psnr"],
                "linear_given_psnr": row["linear"]["given_keyframes"]["psnr_avg"],
                "adapter_given_psnr": row["adapter"]["given_keyframes"]["psnr_avg"],
                "delta_given_psnr": row["delta_given_psnr"],
                "linear_all_ssim": row["linear"]["all"]["ssim_avg"],
                "adapter_all_ssim": row["adapter"]["all"]["ssim_avg"],
                "linear_middle_ssim": row["linear"]["middle_only"]["ssim_avg"],
                "adapter_middle_ssim": row["adapter"]["middle_only"]["ssim_avg"],
            })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage2_csv", type=Path, default=DEFAULT_STAGE2_CSV)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--gaps", nargs="*", type=int, default=[2, 4, 8, 16])
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
    model = load_adapter(args.adapter, args.hidden_dim, device)

    manifest_rows = read_manifest(args.manifest, args.samples, args.gaps)
    grouped = group_rows(manifest_rows)
    rows = []
    for sample in args.samples:
        rates = load_rates(args.stage2_csv, sample, "static_anchor", "q8", 0.0)
        for gap in args.gaps:
            key = (sample, gap)
            if key not in grouped:
                raise RuntimeError(f"Missing manifest rows for {sample} gap={gap}")
            print(f"=== Stage23 sample={sample} gap={gap} ===", flush=True)
            summary = evaluate_group(sample, gap, grouped[key], model, opt, background, device, args.quant_bits)
            summary.update(rates[gap])
            rows.append(summary)

    csv_path = args.summary_root / "stage23_full_video_anchor_only_evaluator.csv"
    summary_path = args.summary_root / "stage23_full_video_anchor_only_evaluator_summary.json"
    write_csv(rows, csv_path)
    mean_delta_all = float(np.mean([row["delta_all_psnr"] for row in rows]))
    mean_delta_middle = float(np.mean([row["delta_middle_psnr"] for row in rows]))
    summary = {
        "stage": 23,
        "mode": "full-video anchor-only evaluator",
        "samples": args.samples,
        "gaps": args.gaps,
        "adapter": str(args.adapter),
        "quant_bits": args.quant_bits,
        "rows": rows,
        "csv": str(csv_path),
        "mean_delta_all_psnr": mean_delta_all,
        "mean_delta_middle_psnr": mean_delta_middle,
        "notes": "Given keyframes are rendered directly from transmitted q8 anchors for both methods. Adapter is used only for non-keyframe middle frames.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "rows": len(rows),
        "mean_delta_all_psnr": mean_delta_all,
        "mean_delta_middle_psnr": mean_delta_middle,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
