import argparse
import csv
import json
import math
import os
import struct
import sys
import zlib
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/streamsplat.safetensors")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage155_streamsplat_base_sideinfo_upper_bound"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage155_streamsplat_base_sideinfo_upper_bound")
DEFAULT_STAGE147_ROWS = REPO_ROOT / "experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_rows.csv"


sys.path.insert(0, str(REPO_ROOT))
import gaussian_renderer_dynamic  # noqa: E402
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402
from mono_dfcgs.render_adapter import static_anchor_to_single_frame_gaussians  # noqa: E402
from scripts.run_stage153_middle_multimetric_badcase_eval import (  # noqa: E402
    format_optional,
    load_metric_modules,
    lpips_metric,
    mean,
    ms_ssim_metric,
    percentile,
    psnr_metric,
    ssim_metric,
    tensor_to_rgb8,
    to_nchw,
)
from scripts.run_stage154_original_streamsplat_middle_base_alignment import (  # noqa: E402
    depth_path_for_rgb,
    load_task_batch,
)
from scripts.run_stage6_export_real_anchor_dataset import load_model  # noqa: E402
from scripts.run_stage72_original_davis_baseline import DEFAULT_DAVIS_ROOT  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import (  # noqa: E402
    DEFAULT_DENSE_MANIFEST,
    DEFAULT_TASK_MANIFEST,
    build_dense_index,
    load_anchor,
    parse_task_rows,
    select_balanced,
)


IMAGE_RESIDUAL_MAGIC = b"IRZ1"
IMAGE_RESIDUAL_HEADER = struct.Struct("<4sBBHHHI")

ROW_FIELDS = [
    "task_id",
    "sequence",
    "gap",
    "codec",
    "target_index",
    "normalized_time",
    "method",
    "side_bits",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "payload_bytes",
    "side_mib_per_intermediate",
    "q12_main_anchor_mib_per_frame_ref",
    "direct_total_mib_per_frame_ref",
    "delta_psnr_vs_original",
    "delta_ssim_vs_original",
    "delta_ms_ssim_vs_original",
    "delta_lpips_vs_original",
]

SUMMARY_FIELDS = [
    "gap",
    "method",
    "side_bits",
    "task_count",
    "mean_psnr",
    "min_psnr",
    "p10_psnr",
    "mean_ssim",
    "min_ssim",
    "p10_ssim",
    "mean_ms_ssim",
    "mean_lpips",
    "p90_lpips",
    "mean_payload_bytes",
    "mean_side_mib_per_intermediate",
    "mean_direct_total_mib_per_frame_ref",
    "mean_delta_psnr_vs_original",
    "mean_delta_ssim_vs_original",
    "mean_delta_ms_ssim_vs_original",
    "mean_delta_lpips_vs_original",
]

BADCASE_FIELDS = [
    "rank_type",
    "rank",
    "task_id",
    "sequence",
    "gap",
    "target_index",
    "normalized_time",
    "method",
    "side_bits",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "payload_bytes",
    "contact_sheet_path",
]

DIAGNOSTIC_FIELDS = [
    "task_id",
    "sequence",
    "gap",
    "target_index",
    "base_gaussian_count",
    "target_gaussian_count",
    "base_attr_shape",
    "target_attr_shape",
    "shape_matches_target_dense_anchor",
    "static_eval_render_max_abs_diff_vs_dynamic",
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


def pack_ints(values, bits):
    max_value = (1 << int(bits)) - 1
    out = bytearray()
    acc = 0
    acc_bits = 0
    for value in values:
        value = int(value)
        if value < 0 or value > max_value:
            raise ValueError(f"value {value} out of range for {bits} bits")
        acc |= value << acc_bits
        acc_bits += int(bits)
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits:
        out.append(acc & 0xFF)
    return bytes(out)


def unpack_ints(data, count, bits):
    values = []
    mask = (1 << int(bits)) - 1
    acc = 0
    acc_bits = 0
    offset = 0
    for _ in range(int(count)):
        while acc_bits < int(bits):
            if offset >= len(data):
                raise ValueError("not enough packed integer data")
            acc |= int(data[offset]) << acc_bits
            offset += 1
            acc_bits += 8
        values.append(acc & mask)
        acc >>= int(bits)
        acc_bits -= int(bits)
    return values


def encode_image_residual(base, target, bits, zlib_level):
    base_cpu = to_nchw(base).detach().cpu().float()
    target_cpu = to_nchw(target).detach().cpu().float()
    if base_cpu.shape != target_cpu.shape or base_cpu.shape[0] != 1:
        raise ValueError(f"expected matching [1,3,H,W], got {base_cpu.shape} and {target_cpu.shape}")
    residual = target_cpu - base_cpu
    _, channels, height, width = residual.shape
    mins = residual.amin(dim=(0, 2, 3))
    maxs = residual.amax(dim=(0, 2, 3))
    mins_half = mins.numpy().astype("<f2")
    maxs_half = maxs.numpy().astype("<f2")
    mins_codec = torch.from_numpy(mins_half.astype("<f4")).reshape(1, channels, 1, 1)
    maxs_codec = torch.from_numpy(maxs_half.astype("<f4")).reshape(1, channels, 1, 1)
    qmax = (1 << int(bits)) - 1
    scales = (maxs_codec - mins_codec).clamp_min(1e-8) / qmax
    q = torch.round((residual - mins_codec) / scales).clamp(0, qmax).to(torch.int64)
    packed = pack_ints(q.reshape(-1).tolist(), int(bits))
    compressed = zlib.compress(packed, int(zlib_level))
    header = IMAGE_RESIDUAL_HEADER.pack(IMAGE_RESIDUAL_MAGIC, 1, int(bits), int(channels), int(height), int(width), int(len(compressed)))
    metadata = mins_half.tobytes() + maxs_half.tobytes()
    payload = header + metadata + compressed
    decoded = decode_image_residual(base_cpu, payload)
    return payload, decoded.to(base.device), {
        "payload_bytes": len(payload),
        "header_bytes": len(header),
        "metadata_bytes": len(metadata),
        "packed_bytes": len(packed),
        "compressed_bytes": len(compressed),
        "compression_ratio": len(compressed) / max(len(packed), 1),
    }


def decode_image_residual(base_cpu, payload):
    magic, version, bits, channels, height, width, compressed_len = IMAGE_RESIDUAL_HEADER.unpack(payload[:IMAGE_RESIDUAL_HEADER.size])
    if magic != IMAGE_RESIDUAL_MAGIC or version != 1:
        raise ValueError("bad image residual payload")
    metadata_start = IMAGE_RESIDUAL_HEADER.size
    metadata_end = metadata_start + int(channels) * 2 * 2
    metadata = payload[metadata_start:metadata_end]
    mins = np.frombuffer(metadata[: int(channels) * 2], dtype="<f2").astype("<f4")
    maxs = np.frombuffer(metadata[int(channels) * 2:], dtype="<f2").astype("<f4")
    compressed = payload[metadata_end:metadata_end + int(compressed_len)]
    packed = zlib.decompress(compressed)
    count = int(channels) * int(height) * int(width)
    q = torch.tensor(unpack_ints(packed, count, int(bits)), dtype=torch.float32).reshape(1, int(channels), int(height), int(width))
    mins_t = torch.from_numpy(mins).reshape(1, int(channels), 1, 1)
    maxs_t = torch.from_numpy(maxs).reshape(1, int(channels), 1, 1)
    scales = (maxs_t - mins_t).clamp_min(1e-8) / ((1 << int(bits)) - 1)
    residual = q * scales + mins_t
    return (base_cpu + residual).clamp(0.0, 1.0)


def stream_gaussians_at_time(pred_gs, t, opt, sample_idx=0):
    device = pred_gs["xyz"].device
    n = int(pred_gs["xyz"].shape[1])
    anchor_time = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device).repeat_interleave(n // 2).reshape(1, n, 1)
    actual_time = torch.tensor(float(t), dtype=torch.float32, device=device).reshape(1, 1, 1)
    time_basis = actual_time - anchor_time
    xyz_static = pred_gs["xyz"][sample_idx:sample_idx + 1, :, 0, :].float()
    dynamic_components = pred_gs["xyz"][sample_idx:sample_idx + 1, :, 1:, :].float()
    if int(opt.forder) > 0:
        polynomial_basis = torch.cat([time_basis ** i for i in range(1, int(opt.forder) + 1)], dim=2).unsqueeze(-1)
        dynamic_poly = (dynamic_components * polynomial_basis).sum(dim=2)
    else:
        dynamic_poly = torch.zeros_like(xyz_static)
    xyz = xyz_static + dynamic_poly
    # Keep xyz in renderer parameter space. The dynamic renderer applies
    # opt.pred_inverse during rendering, so pre-applying it here would double
    # transform the depth coordinate and break static-vs-dynamic diagnostics.
    rot = pred_gs["rot"][sample_idx:sample_idx + 1, :, 0, :].float() + pred_gs["rot"][sample_idx:sample_idx + 1, :, 1, :].float() * time_basis
    opacity = pred_gs["opacity"][sample_idx:sample_idx + 1, :, :1].float()
    if pred_gs["opacity"].shape[2] > 1:
        opacity_dynamic = pred_gs["opacity"][sample_idx:sample_idx + 1, :, 1:].float()
        coef = torch.sigmoid(-opacity_dynamic[:, :, 0:1] * (time_basis.abs() - opacity_dynamic[:, :, 1:])) / torch.sigmoid(opacity_dynamic[:, :, 0:1] * opacity_dynamic[:, :, 1:])
        opacity = opacity * coef
    return {
        "rgb": pred_gs["rgb"][sample_idx:sample_idx + 1].float(),
        "opacity": opacity,
        "scale": pred_gs["scale"][sample_idx:sample_idx + 1].float(),
        "xyz": xyz,
        "rot": rot,
    }


def render_static_anchor(anchor, background, opt):
    gaussians = static_anchor_to_single_frame_gaussians(anchor)
    with autocast("cuda", enabled=False):
        render_pkg = gaussian_renderer_dynamic.render(gaussians, background, timestamps=None, opt=opt, anchor_time=None, training=False)
    return to_nchw(render_pkg["render"].clamp(0.0, 1.0))


def render_batch_with_gs(tasks, model, opt, device):
    frames, depths, timestamps, targets = load_task_batch(tasks, opt, device)
    opt.output_frames = 3
    model.opt.output_frames = 3
    anchor_time = torch.tensor([0.0, 1.0], device=device)
    with torch.no_grad():
        decoder_out = model.forward_gaussians(frames, depths, timestamps)
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
    return decoder_out["pred_gs"], to_nchw(render_pkg["render"][:, 1].clamp(0.0, 1.0)), to_nchw(targets)


def compute_metrics(pred, target, lpips_model, ms_ssim_module):
    return {
        "psnr": psnr_metric(pred, target),
        "ssim": ssim_metric(pred, target),
        "ms_ssim": ms_ssim_metric(pred, target, ms_ssim_module),
        "lpips": lpips_metric(pred, target, lpips_model),
    }


def optional_delta(value, base):
    if value is None or base is None:
        return None
    return float(value) - float(base)


def load_rate_reference(path):
    out = {}
    for row in read_csv(path):
        gap = int(row["reference_gap"])
        out[gap] = float(row["q12_main_anchor_mib_per_frame"])
    return out


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["gap"]), row["method"], int(row["side_bits"]))].append(row)
    out = []
    for (gap, method, side_bits), group in sorted(groups.items()):
        out.append({
            "gap": gap,
            "method": method,
            "side_bits": side_bits,
            "task_count": len(group),
            "mean_psnr": mean(row["psnr"] for row in group),
            "min_psnr": percentile((row["psnr"] for row in group), 0),
            "p10_psnr": percentile((row["psnr"] for row in group), 10),
            "mean_ssim": mean(row["ssim"] for row in group),
            "min_ssim": percentile((row["ssim"] for row in group), 0),
            "p10_ssim": percentile((row["ssim"] for row in group), 10),
            "mean_ms_ssim": mean(row["ms_ssim"] for row in group),
            "mean_lpips": mean(row["lpips"] for row in group),
            "p90_lpips": percentile((row["lpips"] for row in group), 90),
            "mean_payload_bytes": mean(row["payload_bytes"] for row in group),
            "mean_side_mib_per_intermediate": mean(row["side_mib_per_intermediate"] for row in group),
            "mean_direct_total_mib_per_frame_ref": mean(row["direct_total_mib_per_frame_ref"] for row in group),
            "mean_delta_psnr_vs_original": mean(row["delta_psnr_vs_original"] for row in group),
            "mean_delta_ssim_vs_original": mean(row["delta_ssim_vs_original"] for row in group),
            "mean_delta_ms_ssim_vs_original": mean(row["delta_ms_ssim_vs_original"] for row in group),
            "mean_delta_lpips_vs_original": mean(row["delta_lpips_vs_original"] for row in group),
        })
    return out


def put_label(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def make_canvas(target, original, corrected, task, original_metrics, corrected_metrics, payload_bytes, method):
    target_rgb = tensor_to_rgb8(target)
    original_rgb = tensor_to_rgb8(original)
    corrected_rgb = tensor_to_rgb8(corrected)
    h, w, _ = target_rgb.shape
    header = 58
    canvas = np.zeros((h + header, w * 3, 3), dtype=np.uint8)
    canvas[header:, :w] = target_rgb
    canvas[header:, w:2 * w] = original_rgb
    canvas[header:, 2 * w:] = corrected_rgb
    title = f"gap{task['reference_gap']} {task['sequence']} target {task['target_index']} t={task['normalized_time']:.3f}"
    put_label(canvas, title, 8, 18)
    put_label(canvas, "target", 8, 45)
    put_label(canvas, f"original P {original_metrics['psnr']:.2f} L {format_optional(original_metrics['lpips'])}", w + 8, 45)
    put_label(canvas, f"{method} P {corrected_metrics['psnr']:.2f} L {format_optional(corrected_metrics['lpips'])} B {payload_bytes:.0f}", 2 * w + 8, 45)
    return canvas


def save_best_contact_sheet(best_method_rows, rendered_cache, args):
    if not best_method_rows:
        return None
    worst = sorted(best_method_rows, key=lambda row: float(row["lpips"]) if row["lpips"] is not None else -1.0, reverse=True)[: args.top_badcases]
    frames = []
    for row in worst:
        cached = rendered_cache.get((row["task_id"], row["method"], int(row["side_bits"])))
        if cached is None:
            continue
        frames.append(make_canvas(
            cached["target"],
            cached["original"],
            cached["corrected"],
            cached["task"],
            cached["original_metrics"],
            cached["corrected_metrics"],
            cached["payload_bytes"],
            row["method"],
        ))
    if not frames:
        return None
    h, w, _ = frames[0].shape
    columns = min(args.contact_columns, len(frames))
    rows = int(math.ceil(len(frames) / columns))
    sheet = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        r = idx // columns
        c = idx % columns
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame
    path = args.heavy_root / "stage155_best_setting_worst_lpips_contact_sheet.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return str(path)


def choose_best_summary(summary_rows):
    candidates = [row for row in summary_rows if row["method"] == "image_residual_sideinfo_full_frame" and float(row["mean_psnr"]) >= 26.0]
    if candidates:
        return sorted(candidates, key=lambda row: (float(row["mean_payload_bytes"]), float(row["mean_lpips"] or 999.0)))[0]
    candidates = [row for row in summary_rows if row["method"] == "image_residual_sideinfo_full_frame"]
    if candidates:
        return sorted(candidates, key=lambda row: float(row["mean_psnr"]), reverse=True)[0]
    return None


def build_badcases(rows, best):
    if best is None:
        return []
    selected = [row for row in rows if row["method"] == best["method"] and int(row["side_bits"]) == int(best["side_bits"])]
    cases = {
        "best_setting_highest_lpips": sorted([row for row in selected if row["lpips"] is not None], key=lambda row: float(row["lpips"]), reverse=True),
        "best_setting_lowest_psnr": sorted(selected, key=lambda row: float(row["psnr"])),
        "best_setting_lowest_ssim": sorted(selected, key=lambda row: float(row["ssim"])),
    }
    out = []
    for rank_type, group in cases.items():
        for rank, row in enumerate(group[:12], start=1):
            item = {key: row.get(key) for key in BADCASE_FIELDS}
            item["rank_type"] = rank_type
            item["rank"] = rank
            out.append(item)
    return out


def write_report(summary_rows, diagnostics, badcase_rows, package, path):
    lines = [
        "# Stage155 StreamSplat-Base Side-Info Upper-Bound Sweep",
        "",
        "## Summary",
        "",
        "| gap | method | bits | tasks | PSNR mean | PSNR p10 | SSIM mean | MS-SSIM mean | LPIPS mean | LPIPS p90 | payload bytes | direct rate ref | delta PSNR | delta LPIPS |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['gap']} | {row['method']} | {row['side_bits']} | {row['task_count']} | {float(row['mean_psnr']):.6f} | {float(row['p10_psnr']):.6f} | {float(row['mean_ssim']):.6f} | {format_optional(row['mean_ms_ssim'])} | {format_optional(row['mean_lpips'])} | {format_optional(row['p90_lpips'])} | {format_optional(row['mean_payload_bytes'])} | {format_optional(row['mean_direct_total_mib_per_frame_ref'])} | {format_optional(row['mean_delta_psnr_vs_original'])} | {format_optional(row['mean_delta_lpips_vs_original'])} |"
        )
    shape_matches = sum(1 for row in diagnostics if row["shape_matches_target_dense_anchor"])
    lines.extend([
        "",
        "## Gaussian-Domain Diagnostic",
        "",
        f"- Shape-matched tasks: `{shape_matches}/{len(diagnostics)}`.",
        f"- Max static-eval render diff vs dynamic render: `{package['max_static_eval_render_diff_vs_dynamic']}`.",
        "- If shape-matched tasks are zero, direct dense-anchor residual side-info is not a valid final Gaussian-domain codec for original StreamSplat because the base has a different Gaussian correspondence/count from Stage61 dense target anchors.",
        "",
        "## Best Setting",
        "",
        f"- Best sampled setting: `{package['best_setting']}`",
        f"- Worst-LPIPS contact sheet: `{package['best_contact_sheet']}`",
        "",
        "## Bad Cases",
        "",
        "| rank type | rank | sequence | gap | target | method | bits | PSNR | SSIM | LPIPS | payload bytes |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in badcase_rows:
        lines.append(
            f"| {row['rank_type']} | {row['rank']} | {row['sequence']} | {row['gap']} | {row['target_index']} | {row['method']} | {row['side_bits']} | {float(row['psnr']):.6f} | {float(row['ssim']):.6f} | {format_optional(row['lpips'])} | {float(row['payload_bytes']):.3f} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        "- `image_residual_sideinfo_full_frame` is an upper-bound diagnostic, not the final GS-feature method.",
        "- A successful high-rate setting proves that the original StreamSplat base can be made visually and numerically strong if enough rate-counted auxiliary information is available.",
        "- If dense-anchor shape matching fails, the next final-method path should be a StreamSplat-guided adapter or a residual codec defined on the original StreamSplat target-time Gaussian set, not direct residuals to Stage61 dense anchors.",
        "",
        "## Outputs",
        "",
        f"- rows CSV: `{package['rows_csv']}`",
        f"- summary CSV: `{package['summary_csv']}`",
        f"- diagnostics CSV: `{package['diagnostics_csv']}`",
        f"- badcases CSV: `{package['badcases_csv']}`",
        f"- summary JSON: `{package['summary_json']}`",
        f"- package JSON: `{package['package_json']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stage147_rows", type=Path, default=DEFAULT_STAGE147_ROWS)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--task_split", default="eval")
    parser.add_argument("--codecs", nargs="+", default=["q12"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--max_tasks", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_residual_bits", nargs="+", type=int, default=[3, 4, 5, 6])
    parser.add_argument("--zlib_level", type=int, default=9)
    parser.add_argument("--top_badcases", type=int, default=12)
    parser.add_argument("--contact_columns", type=int, default=4)
    parser.add_argument("--disable_lpips", action="store_true")
    parser.add_argument("--disable_ms_ssim", action="store_true")
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.davis_root.exists():
        raise FileNotFoundError(args.davis_root)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    opt = Options()
    opt.resume = str(args.checkpoint)
    opt.compile = False
    opt.input_frames = 1
    opt.output_frames = 3
    opt.epoch = 0
    model = load_model(deepcopy(opt), args.checkpoint, str(device))
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    lpips_model, lpips_error, ms_ssim_module, ms_ssim_error = load_metric_modules(device, args.disable_lpips, args.disable_ms_ssim)
    tasks = select_balanced(parse_task_rows(args.task_manifest, args.task_split, args.codecs, args.gaps), args.max_tasks, args.seed)
    dense_index = build_dense_index(args.dense_manifest, sorted({row["split"] for row in tasks}))
    rate_ref = load_rate_reference(args.stage147_rows)
    rows = []
    diagnostics = []
    rendered_cache = {}
    max_static_diff = 0.0
    for start in range(0, len(tasks), args.batch_size):
        batch_tasks = tasks[start:start + args.batch_size]
        pred_gs, original_render, target = render_batch_with_gs(batch_tasks, model, opt, device)
        for local_idx, task in enumerate(batch_tasks):
            original_i = original_render[local_idx:local_idx + 1]
            target_i = target[local_idx:local_idx + 1]
            original_metrics = compute_metrics(original_i, target_i, lpips_model, ms_ssim_module)
            gap = int(task["reference_gap"])
            q12_ref = float(rate_ref.get(gap, 0.0))
            rows.append({
                "task_id": task["task_id"],
                "sequence": task["sequence"],
                "gap": gap,
                "codec": task["codec"],
                "target_index": task["target_index"],
                "normalized_time": task["normalized_time"],
                "method": "original_streamsplat_base",
                "side_bits": 0,
                **original_metrics,
                "payload_bytes": 0,
                "side_mib_per_intermediate": 0.0,
                "q12_main_anchor_mib_per_frame_ref": q12_ref,
                "direct_total_mib_per_frame_ref": q12_ref,
                "delta_psnr_vs_original": 0.0,
                "delta_ssim_vs_original": 0.0,
                "delta_ms_ssim_vs_original": 0.0,
                "delta_lpips_vs_original": 0.0,
            })
            static_anchor = stream_gaussians_at_time(pred_gs, float(task["normalized_time"]), opt, sample_idx=local_idx)
            static_render = render_static_anchor(static_anchor, background, opt)
            static_diff = float((static_render - original_i).abs().max().item())
            max_static_diff = max(max_static_diff, static_diff)
            dense_key = (task["dataset"], task["split"], task["sequence"], task["target_index"])
            target_item, target_side = dense_index[dense_key]
            target_anchor = load_anchor(target_item, target_side, device, bits=None, cache=None)
            base_attrs = flatten_static_anchor(static_anchor)
            target_attrs = flatten_static_anchor(target_anchor)
            shape_match = tuple(base_attrs.shape) == tuple(target_attrs.shape)
            diagnostics.append({
                "task_id": task["task_id"],
                "sequence": task["sequence"],
                "gap": gap,
                "target_index": task["target_index"],
                "base_gaussian_count": int(base_attrs.shape[1]),
                "target_gaussian_count": int(target_attrs.shape[1]),
                "base_attr_shape": str(tuple(base_attrs.shape)),
                "target_attr_shape": str(tuple(target_attrs.shape)),
                "shape_matches_target_dense_anchor": shape_match,
                "static_eval_render_max_abs_diff_vs_dynamic": static_diff,
            })
            for bits in args.image_residual_bits:
                payload, corrected, info = encode_image_residual(original_i, target_i, bits, args.zlib_level)
                corrected_metrics = compute_metrics(corrected, target_i, lpips_model, ms_ssim_module)
                side_mib = float(info["payload_bytes"]) / (1024.0 * 1024.0)
                method = "image_residual_sideinfo_full_frame"
                row = {
                    "task_id": task["task_id"],
                    "sequence": task["sequence"],
                    "gap": gap,
                    "codec": task["codec"],
                    "target_index": task["target_index"],
                    "normalized_time": task["normalized_time"],
                    "method": method,
                    "side_bits": int(bits),
                    **corrected_metrics,
                    "payload_bytes": info["payload_bytes"],
                    "side_mib_per_intermediate": side_mib,
                    "q12_main_anchor_mib_per_frame_ref": q12_ref,
                    "direct_total_mib_per_frame_ref": q12_ref + side_mib,
                    "delta_psnr_vs_original": optional_delta(corrected_metrics["psnr"], original_metrics["psnr"]),
                    "delta_ssim_vs_original": optional_delta(corrected_metrics["ssim"], original_metrics["ssim"]),
                    "delta_ms_ssim_vs_original": optional_delta(corrected_metrics["ms_ssim"], original_metrics["ms_ssim"]),
                    "delta_lpips_vs_original": optional_delta(corrected_metrics["lpips"], original_metrics["lpips"]),
                }
                rows.append(row)
                rendered_cache[(task["task_id"], method, int(bits))] = {
                    "task": task,
                    "target": target_i.detach().cpu(),
                    "original": original_i.detach().cpu(),
                    "corrected": corrected.detach().cpu(),
                    "original_metrics": original_metrics,
                    "corrected_metrics": corrected_metrics,
                    "payload_bytes": info["payload_bytes"],
                }
        print(json.dumps({"processed": min(start + args.batch_size, len(tasks)), "total": len(tasks)}), flush=True)
        del pred_gs, original_render, target
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary_rows = summarize(rows)
    best = choose_best_summary(summary_rows)
    best_rows = [] if best is None else [row for row in rows if row["method"] == best["method"] and int(row["side_bits"]) == int(best["side_bits"])]
    best_contact = save_best_contact_sheet(best_rows, rendered_cache, args)
    badcase_rows = build_badcases(rows, best)
    for row in badcase_rows:
        row["contact_sheet_path"] = best_contact or ""
    rows_csv = args.summary_root / "stage155_streamsplat_base_sideinfo_rows.csv"
    summary_csv = args.summary_root / "stage155_streamsplat_base_sideinfo_summary.csv"
    diagnostics_csv = args.summary_root / "stage155_streamsplat_base_gaussian_diagnostics.csv"
    badcases_csv = args.summary_root / "stage155_streamsplat_base_sideinfo_badcases.csv"
    summary_json = args.summary_root / "stage155_streamsplat_base_sideinfo_summary.json"
    package_json = args.summary_root / "stage155_streamsplat_base_sideinfo_upper_bound_package.json"
    report_md = args.summary_root / "stage155_streamsplat_base_sideinfo_upper_bound_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(diagnostics, diagnostics_csv, DIAGNOSTIC_FIELDS)
    write_csv(badcase_rows, badcases_csv, BADCASE_FIELDS)
    package = {
        "stage": 155,
        "mode": "StreamSplat-base side-info upper-bound sweep",
        "task_count": len(tasks),
        "correction_status": "image_residual_upper_bound_diagnostic_not_final_gs_codec",
        "gaussian_domain_shape_matched_tasks": sum(1 for row in diagnostics if row["shape_matches_target_dense_anchor"]),
        "gaussian_domain_task_count": len(diagnostics),
        "max_static_eval_render_diff_vs_dynamic": max_static_diff,
        "best_setting": best,
        "best_contact_sheet": best_contact,
        "summary_rows": summary_rows,
        "lpips_available": lpips_model is not None,
        "lpips_error": lpips_error,
        "ms_ssim_available": ms_ssim_module is not None,
        "ms_ssim_error": ms_ssim_error,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "diagnostics_csv": str(diagnostics_csv),
        "badcases_csv": str(badcases_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "heavy_root": str(args.heavy_root),
        "notes": "Image residual side-info is a rate-counted achievability upper bound. It is not the final Gaussian-feature method.",
    }
    summary_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary_rows, diagnostics, badcase_rows, package, report_md)
    print(json.dumps({"package": str(package_json), "best_setting": best, "contact_sheet": best_contact}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
