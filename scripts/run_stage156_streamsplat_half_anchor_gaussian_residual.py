import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/streamsplat.safetensors")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage156_streamsplat_half_anchor_gaussian_residual"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage156_streamsplat_half_anchor_gaussian_residual")
DEFAULT_STAGE147_ROWS = REPO_ROOT / "experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_rows.csv"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, unflatten_static_anchor  # noqa: E402
from mono_dfcgs.residual_sideinfo_codec import (  # noqa: E402
    decode_residual_sideinfo_entropy,
    encode_topk_residual_sideinfo_entropy,
)
from scripts.run_stage153_middle_multimetric_badcase_eval import (  # noqa: E402
    format_optional,
    load_metric_modules,
    mean,
    percentile,
    tensor_to_rgb8,
)
from scripts.run_stage155_streamsplat_base_sideinfo_upper_bound import (  # noqa: E402
    compute_metrics,
    load_rate_reference,
    render_batch_with_gs,
    render_static_anchor,
    stream_gaussians_at_time,
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


ROW_FIELDS = [
    "task_id",
    "sequence",
    "gap",
    "codec",
    "target_index",
    "normalized_time",
    "method",
    "half_policy",
    "selected_half",
    "keep_fraction",
    "side_bits",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "payload_bytes",
    "selector_payload_bytes",
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
    "half_policy",
    "keep_fraction",
    "side_bits",
    "task_count",
    "mean_psnr",
    "min_psnr",
    "p10_psnr",
    "mean_ssim",
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
    "half_policy",
    "selected_half",
    "keep_fraction",
    "side_bits",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "payload_bytes",
    "contact_sheet_path",
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


def split_half_anchor(anchor, half):
    n = int(anchor["rgb"].shape[1])
    mid = n // 2
    if half == "left":
        sl = slice(0, mid)
    elif half == "right":
        sl = slice(mid, n)
    else:
        raise ValueError(f"unknown half {half}")
    return {key: value[:, sl].contiguous() for key, value in anchor.items()}


def side_mib(bytes_count):
    return float(bytes_count) / (1024.0 * 1024.0)


def optional_delta(value, base):
    if value is None or base is None:
        return None
    return float(value) - float(base)


def row_key(row):
    return (row["task_id"], float(row["keep_fraction"]), int(row["side_bits"]))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["gap"]), row["method"], row["half_policy"], float(row["keep_fraction"]), int(row["side_bits"]))].append(row)
    out = []
    for (gap, method, half_policy, keep_fraction, side_bits), group in sorted(groups.items()):
        out.append({
            "gap": gap,
            "method": method,
            "half_policy": half_policy,
            "keep_fraction": keep_fraction,
            "side_bits": side_bits,
            "task_count": len(group),
            "mean_psnr": mean(row["psnr"] for row in group),
            "min_psnr": percentile((row["psnr"] for row in group), 0),
            "p10_psnr": percentile((row["psnr"] for row in group), 10),
            "mean_ssim": mean(row["ssim"] for row in group),
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


def choose_best_summary(summary_rows):
    candidates = [
        row for row in summary_rows
        if row["method"] == "streamsplat_half_anchor_entropy_residual"
        and row["half_policy"] == "best_half_selector"
        and float(row["mean_psnr"]) >= 26.0
        and float(row["mean_delta_lpips_vs_original"]) <= 0.0
        and float(row["mean_delta_ssim_vs_original"]) >= 0.0
    ]
    if candidates:
        return sorted(candidates, key=lambda row: (float(row["mean_payload_bytes"]), float(row["mean_lpips"] or 999.0)))[0]
    candidates = [row for row in summary_rows if row["method"] == "streamsplat_half_anchor_entropy_residual" and row["half_policy"] == "best_half_selector"]
    if candidates:
        return sorted(candidates, key=lambda row: float(row["mean_psnr"]), reverse=True)[0]
    return None


def put_label(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def make_canvas(target_rgb, original_rgb, corrected_rgb, row):
    h, w, _ = target_rgb.shape
    header = 58
    canvas = np.zeros((h + header, w * 3, 3), dtype=np.uint8)
    canvas[header:, :w] = target_rgb
    canvas[header:, w:2 * w] = original_rgb
    canvas[header:, 2 * w:] = corrected_rgb
    title = f"gap{row['gap']} {row['sequence']} target {row['target_index']} t={float(row['normalized_time']):.3f}"
    put_label(canvas, title, 8, 18)
    put_label(canvas, "target", 8, 45)
    put_label(canvas, "original StreamSplat", w + 8, 45)
    put_label(canvas, f"{row['selected_half']} k{row['keep_fraction']} q{row['side_bits']} P {float(row['psnr']):.2f} L {format_optional(row['lpips'])}", 2 * w + 8, 45)
    return canvas


def save_contact_sheet(best_rows, image_cache, args):
    if not best_rows:
        return None
    worst = sorted(best_rows, key=lambda row: float(row["lpips"]) if row["lpips"] is not None else -1.0, reverse=True)[: args.top_badcases]
    frames = []
    for row in worst:
        cached = image_cache.get((row["task_id"], float(row["keep_fraction"]), int(row["side_bits"]), row["selected_half"]))
        if cached is None:
            continue
        frames.append(make_canvas(cached["target"], cached["original"], cached["corrected"], row))
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
    path = args.heavy_root / "stage156_best_half_selector_worst_lpips_contact_sheet.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return str(path)


def build_badcases(best_rows, contact_sheet):
    cases = {
        "best_half_highest_lpips": sorted([row for row in best_rows if row["lpips"] is not None], key=lambda row: float(row["lpips"]), reverse=True),
        "best_half_lowest_psnr": sorted(best_rows, key=lambda row: float(row["psnr"])),
        "best_half_lowest_ssim": sorted(best_rows, key=lambda row: float(row["ssim"])),
    }
    out = []
    for rank_type, group in cases.items():
        for rank, row in enumerate(group[:12], start=1):
            item = {key: row.get(key) for key in BADCASE_FIELDS}
            item["rank_type"] = rank_type
            item["rank"] = rank
            item["contact_sheet_path"] = contact_sheet or ""
            out.append(item)
    return out


def write_report(summary_rows, best, badcase_rows, package, path):
    lines = [
        "# Stage156 StreamSplat Half-Anchor Gaussian Residual Side-Info",
        "",
        "## Summary",
        "",
        "| gap | policy | keep | bits | tasks | PSNR mean | PSNR p10 | SSIM mean | MS-SSIM mean | LPIPS mean | LPIPS p90 | payload bytes | direct rate ref | delta PSNR | delta LPIPS |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["method"] == "original_streamsplat_base":
            continue
        lines.append(
            f"| {row['gap']} | {row['half_policy']} | {row['keep_fraction']} | {row['side_bits']} | {row['task_count']} | {float(row['mean_psnr']):.6f} | {float(row['p10_psnr']):.6f} | {float(row['mean_ssim']):.6f} | {format_optional(row['mean_ms_ssim'])} | {format_optional(row['mean_lpips'])} | {format_optional(row['p90_lpips'])} | {format_optional(row['mean_payload_bytes'])} | {format_optional(row['mean_direct_total_mib_per_frame_ref'])} | {format_optional(row['mean_delta_psnr_vs_original'])} | {format_optional(row['mean_delta_lpips_vs_original'])} |"
        )
    lines.extend([
        "",
        "## Original Baseline",
        "",
        "| gap | PSNR mean | SSIM mean | MS-SSIM mean | LPIPS mean |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        if row["method"] == "original_streamsplat_base":
            lines.append(f"| {row['gap']} | {float(row['mean_psnr']):.6f} | {float(row['mean_ssim']):.6f} | {format_optional(row['mean_ms_ssim'])} | {format_optional(row['mean_lpips'])} |")
    lines.extend([
        "",
        "## Best Setting",
        "",
        f"- Best sampled setting: `{best}`",
        f"- Worst-LPIPS contact sheet: `{package['best_contact_sheet']}`",
        "",
        "## Bad Cases",
        "",
        "| rank type | rank | sequence | gap | target | half | keep | bits | PSNR | SSIM | LPIPS | payload bytes |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in badcase_rows:
        lines.append(
            f"| {row['rank_type']} | {row['rank']} | {row['sequence']} | {row['gap']} | {row['target_index']} | {row['selected_half']} | {row['keep_fraction']} | {row['side_bits']} | {float(row['psnr']):.6f} | {float(row['ssim']):.6f} | {format_optional(row['lpips'])} | {float(row['payload_bytes']):.3f} |"
        )
    lines.extend([
        "",
        "## Contract",
        "",
        "- The target dense anchor is used encoder-side only to produce the residual payload and offline metrics.",
        "- Decoder inputs are original StreamSplat endpoint inputs/base, normalized time, entropy residual payload, and a counted one-byte half selector for `best_half_selector`.",
        "- Decoder does not receive unencoded target dense anchors, target RGB, or unencoded residual tensors.",
        "",
        "## Outputs",
        "",
        f"- rows CSV: `{package['rows_csv']}`",
        f"- summary CSV: `{package['summary_csv']}`",
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
    parser.add_argument("--keep_fractions", nargs="+", type=float, default=[0.2, 0.4, 1.0])
    parser.add_argument("--side_bits", nargs="+", type=int, default=[4, 6])
    parser.add_argument("--selector_payload_bytes", type=int, default=1)
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
    lpips_model, lpips_error, ms_ssim_module, ms_ssim_error = load_metric_modules(device, args.disable_lpips, args.disable_ms_ssim)
    tasks = select_balanced(parse_task_rows(args.task_manifest, args.task_split, args.codecs, args.gaps), args.max_tasks, args.seed)
    dense_index = build_dense_index(args.dense_manifest, sorted({row["split"] for row in tasks}))
    rate_ref = load_rate_reference(args.stage147_rows)
    rows = []
    half_rows = []
    image_cache = {}
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
                "half_policy": "full_original",
                "selected_half": "both",
                "keep_fraction": 0.0,
                "side_bits": 0,
                **original_metrics,
                "payload_bytes": 0,
                "selector_payload_bytes": 0,
                "side_mib_per_intermediate": 0.0,
                "q12_main_anchor_mib_per_frame_ref": q12_ref,
                "direct_total_mib_per_frame_ref": q12_ref,
                "delta_psnr_vs_original": 0.0,
                "delta_ssim_vs_original": 0.0,
                "delta_ms_ssim_vs_original": 0.0,
                "delta_lpips_vs_original": 0.0,
            })
            stream_anchor = stream_gaussians_at_time(pred_gs, float(task["normalized_time"]), opt, sample_idx=local_idx)
            dense_key = (task["dataset"], task["split"], task["sequence"], task["target_index"])
            target_item, target_side = dense_index[dense_key]
            target_anchor = load_anchor(target_item, target_side, device, bits=None, cache=None)
            target_attrs = flatten_static_anchor(target_anchor)
            for half in ["left", "right"]:
                half_anchor = split_half_anchor(stream_anchor, half)
                base_attrs = flatten_static_anchor(half_anchor)
                if tuple(base_attrs.shape) != tuple(target_attrs.shape):
                    raise RuntimeError(f"half anchor shape {tuple(base_attrs.shape)} does not match target {tuple(target_attrs.shape)}")
                for keep_fraction in args.keep_fractions:
                    for side_bits in args.side_bits:
                        payload, info = encode_topk_residual_sideinfo_entropy(
                            base_attrs,
                            target_attrs,
                            float(keep_fraction),
                            int(side_bits),
                            zlib_level=args.zlib_level,
                        )
                        decoded_attrs = decode_residual_sideinfo_entropy(base_attrs, payload)
                        corrected_anchor = unflatten_static_anchor(decoded_attrs)
                        corrected_render = render_static_anchor(corrected_anchor, torch.tensor(opt.background_color, dtype=torch.float32, device=device), opt)
                        corrected_metrics = compute_metrics(corrected_render, target_i, lpips_model, ms_ssim_module)
                        payload_bytes = int(info["payload_bytes"])
                        side = side_mib(payload_bytes)
                        row = {
                            "task_id": task["task_id"],
                            "sequence": task["sequence"],
                            "gap": gap,
                            "codec": task["codec"],
                            "target_index": task["target_index"],
                            "normalized_time": task["normalized_time"],
                            "method": "streamsplat_half_anchor_entropy_residual",
                            "half_policy": f"fixed_{half}_half",
                            "selected_half": half,
                            "keep_fraction": float(keep_fraction),
                            "side_bits": int(side_bits),
                            **corrected_metrics,
                            "payload_bytes": payload_bytes,
                            "selector_payload_bytes": 0,
                            "side_mib_per_intermediate": side,
                            "q12_main_anchor_mib_per_frame_ref": q12_ref,
                            "direct_total_mib_per_frame_ref": q12_ref + side,
                            "delta_psnr_vs_original": optional_delta(corrected_metrics["psnr"], original_metrics["psnr"]),
                            "delta_ssim_vs_original": optional_delta(corrected_metrics["ssim"], original_metrics["ssim"]),
                            "delta_ms_ssim_vs_original": optional_delta(corrected_metrics["ms_ssim"], original_metrics["ms_ssim"]),
                            "delta_lpips_vs_original": optional_delta(corrected_metrics["lpips"], original_metrics["lpips"]),
                        }
                        rows.append(row)
                        half_rows.append(row)
                        image_cache[(task["task_id"], float(keep_fraction), int(side_bits), half)] = {
                            "target": tensor_to_rgb8(target_i.detach().cpu()),
                            "original": tensor_to_rgb8(original_i.detach().cpu()),
                            "corrected": tensor_to_rgb8(corrected_render.detach().cpu()),
                        }
        print(json.dumps({"processed": min(start + args.batch_size, len(tasks)), "total": len(tasks)}), flush=True)
        del pred_gs, original_render, target
        if device.type == "cuda":
            torch.cuda.empty_cache()
    grouped = defaultdict(list)
    for row in half_rows:
        grouped[row_key(row)].append(row)
    best_rows = []
    for (_task_id, _keep, _bits), group in grouped.items():
        best = sorted(group, key=lambda row: float(row["psnr"]), reverse=True)[0]
        oracle = dict(best)
        oracle["half_policy"] = "best_half_selector"
        oracle["selector_payload_bytes"] = int(args.selector_payload_bytes)
        oracle["payload_bytes"] = int(oracle["payload_bytes"]) + int(args.selector_payload_bytes)
        oracle["side_mib_per_intermediate"] = side_mib(oracle["payload_bytes"])
        oracle["direct_total_mib_per_frame_ref"] = float(oracle["q12_main_anchor_mib_per_frame_ref"]) + float(oracle["side_mib_per_intermediate"])
        rows.append(oracle)
        best_rows.append(oracle)
    summary_rows = summarize(rows)
    best_summary = choose_best_summary(summary_rows)
    best_setting_rows = [] if best_summary is None else [
        row for row in best_rows
        if float(row["keep_fraction"]) == float(best_summary["keep_fraction"])
        and int(row["side_bits"]) == int(best_summary["side_bits"])
    ]
    contact_sheet = save_contact_sheet(best_setting_rows, image_cache, args)
    badcase_rows = build_badcases(best_setting_rows, contact_sheet)
    rows_csv = args.summary_root / "stage156_streamsplat_half_anchor_rows.csv"
    summary_csv = args.summary_root / "stage156_streamsplat_half_anchor_summary.csv"
    badcases_csv = args.summary_root / "stage156_streamsplat_half_anchor_badcases.csv"
    summary_json = args.summary_root / "stage156_streamsplat_half_anchor_summary.json"
    package_json = args.summary_root / "stage156_streamsplat_half_anchor_gaussian_residual_package.json"
    report_md = args.summary_root / "stage156_streamsplat_half_anchor_gaussian_residual_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(badcase_rows, badcases_csv, BADCASE_FIELDS)
    package = {
        "stage": 156,
        "mode": "StreamSplat half-anchor Gaussian residual side-info",
        "task_count": len(tasks),
        "method": "streamsplat_half_anchor_entropy_residual",
        "best_setting": best_summary,
        "best_contact_sheet": contact_sheet,
        "summary_rows": summary_rows,
        "lpips_available": lpips_model is not None,
        "lpips_error": lpips_error,
        "ms_ssim_available": ms_ssim_module is not None,
        "ms_ssim_error": ms_ssim_error,
        "selector_payload_bytes": int(args.selector_payload_bytes),
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "badcases_csv": str(badcases_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "heavy_root": str(args.heavy_root),
        "notes": "Target dense anchor is encoder-side only. Decoder receives original StreamSplat base, encoded residual payload, and counted half-selector metadata.",
    }
    summary_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary_rows, best_summary, badcase_rows, package, report_md)
    print(json.dumps({"package": str(package_json), "best_setting": best_summary, "contact_sheet": contact_sheet}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
