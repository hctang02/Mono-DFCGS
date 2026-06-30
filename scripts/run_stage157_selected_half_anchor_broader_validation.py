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
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage157_selected_half_anchor_broader_validation"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage157_selected_half_anchor_broader_validation")
DEFAULT_STAGE147_ROWS = REPO_ROOT / "experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_rows.csv"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, unflatten_static_anchor  # noqa: E402
from mono_dfcgs.residual_sideinfo_codec import decode_residual_sideinfo_entropy, encode_topk_residual_sideinfo_entropy  # noqa: E402
from scripts.run_stage153_middle_multimetric_badcase_eval import format_optional, load_metric_modules, mean, percentile, tensor_to_rgb8  # noqa: E402
from scripts.run_stage155_streamsplat_base_sideinfo_upper_bound import compute_metrics, load_rate_reference, render_batch_with_gs, render_static_anchor, stream_gaussians_at_time  # noqa: E402
from scripts.run_stage156_streamsplat_half_anchor_gaussian_residual import split_half_anchor  # noqa: E402
from scripts.run_stage6_export_real_anchor_dataset import load_model  # noqa: E402
from scripts.run_stage72_original_davis_baseline import DEFAULT_DAVIS_ROOT  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_DENSE_MANIFEST, DEFAULT_TASK_MANIFEST, build_dense_index, load_anchor, parse_task_rows, select_balanced  # noqa: E402


ROW_FIELDS = [
    "task_id", "sequence", "gap", "codec", "target_index", "normalized_time",
    "method", "selected_half", "keep_fraction", "side_bits",
    "psnr", "ssim", "ms_ssim", "lpips",
    "original_psnr", "original_ssim", "original_ms_ssim", "original_lpips",
    "payload_bytes", "selector_payload_bytes", "side_mib_per_intermediate",
    "q12_main_anchor_mib_per_frame_ref", "direct_total_mib_per_frame_ref",
    "delta_psnr_vs_original", "delta_ssim_vs_original", "delta_ms_ssim_vs_original", "delta_lpips_vs_original",
]

SUMMARY_FIELDS = [
    "gap", "method", "keep_fraction", "side_bits", "task_count",
    "mean_psnr", "min_psnr", "p10_psnr",
    "mean_ssim", "p10_ssim", "mean_ms_ssim",
    "mean_lpips", "p90_lpips",
    "mean_original_psnr", "mean_original_ssim", "mean_original_ms_ssim", "mean_original_lpips",
    "mean_payload_bytes", "mean_side_mib_per_intermediate", "mean_direct_total_mib_per_frame_ref",
    "mean_delta_psnr_vs_original", "mean_delta_ssim_vs_original", "mean_delta_ms_ssim_vs_original", "mean_delta_lpips_vs_original",
]

BADCASE_FIELDS = [
    "rank_type", "rank", "task_id", "sequence", "gap", "target_index", "normalized_time",
    "selected_half", "psnr", "ssim", "ms_ssim", "lpips", "payload_bytes", "contact_sheet_path",
]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def side_mib(bytes_count):
    return float(bytes_count) / (1024.0 * 1024.0)


def optional_delta(value, base):
    if value is None or base is None:
        return None
    return float(value) - float(base)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["gap"]), row["method"], float(row["keep_fraction"]), int(row["side_bits"]))].append(row)
    out = []
    for (gap, method, keep_fraction, side_bits), group in sorted(groups.items()):
        out.append({
            "gap": gap,
            "method": method,
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
            "mean_original_psnr": mean(row["original_psnr"] for row in group),
            "mean_original_ssim": mean(row["original_ssim"] for row in group),
            "mean_original_ms_ssim": mean(row["original_ms_ssim"] for row in group),
            "mean_original_lpips": mean(row["original_lpips"] for row in group),
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


def make_canvas(cached, row):
    target_rgb = cached["target"]
    original_rgb = cached["original"]
    corrected_rgb = cached["corrected"]
    h, w, _ = target_rgb.shape
    header = 58
    canvas = np.zeros((h + header, w * 3, 3), dtype=np.uint8)
    canvas[header:, :w] = target_rgb
    canvas[header:, w:2 * w] = original_rgb
    canvas[header:, 2 * w:] = corrected_rgb
    title = f"gap{row['gap']} {row['sequence']} target {row['target_index']} t={float(row['normalized_time']):.3f}"
    put_label(canvas, title, 8, 18)
    put_label(canvas, "target", 8, 45)
    put_label(canvas, f"original P {float(row['original_psnr']):.2f} L {format_optional(row['original_lpips'])}", w + 8, 45)
    put_label(canvas, f"{row['selected_half']} corrected P {float(row['psnr']):.2f} L {format_optional(row['lpips'])}", 2 * w + 8, 45)
    return canvas


def save_contact_sheet(rows, image_cache, args):
    worst = sorted(rows, key=lambda row: float(row["lpips"]) if row["lpips"] is not None else -1.0, reverse=True)[: args.top_badcases]
    frames = []
    for row in worst:
        cached = image_cache.get(row["task_id"])
        if cached is not None:
            frames.append(make_canvas(cached, row))
    if not frames:
        return None
    h, w, _ = frames[0].shape
    columns = min(args.contact_columns, len(frames))
    rows_count = int(math.ceil(len(frames) / columns))
    sheet = np.zeros((rows_count * h, columns * w, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        r = idx // columns
        c = idx % columns
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame
    path = args.heavy_root / "stage157_selected_half_anchor_worst_lpips_contact_sheet.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return str(path)


def build_badcases(rows, contact_sheet):
    cases = {
        "highest_lpips": sorted([row for row in rows if row["lpips"] is not None], key=lambda row: float(row["lpips"]), reverse=True),
        "lowest_psnr": sorted(rows, key=lambda row: float(row["psnr"])),
        "lowest_ssim": sorted(rows, key=lambda row: float(row["ssim"])),
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


def write_report(summary_rows, badcase_rows, package, path):
    lines = [
        "# Stage157 Selected Half-Anchor Broader Validation",
        "",
        "## Summary",
        "",
        "| gap | tasks | PSNR | p10 PSNR | SSIM | MS-SSIM | LPIPS | p90 LPIPS | original PSNR | original LPIPS | payload bytes | direct rate ref | delta PSNR | delta LPIPS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['gap']} | {row['task_count']} | {float(row['mean_psnr']):.6f} | {float(row['p10_psnr']):.6f} | {float(row['mean_ssim']):.6f} | {format_optional(row['mean_ms_ssim'])} | {format_optional(row['mean_lpips'])} | {format_optional(row['p90_lpips'])} | {float(row['mean_original_psnr']):.6f} | {format_optional(row['mean_original_lpips'])} | {format_optional(row['mean_payload_bytes'])} | {format_optional(row['mean_direct_total_mib_per_frame_ref'])} | {format_optional(row['mean_delta_psnr_vs_original'])} | {format_optional(row['mean_delta_lpips_vs_original'])} |"
        )
    lines.extend([
        "",
        "## Bad Cases",
        "",
        f"- Worst-LPIPS contact sheet: `{package['contact_sheet']}`",
        "",
        "| rank type | rank | sequence | gap | target | half | PSNR | SSIM | LPIPS | payload bytes |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|---:|",
    ])
    for row in badcase_rows:
        lines.append(f"| {row['rank_type']} | {row['rank']} | {row['sequence']} | {row['gap']} | {row['target_index']} | {row['selected_half']} | {float(row['psnr']):.6f} | {float(row['ssim']):.6f} | {format_optional(row['lpips'])} | {float(row['payload_bytes']):.3f} |")
    lines.extend([
        "",
        "## Contract",
        "",
        "- This validates only `best_half_selector/keep1.0/q6`.",
        "- The target dense anchor is encoder-side only and is not a decoder input.",
        "- Decoder receives original StreamSplat base, entropy residual payload, normalized time, and one counted half-selector byte.",
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
    parser.add_argument("--max_tasks", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--keep_fraction", type=float, default=1.0)
    parser.add_argument("--side_bits", type=int, default=6)
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
    image_cache = {}
    for start in range(0, len(tasks), args.batch_size):
        batch_tasks = tasks[start:start + args.batch_size]
        pred_gs, original_render, target = render_batch_with_gs(batch_tasks, model, opt, device)
        for local_idx, task in enumerate(batch_tasks):
            original_i = original_render[local_idx:local_idx + 1]
            target_i = target[local_idx:local_idx + 1]
            original_metrics = compute_metrics(original_i, target_i, lpips_model, ms_ssim_module)
            stream_anchor = stream_gaussians_at_time(pred_gs, float(task["normalized_time"]), opt, sample_idx=local_idx)
            dense_key = (task["dataset"], task["split"], task["sequence"], task["target_index"])
            target_item, target_side = dense_index[dense_key]
            target_anchor = load_anchor(target_item, target_side, device, bits=None, cache=None)
            target_attrs = flatten_static_anchor(target_anchor)
            half_candidates = []
            for half in ["left", "right"]:
                half_anchor = split_half_anchor(stream_anchor, half)
                base_attrs = flatten_static_anchor(half_anchor)
                payload, info = encode_topk_residual_sideinfo_entropy(
                    base_attrs,
                    target_attrs,
                    args.keep_fraction,
                    args.side_bits,
                    zlib_level=args.zlib_level,
                )
                corrected_anchor = unflatten_static_anchor(decode_residual_sideinfo_entropy(base_attrs, payload))
                corrected_render = render_static_anchor(corrected_anchor, torch.tensor(opt.background_color, dtype=torch.float32, device=device), opt)
                corrected_metrics = compute_metrics(corrected_render, target_i, lpips_model, ms_ssim_module)
                half_candidates.append((half, corrected_render, corrected_metrics, int(info["payload_bytes"])))
            selected_half, selected_render, selected_metrics, payload_bytes = sorted(half_candidates, key=lambda item: float(item[2]["psnr"]), reverse=True)[0]
            payload_bytes += int(args.selector_payload_bytes)
            gap = int(task["reference_gap"])
            q12_ref = float(rate_ref.get(gap, 0.0))
            row = {
                "task_id": task["task_id"],
                "sequence": task["sequence"],
                "gap": gap,
                "codec": task["codec"],
                "target_index": task["target_index"],
                "normalized_time": task["normalized_time"],
                "method": "streamsplat_half_anchor_entropy_residual_best_half",
                "selected_half": selected_half,
                "keep_fraction": float(args.keep_fraction),
                "side_bits": int(args.side_bits),
                **selected_metrics,
                "original_psnr": original_metrics["psnr"],
                "original_ssim": original_metrics["ssim"],
                "original_ms_ssim": original_metrics["ms_ssim"],
                "original_lpips": original_metrics["lpips"],
                "payload_bytes": payload_bytes,
                "selector_payload_bytes": int(args.selector_payload_bytes),
                "side_mib_per_intermediate": side_mib(payload_bytes),
                "q12_main_anchor_mib_per_frame_ref": q12_ref,
                "direct_total_mib_per_frame_ref": q12_ref + side_mib(payload_bytes),
                "delta_psnr_vs_original": optional_delta(selected_metrics["psnr"], original_metrics["psnr"]),
                "delta_ssim_vs_original": optional_delta(selected_metrics["ssim"], original_metrics["ssim"]),
                "delta_ms_ssim_vs_original": optional_delta(selected_metrics["ms_ssim"], original_metrics["ms_ssim"]),
                "delta_lpips_vs_original": optional_delta(selected_metrics["lpips"], original_metrics["lpips"]),
            }
            rows.append(row)
            image_cache[task["task_id"]] = {
                "target": tensor_to_rgb8(target_i.detach().cpu()),
                "original": tensor_to_rgb8(original_i.detach().cpu()),
                "corrected": tensor_to_rgb8(selected_render.detach().cpu()),
            }
        print(json.dumps({"processed": min(start + args.batch_size, len(tasks)), "total": len(tasks)}), flush=True)
        del pred_gs, original_render, target
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary_rows = summarize(rows)
    contact_sheet = save_contact_sheet(rows, image_cache, args)
    badcase_rows = build_badcases(rows, contact_sheet)
    rows_csv = args.summary_root / "stage157_selected_half_anchor_rows.csv"
    summary_csv = args.summary_root / "stage157_selected_half_anchor_summary.csv"
    badcases_csv = args.summary_root / "stage157_selected_half_anchor_badcases.csv"
    summary_json = args.summary_root / "stage157_selected_half_anchor_summary.json"
    package_json = args.summary_root / "stage157_selected_half_anchor_broader_validation_package.json"
    report_md = args.summary_root / "stage157_selected_half_anchor_broader_validation_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(badcase_rows, badcases_csv, BADCASE_FIELDS)
    package = {
        "stage": 157,
        "mode": "selected half-anchor broader validation",
        "policy": "streamsplat_half_anchor_entropy_residual_best_half_keep1_q6",
        "task_count": len(tasks),
        "summary_rows": summary_rows,
        "lpips_available": lpips_model is not None,
        "lpips_error": lpips_error,
        "ms_ssim_available": ms_ssim_module is not None,
        "ms_ssim_error": ms_ssim_error,
        "contact_sheet": contact_sheet,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "badcases_csv": str(badcases_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "heavy_root": str(args.heavy_root),
    }
    summary_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary_rows, badcase_rows, package, report_md)
    print(json.dumps({"package": str(package_json), "summary": summary_rows, "contact_sheet": contact_sheet}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
