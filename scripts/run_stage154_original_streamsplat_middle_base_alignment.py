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
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage154_original_streamsplat_middle_base_alignment"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage154_original_streamsplat_middle_base_alignment")
DEFAULT_STAGE153_PAIR_ROWS = REPO_ROOT / "experiments/stage153_middle_multimetric_badcase_eval/stage153_middle_multimetric_pair_rows.csv"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
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
from scripts.run_stage6_export_real_anchor_dataset import (  # noqa: E402
    get_depth,
    get_image,
    load_model,
    normalize_depths,
)
from scripts.run_stage72_original_davis_baseline import DEFAULT_DAVIS_ROOT, render_original_batch  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_TASK_MANIFEST, parse_task_rows, select_balanced  # noqa: E402


ROW_FIELDS = [
    "task_id",
    "sequence",
    "gap",
    "codec",
    "left_index",
    "right_index",
    "target_index",
    "normalized_time",
    "method",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "linear_base_psnr",
    "stage151_recovered_psnr",
    "delta_psnr_vs_linear",
    "delta_psnr_vs_stage151_recovered",
    "linear_base_ssim",
    "stage151_recovered_ssim",
    "delta_ssim_vs_linear",
    "delta_ssim_vs_stage151_recovered",
    "linear_base_lpips",
    "stage151_recovered_lpips",
    "delta_lpips_vs_linear",
    "delta_lpips_vs_stage151_recovered",
]

SUMMARY_FIELDS = [
    "gap",
    "method",
    "task_count",
    "mean_psnr",
    "min_psnr",
    "p10_psnr",
    "mean_ssim",
    "min_ssim",
    "p10_ssim",
    "mean_ms_ssim",
    "min_ms_ssim",
    "p10_ms_ssim",
    "mean_lpips",
    "max_lpips",
    "p90_lpips",
    "mean_delta_psnr_vs_stage151_recovered",
    "mean_delta_ssim_vs_stage151_recovered",
    "mean_delta_lpips_vs_stage151_recovered",
]

BADCASE_FIELDS = [
    "rank_type",
    "rank",
    "task_id",
    "sequence",
    "gap",
    "target_index",
    "normalized_time",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "stage151_recovered_psnr",
    "stage151_recovered_ssim",
    "stage151_recovered_lpips",
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


def optional_float(row, key):
    value = row.get(key)
    if value in (None, "", "NA"):
        return None
    return float(value)


def optional_delta(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


def depth_path_for_rgb(rgb_path):
    rgb_path = Path(rgb_path)
    return Path(str(rgb_path).replace("JPEGImages", "depthImages")).with_name(f"{rgb_path.stem}_pred.png")


def load_task_batch(tasks, opt, device):
    frames = []
    depths = []
    targets = []
    timestamps = []
    for task in tasks:
        left_rgb = get_image(task["left_rgb_path"], opt.image_height, opt.image_width)
        right_rgb = get_image(task["right_rgb_path"], opt.image_height, opt.image_width)
        left_depth = get_depth(depth_path_for_rgb(task["left_rgb_path"]), opt.image_height, opt.image_width)
        right_depth = get_depth(depth_path_for_rgb(task["right_rgb_path"]), opt.image_height, opt.image_width)
        target_rgb = get_image(task["target_rgb_path"], opt.image_height, opt.image_width)
        frames.append(np.stack([left_rgb, right_rgb], axis=0))
        depths.append(np.stack([left_depth, right_depth], axis=0))
        targets.append(target_rgb)
        timestamps.append([0.0, float(task["normalized_time"]), 1.0])
    frame_tensor = torch.from_numpy(np.stack(frames, axis=0)).float().to(device) / 255.0
    frame_tensor = frame_tensor.permute(0, 1, 4, 2, 3)
    depth_tensor = torch.from_numpy(np.stack(depths, axis=0)).float().to(device).unsqueeze(2)
    depth_tensor = normalize_depths(depth_tensor)
    timestamp_tensor = torch.tensor(timestamps, dtype=torch.float32, device=device)
    target_tensor = torch.from_numpy(np.stack(targets, axis=0)).float().to(device) / 255.0
    target_tensor = target_tensor.permute(0, 3, 1, 2)
    return frame_tensor, depth_tensor, timestamp_tensor, target_tensor


def render_original_tasks(tasks, model, opt, device):
    frames, depths, timestamps, targets = load_task_batch(tasks, opt, device)
    opt.output_frames = 3
    model.opt.output_frames = 3
    with torch.no_grad():
        pred, gs_mib = render_original_batch(model, opt, frames, depths, timestamps)
    middle = to_nchw(pred[:, 1].clamp(0.0, 1.0))
    return middle, to_nchw(targets), float(gs_mib)


def compute_metrics(pred, target, lpips_model, ms_ssim_module):
    return {
        "psnr": psnr_metric(pred, target),
        "ssim": ssim_metric(pred, target),
        "ms_ssim": ms_ssim_metric(pred, target, ms_ssim_module),
        "lpips": lpips_metric(pred, target, lpips_model),
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["gap"]), row["method"])].append(row)
    out = []
    for (gap, method), group in sorted(groups.items()):
        out.append({
            "gap": gap,
            "method": method,
            "task_count": len(group),
            "mean_psnr": mean(row["psnr"] for row in group),
            "min_psnr": percentile((row["psnr"] for row in group), 0),
            "p10_psnr": percentile((row["psnr"] for row in group), 10),
            "mean_ssim": mean(row["ssim"] for row in group),
            "min_ssim": percentile((row["ssim"] for row in group), 0),
            "p10_ssim": percentile((row["ssim"] for row in group), 10),
            "mean_ms_ssim": mean(row["ms_ssim"] for row in group),
            "min_ms_ssim": percentile((row["ms_ssim"] for row in group), 0),
            "p10_ms_ssim": percentile((row["ms_ssim"] for row in group), 10),
            "mean_lpips": mean(row["lpips"] for row in group),
            "max_lpips": percentile((row["lpips"] for row in group), 100),
            "p90_lpips": percentile((row["lpips"] for row in group), 90),
            "mean_delta_psnr_vs_stage151_recovered": mean(row["delta_psnr_vs_stage151_recovered"] for row in group),
            "mean_delta_ssim_vs_stage151_recovered": mean(row["delta_ssim_vs_stage151_recovered"] for row in group),
            "mean_delta_lpips_vs_stage151_recovered": mean(row["delta_lpips_vs_stage151_recovered"] for row in group),
        })
    return out


def put_label(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def make_canvas(target, original, task, metrics, comparison):
    target_rgb = tensor_to_rgb8(target.unsqueeze(0) if target.dim() == 3 else target)
    original_rgb = tensor_to_rgb8(original.unsqueeze(0) if original.dim() == 3 else original)
    h, w, _ = target_rgb.shape
    header = 58
    canvas = np.zeros((h + header, w * 2, 3), dtype=np.uint8)
    canvas[header:, :w] = target_rgb
    canvas[header:, w:] = original_rgb
    title = f"gap{task['reference_gap']} {task['sequence']} target {task['target_index']} t={task['normalized_time']:.3f}"
    put_label(canvas, title, 8, 18)
    put_label(canvas, "target", 8, 45)
    recovered_psnr = comparison.get("recovered_psnr")
    suffix = "" if recovered_psnr is None else f" vs Stage151 P {float(recovered_psnr):.2f}"
    lpips_text = "NA" if metrics["lpips"] is None else f"{metrics['lpips']:.3f}"
    put_label(canvas, f"original P {metrics['psnr']:.2f} S {metrics['ssim']:.3f} L {lpips_text}{suffix}", w + 8, 45)
    return canvas


def build_badcases(rows, top_n):
    return {
        "highest_original_lpips": sorted([row for row in rows if row["lpips"] is not None], key=lambda row: float(row["lpips"]), reverse=True)[:top_n],
        "lowest_original_ssim": sorted(rows, key=lambda row: float(row["ssim"]))[:top_n],
        "lowest_original_psnr": sorted(rows, key=lambda row: float(row["psnr"]))[:top_n],
        "largest_psnr_drop_vs_stage151": sorted(
            [row for row in rows if row["delta_psnr_vs_stage151_recovered"] is not None],
            key=lambda row: float(row["delta_psnr_vs_stage151_recovered"]),
        )[:top_n],
    }


def save_contact_sheets(cases, rendered_cache, args):
    paths = {}
    for rank_type, group in cases.items():
        frames = []
        for row in group:
            cached = rendered_cache.get(row["task_id"])
            if cached is None:
                continue
            frames.append(make_canvas(cached["target"], cached["pred"], cached["task"], cached["metrics"], cached["comparison"]))
        if not frames:
            continue
        h, w, _ = frames[0].shape
        columns = min(args.contact_columns, len(frames))
        rows = int(math.ceil(len(frames) / columns))
        sheet = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
        for idx, frame in enumerate(frames):
            r = idx // columns
            c = idx % columns
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame
        path = args.heavy_root / f"stage154_badcases_{rank_type}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        paths[rank_type] = str(path)
    return paths


def write_report(summary_rows, badcase_rows, package, path):
    lines = [
        "# Stage154 Original StreamSplat Middle Base Alignment",
        "",
        "## Summary",
        "",
        "| gap | method | tasks | PSNR mean | PSNR p10 | SSIM mean | MS-SSIM mean | LPIPS mean | LPIPS p90 | delta PSNR vs Stage151 | delta LPIPS vs Stage151 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['gap']} | {row['method']} | {row['task_count']} | {float(row['mean_psnr']):.6f} | {float(row['p10_psnr']):.6f} | {float(row['mean_ssim']):.6f} | {format_optional(row['mean_ms_ssim'])} | {format_optional(row['mean_lpips'])} | {format_optional(row['p90_lpips'])} | {format_optional(row['mean_delta_psnr_vs_stage151_recovered'])} | {format_optional(row['mean_delta_lpips_vs_stage151_recovered'])} |"
        )
    lines.extend([
        "",
        "## Bad-Case Contact Sheets",
        "",
    ])
    for key, value in package["badcase_contact_sheets"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Worst Original StreamSplat Cases",
        "",
        "| rank type | rank | sequence | gap | target | original PSNR | original SSIM | original LPIPS | Stage151 PSNR | Stage151 LPIPS |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in badcase_rows[: min(48, len(badcase_rows))]:
        lines.append(
            f"| {row['rank_type']} | {row['rank']} | {row['sequence']} | {row['gap']} | {row['target_index']} | {float(row['psnr']):.6f} | {float(row['ssim']):.6f} | {format_optional(row['lpips'])} | {format_optional(row['stage151_recovered_psnr'])} | {format_optional(row['stage151_recovered_lpips'])} |"
        )
    lines.extend([
        "",
        "## Decision Use",
        "",
        "- Stage154 establishes the original StreamSplat-guided base profile on the same task-sampled diagnostic protocol as Stage153.",
        "- If original StreamSplat is visually more stable but lower PSNR, Stage155 should apply rate-counted side-info on top of this base rather than linear interpolation.",
        "- If original StreamSplat is not better on this task protocol, the next model stage still needs a StreamSplat-guided adapter rather than sparse linear residual correction.",
        "",
        "## Outputs",
        "",
        f"- rows CSV: `{package['rows_csv']}`",
        f"- badcases CSV: `{package['badcases_csv']}`",
        f"- summary CSV: `{package['summary_csv']}`",
        f"- summary JSON: `{package['summary_json']}`",
        f"- package JSON: `{package['package_json']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--stage153_pair_rows", type=Path, default=DEFAULT_STAGE153_PAIR_ROWS)
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--task_split", default="eval")
    parser.add_argument("--codecs", nargs="+", default=["q12"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--max_tasks", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=2)
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
    stage153 = {row["task_id"]: row for row in read_csv(args.stage153_pair_rows)}
    rows = []
    rendered_cache = {}
    total_gs_mib = 0.0
    for start in range(0, len(tasks), args.batch_size):
        batch_tasks = tasks[start:start + args.batch_size]
        pred, target, gs_mib = render_original_tasks(batch_tasks, model, opt, device)
        total_gs_mib += gs_mib
        for local_idx, task in enumerate(batch_tasks):
            pred_i = pred[local_idx:local_idx + 1]
            target_i = target[local_idx:local_idx + 1]
            metrics = compute_metrics(pred_i, target_i, lpips_model, ms_ssim_module)
            comparison = stage153.get(task["task_id"], {})
            linear_psnr = optional_float(comparison, "base_psnr")
            recovered_psnr = optional_float(comparison, "recovered_psnr")
            linear_ssim = optional_float(comparison, "base_ssim")
            recovered_ssim = optional_float(comparison, "recovered_ssim")
            linear_lpips = optional_float(comparison, "base_lpips")
            recovered_lpips = optional_float(comparison, "recovered_lpips")
            row = {
                "task_id": task["task_id"],
                "sequence": task["sequence"],
                "gap": task["reference_gap"],
                "codec": task["codec"],
                "left_index": task["left_index"],
                "right_index": task["right_index"],
                "target_index": task["target_index"],
                "normalized_time": task["normalized_time"],
                "method": "original_streamsplat_middle_base",
                **metrics,
                "linear_base_psnr": linear_psnr,
                "stage151_recovered_psnr": recovered_psnr,
                "delta_psnr_vs_linear": optional_delta(metrics["psnr"], linear_psnr),
                "delta_psnr_vs_stage151_recovered": optional_delta(metrics["psnr"], recovered_psnr),
                "linear_base_ssim": linear_ssim,
                "stage151_recovered_ssim": recovered_ssim,
                "delta_ssim_vs_linear": optional_delta(metrics["ssim"], linear_ssim),
                "delta_ssim_vs_stage151_recovered": optional_delta(metrics["ssim"], recovered_ssim),
                "linear_base_lpips": linear_lpips,
                "stage151_recovered_lpips": recovered_lpips,
                "delta_lpips_vs_linear": optional_delta(metrics["lpips"], linear_lpips),
                "delta_lpips_vs_stage151_recovered": optional_delta(metrics["lpips"], recovered_lpips),
            }
            rows.append(row)
            rendered_cache[task["task_id"]] = {
                "task": task,
                "pred": pred_i.detach().cpu(),
                "target": target_i.detach().cpu(),
                "metrics": metrics,
                "comparison": {"recovered_psnr": recovered_psnr},
            }
        print(json.dumps({"processed": min(start + args.batch_size, len(tasks)), "total": len(tasks)}), flush=True)
        del pred, target
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary_rows = summarize(rows)
    cases = build_badcases(rows, args.top_badcases)
    contact_sheets = save_contact_sheets(cases, rendered_cache, args)
    badcase_rows = []
    for rank_type, group in cases.items():
        for rank, row in enumerate(group, start=1):
            badcase_rows.append({
                "rank_type": rank_type,
                "rank": rank,
                "task_id": row["task_id"],
                "sequence": row["sequence"],
                "gap": row["gap"],
                "target_index": row["target_index"],
                "normalized_time": row["normalized_time"],
                "psnr": row["psnr"],
                "ssim": row["ssim"],
                "ms_ssim": row["ms_ssim"],
                "lpips": row["lpips"],
                "stage151_recovered_psnr": row["stage151_recovered_psnr"],
                "stage151_recovered_ssim": row["stage151_recovered_ssim"],
                "stage151_recovered_lpips": row["stage151_recovered_lpips"],
                "contact_sheet_path": contact_sheets.get(rank_type, ""),
            })
    rows_csv = args.summary_root / "stage154_original_streamsplat_middle_rows.csv"
    badcases_csv = args.summary_root / "stage154_original_streamsplat_middle_badcases.csv"
    summary_csv = args.summary_root / "stage154_original_streamsplat_middle_summary.csv"
    summary_json = args.summary_root / "stage154_original_streamsplat_middle_summary.json"
    package_json = args.summary_root / "stage154_original_streamsplat_middle_base_alignment_package.json"
    report_md = args.summary_root / "stage154_original_streamsplat_middle_base_alignment_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(badcase_rows, badcases_csv, BADCASE_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    package = {
        "stage": 154,
        "mode": "original StreamSplat middle base alignment",
        "method": "original_streamsplat_middle_base",
        "task_count": len(tasks),
        "checkpoint": str(args.checkpoint),
        "davis_root": str(args.davis_root),
        "summary_rows": summary_rows,
        "raw_pred_gs_mib_total_diagnostic": total_gs_mib,
        "raw_pred_gs_mib_per_task_diagnostic": total_gs_mib / max(len(tasks), 1),
        "lpips_available": lpips_model is not None,
        "lpips_error": lpips_error,
        "ms_ssim_available": ms_ssim_module is not None,
        "ms_ssim_error": ms_ssim_error,
        "badcase_contact_sheets": contact_sheets,
        "rows_csv": str(rows_csv),
        "badcases_csv": str(badcases_csv),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "heavy_root": str(args.heavy_root),
        "notes": "raw_pred_gs_mib is diagnostic tensor size, not codec rate. Heavy contact sheets are outside git.",
    }
    summary_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary_rows, badcase_rows, package, report_md)
    print(json.dumps({"package": str(package_json), "summary": summary_rows, "contact_sheets": contact_sheets}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
