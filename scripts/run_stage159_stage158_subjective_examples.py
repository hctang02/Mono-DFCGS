import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/streamsplat.safetensors")
DEFAULT_STAGE157_ROWS = REPO_ROOT / "experiments/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_rows.csv"
DEFAULT_STAGE147_ROWS = REPO_ROOT / "experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_rows.csv"
DEFAULT_STAGE72_KEYFRAME_ROWS = REPO_ROOT / "experiments/stage72_original_davis_baseline/stage72_original_davis_baseline_per_frame.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage159_stage158_subjective_examples"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage159_stage158_subjective_examples")
SELECTED_TASK_IDS = ["stage79_00023460", "stage79_00025922", "stage79_00030917"]


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, unflatten_static_anchor  # noqa: E402
from mono_dfcgs.residual_sideinfo_codec import decode_residual_sideinfo_entropy, encode_topk_residual_sideinfo_entropy  # noqa: E402
from scripts.run_stage153_middle_multimetric_badcase_eval import format_optional, load_metric_modules, tensor_to_rgb8  # noqa: E402
from scripts.run_stage155_streamsplat_base_sideinfo_upper_bound import compute_metrics, load_rate_reference, render_batch_with_gs, render_static_anchor, stream_gaussians_at_time  # noqa: E402
from scripts.run_stage156_streamsplat_half_anchor_gaussian_residual import split_half_anchor  # noqa: E402
from scripts.run_stage157_selected_half_anchor_broader_validation import side_mib  # noqa: E402
from scripts.run_stage6_export_real_anchor_dataset import load_model  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_DENSE_MANIFEST, DEFAULT_TASK_MANIFEST, build_dense_index, load_anchor, parse_task_rows  # noqa: E402


ROW_FIELDS = [
    "sequence", "task_id", "left_index", "target_index", "right_index", "normalized_time",
    "key_left_psnr", "key_left_ssim", "key_right_psnr", "key_right_ssim", "key_avg_psnr", "key_avg_ssim",
    "selected_half", "middle_psnr", "middle_ssim", "middle_ms_ssim", "middle_lpips",
    "original_middle_psnr", "original_middle_ssim", "original_middle_ms_ssim", "original_middle_lpips",
    "delta_psnr_vs_original", "delta_ssim_vs_original", "delta_ms_ssim_vs_original", "delta_lpips_vs_original",
    "payload_bytes", "selector_payload_bytes", "side_mib_per_intermediate", "direct_total_mib_per_frame_ref",
    "video_path", "contact_sheet_path",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def optional_delta(value, base):
    if value is None or base is None:
        return None
    return float(value) - float(base)


def load_keyframe_metrics(path):
    metrics = {}
    for row in read_csv(path):
        if row.get("frame_gap") != "4" or row.get("is_keyframe") != "True":
            continue
        sequence = row["sample"].split("/")[-1]
        metrics[(sequence, int(row["frame_index"]))] = {
            "psnr": float(row["psnr"]),
            "ssim": float(row["ssim"]),
        }
    return metrics


def load_stage157_expected(path):
    return {row["task_id"]: row for row in read_csv(path)}


def load_rgb8(path, height, width):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[0] != height or image.shape[1] != width:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def put_label(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def make_canvas(left_rgb, target_rgb, original_rgb, corrected_rgb, right_rgb, row):
    panels = [left_rgb, target_rgb, original_rgb, corrected_rgb, right_rgb]
    h, w, _ = target_rgb.shape
    header = 70
    canvas = np.zeros((h + header, w * len(panels), 3), dtype=np.uint8)
    for idx, panel in enumerate(panels):
        canvas[header:, idx * w:(idx + 1) * w] = panel
    title = f"gap4 {row['sequence']} frames {row['left_index']}-{row['target_index']}-{row['right_index']}"
    put_label(canvas, title, 8, 18)
    labels = [
        f"left key P {float(row['key_left_psnr']):.2f}",
        "target middle",
        f"orig P {float(row['original_middle_psnr']):.2f} L {format_optional(row['original_middle_lpips'])}",
        f"Stage158 P {float(row['middle_psnr']):.2f} L {format_optional(row['middle_lpips'])}",
        f"right key P {float(row['key_right_psnr']):.2f}",
    ]
    for idx, label in enumerate(labels):
        put_label(canvas, label, idx * w + 8, 47)
    put_label(canvas, f"payload {float(row['payload_bytes']):.0f} B, half {row['selected_half']}", 8, 65)
    return canvas


def render_task(task, expected_row, key_metrics, dense_index, model, opt, device, lpips_model, ms_ssim_module, rate_ref, args):
    pred_gs, original_render, target = render_batch_with_gs([task], model, opt, device)
    original_i = original_render[0:1]
    target_i = target[0:1]
    original_metrics = compute_metrics(original_i, target_i, lpips_model, ms_ssim_module)
    stream_anchor = stream_gaussians_at_time(pred_gs, float(task["normalized_time"]), opt, sample_idx=0)
    dense_key = (task["dataset"], task["split"], task["sequence"], task["target_index"])
    target_item, target_side = dense_index[dense_key]
    target_anchor = load_anchor(target_item, target_side, device, bits=None, cache=None)
    target_attrs = flatten_static_anchor(target_anchor)
    candidates = []
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
        candidates.append((half, corrected_render, corrected_metrics, int(info["payload_bytes"])))
    selected_half, corrected_render, corrected_metrics, payload_bytes = sorted(candidates, key=lambda item: float(item[2]["psnr"]), reverse=True)[0]
    payload_bytes += int(args.selector_payload_bytes)
    sequence = task["sequence"]
    left_index = int(task["left_index"])
    right_index = int(task["right_index"])
    left_key = key_metrics[(sequence, left_index)]
    right_key = key_metrics[(sequence, right_index)]
    gap = int(task["reference_gap"])
    q12_ref = float(rate_ref.get(gap, 0.0))
    row = {
        "sequence": sequence,
        "task_id": task["task_id"],
        "left_index": left_index,
        "target_index": int(task["target_index"]),
        "right_index": right_index,
        "normalized_time": float(task["normalized_time"]),
        "key_left_psnr": left_key["psnr"],
        "key_left_ssim": left_key["ssim"],
        "key_right_psnr": right_key["psnr"],
        "key_right_ssim": right_key["ssim"],
        "key_avg_psnr": (left_key["psnr"] + right_key["psnr"]) / 2.0,
        "key_avg_ssim": (left_key["ssim"] + right_key["ssim"]) / 2.0,
        "selected_half": selected_half,
        "middle_psnr": corrected_metrics["psnr"],
        "middle_ssim": corrected_metrics["ssim"],
        "middle_ms_ssim": corrected_metrics["ms_ssim"],
        "middle_lpips": corrected_metrics["lpips"],
        "original_middle_psnr": original_metrics["psnr"],
        "original_middle_ssim": original_metrics["ssim"],
        "original_middle_ms_ssim": original_metrics["ms_ssim"],
        "original_middle_lpips": original_metrics["lpips"],
        "delta_psnr_vs_original": optional_delta(corrected_metrics["psnr"], original_metrics["psnr"]),
        "delta_ssim_vs_original": optional_delta(corrected_metrics["ssim"], original_metrics["ssim"]),
        "delta_ms_ssim_vs_original": optional_delta(corrected_metrics["ms_ssim"], original_metrics["ms_ssim"]),
        "delta_lpips_vs_original": optional_delta(corrected_metrics["lpips"], original_metrics["lpips"]),
        "payload_bytes": payload_bytes,
        "selector_payload_bytes": int(args.selector_payload_bytes),
        "side_mib_per_intermediate": side_mib(payload_bytes),
        "direct_total_mib_per_frame_ref": q12_ref + side_mib(payload_bytes),
    }
    if expected_row is not None:
        row["stage157_expected_psnr"] = float(expected_row["psnr"])
        row["stage157_expected_lpips"] = float(expected_row["lpips"])
    target_rgb8 = tensor_to_rgb8(target_i.detach().cpu())
    h, w, _ = target_rgb8.shape
    images = {
        "left": load_rgb8(task["left_rgb_path"], h, w),
        "target": target_rgb8,
        "original": tensor_to_rgb8(original_i.detach().cpu()),
        "corrected": tensor_to_rgb8(corrected_render.detach().cpu()),
        "right": load_rgb8(task["right_rgb_path"], h, w),
    }
    del pred_gs, original_render, target
    return row, images


def write_report(package, rows, path):
    lines = [
        "# Stage159 Stage158 Subjective Examples",
        "",
        "## Videos",
        "",
        f"- Video: `{package['video_path']}`",
        f"- Contact sheet: `{package['contact_sheet_path']}`",
        f"- Video file size: `{package['video_file_bytes']}` bytes",
        f"- Contact sheet file size: `{package['contact_sheet_file_bytes']}` bytes",
        "",
        "## Rows",
        "",
        "| sequence | frames | key avg PSNR/SSIM | middle PSNR/SSIM/MS-SSIM/LPIPS | original PSNR/LPIPS | delta PSNR/LPIPS | payload bytes | direct rate ref |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sequence']} | {row['left_index']}-{row['target_index']}-{row['right_index']} | "
            f"{float(row['key_avg_psnr']):.3f}/{float(row['key_avg_ssim']):.4f} | "
            f"{float(row['middle_psnr']):.3f}/{float(row['middle_ssim']):.4f}/{float(row['middle_ms_ssim']):.4f}/{float(row['middle_lpips']):.4f} | "
            f"{float(row['original_middle_psnr']):.3f}/{float(row['original_middle_lpips']):.4f} | "
            f"{float(row['delta_psnr_vs_original']):+.3f}/{float(row['delta_lpips_vs_original']):+.4f} | "
            f"{float(row['payload_bytes']):.0f} | {float(row['direct_total_mib_per_frame_ref']):.6f} |"
        )
    lines.extend([
        "",
        "## Layout",
        "",
        "Each video frame is: left keyframe | target middle | original StreamSplat middle | Stage158 recovered middle | right keyframe.",
        "",
        "## Contract",
        "",
        "The Stage158 recovered middle panel uses original StreamSplat target-time half-anchor plus counted q6/keep1.0 entropy residual side-info and one counted half-selector byte.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--stage157_rows", type=Path, default=DEFAULT_STAGE157_ROWS)
    parser.add_argument("--stage147_rows", type=Path, default=DEFAULT_STAGE147_ROWS)
    parser.add_argument("--stage72_keyframe_rows", type=Path, default=DEFAULT_STAGE72_KEYFRAME_ROWS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--task_ids", nargs="+", default=SELECTED_TASK_IDS)
    parser.add_argument("--keep_fraction", type=float, default=1.0)
    parser.add_argument("--side_bits", type=int, default=6)
    parser.add_argument("--selector_payload_bytes", type=int, default=1)
    parser.add_argument("--zlib_level", type=int, default=9)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--disable_lpips", action="store_true")
    parser.add_argument("--disable_ms_ssim", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
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
    tasks_by_id = {row["task_id"]: row for row in parse_task_rows(args.task_manifest, "eval", ["q12"], [4])}
    tasks = [tasks_by_id[task_id] for task_id in args.task_ids]
    key_metrics = load_keyframe_metrics(args.stage72_keyframe_rows)
    dense_index = build_dense_index(args.dense_manifest, sorted({row["split"] for row in tasks}))
    expected_rows = load_stage157_expected(args.stage157_rows)
    rate_ref = load_rate_reference(args.stage147_rows)
    rows = []
    canvases = []
    for task in tasks:
        row, images = render_task(task, expected_rows.get(task["task_id"]), key_metrics, dense_index, model, opt, device, lpips_model, ms_ssim_module, rate_ref, args)
        canvas = make_canvas(images["left"], images["target"], images["original"], images["corrected"], images["right"], row)
        rows.append(row)
        canvases.append(canvas)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    video_path = args.heavy_root / "stage159_gap4_stage158_subjective_examples.mp4"
    contact_path = args.heavy_root / "stage159_gap4_stage158_subjective_examples_contact_sheet.jpg"
    h, w, _ = canvases[0].shape
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for canvas in canvases:
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()
    sheet = np.zeros((len(canvases) * h, w, 3), dtype=np.uint8)
    for idx, canvas in enumerate(canvases):
        sheet[idx * h:(idx + 1) * h] = canvas
    cv2.imwrite(str(contact_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    for row in rows:
        row["video_path"] = str(video_path)
        row["contact_sheet_path"] = str(contact_path)
    rows_csv = args.summary_root / "stage159_stage158_subjective_examples_rows.csv"
    summary_json = args.summary_root / "stage159_stage158_subjective_examples_summary.json"
    package_json = args.summary_root / "stage159_stage158_subjective_examples_package.json"
    report_md = args.summary_root / "stage159_stage158_subjective_examples_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    package = {
        "stage": 159,
        "mode": "stage158 subjective examples",
        "policy": "streamsplat_guided_half_anchor_entropy_residual_v1",
        "task_ids": args.task_ids,
        "rows": rows,
        "rows_csv": str(rows_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "video_path": str(video_path),
        "contact_sheet_path": str(contact_path),
        "video_file_bytes": video_path.stat().st_size,
        "contact_sheet_file_bytes": contact_path.stat().st_size,
        "heavy_root": str(args.heavy_root),
        "lpips_available": lpips_model is not None,
        "lpips_error": lpips_error,
        "ms_ssim_available": ms_ssim_module is not None,
        "ms_ssim_error": ms_ssim_error,
    }
    summary_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, rows, report_md)
    print(json.dumps({"video": str(video_path), "contact_sheet": str(contact_path), "rows": rows}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
