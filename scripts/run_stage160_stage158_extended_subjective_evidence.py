import argparse
import csv
import json
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
DEFAULT_STAGE157_ROWS = REPO_ROOT / "experiments/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_rows.csv"
DEFAULT_STAGE147_ROWS = REPO_ROOT / "experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_rows.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage160_stage158_extended_subjective_evidence"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage160_stage158_extended_subjective_evidence")
DEFAULT_SEQUENCES = [
    "cows",
    "breakdance",
    "camel",
    "bike-packing",
    "scooter-black",
    "dance-twirl",
    "motocross-jump",
    "soapbox",
    "car-shadow",
    "goat",
    "gold-fish",
    "kite-surf",
]


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, unflatten_static_anchor  # noqa: E402
from mono_dfcgs.residual_sideinfo_codec import decode_residual_sideinfo_entropy, encode_topk_residual_sideinfo_entropy  # noqa: E402
from scripts.run_stage153_middle_multimetric_badcase_eval import format_optional, load_metric_modules, mean, tensor_to_rgb8  # noqa: E402
from scripts.run_stage155_streamsplat_base_sideinfo_upper_bound import compute_metrics, load_rate_reference, render_batch_with_gs, render_static_anchor, stream_gaussians_at_time  # noqa: E402
from scripts.run_stage156_streamsplat_half_anchor_gaussian_residual import split_half_anchor  # noqa: E402
from scripts.run_stage157_selected_half_anchor_broader_validation import side_mib  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import load_rgb  # noqa: E402
from scripts.run_stage6_export_real_anchor_dataset import load_model  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_DENSE_MANIFEST, DEFAULT_TASK_MANIFEST, build_dense_index, load_anchor, parse_task_rows  # noqa: E402


ROW_FIELDS = [
    "frame_number", "sequence", "task_id", "gap", "left_index", "target_index", "right_index", "normalized_time",
    "key_left_psnr", "key_left_ssim", "key_left_ms_ssim", "key_left_lpips",
    "key_right_psnr", "key_right_ssim", "key_right_ms_ssim", "key_right_lpips",
    "key_avg_psnr", "key_avg_ssim", "key_avg_ms_ssim", "key_avg_lpips",
    "selected_half", "middle_psnr", "middle_ssim", "middle_ms_ssim", "middle_lpips",
    "original_middle_psnr", "original_middle_ssim", "original_middle_ms_ssim", "original_middle_lpips",
    "delta_psnr_vs_original", "delta_ssim_vs_original", "delta_ms_ssim_vs_original", "delta_lpips_vs_original",
    "payload_bytes", "selector_payload_bytes", "side_mib_per_intermediate", "q12_main_anchor_mib_per_frame_ref", "direct_total_mib_per_frame_ref",
    "stage157_expected_psnr", "stage157_expected_lpips", "stage157_metric_abs_diff_psnr", "stage157_metric_abs_diff_lpips",
    "video_path", "contact_sheet_path",
]

SUMMARY_FIELDS = [
    "sequence", "task_count", "mean_key_avg_psnr", "mean_key_avg_ssim", "mean_key_avg_ms_ssim", "mean_key_avg_lpips",
    "mean_middle_psnr", "mean_middle_ssim", "mean_middle_ms_ssim", "mean_middle_lpips",
    "mean_original_middle_psnr", "mean_original_middle_lpips", "mean_delta_psnr_vs_original", "mean_delta_lpips_vs_original",
    "mean_payload_bytes", "mean_side_mib_per_intermediate", "mean_direct_total_mib_per_frame_ref",
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


def avg_metric(left, right, key):
    if left.get(key) is None or right.get(key) is None:
        return None
    return (float(left[key]) + float(right[key])) / 2.0


def select_stage157_rows(stage157_rows, sequences, gap, examples_per_sequence):
    seq_set = set(sequences)
    groups = defaultdict(list)
    for row in stage157_rows:
        if row["sequence"] not in seq_set:
            continue
        if int(row["gap"]) != int(gap):
            continue
        groups[row["sequence"]].append(row)
    selected = []
    for sequence in sequences:
        rows = sorted(groups.get(sequence, []), key=lambda row: (float(row["normalized_time"]), int(row["target_index"])))
        selected.extend(rows[:examples_per_sequence])
    return selected


def put_label(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def make_canvas(left_render, target_rgb, original_rgb, corrected_rgb, right_render, row):
    panels = [left_render, target_rgb, original_rgb, corrected_rgb, right_render]
    h, w, _ = target_rgb.shape
    header = 78
    canvas = np.zeros((h + header, w * len(panels), 3), dtype=np.uint8)
    for idx, panel in enumerate(panels):
        canvas[header:, idx * w:(idx + 1) * w] = panel
    title = f"gap{row['gap']} {row['sequence']} frames {row['left_index']}-{row['target_index']}-{row['right_index']} t={float(row['normalized_time']):.3f}"
    put_label(canvas, title, 8, 18)
    labels = [
        f"left q12 P {float(row['key_left_psnr']):.2f} L {format_optional(row['key_left_lpips'])}",
        "target middle",
        f"orig SS P {float(row['original_middle_psnr']):.2f} L {format_optional(row['original_middle_lpips'])}",
        f"Stage158 P {float(row['middle_psnr']):.2f} L {format_optional(row['middle_lpips'])}",
        f"right q12 P {float(row['key_right_psnr']):.2f} L {format_optional(row['key_right_lpips'])}",
    ]
    for idx, label in enumerate(labels):
        put_label(canvas, label, idx * w + 8, 47)
    put_label(canvas, f"payload {float(row['payload_bytes']):.0f} B, half {row['selected_half']}, direct rate {float(row['direct_total_mib_per_frame_ref']):.3f} MiB/fr", 8, 69)
    return canvas


def render_one(task, expected_row, dense_index, model, opt, background, device, lpips_model, ms_ssim_module, rate_ref, args, anchor_cache):
    pred_gs, original_render, target = render_batch_with_gs([task], model, opt, device)
    original_i = original_render[0:1]
    target_i = target[0:1]
    original_metrics = compute_metrics(original_i, target_i, lpips_model, ms_ssim_module)

    left_anchor = load_anchor(task["left_anchor_source_item"], task["left_anchor_source_side"], device, bits=task["bits"], cache=anchor_cache)
    right_anchor = load_anchor(task["right_anchor_source_item"], task["right_anchor_source_side"], device, bits=task["bits"], cache=anchor_cache)
    left_render = render_static_anchor(left_anchor, background, opt)
    right_render = render_static_anchor(right_anchor, background, opt)
    left_rgb = load_rgb(task["left_rgb_path"], opt.image_height, opt.image_width, device)
    right_rgb = load_rgb(task["right_rgb_path"], opt.image_height, opt.image_width, device)
    left_metrics = compute_metrics(left_render, left_rgb, lpips_model, ms_ssim_module)
    right_metrics = compute_metrics(right_render, right_rgb, lpips_model, ms_ssim_module)

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
        corrected_render = render_static_anchor(corrected_anchor, background, opt)
        corrected_metrics = compute_metrics(corrected_render, target_i, lpips_model, ms_ssim_module)
        candidates.append((half, corrected_render, corrected_metrics, int(info["payload_bytes"])))
    selected_half, corrected_render, corrected_metrics, payload_bytes = sorted(candidates, key=lambda item: float(item[2]["psnr"]), reverse=True)[0]
    payload_bytes += int(args.selector_payload_bytes)
    gap = int(task["reference_gap"])
    q12_ref = float(rate_ref.get(gap, 0.0))
    expected_psnr = float(expected_row["psnr"]) if expected_row is not None else None
    expected_lpips = float(expected_row["lpips"]) if expected_row is not None and expected_row.get("lpips") else None
    row = {
        "sequence": task["sequence"],
        "task_id": task["task_id"],
        "gap": gap,
        "left_index": int(task["left_index"]),
        "target_index": int(task["target_index"]),
        "right_index": int(task["right_index"]),
        "normalized_time": float(task["normalized_time"]),
        "key_left_psnr": left_metrics["psnr"],
        "key_left_ssim": left_metrics["ssim"],
        "key_left_ms_ssim": left_metrics["ms_ssim"],
        "key_left_lpips": left_metrics["lpips"],
        "key_right_psnr": right_metrics["psnr"],
        "key_right_ssim": right_metrics["ssim"],
        "key_right_ms_ssim": right_metrics["ms_ssim"],
        "key_right_lpips": right_metrics["lpips"],
        "key_avg_psnr": avg_metric(left_metrics, right_metrics, "psnr"),
        "key_avg_ssim": avg_metric(left_metrics, right_metrics, "ssim"),
        "key_avg_ms_ssim": avg_metric(left_metrics, right_metrics, "ms_ssim"),
        "key_avg_lpips": avg_metric(left_metrics, right_metrics, "lpips"),
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
        "q12_main_anchor_mib_per_frame_ref": q12_ref,
        "direct_total_mib_per_frame_ref": q12_ref + side_mib(payload_bytes),
        "stage157_expected_psnr": expected_psnr,
        "stage157_expected_lpips": expected_lpips,
        "stage157_metric_abs_diff_psnr": abs(float(corrected_metrics["psnr"]) - expected_psnr) if expected_psnr is not None else None,
        "stage157_metric_abs_diff_lpips": abs(float(corrected_metrics["lpips"]) - expected_lpips) if expected_lpips is not None and corrected_metrics["lpips"] is not None else None,
    }
    images = {
        "left_render": tensor_to_rgb8(left_render.detach().cpu()),
        "target": tensor_to_rgb8(target_i.detach().cpu()),
        "original": tensor_to_rgb8(original_i.detach().cpu()),
        "corrected": tensor_to_rgb8(corrected_render.detach().cpu()),
        "right_render": tensor_to_rgb8(right_render.detach().cpu()),
    }
    del pred_gs, original_render, target
    return row, images


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["sequence"]].append(row)
    summary = []
    for sequence, group in sorted(groups.items()):
        summary.append({
            "sequence": sequence,
            "task_count": len(group),
            "mean_key_avg_psnr": mean(row["key_avg_psnr"] for row in group),
            "mean_key_avg_ssim": mean(row["key_avg_ssim"] for row in group),
            "mean_key_avg_ms_ssim": mean(row["key_avg_ms_ssim"] for row in group),
            "mean_key_avg_lpips": mean(row["key_avg_lpips"] for row in group),
            "mean_middle_psnr": mean(row["middle_psnr"] for row in group),
            "mean_middle_ssim": mean(row["middle_ssim"] for row in group),
            "mean_middle_ms_ssim": mean(row["middle_ms_ssim"] for row in group),
            "mean_middle_lpips": mean(row["middle_lpips"] for row in group),
            "mean_original_middle_psnr": mean(row["original_middle_psnr"] for row in group),
            "mean_original_middle_lpips": mean(row["original_middle_lpips"] for row in group),
            "mean_delta_psnr_vs_original": mean(row["delta_psnr_vs_original"] for row in group),
            "mean_delta_lpips_vs_original": mean(row["delta_lpips_vs_original"] for row in group),
            "mean_payload_bytes": mean(row["payload_bytes"] for row in group),
            "mean_side_mib_per_intermediate": mean(row["side_mib_per_intermediate"] for row in group),
            "mean_direct_total_mib_per_frame_ref": mean(row["direct_total_mib_per_frame_ref"] for row in group),
        })
    return summary


def write_report(package, rows, summary_rows, path):
    lines = [
        "# Stage160 Stage158 Extended Subjective Evidence",
        "",
        "## Heavy Outputs",
        "",
        f"- Video: `{package['video_path']}`",
        f"- Contact sheet: `{package['contact_sheet_path']}`",
        f"- Video file size: `{package['video_file_bytes']}` bytes",
        f"- Contact sheet file size: `{package['contact_sheet_file_bytes']}` bytes",
        "",
        "## Layout",
        "",
        "Each frame is: left q12 keyframe render | target middle RGB | original StreamSplat middle | Stage158 recovered middle | right q12 keyframe render.",
        "",
        "## Sequence Summary",
        "",
        "| sequence | tasks | key avg PSNR/LPIPS | middle PSNR/LPIPS | original PSNR/LPIPS | delta PSNR/LPIPS | payload bytes | direct rate ref |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['sequence']} | {row['task_count']} | "
            f"{float(row['mean_key_avg_psnr']):.3f}/{format_optional(row['mean_key_avg_lpips'])} | "
            f"{float(row['mean_middle_psnr']):.3f}/{format_optional(row['mean_middle_lpips'])} | "
            f"{float(row['mean_original_middle_psnr']):.3f}/{format_optional(row['mean_original_middle_lpips'])} | "
            f"{float(row['mean_delta_psnr_vs_original']):+.3f}/{format_optional(row['mean_delta_lpips_vs_original'])} | "
            f"{float(row['mean_payload_bytes']):.0f} | {float(row['mean_direct_total_mib_per_frame_ref']):.6f} |"
        )
    lines.extend([
        "",
        "## Per-Frame Rows",
        "",
        "| # | sequence | frames | t | key avg PSNR/SSIM/MS-SSIM/LPIPS | middle PSNR/SSIM/MS-SSIM/LPIPS | original PSNR/LPIPS | delta PSNR/LPIPS | payload bytes |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['frame_number']} | {row['sequence']} | {row['left_index']}-{row['target_index']}-{row['right_index']} | {float(row['normalized_time']):.3f} | "
            f"{float(row['key_avg_psnr']):.3f}/{float(row['key_avg_ssim']):.4f}/{format_optional(row['key_avg_ms_ssim'])}/{format_optional(row['key_avg_lpips'])} | "
            f"{float(row['middle_psnr']):.3f}/{float(row['middle_ssim']):.4f}/{format_optional(row['middle_ms_ssim'])}/{format_optional(row['middle_lpips'])} | "
            f"{float(row['original_middle_psnr']):.3f}/{format_optional(row['original_middle_lpips'])} | "
            f"{float(row['delta_psnr_vs_original']):+.3f}/{format_optional(row['delta_lpips_vs_original'])} | {float(row['payload_bytes']):.0f} |"
        )
    lines.extend([
        "",
        "## Contract",
        "",
        "Stage158 recovered middle uses original StreamSplat target-time half-anchor plus counted q6/keep1.0 entropy residual side-info and one counted half-selector byte.",
        "Target dense anchor is used encoder-side only for residual construction and is forbidden as decoder input.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--stage157_rows", type=Path, default=DEFAULT_STAGE157_ROWS)
    parser.add_argument("--stage147_rows", type=Path, default=DEFAULT_STAGE147_ROWS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--examples_per_sequence", type=int, default=2)
    parser.add_argument("--keep_fraction", type=float, default=1.0)
    parser.add_argument("--side_bits", type=int, default=6)
    parser.add_argument("--selector_payload_bytes", type=int, default=1)
    parser.add_argument("--zlib_level", type=int, default=9)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--contact_columns", type=int, default=2)
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
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    lpips_model, lpips_error, ms_ssim_module, ms_ssim_error = load_metric_modules(device, args.disable_lpips, args.disable_ms_ssim)
    stage157_rows = read_csv(args.stage157_rows)
    selected_rows = select_stage157_rows(stage157_rows, args.sequences, args.gap, args.examples_per_sequence)
    if not selected_rows:
        raise RuntimeError("No Stage157 rows selected")
    task_ids = [row["task_id"] for row in selected_rows]
    expected_by_id = {row["task_id"]: row for row in selected_rows}
    all_tasks = parse_task_rows(args.task_manifest, "eval", ["q12"], [args.gap])
    tasks_by_id = {row["task_id"]: row for row in all_tasks}
    tasks = [tasks_by_id[task_id] for task_id in task_ids]
    dense_index = build_dense_index(args.dense_manifest, sorted({row["split"] for row in tasks}))
    rate_ref = load_rate_reference(args.stage147_rows)
    anchor_cache = {}
    rows = []
    canvases = []
    video_path = args.heavy_root / "stage160_gap4_stage158_extended_subjective_evidence.mp4"
    contact_path = args.heavy_root / "stage160_gap4_stage158_extended_subjective_evidence_contact_sheet.jpg"
    for frame_number, task in enumerate(tasks, start=1):
        row, images = render_one(task, expected_by_id.get(task["task_id"]), dense_index, model, opt, background, device, lpips_model, ms_ssim_module, rate_ref, args, anchor_cache)
        row["frame_number"] = frame_number
        row["video_path"] = str(video_path)
        row["contact_sheet_path"] = str(contact_path)
        rows.append(row)
        canvases.append(make_canvas(images["left_render"], images["target"], images["original"], images["corrected"], images["right_render"], row))
        print(json.dumps({"processed": frame_number, "total": len(tasks), "sequence": task["sequence"], "target_index": task["target_index"]}), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    h, w, _ = canvases[0].shape
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for canvas in canvases:
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()
    columns = min(int(args.contact_columns), len(canvases))
    sheet_rows = int(np.ceil(len(canvases) / columns))
    sheet = np.zeros((sheet_rows * h, columns * w, 3), dtype=np.uint8)
    for idx, canvas in enumerate(canvases):
        row_idx = idx // columns
        col_idx = idx % columns
        sheet[row_idx * h:(row_idx + 1) * h, col_idx * w:(col_idx + 1) * w] = canvas
    cv2.imwrite(str(contact_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    summary_rows = summarize(rows)
    rows_csv = args.summary_root / "stage160_stage158_extended_subjective_evidence_rows.csv"
    summary_csv = args.summary_root / "stage160_stage158_extended_subjective_evidence_summary.csv"
    summary_json = args.summary_root / "stage160_stage158_extended_subjective_evidence_summary.json"
    package_json = args.summary_root / "stage160_stage158_extended_subjective_evidence_package.json"
    report_md = args.summary_root / "stage160_stage158_extended_subjective_evidence_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    package = {
        "stage": 160,
        "mode": "stage158 extended subjective evidence",
        "policy": "streamsplat_guided_half_anchor_entropy_residual_v1",
        "selected_sequences": args.sequences,
        "task_ids": task_ids,
        "frame_count": len(rows),
        "rows": rows,
        "summary_rows": summary_rows,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
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
    write_report(package, rows, summary_rows, report_md)
    print(json.dumps({"video": str(video_path), "contact_sheet": str(contact_path), "frame_count": len(rows), "summary": summary_rows}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
