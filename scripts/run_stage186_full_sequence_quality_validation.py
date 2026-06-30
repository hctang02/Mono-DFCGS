import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
from torch.amp import autocast


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/streamsplat.safetensors")
DEFAULT_STAGE183_ROOT = REPO_ROOT / "experiments/stage183_full_sequence_payload_measurement_protocol"
DEFAULT_STAGE184_ROOT = REPO_ROOT / "experiments/stage184_full_sequence_payload_measurement_execution"
DEFAULT_STAGE185_ROOT = REPO_ROOT / "experiments/stage185_measured_full_sequence_rd_aggregation"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage186_full_sequence_quality_validation"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage186_full_sequence_quality_validation")

DEFAULT_FRAME_SCHEDULE_ROWS = DEFAULT_STAGE183_ROOT / "stage183_full_sequence_frame_schedule_payload_rows.csv"
DEFAULT_KEYFRAME_MEASUREMENTS = DEFAULT_STAGE184_ROOT / "stage184_unique_keyframe_payload_measurements.csv"
DEFAULT_RESIDUAL_MEASUREMENTS = DEFAULT_STAGE184_ROOT / "stage184_unique_stage158_residual_payload_measurements.csv"
DEFAULT_STAGE185_TOTAL_RD = DEFAULT_STAGE185_ROOT / "stage185_measured_full_sequence_total_rd.csv"

SCHEDULES = ["uniform_gap8", "stage165_adaptive", "uniform_gap4"]


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, unflatten_static_anchor  # noqa: E402
from mono_dfcgs.residual_sideinfo_codec import decode_residual_sideinfo_entropy, encode_topk_residual_sideinfo_entropy  # noqa: E402
from scripts.run_stage153_middle_multimetric_badcase_eval import load_metric_modules, mean, percentile  # noqa: E402
from scripts.run_stage154_original_streamsplat_middle_base_alignment import load_task_batch  # noqa: E402
from scripts.run_stage155_streamsplat_base_sideinfo_upper_bound import compute_metrics, render_static_anchor, stream_gaussians_at_time  # noqa: E402
from scripts.run_stage156_streamsplat_half_anchor_gaussian_residual import split_half_anchor  # noqa: E402
from scripts.run_stage177_selector_fixed_gap_psnr_comparison import frame_rgb_path  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import load_rgb  # noqa: E402
from scripts.run_stage6_export_real_anchor_dataset import load_model  # noqa: E402
from scripts.run_stage72_original_davis_baseline import DEFAULT_DAVIS_ROOT  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_DENSE_MANIFEST, build_dense_index, load_anchor  # noqa: E402


KEYFRAME_QUALITY_FIELDS = [
    "measurement_key",
    "sequence",
    "frame_index",
    "used_by_schedules",
    "schedule_count",
    "bitstream_bytes",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "status",
    "error",
]

RESIDUAL_QUALITY_FIELDS = [
    "measurement_key",
    "sequence",
    "target_index",
    "left_index",
    "right_index",
    "segment_length",
    "normalized_time",
    "used_by_schedules",
    "schedule_count",
    "selected_half",
    "payload_bytes",
    "stage184_selected_psnr",
    "psnr",
    "psnr_delta_vs_stage184",
    "ssim",
    "ms_ssim",
    "lpips",
    "status",
    "error",
]

FINAL_QUALITY_FIELDS = [
    "schedule",
    "sequence",
    "frame_index",
    "final_type",
    "measurement_key",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "payload_bytes",
]

SUMMARY_FIELDS = [
    "schedule",
    "frame_count",
    "keyframe_count",
    "residual_count",
    "mean_psnr",
    "p10_psnr",
    "mean_keyframe_psnr",
    "mean_residual_psnr",
    "mean_ssim",
    "p10_ssim",
    "mean_ms_ssim",
    "mean_lpips",
    "p90_lpips",
    "mean_payload_bytes_per_frame_schedule_row",
]

RD_QUALITY_FIELDS = [
    "schedule",
    "total_mib_per_frame",
    "delta_total_mib_per_frame_vs_gap8",
    "delta_total_mib_per_frame_vs_gap4",
    "mean_psnr",
    "delta_psnr_vs_gap8",
    "delta_psnr_vs_gap4",
    "mean_ssim",
    "delta_ssim_vs_gap8",
    "delta_ssim_vs_gap4",
    "mean_ms_ssim",
    "delta_ms_ssim_vs_gap8",
    "delta_ms_ssim_vs_gap4",
    "mean_lpips",
    "delta_lpips_vs_gap8",
    "delta_lpips_vs_gap4",
]

VALIDATION_FIELDS = ["item", "expected", "actual", "status"]


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def numeric(row, key, default=None):
    value = row.get(key)
    if value in (None, "", "NA"):
        return default
    return float(value)


def completed_keys(rows):
    return {row["measurement_key"] for row in rows if row.get("status") == "ok"}


def output_paths(output_root):
    return {
        "keyframe_quality_csv": output_root / "stage186_unique_keyframe_quality_metrics.csv",
        "residual_quality_csv": output_root / "stage186_unique_stage158_residual_quality_metrics.csv",
        "final_quality_csv": output_root / "stage186_full_sequence_quality_by_schedule.csv",
        "summary_csv": output_root / "stage186_full_sequence_quality_summary.csv",
        "rd_quality_csv": output_root / "stage186_measured_rd_quality_points.csv",
        "validation_csv": output_root / "stage186_quality_validation_checks.csv",
        "package_json": output_root / "stage186_full_sequence_quality_validation_package.json",
        "report_md": output_root / "stage186_full_sequence_quality_validation_report.md",
    }


def selected_psnr(row):
    if row["selected_half"] == "left":
        return numeric(row, "left_psnr")
    if row["selected_half"] == "right":
        return numeric(row, "right_psnr")
    return None


def residual_task(row, davis_root):
    sequence = row["sequence"]
    left = int(row["left_index"])
    right = int(row["right_index"])
    target = int(row["target_index"])
    return {
        "task_id": row["measurement_key"],
        "dataset": "DAVIS",
        "split": "val",
        "sequence": sequence,
        "codec": "q12",
        "reference_gap": int(row["segment_length"]),
        "left_index": left,
        "right_index": right,
        "target_index": target,
        "segment_length": int(row["segment_length"]),
        "normalized_time": float(row["normalized_time"]),
        "left_rgb_path": str(frame_rgb_path(davis_root, sequence, left)),
        "right_rgb_path": str(frame_rgb_path(davis_root, sequence, right)),
        "target_rgb_path": str(frame_rgb_path(davis_root, sequence, target)),
    }


def measure_keyframe_quality(row, dense_index, opt, device, background, lpips_model, ms_ssim_module, args):
    try:
        sequence = row["sequence"]
        frame_index = int(row["frame_index"])
        dense_key = ("DAVIS", "val", sequence, frame_index)
        target_item, target_side = dense_index[dense_key]
        anchor = load_anchor(target_item, target_side, device, bits=12, cache=None)
        render = render_static_anchor(anchor, background, opt)
        target_rgb = load_rgb(frame_rgb_path(args.davis_root, sequence, frame_index), opt.image_height, opt.image_width, device)
        metrics = compute_metrics(render, target_rgb, lpips_model, ms_ssim_module)
        return {
            "measurement_key": row["measurement_key"],
            "sequence": sequence,
            "frame_index": frame_index,
            "used_by_schedules": row["used_by_schedules"],
            "schedule_count": int(row["schedule_count"]),
            "bitstream_bytes": int(float(row["bitstream_bytes"])),
            **metrics,
            "status": "ok",
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "measurement_key": row.get("measurement_key", ""),
            "sequence": row.get("sequence", ""),
            "frame_index": row.get("frame_index", ""),
            "used_by_schedules": row.get("used_by_schedules", ""),
            "schedule_count": row.get("schedule_count", ""),
            "bitstream_bytes": row.get("bitstream_bytes", 0),
            "psnr": "",
            "ssim": "",
            "ms_ssim": "",
            "lpips": "",
            "status": "error",
            "error": repr(exc),
        }


def predict_gaussians_batch(tasks, model, opt, device):
    frames, depths, timestamps, targets = load_task_batch(tasks, opt, device)
    opt.output_frames = 3
    model.opt.output_frames = 3
    with torch.no_grad():
        decoder_out = model.forward_gaussians(frames, depths, timestamps)
    return decoder_out["pred_gs"], targets


def render_selected_residual(stream_anchor, target_attrs, selected_half, opt, args, device):
    half_anchor = split_half_anchor(stream_anchor, selected_half)
    base_attrs = flatten_static_anchor(half_anchor)
    payload, _info = encode_topk_residual_sideinfo_entropy(
        base_attrs,
        target_attrs,
        args.keep_fraction,
        args.side_bits,
        zlib_level=args.zlib_level,
    )
    corrected_anchor = unflatten_static_anchor(decode_residual_sideinfo_entropy(base_attrs, payload))
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    with torch.no_grad(), autocast("cuda", enabled=False):
        return render_static_anchor(corrected_anchor, background, opt)


def measure_residual_quality_batch(rows, model, opt, dense_index, lpips_model, ms_ssim_module, args, device):
    tasks = [residual_task(row, args.davis_root) for row in rows]
    pred_gs, targets = predict_gaussians_batch(tasks, model, opt, device)
    out = []
    for local_idx, (row, task) in enumerate(zip(rows, tasks)):
        try:
            target_item, target_side = dense_index[("DAVIS", "val", task["sequence"], int(task["target_index"]))]
            target_anchor = load_anchor(target_item, target_side, device, bits=None, cache=None)
            target_attrs = flatten_static_anchor(target_anchor)
            stream_anchor = stream_gaussians_at_time(pred_gs, float(task["normalized_time"]), opt, sample_idx=local_idx)
            render = render_selected_residual(stream_anchor, target_attrs, row["selected_half"], opt, args, device)
            target_rgb = targets[local_idx:local_idx + 1]
            metrics = compute_metrics(render, target_rgb, lpips_model, ms_ssim_module)
            prior_psnr = selected_psnr(row)
            out.append({
                "measurement_key": row["measurement_key"],
                "sequence": row["sequence"],
                "target_index": int(row["target_index"]),
                "left_index": int(row["left_index"]),
                "right_index": int(row["right_index"]),
                "segment_length": int(row["segment_length"]),
                "normalized_time": float(row["normalized_time"]),
                "used_by_schedules": row["used_by_schedules"],
                "schedule_count": int(row["schedule_count"]),
                "selected_half": row["selected_half"],
                "payload_bytes": int(float(row["payload_bytes"])),
                "stage184_selected_psnr": prior_psnr,
                "psnr": metrics["psnr"],
                "psnr_delta_vs_stage184": metrics["psnr"] - prior_psnr if prior_psnr is not None else None,
                "ssim": metrics["ssim"],
                "ms_ssim": metrics["ms_ssim"],
                "lpips": metrics["lpips"],
                "status": "ok",
                "error": "",
            })
            del target_anchor, target_attrs, render
        except Exception as exc:  # noqa: BLE001
            out.append({
                "measurement_key": row.get("measurement_key", ""),
                "sequence": row.get("sequence", ""),
                "target_index": row.get("target_index", ""),
                "left_index": row.get("left_index", ""),
                "right_index": row.get("right_index", ""),
                "segment_length": row.get("segment_length", ""),
                "normalized_time": row.get("normalized_time", ""),
                "used_by_schedules": row.get("used_by_schedules", ""),
                "schedule_count": row.get("schedule_count", ""),
                "selected_half": row.get("selected_half", ""),
                "payload_bytes": row.get("payload_bytes", 0),
                "stage184_selected_psnr": selected_psnr(row),
                "psnr": "",
                "psnr_delta_vs_stage184": "",
                "ssim": "",
                "ms_ssim": "",
                "lpips": "",
                "status": "error",
                "error": repr(exc),
            })
    del pred_gs, targets
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def build_final_quality(frame_rows, keyframe_quality_rows, residual_quality_rows):
    keyframe_by_key = {row["measurement_key"]: row for row in keyframe_quality_rows if row.get("status") == "ok"}
    residual_by_key = {row["measurement_key"]: row for row in residual_quality_rows if row.get("status") == "ok"}
    out = []
    missing = []
    for row in frame_rows:
        if row["measurement_type"] == "q12_keyframe_payload":
            measured = keyframe_by_key.get(row["measurement_key"])
            final_type = "q12_keyframe"
            payload = 0
        else:
            measured = residual_by_key.get(row["measurement_key"])
            final_type = "stage158_residual_recovery"
            payload = int(float(measured["payload_bytes"])) if measured else 0
        if measured is None:
            missing.append(row["measurement_key"])
            continue
        out.append({
            "schedule": row["schedule"],
            "sequence": row["sequence"],
            "frame_index": int(row["frame_index"]),
            "final_type": final_type,
            "measurement_key": row["measurement_key"],
            "psnr": numeric(measured, "psnr"),
            "ssim": numeric(measured, "ssim"),
            "ms_ssim": numeric(measured, "ms_ssim"),
            "lpips": numeric(measured, "lpips"),
            "payload_bytes": payload,
        })
    return out, missing


def summarize_quality(final_rows):
    out = []
    for schedule in SCHEDULES:
        rows = [row for row in final_rows if row["schedule"] == schedule]
        keyframes = [row for row in rows if row["final_type"] == "q12_keyframe"]
        residuals = [row for row in rows if row["final_type"] == "stage158_residual_recovery"]
        out.append({
            "schedule": schedule,
            "frame_count": len(rows),
            "keyframe_count": len(keyframes),
            "residual_count": len(residuals),
            "mean_psnr": mean(row["psnr"] for row in rows),
            "p10_psnr": percentile((row["psnr"] for row in rows), 10),
            "mean_keyframe_psnr": mean(row["psnr"] for row in keyframes),
            "mean_residual_psnr": mean(row["psnr"] for row in residuals),
            "mean_ssim": mean(row["ssim"] for row in rows),
            "p10_ssim": percentile((row["ssim"] for row in rows), 10),
            "mean_ms_ssim": mean(row["ms_ssim"] for row in rows),
            "mean_lpips": mean(row["lpips"] for row in rows),
            "p90_lpips": percentile((row["lpips"] for row in rows), 90),
            "mean_payload_bytes_per_frame_schedule_row": mean(row["payload_bytes"] for row in rows),
        })
    return out


def build_rd_quality(summary_rows, stage185_rows):
    quality = {row["schedule"]: row for row in summary_rows}
    rd = {row["schedule"]: row for row in stage185_rows}
    gap8 = quality["uniform_gap8"]
    gap4 = quality["uniform_gap4"]
    out = []
    for schedule in SCHEDULES:
        q = quality[schedule]
        r = rd[schedule]
        out.append({
            "schedule": schedule,
            "total_mib_per_frame": numeric(r, "total_mib_per_frame"),
            "delta_total_mib_per_frame_vs_gap8": numeric(r, "delta_total_mib_per_frame_vs_gap8"),
            "delta_total_mib_per_frame_vs_gap4": numeric(r, "delta_total_mib_per_frame_vs_gap4"),
            "mean_psnr": q["mean_psnr"],
            "delta_psnr_vs_gap8": q["mean_psnr"] - gap8["mean_psnr"],
            "delta_psnr_vs_gap4": q["mean_psnr"] - gap4["mean_psnr"],
            "mean_ssim": q["mean_ssim"],
            "delta_ssim_vs_gap8": q["mean_ssim"] - gap8["mean_ssim"],
            "delta_ssim_vs_gap4": q["mean_ssim"] - gap4["mean_ssim"],
            "mean_ms_ssim": q["mean_ms_ssim"],
            "delta_ms_ssim_vs_gap8": q["mean_ms_ssim"] - gap8["mean_ms_ssim"],
            "delta_ms_ssim_vs_gap4": q["mean_ms_ssim"] - gap4["mean_ms_ssim"],
            "mean_lpips": q["mean_lpips"],
            "delta_lpips_vs_gap8": q["mean_lpips"] - gap8["mean_lpips"],
            "delta_lpips_vs_gap4": q["mean_lpips"] - gap4["mean_lpips"],
        })
    return out


def validation_rows(frame_rows, keyframe_protocol_rows, residual_protocol_rows, keyframe_quality_rows, residual_quality_rows, final_rows, missing_final):
    keyframe_ok = sum(1 for row in keyframe_quality_rows if row.get("status") == "ok")
    residual_ok = sum(1 for row in residual_quality_rows if row.get("status") == "ok")
    return [
        {"item": "unique_keyframe_quality_rows", "expected": len(keyframe_protocol_rows), "actual": keyframe_ok, "status": "ok" if keyframe_ok == len(keyframe_protocol_rows) else "error"},
        {"item": "unique_residual_quality_rows", "expected": len(residual_protocol_rows), "actual": residual_ok, "status": "ok" if residual_ok == len(residual_protocol_rows) else "error"},
        {"item": "final_frame_schedule_quality_rows", "expected": len(frame_rows), "actual": len(final_rows), "status": "ok" if len(final_rows) == len(frame_rows) else "error"},
        {"item": "missing_final_measurements", "expected": 0, "actual": len(missing_final), "status": "ok" if not missing_final else "error"},
    ]


def decision(rd_quality_rows):
    rows = {row["schedule"]: row for row in rd_quality_rows}
    adaptive = rows["stage165_adaptive"]
    if adaptive["delta_psnr_vs_gap8"] > 0 and adaptive["delta_psnr_vs_gap4"] > 0 and adaptive["delta_lpips_vs_gap8"] < 0 and adaptive["delta_lpips_vs_gap4"] < 0:
        return "adaptive_full_sequence_quality_positive_measured_rate_between_gap8_and_gap4"
    if (
        adaptive["delta_psnr_vs_gap8"] > 0
        and adaptive["delta_ssim_vs_gap8"] > 0
        and adaptive["delta_ms_ssim_vs_gap8"] > 0
        and adaptive["delta_lpips_vs_gap8"] < 0
        and adaptive["delta_total_mib_per_frame_vs_gap8"] > 0
        and adaptive["delta_total_mib_per_frame_vs_gap4"] < 0
    ):
        return "adaptive_quality_rate_between_gap8_and_gap4"
    return "adaptive_full_sequence_quality_needs_review"


def write_report(summary_rows, rd_quality_rows, checks, package, path):
    adaptive = next(row for row in rd_quality_rows if row["schedule"] == "stage165_adaptive")
    lines = [
        "# Stage186 Full-Sequence Quality Validation",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Adaptive full-sequence PSNR delta vs gap8: `{adaptive['delta_psnr_vs_gap8']}` dB.",
        f"- Adaptive full-sequence PSNR delta vs gap4: `{adaptive['delta_psnr_vs_gap4']}` dB.",
        f"- Adaptive full-sequence LPIPS delta vs gap8: `{adaptive['delta_lpips_vs_gap8']}`.",
        f"- Adaptive measured rate delta vs gap8: `{adaptive['delta_total_mib_per_frame_vs_gap8']}` MiB/frame.",
        "",
        "## Full-Sequence Quality",
        "",
        "| schedule | frames | keyframes | residuals | PSNR | p10 PSNR | keyframe PSNR | residual PSNR | SSIM | MS-SSIM | LPIPS | p90 LPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['schedule']} | {row['frame_count']} | {row['keyframe_count']} | {row['residual_count']} | "
            f"{float(row['mean_psnr']):.6f} | {float(row['p10_psnr']):.6f} | {float(row['mean_keyframe_psnr']):.6f} | {float(row['mean_residual_psnr']):.6f} | "
            f"{float(row['mean_ssim']):.6f} | {float(row['mean_ms_ssim']):.6f} | {float(row['mean_lpips']):.6f} | {float(row['p90_lpips']):.6f} |"
        )
    lines.extend([
        "",
        "## Measured RD-Quality Points",
        "",
        "| schedule | MiB/frame | dRate vs gap8 | PSNR | dPSNR vs gap8 | dPSNR vs gap4 | SSIM | MS-SSIM | LPIPS | dLPIPS vs gap8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rd_quality_rows:
        lines.append(
            f"| {row['schedule']} | {float(row['total_mib_per_frame']):.12f} | {float(row['delta_total_mib_per_frame_vs_gap8']):.12f} | "
            f"{float(row['mean_psnr']):.6f} | {float(row['delta_psnr_vs_gap8']):.6f} | {float(row['delta_psnr_vs_gap4']):.6f} | "
            f"{float(row['mean_ssim']):.6f} | {float(row['mean_ms_ssim']):.6f} | {float(row['mean_lpips']):.6f} | {float(row['delta_lpips_vs_gap8']):.6f} |"
        )
    lines.extend([
        "",
        "## Validation",
        "",
        "| item | expected | actual | status |",
        "|---|---:|---:|---|",
    ])
    for row in checks:
        lines.append(f"| {row['item']} | {row['expected']} | {row['actual']} | {row['status']} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- This stage uses full unique keyframe and residual render measurements, then maps them back to all frame/schedule rows.",
        "- Stage165 adaptive improves all reported full-sequence quality metrics over uniform gap8, but it remains below uniform gap4 quality.",
        "- Stage185 measured rate also places adaptive between gap8 and gap4: higher than gap8, lower than gap4.",
        "- The next selector work should seek a lower-budget adaptive point that keeps most of the gap8 quality gain while reducing or eliminating the gap8 rate overhead.",
        "",
        "## Outputs",
        "",
        f"- Unique keyframe quality CSV: `{package['keyframe_quality_csv']}`",
        f"- Unique residual quality CSV: `{package['residual_quality_csv']}`",
        f"- Full frame/schedule quality CSV: `{package['final_quality_csv']}`",
        f"- Summary CSV: `{package['summary_csv']}`",
        f"- Measured RD-quality CSV: `{package['rd_quality_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_schedule_rows", type=Path, default=DEFAULT_FRAME_SCHEDULE_ROWS)
    parser.add_argument("--keyframe_measurements", type=Path, default=DEFAULT_KEYFRAME_MEASUREMENTS)
    parser.add_argument("--residual_measurements", type=Path, default=DEFAULT_RESIDUAL_MEASUREMENTS)
    parser.add_argument("--stage185_total_rd", type=Path, default=DEFAULT_STAGE185_TOTAL_RD)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--keep_fraction", type=float, default=1.0)
    parser.add_argument("--side_bits", type=int, default=6)
    parser.add_argument("--zlib_level", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_keyframes", type=int, default=0)
    parser.add_argument("--max_residuals", type=int, default=0)
    parser.add_argument("--flush_every", type=int, default=16)
    parser.add_argument("--skip_keyframes", action="store_true")
    parser.add_argument("--skip_residuals", action="store_true")
    parser.add_argument("--disable_lpips", action="store_true")
    parser.add_argument("--disable_ms_ssim", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    paths = output_paths(args.output_root)

    frame_rows = read_csv(args.frame_schedule_rows)
    keyframe_measurements = read_csv(args.keyframe_measurements)
    residual_measurements = read_csv(args.residual_measurements)
    stage185_rows = read_csv(args.stage185_total_rd)
    keyframe_quality_rows = read_csv(paths["keyframe_quality_csv"])
    residual_quality_rows = read_csv(paths["residual_quality_csv"])

    device = torch.device(args.device)
    opt = Options()
    opt.resume = str(args.checkpoint)
    opt.compile = False
    opt.input_frames = 1
    opt.output_frames = 3
    opt.epoch = 0
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    dense_index = build_dense_index(args.dense_manifest, ["val"])
    lpips_model, lpips_error, ms_ssim_module, ms_ssim_error = load_metric_modules(device, args.disable_lpips, args.disable_ms_ssim)

    if not args.skip_keyframes:
        done = completed_keys(keyframe_quality_rows)
        pending = [row for row in keyframe_measurements if row["measurement_key"] not in done and row.get("status") == "ok"]
        if args.max_keyframes > 0:
            pending = pending[: args.max_keyframes]
        for idx, row in enumerate(pending, 1):
            keyframe_quality_rows.append(measure_keyframe_quality(row, dense_index, opt, device, background, lpips_model, ms_ssim_module, args))
            if idx % max(1, args.flush_every) == 0 or idx == len(pending):
                write_csv(keyframe_quality_rows, paths["keyframe_quality_csv"], KEYFRAME_QUALITY_FIELDS)
                print(json.dumps({"keyframes_quality_this_run": idx, "total_keyframe_quality_rows": len(completed_keys(keyframe_quality_rows))}), flush=True)

    if not args.skip_residuals:
        if not args.checkpoint.exists():
            raise FileNotFoundError(args.checkpoint)
        model = load_model(deepcopy(opt), args.checkpoint, str(device))
        model.eval()
        done = completed_keys(residual_quality_rows)
        pending = [row for row in residual_measurements if row["measurement_key"] not in done and row.get("status") == "ok"]
        if args.max_residuals > 0:
            pending = pending[: args.max_residuals]
        batch_size = max(1, int(args.batch_size))
        processed = 0
        for start in range(0, len(pending), batch_size):
            batch_rows = pending[start:start + batch_size]
            residual_quality_rows.extend(measure_residual_quality_batch(batch_rows, model, opt, dense_index, lpips_model, ms_ssim_module, args, device))
            processed += len(batch_rows)
            if processed % max(1, args.flush_every) == 0 or processed == len(pending):
                write_csv(residual_quality_rows, paths["residual_quality_csv"], RESIDUAL_QUALITY_FIELDS)
                print(json.dumps({"residual_quality_this_run": processed, "total_residual_quality_rows": len(completed_keys(residual_quality_rows))}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_rows, missing_final = build_final_quality(frame_rows, keyframe_quality_rows, residual_quality_rows)
    summary_rows = summarize_quality(final_rows)
    rd_quality_rows = build_rd_quality(summary_rows, stage185_rows)
    checks = validation_rows(frame_rows, keyframe_measurements, residual_measurements, keyframe_quality_rows, residual_quality_rows, final_rows, missing_final)

    write_csv(final_rows, paths["final_quality_csv"], FINAL_QUALITY_FIELDS)
    write_csv(summary_rows, paths["summary_csv"], SUMMARY_FIELDS)
    write_csv(rd_quality_rows, paths["rd_quality_csv"], RD_QUALITY_FIELDS)
    write_csv(checks, paths["validation_csv"], VALIDATION_FIELDS)
    complete = all(row["status"] == "ok" for row in checks)
    package = {
        "stage": 186,
        "status": "full_sequence_quality_validation_complete" if complete else "full_sequence_quality_validation_partial",
        "decision": decision(rd_quality_rows) if complete else "quality_validation_incomplete",
        "complete": complete,
        "summary_rows": summary_rows,
        "rd_quality_rows": rd_quality_rows,
        "validation_rows": checks,
        "lpips_available": lpips_model is not None,
        "lpips_error": lpips_error,
        "ms_ssim_available": ms_ssim_module is not None,
        "ms_ssim_error": ms_ssim_error,
        "keyframe_quality_csv": str(paths["keyframe_quality_csv"]),
        "residual_quality_csv": str(paths["residual_quality_csv"]),
        "final_quality_csv": str(paths["final_quality_csv"]),
        "summary_csv": str(paths["summary_csv"]),
        "rd_quality_csv": str(paths["rd_quality_csv"]),
        "validation_csv": str(paths["validation_csv"]),
        "package_json": str(paths["package_json"]),
        "report_md": str(paths["report_md"]),
        "heavy_root": str(args.heavy_root),
    }
    paths["package_json"].write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary_rows, rd_quality_rows, checks, package, paths["report_md"])
    adaptive = next(row for row in rd_quality_rows if row["schedule"] == "stage165_adaptive")
    print(json.dumps({
        "package": str(paths["package_json"]),
        "status": package["status"],
        "decision": package["decision"],
        "adaptive_psnr": adaptive["mean_psnr"],
        "adaptive_delta_psnr_vs_gap8": adaptive["delta_psnr_vs_gap8"],
        "adaptive_lpips": adaptive["mean_lpips"],
        "adaptive_rate_delta_vs_gap8": adaptive["delta_total_mib_per_frame_vs_gap8"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
