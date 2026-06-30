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
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage184_full_sequence_payload_measurement_execution"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage184_full_sequence_payload_measurement_execution")

DEFAULT_UNIQUE_KEYFRAME_ROWS = DEFAULT_STAGE183_ROOT / "stage183_unique_keyframe_payload_measurement_rows.csv"
DEFAULT_UNIQUE_RESIDUAL_ROWS = DEFAULT_STAGE183_ROOT / "stage183_unique_stage158_residual_payload_measurement_rows.csv"
DEFAULT_FRAME_SCHEDULE_ROWS = DEFAULT_STAGE183_ROOT / "stage183_full_sequence_frame_schedule_payload_rows.csv"

SCHEDULES = ["uniform_gap8", "stage165_adaptive", "uniform_gap4"]
MIB = 1024.0 * 1024.0


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.anchor_bitstream import encode_anchor_bitstream  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, unflatten_static_anchor  # noqa: E402
from mono_dfcgs.residual_sideinfo_codec import decode_residual_sideinfo_entropy, encode_topk_residual_sideinfo_entropy  # noqa: E402
from scripts.run_stage153_middle_multimetric_badcase_eval import psnr_metric  # noqa: E402
from scripts.run_stage154_original_streamsplat_middle_base_alignment import load_task_batch  # noqa: E402
from scripts.run_stage155_streamsplat_base_sideinfo_upper_bound import render_static_anchor, stream_gaussians_at_time  # noqa: E402
from scripts.run_stage156_streamsplat_half_anchor_gaussian_residual import split_half_anchor  # noqa: E402
from scripts.run_stage6_export_real_anchor_dataset import load_model  # noqa: E402
from scripts.run_stage72_original_davis_baseline import DEFAULT_DAVIS_ROOT  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_DENSE_MANIFEST, build_dense_index, load_anchor  # noqa: E402


KEYFRAME_FIELDS = [
    "measurement_key",
    "sequence",
    "frame_index",
    "used_by_schedules",
    "schedule_count",
    "bits",
    "compression",
    "payload_encoding",
    "anchor_values",
    "bitstream_bytes",
    "bitstream_mib",
    "status",
    "error",
]

RESIDUAL_FIELDS = [
    "measurement_key",
    "sequence",
    "target_index",
    "left_index",
    "right_index",
    "segment_length",
    "normalized_time",
    "used_by_schedules",
    "schedule_count",
    "keep_fraction",
    "side_bits",
    "selector_mode",
    "selector_payload_bytes",
    "left_payload_bytes",
    "right_payload_bytes",
    "left_psnr",
    "right_psnr",
    "selected_half",
    "selected_residual_payload_bytes",
    "payload_bytes",
    "payload_mib",
    "status",
    "error",
]

SCHEDULE_KEYFRAME_FIELDS = [
    "schedule",
    "sequence",
    "keyframe_count",
    "first_frame_index",
    "last_frame_index",
    "bits",
    "compression",
    "payload_encoding",
    "bitstream_bytes",
    "bitstream_mib",
    "status",
    "error",
]

SUMMARY_FIELDS = [
    "item",
    "expected_count",
    "measured_count",
    "complete",
    "total_bytes",
    "total_mib",
    "mean_bytes",
]


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


def numeric(row, key, default=0.0):
    value = row.get(key)
    if value in (None, "", "NA"):
        return default
    return float(value)


def frame_rgb_path(davis_root, sequence, index):
    return davis_root / "JPEGImages" / "Full-Resolution" / sequence / f"{int(index):05d}.jpg"


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


def output_paths(summary_root):
    return {
        "keyframe_csv": summary_root / "stage184_unique_keyframe_payload_measurements.csv",
        "residual_csv": summary_root / "stage184_unique_stage158_residual_payload_measurements.csv",
        "schedule_keyframe_csv": summary_root / "stage184_schedule_packed_keyframe_payload_measurements.csv",
        "summary_csv": summary_root / "stage184_payload_measurement_summary.csv",
        "package_json": summary_root / "stage184_full_sequence_payload_measurement_execution_package.json",
        "report_md": summary_root / "stage184_full_sequence_payload_measurement_execution_report.md",
    }


def completed_keys(rows, key_name="measurement_key"):
    return {row[key_name] for row in rows if row.get("status") == "ok"}


def completed_schedule_keys(rows):
    return {(row["schedule"], row["sequence"]) for row in rows if row.get("status") == "ok"}


def measure_keyframe_row(row, dense_index, args, device):
    try:
        sequence = row["sequence"]
        frame_index = int(row["frame_index"])
        dense_key = ("DAVIS", "val", sequence, frame_index)
        target_item, target_side = dense_index[dense_key]
        anchor = load_anchor(target_item, target_side, device, bits=None, cache=None)
        attrs = flatten_static_anchor(anchor)
        blob = encode_anchor_bitstream(
            [anchor],
            [frame_index],
            timestamps=[frame_index],
            bits=args.keyframe_bits,
            compression=args.keyframe_compression,
            payload_encoding=args.keyframe_payload_encoding,
        )
        bitstream_bytes = len(blob)
        return {
            "measurement_key": row["measurement_key"],
            "sequence": sequence,
            "frame_index": frame_index,
            "used_by_schedules": row["used_by_schedules"],
            "schedule_count": int(row["schedule_count"]),
            "bits": int(args.keyframe_bits),
            "compression": args.keyframe_compression,
            "payload_encoding": args.keyframe_payload_encoding,
            "anchor_values": int(attrs.numel()),
            "bitstream_bytes": bitstream_bytes,
            "bitstream_mib": bitstream_bytes / MIB,
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
            "bits": int(args.keyframe_bits),
            "compression": args.keyframe_compression,
            "payload_encoding": args.keyframe_payload_encoding,
            "anchor_values": 0,
            "bitstream_bytes": 0,
            "bitstream_mib": 0.0,
            "status": "error",
            "error": repr(exc),
        }


def group_schedule_keyframes(frame_rows):
    groups = {}
    for row in frame_rows:
        if row["measurement_type"] != "q12_keyframe_payload":
            continue
        key = (row["schedule"], row["sequence"])
        groups.setdefault(key, set()).add(int(row["frame_index"]))
    return {key: sorted(values) for key, values in groups.items()}


def measure_schedule_keyframe_group(schedule, sequence, frame_indices, dense_index, args, device):
    try:
        anchors = []
        for frame_index in frame_indices:
            dense_key = ("DAVIS", "val", sequence, int(frame_index))
            target_item, target_side = dense_index[dense_key]
            anchors.append(load_anchor(target_item, target_side, device, bits=None, cache=None))
        blob = encode_anchor_bitstream(
            anchors,
            frame_indices,
            timestamps=frame_indices,
            bits=args.keyframe_bits,
            compression=args.keyframe_compression,
            payload_encoding=args.keyframe_payload_encoding,
        )
        bitstream_bytes = len(blob)
        return {
            "schedule": schedule,
            "sequence": sequence,
            "keyframe_count": len(frame_indices),
            "first_frame_index": frame_indices[0] if frame_indices else "",
            "last_frame_index": frame_indices[-1] if frame_indices else "",
            "bits": int(args.keyframe_bits),
            "compression": args.keyframe_compression,
            "payload_encoding": args.keyframe_payload_encoding,
            "bitstream_bytes": bitstream_bytes,
            "bitstream_mib": bitstream_bytes / MIB,
            "status": "ok",
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "schedule": schedule,
            "sequence": sequence,
            "keyframe_count": len(frame_indices),
            "first_frame_index": frame_indices[0] if frame_indices else "",
            "last_frame_index": frame_indices[-1] if frame_indices else "",
            "bits": int(args.keyframe_bits),
            "compression": args.keyframe_compression,
            "payload_encoding": args.keyframe_payload_encoding,
            "bitstream_bytes": 0,
            "bitstream_mib": 0.0,
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


def residual_candidate_payload_and_psnr(stream_anchor, target_attrs, target_rgb, half, opt, args, device):
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
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    with torch.no_grad(), autocast("cuda", enabled=False):
        corrected_render = render_static_anchor(corrected_anchor, background, opt)
    psnr = psnr_metric(corrected_render, target_rgb)
    return int(info["payload_bytes"]), float(psnr)


def measure_residual_batch(rows, model, opt, dense_index, args, device):
    tasks = [residual_task(row, args.davis_root) for row in rows]
    pred_gs, targets = predict_gaussians_batch(tasks, model, opt, device)
    out = []
    for local_idx, (row, task) in enumerate(zip(rows, tasks)):
        try:
            stream_anchor = stream_gaussians_at_time(pred_gs, float(task["normalized_time"]), opt, sample_idx=local_idx)
            dense_key = ("DAVIS", "val", task["sequence"], int(task["target_index"]))
            target_item, target_side = dense_index[dense_key]
            target_anchor = load_anchor(target_item, target_side, device, bits=None, cache=None)
            target_attrs = flatten_static_anchor(target_anchor)
            target_rgb = targets[local_idx:local_idx + 1]
            left_payload, left_psnr = residual_candidate_payload_and_psnr(stream_anchor, target_attrs, target_rgb, "left", opt, args, device)
            right_payload, right_psnr = residual_candidate_payload_and_psnr(stream_anchor, target_attrs, target_rgb, "right", opt, args, device)
            if left_psnr >= right_psnr:
                selected_half = "left"
                selected_payload = left_payload
            else:
                selected_half = "right"
                selected_payload = right_payload
            total_payload = int(selected_payload) + int(args.selector_payload_bytes)
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
                "keep_fraction": float(args.keep_fraction),
                "side_bits": int(args.side_bits),
                "selector_mode": "psnr_best_half",
                "selector_payload_bytes": int(args.selector_payload_bytes),
                "left_payload_bytes": left_payload,
                "right_payload_bytes": right_payload,
                "left_psnr": left_psnr,
                "right_psnr": right_psnr,
                "selected_half": selected_half,
                "selected_residual_payload_bytes": selected_payload,
                "payload_bytes": total_payload,
                "payload_mib": total_payload / MIB,
                "status": "ok",
                "error": "",
            })
            del target_anchor, target_attrs
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
                "keep_fraction": float(args.keep_fraction),
                "side_bits": int(args.side_bits),
                "selector_mode": "psnr_best_half",
                "selector_payload_bytes": int(args.selector_payload_bytes),
                "left_payload_bytes": 0,
                "right_payload_bytes": 0,
                "left_psnr": "",
                "right_psnr": "",
                "selected_half": "",
                "selected_residual_payload_bytes": 0,
                "payload_bytes": 0,
                "payload_mib": 0.0,
                "status": "error",
                "error": repr(exc),
            })
    del pred_gs, targets
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def measured_total(rows, byte_key):
    ok = [row for row in rows if row.get("status") == "ok"]
    total = sum(numeric(row, byte_key) for row in ok)
    return len(ok), total


def build_summary(expected_keyframes, expected_residuals, expected_schedule_groups, keyframe_rows, residual_rows, schedule_keyframe_rows):
    measured_keyframes, keyframe_bytes = measured_total(keyframe_rows, "bitstream_bytes")
    measured_residuals, residual_bytes = measured_total(residual_rows, "payload_bytes")
    measured_schedule, schedule_keyframe_bytes = measured_total(schedule_keyframe_rows, "bitstream_bytes")
    items = [
        ("unique_keyframe_single_anchor_bitstreams", expected_keyframes, measured_keyframes, keyframe_bytes),
        ("unique_stage158_residual_payloads", expected_residuals, measured_residuals, residual_bytes),
        ("schedule_sequence_packed_keyframe_bitstreams", expected_schedule_groups, measured_schedule, schedule_keyframe_bytes),
    ]
    return [
        {
            "item": item,
            "expected_count": expected,
            "measured_count": measured,
            "complete": int(expected == measured),
            "total_bytes": total_bytes,
            "total_mib": total_bytes / MIB,
            "mean_bytes": total_bytes / float(measured) if measured else 0.0,
        }
        for item, expected, measured, total_bytes in items
    ]


def write_report(summary_rows, package, path):
    lines = [
        "# Stage184 Full-Sequence Payload Measurement Execution",
        "",
        "## Scope",
        "",
        "This stage measures actual q12 keyframe bitstream bytes and Stage158 q6/keep1.0 residual payload bytes from the Stage183 protocol.",
        "",
        "## Summary",
        "",
        "| item | expected | measured | complete | total MiB | mean bytes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['item']} | {row['expected_count']} | {row['measured_count']} | {row['complete']} | "
            f"{float(row['total_mib']):.6f} | {float(row['mean_bytes']):.3f} |"
        )
    lines.extend([
        "",
        "## Policy Contract",
        "",
        "- Keyframes are encoded as q12 Gaussian anchor bitstreams using the Mono-DFCGS anchor container.",
        "- Residual rows use Stage158 `best_half_selector` with PSNR-based half selection, q6 residual entropy payload, keep fraction 1.0, and one counted selector byte.",
        "- Target dense anchors and target RGB are encoder-side measurement inputs only; decoder-side inputs remain the transmitted schedule, keyframe bitstreams, normalized time, residual payload, and half selector byte.",
        "",
        "## Outputs",
        "",
        f"- Unique keyframe payload CSV: `{package['keyframe_csv']}`",
        f"- Unique residual payload CSV: `{package['residual_csv']}`",
        f"- Schedule-packed keyframe payload CSV: `{package['schedule_keyframe_csv']}`",
        f"- Summary CSV: `{package['summary_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unique_keyframe_rows", type=Path, default=DEFAULT_UNIQUE_KEYFRAME_ROWS)
    parser.add_argument("--unique_residual_rows", type=Path, default=DEFAULT_UNIQUE_RESIDUAL_ROWS)
    parser.add_argument("--frame_schedule_rows", type=Path, default=DEFAULT_FRAME_SCHEDULE_ROWS)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--keyframe_bits", type=int, default=12)
    parser.add_argument("--keyframe_compression", choices=["none", "zlib"], default="none")
    parser.add_argument("--keyframe_payload_encoding", choices=["bitpack", "dtype"], default="bitpack")
    parser.add_argument("--keep_fraction", type=float, default=1.0)
    parser.add_argument("--side_bits", type=int, default=6)
    parser.add_argument("--selector_payload_bytes", type=int, default=1)
    parser.add_argument("--zlib_level", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_keyframes", type=int, default=0)
    parser.add_argument("--max_residuals", type=int, default=0)
    parser.add_argument("--max_schedule_keyframe_groups", type=int, default=0)
    parser.add_argument("--flush_every", type=int, default=5)
    parser.add_argument("--skip_keyframes", action="store_true")
    parser.add_argument("--skip_residuals", action="store_true")
    parser.add_argument("--skip_schedule_keyframes", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
    args.summary_root.mkdir(parents=True, exist_ok=True)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    paths = output_paths(args.summary_root)

    keyframe_protocol_rows = read_csv(args.unique_keyframe_rows)
    residual_protocol_rows = read_csv(args.unique_residual_rows)
    frame_schedule_rows = read_csv(args.frame_schedule_rows)
    dense_index = build_dense_index(args.dense_manifest, ["val"])
    cpu_device = torch.device("cpu")

    keyframe_rows = read_csv(paths["keyframe_csv"])
    residual_rows = read_csv(paths["residual_csv"])
    schedule_keyframe_rows = read_csv(paths["schedule_keyframe_csv"])

    if not args.skip_keyframes:
        done = completed_keys(keyframe_rows)
        pending = [row for row in keyframe_protocol_rows if row["measurement_key"] not in done]
        if args.max_keyframes > 0:
            pending = pending[: args.max_keyframes]
        for idx, row in enumerate(pending, 1):
            measured = measure_keyframe_row(row, dense_index, args, cpu_device)
            keyframe_rows.append(measured)
            if idx % max(1, args.flush_every) == 0 or idx == len(pending):
                write_csv(keyframe_rows, paths["keyframe_csv"], KEYFRAME_FIELDS)
                print(json.dumps({"keyframes_measured_this_run": idx, "total_keyframe_rows": len(completed_keys(keyframe_rows))}), flush=True)

    if not args.skip_schedule_keyframes:
        groups = group_schedule_keyframes(frame_schedule_rows)
        done_groups = completed_schedule_keys(schedule_keyframe_rows)
        pending_groups = [(schedule, sequence, frames) for (schedule, sequence), frames in sorted(groups.items()) if (schedule, sequence) not in done_groups]
        if args.max_schedule_keyframe_groups > 0:
            pending_groups = pending_groups[: args.max_schedule_keyframe_groups]
        for idx, (schedule, sequence, frames) in enumerate(pending_groups, 1):
            measured = measure_schedule_keyframe_group(schedule, sequence, frames, dense_index, args, cpu_device)
            schedule_keyframe_rows.append(measured)
            if idx % max(1, args.flush_every) == 0 or idx == len(pending_groups):
                write_csv(schedule_keyframe_rows, paths["schedule_keyframe_csv"], SCHEDULE_KEYFRAME_FIELDS)
                print(json.dumps({"schedule_keyframe_groups_this_run": idx, "total_schedule_keyframe_groups": len(completed_schedule_keys(schedule_keyframe_rows))}), flush=True)

    if not args.skip_residuals:
        if not args.checkpoint.exists():
            raise FileNotFoundError(args.checkpoint)
        device = torch.device(args.device)
        opt = Options()
        opt.resume = str(args.checkpoint)
        opt.compile = False
        opt.input_frames = 1
        opt.output_frames = 3
        opt.epoch = 0
        model = load_model(deepcopy(opt), args.checkpoint, str(device))
        model.eval()
        done = completed_keys(residual_rows)
        pending = [row for row in residual_protocol_rows if row["measurement_key"] not in done]
        if args.max_residuals > 0:
            pending = pending[: args.max_residuals]
        batch_size = max(1, int(args.batch_size))
        processed = 0
        for start in range(0, len(pending), batch_size):
            batch_rows = pending[start:start + batch_size]
            measured_rows = measure_residual_batch(batch_rows, model, opt, dense_index, args, device)
            residual_rows.extend(measured_rows)
            processed += len(batch_rows)
            if processed % max(1, args.flush_every) == 0 or processed == len(pending):
                write_csv(residual_rows, paths["residual_csv"], RESIDUAL_FIELDS)
                print(json.dumps({"residuals_measured_this_run": processed, "total_residual_rows": len(completed_keys(residual_rows))}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    expected_schedule_groups = len(group_schedule_keyframes(frame_schedule_rows))
    summary_rows = build_summary(
        len(keyframe_protocol_rows),
        len(residual_protocol_rows),
        expected_schedule_groups,
        keyframe_rows,
        residual_rows,
        schedule_keyframe_rows,
    )
    write_csv(summary_rows, paths["summary_csv"], SUMMARY_FIELDS)
    complete = all(int(row["complete"]) == 1 for row in summary_rows)
    package = {
        "stage": 184,
        "status": "full_sequence_payload_measurement_complete" if complete else "full_sequence_payload_measurement_partial",
        "complete": complete,
        "keyframe_bits": int(args.keyframe_bits),
        "keyframe_compression": args.keyframe_compression,
        "keyframe_payload_encoding": args.keyframe_payload_encoding,
        "residual_policy": "stage158_psnr_best_half_q6_keep1_entropy_residual",
        "keep_fraction": float(args.keep_fraction),
        "side_bits": int(args.side_bits),
        "selector_payload_bytes": int(args.selector_payload_bytes),
        "summary_rows": summary_rows,
        "keyframe_csv": str(paths["keyframe_csv"]),
        "residual_csv": str(paths["residual_csv"]),
        "schedule_keyframe_csv": str(paths["schedule_keyframe_csv"]),
        "summary_csv": str(paths["summary_csv"]),
        "package_json": str(paths["package_json"]),
        "report_md": str(paths["report_md"]),
        "heavy_root": str(args.heavy_root),
    }
    paths["package_json"].write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary_rows, package, paths["report_md"])
    print(json.dumps({
        "package": str(paths["package_json"]),
        "status": package["status"],
        "complete": complete,
        "summary": summary_rows,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
