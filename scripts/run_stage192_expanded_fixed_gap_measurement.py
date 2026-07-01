import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/streamsplat.safetensors")
DEFAULT_STAGE191_ROOT = REPO_ROOT / "experiments/stage191_fixed_gap_expansion_protocol"
DEFAULT_STAGE184_ROOT = REPO_ROOT / "experiments/stage184_full_sequence_payload_measurement_execution"
DEFAULT_STAGE186_ROOT = REPO_ROOT / "experiments/stage186_full_sequence_quality_validation"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage192_expanded_fixed_gap_measurement"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage192_expanded_fixed_gap_measurement")

DEFAULT_FRAME_ROWS = DEFAULT_STAGE191_ROOT / "stage191_expanded_fixed_gap_frame_schedule_rows.csv"
DEFAULT_KEYFRAME_ROWS = DEFAULT_STAGE191_ROOT / "stage191_unique_keyframe_measurement_rows.csv"
DEFAULT_RESIDUAL_ROWS = DEFAULT_STAGE191_ROOT / "stage191_unique_stage158_residual_measurement_rows.csv"
DEFAULT_SCHEDULE_SUMMARY = DEFAULT_STAGE191_ROOT / "stage191_expanded_fixed_gap_schedule_summary.csv"

MIB = 1024.0 * 1024.0


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
import scripts.run_stage184_full_sequence_payload_measurement_execution as stage184  # noqa: E402
import scripts.run_stage186_full_sequence_quality_validation as stage186  # noqa: E402
from scripts.run_stage153_middle_multimetric_badcase_eval import mean, percentile  # noqa: E402
from scripts.run_stage72_original_davis_baseline import DEFAULT_DAVIS_ROOT  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_DENSE_MANIFEST  # noqa: E402


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

QUALITY_SUMMARY_FIELDS = [
    "schedule",
    "schedule_family",
    "gap",
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

TOTAL_RD_FIELDS = [
    "schedule",
    "schedule_family",
    "gap",
    "sequence_count",
    "total_frames",
    "keyframe_count",
    "residual_count",
    "keyframe_bitstream_bytes",
    "keyframe_mib_per_frame",
    "residual_payload_bytes",
    "residual_mib_per_frame",
    "metadata_bytes",
    "metadata_mib_per_frame",
    "total_payload_bytes",
    "total_mib_per_frame",
    "rate_scope",
]

RD_QUALITY_FIELDS = [
    "schedule",
    "schedule_family",
    "gap",
    "total_mib_per_frame",
    "keyframe_count",
    "residual_count",
    "mean_psnr",
    "mean_ssim",
    "mean_ms_ssim",
    "mean_lpips",
    "delta_rate_vs_gap8",
    "delta_psnr_vs_gap8",
    "delta_ssim_vs_gap8",
    "delta_ms_ssim_vs_gap8",
    "delta_lpips_vs_gap8",
    "best_fixed_by_psnr",
    "delta_rate_vs_best_fixed_psnr",
    "delta_psnr_vs_best_fixed_psnr",
    "delta_ssim_vs_best_fixed_psnr",
    "delta_ms_ssim_vs_best_fixed_psnr",
    "delta_lpips_vs_best_fixed_psnr",
    "beats_best_fixed_psnr_by_1db_and_no_metric_regression",
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


def numeric(row, key, default=0.0):
    value = row.get(key) if row else None
    if value in (None, "", "NA"):
        return default
    return float(value)


def paths(output_root):
    return {
        "keyframe_payload_csv": output_root / "stage192_unique_keyframe_payload_measurements.csv",
        "residual_payload_csv": output_root / "stage192_unique_stage158_residual_payload_measurements.csv",
        "schedule_keyframe_payload_csv": output_root / "stage192_schedule_packed_keyframe_payload_measurements.csv",
        "keyframe_quality_csv": output_root / "stage192_unique_keyframe_quality_metrics.csv",
        "residual_quality_csv": output_root / "stage192_unique_stage158_residual_quality_metrics.csv",
        "total_rd_csv": output_root / "stage192_expanded_fixed_gap_total_rd.csv",
        "final_quality_csv": output_root / "stage192_full_sequence_quality_by_schedule.csv",
        "quality_summary_csv": output_root / "stage192_full_sequence_quality_summary.csv",
        "rd_quality_csv": output_root / "stage192_expanded_fixed_gap_rd_quality_points.csv",
        "validation_csv": output_root / "stage192_expanded_fixed_gap_validation.csv",
        "package_json": output_root / "stage192_expanded_fixed_gap_measurement_package.json",
        "report_md": output_root / "stage192_expanded_fixed_gap_measurement_report.md",
    }


def ok_by_key(rows, key="measurement_key"):
    return {row[key]: row for row in rows if row.get("status") == "ok"}


def ok_keys(rows, key="measurement_key"):
    return set(ok_by_key(rows, key).keys())


def ok_schedule_groups(rows):
    return {(row["schedule"], row["sequence"]): row for row in rows if row.get("status") == "ok"}


def proto_by_key(rows, key="measurement_key"):
    return {row[key]: row for row in rows}


def normalize_keyed_row(proto, measured, meta_fields):
    row = dict(measured)
    for field in meta_fields:
        if field in proto:
            row[field] = proto[field]
    return row


def merge_seeded_keyed_rows(protocol_rows, existing_source_rows, existing_output_rows, meta_fields):
    protocol = proto_by_key(protocol_rows)
    source_ok = ok_by_key(existing_source_rows)
    output_ok = ok_by_key(existing_output_rows)
    output_all = {row["measurement_key"]: row for row in existing_output_rows}
    merged = {}
    for key, proto in protocol.items():
        if key in source_ok:
            merged[key] = normalize_keyed_row(proto, source_ok[key], meta_fields)
        if key in output_all:
            if output_all[key].get("status") == "ok" or key not in merged:
                merged[key] = normalize_keyed_row(proto, output_all[key], meta_fields)
    return [merged[key] for key in sorted(merged)]


def merge_seeded_schedule_groups(frame_rows, existing_source_rows, existing_output_rows):
    expected = stage184.group_schedule_keyframes(frame_rows)
    source_ok = ok_schedule_groups(existing_source_rows)
    output_all = {(row["schedule"], row["sequence"]): row for row in existing_output_rows}
    merged = {}
    for key in expected:
        if key in source_ok:
            merged[key] = dict(source_ok[key])
        if key in output_all:
            if output_all[key].get("status") == "ok" or key not in merged:
                merged[key] = dict(output_all[key])
    return [merged[key] for key in sorted(merged)]


def seed_outputs(args, p, keyframe_rows, residual_rows, frame_rows):
    keyframe_payload = merge_seeded_keyed_rows(
        keyframe_rows,
        read_csv(args.stage184_root / "stage184_unique_keyframe_payload_measurements.csv"),
        read_csv(p["keyframe_payload_csv"]),
        ["measurement_key", "sequence", "frame_index", "used_by_schedules", "schedule_count"],
    )
    residual_payload = merge_seeded_keyed_rows(
        residual_rows,
        read_csv(args.stage184_root / "stage184_unique_stage158_residual_payload_measurements.csv"),
        read_csv(p["residual_payload_csv"]),
        ["measurement_key", "sequence", "target_index", "left_index", "right_index", "segment_length", "normalized_time", "used_by_schedules", "schedule_count"],
    )
    schedule_payload = merge_seeded_schedule_groups(
        frame_rows,
        read_csv(args.stage184_root / "stage184_schedule_packed_keyframe_payload_measurements.csv"),
        read_csv(p["schedule_keyframe_payload_csv"]),
    )
    keyframe_quality = merge_seeded_keyed_rows(
        keyframe_rows,
        read_csv(args.stage186_root / "stage186_unique_keyframe_quality_metrics.csv"),
        read_csv(p["keyframe_quality_csv"]),
        ["measurement_key", "sequence", "frame_index", "used_by_schedules", "schedule_count"],
    )
    residual_quality = merge_seeded_keyed_rows(
        residual_rows,
        read_csv(args.stage186_root / "stage186_unique_stage158_residual_quality_metrics.csv"),
        read_csv(p["residual_quality_csv"]),
        ["measurement_key", "sequence", "target_index", "left_index", "right_index", "segment_length", "normalized_time", "used_by_schedules", "schedule_count"],
    )
    write_csv(keyframe_payload, p["keyframe_payload_csv"], stage184.KEYFRAME_FIELDS)
    write_csv(residual_payload, p["residual_payload_csv"], stage184.RESIDUAL_FIELDS)
    write_csv(schedule_payload, p["schedule_keyframe_payload_csv"], stage184.SCHEDULE_KEYFRAME_FIELDS)
    write_csv(keyframe_quality, p["keyframe_quality_csv"], stage186.KEYFRAME_QUALITY_FIELDS)
    write_csv(residual_quality, p["residual_quality_csv"], stage186.RESIDUAL_QUALITY_FIELDS)
    return keyframe_payload, residual_payload, schedule_payload, keyframe_quality, residual_quality


def measure_payloads(args, p, keyframe_rows, residual_rows, frame_rows, dense_index):
    keyframe_payload = read_csv(p["keyframe_payload_csv"])
    residual_payload = read_csv(p["residual_payload_csv"])
    schedule_payload = read_csv(p["schedule_keyframe_payload_csv"])
    cpu_device = torch.device("cpu")

    if not args.skip_payload_keyframes:
        done = ok_keys(keyframe_payload)
        pending = [row for row in keyframe_rows if row["measurement_key"] not in done]
        if args.max_payload_keyframes > 0:
            pending = pending[: args.max_payload_keyframes]
        for idx, row in enumerate(pending, 1):
            keyframe_payload.append(stage184.measure_keyframe_row(row, dense_index, args, cpu_device))
            if idx % max(1, args.flush_every) == 0 or idx == len(pending):
                write_csv(keyframe_payload, p["keyframe_payload_csv"], stage184.KEYFRAME_FIELDS)
                print(json.dumps({"payload_keyframes_this_run": idx, "payload_keyframes_ok_total": len(ok_keys(keyframe_payload))}), flush=True)

    if not args.skip_schedule_keyframes:
        groups = stage184.group_schedule_keyframes(frame_rows)
        done = set(ok_schedule_groups(schedule_payload).keys())
        pending = [(schedule, sequence, frames) for (schedule, sequence), frames in sorted(groups.items()) if (schedule, sequence) not in done]
        if args.max_schedule_keyframe_groups > 0:
            pending = pending[: args.max_schedule_keyframe_groups]
        for idx, (schedule, sequence, frames) in enumerate(pending, 1):
            schedule_payload.append(stage184.measure_schedule_keyframe_group(schedule, sequence, frames, dense_index, args, cpu_device))
            if idx % max(1, args.flush_every) == 0 or idx == len(pending):
                write_csv(schedule_payload, p["schedule_keyframe_payload_csv"], stage184.SCHEDULE_KEYFRAME_FIELDS)
                print(json.dumps({"schedule_keyframe_groups_this_run": idx, "schedule_keyframe_groups_ok_total": len(ok_schedule_groups(schedule_payload))}), flush=True)

    if not args.skip_payload_residuals:
        if not args.checkpoint.exists():
            raise FileNotFoundError(args.checkpoint)
        device = torch.device(args.device)
        opt = Options()
        opt.resume = str(args.checkpoint)
        opt.compile = False
        opt.input_frames = 1
        opt.output_frames = 3
        opt.epoch = 0
        model = stage184.load_model(deepcopy(opt), args.checkpoint, str(device))
        model.eval()
        done = ok_keys(residual_payload)
        pending = [row for row in residual_rows if row["measurement_key"] not in done]
        if args.max_payload_residuals > 0:
            pending = pending[: args.max_payload_residuals]
        processed = 0
        batch_size = max(1, int(args.payload_batch_size))
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            residual_payload.extend(stage184.measure_residual_batch(batch, model, opt, dense_index, args, device))
            processed += len(batch)
            if processed % max(1, args.flush_every) == 0 or processed == len(pending):
                write_csv(residual_payload, p["residual_payload_csv"], stage184.RESIDUAL_FIELDS)
                print(json.dumps({"payload_residuals_this_run": processed, "payload_residuals_ok_total": len(ok_keys(residual_payload))}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def measure_quality(args, p, keyframe_rows, residual_rows, dense_index):
    keyframe_payload = read_csv(p["keyframe_payload_csv"])
    residual_payload = read_csv(p["residual_payload_csv"])
    keyframe_quality = read_csv(p["keyframe_quality_csv"])
    residual_quality = read_csv(p["residual_quality_csv"])
    keyframe_payload_ok = ok_by_key(keyframe_payload)
    residual_payload_ok = ok_by_key(residual_payload)

    device = torch.device(args.device)
    opt = Options()
    opt.resume = str(args.checkpoint)
    opt.compile = False
    opt.input_frames = 1
    opt.output_frames = 3
    opt.epoch = 0
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    lpips_model, lpips_error, ms_ssim_module, ms_ssim_error = stage186.load_metric_modules(device, args.disable_lpips, args.disable_ms_ssim)

    if not args.skip_quality_keyframes:
        done = ok_keys(keyframe_quality)
        pending = [keyframe_payload_ok[row["measurement_key"]] for row in keyframe_rows if row["measurement_key"] in keyframe_payload_ok and row["measurement_key"] not in done]
        if args.max_quality_keyframes > 0:
            pending = pending[: args.max_quality_keyframes]
        for idx, row in enumerate(pending, 1):
            keyframe_quality.append(stage186.measure_keyframe_quality(row, dense_index, opt, device, background, lpips_model, ms_ssim_module, args))
            if idx % max(1, args.flush_every) == 0 or idx == len(pending):
                write_csv(keyframe_quality, p["keyframe_quality_csv"], stage186.KEYFRAME_QUALITY_FIELDS)
                print(json.dumps({"quality_keyframes_this_run": idx, "quality_keyframes_ok_total": len(ok_keys(keyframe_quality))}), flush=True)

    if not args.skip_quality_residuals:
        if not args.checkpoint.exists():
            raise FileNotFoundError(args.checkpoint)
        model = stage186.load_model(deepcopy(opt), args.checkpoint, str(device))
        model.eval()
        done = ok_keys(residual_quality)
        pending = [residual_payload_ok[row["measurement_key"]] for row in residual_rows if row["measurement_key"] in residual_payload_ok and row["measurement_key"] not in done]
        if args.max_quality_residuals > 0:
            pending = pending[: args.max_quality_residuals]
        processed = 0
        batch_size = max(1, int(args.quality_batch_size))
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            residual_quality.extend(stage186.measure_residual_quality_batch(batch, model, opt, dense_index, lpips_model, ms_ssim_module, args, device))
            processed += len(batch)
            if processed % max(1, args.flush_every) == 0 or processed == len(pending):
                write_csv(residual_quality, p["residual_quality_csv"], stage186.RESIDUAL_QUALITY_FIELDS)
                print(json.dumps({"quality_residuals_this_run": processed, "quality_residuals_ok_total": len(ok_keys(residual_quality))}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return lpips_error, ms_ssim_error


def schedule_order(summary_rows):
    return [row["schedule"] for row in summary_rows]


def metadata_by_schedule(summary_rows):
    return {row["schedule"]: int(float(row["metadata_bytes"])) for row in summary_rows}


def aggregate(args, p, frame_rows, summary_rows):
    keyframe_payload = ok_by_key(read_csv(p["keyframe_payload_csv"]))
    residual_payload = ok_by_key(read_csv(p["residual_payload_csv"]))
    schedule_payload = ok_schedule_groups(read_csv(p["schedule_keyframe_payload_csv"]))
    keyframe_quality = ok_by_key(read_csv(p["keyframe_quality_csv"]))
    residual_quality = ok_by_key(read_csv(p["residual_quality_csv"]))

    schedules = schedule_order(summary_rows)
    summary_by_schedule = {row["schedule"]: row for row in summary_rows}
    metadata = metadata_by_schedule(summary_rows)
    rows_by_schedule = {schedule: [row for row in frame_rows if row["schedule"] == schedule] for schedule in schedules}

    total_rows = []
    validation = []
    for schedule in schedules:
        group = rows_by_schedule[schedule]
        sequences = sorted({row["sequence"] for row in group})
        keyframe_count = sum(1 for row in group if row["measurement_type"] == "q12_keyframe_payload")
        residual_group = [row for row in group if row["measurement_type"] == "stage158_residual_payload"]
        missing_residual_payload = [row["measurement_key"] for row in residual_group if row["measurement_key"] not in residual_payload]
        missing_schedule_payload = [sequence for sequence in sequences if (schedule, sequence) not in schedule_payload]
        residual_bytes = sum(numeric(residual_payload[row["measurement_key"]], "payload_bytes") for row in residual_group if row["measurement_key"] in residual_payload)
        keyframe_bytes = sum(numeric(schedule_payload[(schedule, sequence)], "bitstream_bytes") for sequence in sequences if (schedule, sequence) in schedule_payload)
        metadata_bytes = metadata[schedule]
        total_bytes = keyframe_bytes + residual_bytes + metadata_bytes
        total_frames = len(group)
        total_rows.append(
            {
                "schedule": schedule,
                "schedule_family": summary_by_schedule[schedule]["schedule_family"],
                "gap": summary_by_schedule[schedule]["gap"],
                "sequence_count": len(sequences),
                "total_frames": total_frames,
                "keyframe_count": keyframe_count,
                "residual_count": len(residual_group),
                "keyframe_bitstream_bytes": keyframe_bytes,
                "keyframe_mib_per_frame": keyframe_bytes / MIB / float(total_frames),
                "residual_payload_bytes": residual_bytes,
                "residual_mib_per_frame": residual_bytes / MIB / float(total_frames),
                "metadata_bytes": metadata_bytes,
                "metadata_mib_per_frame": metadata_bytes / MIB / float(total_frames),
                "total_payload_bytes": total_bytes,
                "total_mib_per_frame": total_bytes / MIB / float(total_frames),
                "rate_scope": "measured_schedule_packed_q12_keyframes_plus_measured_stage158_residual_payloads_plus_exact_metadata",
            }
        )
        validation.append({"item": f"{schedule}_missing_residual_payload", "expected": 0, "actual": len(missing_residual_payload), "status": "ok" if not missing_residual_payload else "error"})
        validation.append({"item": f"{schedule}_missing_schedule_keyframe_payload", "expected": 0, "actual": len(missing_schedule_payload), "status": "ok" if not missing_schedule_payload else "error"})

    final_rows = []
    missing_final = []
    for row in frame_rows:
        if row["measurement_type"] == "q12_keyframe_payload":
            measured = keyframe_quality.get(row["measurement_key"])
            final_type = "q12_keyframe"
            payload = 0
        else:
            measured = residual_quality.get(row["measurement_key"])
            final_type = "stage158_residual_recovery"
            payload = int(numeric(measured, "payload_bytes", 0.0)) if measured else 0
        if measured is None:
            missing_final.append(row["measurement_key"])
            continue
        final_rows.append(
            {
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
            }
        )

    quality_rows = []
    for schedule in schedules:
        group = [row for row in final_rows if row["schedule"] == schedule]
        keyframes = [row for row in group if row["final_type"] == "q12_keyframe"]
        residuals = [row for row in group if row["final_type"] == "stage158_residual_recovery"]
        meta = summary_by_schedule[schedule]
        quality_rows.append(
            {
                "schedule": schedule,
                "schedule_family": meta["schedule_family"],
                "gap": meta["gap"],
                "frame_count": len(group),
                "keyframe_count": len(keyframes),
                "residual_count": len(residuals),
                "mean_psnr": mean(row["psnr"] for row in group),
                "p10_psnr": percentile((row["psnr"] for row in group), 10),
                "mean_keyframe_psnr": mean(row["psnr"] for row in keyframes),
                "mean_residual_psnr": mean(row["psnr"] for row in residuals),
                "mean_ssim": mean(row["ssim"] for row in group),
                "p10_ssim": percentile((row["ssim"] for row in group), 10),
                "mean_ms_ssim": mean(row["ms_ssim"] for row in group),
                "mean_lpips": mean(row["lpips"] for row in group),
                "p90_lpips": percentile((row["lpips"] for row in group), 90),
                "mean_payload_bytes_per_frame_schedule_row": mean(row["payload_bytes"] for row in group),
            }
        )
        validation.append({"item": f"{schedule}_final_quality_rows", "expected": len(rows_by_schedule[schedule]), "actual": len(group), "status": "ok" if len(group) == len(rows_by_schedule[schedule]) else "error"})

    expected_keyframe_count = len({row["measurement_key"] for row in frame_rows if row["measurement_type"] == "q12_keyframe_payload"})
    expected_residual_count = len({row["measurement_key"] for row in frame_rows if row["measurement_type"] == "stage158_residual_payload"})
    validation.extend(
        [
            {"item": "unique_keyframe_payload_rows", "expected": expected_keyframe_count, "actual": len(keyframe_payload), "status": "ok" if len(keyframe_payload) == expected_keyframe_count else "error"},
            {"item": "unique_residual_payload_rows", "expected": expected_residual_count, "actual": len(residual_payload), "status": "ok" if len(residual_payload) == expected_residual_count else "error"},
            {"item": "unique_keyframe_quality_rows", "expected": expected_keyframe_count, "actual": len(keyframe_quality), "status": "ok" if len(keyframe_quality) == expected_keyframe_count else "error"},
            {"item": "unique_residual_quality_rows", "expected": expected_residual_count, "actual": len(residual_quality), "status": "ok" if len(residual_quality) == expected_residual_count else "error"},
            {"item": "missing_final_measurements", "expected": 0, "actual": len(missing_final), "status": "ok" if not missing_final else "error"},
        ]
    )

    rd_by_schedule = {row["schedule"]: row for row in total_rows}
    q_by_schedule = {row["schedule"]: row for row in quality_rows}
    fixed_quality = [row for row in quality_rows if row["schedule_family"] == "fixed_gap"]
    best_fixed = max(fixed_quality, key=lambda row: row["mean_psnr"])
    gap8 = q_by_schedule.get("uniform_gap8")
    gap8_rd = rd_by_schedule.get("uniform_gap8")
    rd_quality = []
    for schedule in schedules:
        q = q_by_schedule[schedule]
        r = rd_by_schedule[schedule]
        pass_flag = int(
            q["mean_psnr"] >= best_fixed["mean_psnr"] + 1.0
            and q["mean_ssim"] >= best_fixed["mean_ssim"]
            and q["mean_ms_ssim"] >= best_fixed["mean_ms_ssim"]
            and q["mean_lpips"] <= best_fixed["mean_lpips"]
        )
        rd_quality.append(
            {
                "schedule": schedule,
                "schedule_family": q["schedule_family"],
                "gap": q["gap"],
                "total_mib_per_frame": r["total_mib_per_frame"],
                "keyframe_count": r["keyframe_count"],
                "residual_count": r["residual_count"],
                "mean_psnr": q["mean_psnr"],
                "mean_ssim": q["mean_ssim"],
                "mean_ms_ssim": q["mean_ms_ssim"],
                "mean_lpips": q["mean_lpips"],
                "delta_rate_vs_gap8": r["total_mib_per_frame"] - gap8_rd["total_mib_per_frame"] if gap8_rd else "",
                "delta_psnr_vs_gap8": q["mean_psnr"] - gap8["mean_psnr"] if gap8 else "",
                "delta_ssim_vs_gap8": q["mean_ssim"] - gap8["mean_ssim"] if gap8 else "",
                "delta_ms_ssim_vs_gap8": q["mean_ms_ssim"] - gap8["mean_ms_ssim"] if gap8 else "",
                "delta_lpips_vs_gap8": q["mean_lpips"] - gap8["mean_lpips"] if gap8 else "",
                "best_fixed_by_psnr": best_fixed["schedule"],
                "delta_rate_vs_best_fixed_psnr": r["total_mib_per_frame"] - rd_by_schedule[best_fixed["schedule"]]["total_mib_per_frame"],
                "delta_psnr_vs_best_fixed_psnr": q["mean_psnr"] - best_fixed["mean_psnr"],
                "delta_ssim_vs_best_fixed_psnr": q["mean_ssim"] - best_fixed["mean_ssim"],
                "delta_ms_ssim_vs_best_fixed_psnr": q["mean_ms_ssim"] - best_fixed["mean_ms_ssim"],
                "delta_lpips_vs_best_fixed_psnr": q["mean_lpips"] - best_fixed["mean_lpips"],
                "beats_best_fixed_psnr_by_1db_and_no_metric_regression": pass_flag,
            }
        )
    return total_rows, final_rows, quality_rows, rd_quality, validation, best_fixed


def decision(complete, rd_quality_rows):
    if not complete:
        return "expanded_fixed_gap_measurement_partial"
    adaptive = next(row for row in rd_quality_rows if row["schedule"] == "stage165_adaptive")
    if int(adaptive["beats_best_fixed_psnr_by_1db_and_no_metric_regression"]) == 1:
        return "current_adaptive_beats_best_fixed_by_target_margin"
    if adaptive["delta_psnr_vs_best_fixed_psnr"] > 0:
        return "current_adaptive_beats_best_fixed_psnr_but_not_target_margin"
    return "current_adaptive_not_strong_against_expanded_fixed_gaps"


def write_report(rd_quality_rows, validation_rows, package, path):
    lines = [
        "# Stage192 Expanded Fixed-Gap Measurement",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Complete: `{package['complete']}`.",
        f"- Best fixed gap by PSNR: `{package['best_fixed_by_psnr']}`.",
        "",
        "## Expanded RD-Quality",
        "",
        "| schedule | MiB/frame | keyframes | PSNR | SSIM | MS-SSIM | LPIPS | dPSNR vs best fixed | dLPIPS vs best fixed | pass +1dB/no-regression |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rd_quality_rows:
        lines.append(
            f"| {row['schedule']} | {float(row['total_mib_per_frame']):.12f} | {row['keyframe_count']} | "
            f"{float(row['mean_psnr']):.6f} | {float(row['mean_ssim']):.6f} | {float(row['mean_ms_ssim']):.6f} | {float(row['mean_lpips']):.6f} | "
            f"{float(row['delta_psnr_vs_best_fixed_psnr']):.6f} | {float(row['delta_lpips_vs_best_fixed_psnr']):.6f} | {row['beats_best_fixed_psnr_by_1db_and_no_metric_regression']} |"
        )
    lines.extend([
        "",
        "## Validation",
        "",
        "| item | expected | actual | status |",
        "|---|---:|---:|---|",
    ])
    for row in validation_rows:
        lines.append(f"| {row['item']} | {row['expected']} | {row['actual']} | {row['status']} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- This stage is the expanded fixed-gap baseline check requested before stronger selector claims.",
        "- If current adaptive does not beat the best fixed-gap baseline, Stage193 must compute oracle headroom before additional selector tuning.",
        "- The target for a strong selector claim is about +1 dB PSNR over the best tested fixed gap with no SSIM/MS-SSIM/LPIPS regression.",
        "",
        "## Outputs",
        "",
        f"- RD-quality CSV: `{package['rd_quality_csv']}`",
        f"- Total RD CSV: `{package['total_rd_csv']}`",
        f"- Quality summary CSV: `{package['quality_summary_csv']}`",
        f"- Final quality CSV: `{package['final_quality_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_rows", type=Path, default=DEFAULT_FRAME_ROWS)
    parser.add_argument("--keyframe_rows", type=Path, default=DEFAULT_KEYFRAME_ROWS)
    parser.add_argument("--residual_rows", type=Path, default=DEFAULT_RESIDUAL_ROWS)
    parser.add_argument("--schedule_summary", type=Path, default=DEFAULT_SCHEDULE_SUMMARY)
    parser.add_argument("--stage184_root", type=Path, default=DEFAULT_STAGE184_ROOT)
    parser.add_argument("--stage186_root", type=Path, default=DEFAULT_STAGE186_ROOT)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--keyframe_bits", type=int, default=12)
    parser.add_argument("--keyframe_compression", choices=["none", "zlib"], default="none")
    parser.add_argument("--keyframe_payload_encoding", choices=["bitpack", "dtype"], default="bitpack")
    parser.add_argument("--keep_fraction", type=float, default=1.0)
    parser.add_argument("--side_bits", type=int, default=6)
    parser.add_argument("--selector_payload_bytes", type=int, default=1)
    parser.add_argument("--zlib_level", type=int, default=9)
    parser.add_argument("--payload_batch_size", type=int, default=2)
    parser.add_argument("--quality_batch_size", type=int, default=8)
    parser.add_argument("--max_payload_keyframes", type=int, default=0)
    parser.add_argument("--max_payload_residuals", type=int, default=0)
    parser.add_argument("--max_schedule_keyframe_groups", type=int, default=0)
    parser.add_argument("--max_quality_keyframes", type=int, default=0)
    parser.add_argument("--max_quality_residuals", type=int, default=0)
    parser.add_argument("--flush_every", type=int, default=200)
    parser.add_argument("--skip_payload_keyframes", action="store_true")
    parser.add_argument("--skip_payload_residuals", action="store_true")
    parser.add_argument("--skip_schedule_keyframes", action="store_true")
    parser.add_argument("--skip_quality_keyframes", action="store_true")
    parser.add_argument("--skip_quality_residuals", action="store_true")
    parser.add_argument("--disable_lpips", action="store_true")
    parser.add_argument("--disable_ms_ssim", action="store_true")
    parser.add_argument("--seed_only", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    p = paths(args.output_root)
    frame_rows = read_csv(args.frame_rows)
    keyframe_rows = read_csv(args.keyframe_rows)
    residual_rows = read_csv(args.residual_rows)
    summary_rows = read_csv(args.schedule_summary)

    seed_outputs(args, p, keyframe_rows, residual_rows, frame_rows)
    if not args.seed_only and not args.aggregate_only:
        dense_index = stage184.build_dense_index(args.dense_manifest, ["val"])
        measure_payloads(args, p, keyframe_rows, residual_rows, frame_rows, dense_index)
        measure_quality(args, p, keyframe_rows, residual_rows, dense_index)

    total_rows, final_rows, quality_rows, rd_quality_rows, validation_rows, best_fixed = aggregate(args, p, frame_rows, summary_rows)
    complete = all(row["status"] == "ok" for row in validation_rows)
    stage_decision = decision(complete, rd_quality_rows)

    write_csv(total_rows, p["total_rd_csv"], TOTAL_RD_FIELDS)
    write_csv(final_rows, p["final_quality_csv"], FINAL_QUALITY_FIELDS)
    write_csv(quality_rows, p["quality_summary_csv"], QUALITY_SUMMARY_FIELDS)
    write_csv(rd_quality_rows, p["rd_quality_csv"], RD_QUALITY_FIELDS)
    write_csv(validation_rows, p["validation_csv"], VALIDATION_FIELDS)

    package = {
        "stage": 192,
        "status": "expanded_fixed_gap_measurement_complete" if complete else "expanded_fixed_gap_measurement_partial",
        "complete": complete,
        "decision": stage_decision,
        "best_fixed_by_psnr": best_fixed["schedule"],
        "best_fixed_psnr": best_fixed["mean_psnr"],
        "rd_quality_rows": rd_quality_rows,
        "validation_rows": validation_rows,
        "keyframe_payload_csv": str(p["keyframe_payload_csv"].relative_to(REPO_ROOT)),
        "residual_payload_csv": str(p["residual_payload_csv"].relative_to(REPO_ROOT)),
        "schedule_keyframe_payload_csv": str(p["schedule_keyframe_payload_csv"].relative_to(REPO_ROOT)),
        "keyframe_quality_csv": str(p["keyframe_quality_csv"].relative_to(REPO_ROOT)),
        "residual_quality_csv": str(p["residual_quality_csv"].relative_to(REPO_ROOT)),
        "total_rd_csv": str(p["total_rd_csv"].relative_to(REPO_ROOT)),
        "final_quality_csv": str(p["final_quality_csv"].relative_to(REPO_ROOT)),
        "quality_summary_csv": str(p["quality_summary_csv"].relative_to(REPO_ROOT)),
        "rd_quality_csv": str(p["rd_quality_csv"].relative_to(REPO_ROOT)),
        "validation_csv": str(p["validation_csv"].relative_to(REPO_ROOT)),
        "package_json": str(p["package_json"].relative_to(REPO_ROOT)),
        "report_md": str(p["report_md"].relative_to(REPO_ROOT)),
        "heavy_root": str(args.heavy_root),
    }
    p["package_json"].write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(rd_quality_rows, validation_rows, package, p["report_md"])
    adaptive = next(row for row in rd_quality_rows if row["schedule"] == "stage165_adaptive")
    print(
        json.dumps(
            {
                "package": str(p["package_json"]),
                "status": package["status"],
                "decision": stage_decision,
                "best_fixed_by_psnr": best_fixed["schedule"],
                "adaptive_delta_psnr_vs_best_fixed": adaptive["delta_psnr_vs_best_fixed_psnr"],
                "complete": complete,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
