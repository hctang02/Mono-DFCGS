import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/streamsplat.safetensors")
DEFAULT_STAGE192_ROOT = REPO_ROOT / "experiments/stage192_expanded_fixed_gap_measurement"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage194_all_keyframe_q12_upper_bound"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage194_all_keyframe_q12_upper_bound")

DEFAULT_STAGE192_FINAL_ROWS = DEFAULT_STAGE192_ROOT / "stage192_full_sequence_quality_by_schedule.csv"
DEFAULT_STAGE192_KEYFRAME_PAYLOAD = DEFAULT_STAGE192_ROOT / "stage192_unique_keyframe_payload_measurements.csv"
DEFAULT_STAGE192_KEYFRAME_QUALITY = DEFAULT_STAGE192_ROOT / "stage192_unique_keyframe_quality_metrics.csv"
DEFAULT_STAGE192_RD_QUALITY = DEFAULT_STAGE192_ROOT / "stage192_expanded_fixed_gap_rd_quality_points.csv"

SCHEDULE = "uniform_gap1"
SCHEDULE_FAMILY = "fixed_gap"
GAP = 1
METADATA_BYTES = 1
MIB = 1024.0 * 1024.0


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
import scripts.run_stage184_full_sequence_payload_measurement_execution as stage184  # noqa: E402
import scripts.run_stage186_full_sequence_quality_validation as stage186  # noqa: E402
from scripts.run_stage153_middle_multimetric_badcase_eval import mean, percentile  # noqa: E402
from scripts.run_stage72_original_davis_baseline import DEFAULT_DAVIS_ROOT  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_DENSE_MANIFEST  # noqa: E402


PROTOCOL_FIELDS = [
    "schedule",
    "schedule_family",
    "gap",
    "measurement_type",
    "measurement_key",
    "sequence",
    "frame_index",
    "used_by_schedules",
    "schedule_count",
]

FINAL_FIELDS = [
    "schedule",
    "sequence",
    "frame_index",
    "final_type",
    "measurement_key",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "bitstream_bytes",
]

SUMMARY_FIELDS = [
    "schedule",
    "schedule_family",
    "gap",
    "frame_count",
    "keyframe_count",
    "mean_psnr",
    "p10_psnr",
    "mean_ssim",
    "p10_ssim",
    "mean_ms_ssim",
    "mean_lpips",
    "p90_lpips",
    "keyframe_bitstream_bytes",
    "metadata_bytes",
    "total_payload_bytes",
    "total_mib_per_frame",
    "best_fixed_reference",
    "best_fixed_psnr",
    "best_fixed_ssim",
    "best_fixed_ms_ssim",
    "best_fixed_lpips",
    "delta_rate_vs_best_fixed",
    "delta_psnr_vs_best_fixed",
    "delta_ssim_vs_best_fixed",
    "delta_ms_ssim_vs_best_fixed",
    "delta_lpips_vs_best_fixed",
    "beats_best_fixed_by_1db_no_metric_regression",
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
        "protocol_csv": output_root / "stage194_all_keyframe_protocol_rows.csv",
        "keyframe_payload_csv": output_root / "stage194_unique_keyframe_payload_measurements.csv",
        "schedule_keyframe_payload_csv": output_root / "stage194_schedule_packed_keyframe_payload_measurements.csv",
        "keyframe_quality_csv": output_root / "stage194_unique_keyframe_quality_metrics.csv",
        "final_quality_csv": output_root / "stage194_all_keyframe_quality_rows.csv",
        "summary_csv": output_root / "stage194_all_keyframe_q12_summary.csv",
        "validation_csv": output_root / "stage194_all_keyframe_q12_validation.csv",
        "package_json": output_root / "stage194_all_keyframe_q12_upper_bound_package.json",
        "report_md": output_root / "stage194_all_keyframe_q12_upper_bound_report.md",
    }


def keyframe_key(sequence, frame_index):
    return f"keyframe::{sequence}:{int(frame_index):05d}"


def ok_by_key(rows, key="measurement_key"):
    return {row[key]: row for row in rows if row.get("status") == "ok"}


def ok_schedule_groups(rows):
    return {(row["schedule"], row["sequence"]): row for row in rows if row.get("status") == "ok"}


def build_protocol(stage192_final_rows):
    frames = sorted({(row["sequence"], int(row["frame_index"])) for row in stage192_final_rows})
    return [
        {
            "schedule": SCHEDULE,
            "schedule_family": SCHEDULE_FAMILY,
            "gap": GAP,
            "measurement_type": "q12_keyframe_payload",
            "measurement_key": keyframe_key(sequence, frame_index),
            "sequence": sequence,
            "frame_index": frame_index,
            "used_by_schedules": SCHEDULE,
            "schedule_count": 1,
        }
        for sequence, frame_index in frames
    ]


def normalize_keyed_row(proto, measured, meta_fields):
    row = dict(measured)
    for field in meta_fields:
        row[field] = proto[field]
    return row


def merge_seeded_keyed_rows(protocol_rows, existing_source_rows, existing_output_rows, meta_fields):
    protocol = {row["measurement_key"]: row for row in protocol_rows}
    source_ok = ok_by_key(existing_source_rows)
    output_all = {row["measurement_key"]: row for row in existing_output_rows}
    merged = {}
    for key, proto in protocol.items():
        if key in source_ok:
            merged[key] = normalize_keyed_row(proto, source_ok[key], meta_fields)
        if key in output_all:
            if output_all[key].get("status") == "ok" or key not in merged:
                merged[key] = normalize_keyed_row(proto, output_all[key], meta_fields)
    return [merged[key] for key in sorted(merged)]


def seed_outputs(args, p, protocol_rows):
    keyframe_payload = merge_seeded_keyed_rows(
        protocol_rows,
        read_csv(args.stage192_keyframe_payload),
        read_csv(p["keyframe_payload_csv"]),
        ["measurement_key", "sequence", "frame_index", "used_by_schedules", "schedule_count"],
    )
    keyframe_quality = merge_seeded_keyed_rows(
        protocol_rows,
        read_csv(args.stage192_keyframe_quality),
        read_csv(p["keyframe_quality_csv"]),
        ["measurement_key", "sequence", "frame_index", "used_by_schedules", "schedule_count"],
    )
    write_csv(keyframe_payload, p["keyframe_payload_csv"], stage184.KEYFRAME_FIELDS)
    write_csv(keyframe_quality, p["keyframe_quality_csv"], stage186.KEYFRAME_QUALITY_FIELDS)
    return keyframe_payload, keyframe_quality


def protocol_groups(protocol_rows):
    groups = {}
    for row in protocol_rows:
        groups.setdefault(row["sequence"], []).append(int(row["frame_index"]))
    return {sequence: sorted(frames) for sequence, frames in groups.items()}


def measure_payloads(args, p, protocol_rows, dense_index):
    keyframe_payload = read_csv(p["keyframe_payload_csv"])
    schedule_payload = read_csv(p["schedule_keyframe_payload_csv"])
    cpu_device = torch.device("cpu")

    if not args.skip_payload_keyframes:
        done = set(ok_by_key(keyframe_payload).keys())
        pending = [row for row in protocol_rows if row["measurement_key"] not in done]
        if args.max_payload_keyframes > 0:
            pending = pending[: args.max_payload_keyframes]
        for idx, row in enumerate(pending, 1):
            keyframe_payload.append(stage184.measure_keyframe_row(row, dense_index, args, cpu_device))
            if idx % max(1, args.flush_every) == 0 or idx == len(pending):
                write_csv(keyframe_payload, p["keyframe_payload_csv"], stage184.KEYFRAME_FIELDS)
                print(json.dumps({"payload_keyframes_this_run": idx, "payload_keyframes_ok_total": len(ok_by_key(keyframe_payload))}), flush=True)

    if not args.skip_schedule_keyframes:
        groups = protocol_groups(protocol_rows)
        done = set(ok_schedule_groups(schedule_payload).keys())
        pending = [(sequence, frames) for sequence, frames in sorted(groups.items()) if (SCHEDULE, sequence) not in done]
        if args.max_schedule_keyframe_groups > 0:
            pending = pending[: args.max_schedule_keyframe_groups]
        for idx, (sequence, frames) in enumerate(pending, 1):
            schedule_payload.append(stage184.measure_schedule_keyframe_group(SCHEDULE, sequence, frames, dense_index, args, cpu_device))
            if idx % max(1, args.flush_every) == 0 or idx == len(pending):
                write_csv(schedule_payload, p["schedule_keyframe_payload_csv"], stage184.SCHEDULE_KEYFRAME_FIELDS)
                print(json.dumps({"schedule_keyframe_groups_this_run": idx, "schedule_keyframe_groups_ok_total": len(ok_schedule_groups(schedule_payload))}), flush=True)


def measure_quality(args, p, protocol_rows, dense_index):
    keyframe_payload = ok_by_key(read_csv(p["keyframe_payload_csv"]))
    keyframe_quality = read_csv(p["keyframe_quality_csv"])
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
        done = set(ok_by_key(keyframe_quality).keys())
        pending = [keyframe_payload[row["measurement_key"]] for row in protocol_rows if row["measurement_key"] in keyframe_payload and row["measurement_key"] not in done]
        if args.max_quality_keyframes > 0:
            pending = pending[: args.max_quality_keyframes]
        for idx, row in enumerate(pending, 1):
            keyframe_quality.append(stage186.measure_keyframe_quality(row, dense_index, opt, device, background, lpips_model, ms_ssim_module, args))
            if idx % max(1, args.flush_every) == 0 or idx == len(pending):
                write_csv(keyframe_quality, p["keyframe_quality_csv"], stage186.KEYFRAME_QUALITY_FIELDS)
                print(json.dumps({"quality_keyframes_this_run": idx, "quality_keyframes_ok_total": len(ok_by_key(keyframe_quality))}), flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return lpips_error, ms_ssim_error


def best_fixed_gap2(stage192_rd_quality_rows):
    for row in stage192_rd_quality_rows:
        if row["schedule"] == "uniform_gap2":
            return row
    fixed = [row for row in stage192_rd_quality_rows if row.get("schedule_family") == "fixed_gap"]
    return max(fixed, key=lambda row: numeric(row, "mean_psnr"))


def aggregate(args, p, protocol_rows):
    keyframe_payload = ok_by_key(read_csv(p["keyframe_payload_csv"]))
    schedule_payload = ok_schedule_groups(read_csv(p["schedule_keyframe_payload_csv"]))
    keyframe_quality = ok_by_key(read_csv(p["keyframe_quality_csv"]))
    best_fixed = best_fixed_gap2(read_csv(args.stage192_rd_quality))
    best_fixed_rate = numeric(best_fixed, "total_mib_per_frame")

    final_rows = []
    missing_payload = []
    missing_quality = []
    for row in protocol_rows:
        key = row["measurement_key"]
        payload = keyframe_payload.get(key)
        quality = keyframe_quality.get(key)
        if payload is None:
            missing_payload.append(key)
        if quality is None:
            missing_quality.append(key)
            continue
        final_rows.append(
            {
                "schedule": SCHEDULE,
                "sequence": row["sequence"],
                "frame_index": int(row["frame_index"]),
                "final_type": "q12_keyframe",
                "measurement_key": key,
                "psnr": numeric(quality, "psnr"),
                "ssim": numeric(quality, "ssim"),
                "ms_ssim": numeric(quality, "ms_ssim"),
                "lpips": numeric(quality, "lpips"),
                "bitstream_bytes": int(numeric(payload, "bitstream_bytes", 0.0)) if payload else 0,
            }
        )

    groups = protocol_groups(protocol_rows)
    missing_schedule_groups = [sequence for sequence in groups if (SCHEDULE, sequence) not in schedule_payload]
    keyframe_bitstream_bytes = sum(int(numeric(schedule_payload[(SCHEDULE, sequence)], "bitstream_bytes")) for sequence in groups if (SCHEDULE, sequence) in schedule_payload)
    total_payload_bytes = keyframe_bitstream_bytes + METADATA_BYTES
    frame_count = len(protocol_rows)
    mean_psnr = mean(row["psnr"] for row in final_rows)
    mean_ssim = mean(row["ssim"] for row in final_rows)
    mean_ms_ssim = mean(row["ms_ssim"] for row in final_rows)
    mean_lpips = mean(row["lpips"] for row in final_rows)
    delta_psnr = mean_psnr - numeric(best_fixed, "mean_psnr") if mean_psnr is not None else ""
    delta_ssim = mean_ssim - numeric(best_fixed, "mean_ssim") if mean_ssim is not None else ""
    delta_ms_ssim = mean_ms_ssim - numeric(best_fixed, "mean_ms_ssim") if mean_ms_ssim is not None else ""
    delta_lpips = mean_lpips - numeric(best_fixed, "mean_lpips") if mean_lpips is not None else ""
    pass_flag = int(
        mean_psnr is not None
        and delta_psnr >= 1.0
        and delta_ssim >= 0.0
        and delta_ms_ssim >= 0.0
        and delta_lpips <= 0.0
    )
    total_mib = total_payload_bytes / MIB / frame_count if frame_count else 0.0
    summary_rows = [
        {
            "schedule": SCHEDULE,
            "schedule_family": SCHEDULE_FAMILY,
            "gap": GAP,
            "frame_count": len(final_rows),
            "keyframe_count": len(final_rows),
            "mean_psnr": mean_psnr,
            "p10_psnr": percentile((row["psnr"] for row in final_rows), 10),
            "mean_ssim": mean_ssim,
            "p10_ssim": percentile((row["ssim"] for row in final_rows), 10),
            "mean_ms_ssim": mean_ms_ssim,
            "mean_lpips": mean_lpips,
            "p90_lpips": percentile((row["lpips"] for row in final_rows), 90),
            "keyframe_bitstream_bytes": keyframe_bitstream_bytes,
            "metadata_bytes": METADATA_BYTES,
            "total_payload_bytes": total_payload_bytes,
            "total_mib_per_frame": total_mib,
            "best_fixed_reference": best_fixed["schedule"],
            "best_fixed_psnr": numeric(best_fixed, "mean_psnr"),
            "best_fixed_ssim": numeric(best_fixed, "mean_ssim"),
            "best_fixed_ms_ssim": numeric(best_fixed, "mean_ms_ssim"),
            "best_fixed_lpips": numeric(best_fixed, "mean_lpips"),
            "delta_rate_vs_best_fixed": total_mib - best_fixed_rate,
            "delta_psnr_vs_best_fixed": delta_psnr,
            "delta_ssim_vs_best_fixed": delta_ssim,
            "delta_ms_ssim_vs_best_fixed": delta_ms_ssim,
            "delta_lpips_vs_best_fixed": delta_lpips,
            "beats_best_fixed_by_1db_no_metric_regression": pass_flag,
        }
    ]
    validation_rows = [
        {"item": "protocol_frame_count", "expected": 1999, "actual": len(protocol_rows), "status": "ok" if len(protocol_rows) == 1999 else "error"},
        {"item": "unique_keyframe_payload_rows", "expected": len(protocol_rows), "actual": len(keyframe_payload), "status": "ok" if len(keyframe_payload) == len(protocol_rows) else "error"},
        {"item": "unique_keyframe_quality_rows", "expected": len(protocol_rows), "actual": len(keyframe_quality), "status": "ok" if len(keyframe_quality) == len(protocol_rows) else "error"},
        {"item": "schedule_keyframe_groups", "expected": len(groups), "actual": len(schedule_payload), "status": "ok" if len(schedule_payload) == len(groups) else "error"},
        {"item": "missing_payload_rows", "expected": 0, "actual": len(missing_payload), "status": "ok" if not missing_payload else "error"},
        {"item": "missing_quality_rows", "expected": 0, "actual": len(missing_quality), "status": "ok" if not missing_quality else "error"},
        {"item": "missing_schedule_groups", "expected": 0, "actual": len(missing_schedule_groups), "status": "ok" if not missing_schedule_groups else "error"},
        {"item": "final_quality_rows", "expected": len(protocol_rows), "actual": len(final_rows), "status": "ok" if len(final_rows) == len(protocol_rows) else "error"},
    ]
    return final_rows, summary_rows, validation_rows, best_fixed


def decision(complete, summary_row):
    if not complete:
        return "all_keyframe_q12_upper_bound_partial"
    if int(summary_row["beats_best_fixed_by_1db_no_metric_regression"]) == 1:
        return "all_keyframe_q12_has_target_headroom"
    if float(summary_row["delta_psnr_vs_best_fixed"]) > 0.0:
        return "all_keyframe_q12_improves_gap2_but_below_target_margin"
    return "all_keyframe_q12_no_quality_headroom_vs_gap2"


def write_report(summary_rows, validation_rows, package, path):
    row = summary_rows[0]
    lines = [
        "# Stage194 All-Keyframe Q12 Upper-Bound",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Complete: `{package['complete']}`.",
        f"- Reference best fixed baseline: `{row['best_fixed_reference']}`.",
        "",
        "## All-Keyframe RD-Quality",
        "",
        "| schedule | MiB/frame | keyframes | PSNR | SSIM | MS-SSIM | LPIPS | dPSNR vs gap2 | dLPIPS vs gap2 | +1dB pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {row['schedule']} | {float(row['total_mib_per_frame']):.12f} | {row['keyframe_count']} | "
        f"{float(row['mean_psnr']):.6f} | {float(row['mean_ssim']):.6f} | {float(row['mean_ms_ssim']):.6f} | {float(row['mean_lpips']):.6f} | "
        f"{float(row['delta_psnr_vs_best_fixed']):.6f} | {float(row['delta_lpips_vs_best_fixed']):.6f} | {row['beats_best_fixed_by_1db_no_metric_regression']} |",
        "",
        "## Validation",
        "",
        "| item | expected | actual | status |",
        "|---|---:|---:|---|",
    ]
    for check in validation_rows:
        lines.append(f"| {check['item']} | {check['expected']} | {check['actual']} | {check['status']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `uniform_gap1` is an upper-bound diagnostic, not an adaptive selector or a practical low-rate point.",
            "- If even all-frame q12 keyframes do not reach the +1 dB target over Stage192 `uniform_gap2`, the bottleneck is representation/codec quality rather than selector thresholding.",
            "- Rate uses measured schedule-packed q12 keyframe bitstreams plus fixed-gap metadata for consistency with Stage192.",
            "",
            "## Outputs",
            "",
            f"- Summary CSV: `{package['summary_csv']}`",
            f"- Final quality rows: `{package['final_quality_csv']}`",
            f"- Keyframe quality CSV: `{package['keyframe_quality_csv']}`",
            f"- Schedule-packed payload CSV: `{package['schedule_keyframe_payload_csv']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage192_final_rows", type=Path, default=DEFAULT_STAGE192_FINAL_ROWS)
    parser.add_argument("--stage192_keyframe_payload", type=Path, default=DEFAULT_STAGE192_KEYFRAME_PAYLOAD)
    parser.add_argument("--stage192_keyframe_quality", type=Path, default=DEFAULT_STAGE192_KEYFRAME_QUALITY)
    parser.add_argument("--stage192_rd_quality", type=Path, default=DEFAULT_STAGE192_RD_QUALITY)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--keyframe_bits", type=int, default=12)
    parser.add_argument("--keyframe_compression", choices=["none", "zlib"], default="none")
    parser.add_argument("--keyframe_payload_encoding", choices=["bitpack", "dtype"], default="bitpack")
    parser.add_argument("--max_payload_keyframes", type=int, default=0)
    parser.add_argument("--max_schedule_keyframe_groups", type=int, default=0)
    parser.add_argument("--max_quality_keyframes", type=int, default=0)
    parser.add_argument("--flush_every", type=int, default=200)
    parser.add_argument("--skip_payload_keyframes", action="store_true")
    parser.add_argument("--skip_schedule_keyframes", action="store_true")
    parser.add_argument("--skip_quality_keyframes", action="store_true")
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
    protocol_rows = build_protocol(read_csv(args.stage192_final_rows))
    write_csv(protocol_rows, p["protocol_csv"], PROTOCOL_FIELDS)
    seed_outputs(args, p, protocol_rows)

    lpips_error = ""
    ms_ssim_error = ""
    if not args.seed_only and not args.aggregate_only:
        dense_index = stage184.build_dense_index(args.dense_manifest, ["val"])
        measure_payloads(args, p, protocol_rows, dense_index)
        lpips_error, ms_ssim_error = measure_quality(args, p, protocol_rows, dense_index)

    final_rows, summary_rows, validation_rows, best_fixed = aggregate(args, p, protocol_rows)
    complete = all(row["status"] == "ok" for row in validation_rows)
    stage_decision = decision(complete, summary_rows[0])
    write_csv(final_rows, p["final_quality_csv"], FINAL_FIELDS)
    write_csv(summary_rows, p["summary_csv"], SUMMARY_FIELDS)
    write_csv(validation_rows, p["validation_csv"], VALIDATION_FIELDS)
    package = {
        "stage": 194,
        "status": "all_keyframe_q12_upper_bound_complete" if complete else "all_keyframe_q12_upper_bound_partial",
        "complete": complete,
        "decision": stage_decision,
        "reference_best_fixed": best_fixed["schedule"],
        "reference_best_fixed_psnr": numeric(best_fixed, "mean_psnr"),
        "summary_rows": summary_rows,
        "validation_rows": validation_rows,
        "lpips_error": lpips_error,
        "ms_ssim_error": ms_ssim_error,
        "protocol_csv": str(p["protocol_csv"].relative_to(REPO_ROOT)),
        "keyframe_payload_csv": str(p["keyframe_payload_csv"].relative_to(REPO_ROOT)),
        "schedule_keyframe_payload_csv": str(p["schedule_keyframe_payload_csv"].relative_to(REPO_ROOT)),
        "keyframe_quality_csv": str(p["keyframe_quality_csv"].relative_to(REPO_ROOT)),
        "final_quality_csv": str(p["final_quality_csv"].relative_to(REPO_ROOT)),
        "summary_csv": str(p["summary_csv"].relative_to(REPO_ROOT)),
        "validation_csv": str(p["validation_csv"].relative_to(REPO_ROOT)),
        "package_json": str(p["package_json"].relative_to(REPO_ROOT)),
        "report_md": str(p["report_md"].relative_to(REPO_ROOT)),
        "heavy_root": str(args.heavy_root),
    }
    p["package_json"].write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary_rows, validation_rows, package, p["report_md"])
    print(
        json.dumps(
            {
                "package": str(p["package_json"]),
                "status": package["status"],
                "decision": stage_decision,
                "complete": complete,
                "summary": summary_rows[0],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
