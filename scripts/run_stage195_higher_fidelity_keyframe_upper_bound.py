import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE192_ROOT = REPO_ROOT / "experiments/stage192_expanded_fixed_gap_measurement"
DEFAULT_STAGE194_ROOT = REPO_ROOT / "experiments/stage194_all_keyframe_q12_upper_bound"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage195_higher_fidelity_keyframe_upper_bound"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage195_higher_fidelity_keyframe_upper_bound")
DEFAULT_CHECKPOINT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/streamsplat.safetensors")

DEFAULT_PROTOCOL_ROWS = DEFAULT_STAGE194_ROOT / "stage194_all_keyframe_protocol_rows.csv"
DEFAULT_STAGE192_RD_QUALITY = DEFAULT_STAGE192_ROOT / "stage192_expanded_fixed_gap_rd_quality_points.csv"

Q16_SCHEDULE = "uniform_gap1_q16"
REPRESENTATIONS = [("q16_keyframe", 16), ("float_dense_anchor", None)]
METADATA_BYTES = 1
MIB = 1024.0 * 1024.0


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
import scripts.run_stage184_full_sequence_payload_measurement_execution as stage184  # noqa: E402
from scripts.run_stage153_middle_multimetric_badcase_eval import mean, percentile  # noqa: E402
from scripts.run_stage155_streamsplat_base_sideinfo_upper_bound import compute_metrics, render_static_anchor  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import load_rgb  # noqa: E402
from scripts.run_stage72_original_davis_baseline import DEFAULT_DAVIS_ROOT  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_DENSE_MANIFEST, load_anchor  # noqa: E402
from scripts.run_stage177_selector_fixed_gap_psnr_comparison import frame_rgb_path  # noqa: E402
import scripts.run_stage186_full_sequence_quality_validation as stage186  # noqa: E402


QUALITY_FIELDS = [
    "representation",
    "bits",
    "measurement_key",
    "sequence",
    "frame_index",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "status",
    "error",
]

SUMMARY_FIELDS = [
    "representation",
    "bits",
    "frame_count",
    "mean_psnr",
    "p10_psnr",
    "mean_ssim",
    "p10_ssim",
    "mean_ms_ssim",
    "mean_lpips",
    "p90_lpips",
    "q16_schedule_keyframe_bytes",
    "metadata_bytes",
    "total_payload_bytes",
    "total_mib_per_frame",
    "rate_scope",
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
        "q16_schedule_payload_csv": output_root / "stage195_q16_schedule_packed_keyframe_payload_measurements.csv",
        "quality_csv": output_root / "stage195_higher_fidelity_keyframe_quality_metrics.csv",
        "summary_csv": output_root / "stage195_higher_fidelity_keyframe_summary.csv",
        "validation_csv": output_root / "stage195_higher_fidelity_keyframe_validation.csv",
        "package_json": output_root / "stage195_higher_fidelity_keyframe_upper_bound_package.json",
        "report_md": output_root / "stage195_higher_fidelity_keyframe_upper_bound_report.md",
    }


def ok_schedule_groups(rows):
    return {(row["schedule"], row["sequence"]): row for row in rows if row.get("status") == "ok"}


def protocol_groups(protocol_rows):
    groups = {}
    for row in protocol_rows:
        groups.setdefault(row["sequence"], []).append(int(row["frame_index"]))
    return {sequence: sorted(frames) for sequence, frames in groups.items()}


def quality_key(row):
    return (row["representation"], row["measurement_key"])


def ok_quality_rows(rows):
    return {quality_key(row): row for row in rows if row.get("status") == "ok"}


def best_fixed_gap2(stage192_rd_quality_rows):
    for row in stage192_rd_quality_rows:
        if row["schedule"] == "uniform_gap2":
            return row
    fixed = [row for row in stage192_rd_quality_rows if row.get("schedule_family") == "fixed_gap"]
    return max(fixed, key=lambda row: numeric(row, "mean_psnr"))


def measure_q16_schedule_payload(args, p, protocol_rows, dense_index):
    schedule_rows = read_csv(p["q16_schedule_payload_csv"])
    groups = protocol_groups(protocol_rows)
    done = set(ok_schedule_groups(schedule_rows).keys())
    pending = [(sequence, frames) for sequence, frames in sorted(groups.items()) if (Q16_SCHEDULE, sequence) not in done]
    if args.max_schedule_keyframe_groups > 0:
        pending = pending[: args.max_schedule_keyframe_groups]
    cpu_device = torch.device("cpu")
    for idx, (sequence, frames) in enumerate(pending, 1):
        schedule_rows.append(stage184.measure_schedule_keyframe_group(Q16_SCHEDULE, sequence, frames, dense_index, args, cpu_device))
        if idx % max(1, args.flush_every) == 0 or idx == len(pending):
            write_csv(schedule_rows, p["q16_schedule_payload_csv"], stage184.SCHEDULE_KEYFRAME_FIELDS)
            print(json.dumps({"q16_schedule_groups_this_run": idx, "q16_schedule_groups_ok_total": len(ok_schedule_groups(schedule_rows))}), flush=True)


def measure_quality_row(proto, representation, bits, dense_index, opt, device, background, lpips_model, ms_ssim_module, args):
    try:
        sequence = proto["sequence"]
        frame_index = int(proto["frame_index"])
        dense_key = ("DAVIS", "val", sequence, frame_index)
        target_item, target_side = dense_index[dense_key]
        anchor = load_anchor(target_item, target_side, device, bits=bits, cache=None)
        render = render_static_anchor(anchor, background, opt)
        target_rgb = load_rgb(frame_rgb_path(args.davis_root, sequence, frame_index), opt.image_height, opt.image_width, device)
        metrics = compute_metrics(render, target_rgb, lpips_model, ms_ssim_module)
        return {
            "representation": representation,
            "bits": "float" if bits is None else int(bits),
            "measurement_key": proto["measurement_key"],
            "sequence": sequence,
            "frame_index": frame_index,
            **metrics,
            "status": "ok",
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "representation": representation,
            "bits": "float" if bits is None else int(bits),
            "measurement_key": proto.get("measurement_key", ""),
            "sequence": proto.get("sequence", ""),
            "frame_index": proto.get("frame_index", ""),
            "psnr": "",
            "ssim": "",
            "ms_ssim": "",
            "lpips": "",
            "status": "error",
            "error": repr(exc),
        }


def measure_quality(args, p, protocol_rows, dense_index):
    quality_rows = read_csv(p["quality_csv"])
    done = set(ok_quality_rows(quality_rows).keys())
    device = torch.device(args.device)
    opt = Options()
    opt.resume = str(args.checkpoint)
    opt.compile = False
    opt.input_frames = 1
    opt.output_frames = 3
    opt.epoch = 0
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    lpips_model, lpips_error, ms_ssim_module, ms_ssim_error = stage186.load_metric_modules(device, args.disable_lpips, args.disable_ms_ssim)

    for representation, bits in REPRESENTATIONS:
        pending = [row for row in protocol_rows if (representation, row["measurement_key"]) not in done]
        if args.max_quality_rows > 0:
            pending = pending[: args.max_quality_rows]
        for idx, proto in enumerate(pending, 1):
            quality_rows.append(measure_quality_row(proto, representation, bits, dense_index, opt, device, background, lpips_model, ms_ssim_module, args))
            if idx % max(1, args.flush_every) == 0 or idx == len(pending):
                write_csv(quality_rows, p["quality_csv"], QUALITY_FIELDS)
                print(json.dumps({"representation": representation, "quality_rows_this_run": idx, "quality_rows_ok_total": len(ok_quality_rows(quality_rows))}), flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return lpips_error, ms_ssim_error


def aggregate(args, p, protocol_rows):
    quality_by_key = ok_quality_rows(read_csv(p["quality_csv"]))
    schedule_payload = ok_schedule_groups(read_csv(p["q16_schedule_payload_csv"]))
    groups = protocol_groups(protocol_rows)
    best_fixed = best_fixed_gap2(read_csv(args.stage192_rd_quality))
    best_fixed_rate = numeric(best_fixed, "total_mib_per_frame")
    q16_schedule_bytes = sum(int(numeric(schedule_payload[(Q16_SCHEDULE, sequence)], "bitstream_bytes")) for sequence in groups if (Q16_SCHEDULE, sequence) in schedule_payload)
    q16_total_bytes = q16_schedule_bytes + METADATA_BYTES
    q16_rate = q16_total_bytes / MIB / len(protocol_rows) if len(schedule_payload) == len(groups) else ""
    summary_rows = []
    validation_rows = [
        {"item": "protocol_frame_count", "expected": 1999, "actual": len(protocol_rows), "status": "ok" if len(protocol_rows) == 1999 else "error"},
        {"item": "q16_schedule_keyframe_groups", "expected": len(groups), "actual": len(schedule_payload), "status": "ok" if len(schedule_payload) == len(groups) else "error"},
    ]

    for representation, bits in REPRESENTATIONS:
        rows = [quality_by_key[(representation, row["measurement_key"])] for row in protocol_rows if (representation, row["measurement_key"]) in quality_by_key]
        missing = len(protocol_rows) - len(rows)
        mean_psnr = mean(row["psnr"] for row in rows)
        mean_ssim = mean(row["ssim"] for row in rows)
        mean_ms_ssim = mean(row["ms_ssim"] for row in rows)
        mean_lpips = mean(row["lpips"] for row in rows)
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
        if representation == "q16_keyframe":
            total_bytes = q16_total_bytes if q16_rate != "" else ""
            rate = q16_rate
            delta_rate = q16_rate - best_fixed_rate if q16_rate != "" else ""
            rate_scope = "measured_schedule_packed_q16_keyframes_plus_fixed_gap_metadata"
            schedule_bytes = q16_schedule_bytes
            metadata = METADATA_BYTES
        else:
            total_bytes = ""
            rate = ""
            delta_rate = ""
            rate_scope = "quality_upper_bound_no_payload_rate"
            schedule_bytes = ""
            metadata = ""
        summary_rows.append(
            {
                "representation": representation,
                "bits": "float" if bits is None else int(bits),
                "frame_count": len(rows),
                "mean_psnr": mean_psnr,
                "p10_psnr": percentile((row["psnr"] for row in rows), 10),
                "mean_ssim": mean_ssim,
                "p10_ssim": percentile((row["ssim"] for row in rows), 10),
                "mean_ms_ssim": mean_ms_ssim,
                "mean_lpips": mean_lpips,
                "p90_lpips": percentile((row["lpips"] for row in rows), 90),
                "q16_schedule_keyframe_bytes": schedule_bytes,
                "metadata_bytes": metadata,
                "total_payload_bytes": total_bytes,
                "total_mib_per_frame": rate,
                "rate_scope": rate_scope,
                "best_fixed_reference": best_fixed["schedule"],
                "best_fixed_psnr": numeric(best_fixed, "mean_psnr"),
                "best_fixed_ssim": numeric(best_fixed, "mean_ssim"),
                "best_fixed_ms_ssim": numeric(best_fixed, "mean_ms_ssim"),
                "best_fixed_lpips": numeric(best_fixed, "mean_lpips"),
                "delta_rate_vs_best_fixed": delta_rate,
                "delta_psnr_vs_best_fixed": delta_psnr,
                "delta_ssim_vs_best_fixed": delta_ssim,
                "delta_ms_ssim_vs_best_fixed": delta_ms_ssim,
                "delta_lpips_vs_best_fixed": delta_lpips,
                "beats_best_fixed_by_1db_no_metric_regression": pass_flag,
            }
        )
        validation_rows.append({"item": f"{representation}_quality_rows", "expected": len(protocol_rows), "actual": len(rows), "status": "ok" if missing == 0 else "error"})
        validation_rows.append({"item": f"{representation}_missing_quality_rows", "expected": 0, "actual": missing, "status": "ok" if missing == 0 else "error"})
    return summary_rows, validation_rows, best_fixed


def decision(complete, summary_rows):
    if not complete:
        return "higher_fidelity_keyframe_upper_bound_partial"
    q16 = next(row for row in summary_rows if row["representation"] == "q16_keyframe")
    float_row = next(row for row in summary_rows if row["representation"] == "float_dense_anchor")
    if int(q16["beats_best_fixed_by_1db_no_metric_regression"]) == 1:
        return "q16_keyframe_has_target_headroom"
    if int(float_row["beats_best_fixed_by_1db_no_metric_regression"]) == 1:
        return "float_dense_anchor_has_quality_target_headroom"
    if float(float_row["delta_psnr_vs_best_fixed"]) > 0.0:
        return "higher_fidelity_keyframes_improve_gap2_but_below_target_margin"
    return "higher_fidelity_keyframes_no_quality_headroom_vs_gap2"


def fmt(value, digits=6):
    if value in (None, ""):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_report(summary_rows, validation_rows, package, path):
    lines = [
        "# Stage195 Higher-Fidelity Keyframe Upper-Bound",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Complete: `{package['complete']}`.",
        "",
        "## RD-Quality / Quality Upper Bounds",
        "",
        "| representation | MiB/frame | PSNR | SSIM | MS-SSIM | LPIPS | dPSNR vs gap2 | dLPIPS vs gap2 | +1dB pass | rate scope |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['representation']} | {fmt(row['total_mib_per_frame'], 12)} | {fmt(row['mean_psnr'])} | {fmt(row['mean_ssim'])} | "
            f"{fmt(row['mean_ms_ssim'])} | {fmt(row['mean_lpips'])} | {fmt(row['delta_psnr_vs_best_fixed'])} | "
            f"{fmt(row['delta_lpips_vs_best_fixed'])} | {row['beats_best_fixed_by_1db_no_metric_regression']} | {row['rate_scope']} |"
        )
    lines.extend(["", "## Validation", "", "| item | expected | actual | status |", "|---|---:|---:|---|"])
    for check in validation_rows:
        lines.append(f"| {check['item']} | {check['expected']} | {check['actual']} | {check['status']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- q16 includes measured schedule-packed keyframe rate; float dense-anchor is quality-only and has no deployable payload claim.",
            "- If float dense-anchor quality is still below the +1 dB target over Stage192 `uniform_gap2`, a stronger selector or higher quantization alone cannot satisfy the requested full-sequence gain.",
            "",
            "## Outputs",
            "",
            f"- Summary CSV: `{package['summary_csv']}`",
            f"- Quality CSV: `{package['quality_csv']}`",
            f"- q16 schedule payload CSV: `{package['q16_schedule_payload_csv']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol_rows", type=Path, default=DEFAULT_PROTOCOL_ROWS)
    parser.add_argument("--stage192_rd_quality", type=Path, default=DEFAULT_STAGE192_RD_QUALITY)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--keyframe_bits", type=int, default=16)
    parser.add_argument("--keyframe_compression", choices=["none", "zlib"], default="none")
    parser.add_argument("--keyframe_payload_encoding", choices=["bitpack", "dtype"], default="bitpack")
    parser.add_argument("--max_schedule_keyframe_groups", type=int, default=0)
    parser.add_argument("--max_quality_rows", type=int, default=0)
    parser.add_argument("--flush_every", type=int, default=200)
    parser.add_argument("--skip_q16_schedule_payload", action="store_true")
    parser.add_argument("--skip_quality", action="store_true")
    parser.add_argument("--disable_lpips", action="store_true")
    parser.add_argument("--disable_ms_ssim", action="store_true")
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
    protocol_rows = read_csv(args.protocol_rows)
    lpips_error = ""
    ms_ssim_error = ""
    if not args.aggregate_only:
        dense_index = stage184.build_dense_index(args.dense_manifest, ["val"])
        if not args.skip_q16_schedule_payload:
            measure_q16_schedule_payload(args, p, protocol_rows, dense_index)
        if not args.skip_quality:
            lpips_error, ms_ssim_error = measure_quality(args, p, protocol_rows, dense_index)
    summary_rows, validation_rows, best_fixed = aggregate(args, p, protocol_rows)
    complete = all(row["status"] == "ok" for row in validation_rows)
    stage_decision = decision(complete, summary_rows)
    write_csv(summary_rows, p["summary_csv"], SUMMARY_FIELDS)
    write_csv(validation_rows, p["validation_csv"], VALIDATION_FIELDS)
    package = {
        "stage": 195,
        "status": "higher_fidelity_keyframe_upper_bound_complete" if complete else "higher_fidelity_keyframe_upper_bound_partial",
        "complete": complete,
        "decision": stage_decision,
        "reference_best_fixed": best_fixed["schedule"],
        "reference_best_fixed_psnr": numeric(best_fixed, "mean_psnr"),
        "summary_rows": summary_rows,
        "validation_rows": validation_rows,
        "lpips_error": lpips_error,
        "ms_ssim_error": ms_ssim_error,
        "q16_schedule_payload_csv": str(p["q16_schedule_payload_csv"].relative_to(REPO_ROOT)),
        "quality_csv": str(p["quality_csv"].relative_to(REPO_ROOT)),
        "summary_csv": str(p["summary_csv"].relative_to(REPO_ROOT)),
        "validation_csv": str(p["validation_csv"].relative_to(REPO_ROOT)),
        "package_json": str(p["package_json"].relative_to(REPO_ROOT)),
        "report_md": str(p["report_md"].relative_to(REPO_ROOT)),
        "heavy_root": str(args.heavy_root),
    }
    p["package_json"].write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary_rows, validation_rows, package, p["report_md"])
    print(json.dumps({"package": str(p["package_json"]), "status": package["status"], "decision": stage_decision, "complete": complete, "summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
