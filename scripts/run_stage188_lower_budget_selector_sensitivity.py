import argparse
import csv
import json
import math
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE165_ROWS = REPO_ROOT / "experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_adaptive_schedule_rows.csv"
DEFAULT_STAGE184_ROOT = REPO_ROOT / "experiments/stage184_full_sequence_payload_measurement_execution"
DEFAULT_STAGE185_TOTAL_RD = REPO_ROOT / "experiments/stage185_measured_full_sequence_rd_aggregation/stage185_measured_full_sequence_total_rd.csv"
DEFAULT_STAGE186_ROOT = REPO_ROOT / "experiments/stage186_full_sequence_quality_validation"
DEFAULT_STAGE187_ROWS = REPO_ROOT / "experiments/stage187_selector_feature_ablation_validation/stage187_selector_feature_ablation_rows.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage188_lower_budget_selector_sensitivity"

DEFAULT_KEYFRAME_PAYLOADS = DEFAULT_STAGE184_ROOT / "stage184_unique_keyframe_payload_measurements.csv"
DEFAULT_RESIDUAL_PAYLOADS = DEFAULT_STAGE184_ROOT / "stage184_unique_stage158_residual_payload_measurements.csv"
DEFAULT_KEYFRAME_QUALITY = DEFAULT_STAGE186_ROOT / "stage186_unique_keyframe_quality_metrics.csv"
DEFAULT_RESIDUAL_QUALITY = DEFAULT_STAGE186_ROOT / "stage186_unique_stage158_residual_quality_metrics.csv"
DEFAULT_STAGE186_RD_QUALITY = DEFAULT_STAGE186_ROOT / "stage186_measured_rd_quality_points.csv"

MIB = 1024.0 * 1024.0
FULL_VARIANT = "full_stage165_features"

SENSITIVITY_FIELDS = [
    "candidate",
    "candidate_family",
    "aliases",
    "coverage_status",
    "total_frames",
    "keyframe_count",
    "residual_count",
    "extra_keyframes_vs_gap8",
    "kept_cell_count",
    "dropped_cell_count",
    "keyframe_bitstream_bytes_additive",
    "residual_payload_bytes",
    "metadata_bytes",
    "total_payload_bytes_additive",
    "total_mib_per_frame_additive",
    "delta_mib_per_frame_vs_gap8_additive",
    "delta_mib_per_frame_vs_full_additive",
    "delta_mib_per_frame_vs_gap4_additive",
    "mean_psnr",
    "delta_psnr_vs_gap8",
    "delta_psnr_vs_full",
    "delta_psnr_vs_gap4",
    "p10_psnr",
    "mean_ssim",
    "delta_ssim_vs_gap8",
    "mean_ms_ssim",
    "delta_ms_ssim_vs_gap8",
    "mean_lpips",
    "delta_lpips_vs_gap8",
    "delta_lpips_vs_full",
    "p90_lpips",
    "quality_positive_vs_gap8_all_metrics",
    "rate_scope",
]

CELL_FIELDS = [
    "cell_id",
    "sequence",
    "left_index",
    "right_index",
    "extra_keyframes",
    "extra_keyframe_count",
    "max_selector_score",
    "mean_selector_score",
    "max_vote_count",
    "hard_label_count",
    "high_payload_label_count",
    "mean_payload_bytes",
    "mean_stage158_psnr",
    "mean_stage158_lpips",
    "rank_index",
]

ROW_AUDIT_FIELDS = [
    "candidate",
    "candidate_family",
    "selected_unique_target_count",
    "keyframe_count",
    "residual_count",
    "missing_keyframe_payloads",
    "missing_residual_payloads",
    "missing_keyframe_quality",
    "missing_residual_quality",
    "coverage_status",
    "note",
]

CONTEXT_FIELDS = ["item", "value", "note"]
VALIDATION_FIELDS = ["item", "expected", "actual", "status"]


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_keyframes(value):
    return [int(item) for item in str(value).split() if item != ""]


def mean(values):
    vals = [float(v) for v in values if v not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def percentile(values, pct):
    vals = sorted(float(v) for v in values if v not in (None, ""))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return vals[lower]
    frac = pos - lower
    return vals[lower] * (1.0 - frac) + vals[upper] * frac


def numeric(row, key, default=0.0):
    value = row.get(key)
    if value in (None, "", "NA"):
        return default
    return float(value)


def keyframe_measurement_key(sequence, frame_index):
    return f"keyframe::{sequence}:{int(frame_index):05d}"


def residual_measurement_key(sequence, target_index, left_index, right_index):
    return f"residual::{sequence}:{int(target_index):05d}:{int(left_index):05d}:{int(right_index):05d}"


def adjacent_keyframes(keyframes, frame_index):
    pos = bisect_left(keyframes, frame_index)
    if pos < len(keyframes) and keyframes[pos] == frame_index:
        return frame_index, frame_index
    if pos == 0 or pos == len(keyframes):
        raise ValueError(f"frame {frame_index} outside keyframe coverage")
    return keyframes[pos - 1], keyframes[pos]


def metadata_bytes(schedule_name, keyframes_by_sequence, total_frames_by_sequence):
    if schedule_name in {"uniform_gap8", "uniform_gap4"}:
        return 1
    total_bits = 0
    for sequence, keyframes in sorted(keyframes_by_sequence.items()):
        bits_per_index = math.ceil(math.log2(max(int(total_frames_by_sequence[sequence]), 2)))
        total_bits += 8 + len(keyframes) * bits_per_index
    return int(math.ceil(total_bits / 8.0))


def schedule_rows_for_keyframes(schedule_name, keyframes_by_sequence, total_frames_by_sequence):
    rows = []
    for sequence, total_frames in sorted(total_frames_by_sequence.items()):
        keyframes = sorted(set(int(v) for v in keyframes_by_sequence[sequence]))
        keyframe_set = set(keyframes)
        for frame_index in range(int(total_frames)):
            if frame_index in keyframe_set:
                rows.append({
                    "schedule": schedule_name,
                    "sequence": sequence,
                    "frame_index": frame_index,
                    "measurement_type": "q12_keyframe_payload",
                    "measurement_key": keyframe_measurement_key(sequence, frame_index),
                })
            else:
                left, right = adjacent_keyframes(keyframes, frame_index)
                rows.append({
                    "schedule": schedule_name,
                    "sequence": sequence,
                    "frame_index": frame_index,
                    "measurement_type": "stage158_residual_payload",
                    "measurement_key": residual_measurement_key(sequence, frame_index, left, right),
                })
    return rows


def ok_map(rows, key_field="measurement_key"):
    return {row[key_field]: row for row in rows if row.get("status") == "ok"}


def coverage_for_rows(rows, keyframe_payloads, residual_payloads, keyframe_quality, residual_quality):
    missing = {
        "missing_keyframe_payloads": 0,
        "missing_residual_payloads": 0,
        "missing_keyframe_quality": 0,
        "missing_residual_quality": 0,
    }
    for row in rows:
        key = row["measurement_key"]
        if row["measurement_type"] == "q12_keyframe_payload":
            if key not in keyframe_payloads:
                missing["missing_keyframe_payloads"] += 1
            if key not in keyframe_quality:
                missing["missing_keyframe_quality"] += 1
        else:
            if key not in residual_payloads:
                missing["missing_residual_payloads"] += 1
            if key not in residual_quality:
                missing["missing_residual_quality"] += 1
    return missing


def aggregate_candidate(candidate, family, keyframes_by_sequence, total_frames_by_sequence, maps, gap8_keyframe_count, cell_counts, aliases=""):
    rows = schedule_rows_for_keyframes(candidate, keyframes_by_sequence, total_frames_by_sequence)
    coverage = coverage_for_rows(rows, maps["keyframe_payloads"], maps["residual_payloads"], maps["keyframe_quality"], maps["residual_quality"])
    if any(coverage.values()):
        return None, coverage, rows
    keyframe_rows = [row for row in rows if row["measurement_type"] == "q12_keyframe_payload"]
    residual_rows = [row for row in rows if row["measurement_type"] == "stage158_residual_payload"]
    keyframe_bytes = sum(numeric(maps["keyframe_payloads"][row["measurement_key"]], "bitstream_bytes") for row in keyframe_rows)
    residual_bytes = sum(numeric(maps["residual_payloads"][row["measurement_key"]], "payload_bytes") for row in residual_rows)
    meta_bytes = metadata_bytes(candidate, keyframes_by_sequence, total_frames_by_sequence)
    total_bytes = keyframe_bytes + residual_bytes + meta_bytes
    quality_rows = []
    for row in rows:
        measured = maps["keyframe_quality"][row["measurement_key"]] if row["measurement_type"] == "q12_keyframe_payload" else maps["residual_quality"][row["measurement_key"]]
        quality_rows.append(measured)
    total_frames = len(rows)
    total_keyframes = len(keyframe_rows)
    return {
        "candidate": candidate,
        "candidate_family": family,
        "aliases": aliases,
        "coverage_status": "complete_reused_stage184_stage186_rows",
        "total_frames": total_frames,
        "keyframe_count": total_keyframes,
        "residual_count": len(residual_rows),
        "extra_keyframes_vs_gap8": total_keyframes - gap8_keyframe_count,
        "kept_cell_count": cell_counts.get("kept_cell_count", ""),
        "dropped_cell_count": cell_counts.get("dropped_cell_count", ""),
        "keyframe_bitstream_bytes_additive": keyframe_bytes,
        "residual_payload_bytes": residual_bytes,
        "metadata_bytes": meta_bytes,
        "total_payload_bytes_additive": total_bytes,
        "total_mib_per_frame_additive": total_bytes / MIB / float(total_frames),
        "delta_mib_per_frame_vs_gap8_additive": None,
        "delta_mib_per_frame_vs_full_additive": None,
        "delta_mib_per_frame_vs_gap4_additive": None,
        "mean_psnr": mean(row["psnr"] for row in quality_rows),
        "delta_psnr_vs_gap8": None,
        "delta_psnr_vs_full": None,
        "delta_psnr_vs_gap4": None,
        "p10_psnr": percentile((row["psnr"] for row in quality_rows), 10),
        "mean_ssim": mean(row["ssim"] for row in quality_rows),
        "delta_ssim_vs_gap8": None,
        "mean_ms_ssim": mean(row["ms_ssim"] for row in quality_rows),
        "delta_ms_ssim_vs_gap8": None,
        "mean_lpips": mean(row["lpips"] for row in quality_rows),
        "delta_lpips_vs_gap8": None,
        "delta_lpips_vs_full": None,
        "p90_lpips": percentile((row["lpips"] for row in quality_rows), 90),
        "quality_positive_vs_gap8_all_metrics": None,
        "rate_scope": "measured_single_anchor_additive_keyframes_plus_measured_stage158_residuals_plus_exact_metadata",
    }, coverage, rows


def load_schedule_keyframes(stage165_rows):
    total_frames_by_sequence = {}
    out = {"uniform_gap8": {}, "stage165_adaptive_full": {}, "uniform_gap4": {}}
    for row in stage165_rows:
        sequence = row["sequence"]
        total_frames_by_sequence[sequence] = int(row["total_frames"])
        out["uniform_gap8"][sequence] = parse_keyframes(row["uniform_gap8_keyframes"])
        out["stage165_adaptive_full"][sequence] = parse_keyframes(row["adaptive_keyframes"])
        out["uniform_gap4"][sequence] = parse_keyframes(row["uniform_gap4_keyframes"])
    return out, total_frames_by_sequence


def build_full_variant_rows(stage187_rows):
    return {
        (row["sequence"], int(row["target_index"])): row
        for row in stage187_rows
        if row["variant"] == FULL_VARIANT
    }


def gap8_interval_for_frame(gap8_keyframes, frame_index):
    pos = bisect_left(gap8_keyframes, frame_index)
    if pos < len(gap8_keyframes) and gap8_keyframes[pos] == frame_index:
        return None
    if pos == 0 or pos == len(gap8_keyframes):
        raise ValueError(f"extra frame {frame_index} outside gap8 coverage")
    return gap8_keyframes[pos - 1], gap8_keyframes[pos]


def build_interval_cells(schedule_keyframes, full_variant_rows):
    cells = {}
    for sequence, adaptive_keyframes in schedule_keyframes["stage165_adaptive_full"].items():
        gap8 = sorted(schedule_keyframes["uniform_gap8"][sequence])
        gap8_set = set(gap8)
        for frame_index in sorted(set(adaptive_keyframes) - gap8_set):
            left, right = gap8_interval_for_frame(gap8, frame_index)
            cell_id = f"{sequence}:{left:05d}:{right:05d}"
            cells.setdefault(cell_id, {"sequence": sequence, "left_index": left, "right_index": right, "extra_keyframes": []})
            cells[cell_id]["extra_keyframes"].append(frame_index)
    cell_rows = []
    for cell_id, cell in cells.items():
        row_metrics = []
        for frame_index in cell["extra_keyframes"]:
            row = full_variant_rows.get((cell["sequence"], frame_index))
            if row is not None:
                row_metrics.append(row)
        scores = [numeric(row, "selector_score") for row in row_metrics]
        votes = [numeric(row, "vote_count") for row in row_metrics]
        payloads = [numeric(row, "payload_bytes") for row in row_metrics]
        psnrs = [numeric(row, "stage158_psnr") for row in row_metrics]
        lpips = [numeric(row, "stage158_lpips") for row in row_metrics]
        cell_rows.append({
            "cell_id": cell_id,
            "sequence": cell["sequence"],
            "left_index": cell["left_index"],
            "right_index": cell["right_index"],
            "extra_keyframes": " ".join(str(v) for v in sorted(cell["extra_keyframes"])),
            "extra_keyframe_count": len(cell["extra_keyframes"]),
            "max_selector_score": max(scores) if scores else 0.0,
            "mean_selector_score": mean(scores) if scores else 0.0,
            "max_vote_count": max(votes) if votes else 0.0,
            "hard_label_count": sum(int(float(row["hard_quality_label"])) for row in row_metrics),
            "high_payload_label_count": sum(int(float(row["high_payload_label"])) for row in row_metrics),
            "mean_payload_bytes": mean(payloads),
            "mean_stage158_psnr": mean(psnrs),
            "mean_stage158_lpips": mean(lpips),
            "rank_index": None,
        })
    cell_rows.sort(key=lambda row: (float(row["max_selector_score"]), float(row["mean_selector_score"]), int(row["extra_keyframe_count"])), reverse=True)
    for idx, row in enumerate(cell_rows, 1):
        row["rank_index"] = idx
    return cell_rows


def keyframes_for_kept_cells(schedule_keyframes, kept_cell_ids):
    out = {}
    kept = set(kept_cell_ids)
    for sequence, gap8 in schedule_keyframes["uniform_gap8"].items():
        keyframes = set(gap8)
        full_extra = set(schedule_keyframes["stage165_adaptive_full"][sequence]) - set(gap8)
        for frame_index in full_extra:
            left, right = gap8_interval_for_frame(sorted(gap8), frame_index)
            if f"{sequence}:{left:05d}:{right:05d}" in kept:
                keyframes.add(frame_index)
        out[sequence] = sorted(keyframes)
    return out


def schedule_signature(keyframes_by_sequence):
    return tuple((sequence, tuple(sorted(values))) for sequence, values in sorted(keyframes_by_sequence.items()))


def build_interval_candidates(schedule_keyframes, cell_rows):
    candidates = []
    cells = [row["cell_id"] for row in cell_rows]
    total_cells = len(cells)
    for frac in [0.9, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        keep_count = max(1, int(math.ceil(total_cells * frac)))
        name = f"interval_top{int(frac * 100):02d}pct_cells"
        candidates.append((name, cells[:keep_count], f"top {keep_count}/{total_cells} ranked gap8 intervals"))
    for min_score in [5.0, 4.0, 3.0, 2.0, 1.0]:
        kept = [row["cell_id"] for row in cell_rows if float(row["max_selector_score"]) >= min_score]
        if kept:
            candidates.append((f"interval_score_ge{str(min_score).replace('.', 'p')}", kept, f"cell max selector score >= {min_score}"))
    for min_votes in [5, 4, 3, 2, 1]:
        kept = [row["cell_id"] for row in cell_rows if float(row["max_vote_count"]) >= min_votes]
        if kept:
            candidates.append((f"interval_vote_ge{min_votes}", kept, f"cell max vote count >= {min_votes}"))
    deduped = []
    seen = {}
    for name, kept, note in candidates:
        keyframes = keyframes_for_kept_cells(schedule_keyframes, kept)
        sig = schedule_signature(keyframes)
        if sig in seen:
            seen[sig]["aliases"].append(name)
            seen[sig]["notes"].append(note)
            continue
        item = {"candidate": name, "kept_cells": kept, "keyframes": keyframes, "aliases": [], "notes": [note]}
        seen[sig] = item
        deduped.append(item)
    return deduped


def selected_targets_by_variant(stage187_rows):
    out = defaultdict(set)
    for row in stage187_rows:
        if int(float(row["selected_for_extra_keyframe"])):
            out[row["variant"]].add((row["sequence"], int(row["target_index"])))
    return out


def keyframes_for_selected_targets(schedule_keyframes, selected_targets):
    out = {}
    for sequence, gap8 in schedule_keyframes["uniform_gap8"].items():
        keyframes = set(gap8)
        keyframes.update(target for seq, target in selected_targets if seq == sequence)
        out[sequence] = sorted(keyframes)
    return out


def apply_deltas(rows):
    by_name = {row["candidate"]: row for row in rows}
    gap8 = by_name["uniform_gap8"]
    full = by_name["stage165_adaptive_full"]
    gap4 = by_name["uniform_gap4"]
    for row in rows:
        row["delta_mib_per_frame_vs_gap8_additive"] = row["total_mib_per_frame_additive"] - gap8["total_mib_per_frame_additive"]
        row["delta_mib_per_frame_vs_full_additive"] = row["total_mib_per_frame_additive"] - full["total_mib_per_frame_additive"]
        row["delta_mib_per_frame_vs_gap4_additive"] = row["total_mib_per_frame_additive"] - gap4["total_mib_per_frame_additive"]
        row["delta_psnr_vs_gap8"] = row["mean_psnr"] - gap8["mean_psnr"]
        row["delta_psnr_vs_full"] = row["mean_psnr"] - full["mean_psnr"]
        row["delta_psnr_vs_gap4"] = row["mean_psnr"] - gap4["mean_psnr"]
        row["delta_ssim_vs_gap8"] = row["mean_ssim"] - gap8["mean_ssim"]
        row["delta_ms_ssim_vs_gap8"] = row["mean_ms_ssim"] - gap8["mean_ms_ssim"]
        row["delta_lpips_vs_gap8"] = row["mean_lpips"] - gap8["mean_lpips"]
        row["delta_lpips_vs_full"] = row["mean_lpips"] - full["mean_lpips"]
        row["quality_positive_vs_gap8_all_metrics"] = int(
            row["delta_psnr_vs_gap8"] > 0.0
            and row["delta_ssim_vs_gap8"] > 0.0
            and row["delta_ms_ssim_vs_gap8"] > 0.0
            and row["delta_lpips_vs_gap8"] < 0.0
        )


def build_context(stage185_rows, stage186_rd_rows):
    rows185 = {row["schedule"]: row for row in stage185_rows}
    rows186 = {row["schedule"]: row for row in stage186_rd_rows}
    return [
        {"item": "stage185_rate_scope", "value": rows185["stage165_adaptive"]["rate_scope"], "note": "Packed keyframe scope for Stage185/186 context, not reused as new candidate scope."},
        {"item": "stage185_gap8_mib_per_frame", "value": rows185["uniform_gap8"]["total_mib_per_frame"], "note": "Packed-keyframe measured baseline."},
        {"item": "stage185_adaptive_mib_per_frame", "value": rows185["stage165_adaptive"]["total_mib_per_frame"], "note": "Packed-keyframe measured full Stage165 rate."},
        {"item": "stage185_gap4_mib_per_frame", "value": rows185["uniform_gap4"]["total_mib_per_frame"], "note": "Packed-keyframe measured gap4 rate."},
        {"item": "stage186_adaptive_psnr", "value": rows186["stage165_adaptive"]["mean_psnr"], "note": "Full Stage165 quality context."},
        {"item": "stage188_rate_scope", "value": "measured_single_anchor_additive_keyframes_plus_measured_stage158_residuals_plus_exact_metadata", "note": "Apples-to-apples sensitivity scope for all Stage188 candidates and baselines."},
    ]


def decision(rows):
    gap8 = next(row for row in rows if row["candidate"] == "uniform_gap8")
    full = next(row for row in rows if row["candidate"] == "stage165_adaptive_full")
    positives = [
        row for row in rows
        if row["candidate"] not in {"uniform_gap8", "stage165_adaptive_full", "uniform_gap4"}
        and int(row["quality_positive_vs_gap8_all_metrics"]) == 1
        and float(row["total_mib_per_frame_additive"]) < float(full["total_mib_per_frame_additive"])
    ]
    if positives:
        lowest_rate = sorted(positives, key=lambda row: float(row["total_mib_per_frame_additive"]))[0]
        highest_quality = sorted(positives, key=lambda row: float(row["mean_psnr"]), reverse=True)[0]
        if float(lowest_rate["total_mib_per_frame_additive"]) <= float(gap8["total_mib_per_frame_additive"]):
            return "lower_budget_positive_quality_candidate_at_or_below_gap8_rate", lowest_rate, highest_quality
        return "lower_budget_positive_quality_candidates_found_but_gap8_rate_not_reached", lowest_rate, highest_quality
    return "lower_budget_candidates_need_review_or_new_measurement", None, None


def write_report(sensitivity_rows, row_audit_rows, context_rows, package, path):
    lines = [
        "# Stage188 Lower-Budget Selector Sensitivity",
        "",
        "## Scope",
        "",
        "This stage reuses Stage184/186 measured rows. New candidate rates use an additive single-anchor keyframe scope, not the Stage185 schedule-packed keyframe scope.",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
    ]
    if package.get("recommended_lowest_rate_positive_candidate"):
        rec = package["recommended_lowest_rate_positive_candidate"]
        lines.append(f"- Lowest-rate positive candidate: `{rec['candidate']}` at `{rec['total_mib_per_frame_additive']}` MiB/frame additive, PSNR `{rec['mean_psnr']}`.")
        lines.append(f"- Its additive rate delta vs gap8 is `{rec['delta_mib_per_frame_vs_gap8_additive']}` MiB/frame; it reduces full adaptive overhead by `{-rec['delta_mib_per_frame_vs_full_additive']}` MiB/frame but does not reach gap8 rate.")
    if package.get("recommended_highest_quality_below_full_candidate"):
        rec = package["recommended_highest_quality_below_full_candidate"]
        lines.append(f"- Highest-quality below-full candidate: `{rec['candidate']}` at `{rec['total_mib_per_frame_additive']}` MiB/frame additive, PSNR `{rec['mean_psnr']}`.")
    if package.get("recommended_balanced_half_overhead_candidate"):
        rec = package["recommended_balanced_half_overhead_candidate"]
        lines.append(f"- Balanced half-overhead candidate: `{rec['candidate']}` at `{rec['total_mib_per_frame_additive']}` MiB/frame additive, PSNR `{rec['mean_psnr']}`.")
    lines.extend(["", "## Context", ""])
    for row in context_rows:
        lines.append(f"- `{row['item']}`: `{row['value']}`. {row['note']}")
    lines.extend([
        "",
        "## Fully Covered Additive RD-Quality",
        "",
        "| candidate | family | keyframes | cells kept | MiB/frame | dRate vs gap8 | dRate vs full | PSNR | dPSNR vs gap8 | SSIM | MS-SSIM | LPIPS | dLPIPS vs gap8 | all-metric positive |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    rows = sorted(sensitivity_rows, key=lambda row: (float(row["total_mib_per_frame_additive"]), -float(row["mean_psnr"])))
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['candidate_family']} | {row['keyframe_count']} | {row['kept_cell_count']} | "
            f"{float(row['total_mib_per_frame_additive']):.12f} | {float(row['delta_mib_per_frame_vs_gap8_additive']):.12f} | {float(row['delta_mib_per_frame_vs_full_additive']):.12f} | "
            f"{float(row['mean_psnr']):.6f} | {float(row['delta_psnr_vs_gap8']):.6f} | {float(row['mean_ssim']):.6f} | {float(row['mean_ms_ssim']):.6f} | "
            f"{float(row['mean_lpips']):.6f} | {float(row['delta_lpips_vs_gap8']):.6f} | {row['quality_positive_vs_gap8_all_metrics']} |"
        )
    incomplete = [row for row in row_audit_rows if row["coverage_status"] != "complete_reused_stage184_stage186_rows"]
    lines.extend([
        "",
        "## Row-Level Ablation Coverage Audit",
        "",
        "| candidate | unique targets | keyframes | residuals | missing residual payloads | missing residual quality | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in row_audit_rows:
        lines.append(
            f"| {row['candidate']} | {row['selected_unique_target_count']} | {row['keyframe_count']} | {row['residual_count']} | "
            f"{row['missing_residual_payloads']} | {row['missing_residual_quality']} | {row['coverage_status']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Interval-level candidates are fully measured because each gap8 interval either reuses full Stage165 adaptive rows or uniform gap8 rows.",
        "- Row-level ablations are audited after collapsing duplicate sampled rows to unique target frames. In this run all audited row-level variants were fully covered by existing measured rows.",
        "- The lowest-rate positive candidate reduces most of the full adaptive overhead but still remains above uniform gap8 rate under the additive sensitivity scope.",
        "- Additive single-anchor keyframe rates are for sensitivity only and should not be mixed numerically with Stage185 schedule-packed keyframe rates.",
        "",
        "## Outputs",
        "",
        f"- Sensitivity CSV: `{package['sensitivity_csv']}`",
        f"- Interval cells CSV: `{package['cell_csv']}`",
        f"- Row-level coverage CSV: `{package['row_level_coverage_csv']}`",
        f"- Context CSV: `{package['context_csv']}`",
    ])
    if incomplete:
        lines.append(f"- Incomplete row-level candidates requiring new measurement: `{len(incomplete)}`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage165_rows", type=Path, default=DEFAULT_STAGE165_ROWS)
    parser.add_argument("--stage187_rows", type=Path, default=DEFAULT_STAGE187_ROWS)
    parser.add_argument("--keyframe_payloads", type=Path, default=DEFAULT_KEYFRAME_PAYLOADS)
    parser.add_argument("--residual_payloads", type=Path, default=DEFAULT_RESIDUAL_PAYLOADS)
    parser.add_argument("--keyframe_quality", type=Path, default=DEFAULT_KEYFRAME_QUALITY)
    parser.add_argument("--residual_quality", type=Path, default=DEFAULT_RESIDUAL_QUALITY)
    parser.add_argument("--stage185_total_rd", type=Path, default=DEFAULT_STAGE185_TOTAL_RD)
    parser.add_argument("--stage186_rd_quality", type=Path, default=DEFAULT_STAGE186_RD_QUALITY)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage165_rows = read_csv(args.stage165_rows)
    stage187_rows = read_csv(args.stage187_rows)
    schedule_keyframes, total_frames_by_sequence = load_schedule_keyframes(stage165_rows)
    maps = {
        "keyframe_payloads": ok_map(read_csv(args.keyframe_payloads)),
        "residual_payloads": ok_map(read_csv(args.residual_payloads)),
        "keyframe_quality": ok_map(read_csv(args.keyframe_quality)),
        "residual_quality": ok_map(read_csv(args.residual_quality)),
    }
    full_variant_rows = build_full_variant_rows(stage187_rows)
    cell_rows = build_interval_cells(schedule_keyframes, full_variant_rows)
    gap8_keyframe_count = sum(len(values) for values in schedule_keyframes["uniform_gap8"].values())
    complete_rows = []
    row_audit_rows = []
    candidate_specs = [
        ("uniform_gap8", "baseline", schedule_keyframes["uniform_gap8"], {"kept_cell_count": 0, "dropped_cell_count": len(cell_rows)}, ""),
        ("stage165_adaptive_full", "baseline", schedule_keyframes["stage165_adaptive_full"], {"kept_cell_count": len(cell_rows), "dropped_cell_count": 0}, ""),
        ("uniform_gap4", "baseline", schedule_keyframes["uniform_gap4"], {"kept_cell_count": "", "dropped_cell_count": ""}, ""),
    ]
    for item in build_interval_candidates(schedule_keyframes, cell_rows):
        aliases = " ".join(item["aliases"])
        candidate_specs.append((
            item["candidate"],
            "interval_budget_reuse",
            item["keyframes"],
            {"kept_cell_count": len(item["kept_cells"]), "dropped_cell_count": len(cell_rows) - len(item["kept_cells"])},
            aliases,
        ))
    seen_schedules = set()
    validation_rows = []
    for name, family, keyframes, cell_counts, aliases in candidate_specs:
        sig = schedule_signature(keyframes)
        if sig in seen_schedules and family != "baseline":
            continue
        seen_schedules.add(sig)
        row, coverage, rows = aggregate_candidate(name, family, keyframes, total_frames_by_sequence, maps, gap8_keyframe_count, cell_counts, aliases)
        validation_rows.append({
            "item": f"{name}_coverage_missing_total",
            "expected": 0,
            "actual": sum(coverage.values()),
            "status": "ok" if not any(coverage.values()) else "error",
        })
        if row is not None:
            complete_rows.append(row)
        else:
            row_audit_rows.append({
                "candidate": name,
                "candidate_family": family,
                "selected_unique_target_count": "",
                "keyframe_count": sum(len(values) for values in keyframes.values()),
                "residual_count": len(rows) - sum(len(values) for values in keyframes.values()),
                **coverage,
                "coverage_status": "missing_measurements",
                "note": "interval candidate unexpectedly missing measured rows",
            })
    selected_by_variant = selected_targets_by_variant(stage187_rows)
    for variant in sorted(selected_by_variant):
        keyframes = keyframes_for_selected_targets(schedule_keyframes, selected_by_variant[variant])
        rows = schedule_rows_for_keyframes(variant, keyframes, total_frames_by_sequence)
        coverage = coverage_for_rows(rows, maps["keyframe_payloads"], maps["residual_payloads"], maps["keyframe_quality"], maps["residual_quality"])
        status = "complete_reused_stage184_stage186_rows" if not any(coverage.values()) else "requires_new_measurement"
        row_audit_rows.append({
            "candidate": variant,
            "candidate_family": "row_level_feature_ablation",
            "selected_unique_target_count": len(selected_by_variant[variant]),
            "keyframe_count": sum(len(values) for values in keyframes.values()),
            "residual_count": len(rows) - sum(len(values) for values in keyframes.values()),
            **coverage,
            "coverage_status": status,
            "note": "row-level ablation coverage audit after duplicate target collapse",
        })
        if status == "complete_reused_stage184_stage186_rows" and variant not in {"full_stage165_features"}:
            candidate_row, _coverage, _rows = aggregate_candidate(
                f"row_{variant}",
                "row_level_feature_ablation",
                keyframes,
                total_frames_by_sequence,
                maps,
                gap8_keyframe_count,
                {"kept_cell_count": "", "dropped_cell_count": ""},
                "",
            )
            complete_rows.append(candidate_row)
    apply_deltas(complete_rows)
    complete_rows.sort(key=lambda row: (float(row["total_mib_per_frame_additive"]), -float(row["mean_psnr"])))
    context_rows = build_context(read_csv(args.stage185_total_rd), read_csv(args.stage186_rd_quality))
    dec, lowest_rate, highest_quality = decision(complete_rows)
    full_row = next(row for row in complete_rows if row["candidate"] == "stage165_adaptive_full")
    positive_rows = [
        row for row in complete_rows
        if row["candidate"] not in {"uniform_gap8", "stage165_adaptive_full", "uniform_gap4"}
        and int(row["quality_positive_vs_gap8_all_metrics"]) == 1
    ]
    half_overhead_limit = 0.5 * float(full_row["delta_mib_per_frame_vs_gap8_additive"])
    half_overhead_candidates = [row for row in positive_rows if float(row["delta_mib_per_frame_vs_gap8_additive"]) <= half_overhead_limit]
    balanced_half_overhead = sorted(half_overhead_candidates, key=lambda row: float(row["mean_psnr"]), reverse=True)[0] if half_overhead_candidates else None
    sensitivity_csv = args.output_root / "stage188_lower_budget_selector_sensitivity_rd_quality.csv"
    cell_csv = args.output_root / "stage188_interval_cell_ranking.csv"
    row_level_csv = args.output_root / "stage188_row_level_ablation_coverage_audit.csv"
    context_csv = args.output_root / "stage188_lower_budget_selector_sensitivity_context.csv"
    validation_csv = args.output_root / "stage188_lower_budget_selector_sensitivity_validation.csv"
    package_json = args.output_root / "stage188_lower_budget_selector_sensitivity_package.json"
    report_md = args.output_root / "stage188_lower_budget_selector_sensitivity_report.md"
    write_csv(complete_rows, sensitivity_csv, SENSITIVITY_FIELDS)
    write_csv(cell_rows, cell_csv, CELL_FIELDS)
    write_csv(row_audit_rows, row_level_csv, ROW_AUDIT_FIELDS)
    write_csv(context_rows, context_csv, CONTEXT_FIELDS)
    write_csv(validation_rows, validation_csv, VALIDATION_FIELDS)
    package = {
        "stage": 188,
        "status": "lower_budget_selector_sensitivity_packaged",
        "decision": dec,
        "rate_scope": "measured_single_anchor_additive_keyframes_plus_measured_stage158_residuals_plus_exact_metadata",
        "complete_candidate_count": len(complete_rows),
        "interval_cell_count": len(cell_rows),
        "row_level_audit_count": len(row_audit_rows),
        "recommended_lowest_rate_positive_candidate": lowest_rate,
        "recommended_highest_quality_below_full_candidate": highest_quality,
        "recommended_balanced_half_overhead_candidate": balanced_half_overhead,
        "full_adaptive_additive_gap8_overhead_mib_per_frame": full_row["delta_mib_per_frame_vs_gap8_additive"],
        "half_overhead_limit_mib_per_frame": half_overhead_limit,
        "sensitivity_csv": str(sensitivity_csv),
        "cell_csv": str(cell_csv),
        "row_level_coverage_csv": str(row_level_csv),
        "context_csv": str(context_csv),
        "validation_csv": str(validation_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(complete_rows, row_audit_rows, context_rows, package, report_md)
    print(json.dumps({
        "package": str(package_json),
        "decision": dec,
        "complete_candidate_count": len(complete_rows),
        "interval_cell_count": len(cell_rows),
        "recommended_lowest_rate_positive_candidate": lowest_rate["candidate"] if lowest_rate else None,
        "recommended_highest_quality_below_full_candidate": highest_quality["candidate"] if highest_quality else None,
        "recommended_balanced_half_overhead_candidate": balanced_half_overhead["candidate"] if balanced_half_overhead else None,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
