import argparse
import csv
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE165_ROWS = REPO_ROOT / "experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_adaptive_schedule_rows.csv"
DEFAULT_STAGE184_ROOT = REPO_ROOT / "experiments/stage184_full_sequence_payload_measurement_execution"
DEFAULT_STAGE186_ROOT = REPO_ROOT / "experiments/stage186_full_sequence_quality_validation"
DEFAULT_STAGE187_ROWS = REPO_ROOT / "experiments/stage187_selector_feature_ablation_validation/stage187_selector_feature_ablation_rows.csv"
DEFAULT_STAGE188_ROOT = REPO_ROOT / "experiments/stage188_lower_budget_selector_sensitivity"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage189_failure_case_analysis"

DEFAULT_KEYFRAME_PAYLOADS = DEFAULT_STAGE184_ROOT / "stage184_unique_keyframe_payload_measurements.csv"
DEFAULT_RESIDUAL_PAYLOADS = DEFAULT_STAGE184_ROOT / "stage184_unique_stage158_residual_payload_measurements.csv"
DEFAULT_KEYFRAME_QUALITY = DEFAULT_STAGE186_ROOT / "stage186_unique_keyframe_quality_metrics.csv"
DEFAULT_RESIDUAL_QUALITY = DEFAULT_STAGE186_ROOT / "stage186_unique_stage158_residual_quality_metrics.csv"
DEFAULT_STAGE186_FINAL_ROWS = DEFAULT_STAGE186_ROOT / "stage186_full_sequence_quality_by_schedule.csv"
DEFAULT_STAGE188_RD = DEFAULT_STAGE188_ROOT / "stage188_lower_budget_selector_sensitivity_rd_quality.csv"
DEFAULT_STAGE188_CELLS = DEFAULT_STAGE188_ROOT / "stage188_interval_cell_ranking.csv"

sys_path_added = False
try:
    from scripts.run_stage188_lower_budget_selector_sensitivity import (  # noqa: E402
        build_full_variant_rows,
        build_interval_cells,
        keyframes_for_kept_cells,
        load_schedule_keyframes,
        ok_map,
        read_csv,
        schedule_rows_for_keyframes,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    sys_path_added = True
    from scripts.run_stage188_lower_budget_selector_sensitivity import (  # noqa: E402
        build_full_variant_rows,
        build_interval_cells,
        keyframes_for_kept_cells,
        load_schedule_keyframes,
        ok_map,
        read_csv,
        schedule_rows_for_keyframes,
    )


PROMOTION_FIELDS = [
    "sequence",
    "frame_index",
    "cell_id",
    "selector_score",
    "vote_count",
    "hard_quality_label",
    "high_payload_label",
    "gap8_psnr",
    "adaptive_psnr",
    "delta_psnr_adaptive_vs_gap8",
    "gap8_ssim",
    "adaptive_ssim",
    "delta_ssim_adaptive_vs_gap8",
    "gap8_ms_ssim",
    "adaptive_ms_ssim",
    "delta_ms_ssim_adaptive_vs_gap8",
    "gap8_lpips",
    "adaptive_lpips",
    "delta_lpips_adaptive_vs_gap8",
    "gap8_residual_payload_bytes",
    "adaptive_keyframe_bitstream_bytes",
    "local_payload_delta_bytes",
    "false_positive_risk_flag",
    "false_positive_reason",
]

RESIDUAL_RISK_FIELDS = [
    "sequence",
    "frame_index",
    "adaptive_psnr",
    "adaptive_ssim",
    "adaptive_ms_ssim",
    "adaptive_lpips",
    "adaptive_payload_bytes",
    "gap8_psnr",
    "gap8_lpips",
    "gap4_psnr",
    "gap4_lpips",
    "delta_psnr_gap4_vs_adaptive",
    "delta_lpips_gap4_vs_adaptive",
    "low_psnr_flag",
    "high_lpips_flag",
    "high_payload_flag",
    "risk_score",
]

CANDIDATE_DROP_FIELDS = [
    "candidate",
    "sequence",
    "frame_index",
    "full_type",
    "candidate_type",
    "full_psnr",
    "candidate_psnr",
    "delta_psnr_candidate_vs_full",
    "gap8_psnr",
    "delta_psnr_candidate_vs_gap8",
    "full_lpips",
    "candidate_lpips",
    "delta_lpips_candidate_vs_full",
    "gap8_lpips",
    "delta_lpips_candidate_vs_gap8",
    "candidate_payload_bytes",
]

CANDIDATE_SUMMARY_FIELDS = [
    "candidate",
    "keyframe_count",
    "rate_mib_per_frame_additive",
    "delta_rate_vs_gap8",
    "delta_rate_vs_full",
    "mean_psnr",
    "delta_psnr_vs_gap8",
    "delta_psnr_vs_full",
    "mean_lpips",
    "delta_lpips_vs_gap8",
    "changed_frame_count_vs_full",
    "mean_changed_delta_psnr_vs_full",
    "worst_changed_delta_psnr_vs_full",
    "mean_changed_delta_lpips_vs_full",
]

SEQUENCE_FIELDS = [
    "schedule",
    "sequence",
    "frame_count",
    "keyframe_count",
    "mean_psnr",
    "p10_psnr",
    "mean_lpips",
    "p90_lpips",
    "mean_payload_bytes",
    "delta_psnr_vs_gap8",
    "delta_lpips_vs_gap8",
]

HOTSPOT_FIELDS = [
    "sequence",
    "promotion_count",
    "promotion_rate_risk_count",
    "residual_risk_count",
    "low_psnr_count",
    "high_lpips_count",
    "high_payload_count",
    "mean_residual_risk_score",
    "max_residual_risk_score",
    "worst_residual_frame_index",
]

CONTEXT_FIELDS = ["item", "value", "note"]


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
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def final_key(row):
    return row["schedule"], row["sequence"], int(row["frame_index"])


def by_schedule_frame(rows):
    return {final_key(row): row for row in rows}


def by_frame(rows):
    return {(row["sequence"], int(row["frame_index"])): row for row in rows}


def keyframe_key(sequence, frame_index):
    return f"keyframe::{sequence}:{int(frame_index):05d}"


def cell_for_extra(cell_rows):
    out = {}
    for row in cell_rows:
        for item in str(row["extra_keyframes"]).split():
            out[(row["sequence"], int(item))] = row["cell_id"]
    return out


def variant_row_map(stage187_rows, variant="full_stage165_features"):
    out = {}
    for row in stage187_rows:
        if row["variant"] == variant:
            out[(row["sequence"], int(row["target_index"]))] = row
    return out


def build_candidate_final_rows(candidate, keyframes_by_sequence, total_frames_by_sequence, keyframe_quality, residual_quality, residual_payloads):
    out = []
    rows = schedule_rows_for_keyframes(candidate, keyframes_by_sequence, total_frames_by_sequence)
    for row in rows:
        key = row["measurement_key"]
        if row["measurement_type"] == "q12_keyframe_payload":
            measured = keyframe_quality[key]
            final_type = "q12_keyframe"
            payload = 0
        else:
            measured = residual_quality[key]
            final_type = "stage158_residual_recovery"
            payload = int(numeric(residual_payloads[key], "payload_bytes"))
        out.append({
            "schedule": candidate,
            "sequence": row["sequence"],
            "frame_index": int(row["frame_index"]),
            "final_type": final_type,
            "measurement_key": key,
            "psnr": numeric(measured, "psnr"),
            "ssim": numeric(measured, "ssim"),
            "ms_ssim": numeric(measured, "ms_ssim"),
            "lpips": numeric(measured, "lpips"),
            "payload_bytes": payload,
        })
    return out


def kept_cells_for_candidate(candidate, cell_rows):
    if candidate == "interval_top10pct_cells":
        return [row["cell_id"] for row in cell_rows[: max(1, math.ceil(len(cell_rows) * 0.1))]]
    if candidate == "interval_top90pct_cells":
        return [row["cell_id"] for row in cell_rows[: max(1, math.ceil(len(cell_rows) * 0.9))]]
    if candidate == "interval_score_ge4p0":
        return [row["cell_id"] for row in cell_rows if numeric(row, "max_selector_score") >= 4.0]
    raise ValueError(candidate)


def promotion_analysis(stage186_rows, keyframe_payloads, stage187_rows, cell_rows):
    rows = by_schedule_frame(stage186_rows)
    full_variant = variant_row_map(stage187_rows)
    cell_by_frame = cell_for_extra(cell_rows)
    out = []
    for (schedule, sequence, frame_index), adaptive in rows.items():
        if schedule != "stage165_adaptive" or adaptive["final_type"] != "q12_keyframe":
            continue
        gap8 = rows.get(("uniform_gap8", sequence, frame_index))
        if gap8 is None or gap8["final_type"] == "q12_keyframe":
            continue
        label = full_variant.get((sequence, frame_index), {})
        keyframe_payload = keyframe_payloads[keyframe_key(sequence, frame_index)]
        delta_psnr = numeric(adaptive, "psnr") - numeric(gap8, "psnr")
        delta_lpips = numeric(adaptive, "lpips") - numeric(gap8, "lpips")
        local_delta = numeric(keyframe_payload, "bitstream_bytes") - numeric(gap8, "payload_bytes")
        low_gain = delta_psnr < 0.01 and delta_lpips > -0.0001
        small_unlabeled_gain = delta_psnr < 0.25 and delta_lpips > -0.03
        no_label_need = int(numeric(label, "hard_quality_label")) == 0 and int(numeric(label, "high_payload_label")) == 0
        large_local_delta = local_delta > 450000
        rate_risk = low_gain or (no_label_need and small_unlabeled_gain and large_local_delta)
        reasons = []
        if low_gain:
            reasons.append("low_local_quality_gain")
        if no_label_need and small_unlabeled_gain:
            reasons.append("small_unlabeled_gain")
        elif no_label_need:
            reasons.append("no_stage187_hard_or_payload_label")
        if large_local_delta:
            reasons.append("large_local_payload_delta")
        out.append({
            "sequence": sequence,
            "frame_index": frame_index,
            "cell_id": cell_by_frame.get((sequence, frame_index), ""),
            "selector_score": numeric(label, "selector_score"),
            "vote_count": numeric(label, "vote_count"),
            "hard_quality_label": int(numeric(label, "hard_quality_label")),
            "high_payload_label": int(numeric(label, "high_payload_label")),
            "gap8_psnr": numeric(gap8, "psnr"),
            "adaptive_psnr": numeric(adaptive, "psnr"),
            "delta_psnr_adaptive_vs_gap8": delta_psnr,
            "gap8_ssim": numeric(gap8, "ssim"),
            "adaptive_ssim": numeric(adaptive, "ssim"),
            "delta_ssim_adaptive_vs_gap8": numeric(adaptive, "ssim") - numeric(gap8, "ssim"),
            "gap8_ms_ssim": numeric(gap8, "ms_ssim"),
            "adaptive_ms_ssim": numeric(adaptive, "ms_ssim"),
            "delta_ms_ssim_adaptive_vs_gap8": numeric(adaptive, "ms_ssim") - numeric(gap8, "ms_ssim"),
            "gap8_lpips": numeric(gap8, "lpips"),
            "adaptive_lpips": numeric(adaptive, "lpips"),
            "delta_lpips_adaptive_vs_gap8": delta_lpips,
            "gap8_residual_payload_bytes": int(numeric(gap8, "payload_bytes")),
            "adaptive_keyframe_bitstream_bytes": int(numeric(keyframe_payload, "bitstream_bytes")),
            "local_payload_delta_bytes": int(local_delta),
            "false_positive_risk_flag": int(rate_risk),
            "false_positive_reason": " ".join(reasons),
        })
    out.sort(key=lambda row: (int(row["false_positive_risk_flag"]), row["delta_psnr_adaptive_vs_gap8"], -row["local_payload_delta_bytes"]), reverse=True)
    return out


def residual_risk_analysis(stage186_rows):
    rows = by_schedule_frame(stage186_rows)
    out = []
    for (schedule, sequence, frame_index), adaptive in rows.items():
        if schedule != "stage165_adaptive" or adaptive["final_type"] != "stage158_residual_recovery":
            continue
        low_psnr = numeric(adaptive, "psnr") < 26.0
        high_lpips = numeric(adaptive, "lpips") > 0.22
        high_payload = numeric(adaptive, "payload_bytes") > 220000.0
        if not (low_psnr or high_lpips or high_payload):
            continue
        gap8 = rows[("uniform_gap8", sequence, frame_index)]
        gap4 = rows[("uniform_gap4", sequence, frame_index)]
        risk_score = max(0.0, 26.0 - numeric(adaptive, "psnr")) + max(0.0, numeric(adaptive, "lpips") - 0.22) * 20.0 + max(0.0, numeric(adaptive, "payload_bytes") - 220000.0) / 50000.0
        out.append({
            "sequence": sequence,
            "frame_index": frame_index,
            "adaptive_psnr": numeric(adaptive, "psnr"),
            "adaptive_ssim": numeric(adaptive, "ssim"),
            "adaptive_ms_ssim": numeric(adaptive, "ms_ssim"),
            "adaptive_lpips": numeric(adaptive, "lpips"),
            "adaptive_payload_bytes": int(numeric(adaptive, "payload_bytes")),
            "gap8_psnr": numeric(gap8, "psnr"),
            "gap8_lpips": numeric(gap8, "lpips"),
            "gap4_psnr": numeric(gap4, "psnr"),
            "gap4_lpips": numeric(gap4, "lpips"),
            "delta_psnr_gap4_vs_adaptive": numeric(gap4, "psnr") - numeric(adaptive, "psnr"),
            "delta_lpips_gap4_vs_adaptive": numeric(gap4, "lpips") - numeric(adaptive, "lpips"),
            "low_psnr_flag": int(low_psnr),
            "high_lpips_flag": int(high_lpips),
            "high_payload_flag": int(high_payload),
            "risk_score": risk_score,
        })
    out.sort(key=lambda row: row["risk_score"], reverse=True)
    return out


def candidate_drop_analysis(candidate_rows_by_name, stage186_rows, stage188_rows):
    gap8 = {k[1:]: row for k, row in by_schedule_frame(stage186_rows).items() if k[0] == "uniform_gap8"}
    full = {k[1:]: row for k, row in by_schedule_frame(stage186_rows).items() if k[0] == "stage165_adaptive"}
    rd = {row["candidate"]: row for row in stage188_rows}
    out = []
    summary = []
    for candidate, rows in candidate_rows_by_name.items():
        candidate_map = by_frame(rows)
        changed = []
        for key, full_row in full.items():
            cand = candidate_map[key]
            if cand["measurement_key"] == full_row["measurement_key"]:
                continue
            gap8_row = gap8[key]
            delta_psnr_full = numeric(cand, "psnr") - numeric(full_row, "psnr")
            delta_lpips_full = numeric(cand, "lpips") - numeric(full_row, "lpips")
            item = {
                "candidate": candidate,
                "sequence": key[0],
                "frame_index": key[1],
                "full_type": full_row["final_type"],
                "candidate_type": cand["final_type"],
                "full_psnr": numeric(full_row, "psnr"),
                "candidate_psnr": numeric(cand, "psnr"),
                "delta_psnr_candidate_vs_full": delta_psnr_full,
                "gap8_psnr": numeric(gap8_row, "psnr"),
                "delta_psnr_candidate_vs_gap8": numeric(cand, "psnr") - numeric(gap8_row, "psnr"),
                "full_lpips": numeric(full_row, "lpips"),
                "candidate_lpips": numeric(cand, "lpips"),
                "delta_lpips_candidate_vs_full": delta_lpips_full,
                "gap8_lpips": numeric(gap8_row, "lpips"),
                "delta_lpips_candidate_vs_gap8": numeric(cand, "lpips") - numeric(gap8_row, "lpips"),
                "candidate_payload_bytes": int(numeric(cand, "payload_bytes")),
            }
            changed.append(item)
            out.append(item)
        r = rd[candidate]
        summary.append({
            "candidate": candidate,
            "keyframe_count": r["keyframe_count"],
            "rate_mib_per_frame_additive": r["total_mib_per_frame_additive"],
            "delta_rate_vs_gap8": r["delta_mib_per_frame_vs_gap8_additive"],
            "delta_rate_vs_full": r["delta_mib_per_frame_vs_full_additive"],
            "mean_psnr": r["mean_psnr"],
            "delta_psnr_vs_gap8": r["delta_psnr_vs_gap8"],
            "delta_psnr_vs_full": r["delta_psnr_vs_full"],
            "mean_lpips": r["mean_lpips"],
            "delta_lpips_vs_gap8": r["delta_lpips_vs_gap8"],
            "changed_frame_count_vs_full": len(changed),
            "mean_changed_delta_psnr_vs_full": mean(row["delta_psnr_candidate_vs_full"] for row in changed),
            "worst_changed_delta_psnr_vs_full": min((row["delta_psnr_candidate_vs_full"] for row in changed), default=0.0),
            "mean_changed_delta_lpips_vs_full": mean(row["delta_lpips_candidate_vs_full"] for row in changed),
        })
    out.sort(key=lambda row: (row["candidate"], row["delta_psnr_candidate_vs_full"]))
    return out, summary


def sequence_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["schedule"], row["sequence"]), []).append(row)
    gap8_means = {}
    for (schedule, sequence), group in grouped.items():
        if schedule == "uniform_gap8":
            gap8_means[sequence] = {
                "psnr": mean(row["psnr"] for row in group),
                "lpips": mean(row["lpips"] for row in group),
            }
    out = []
    for (schedule, sequence), group in sorted(grouped.items()):
        psnr = mean(row["psnr"] for row in group)
        lpips = mean(row["lpips"] for row in group)
        out.append({
            "schedule": schedule,
            "sequence": sequence,
            "frame_count": len(group),
            "keyframe_count": sum(1 for row in group if row["final_type"] == "q12_keyframe"),
            "mean_psnr": psnr,
            "p10_psnr": percentile((row["psnr"] for row in group), 10),
            "mean_lpips": lpips,
            "p90_lpips": percentile((row["lpips"] for row in group), 90),
            "mean_payload_bytes": mean(row["payload_bytes"] for row in group),
            "delta_psnr_vs_gap8": psnr - gap8_means[sequence]["psnr"] if sequence in gap8_means else 0.0,
            "delta_lpips_vs_gap8": lpips - gap8_means[sequence]["lpips"] if sequence in gap8_means else 0.0,
        })
    return out


def sequence_hotspots(promotions, residual_risks):
    sequences = sorted({row["sequence"] for row in promotions} | {row["sequence"] for row in residual_risks})
    out = []
    for sequence in sequences:
        seq_promotions = [row for row in promotions if row["sequence"] == sequence]
        seq_residuals = [row for row in residual_risks if row["sequence"] == sequence]
        worst = max(seq_residuals, key=lambda row: row["risk_score"], default=None)
        out.append({
            "sequence": sequence,
            "promotion_count": len(seq_promotions),
            "promotion_rate_risk_count": sum(int(row["false_positive_risk_flag"]) for row in seq_promotions),
            "residual_risk_count": len(seq_residuals),
            "low_psnr_count": sum(int(row["low_psnr_flag"]) for row in seq_residuals),
            "high_lpips_count": sum(int(row["high_lpips_flag"]) for row in seq_residuals),
            "high_payload_count": sum(int(row["high_payload_flag"]) for row in seq_residuals),
            "mean_residual_risk_score": mean(row["risk_score"] for row in seq_residuals),
            "max_residual_risk_score": worst["risk_score"] if worst else 0.0,
            "worst_residual_frame_index": worst["frame_index"] if worst else "",
        })
    out.sort(key=lambda row: (int(row["residual_risk_count"]), float(row["max_residual_risk_score"])), reverse=True)
    return out


def write_report(promotions, residual_risks, candidate_drop_rows, candidate_summary, sequence_rows, hotspot_rows, package, path):
    risk_promotions = [row for row in promotions if int(row["false_positive_risk_flag"])]
    lines = [
        "# Stage189 Failure-Case Analysis",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Promotion rows analyzed: `{len(promotions)}`; promoted rate-risk rows: `{len(risk_promotions)}`.",
        f"- Residual risk rows: `{len(residual_risks)}`.",
        "",
        "## Candidate Summary",
        "",
        "| candidate | keyframes | MiB/frame | dRate vs gap8 | dRate vs full | PSNR | dPSNR vs gap8 | LPIPS | changed frames vs full | worst changed dPSNR vs full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidate_summary:
        lines.append(
            f"| {row['candidate']} | {row['keyframe_count']} | {float(row['rate_mib_per_frame_additive']):.12f} | {float(row['delta_rate_vs_gap8']):.12f} | "
            f"{float(row['delta_rate_vs_full']):.12f} | {float(row['mean_psnr']):.6f} | {float(row['delta_psnr_vs_gap8']):.6f} | {float(row['mean_lpips']):.6f} | "
            f"{row['changed_frame_count_vs_full']} | {float(row['worst_changed_delta_psnr_vs_full']):.6f} |"
        )
    lines.extend([
        "",
        "## Worst Promotion Rate Risks",
        "",
        "| sequence | frame | dPSNR adaptive-gap8 | dLPIPS adaptive-gap8 | local payload delta | reason |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in sorted(risk_promotions, key=lambda item: (item["delta_psnr_adaptive_vs_gap8"], item["delta_lpips_adaptive_vs_gap8"]))[:12]:
        lines.append(
            f"| {row['sequence']} | {row['frame_index']} | {float(row['delta_psnr_adaptive_vs_gap8']):.6f} | "
            f"{float(row['delta_lpips_adaptive_vs_gap8']):.6f} | {row['local_payload_delta_bytes']} | {row['false_positive_reason']} |"
        )
    lines.extend([
        "",
        "## Worst Residual False-Negative Risks",
        "",
        "| sequence | frame | adaptive PSNR | adaptive LPIPS | payload | gap4 dPSNR | gap4 dLPIPS | flags |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in residual_risks[:12]:
        flags = " ".join(name for name, key in [("low_psnr", "low_psnr_flag"), ("high_lpips", "high_lpips_flag"), ("high_payload", "high_payload_flag")] if int(row[key]))
        lines.append(
            f"| {row['sequence']} | {row['frame_index']} | {float(row['adaptive_psnr']):.6f} | {float(row['adaptive_lpips']):.6f} | "
            f"{row['adaptive_payload_bytes']} | {float(row['delta_psnr_gap4_vs_adaptive']):.6f} | {float(row['delta_lpips_gap4_vs_adaptive']):.6f} | {flags} |"
        )
    lines.extend([
        "",
        "## Sequence Hotspots",
        "",
        "| sequence | promotion risks | residual risks | low PSNR | high LPIPS | high payload | max residual risk | worst residual frame |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in hotspot_rows[:10]:
        lines.append(
            f"| {row['sequence']} | {row['promotion_rate_risk_count']} | {row['residual_risk_count']} | {row['low_psnr_count']} | "
            f"{row['high_lpips_count']} | {row['high_payload_count']} | {float(row['max_residual_risk_score']):.6f} | {row['worst_residual_frame_index']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Rate overhead is driven by promoted keyframes whose local keyframe cost is much larger than the replaced gap8 residual payload.",
        "- The lowest-rate Stage188 candidate keeps only the strongest cells and therefore gives only a small quality gain over gap8.",
        "- The balanced candidate keeps enough high-score cells to preserve most visible/metric gains while cutting about half the full adaptive overhead.",
        "- Remaining false negatives are residual frames with low PSNR, high LPIPS, or high residual payload that were not promoted.",
        "",
        "## Outputs",
        "",
        f"- Promotion analysis CSV: `{package['promotion_csv']}`",
        f"- Residual risk CSV: `{package['residual_risk_csv']}`",
        f"- Candidate dropped-frame CSV: `{package['candidate_drop_csv']}`",
        f"- Candidate summary CSV: `{package['candidate_summary_csv']}`",
        f"- Sequence summary CSV: `{package['sequence_summary_csv']}`",
        f"- Sequence hotspot CSV: `{package['sequence_hotspot_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage165_rows", type=Path, default=DEFAULT_STAGE165_ROWS)
    parser.add_argument("--stage187_rows", type=Path, default=DEFAULT_STAGE187_ROWS)
    parser.add_argument("--stage186_final_rows", type=Path, default=DEFAULT_STAGE186_FINAL_ROWS)
    parser.add_argument("--stage188_rd", type=Path, default=DEFAULT_STAGE188_RD)
    parser.add_argument("--stage188_cells", type=Path, default=DEFAULT_STAGE188_CELLS)
    parser.add_argument("--keyframe_payloads", type=Path, default=DEFAULT_KEYFRAME_PAYLOADS)
    parser.add_argument("--residual_payloads", type=Path, default=DEFAULT_RESIDUAL_PAYLOADS)
    parser.add_argument("--keyframe_quality", type=Path, default=DEFAULT_KEYFRAME_QUALITY)
    parser.add_argument("--residual_quality", type=Path, default=DEFAULT_RESIDUAL_QUALITY)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage165_rows = read_csv(args.stage165_rows)
    stage187_rows = read_csv(args.stage187_rows)
    stage186_rows = read_csv(args.stage186_final_rows)
    stage188_rows = read_csv(args.stage188_rd)
    schedule_keyframes, total_frames_by_sequence = load_schedule_keyframes(stage165_rows)
    keyframe_payloads = ok_map(read_csv(args.keyframe_payloads))
    residual_payloads = ok_map(read_csv(args.residual_payloads))
    keyframe_quality = ok_map(read_csv(args.keyframe_quality))
    residual_quality = ok_map(read_csv(args.residual_quality))
    cell_rows = build_interval_cells(schedule_keyframes, build_full_variant_rows(stage187_rows))
    promotions = promotion_analysis(stage186_rows, keyframe_payloads, stage187_rows, cell_rows)
    residual_risks = residual_risk_analysis(stage186_rows)
    candidate_rows_by_name = {}
    for candidate in ["interval_top10pct_cells", "interval_score_ge4p0", "interval_top90pct_cells"]:
        keyframes = keyframes_for_kept_cells(schedule_keyframes, kept_cells_for_candidate(candidate, cell_rows))
        candidate_rows_by_name[candidate] = build_candidate_final_rows(candidate, keyframes, total_frames_by_sequence, keyframe_quality, residual_quality, residual_payloads)
    candidate_drop_rows, candidate_summary = candidate_drop_analysis(candidate_rows_by_name, stage186_rows, stage188_rows)
    sequence_rows = sequence_summary(stage186_rows + [row for rows in candidate_rows_by_name.values() for row in rows])
    hotspot_rows = sequence_hotspots(promotions, residual_risks)
    promotion_csv = args.output_root / "stage189_promoted_keyframe_false_positive_analysis.csv"
    residual_csv = args.output_root / "stage189_unpromoted_residual_false_negative_risks.csv"
    candidate_drop_csv = args.output_root / "stage189_candidate_dropped_frame_loss_analysis.csv"
    candidate_summary_csv = args.output_root / "stage189_candidate_failure_summary.csv"
    sequence_csv = args.output_root / "stage189_sequence_level_failure_summary.csv"
    hotspot_csv = args.output_root / "stage189_sequence_hotspots.csv"
    package_json = args.output_root / "stage189_failure_case_analysis_package.json"
    report_md = args.output_root / "stage189_failure_case_analysis_report.md"
    write_csv(promotions, promotion_csv, PROMOTION_FIELDS)
    write_csv(residual_risks, residual_csv, RESIDUAL_RISK_FIELDS)
    write_csv(candidate_drop_rows, candidate_drop_csv, CANDIDATE_DROP_FIELDS)
    write_csv(candidate_summary, candidate_summary_csv, CANDIDATE_SUMMARY_FIELDS)
    write_csv(sequence_rows, sequence_csv, SEQUENCE_FIELDS)
    write_csv(hotspot_rows, hotspot_csv, HOTSPOT_FIELDS)
    risk_promotions = [row for row in promotions if int(row["false_positive_risk_flag"])]
    package = {
        "stage": 189,
        "status": "failure_case_analysis_packaged",
        "decision": "failure_cases_identified_for_paper_and_next_selector_refinement",
        "promotion_count": len(promotions),
        "false_positive_risk_count": len(risk_promotions),
        "residual_risk_count": len(residual_risks),
        "candidate_count": len(candidate_rows_by_name),
        "top_false_positive_risks": risk_promotions[:10],
        "top_residual_risks": residual_risks[:10],
        "candidate_summary": candidate_summary,
        "sequence_hotspots": hotspot_rows[:10],
        "promotion_csv": str(promotion_csv),
        "residual_risk_csv": str(residual_csv),
        "candidate_drop_csv": str(candidate_drop_csv),
        "candidate_summary_csv": str(candidate_summary_csv),
        "sequence_summary_csv": str(sequence_csv),
        "sequence_hotspot_csv": str(hotspot_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(promotions, residual_risks, candidate_drop_rows, candidate_summary, sequence_rows, hotspot_rows, package, report_md)
    print(json.dumps({
        "package": str(package_json),
        "decision": package["decision"],
        "promotion_count": len(promotions),
        "false_positive_risk_count": len(risk_promotions),
        "residual_risk_count": len(residual_risks),
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
