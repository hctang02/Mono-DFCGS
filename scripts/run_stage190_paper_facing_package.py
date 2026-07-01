import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

STAGE185_TOTAL_RD = REPO_ROOT / "experiments/stage185_measured_full_sequence_rd_aggregation/stage185_measured_full_sequence_total_rd.csv"
STAGE186_RD_QUALITY = REPO_ROOT / "experiments/stage186_full_sequence_quality_validation/stage186_measured_rd_quality_points.csv"
STAGE187_ABLATION = REPO_ROOT / "experiments/stage187_selector_feature_ablation_validation/stage187_selector_feature_ablation_summary.csv"
STAGE188_SENSITIVITY = REPO_ROOT / "experiments/stage188_lower_budget_selector_sensitivity/stage188_lower_budget_selector_sensitivity_rd_quality.csv"
STAGE189_CANDIDATE_FAILURE = REPO_ROOT / "experiments/stage189_failure_case_analysis/stage189_candidate_failure_summary.csv"
STAGE189_PROMOTED_RISKS = REPO_ROOT / "experiments/stage189_failure_case_analysis/stage189_promoted_keyframe_false_positive_analysis.csv"
STAGE189_HOTSPOTS = REPO_ROOT / "experiments/stage189_failure_case_analysis/stage189_sequence_hotspots.csv"

OUTPUT_ROOT = REPO_ROOT / "experiments/stage190_paper_facing_package"

RECOMMENDED_TITLE = "Mono-DFCGS: Recovery-Aware Adaptive Keyframe Scheduling for Monocular Dynamic Gaussian Splatting Compression"

ABSTRACT_DRAFT = (
    "We present Mono-DFCGS, a monocular dynamic Gaussian splatting compression pipeline that combines "
    "StreamSplat-guided Gaussian-domain middle-frame recovery with a recovery-aware adaptive keyframe "
    "schedule. The recovery module preserves the original dynamic prediction as the motion/geometry base "
    "and transmits a counted entropy-coded residual for one selected half-anchor, avoiding decoder-side "
    "target anchors, target RGB, or unencoded oracle residuals. The adaptive scheduler uses encoder-side "
    "RGB/motion cues to promote difficult or high-payload frames while transmitting only the resulting "
    "schedule. On full DAVIS sequence accounting, the current adaptive schedule is a measured middle "
    "rate-distortion point: it improves PSNR, SSIM, MS-SSIM, and LPIPS over uniform gap8, and remains below "
    "uniform gap4 rate, but it is not yet lower-rate than gap8. We further analyze feature ablations, "
    "lower-budget sensitivity, and failure cases to identify the next selector refinement targets."
)


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def numeric(row, key, default=0.0):
    value = row.get(key) if row else None
    if value in (None, "", "NA"):
        return default
    return float(value)


def fmt(value, digits=6):
    if value in (None, ""):
        return ""
    return f"{float(value):.{digits}f}"


def get_rows_by_key(rows, key):
    return {row[key]: row for row in rows}


def ordered_subset(rows, key, order):
    by_key = get_rows_by_key(rows, key)
    return [by_key[item] for item in order if item in by_key]


def markdown_table(rows, columns):
    header = "| " + " | ".join(title for _, title in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join([header, sep] + body)


def build_measured_rd_quality():
    rd_rows = read_csv(STAGE185_TOTAL_RD)
    quality_rows = get_rows_by_key(read_csv(STAGE186_RD_QUALITY), "schedule")
    names = {
        "uniform_gap8": "Uniform gap8",
        "stage165_adaptive": "Mono-DFCGS adaptive",
        "uniform_gap4": "Uniform gap4",
    }
    notes = {
        "uniform_gap8": "lower-rate fixed-gap baseline",
        "stage165_adaptive": "middle RD point; better quality than gap8 but higher rate",
        "uniform_gap4": "higher-quality fixed-gap reference",
    }
    out = []
    for row in rd_rows:
        schedule = row["schedule"]
        q = quality_rows[schedule]
        out.append(
            {
                "schedule": schedule,
                "paper_name": names.get(schedule, schedule),
                "keyframe_count": row["keyframe_count"],
                "residual_count": row["residual_count"],
                "packed_rate_mib_per_frame": row["total_mib_per_frame"],
                "delta_rate_vs_gap8": row["delta_total_mib_per_frame_vs_gap8"],
                "psnr": q["mean_psnr"],
                "delta_psnr_vs_gap8": q["delta_psnr_vs_gap8"],
                "ssim": q["mean_ssim"],
                "ms_ssim": q["mean_ms_ssim"],
                "lpips": q["mean_lpips"],
                "delta_lpips_vs_gap8": q["delta_lpips_vs_gap8"],
                "scope": row["rate_scope"],
                "paper_note": notes.get(schedule, ""),
            }
        )
    return out


def build_selector_ablation():
    rows = read_csv(STAGE187_ABLATION)
    order = [
        "full_stage165_features",
        "drop_interp_rgb",
        "motion_proxy_edge_hist",
        "edge_hist_only",
        "drop_hist_motion",
        "drop_edge_motion",
        "proxy_only",
        "rgb_only",
    ]
    notes = {
        "full_stage165_features": "full selector reference; highest payload recall",
        "drop_interp_rgb": "conservative lower-budget candidate",
        "motion_proxy_edge_hist": "small reduction with same hard recall",
        "edge_hist_only": "motion/edge stress point",
        "drop_hist_motion": "aggressive lower-budget candidate",
        "drop_edge_motion": "aggressive but lower hard recall",
        "proxy_only": "minimal one-feature proxy",
        "rgb_only": "direct RGB-only stress point",
    }
    out = []
    for row in ordered_subset(rows, "variant", order):
        out.append(
            {
                "variant": row["variant"],
                "feature_count": row["feature_count"],
                "selected_rows": row["selected_count"],
                "keyframes": row["keyframe_count"],
                "hard_recall": row["recall_hard"],
                "payload_recall": row["recall_payload"],
                "selected_delta_vs_full": row["selected_count_delta_vs_full"],
                "payload_recall_delta_vs_full": row["recall_payload_delta_vs_full"],
                "paper_note": notes.get(row["variant"], row.get("description", "")),
            }
        )
    return out


def build_lower_budget_sensitivity():
    rows = read_csv(STAGE188_SENSITIVITY)
    order = [
        "uniform_gap8",
        "interval_top10pct_cells",
        "interval_score_ge4p0",
        "interval_top90pct_cells",
        "stage165_adaptive_full",
        "uniform_gap4",
    ]
    notes = {
        "uniform_gap8": "additive sensitivity baseline",
        "interval_top10pct_cells": "lowest-rate positive-quality candidate",
        "interval_score_ge4p0": "balanced half-overhead candidate",
        "interval_top90pct_cells": "near-full quality below full adaptive rate",
        "stage165_adaptive_full": "full frozen selector under additive scope",
        "uniform_gap4": "additive fixed-gap high-quality reference",
    }
    out = []
    for row in ordered_subset(rows, "candidate", order):
        out.append(
            {
                "candidate": row["candidate"],
                "keyframes": row["keyframe_count"],
                "kept_cells": row.get("kept_cell_count", ""),
                "additive_rate_mib_per_frame": row["total_mib_per_frame_additive"],
                "delta_rate_vs_gap8": row["delta_mib_per_frame_vs_gap8_additive"],
                "delta_rate_vs_full": row["delta_mib_per_frame_vs_full_additive"],
                "psnr": row["mean_psnr"],
                "delta_psnr_vs_gap8": row["delta_psnr_vs_gap8"],
                "lpips": row["mean_lpips"],
                "delta_lpips_vs_gap8": row["delta_lpips_vs_gap8"],
                "quality_positive_all_metrics": row["quality_positive_vs_gap8_all_metrics"],
                "scope": row["rate_scope"],
                "paper_note": notes.get(row["candidate"], ""),
            }
        )
    return out


def build_failure_case_tables():
    candidate_rows = read_csv(STAGE189_CANDIDATE_FAILURE)
    out_candidates = []
    notes = {
        "interval_top10pct_cells": "most rate-efficient but many changed frames vs full adaptive",
        "interval_score_ge4p0": "balanced point with fewer changed frames",
        "interval_top90pct_cells": "near-full point with small quality loss",
    }
    for row in candidate_rows:
        out_candidates.append(
            {
                "candidate": row["candidate"],
                "keyframes": row["keyframe_count"],
                "additive_rate_mib_per_frame": row["rate_mib_per_frame_additive"],
                "delta_rate_vs_gap8": row["delta_rate_vs_gap8"],
                "delta_rate_vs_full": row["delta_rate_vs_full"],
                "psnr": row["mean_psnr"],
                "delta_psnr_vs_gap8": row["delta_psnr_vs_gap8"],
                "lpips": row["mean_lpips"],
                "changed_frames_vs_full": row["changed_frame_count_vs_full"],
                "worst_changed_dpsnr_vs_full": row["worst_changed_delta_psnr_vs_full"],
                "paper_note": notes.get(row["candidate"], ""),
            }
        )

    promoted = read_csv(STAGE189_PROMOTED_RISKS)
    promoted_risks = [row for row in promoted if row.get("false_positive_risk_flag") == "1"]
    promoted_out = []
    for row in promoted_risks:
        promoted_out.append(
            {
                "sequence": row["sequence"],
                "frame_index": row["frame_index"],
                "dpsnr_adaptive_vs_gap8": row["delta_psnr_adaptive_vs_gap8"],
                "dlpips_adaptive_vs_gap8": row["delta_lpips_adaptive_vs_gap8"],
                "local_payload_delta_bytes": row["local_payload_delta_bytes"],
                "reason": row["false_positive_reason"],
            }
        )

    hotspots = read_csv(STAGE189_HOTSPOTS)[:10]
    hotspot_out = []
    for row in hotspots:
        hotspot_out.append(
            {
                "sequence": row["sequence"],
                "residual_risks": row["residual_risk_count"],
                "low_psnr": row["low_psnr_count"],
                "high_lpips": row["high_lpips_count"],
                "high_payload": row["high_payload_count"],
                "max_residual_risk": row["max_residual_risk_score"],
                "worst_frame": row["worst_residual_frame_index"],
            }
        )
    return out_candidates, promoted_out, hotspot_out


def build_claims_and_limitations():
    rows = [
        {
            "category": "claim",
            "item": "measured adaptive RD position",
            "status": "supported",
            "paper_wording": "The current adaptive schedule is a middle rate-distortion point between uniform gap8 and uniform gap4.",
        },
        {
            "category": "claim",
            "item": "quality over gap8",
            "status": "supported",
            "paper_wording": "Adaptive improves PSNR, SSIM, MS-SSIM, and LPIPS over uniform gap8 on full-sequence evaluation.",
        },
        {
            "category": "non-claim",
            "item": "lower rate than gap8",
            "status": "not supported",
            "paper_wording": "Do not claim the frozen adaptive schedule is lower-rate than uniform gap8.",
        },
        {
            "category": "scope caveat",
            "item": "Stage188 additive rates",
            "status": "separate scope",
            "paper_wording": "Stage188 additive rates compare lower-budget candidates internally and must not be numerically mixed with Stage185 schedule-packed rates.",
        },
        {
            "category": "decoder contract",
            "item": "allowed inputs",
            "status": "fixed",
            "paper_wording": "The decoder receives original StreamSplat endpoint/base inputs, normalized time, encoded q6 keep1.0 entropy residual payload, counted half selector, and transmitted schedule/keyframe metadata.",
        },
        {
            "category": "decoder contract",
            "item": "forbidden inputs",
            "status": "fixed",
            "paper_wording": "The decoder does not receive target dense anchors, target RGB, unencoded target residuals, or oracle labels.",
        },
        {
            "category": "selector contract",
            "item": "encoder-side features",
            "status": "fixed",
            "paper_wording": "RGB/motion selector features are encoder-side only; the transmitted schedule is counted and sufficient for decoding.",
        },
        {
            "category": "limitation",
            "item": "unoptimized selector",
            "status": "open",
            "paper_wording": "Stage189 shows broad residual-risk hotspots and rare high-cost promotions; further selector refinement is needed for an optimized RD frontier.",
        },
        {
            "category": "next measurement",
            "item": "schedule-packed lower-budget candidates",
            "status": "optional",
            "paper_wording": "If final claims require same-scope candidate RD, selected Stage188 candidates should be measured with schedule-packed keyframe streams.",
        },
    ]
    return rows


def build_report(measured, ablation, sensitivity, failure_candidates, promoted, hotspots, claims):
    report_lines = [
        "# Stage190 Paper-Facing Package",
        "",
        "## Recommended Title",
        "",
        RECOMMENDED_TITLE,
        "",
        "## Abstract Draft",
        "",
        ABSTRACT_DRAFT,
        "",
        "## Core Claim",
        "",
        "The current measured result should be framed as a recovery-aware adaptive middle RD point: better quality than uniform gap8 at higher rate, and lower rate than uniform gap4 at lower quality.",
        "",
        "## Table 1. Full-Sequence RD-Quality",
        "",
        markdown_table(
            [
                {
                    "schedule": row["paper_name"],
                    "keyframes": row["keyframe_count"],
                    "rate": fmt(row["packed_rate_mib_per_frame"]),
                    "d_rate_gap8": fmt(row["delta_rate_vs_gap8"]),
                    "psnr": fmt(row["psnr"]),
                    "ssim": fmt(row["ssim"]),
                    "ms_ssim": fmt(row["ms_ssim"]),
                    "lpips": fmt(row["lpips"]),
                    "note": row["paper_note"],
                }
                for row in measured
            ],
            [
                ("schedule", "schedule"),
                ("keyframes", "keyframes"),
                ("rate", "MiB/frame"),
                ("d_rate_gap8", "dRate vs gap8"),
                ("psnr", "PSNR"),
                ("ssim", "SSIM"),
                ("ms_ssim", "MS-SSIM"),
                ("lpips", "LPIPS"),
                ("note", "note"),
            ],
        ),
        "",
        "Scope: Stage185/186 measured schedule-packed q12 keyframes plus measured Stage158 residual payloads plus exact metadata.",
        "",
        "## Table 2. Selector Feature Ablation",
        "",
        markdown_table(
            [
                {
                    "variant": row["variant"],
                    "features": row["feature_count"],
                    "selected": row["selected_rows"],
                    "keyframes": row["keyframes"],
                    "hard_recall": fmt(row["hard_recall"]),
                    "payload_recall": fmt(row["payload_recall"]),
                    "note": row["paper_note"],
                }
                for row in ablation
            ],
            [
                ("variant", "variant"),
                ("features", "features"),
                ("selected", "selected rows"),
                ("keyframes", "keyframes"),
                ("hard_recall", "hard recall"),
                ("payload_recall", "payload recall"),
                ("note", "note"),
            ],
        ),
        "",
        "Scope: Stage187 is a selector-label/protocol ablation, not measured full-sequence RD for each ablation schedule.",
        "",
        "## Table 3. Lower-Budget Sensitivity",
        "",
        markdown_table(
            [
                {
                    "candidate": row["candidate"],
                    "keyframes": row["keyframes"],
                    "rate": fmt(row["additive_rate_mib_per_frame"]),
                    "d_rate_gap8": fmt(row["delta_rate_vs_gap8"]),
                    "psnr": fmt(row["psnr"]),
                    "d_psnr_gap8": fmt(row["delta_psnr_vs_gap8"]),
                    "lpips": fmt(row["lpips"]),
                    "note": row["paper_note"],
                }
                for row in sensitivity
            ],
            [
                ("candidate", "candidate"),
                ("keyframes", "keyframes"),
                ("rate", "additive MiB/frame"),
                ("d_rate_gap8", "dRate vs gap8"),
                ("psnr", "PSNR"),
                ("d_psnr_gap8", "dPSNR vs gap8"),
                ("lpips", "LPIPS"),
                ("note", "note"),
            ],
        ),
        "",
        "Scope caveat: Stage188 uses measured single-anchor additive keyframes plus measured residuals and exact metadata. It is internally comparable across Stage188 candidates but not numerically interchangeable with Stage185 schedule-packed rates.",
        "",
        "## Table 4. Candidate Failure Cases",
        "",
        markdown_table(
            [
                {
                    "candidate": row["candidate"],
                    "keyframes": row["keyframes"],
                    "rate": fmt(row["additive_rate_mib_per_frame"]),
                    "d_rate_full": fmt(row["delta_rate_vs_full"]),
                    "psnr": fmt(row["psnr"]),
                    "changed": row["changed_frames_vs_full"],
                    "worst": fmt(row["worst_changed_dpsnr_vs_full"]),
                    "note": row["paper_note"],
                }
                for row in failure_candidates
            ],
            [
                ("candidate", "candidate"),
                ("keyframes", "keyframes"),
                ("rate", "MiB/frame"),
                ("d_rate_full", "dRate vs full"),
                ("psnr", "PSNR"),
                ("changed", "changed frames"),
                ("worst", "worst changed dPSNR"),
                ("note", "note"),
            ],
        ),
        "",
        "## Promoted Rate-Risk Examples",
        "",
        markdown_table(
            [
                {
                    "sequence": row["sequence"],
                    "frame": row["frame_index"],
                    "dpsnr": fmt(row["dpsnr_adaptive_vs_gap8"]),
                    "dlpips": fmt(row["dlpips_adaptive_vs_gap8"]),
                    "payload_delta": row["local_payload_delta_bytes"],
                    "reason": row["reason"],
                }
                for row in promoted
            ],
            [
                ("sequence", "sequence"),
                ("frame", "frame"),
                ("dpsnr", "dPSNR"),
                ("dlpips", "dLPIPS"),
                ("payload_delta", "payload delta bytes"),
                ("reason", "reason"),
            ],
        ),
        "",
        "## Residual-Risk Hotspots",
        "",
        markdown_table(
            [
                {
                    "sequence": row["sequence"],
                    "risks": row["residual_risks"],
                    "low_psnr": row["low_psnr"],
                    "high_lpips": row["high_lpips"],
                    "high_payload": row["high_payload"],
                    "max_risk": fmt(row["max_residual_risk"]),
                    "worst": row["worst_frame"],
                }
                for row in hotspots
            ],
            [
                ("sequence", "sequence"),
                ("risks", "residual risks"),
                ("low_psnr", "low PSNR"),
                ("high_lpips", "high LPIPS"),
                ("high_payload", "high payload"),
                ("max_risk", "max risk"),
                ("worst", "worst frame"),
            ],
        ),
        "",
        "## Decoder Contract",
        "",
        "Allowed decoder inputs: original StreamSplat endpoint/base inputs, normalized time, encoded q6/keep1.0 entropy residual payload, counted one-byte half selector, and transmitted schedule/keyframe metadata.",
        "",
        "Forbidden decoder inputs: target dense anchor, target RGB, unencoded target residual, rendered quality/oracle labels, or selector features that are not represented by transmitted schedule metadata.",
        "",
        "## Claim Boundaries",
        "",
        markdown_table(claims, [("category", "category"), ("item", "item"), ("status", "status"), ("paper_wording", "paper wording")]),
        "",
        "## Next Paper/Experiment Step",
        "",
        "Use this package for the paper-facing method/results section. If same-scope lower-budget RD is needed, measure schedule-packed keyframe streams for one or two selected Stage188 candidates before making final rate-frontier claims.",
        "",
    ]
    return "\n".join(report_lines)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    measured = build_measured_rd_quality()
    ablation = build_selector_ablation()
    sensitivity = build_lower_budget_sensitivity()
    failure_candidates, promoted_risks, hotspots = build_failure_case_tables()
    claims = build_claims_and_limitations()

    write_csv(
        OUTPUT_ROOT / "stage190_paper_table_measured_rd_quality.csv",
        measured,
        [
            "schedule",
            "paper_name",
            "keyframe_count",
            "residual_count",
            "packed_rate_mib_per_frame",
            "delta_rate_vs_gap8",
            "psnr",
            "delta_psnr_vs_gap8",
            "ssim",
            "ms_ssim",
            "lpips",
            "delta_lpips_vs_gap8",
            "scope",
            "paper_note",
        ],
    )
    write_csv(
        OUTPUT_ROOT / "stage190_paper_table_selector_ablation.csv",
        ablation,
        [
            "variant",
            "feature_count",
            "selected_rows",
            "keyframes",
            "hard_recall",
            "payload_recall",
            "selected_delta_vs_full",
            "payload_recall_delta_vs_full",
            "paper_note",
        ],
    )
    write_csv(
        OUTPUT_ROOT / "stage190_paper_table_lower_budget_sensitivity.csv",
        sensitivity,
        [
            "candidate",
            "keyframes",
            "kept_cells",
            "additive_rate_mib_per_frame",
            "delta_rate_vs_gap8",
            "delta_rate_vs_full",
            "psnr",
            "delta_psnr_vs_gap8",
            "lpips",
            "delta_lpips_vs_gap8",
            "quality_positive_all_metrics",
            "scope",
            "paper_note",
        ],
    )
    write_csv(
        OUTPUT_ROOT / "stage190_paper_table_candidate_failures.csv",
        failure_candidates,
        [
            "candidate",
            "keyframes",
            "additive_rate_mib_per_frame",
            "delta_rate_vs_gap8",
            "delta_rate_vs_full",
            "psnr",
            "delta_psnr_vs_gap8",
            "lpips",
            "changed_frames_vs_full",
            "worst_changed_dpsnr_vs_full",
            "paper_note",
        ],
    )
    write_csv(
        OUTPUT_ROOT / "stage190_paper_table_promoted_rate_risks.csv",
        promoted_risks,
        ["sequence", "frame_index", "dpsnr_adaptive_vs_gap8", "dlpips_adaptive_vs_gap8", "local_payload_delta_bytes", "reason"],
    )
    write_csv(
        OUTPUT_ROOT / "stage190_paper_table_residual_hotspots.csv",
        hotspots,
        ["sequence", "residual_risks", "low_psnr", "high_lpips", "high_payload", "max_residual_risk", "worst_frame"],
    )
    write_csv(
        OUTPUT_ROOT / "stage190_claims_and_limitations.csv",
        claims,
        ["category", "item", "status", "paper_wording"],
    )

    report = build_report(measured, ablation, sensitivity, failure_candidates, promoted_risks, hotspots, claims)
    report_path = OUTPUT_ROOT / "stage190_paper_facing_report.md"
    report_path.write_text(report, encoding="utf-8")

    package = {
        "stage": 190,
        "status": "paper_facing_package_created",
        "decision": "paper_facing_tables_and_claim_boundaries_packaged",
        "recommended_title": RECOMMENDED_TITLE,
        "abstract_draft": ABSTRACT_DRAFT,
        "primary_claim": "Mono-DFCGS adaptive is a measured middle RD point: quality above gap8 at higher rate and rate below gap4 at lower quality.",
        "non_claims": [
            "Do not claim the frozen adaptive schedule is lower-rate than uniform gap8.",
            "Do not mix Stage188 additive rates with Stage185 schedule-packed rates.",
            "Do not present teacher/oracle side-info as final decoder input.",
        ],
        "decoder_allowed_inputs": [
            "original StreamSplat endpoint/base inputs",
            "normalized time",
            "encoded q6/keep1.0 entropy residual payload",
            "counted one-byte half selector",
            "transmitted schedule/keyframe metadata",
        ],
        "decoder_forbidden_inputs": [
            "target dense anchor",
            "target RGB",
            "unencoded target residual",
            "rendered quality or oracle labels",
            "selector features not represented by transmitted schedule metadata",
        ],
        "table_counts": {
            "measured_rd_quality": len(measured),
            "selector_ablation": len(ablation),
            "lower_budget_sensitivity": len(sensitivity),
            "candidate_failures": len(failure_candidates),
            "promoted_rate_risks": len(promoted_risks),
            "residual_hotspots": len(hotspots),
            "claims_and_limitations": len(claims),
        },
        "outputs": {
            "report": str(report_path.relative_to(REPO_ROOT)),
            "measured_rd_quality": "experiments/stage190_paper_facing_package/stage190_paper_table_measured_rd_quality.csv",
            "selector_ablation": "experiments/stage190_paper_facing_package/stage190_paper_table_selector_ablation.csv",
            "lower_budget_sensitivity": "experiments/stage190_paper_facing_package/stage190_paper_table_lower_budget_sensitivity.csv",
            "candidate_failures": "experiments/stage190_paper_facing_package/stage190_paper_table_candidate_failures.csv",
            "promoted_rate_risks": "experiments/stage190_paper_facing_package/stage190_paper_table_promoted_rate_risks.csv",
            "residual_hotspots": "experiments/stage190_paper_facing_package/stage190_paper_table_residual_hotspots.csv",
            "claims_and_limitations": "experiments/stage190_paper_facing_package/stage190_claims_and_limitations.csv",
        },
        "next": "Use the package for paper writing; optionally measure schedule-packed Stage188 candidates for same-scope final RD claims.",
    }
    package_path = OUTPUT_ROOT / "stage190_paper_facing_package.json"
    package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "package": str(package_path),
                "decision": package["decision"],
                "table_counts": package["table_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
