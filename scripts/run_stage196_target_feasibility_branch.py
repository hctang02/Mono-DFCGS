import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE192 = REPO_ROOT / "experiments/stage192_expanded_fixed_gap_measurement/stage192_expanded_fixed_gap_rd_quality_points.csv"
STAGE193 = REPO_ROOT / "experiments/stage193_oracle_upper_bound_analysis/stage193_oracle_summary.csv"
STAGE194 = REPO_ROOT / "experiments/stage194_all_keyframe_q12_upper_bound/stage194_all_keyframe_q12_summary.csv"
STAGE195 = REPO_ROOT / "experiments/stage195_higher_fidelity_keyframe_upper_bound/stage195_higher_fidelity_keyframe_summary.csv"
OUTPUT_ROOT = REPO_ROOT / "experiments/stage196_target_feasibility_branch"

SUMMARY_FIELDS = [
    "source_stage",
    "ceiling",
    "schedule_consistent",
    "rate_scope",
    "mib_per_frame",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "delta_psnr_vs_best_fixed_gap2",
    "target_psnr_gap2_plus_1db",
    "delta_psnr_vs_target",
    "passes_target_no_metric_regression",
    "interpretation",
]

BRANCH_FIELDS = ["branch", "status", "rationale", "next_stage", "claim_risk"]


def read_csv(path):
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


def best_gap2():
    for row in read_csv(STAGE192):
        if row["schedule"] == "uniform_gap2":
            return row
    raise ValueError("uniform_gap2 missing from Stage192")


def pass_flag(row, best, psnr, ssim, ms_ssim, lpips, target_psnr):
    return int(
        psnr >= target_psnr
        and ssim >= numeric(best, "mean_ssim")
        and ms_ssim >= numeric(best, "mean_ms_ssim")
        and lpips <= numeric(best, "mean_lpips")
    )


def summary_rows(best):
    target_psnr = numeric(best, "mean_psnr") + 1.0
    rows = []
    for row in read_csv(STAGE193):
        psnr = numeric(row, "mean_psnr")
        rows.append(
            {
                "source_stage": 193,
                "ceiling": row["oracle"],
                "schedule_consistent": row["schedule_consistent"],
                "rate_scope": row["note"],
                "mib_per_frame": row["mib_per_frame_additive_proxy"],
                "psnr": psnr,
                "ssim": row["mean_ssim"],
                "ms_ssim": row["mean_ms_ssim"],
                "lpips": row["mean_lpips"],
                "delta_psnr_vs_best_fixed_gap2": psnr - numeric(best, "mean_psnr"),
                "target_psnr_gap2_plus_1db": target_psnr,
                "delta_psnr_vs_target": psnr - target_psnr,
                "passes_target_no_metric_regression": row["beats_best_fixed_by_1db_no_metric_regression"],
                "interpretation": "current measured selector/fixed-gap candidate-space ceiling",
            }
        )
    row194 = read_csv(STAGE194)[0]
    psnr194 = numeric(row194, "mean_psnr")
    rows.append(
        {
            "source_stage": 194,
            "ceiling": "all_q12_keyframes",
            "schedule_consistent": 1,
            "rate_scope": "measured_schedule_packed_q12_keyframes_plus_fixed_gap_metadata",
            "mib_per_frame": row194["total_mib_per_frame"],
            "psnr": psnr194,
            "ssim": row194["mean_ssim"],
            "ms_ssim": row194["mean_ms_ssim"],
            "lpips": row194["mean_lpips"],
            "delta_psnr_vs_best_fixed_gap2": psnr194 - numeric(best, "mean_psnr"),
            "target_psnr_gap2_plus_1db": target_psnr,
            "delta_psnr_vs_target": psnr194 - target_psnr,
            "passes_target_no_metric_regression": row194["beats_best_fixed_by_1db_no_metric_regression"],
            "interpretation": "q12 keyframe representation ceiling",
        }
    )
    for row in read_csv(STAGE195):
        psnr = numeric(row, "mean_psnr")
        rows.append(
            {
                "source_stage": 195,
                "ceiling": row["representation"],
                "schedule_consistent": 1,
                "rate_scope": row["rate_scope"],
                "mib_per_frame": row["total_mib_per_frame"],
                "psnr": psnr,
                "ssim": row["mean_ssim"],
                "ms_ssim": row["mean_ms_ssim"],
                "lpips": row["mean_lpips"],
                "delta_psnr_vs_best_fixed_gap2": psnr - numeric(best, "mean_psnr"),
                "target_psnr_gap2_plus_1db": target_psnr,
                "delta_psnr_vs_target": psnr - target_psnr,
                "passes_target_no_metric_regression": row["beats_best_fixed_by_1db_no_metric_regression"],
                "interpretation": "higher-fidelity dense-anchor/rendering representation ceiling",
            }
        )
    return rows


def branch_rows():
    return [
        {
            "branch": "selector_or_keyframe_quantization_tuning",
            "status": "stop",
            "rationale": "Stage193-195 ceilings remain at least 0.7698838629920566 dB below gap2+1dB target.",
            "next_stage": "none",
            "claim_risk": "Cannot honestly claim a large full-sequence selector gain over best fixed gap.",
        },
        {
            "branch": "counted_rgb_or_image_residual_correction",
            "status": "viable_next_diagnostic",
            "rationale": "Prior Stage155 sampled image-residual upper bound exceeded 31 dB, so this is the most likely source of +1 dB headroom if all bytes are counted.",
            "next_stage": "full_sequence_counted_rgb_residual_upper_bound",
            "claim_risk": "Less Gaussian-native; must be framed as a counted correction payload, not a pure GS keyframe selector.",
        },
        {
            "branch": "new_dense_anchor_reconstruction_objective_or_model",
            "status": "viable_but_heavy",
            "rationale": "Float dense-anchor target quality is only 29.884931 dB, so the renderer/anchor representation must improve to reach 30.654815 dB without image residuals.",
            "next_stage": "training_or_representation_redesign",
            "claim_risk": "Requires new training and may not finish quickly.",
        },
        {
            "branch": "paper_claim_scope_adjustment",
            "status": "viable_writing_fallback",
            "rationale": "Existing evidence supports sampled-target selector gains and measured middle RD points, but not +1 dB over best full-sequence fixed gap.",
            "next_stage": "revise_claims_and_tables",
            "claim_risk": "Does not satisfy user's requested stronger full-sequence result.",
        },
    ]


def decision(rows):
    if any(int(row["passes_target_no_metric_regression"]) for row in rows):
        return "target_headroom_found"
    return "selector_keyframe_representation_cannot_meet_target"


def write_report(rows, branches, package, path):
    lines = [
        "# Stage196 Target Feasibility Branch",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Target PSNR: `{package['target_psnr_gap2_plus_1db']}` (`uniform_gap2 + 1 dB`).",
        "",
        "## Ceilings",
        "",
        "| source | ceiling | PSNR | dPSNR vs gap2 | dPSNR vs target | pass | interpretation |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['source_stage']} | {row['ceiling']} | {float(row['psnr']):.6f} | "
            f"{float(row['delta_psnr_vs_best_fixed_gap2']):.6f} | {float(row['delta_psnr_vs_target']):.6f} | "
            f"{row['passes_target_no_metric_regression']} | {row['interpretation']} |"
        )
    lines.extend(["", "## Branch Options", "", "| branch | status | next stage | claim risk |", "|---|---|---|---|"])
    for row in branches:
        lines.append(f"| {row['branch']} | {row['status']} | {row['next_stage']} | {row['claim_risk']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current selector/keyframe branch is exhausted for the requested `+1 dB` full-sequence target.",
            "- The next technically plausible route is a counted correction payload or a new reconstruction model/objective, not selector threshold tuning.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    best = best_gap2()
    rows = summary_rows(best)
    branches = branch_rows()
    stage_decision = decision(rows)
    target_psnr = numeric(best, "mean_psnr") + 1.0
    summary_csv = OUTPUT_ROOT / "stage196_target_feasibility_summary.csv"
    branch_csv = OUTPUT_ROOT / "stage196_branch_options.csv"
    package_json = OUTPUT_ROOT / "stage196_target_feasibility_branch_package.json"
    report_md = OUTPUT_ROOT / "stage196_target_feasibility_branch_report.md"
    write_csv(rows, summary_csv, SUMMARY_FIELDS)
    write_csv(branches, branch_csv, BRANCH_FIELDS)
    package = {
        "stage": 196,
        "status": "target_feasibility_branch_complete",
        "decision": stage_decision,
        "best_fixed_reference": "uniform_gap2",
        "best_fixed_psnr": numeric(best, "mean_psnr"),
        "target_psnr_gap2_plus_1db": target_psnr,
        "summary_rows": rows,
        "branch_rows": branches,
        "summary_csv": str(summary_csv.relative_to(REPO_ROOT)),
        "branch_csv": str(branch_csv.relative_to(REPO_ROOT)),
        "package_json": str(package_json.relative_to(REPO_ROOT)),
        "report_md": str(report_md.relative_to(REPO_ROOT)),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(rows, branches, package, report_md)
    print(json.dumps({"package": str(package_json), "decision": stage_decision, "target_psnr": target_psnr}, indent=2))


if __name__ == "__main__":
    main()
