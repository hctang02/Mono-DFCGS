import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "experiments/stage198_prior_predictor_training_audit"

STAGE78 = REPO_ROOT / "experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_psnr_table.csv"
STAGE143 = REPO_ROOT / "experiments/stage143_middle_frame_psnr_collapse_diagnostic/stage143_middle_frame_psnr_collapse_summary.csv"
STAGE145 = REPO_ROOT / "experiments/stage145_large_scale_adapter_training_launch/stage145_large_scale_adapter_training_launch_summary.json"
STAGE146 = REPO_ROOT / "experiments/stage146_gap_balanced_adapter_training_v2/stage146_gap_balanced_adapter_training_v2_summary.json"
STAGE154 = REPO_ROOT / "experiments/stage154_original_streamsplat_middle_base_alignment/stage154_original_streamsplat_middle_summary.csv"
STAGE157 = REPO_ROOT / "experiments/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_summary.csv"
STAGE196 = REPO_ROOT / "experiments/stage196_target_feasibility_branch/stage196_target_feasibility_summary.csv"

EVIDENCE_FIELDS = ["source_stage", "evidence", "gap", "metric", "value", "comparison", "interpretation"]
DECISION_FIELDS = ["item", "decision", "evidence", "action_for_new_route"]
REQUIREMENT_FIELDS = ["requirement", "reason", "stage_gate"]


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(row, key, default=0.0):
    value = row.get(key) if row else None
    if value in (None, "", "NA"):
        return default
    return float(value)


def find_row(rows, **kwargs):
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in kwargs.items()):
            return row
    raise KeyError(kwargs)


def evidence_rows():
    rows = []
    stage78 = read_csv(STAGE78)
    for gap in (4, 8):
        adapter = find_row(stage78, codec="q12", frame_gap=gap, method="adapter")
        linear = find_row(stage78, codec="q12", frame_gap=gap, method="linear")
        rows.append({
            "source_stage": 78,
            "evidence": "old_adapter_middle_psnr",
            "gap": gap,
            "metric": "mean_middle_psnr",
            "value": numeric(adapter, "mean_middle_psnr"),
            "comparison": f"linear={numeric(linear, 'mean_middle_psnr')}",
            "interpretation": "Old adapter gives only small gain over linear and remains far below recovery targets.",
        })
    stage143 = read_csv(STAGE143)
    for gap in (4, 8):
        q12 = find_row(stage143, codec="q12", reference_gap=gap, method="adapter")
        f32 = find_row(stage143, codec="float32", reference_gap=gap, method="adapter")
        dense = find_row(stage143, codec="float32", reference_gap=gap, method="dense_direct")
        rows.append({
            "source_stage": 143,
            "evidence": "qbit_sensitivity_adapter",
            "gap": gap,
            "metric": "float32_minus_q12_middle_psnr",
            "value": numeric(f32, "mean_middle_psnr") - numeric(q12, "mean_middle_psnr"),
            "comparison": f"q12={numeric(q12, 'mean_middle_psnr')}, float32={numeric(f32, 'mean_middle_psnr')}",
            "interpretation": "Raising anchor q-bit does not repair the old adapter.",
        })
        rows.append({
            "source_stage": 143,
            "evidence": "dense_direct_vs_adapter_ceiling",
            "gap": gap,
            "metric": "dense_direct_minus_adapter_middle_psnr",
            "value": numeric(dense, "mean_middle_psnr") - numeric(f32, "mean_middle_psnr"),
            "comparison": f"dense_direct={numeric(dense, 'mean_middle_psnr')}, adapter={numeric(f32, 'mean_middle_psnr')}",
            "interpretation": "The old adapter is the bottleneck, not renderer/data loading.",
        })
    for stage, path in [(145, STAGE145), (146, STAGE146)]:
        summary = read_json(path)
        initial = summary["initial_eval"]
        best = summary["best_eval"]
        final = summary["final_eval"]
        rows.append({
            "source_stage": stage,
            "evidence": "training_best_gain",
            "gap": "4,8",
            "metric": "best_mean_psnr_minus_initial",
            "value": best["model_psnr_avg"] - initial["model_psnr_avg"],
            "comparison": f"steps={summary['steps']}, selected_train={summary['selected_train_rows']}, selected_eval={summary['selected_eval_rows']}",
            "interpretation": "Training gain is tiny or absent under the old architecture/objective.",
        })
        rows.append({
            "source_stage": stage,
            "evidence": "training_final_change",
            "gap": "4,8",
            "metric": "final_mean_psnr_minus_initial",
            "value": final["model_psnr_avg"] - initial["model_psnr_avg"],
            "comparison": f"best_step={summary['best_step']}",
            "interpretation": "Longer continuation did not provide reliable improvement.",
        })
    stage154 = read_csv(STAGE154)
    for row in stage154:
        rows.append({
            "source_stage": 154,
            "evidence": "original_streamsplat_base",
            "gap": row["gap"],
            "metric": "mean_psnr",
            "value": numeric(row, "mean_psnr"),
            "comparison": f"lpips={numeric(row, 'mean_lpips')}",
            "interpretation": "StreamSplat base is more plausible than old adapter but still not enough without GS residual side-info.",
        })
    stage157 = read_csv(STAGE157)
    for row in stage157:
        rows.append({
            "source_stage": 157,
            "evidence": "stage158_residual_sideinfo_success",
            "gap": row["gap"],
            "metric": "mean_psnr",
            "value": numeric(row, "mean_psnr"),
            "comparison": f"payload_bytes={numeric(row, 'mean_payload_bytes')}, lpips={numeric(row, 'mean_lpips')}",
            "interpretation": "Quality gain came from counted GS-domain residual side-info, not from the old adapter predictor alone.",
        })
    stage196 = read_csv(STAGE196)
    best = max(stage196, key=lambda row: numeric(row, "psnr"))
    rows.append({
        "source_stage": 196,
        "evidence": "target_gap_remaining",
        "gap": "full_sequence",
        "metric": "best_ceiling_delta_psnr_vs_target",
        "value": numeric(best, "delta_psnr_vs_target"),
        "comparison": f"best_ceiling={best['ceiling']}, target={best['target_psnr_gap2_plus_1db']}",
        "interpretation": "Even best current keyframe/selector ceiling remains below the requested target.",
    })
    return rows


def decision_rows(evidence):
    stage145_gain = next(row for row in evidence if row["source_stage"] == 145 and row["evidence"] == "training_best_gain")
    stage146_gain = next(row for row in evidence if row["source_stage"] == 146 and row["evidence"] == "training_best_gain")
    return [
        {
            "item": "continue_old_adapter_training",
            "decision": "reject",
            "evidence": f"Stage145 best gain {stage145_gain['value']} dB; Stage146 best gain {stage146_gain['value']} dB with best step 0.",
            "action_for_new_route": "Do not spend Stage201+ on continuing GaussianAnchorDynamicPredictor unchanged.",
        },
        {
            "item": "raise_keyframe_qbits_for_old_adapter",
            "decision": "reject",
            "evidence": "Stage143 q12-to-float32 old adapter middle PSNR changes are about 0.000 dB.",
            "action_for_new_route": "Use qbit as rate/ablation variable only, not the primary quality fix.",
        },
        {
            "item": "use_stage158_as_evidence_predictor_is_solved",
            "decision": "reject",
            "evidence": "Stage157/158 quality is achieved by counted GS residual side-info over StreamSplat base, not by the old adapter alone.",
            "action_for_new_route": "Design predictor and residual codec jointly; measure predictor-only before selector training.",
        },
        {
            "item": "new_predictor_requirement",
            "decision": "require_new_architecture_and_loss",
            "evidence": "Old per-Gaussian MLP residual over linear interpolation underfits motion/occlusion and saturates.",
            "action_for_new_route": "Stage200 must propose a stronger temporal refiner/motion-field architecture with render-aware and RD-aware gates.",
        },
    ]


def requirement_rows():
    return [
        {
            "requirement": "predictor_only_gate_before_selector",
            "reason": "Selector cannot help if non-keyframe reconstruction is weak.",
            "stage_gate": "Stage201 must beat old adapter and show stable render metrics before Stage206+.",
        },
        {
            "requirement": "gs_native_residual_payload",
            "reason": "Stage158 showed residual side-info is the practical quality lever, but the final method must remain GS-domain.",
            "stage_gate": "Stage203/204 must define counted GS latent/residual bitstreams, not image residuals.",
        },
        {
            "requirement": "edge_rd_headroom_before_selector_training",
            "reason": "Stage193 showed selector training is wasted without candidate-space headroom.",
            "stage_gate": "Stage207 DP oracle must beat fixed baselines before Stage209 selector training is promoted.",
        },
        {
            "requirement": "full_sequence_metrics_only_for_strong_claim",
            "reason": "Stage177/180 sampled gains overstated full-sequence effect.",
            "stage_gate": "Stage211 must include full-sequence PSNR/SSIM/MS-SSIM/LPIPS and measured bytes.",
        },
    ]


def write_report(package, evidence, decisions, requirements, path):
    lines = [
        "# Stage198 Prior Predictor Training Audit",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        "- The old `GaussianAnchorDynamicPredictor` / Stage65 adapter route should not be continued unchanged.",
        "- Stage201+ must test a new predictor/refiner architecture and a GS-native residual codec before selector training.",
        "",
        "## Evidence",
        "",
        "| stage | evidence | gap | metric | value | interpretation |",
        "|---:|---|---:|---|---:|---|",
    ]
    for row in evidence:
        gap = row["gap"]
        lines.append(f"| {row['source_stage']} | {row['evidence']} | {gap} | {row['metric']} | {float(row['value']):.6f} | {row['interpretation']} |")
    lines.extend(["", "## Decisions", "", "| item | decision | action |", "|---|---|---|"])
    for row in decisions:
        lines.append(f"| {row['item']} | {row['decision']} | {row['action_for_new_route']} |")
    lines.extend(["", "## Requirements", "", "| requirement | stage gate |", "|---|---|"])
    for row in requirements:
        lines.append(f"| {row['requirement']} | {row['stage_gate']} |")
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Evidence CSV: `{package['evidence_csv']}`",
        f"- Decision CSV: `{package['decision_csv']}`",
        f"- Requirement CSV: `{package['requirement_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    evidence = evidence_rows()
    decisions = decision_rows(evidence)
    requirements = requirement_rows()
    evidence_csv = OUTPUT_ROOT / "stage198_prior_predictor_evidence.csv"
    decision_csv = OUTPUT_ROOT / "stage198_prior_predictor_decisions.csv"
    requirement_csv = OUTPUT_ROOT / "stage198_new_route_requirements.csv"
    package_json = OUTPUT_ROOT / "stage198_prior_predictor_training_audit_package.json"
    report_md = OUTPUT_ROOT / "stage198_prior_predictor_training_audit_report.md"
    write_csv(evidence, evidence_csv, EVIDENCE_FIELDS)
    write_csv(decisions, decision_csv, DECISION_FIELDS)
    write_csv(requirements, requirement_csv, REQUIREMENT_FIELDS)
    package = {
        "stage": 198,
        "status": "prior_predictor_training_audit_complete",
        "decision": "old_adapter_route_rejected_new_predictor_required",
        "evidence_rows": evidence,
        "decision_rows": decisions,
        "requirement_rows": requirements,
        "evidence_csv": str(evidence_csv.relative_to(REPO_ROOT)),
        "decision_csv": str(decision_csv.relative_to(REPO_ROOT)),
        "requirement_csv": str(requirement_csv.relative_to(REPO_ROOT)),
        "package_json": str(package_json.relative_to(REPO_ROOT)),
        "report_md": str(report_md.relative_to(REPO_ROOT)),
        "next": "Build Stage199 learned GS training manifest under Stage197 decoder contract.",
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, evidence, decisions, requirements, report_md)
    print(json.dumps({"package": str(package_json), "decision": package["decision"]}, indent=2))


if __name__ == "__main__":
    main()
