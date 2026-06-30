import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE162_PACKAGE = REPO_ROOT / "experiments/stage162_keyframe_selector_protocol_source_audit/stage162_keyframe_selector_protocol_source_audit_package.json"
DEFAULT_STAGE165_PACKAGE = REPO_ROOT / "experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_multifeature_keyframe_schedule_preflight_package.json"
DEFAULT_STAGE172_PACKAGE = REPO_ROOT / "experiments/stage172_keyframe_rate_accounting_audit/stage172_keyframe_rate_accounting_audit_package.json"
DEFAULT_STAGE174_PACKAGE = REPO_ROOT / "experiments/stage174_medium_rendered_validation_execution/stage174_medium_rendered_validation_execution_package.json"
DEFAULT_STAGE175_PACKAGE = REPO_ROOT / "experiments/stage175_adaptive_schedule_decision_branch/stage175_adaptive_schedule_decision_branch_package.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage176_adaptive_schedule_candidate_package"


EVIDENCE_FIELDS = ["area", "status", "metric", "value", "source"]
LIMITATION_FIELDS = ["limitation", "impact", "required_followup"]
NEXT_FIELDS = ["step", "goal", "requirements"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def component(stage172, schedule):
    for row in stage172["component_rows"]:
        if row["schedule"] == schedule:
            return row
    raise KeyError(schedule)


def summary(stage174, category, schedule):
    for row in stage174["summary_rows"]:
        if row["category"] == category and row["schedule"] == schedule:
            return row
    raise KeyError((category, schedule))


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def build_policy(stage162, stage165, stage172, stage175):
    selected = stage165["selected_policy"]
    schedule_summary = stage165["schedule_summary"]
    adaptive = component(stage172, "stage165_adaptive")
    return {
        "policy_name": "rgb_motion_rank_gate_gap8_plus_extra_targets_v1_sampled_candidate",
        "status": "sampled_validated_candidate_not_final_full_sequence_rd",
        "fixed_middle_recovery_policy": stage165["fixed_middle_recovery_policy"],
        "schedule_logic": stage165["schedule_logic"],
        "selector_candidate": selected["candidate"],
        "rank_threshold": selected["rank_threshold"],
        "min_votes": selected["min_votes"],
        "feature_scope": stage165["inference_feature_scope"],
        "encoder_allowed_inputs": [
            "input RGB frames",
            "deterministic RGB/motion proxy features from input frames",
            "optional fixed RGB-only pretrained motion/feature networks in a higher-compute tier",
        ],
        "decoder_allowed_inputs": [
            "transmitted keyframe schedule or keyframe indices",
            "normal keyframe anchors/payloads",
            "Stage158 residual payloads for non-keyframe middle recovery rows",
            "normal StreamSplat endpoint inputs and normalized time for recovered middle rows",
        ],
        "decoder_forbidden_inputs": [
            "RGB/motion selector features",
            "target dense anchors except through encoded Stage158 residual payload",
            "target RGB",
            "rendered quality metrics or oracle labels",
            "unencoded target residual tensors",
        ],
        "rate_accounting_contract": stage162["rate_accounting"],
        "schedule_summary": schedule_summary,
        "sampled_rate_proxy": {
            "scope": stage172["accounting_scope"],
            "total_proxy_mib_per_frame": adaptive["total_proxy_mib_per_frame_recomputed"],
            "keyframe_count": adaptive["keyframe_count"],
            "extra_keyframes_vs_gap8": adaptive["keyframe_delta_vs_uniform_gap8"],
            "metadata_bytes": adaptive["metadata_bytes"],
            "sampled_residual_row_count": adaptive["sampled_residual_row_count"],
            "sampled_promoted_row_count": adaptive["sampled_promoted_row_count"],
        },
        "stage175_decision": stage175["decision"],
    }


def build_evidence(stage165, stage172, stage174, stage175):
    selected = stage165["selected_policy"]
    sched = stage165["schedule_summary"]
    gap8 = component(stage172, "uniform_gap8")
    adaptive = component(stage172, "stage165_adaptive")
    gap4 = component(stage172, "uniform_gap4")
    false_adaptive = summary(stage174, "false_negative_residual", "stage165_adaptive")
    positive_gap8 = summary(stage174, "positive_promoted_extension", "uniform_gap8")
    false_positive_gap8 = summary(stage174, "selector_false_positive_keyframe_control", "uniform_gap8")
    return [
        {
            "area": "selector_policy",
            "status": "selected_candidate",
            "metric": "rank threshold / min votes / selected rows",
            "value": f"{selected['rank_threshold']} / {selected['min_votes']} / {selected['selected_count']}",
            "source": "Stage165",
        },
        {
            "area": "selector_recall",
            "status": "positive_signal",
            "metric": "hard recall / payload recall",
            "value": f"{fmt(selected['recall_hard'])} / {fmt(selected['recall_payload'])}",
            "source": "Stage165",
        },
        {
            "area": "schedule_size",
            "status": "bounded_between_uniforms",
            "metric": "adaptive keyframes / frames / metadata bytes",
            "value": f"{sched['total_keyframe_count']} / {sched['total_frame_count']} / {sched['metadata_bytes']}",
            "source": "Stage165",
        },
        {
            "area": "rate_proxy",
            "status": "rate_promising_sampled_proxy",
            "metric": "adaptive / gap8 / gap4 MiB/frame",
            "value": f"{fmt(adaptive['total_proxy_mib_per_frame_recomputed'])} / {fmt(gap8['total_proxy_mib_per_frame_recomputed'])} / {fmt(gap4['total_proxy_mib_per_frame_recomputed'])}",
            "source": "Stage172",
        },
        {
            "area": "medium_validation",
            "status": "complete",
            "metric": "protocol rows / new renders",
            "value": f"{stage174['output_row_count']} / {stage174['new_render_count']}",
            "source": "Stage174",
        },
        {
            "area": "false_negatives",
            "status": "risk_neutral_quality",
            "metric": "adaptive delta vs gap8 PSNR / LPIPS",
            "value": f"{fmt(false_adaptive['mean_delta_psnr_vs_uniform_gap8'])} / {fmt(false_adaptive['mean_delta_lpips_vs_uniform_gap8'])}",
            "source": "Stage174",
        },
        {
            "area": "positive_promotions",
            "status": "supports_schedule",
            "metric": "positive extension gap8 PSNR / LPIPS / payload",
            "value": f"{fmt(positive_gap8['mean_psnr'])} / {fmt(positive_gap8['mean_lpips'])} / {fmt(positive_gap8['mean_payload_bytes'])}",
            "source": "Stage174",
        },
        {
            "area": "false_positive_keyframes",
            "status": "risk_precision",
            "metric": "false-positive controls gap8 PSNR / LPIPS / payload",
            "value": f"{fmt(false_positive_gap8['mean_psnr'])} / {fmt(false_positive_gap8['mean_lpips'])} / {fmt(false_positive_gap8['mean_payload_bytes'])}",
            "source": "Stage174/175",
        },
        {
            "area": "decision",
            "status": "candidate_package",
            "metric": "Stage175 branch",
            "value": stage175["decision"],
            "source": "Stage175",
        },
    ]


def build_limitations():
    return [
        {
            "limitation": "not_final_full_sequence_rd",
            "impact": "Current rate is sampled/proxy and rendered evidence is medium sampled validation.",
            "required_followup": "Run broader or full-sequence RD with all keyframe, metadata, and residual payload counted.",
        },
        {
            "limitation": "selector_false_positive_keyframes",
            "impact": "Some promoted targets are not hard/high-payload labels and may waste extra keyframes.",
            "required_followup": "Broader validation should quantify false-positive keyframe overhead; selector refinement may add per-sequence budgets or stricter gates.",
        },
        {
            "limitation": "false_negatives_remain",
            "impact": "Hard rows missed by the selector remain essentially uniform-gap8 residual cases.",
            "required_followup": "Keep false-negative stress sets in broader validation and consider selector features that improve recall without large false-positive cost.",
        },
        {
            "limitation": "adaptive_keyframe_rows_have_no_middle_metrics",
            "impact": "Promoted rows cannot be compared by middle-render PSNR/LPIPS; they are transmitted keyframes.",
            "required_followup": "Judge promoted rows through keyframe rate, all-frame sequence metrics, and subjective keyframe continuity.",
        },
        {
            "limitation": "offline_feedforward_scope",
            "impact": "DAVIS experiments use offline input RGB; online streaming variants need declared lookahead.",
            "required_followup": "Document lookahead or restrict selector features for online settings.",
        },
    ]


def build_next_steps():
    return [
        {
            "step": "broader_sampled_validation",
            "goal": "Scale beyond the 50-target medium protocol before final claims.",
            "requirements": "Reuse existing rows where possible; include false negatives, false-positive keyframes, high-payload controls, normal controls, and weak subjective sequences.",
        },
        {
            "step": "full_sequence_rd_accounting",
            "goal": "Convert sampled proxy rate into full sequence/frame accounting.",
            "requirements": "Count all keyframe anchors, adaptive schedule metadata, Stage158 residual payloads for every non-keyframe recovery, and report MiB/frame.",
        },
        {
            "step": "all_frame_quality_report",
            "goal": "Move beyond target-row summaries.",
            "requirements": "Report all-frame, keyframe-only, middle-only, per-sequence, and per-category PSNR/SSIM/MS-SSIM/LPIPS.",
        },
        {
            "step": "selector_refinement_if_needed",
            "goal": "Reduce false-positive keyframes or false negatives if broader validation shows rate/quality issues.",
            "requirements": "Stay within Stage162-allowed encoder-side RGB/motion features; no rendered quality or target dense anchors as inference inputs.",
        },
    ]


def write_report(policy, evidence, limitations, next_steps, package, path):
    lines = [
        "# Stage176 Adaptive Schedule Candidate Package",
        "",
        "## Status",
        "",
        f"- Policy: `{policy['policy_name']}`.",
        "- Status: sampled-validated candidate, not final full-sequence RD.",
        f"- Stage175 decision: `{policy['stage175_decision']}`.",
        "",
        "## Decoder Contract",
        "",
        "- Decoder receives the transmitted keyframe schedule/keyframe indices and normal keyframe payloads.",
        "- Decoder does not compute or receive RGB/motion selector features.",
        "- Non-keyframe middle rows use fixed Stage158 recovery with counted residual payload and half selector.",
        "- Target RGB, target dense anchors, rendered metrics, oracle labels, and unencoded residuals are forbidden inference inputs.",
        "",
        "## Evidence",
        "",
        "| area | status | metric | value | source |",
        "|---|---|---|---:|---|",
    ]
    for row in evidence:
        lines.append(f"| {row['area']} | {row['status']} | {row['metric']} | {row['value']} | {row['source']} |")
    lines.extend(["", "## Limitations", ""])
    for row in limitations:
        lines.append(f"- `{row['limitation']}`: {row['impact']} Follow-up: {row['required_followup']}")
    lines.extend(["", "## Next Validation", ""])
    for row in next_steps:
        lines.append(f"- `{row['step']}`: {row['goal']} Requirements: {row['requirements']}")
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Candidate policy JSON: `{package['candidate_policy_json']}`",
        f"- Evidence CSV: `{package['evidence_csv']}`",
        f"- Limitations CSV: `{package['limitations_csv']}`",
        f"- Next validation CSV: `{package['next_validation_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage162_package", type=Path, default=DEFAULT_STAGE162_PACKAGE)
    parser.add_argument("--stage165_package", type=Path, default=DEFAULT_STAGE165_PACKAGE)
    parser.add_argument("--stage172_package", type=Path, default=DEFAULT_STAGE172_PACKAGE)
    parser.add_argument("--stage174_package", type=Path, default=DEFAULT_STAGE174_PACKAGE)
    parser.add_argument("--stage175_package", type=Path, default=DEFAULT_STAGE175_PACKAGE)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage162 = read_json(args.stage162_package)
    stage165 = read_json(args.stage165_package)
    stage172 = read_json(args.stage172_package)
    stage174 = read_json(args.stage174_package)
    stage175 = read_json(args.stage175_package)
    policy = build_policy(stage162, stage165, stage172, stage175)
    evidence = build_evidence(stage165, stage172, stage174, stage175)
    limitations = build_limitations()
    next_steps = build_next_steps()
    candidate_policy_json = args.output_root / "stage176_adaptive_keyframe_schedule_candidate_policy.json"
    evidence_csv = args.output_root / "stage176_candidate_evidence.csv"
    limitations_csv = args.output_root / "stage176_candidate_limitations.csv"
    next_validation_csv = args.output_root / "stage176_next_validation_requirements.csv"
    package_json = args.output_root / "stage176_adaptive_schedule_candidate_package.json"
    report_md = args.output_root / "stage176_adaptive_schedule_candidate_package_report.md"
    write_csv(evidence, evidence_csv, EVIDENCE_FIELDS)
    write_csv(limitations, limitations_csv, LIMITATION_FIELDS)
    write_csv(next_steps, next_validation_csv, NEXT_FIELDS)
    candidate_policy_json.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    package = {
        "stage": 176,
        "status": "adaptive_schedule_candidate_packaged",
        "candidate_status": policy["status"],
        "policy_name": policy["policy_name"],
        "stage175_decision": stage175["decision"],
        "candidate_policy_json": str(candidate_policy_json),
        "evidence_csv": str(evidence_csv),
        "limitations_csv": str(limitations_csv),
        "next_validation_csv": str(next_validation_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "evidence": evidence,
        "limitations": limitations,
        "next_steps": next_steps,
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(policy, evidence, limitations, next_steps, package, report_md)
    print(json.dumps({"package": str(package_json), "policy": policy["policy_name"], "status": policy["status"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
