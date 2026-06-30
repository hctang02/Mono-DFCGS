import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE176_PACKAGE = REPO_ROOT / "experiments/stage176_adaptive_schedule_candidate_package/stage176_adaptive_schedule_candidate_package.json"
DEFAULT_STAGE180_PACKAGE = REPO_ROOT / "experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_broader_sampled_adaptive_validation_execution_package.json"
DEFAULT_STAGE180_DELTAS = REPO_ROOT / "experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_category_delta_summary.csv"
DEFAULT_STAGE181_PACKAGE = REPO_ROOT / "experiments/stage181_full_sequence_rd_accounting_preflight/stage181_full_sequence_rd_accounting_preflight_package.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage182_selector_refinement_or_freeze_decision"

EVIDENCE_FIELDS = ["area", "status", "metric", "value", "source", "decision_weight"]
RISK_FIELDS = ["risk", "status", "evidence", "mitigation_or_next_step"]
NEXT_FIELDS = ["step", "goal", "required_before_final_claim"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def row_by_group(rows, group):
    return next(row for row in rows if row["group"] == group)


def total_proxy_row(stage181, schedule):
    return next(row for row in stage181["total_proxy_rows"] if row["schedule"] == schedule)


def build_evidence(stage176, stage180, deltas, stage181):
    overall = row_by_group(deltas, "overall")
    false_negative = row_by_group(deltas, "false_negative_residual")
    false_positive = row_by_group(deltas, "selector_false_positive_keyframe_control")
    broader_positive = row_by_group(deltas, "broader_positive_promoted")
    broader_weak = row_by_group(deltas, "broader_weak_sequence_probe")
    adaptive_rate = total_proxy_row(stage181, "stage165_adaptive")
    gap8_rate = total_proxy_row(stage181, "uniform_gap8")
    gap4_rate = total_proxy_row(stage181, "uniform_gap4")
    return [
        {
            "area": "candidate_policy",
            "status": "current_candidate",
            "metric": "policy name",
            "value": stage176["policy_name"],
            "source": "Stage176",
            "decision_weight": "freeze_support",
        },
        {
            "area": "broader_quality",
            "status": "positive",
            "metric": "adaptive PSNR delta vs gap8 / gap4",
            "value": f"{overall['mean_delta_psnr_vs_gap8']} / {overall['mean_delta_psnr_vs_gap4']}",
            "source": "Stage180 overall 90 targets",
            "decision_weight": "strong_freeze_support",
        },
        {
            "area": "broader_perceptual",
            "status": "positive",
            "metric": "adaptive LPIPS delta vs gap8 / gap4",
            "value": f"{overall['mean_delta_lpips_vs_gap8']} / {overall['mean_delta_lpips_vs_gap4']}",
            "source": "Stage180 overall 90 targets",
            "decision_weight": "strong_freeze_support",
        },
        {
            "area": "rate_proxy",
            "status": "positive",
            "metric": "adaptive/gap8/gap4 total proxy MiB/frame",
            "value": f"{adaptive_rate['stage180_broader_total_proxy_mib_per_frame']} / {gap8_rate['stage180_broader_total_proxy_mib_per_frame']} / {gap4_rate['stage180_broader_total_proxy_mib_per_frame']}",
            "source": "Stage181",
            "decision_weight": "strong_freeze_support",
        },
        {
            "area": "positive_promotions",
            "status": "positive",
            "metric": "broader_positive_promoted PSNR delta vs gap8 / gap4",
            "value": f"{broader_positive['mean_delta_psnr_vs_gap8']} / {broader_positive['mean_delta_psnr_vs_gap4']}",
            "source": "Stage180 category deltas",
            "decision_weight": "freeze_support",
        },
        {
            "area": "weak_sequences",
            "status": "positive",
            "metric": "broader_weak_sequence_probe PSNR delta vs gap8 / gap4",
            "value": f"{broader_weak['mean_delta_psnr_vs_gap8']} / {broader_weak['mean_delta_psnr_vs_gap4']}",
            "source": "Stage180 category deltas",
            "decision_weight": "freeze_support",
        },
        {
            "area": "false_negatives",
            "status": "remaining_risk_but_bounded",
            "metric": "false_negative_residual PSNR delta vs gap8 / gap4",
            "value": f"{false_negative['mean_delta_psnr_vs_gap8']} / {false_negative['mean_delta_psnr_vs_gap4']}",
            "source": "Stage180 category deltas",
            "decision_weight": "risk_track_not_blocking_freeze",
        },
        {
            "area": "false_positive_keyframes",
            "status": "precision_risk_but_quality_positive",
            "metric": "false-positive control PSNR delta vs gap8 / gap4",
            "value": f"{false_positive['mean_delta_psnr_vs_gap8']} / {false_positive['mean_delta_psnr_vs_gap4']}",
            "source": "Stage180 category deltas",
            "decision_weight": "risk_track_not_blocking_freeze",
        },
        {
            "area": "rd_claim_scope",
            "status": "not_final_full_sequence_rd",
            "metric": "residual payload scope",
            "value": stage181["accounting_scope"],
            "source": "Stage181",
            "decision_weight": "requires_next_measurement",
        },
        {
            "area": "decoder_contract",
            "status": "unchanged_valid",
            "metric": "decoder receives schedule not RGB/motion features",
            "value": "schedule_metadata_transmitted_no_selector_feature_at_decoder",
            "source": "Stage162/176/181",
            "decision_weight": "freeze_support",
        },
    ]


def build_risks():
    return [
        {
            "risk": "false_negatives_remain",
            "status": "known_bounded_risk",
            "evidence": "Stage180 false-negative residual delta vs gap8 is near neutral but remains below gap4.",
            "mitigation_or_next_step": "Keep false-negative stress categories in final payload/full-sequence measurement; do not claim recall solved.",
        },
        {
            "risk": "false_positive_keyframes",
            "status": "not_blocking_current_freeze",
            "evidence": "Stage180 false-positive keyframe controls improve final quality but may still waste keyframes in final RD.",
            "mitigation_or_next_step": "Track exact keyframe payload in full RD; only tune threshold if actual bitstreams show rate regression.",
        },
        {
            "risk": "sampled_residual_proxy",
            "status": "blocks_final_rd_claim",
            "evidence": "Stage181 residual payload is Stage180 broader sampled estimate, not all-frame measurement.",
            "mitigation_or_next_step": "Run all-frame/full-sequence residual payload encode for non-keyframe recovered frames.",
        },
        {
            "risk": "main_anchor_proxy",
            "status": "blocks_paper_level_rd_claim",
            "evidence": "Stage181 main-anchor payload is inherited from Stage172 proxy/interpolation.",
            "mitigation_or_next_step": "Measure actual q12 keyframe bitstreams for every transmitted keyframe.",
        },
        {
            "risk": "online_streaming_scope",
            "status": "not_claimed",
            "evidence": "Selector uses offline encoder-side RGB/motion features unless lookahead is specified.",
            "mitigation_or_next_step": "State offline encoding scope or define lookahead for streaming experiments.",
        },
    ]


def build_next_steps():
    return [
        {
            "step": "freeze_current_candidate",
            "goal": "Treat Stage165 adaptive selector as the current frozen candidate for the next RD measurement.",
            "required_before_final_claim": "No; this is the decision for the next stage.",
        },
        {
            "step": "full_sequence_payload_measurement",
            "goal": "Measure actual q12 keyframe bitstreams and all-frame Stage158 residual payloads under gap8/adaptive/gap4 schedules.",
            "required_before_final_claim": "Yes.",
        },
        {
            "step": "all_frame_quality_report",
            "goal": "Report all-frame, keyframe-only, middle-only, per-sequence and per-category quality.",
            "required_before_final_claim": "Yes.",
        },
        {
            "step": "selector_threshold_tuning",
            "goal": "Only revisit threshold/min-votes if actual full payload measurement shows rate regression or unacceptable false-positive cost.",
            "required_before_final_claim": "Conditional.",
        },
    ]


def write_report(evidence, risks, next_steps, package, path):
    lines = [
        "# Stage182 Selector Refinement Or Freeze Decision",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Frozen policy: `{package['frozen_policy_name']}`.",
        "- Rationale: Stage180 broader quality and Stage181 rate proxy are both positive, while remaining risks are final-RD measurement risks rather than reasons to tune the selector immediately.",
        "",
        "## Evidence",
        "",
        "| area | status | metric | value | source | weight |",
        "|---|---|---|---|---|---|",
    ]
    for row in evidence:
        lines.append(f"| {row['area']} | {row['status']} | {row['metric']} | {row['value']} | {row['source']} | {row['decision_weight']} |")
    lines.extend([
        "",
        "## Risks",
        "",
        "| risk | status | evidence | mitigation / next step |",
        "|---|---|---|---|",
    ])
    for row in risks:
        lines.append(f"| {row['risk']} | {row['status']} | {row['evidence']} | {row['mitigation_or_next_step']} |")
    lines.extend([
        "",
        "## Next Steps",
        "",
        "| step | goal | required before final claim |",
        "|---|---|---|",
    ])
    for row in next_steps:
        lines.append(f"| {row['step']} | {row['goal']} | {row['required_before_final_claim']} |")
    lines.extend([
        "",
        "## Non-Claims",
        "",
        "- This does not claim final full-sequence RD.",
        "- This does not claim false negatives are solved.",
        "- This does not claim selector precision is optimal.",
        "- This does not allow target dense anchors, target RGB, rendered quality/oracle labels, or unencoded residuals as decoder-side inputs.",
        "",
        "## Outputs",
        "",
        f"- Evidence CSV: `{package['evidence_csv']}`",
        f"- Risks CSV: `{package['risks_csv']}`",
        f"- Next steps CSV: `{package['next_steps_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage176_package", type=Path, default=DEFAULT_STAGE176_PACKAGE)
    parser.add_argument("--stage180_package", type=Path, default=DEFAULT_STAGE180_PACKAGE)
    parser.add_argument("--stage180_deltas", type=Path, default=DEFAULT_STAGE180_DELTAS)
    parser.add_argument("--stage181_package", type=Path, default=DEFAULT_STAGE181_PACKAGE)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage176 = read_json(args.stage176_package)
    stage180 = read_json(args.stage180_package)
    deltas = read_csv(args.stage180_deltas)
    stage181 = read_json(args.stage181_package)
    evidence = build_evidence(stage176, stage180, deltas, stage181)
    risks = build_risks()
    next_steps = build_next_steps()
    decision = "freeze_current_candidate_and_run_full_sequence_payload_measurement_next"
    evidence_csv = args.output_root / "stage182_freeze_decision_evidence.csv"
    risks_csv = args.output_root / "stage182_freeze_decision_risks.csv"
    next_steps_csv = args.output_root / "stage182_freeze_decision_next_steps.csv"
    package_json = args.output_root / "stage182_selector_refinement_or_freeze_decision_package.json"
    report_md = args.output_root / "stage182_selector_refinement_or_freeze_decision_report.md"
    write_csv(evidence, evidence_csv, EVIDENCE_FIELDS)
    write_csv(risks, risks_csv, RISK_FIELDS)
    write_csv(next_steps, next_steps_csv, NEXT_FIELDS)
    package = {
        "stage": 182,
        "status": "selector_refinement_or_freeze_decision_packaged",
        "decision": decision,
        "frozen_policy_name": stage176["policy_name"],
        "next_required_stage": "full_sequence_payload_measurement",
        "stage180_decision": stage180["decision"],
        "stage181_decision": stage181["decision"],
        "quality_delta_psnr_vs_gap8": stage180["mean_delta_psnr_vs_uniform_gap8"],
        "quality_delta_psnr_vs_gap4": stage180["mean_delta_psnr_vs_uniform_gap4"],
        "rate_delta_proxy_vs_gap8": stage181["adaptive_delta_total_proxy_vs_gap8"],
        "rate_delta_proxy_vs_gap4": stage181["adaptive_delta_total_proxy_vs_gap4"],
        "not_final_full_sequence_rd": True,
        "evidence": evidence,
        "risks": risks,
        "next_steps": next_steps,
        "evidence_csv": str(evidence_csv),
        "risks_csv": str(risks_csv),
        "next_steps_csv": str(next_steps_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(evidence, risks, next_steps, package, report_md)
    print(json.dumps({"package": str(package_json), "decision": decision}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
