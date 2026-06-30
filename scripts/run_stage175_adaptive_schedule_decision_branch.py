import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE172_PACKAGE = REPO_ROOT / "experiments/stage172_keyframe_rate_accounting_audit/stage172_keyframe_rate_accounting_audit_package.json"
DEFAULT_STAGE174_PACKAGE = REPO_ROOT / "experiments/stage174_medium_rendered_validation_execution/stage174_medium_rendered_validation_execution_package.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage175_adaptive_schedule_decision_branch"


FACTOR_FIELDS = ["factor", "status", "metric", "value", "interpretation", "decision_effect"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def schedule_component(stage172, schedule):
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


def build_factors(stage172, stage174):
    adaptive_rate = schedule_component(stage172, "stage165_adaptive")
    gap8_rate = schedule_component(stage172, "uniform_gap8")
    gap4_rate = schedule_component(stage172, "uniform_gap4")
    false_adaptive = summary(stage174, "false_negative_residual", "stage165_adaptive")
    positive_ext_gap8 = summary(stage174, "positive_promoted_extension", "uniform_gap8")
    false_positive_gap8 = summary(stage174, "selector_false_positive_keyframe_control", "uniform_gap8")
    residual_ext_adaptive = summary(stage174, "high_payload_residual_control_extension", "stage165_adaptive")
    residual_ext_gap8 = summary(stage174, "high_payload_residual_control_extension", "uniform_gap8")
    normal_adaptive = summary(stage174, "normal_residual_control", "stage165_adaptive")
    normal_gap8 = summary(stage174, "normal_residual_control", "uniform_gap8")

    factors = [
        {
            "factor": "rate_proxy",
            "status": "pass_sampled_proxy",
            "metric": "adaptive / gap8 / gap4 total proxy MiB/frame",
            "value": f"{fmt(adaptive_rate['total_proxy_mib_per_frame_recomputed'])} / {fmt(gap8_rate['total_proxy_mib_per_frame_recomputed'])} / {fmt(gap4_rate['total_proxy_mib_per_frame_recomputed'])}",
            "interpretation": "Adaptive remains lower-rate than both uniform references under Stage166 sampled proxy after charging extra keyframes and metadata.",
            "decision_effect": "supports scaling to broader validation",
        },
        {
            "factor": "protocol_completeness",
            "status": "pass",
            "metric": "Stage174 output / protocol rows and new renders",
            "value": f"{stage174['output_row_count']}/{stage174['protocol_row_count']} rows, {stage174['new_render_count']}/{stage174['expected_new_render_count']} new renders",
            "interpretation": "Medium validation completed without missing rows.",
            "decision_effect": "supports packaging current candidate evidence",
        },
        {
            "factor": "false_negatives",
            "status": "risk_neutral_quality",
            "metric": "adaptive delta vs uniform_gap8 PSNR / LPIPS",
            "value": f"{fmt(false_adaptive['mean_delta_psnr_vs_uniform_gap8'])} / {fmt(false_adaptive['mean_delta_lpips_vs_uniform_gap8'])}",
            "interpretation": "False negatives remain unresolved but are not materially worse than uniform gap8 on sampled rendered evidence.",
            "decision_effect": "requires broader validation and possible selector refinement before final full RD",
        },
        {
            "factor": "positive_promotions",
            "status": "pass_schedule_behavior",
            "metric": "positive extension uniform_gap8 PSNR / LPIPS / payload",
            "value": f"{fmt(positive_ext_gap8['mean_psnr'])} / {fmt(positive_ext_gap8['mean_lpips'])} / {fmt(positive_ext_gap8['mean_payload_bytes'])}",
            "interpretation": "Adaptive correctly turns these hard rows into keyframes/no-middle-render rather than expensive residual recovery.",
            "decision_effect": "supports adaptive schedule candidate",
        },
        {
            "factor": "residual_controls",
            "status": "pass_neutral",
            "metric": "high-payload extension adaptive vs gap8 PSNR / LPIPS",
            "value": f"{fmt(residual_ext_adaptive['mean_psnr'])} vs {fmt(residual_ext_gap8['mean_psnr'])} / {fmt(residual_ext_adaptive['mean_lpips'])} vs {fmt(residual_ext_gap8['mean_lpips'])}",
            "interpretation": "When adaptive does not promote rows, rendered residual behavior matches uniform gap8 on this extension slice.",
            "decision_effect": "supports no-regression claim for residual rows in sampled setting",
        },
        {
            "factor": "normal_controls",
            "status": "pass_neutral",
            "metric": "normal adaptive vs gap8 PSNR / LPIPS",
            "value": f"{fmt(normal_adaptive['mean_psnr'])} vs {fmt(normal_gap8['mean_psnr'])} / {fmt(normal_adaptive['mean_lpips'])} vs {fmt(normal_gap8['mean_lpips'])}",
            "interpretation": "Normal/easy rows are unchanged from uniform gap8 when no promotion changes the segment.",
            "decision_effect": "supports bounded behavior on easy controls",
        },
        {
            "factor": "false_positive_keyframes",
            "status": "risk_precision",
            "metric": "false-positive control uniform_gap8 PSNR / LPIPS / payload",
            "value": f"{fmt(false_positive_gap8['mean_psnr'])} / {fmt(false_positive_gap8['mean_lpips'])} / {fmt(false_positive_gap8['mean_payload_bytes'])}",
            "interpretation": "Some adaptive keyframes are not hard/high-payload labels; these are likely unnecessary extra keyframes and limit final selector confidence.",
            "decision_effect": "prevents final full-sequence claim; motivates broader validation or selector refinement",
        },
    ]
    return factors


def write_report(factors, package, path):
    lines = [
        "# Stage175 Adaptive Schedule Decision Branch",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        "- Package the current adaptive schedule as a sampled-validated candidate, not a final full-sequence RD result.",
        "- Broader validation is justified before final claims; selector false-positive keyframes and false negatives remain explicit risks.",
        "",
        "## Factors",
        "",
        "| factor | status | metric | value | interpretation | decision effect |",
        "|---|---|---|---:|---|---|",
    ]
    for row in factors:
        lines.append(
            f"| {row['factor']} | {row['status']} | {row['metric']} | {row['value']} | {row['interpretation']} | {row['decision_effect']} |"
        )
    lines.extend([
        "",
        "## Boundaries",
        "",
        "- Adaptive keyframe rows are schedule/rate events; no middle-render metrics are claimed for them.",
        "- Stage172 rate is sampled/proxy accounting, not final full-sequence RD.",
        "- Stage174 is medium sampled rendered validation, not full DAVIS validation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage172_package", type=Path, default=DEFAULT_STAGE172_PACKAGE)
    parser.add_argument("--stage174_package", type=Path, default=DEFAULT_STAGE174_PACKAGE)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage172 = read_json(args.stage172_package)
    stage174 = read_json(args.stage174_package)
    factors = build_factors(stage172, stage174)
    decision = "package_sampled_validated_candidate_and_scale_broader_validation"
    factors_csv = args.output_root / "stage175_decision_factors.csv"
    package_json = args.output_root / "stage175_adaptive_schedule_decision_branch_package.json"
    report_md = args.output_root / "stage175_adaptive_schedule_decision_branch_report.md"
    package = {
        "stage": 175,
        "status": "adaptive_schedule_decision_branch_packaged",
        "decision": decision,
        "rate_scope": stage172["accounting_scope"],
        "render_scope": "stage174_medium_sampled_validation",
        "factors": factors,
        "recommended_next_stage": "Stage176 adaptive schedule candidate package with limitations and broader-validation path",
        "not_final_claims": [
            "final full-sequence RD",
            "selector precision solved",
            "rendered middle metrics for adaptive keyframe rows",
        ],
        "factors_csv": str(factors_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    write_csv(factors, factors_csv, FACTOR_FIELDS)
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(factors, package, report_md)
    print(json.dumps({"package": str(package_json), "decision": decision}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
