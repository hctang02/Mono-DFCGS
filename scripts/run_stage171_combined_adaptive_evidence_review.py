import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE165_PACKAGE = REPO_ROOT / "experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_multifeature_keyframe_schedule_preflight_package.json"
DEFAULT_STAGE166_PACKAGE = REPO_ROOT / "experiments/stage166_adaptive_schedule_label_rd_comparison/stage166_adaptive_schedule_label_rd_comparison_package.json"
DEFAULT_STAGE170_PACKAGE = REPO_ROOT / "experiments/stage170_combined_adaptive_rendered_validation_execution/stage170_combined_adaptive_rendered_validation_execution_package.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage171_combined_adaptive_evidence_review"


EVIDENCE_FIELDS = [
    "area",
    "status",
    "metric",
    "value",
    "interpretation",
    "next_action",
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def get_schedule(stage166, schedule):
    for row in stage166["schedule_comparison"]:
        if row["schedule"] == schedule:
            return row
    raise KeyError(schedule)


def get_summary(stage170, category, schedule):
    for row in stage170["summary_rows"]:
        if row["category"] == category and row["schedule"] == schedule:
            return row
    raise KeyError((category, schedule))


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def build_evidence(stage165, stage166, stage170):
    selected = stage165["selected_policy"]
    adaptive = get_schedule(stage166, "stage165_adaptive")
    gap8 = get_schedule(stage166, "uniform_gap8")
    gap4 = get_schedule(stage166, "uniform_gap4")
    false_adaptive = get_summary(stage170, "false_negative_residual", "stage165_adaptive")
    false_gap8 = get_summary(stage170, "false_negative_residual", "uniform_gap8")
    positive_adaptive = get_summary(stage170, "positive_promoted", "stage165_adaptive")
    positive_gap8 = get_summary(stage170, "positive_promoted", "uniform_gap8")
    control_adaptive = get_summary(stage170, "high_payload_residual_control", "stage165_adaptive")
    control_gap8 = get_summary(stage170, "high_payload_residual_control", "uniform_gap8")

    rows = [
        {
            "area": "selector_coverage",
            "status": "positive_signal",
            "metric": "hard_recall / payload_recall",
            "value": f"{fmt(selected['recall_hard'])} / {fmt(selected['recall_payload'])}",
            "interpretation": "RGB/motion rank gate catches most sampled hard and high-payload labels, but precision is modest.",
            "next_action": "Proceed, but keep false-negative controls in the next protocol.",
        },
        {
            "area": "schedule_size",
            "status": "bounded_between_uniforms",
            "metric": "keyframes adaptive / gap8 / gap4",
            "value": f"{adaptive['total_keyframe_count']} / {gap8['total_keyframe_count']} / {gap4['total_keyframe_count']}",
            "interpretation": "Adaptive adds keyframes over uniform gap8 but remains much smaller than uniform gap4.",
            "next_action": "Audit extra-keyframe rate before larger rendering.",
        },
        {
            "area": "sampled_proxy_rate",
            "status": "promising_proxy",
            "metric": "total_proxy_mib_per_frame adaptive vs gap8",
            "value": f"{fmt(adaptive['total_proxy_mib_per_frame'])} vs {fmt(gap8['total_proxy_mib_per_frame'])}",
            "interpretation": "Stage166 proxy suggests residual-payload savings can outweigh extra keyframe cost, but this is not yet final accounting.",
            "next_action": "Run Stage172 keyframe/residual/metadata decomposition.",
        },
        {
            "area": "rendered_false_negatives",
            "status": "neutral_risk",
            "metric": "adaptive delta vs uniform_gap8 PSNR / LPIPS",
            "value": f"{fmt(false_adaptive['mean_delta_psnr_vs_uniform_gap8'])} / {fmt(false_adaptive['mean_delta_lpips_vs_uniform_gap8'])}",
            "interpretation": "False negatives are essentially unchanged from uniform gap8, so adaptive does not fix missed hard cases.",
            "next_action": "Keep false negatives in medium validation and do not scale directly to full validation yet.",
        },
        {
            "area": "positive_promotions",
            "status": "confirmed_behavior",
            "metric": "adaptive keyframe markers / uniform_gap8 payload",
            "value": f"{positive_adaptive['keyframe_marker_count']} / {fmt(positive_gap8['mean_payload_bytes'])}",
            "interpretation": "Promoted adaptive targets are transmitted keyframes; no middle-render metrics are claimed, and expensive uniform-gap8 residual recovery is avoided on these rows.",
            "next_action": "Continue marking promoted targets as keyframes/no-middle-render.",
        },
        {
            "area": "high_payload_controls",
            "status": "small_positive_control",
            "metric": "adaptive minus uniform_gap8 PSNR / LPIPS",
            "value": f"{fmt(control_adaptive['mean_delta_psnr_vs_uniform_gap8'])} / {fmt(control_adaptive['mean_delta_lpips_vs_uniform_gap8'])}",
            "interpretation": "On the small residual-control slice, adaptive residual rows are not worse than uniform gap8.",
            "next_action": "Expand this control category in Stage173/174.",
        },
        {
            "area": "validation_scope",
            "status": "sampled_only",
            "metric": "Stage170 protocol rows / new renders",
            "value": f"{stage170['protocol_row_count']} / {stage170['new_render_count']}",
            "interpretation": "Current rendered evidence is intentionally sampled and cannot justify final full-sequence claims alone.",
            "next_action": "Use Stage173 medium protocol before any full-sequence validation.",
        },
    ]
    return rows


def write_report(rows, package, path):
    lines = [
        "# Stage171 Combined Adaptive Evidence Review",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        "- Continue the selector/adaptive-keyframe line, but do not jump directly to full validation.",
        "- Next required stage is keyframe/residual/metadata rate accounting, then a medium rendered protocol.",
        "",
        "## Evidence",
        "",
        "| area | status | metric | value | interpretation | next action |",
        "|---|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['area']} | {row['status']} | {row['metric']} | {row['value']} | {row['interpretation']} | {row['next_action']} |"
        )
    lines.extend([
        "",
        "## Boundaries",
        "",
        "- This review does not change Stage158 `streamsplat_guided_half_anchor_entropy_residual_v1`.",
        "- Adaptive promoted rows remain keyframes/no-middle-render; middle-frame metrics are not invented for them.",
        "- Stage172 must explicitly charge extra keyframes, residual payloads, and schedule metadata.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage165_package", type=Path, default=DEFAULT_STAGE165_PACKAGE)
    parser.add_argument("--stage166_package", type=Path, default=DEFAULT_STAGE166_PACKAGE)
    parser.add_argument("--stage170_package", type=Path, default=DEFAULT_STAGE170_PACKAGE)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage165 = read_json(args.stage165_package)
    stage166 = read_json(args.stage166_package)
    stage170 = read_json(args.stage170_package)
    evidence_rows = build_evidence(stage165, stage166, stage170)
    decision = "proceed_to_rate_audit_and_medium_protocol"
    evidence_csv = args.output_root / "stage171_combined_adaptive_evidence_rows.csv"
    package_json = args.output_root / "stage171_combined_adaptive_evidence_review_package.json"
    report_md = args.output_root / "stage171_combined_adaptive_evidence_review_report.md"
    package = {
        "stage": 171,
        "status": "combined_adaptive_evidence_review_packaged",
        "decision": decision,
        "fixed_middle_recovery_policy": "streamsplat_guided_half_anchor_entropy_residual_v1",
        "evidence_rows": evidence_rows,
        "required_next_stages": [
            "Stage172 keyframe rate accounting audit",
            "Stage173 medium rendered validation protocol",
            "Stage174 medium rendered validation execution",
        ],
        "do_not_claim": [
            "final full-sequence RD",
            "rendered middle metrics for adaptive keyframe targets",
            "decoder-side RGB/motion selector reproduction",
        ],
        "evidence_csv": str(evidence_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    write_csv(evidence_rows, evidence_csv, EVIDENCE_FIELDS)
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(evidence_rows, package, report_md)
    print(json.dumps({"package": str(package_json), "decision": decision}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
