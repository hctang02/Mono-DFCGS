import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage162_keyframe_selector_protocol_source_audit"

STAGE11_KEYFRAME_SELECTION = REPO_ROOT / "experiments/stage11_keyframe_selection/stage11_keyframe_selection_summary.csv"
STAGE16_SEGMENT_SELECTION = REPO_ROOT / "experiments/stage16_segment_error_keyframe_selection/stage16_segment_error_keyframe_selection_summary.csv"
STAGE17_SEGMENT_RD = REPO_ROOT / "experiments/stage17_segment_error_rd_curve/stage17_segment_error_rd_curve_summary.csv"
STAGE161_METHOD_PACKAGE = REPO_ROOT / "experiments/stage161_stage158_method_narrative_package/stage161_stage158_method_narrative_package.json"

FEATURE_FIELDS = [
    "feature_group", "example_features", "source", "encoder_available", "decoder_required", "feedforward_status",
    "allowed_for_inference", "allowed_for_training_label", "metadata_transmitted", "cost_or_caveat",
]
RATE_FIELDS = ["component", "counting_rule", "notes"]
BASELINE_FIELDS = ["baseline", "selector_inputs", "transmitted_metadata", "role", "status"]
PROTOCOL_FIELDS = ["item", "decision", "details"]
HISTORICAL_FIELDS = ["stage", "artifact", "relevance", "use_in_stage162"]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def feature_rows():
    return [
        {
            "feature_group": "raw_rgb_encoder_input",
            "example_features": "frame RGB, grayscale, downsampled RGB, frame timestamp/index",
            "source": "input video frames available to encoder; DAVIS RGB files in current experiments",
            "encoder_available": "yes",
            "decoder_required": "no",
            "feedforward_status": "feed-forward for offline video encoding; for online streaming requires declared lookahead window",
            "allowed_for_inference": "yes",
            "allowed_for_training_label": "yes",
            "metadata_transmitted": "only selected keyframe indices/schedule, not RGB features",
            "cost_or_caveat": "dataset RGB is the source signal being compressed, not extra side information",
        },
        {
            "feature_group": "deterministic_rgb_motion_proxy",
            "example_features": "frame difference, block SAD/MSE, edge-change, histogram difference, local gradient magnitude",
            "source": "deterministic functions of encoder input RGB frames",
            "encoder_available": "yes",
            "decoder_required": "no",
            "feedforward_status": "feed-forward if computed only from input frames inside declared lookahead",
            "allowed_for_inference": "yes_primary_cheap_tier",
            "allowed_for_training_label": "yes",
            "metadata_transmitted": "only keyframe schedule",
            "cost_or_caveat": "preferred first heuristic selector feature family",
        },
        {
            "feature_group": "pretrained_motion_or_feature_network",
            "example_features": "RAFT/GMFlow optical flow, DINO/ResNet frame embeddings, learned motion confidence",
            "source": "fixed pretrained network fed only encoder input RGB frames",
            "encoder_available": "yes_if_weights_are_fixed_and_inputs_are_raw_rgb_only",
            "decoder_required": "no",
            "feedforward_status": "feed-forward with compute-cost caveat; no target dense anchors or rendered metrics may enter",
            "allowed_for_inference": "yes_optional_higher_compute_tier",
            "allowed_for_training_label": "yes",
            "metadata_transmitted": "only keyframe schedule",
            "cost_or_caveat": "network compute and dependency must be documented; no bitrate charge unless features are transmitted",
        },
        {
            "feature_group": "encoder_rd_probe",
            "example_features": "candidate Stage158 residual payload bytes, candidate original StreamSplat middle residual estimates",
            "source": "encoder-side candidate encodes/renders using available input and target anchors during compression",
            "encoder_available": "yes_but_expensive",
            "decoder_required": "no",
            "feedforward_status": "offline encoder-side RD probe, not a cheap single-pass feed-forward selector",
            "allowed_for_inference": "allowed_as_expensive_encoder_optimization_tier_only_if_schedule_is_transmitted",
            "allowed_for_training_label": "yes",
            "metadata_transmitted": "keyframe schedule plus normal Stage158 payloads",
            "cost_or_caveat": "must not be described as low-compute feed-forward; useful upper/teacher policy",
        },
        {
            "feature_group": "rendered_quality_oracle",
            "example_features": "rendered PSNR, SSIM, MS-SSIM, LPIPS, bad-case rank",
            "source": "requires target RGB and rendered candidate output",
            "encoder_available": "yes_for_offline_evaluation_or_oracle_search",
            "decoder_required": "no",
            "feedforward_status": "not feed-forward selector input for deployable inference",
            "allowed_for_inference": "no",
            "allowed_for_training_label": "yes_offline_label_only",
            "metadata_transmitted": "none; labels are not sent",
            "cost_or_caveat": "allowed only to create labels/evaluate oracle schedules",
        },
        {
            "feature_group": "target_dense_anchor_or_residual",
            "example_features": "target dense anchor, target Gaussian residual, unencoded residual tensor",
            "source": "Stage61 dense anchors or residual against target anchor",
            "encoder_available": "yes_for_training_or_stage158_residual_encoding",
            "decoder_required": "forbidden",
            "feedforward_status": "not allowed as selector inference feature except when it is encoded into transmitted Stage158 residual payload",
            "allowed_for_inference": "no_for_selector_features",
            "allowed_for_training_label": "yes_encoder_side_label_or_diagnostic_only",
            "metadata_transmitted": "must be actual encoded residual payload if used by decoder",
            "cost_or_caveat": "prevents oracle leakage into keyframe selector",
        },
    ]


def rate_rows():
    return [
        {
            "component": "keyframe_anchor_payload",
            "counting_rule": "count q-bit Gaussian anchor payload/metadata for every transmitted keyframe",
            "notes": "Use existing q12/qbit anchor accounting tables; if schedule changes keyframe count, anchor rate changes accordingly.",
        },
        {
            "component": "adaptive_keyframe_indices",
            "counting_rule": "count schedule metadata for non-uniform selectors: keyframe count plus sorted indices or segment lengths packed with ceil(log2(total_frames)) bits/index",
            "notes": "Uniform fixed-gap schedules can be signaled by mode id only; adaptive schedules must transmit their indices or equivalent segment lengths.",
        },
        {
            "component": "selector_mode_id",
            "counting_rule": "count at least one small mode id when multiple schedule policies are available",
            "notes": "Examples: uniform_gap4, uniform_gap8, rgb_motion_heuristic, learned_selector.",
        },
        {
            "component": "stage158_middle_residual_payload",
            "counting_rule": "count q6/keep1.0 entropy residual payload plus one-byte half selector for every recovered middle/intermediate frame",
            "notes": "Payload may vary by segment difficulty; Stage158 current policy is quality-first and not rate-minimized.",
        },
        {
            "component": "feature_computation",
            "counting_rule": "not counted as bitrate if features are not transmitted, but compute/dependency tier must be reported",
            "notes": "Pretrained optical flow or embeddings must be documented as higher-compute encoder tools.",
        },
    ]


def baseline_rows():
    return [
        {
            "baseline": "uniform_gap4_plus_stage158",
            "selector_inputs": "none beyond total frame count",
            "transmitted_metadata": "mode id or fixed policy id only",
            "role": "high-quality reference",
            "status": "primary baseline",
        },
        {
            "baseline": "uniform_gap8_plus_stage158",
            "selector_inputs": "none beyond total frame count",
            "transmitted_metadata": "mode id or fixed policy id only",
            "role": "lower-keyframe-rate reference",
            "status": "primary baseline",
        },
        {
            "baseline": "stage16_segment_motion_or_rd_schedule",
            "selector_inputs": "historical motion/segment-error features",
            "transmitted_metadata": "adaptive keyframe indices or segment lengths",
            "role": "historical adaptive schedule reference",
            "status": "reuse for comparison, not final without DAVIS+Stage158 validation",
        },
        {
            "baseline": "rgb_motion_heuristic_v1",
            "selector_inputs": "raw RGB frame differences, block motion proxy, edge-change statistics",
            "transmitted_metadata": "adaptive keyframe indices or segment lengths",
            "role": "first deployable selector candidate",
            "status": "next implementation target",
        },
        {
            "baseline": "learned_rgb_motion_selector_v1",
            "selector_inputs": "RGB/motion features only; no target anchors or rendered metrics at inference",
            "transmitted_metadata": "adaptive keyframe indices or segment lengths",
            "role": "later learned selector candidate",
            "status": "after heuristic/oracle data package",
        },
        {
            "baseline": "oracle_rd_schedule",
            "selector_inputs": "rendered RD/quality labels from candidate schedules",
            "transmitted_metadata": "oracle schedule indices for accounting only",
            "role": "upper bound and training label source",
            "status": "not deployable inference selector",
        },
    ]


def protocol_rows():
    return [
        {
            "item": "codec_setting",
            "decision": "quality_first_stage158_middle_recovery",
            "details": "Use Stage158 policy as middle/intermediate recovery component; do not over-optimize its residual rate in this phase.",
        },
        {
            "item": "selector_output",
            "decision": "transmitted_keyframe_schedule",
            "details": "Selector outputs keyframe indices or segment lengths; decoder follows transmitted schedule and does not reproduce selector features.",
        },
        {
            "item": "feature_scope",
            "decision": "encoder_side_rgb_motion_allowed",
            "details": "RGB/motion features are allowed if derived from input video frames available to encoder; source and compute tier must be logged.",
        },
        {
            "item": "feedforward_protocol",
            "decision": "offline_video_feedforward_with_optional_lookahead",
            "details": "Current DAVIS protocol is offline encoding, so full input RGB can be used encoder-side. Online/streaming variants must declare a lookahead window.",
        },
        {
            "item": "forbidden_inference_inputs",
            "decision": "no_target_dense_or_rendered_metric_leakage",
            "details": "Target dense anchors, unencoded residuals, target RGB quality labels, rendered PSNR/LPIPS, and oracle labels are not selector inference inputs.",
        },
        {
            "item": "evaluation_metrics",
            "decision": "all_frame_keyframe_middle_and_sequence_level",
            "details": "Report all-frame, keyframe-only, middle-only, per-gap, and per-sequence PSNR/SSIM/MS-SSIM/LPIPS plus rate.",
        },
        {
            "item": "splitting",
            "decision": "sequence_aware_validation",
            "details": "Learned selector validation should split by sequence; heuristic selector should report all selected DAVIS sequences and weak cases.",
        },
    ]


def historical_rows():
    return [
        {
            "stage": 11,
            "artifact": str(STAGE11_KEYFRAME_SELECTION),
            "relevance": "early uniform/motion/gaussian/RD-aware keyframe index baselines",
            "use_in_stage162": "historical terminology and baseline family",
        },
        {
            "stage": 16,
            "artifact": str(STAGE16_SEGMENT_SELECTION),
            "relevance": "segment-length-aware keyframe schedules",
            "use_in_stage162": "adaptive segment baseline reference",
        },
        {
            "stage": 17,
            "artifact": str(STAGE17_SEGMENT_RD),
            "relevance": "segment-error RD curve on earlier data/protocol",
            "use_in_stage162": "historical comparison only; must revalidate on DAVIS + Stage158",
        },
        {
            "stage": 161,
            "artifact": str(STAGE161_METHOD_PACKAGE),
            "relevance": "current Stage158 middle recovery method",
            "use_in_stage162": "fixed middle-recovery component for adaptive schedule evaluation",
        },
    ]


def write_report(package, path):
    lines = [
        "# Stage162 Keyframe Selector Protocol Source Audit",
        "",
        "## Scope",
        "",
        "Adaptive keyframe selection is now the next component after Stage158/161 middle recovery.",
        "The selector chooses a keyframe schedule; decoder receives the transmitted schedule and does not need RGB/motion features.",
        "",
        "## Feature Source Audit",
        "",
        "| feature group | source | feed-forward status | inference | caveat |",
        "|---|---|---|---|---|",
    ]
    for row in package["feature_source_audit"]:
        lines.append(
            f"| {row['feature_group']} | {row['source']} | {row['feedforward_status']} | {row['allowed_for_inference']} | {row['cost_or_caveat']} |"
        )
    lines.extend([
        "",
        "## Rate Accounting",
        "",
        "| component | counting rule | notes |",
        "|---|---|---|",
    ])
    for row in package["rate_accounting"]:
        lines.append(f"| {row['component']} | {row['counting_rule']} | {row['notes']} |")
    lines.extend([
        "",
        "## Baselines",
        "",
        "| baseline | selector inputs | metadata | role | status |",
        "|---|---|---|---|---|",
    ])
    for row in package["baselines"]:
        lines.append(f"| {row['baseline']} | {row['selector_inputs']} | {row['transmitted_metadata']} | {row['role']} | {row['status']} |")
    lines.extend([
        "",
        "## Decisions",
        "",
        "| item | decision | details |",
        "|---|---|---|",
    ])
    for row in package["protocol_decisions"]:
        lines.append(f"| {row['item']} | {row['decision']} | {row['details']} |")
    lines.extend([
        "",
        "## Next Stage",
        "",
        "Stage163 should build the first DAVIS selector data package: compute cheap RGB/motion segment features, define candidate schedules, and attach Stage158-compatible rate/quality labels or oracle references for training/evaluation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage161 = read_json(STAGE161_METHOD_PACKAGE)
    features = feature_rows()
    rates = rate_rows()
    baselines = baseline_rows()
    protocol = protocol_rows()
    historical = historical_rows()
    feature_csv = args.summary_root / "stage162_feature_source_audit.csv"
    rate_csv = args.summary_root / "stage162_rate_accounting_rules.csv"
    baseline_csv = args.summary_root / "stage162_selector_baselines.csv"
    protocol_csv = args.summary_root / "stage162_protocol_decisions.csv"
    historical_csv = args.summary_root / "stage162_historical_selector_artifacts.csv"
    package_json = args.summary_root / "stage162_keyframe_selector_protocol_source_audit_package.json"
    report_md = args.summary_root / "stage162_keyframe_selector_protocol_source_audit_report.md"
    write_csv(features, feature_csv, FEATURE_FIELDS)
    write_csv(rates, rate_csv, RATE_FIELDS)
    write_csv(baselines, baseline_csv, BASELINE_FIELDS)
    write_csv(protocol, protocol_csv, PROTOCOL_FIELDS)
    write_csv(historical, historical_csv, HISTORICAL_FIELDS)
    package = {
        "stage": 162,
        "status": "keyframe_selector_protocol_and_source_audit_packaged",
        "fixed_middle_recovery_policy": stage161["policy"],
        "middle_recovery_rate_position": stage161["rate_position"],
        "selector_output": "transmitted keyframe indices or segment lengths",
        "feature_source_audit": features,
        "rate_accounting": rates,
        "baselines": baselines,
        "protocol_decisions": protocol,
        "historical_selector_artifacts": historical,
        "next_stage": "Stage163 build DAVIS RGB/motion selector data and oracle/reference labels",
        "feature_csv": str(feature_csv),
        "rate_csv": str(rate_csv),
        "baseline_csv": str(baseline_csv),
        "protocol_csv": str(protocol_csv),
        "historical_csv": str(historical_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, report_md)
    print(json.dumps({"package": str(package_json), "report": str(report_md), "status": package["status"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
