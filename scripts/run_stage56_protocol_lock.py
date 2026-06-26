import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage56_protocol_lock"

RATE_FIELDS = [
    "rate_name",
    "counted_in_main_rate",
    "counted_in_total_rate",
    "unit",
    "description",
]

METHOD_FIELDS = [
    "method_family",
    "deployable_final_claim",
    "allowed_test_time_inputs",
    "forbidden_test_time_inputs",
    "notes",
]

TABLE_FIELDS = [
    "table_name",
    "required_fields",
    "notes",
]


def write_csv(rows, fields, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_rate_rows():
    return [
        {
            "rate_name": "anchor_bitstream_rate",
            "counted_in_main_rate": "yes",
            "counted_in_total_rate": "yes",
            "unit": "MiB/frame",
            "description": "Compressed transmitted keyframe Gaussian anchor payload plus required anchor container metadata.",
        },
        {
            "rate_name": "keyframe_indices_timestamps_metadata",
            "counted_in_main_rate": "yes",
            "counted_in_total_rate": "yes",
            "unit": "MiB/frame or bytes/sequence",
            "description": "Keyframe indices, timestamps, quantization bits, tensor shapes, field schema, and codec header metadata.",
        },
        {
            "rate_name": "side_information_rate",
            "counted_in_main_rate": "no_for_keyframe_gaussian_only; yes_for_side_info_variant_total",
            "counted_in_total_rate": "yes_if_transmitted",
            "unit": "MiB/frame",
            "description": "Any transmitted non-keyframe depth, motion hint, residual, importance map, or correction payload.",
        },
        {
            "rate_name": "decoder_model_weights",
            "counted_in_main_rate": "no",
            "counted_in_total_rate": "reported_separately",
            "unit": "MiB/model",
            "description": "Shared decoder/adapter weights are not counted as per-video payload, but must be reported separately where relevant.",
        },
        {
            "rate_name": "non_keyframe_rgb_depth_or_gaussians",
            "counted_in_main_rate": "not_allowed_in_current_main_method",
            "counted_in_total_rate": "yes_if_future_variant_transmits_it",
            "unit": "MiB/frame",
            "description": "Current Mono-DFCGS-KG does not transmit these. Future side-info variants must count them if used.",
        },
    ]


def build_method_rows():
    return [
        {
            "method_family": "uniform_keyframes",
            "deployable_final_claim": "yes",
            "allowed_test_time_inputs": "video length, fixed gap, transmitted keyframe Gaussian anchors",
            "forbidden_test_time_inputs": "rendered error, PSNR, reconstructed output lookahead",
            "notes": "Baseline for all adaptive comparisons.",
        },
        {
            "method_family": "rendered_prior_0p1_oracle_calibrated",
            "deployable_final_claim": "no",
            "allowed_test_time_inputs": "offline analysis labels only",
            "forbidden_test_time_inputs": "must not be used as final deployed selector because it relies on rendered-error-related information",
            "notes": "Upper bound / training target only.",
        },
        {
            "method_family": "feed_forward_selector",
            "deployable_final_claim": "yes_if_frozen_and_deterministic",
            "allowed_test_time_inputs": "original input video, encoder-side features, endpoint keyframe candidate stats, deterministic DP output",
            "forbidden_test_time_inputs": "rendered oracle, PSNR labels, test-time optimization over reconstructed frames",
            "notes": "Final adaptive selector target.",
        },
        {
            "method_family": "teacher_distilled_adapter",
            "deployable_final_claim": "yes_if_teacher_not_used_at_test_time",
            "allowed_test_time_inputs": "transmitted keyframe anchors, timestamps, optional counted side information",
            "forbidden_test_time_inputs": "StreamSplat RGB/depth-conditioned teacher output at test time",
            "notes": "Training may use teacher supervision; deployment may not depend on teacher inputs.",
        },
        {
            "method_family": "side_information_variant",
            "deployable_final_claim": "yes_if_all_side_info_is_transmitted_and_counted",
            "allowed_test_time_inputs": "anchor bitstream plus explicitly counted side-information bitstream",
            "forbidden_test_time_inputs": "free non-keyframe RGB/depth/motion/residual inputs",
            "notes": "Report anchor rate, side-info rate, total rate, and all-frame PSNR separately.",
        },
    ]


def build_table_rows():
    return [
        {
            "table_name": "main_rd_table",
            "required_fields": "method, compression, adapter, selector, side_info, anchor_rate_mib_per_frame, side_info_rate_mib_per_frame, total_rate_mib_per_frame, all_psnr",
            "notes": "Main paper/user-facing table. Do not include middle-only metrics unless requested.",
        },
        {
            "table_name": "compression_ablation_table",
            "required_fields": "codec, bits_or_profile, raw_rate_mib_per_frame, compressed_rate_mib_per_frame, rate_saving_percent, all_psnr",
            "notes": "Shows compression contribution.",
        },
        {
            "table_name": "adapter_ablation_table",
            "required_fields": "adapter_variant, training_steps, train_dataset, val_dataset, best_step, all_psnr, delta_vs_linear",
            "notes": "Shows adapter contribution and training scale.",
        },
        {
            "table_name": "selector_ablation_table",
            "required_fields": "selector_variant, deployable, oracle_used_at_test_time, rate_mib_per_frame, all_psnr, delta_vs_uniform, negative_point_count",
            "notes": "Separates oracle upper bounds from final feed-forward selectors.",
        },
        {
            "table_name": "training_run_summary",
            "required_fields": "run_id, stage, model_variant, dataset, steps, best_step, checkpoint_path_external, validation_all_psnr, notes",
            "notes": "Used for medium/long training runs. Checkpoints stay outside git.",
        },
    ]


def write_report(summary, rate_rows, method_rows, table_rows, path):
    lines = [
        "# Stage56 Protocol Lock",
        "",
        "## Locked Decisions",
        "",
        "- Default user-facing quality metric: all-frame PSNR.",
        "- Main keyframe-Gaussian-only rate: transmitted Gaussian anchor bitstream MiB/frame plus required anchor metadata.",
        "- Optional side-information variants must report anchor rate, side-info rate, and total rate.",
        "- `rendered_prior_0p1` is an oracle/calibrated upper bound, not the final deployable selector.",
        "- Final adaptive selector must be frozen, feed-forward, deterministic, and must not use rendered oracle or PSNR lookahead at test time.",
        "- Medium/long training runs are required for strong adapter/selector claims; short runs are smoke/infrastructure only.",
        "",
        "## Rate Accounting",
        "",
        "| Rate name | Main rate | Total rate | Unit | Description |",
        "|---|---|---|---|---|",
    ]
    for row in rate_rows:
        lines.append(
            f"| {row['rate_name']} | {row['counted_in_main_rate']} | {row['counted_in_total_rate']} | {row['unit']} | {row['description']} |"
        )
    lines.extend([
        "",
        "## Method Deployability",
        "",
        "| Method family | Final claim | Allowed test-time inputs | Forbidden test-time inputs | Notes |",
        "|---|---|---|---|---|",
    ])
    for row in method_rows:
        lines.append(
            f"| {row['method_family']} | {row['deployable_final_claim']} | {row['allowed_test_time_inputs']} | {row['forbidden_test_time_inputs']} | {row['notes']} |"
        )
    lines.extend([
        "",
        "## Standard Tables",
        "",
        "| Table | Required fields | Notes |",
        "|---|---|---|",
    ])
    for row in table_rows:
        lines.append(f"| {row['table_name']} | {row['required_fields']} | {row['notes']} |")
    lines.extend([
        "",
        "## Output Files",
        "",
        f"- Summary JSON: `{summary['summary_json']}`",
        f"- Rate accounting CSV: `{summary['rate_accounting_csv']}`",
        f"- Method deployability CSV: `{summary['method_deployability_csv']}`",
        f"- Standard table schema CSV: `{summary['standard_table_schema_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)

    rate_rows = build_rate_rows()
    method_rows = build_method_rows()
    table_rows = build_table_rows()

    rate_csv = args.summary_root / "stage56_rate_accounting_rules.csv"
    method_csv = args.summary_root / "stage56_method_deployability_rules.csv"
    table_csv = args.summary_root / "stage56_standard_table_schemas.csv"
    summary_json = args.summary_root / "stage56_protocol_lock_summary.json"
    report_md = args.summary_root / "stage56_protocol_lock_report.md"

    write_csv(rate_rows, RATE_FIELDS, rate_csv)
    write_csv(method_rows, METHOD_FIELDS, method_csv)
    write_csv(table_rows, TABLE_FIELDS, table_csv)

    summary = {
        "stage": 56,
        "mode": "protocol lock",
        "default_quality_metric": "all-frame PSNR",
        "main_rate_metric": "transmitted Gaussian anchor bitstream MiB/frame",
        "final_selector_requirement": "fully feed-forward frozen predictor plus deterministic selection/DP; no rendered oracle at test time",
        "side_information_rule": "if transmitted, count in side-info rate and total rate",
        "training_scale_rule": "short runs are smoke only; medium/long runs required for final adapter/selector claims",
        "rate_accounting_csv": str(rate_csv),
        "method_deployability_csv": str(method_csv),
        "standard_table_schema_csv": str(table_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "next_stage": "Stage57 compact Gaussian anchor codec",
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, rate_rows, method_rows, table_rows, report_md)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
