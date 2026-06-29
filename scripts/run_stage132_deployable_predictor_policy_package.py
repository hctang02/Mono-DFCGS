import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE125_SUMMARY = REPO_ROOT / "experiments/stage125_broader_feedforward_residual_value_predictor_validation/stage125_broader_feedforward_residual_value_predictor_validation_summary.csv"
DEFAULT_STAGE130_PACKAGE = REPO_ROOT / "experiments/stage130_teacher_sideinfo_vs_predictor_comparison/stage130_teacher_sideinfo_vs_predictor_comparison_package.json"
DEFAULT_STAGE131_PACKAGE = REPO_ROOT / "experiments/stage131_predictor_ablation_package/stage131_predictor_ablation_package.json"
DEFAULT_STAGE65_ADAPTER = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage132_deployable_predictor_policy_package"

SETTING_FIELDS = [
    "role",
    "setting_label",
    "keep_fraction",
    "side_bits",
    "direct_total_mib_per_frame",
    "psnr",
    "delta_psnr_vs_linear_base",
    "delta_psnr_vs_full_adapter",
    "transmitted_residual_payload_bytes",
    "transmitted_selected_index_bytes",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_settings(stage125_rows):
    out = []
    for row in stage125_rows:
        if row["setting_label"] not in {"q4_top20", "q4_top10"}:
            continue
        out.append({
            "role": "primary" if row["setting_label"] == "q4_top20" else "low_rate",
            "setting_label": row["setting_label"],
            "keep_fraction": row["keep_fraction"],
            "side_bits": row["side_bits"],
            "direct_total_mib_per_frame": row["mean_direct_total_mib_per_frame"],
            "psnr": row["mean_selected_predicted_psnr"],
            "delta_psnr_vs_linear_base": row["mean_delta_psnr_vs_base"],
            "delta_psnr_vs_full_adapter": row["mean_delta_psnr_vs_full_predictor"],
            "transmitted_residual_payload_bytes": 0,
            "transmitted_selected_index_bytes": 0,
        })
    return sorted(out, key=lambda row: 0 if row["role"] == "primary" else 1)


def build_policy(settings, stage130_package, stage131_package, adapter_path):
    primary = next(row for row in settings if row["role"] == "primary")
    low_rate = next(row for row in settings if row["role"] == "low_rate")
    return {
        "stage": 132,
        "policy_name": "deployable_adapter_delta_selected_residual_codec_v1",
        "policy_type": "deployable_predictor_policy",
        "status": "current_best_no_teacher_deployable",
        "selected_primary_setting": primary["setting_label"],
        "optional_low_rate_setting": low_rate["setting_label"],
        "predictor": {
            "name": "adapter_delta_selected_predictor",
            "rule": "run pre-shared Stage65 adapter and use adapter_attrs - linear_attrs at deterministic selected indices",
            "stage65_adapter_checkpoint": str(adapter_path),
            "checkpoint_exists": int(adapter_path.exists()),
            "checkpoint_rate_accounting": "pre-shared model; not counted per frame. If transmitted in-session, account separately.",
        },
        "index_selection": {
            "rule_name": "endpoint_diff_topk_v1",
            "keep_count_rule": "min(max(round(N * keep_fraction), 0), N)",
            "score_rule": "sum((right_attrs[0].float() - left_attrs[0].float()) ** 2, dim=-1)",
            "selection_rule": "select top keep_count largest scores, then sort selected indices ascending",
            "transmitted_selected_index_bytes": 0,
        },
        "rate_accounting": {
            "transmitted_residual_payload_bytes": 0,
            "transmitted_selected_index_bytes": 0,
            "direct_total_rate": "q12_main_anchor_mib_per_frame only for predictor residuals",
            "amortized_total_rate": "same as direct for predictor residuals in this package",
        },
        "settings": settings,
        "decoder_contract": {
            "inputs": [
                "left_anchor",
                "right_anchor",
                "normalized_time",
                "pre-shared Stage65 adapter checkpoint",
                "policy setting keep_fraction",
            ],
            "steps": [
                "compute linear base attrs from left/right anchors and normalized time",
                "compute Stage65 adapter attrs from left/right anchors and normalized time",
                "recompute endpoint-diff selected indices",
                "apply adapter_attrs - linear_attrs residual values at selected indices",
            ],
            "forbidden_inputs": [
                "target_dense_anchor",
                "target_residual",
                "target_rgb",
                "oracle_task_label",
                "transmitted selected indices",
                "transmitted residual values",
            ],
        },
        "references": {
            "stage130_best_deployable": stage130_package["best_deployable_no_teacher"],
            "stage131_recommended": stage131_package["recommended_deployable_predictor"],
            "stage131_rejected_final_predictor": stage131_package["rejected_final_predictor"],
            "stage131_rejection_reason": stage131_package["rejection_reason"],
        },
        "limitations": [
            "Quality gain is small relative to teacher residual side-info.",
            "Pre-shared Stage65 adapter checkpoint is required for deployment.",
            "Dedicated selected residual MLP is not selected because Stage129 rendered PSNR regressed.",
        ],
    }


def format_float(value):
    return f"{float(value):.6f}"


def write_report(policy, package, path):
    lines = [
        "# Stage132 Deployable Predictor Policy Package",
        "",
        "## Policy",
        "",
        f"- policy: `{policy['policy_name']}`",
        f"- status: `{policy['status']}`",
        f"- primary setting: `{policy['selected_primary_setting']}`",
        f"- optional low-rate setting: `{policy['optional_low_rate_setting']}`",
        f"- adapter checkpoint exists: `{policy['predictor']['checkpoint_exists']}`",
        "",
        "## Settings",
        "",
        "| role | setting | keep | rate | PSNR | delta base | delta full | residual bytes | index bytes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy["settings"]:
        lines.append(
            f"| {row['role']} | {row['setting_label']} | {row['keep_fraction']} | {format_float(row['direct_total_mib_per_frame'])} | {format_float(row['psnr'])} | {format_float(row['delta_psnr_vs_linear_base'])} | {format_float(row['delta_psnr_vs_full_adapter'])} | {row['transmitted_residual_payload_bytes']} | {row['transmitted_selected_index_bytes']} |"
        )
    lines.extend([
        "",
        "## Decoder Contract",
        "",
        "- Inputs: left anchor, right anchor, normalized time, pre-shared Stage65 adapter, policy keep fraction.",
        "- Forbidden: target dense anchor, target residual, target RGB, oracle labels, transmitted selected indices, transmitted residual values.",
        "- Per-frame residual payload bytes: 0.",
        "- Per-frame selected-index payload bytes: 0.",
        "",
        "## Outputs",
        "",
        f"- policy JSON: `{package['policy_json']}`",
        f"- settings CSV: `{package['settings_csv']}`",
        f"- package JSON: `{package['package_json']}`",
        f"- report Markdown: `{package['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage125_summary", type=Path, default=DEFAULT_STAGE125_SUMMARY)
    parser.add_argument("--stage130_package", type=Path, default=DEFAULT_STAGE130_PACKAGE)
    parser.add_argument("--stage131_package", type=Path, default=DEFAULT_STAGE131_PACKAGE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_STAGE65_ADAPTER)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    settings = build_settings(read_csv(args.stage125_summary))
    policy = build_policy(settings, read_json(args.stage130_package), read_json(args.stage131_package), args.adapter)
    policy_json = args.summary_root / "stage132_deployable_predictor_policy.json"
    settings_csv = args.summary_root / "stage132_deployable_predictor_policy_settings.csv"
    package_json = args.summary_root / "stage132_deployable_predictor_policy_package.json"
    report_md = args.summary_root / "stage132_deployable_predictor_policy_report.md"
    write_csv(settings, settings_csv, SETTING_FIELDS)
    policy_json.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    package = {
        "stage": 132,
        "mode": "deployable predictor policy package",
        "policy_name": policy["policy_name"],
        "status": policy["status"],
        "policy_json": str(policy_json),
        "settings_csv": str(settings_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "primary_setting": policy["selected_primary_setting"],
        "optional_low_rate_setting": policy["optional_low_rate_setting"],
        "adapter_checkpoint_exists": policy["predictor"]["checkpoint_exists"],
        "notes": policy["limitations"],
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(policy, package, report_md)
    print(json.dumps({"package": str(package_json), "policy": policy["policy_name"], "adapter_checkpoint_exists": policy["predictor"]["checkpoint_exists"]}, indent=2))


if __name__ == "__main__":
    main()
