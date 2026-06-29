import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE132_POLICY = REPO_ROOT / "experiments/stage132_deployable_predictor_policy_package/stage132_deployable_predictor_policy.json"
DEFAULT_STAGE137_SUMMARY = REPO_ROOT / "experiments/stage137_broader_render_aware_scale_validation/stage137_broader_render_aware_scale_validation_summary.json"
DEFAULT_STAGE65_ADAPTER = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage138_render_aware_deployable_policy_package"

SETTING_FIELDS = [
    "role",
    "setting_label",
    "keep_fraction",
    "side_bits",
    "adapter_delta_scale",
    "direct_total_mib_per_frame",
    "psnr",
    "delta_psnr_vs_linear_base",
    "delta_psnr_vs_full_adapter",
    "transmitted_residual_payload_bytes",
    "transmitted_selected_index_bytes",
    "previous_stage132_psnr",
    "delta_psnr_vs_stage132_policy",
]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def select_setting(summary_rows, role):
    candidates = [row for row in summary_rows if row["setting_role"] == role and float(row["mean_delta_psnr_vs_base"]) >= 0.0]
    if not candidates:
        raise RuntimeError(f"no non-negative candidates for role {role}")
    return max(
        candidates,
        key=lambda row: (
            float(row["mean_selected_predicted_psnr"]),
            float(row["mean_delta_psnr_vs_base"]),
            -float(row["adapter_delta_scale"]),
        ),
    )


def previous_stage132_row(stage132_policy, role):
    for row in stage132_policy["settings"]:
        if row["role"] == role:
            return row
    raise KeyError(f"missing Stage132 row for role {role}")


def build_setting(row, role, previous):
    psnr = float(row["mean_selected_predicted_psnr"])
    previous_psnr = float(previous["psnr"])
    return {
        "role": role,
        "setting_label": row["setting_label"],
        "keep_fraction": row["keep_fraction"],
        "side_bits": row["side_bits"],
        "adapter_delta_scale": row["adapter_delta_scale"],
        "direct_total_mib_per_frame": row["mean_direct_total_mib_per_frame"],
        "psnr": psnr,
        "delta_psnr_vs_linear_base": row["mean_delta_psnr_vs_base"],
        "delta_psnr_vs_full_adapter": row["mean_delta_psnr_vs_full_predictor"],
        "transmitted_residual_payload_bytes": 0,
        "transmitted_selected_index_bytes": 0,
        "previous_stage132_psnr": previous_psnr,
        "delta_psnr_vs_stage132_policy": psnr - previous_psnr,
    }


def build_policy(settings, stage132_policy, stage132_policy_path, stage137_summary, stage137_summary_path, adapter_path):
    primary = next(row for row in settings if row["role"] == "primary")
    low_rate = next(row for row in settings if row["role"] == "low_rate")
    return {
        "stage": 138,
        "policy_name": "deployable_render_aware_scaled_adapter_delta_selected_residual_codec_v1",
        "policy_type": "deployable_predictor_policy",
        "status": "current_best_no_teacher_deployable_render_aware",
        "replaces_policy": stage132_policy["policy_name"],
        "selected_primary_setting": primary["setting_label"],
        "selected_primary_adapter_delta_scale": primary["adapter_delta_scale"],
        "optional_low_rate_setting": low_rate["setting_label"],
        "optional_low_rate_adapter_delta_scale": low_rate["adapter_delta_scale"],
        "predictor": {
            "name": "render_aware_scaled_adapter_delta_selected_predictor",
            "rule": "run pre-shared Stage65 adapter and use adapter_delta_scale * (adapter_attrs - linear_attrs) at deterministic selected indices",
            "stage65_adapter_checkpoint": str(adapter_path),
            "checkpoint_exists": int(adapter_path.exists()),
            "checkpoint_rate_accounting": "pre-shared model; not counted per frame. If transmitted in-session, account separately.",
        },
        "index_selection": stage132_policy["index_selection"],
        "rate_accounting": {
            "transmitted_residual_payload_bytes": 0,
            "transmitted_selected_index_bytes": 0,
            "direct_total_rate": "q12_main_anchor_mib_per_frame only for predictor residuals",
            "amortized_total_rate": "same as direct for predictor residuals in this package",
            "adapter_delta_scale_rate": "policy constant; no per-frame side-info bytes",
        },
        "settings": settings,
        "decoder_contract": {
            "inputs": [
                "left_anchor",
                "right_anchor",
                "normalized_time",
                "pre-shared Stage65 adapter checkpoint",
                "policy setting keep_fraction",
                "policy setting adapter_delta_scale",
            ],
            "steps": [
                "compute linear base attrs from left/right anchors and normalized time",
                "compute Stage65 adapter attrs from left/right anchors and normalized time",
                "recompute endpoint-diff selected indices",
                "compute adapter_delta_scale * (adapter_attrs - linear_attrs) residual values",
                "apply scaled residual values at selected indices",
            ],
            "forbidden_inputs": [
                "target_dense_anchor",
                "target_residual",
                "target_rgb",
                "oracle_task_label",
                "transmitted selected indices",
                "transmitted residual values",
                "teacher residual side-info",
            ],
        },
        "references": {
            "stage132_previous_policy": str(stage132_policy_path),
            "stage137_summary": str(stage137_summary_path),
            "stage137_broader_best_summary_row": stage137_summary["broader_best_summary_row"],
            "stage137_smoke_selected_summary_row": stage137_summary["smoke_selected_summary_row"],
        },
        "limitations": [
            "Quality gain over Stage132 is small but positive on the 60-task Stage137 validation slice.",
            "Pre-shared Stage65 adapter checkpoint is required for deployment.",
            "Scale is a fixed policy constant selected offline; target RGB is not available to the decoder.",
            "Dedicated selected residual MLP remains rejected because Stage129/Stage134 showed rendered PSNR regression.",
        ],
    }


def format_float(value):
    return f"{float(value):.6f}"


def write_report(policy, package, path):
    lines = [
        "# Stage138 Render-Aware Deployable Policy Package",
        "",
        "## Policy",
        "",
        f"- policy: `{policy['policy_name']}`",
        f"- status: `{policy['status']}`",
        f"- replaces: `{policy['replaces_policy']}`",
        f"- primary setting: `{policy['selected_primary_setting']}` scale `{policy['selected_primary_adapter_delta_scale']}`",
        f"- optional low-rate setting: `{policy['optional_low_rate_setting']}` scale `{policy['optional_low_rate_adapter_delta_scale']}`",
        f"- adapter checkpoint exists: `{policy['predictor']['checkpoint_exists']}`",
        "",
        "## Settings",
        "",
        "| role | setting | keep | scale | rate | PSNR | delta base | delta full | delta Stage132 | residual bytes | index bytes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy["settings"]:
        lines.append(
            f"| {row['role']} | {row['setting_label']} | {row['keep_fraction']} | {row['adapter_delta_scale']} | {format_float(row['direct_total_mib_per_frame'])} | {format_float(row['psnr'])} | {format_float(row['delta_psnr_vs_linear_base'])} | {format_float(row['delta_psnr_vs_full_adapter'])} | {format_float(row['delta_psnr_vs_stage132_policy'])} | {row['transmitted_residual_payload_bytes']} | {row['transmitted_selected_index_bytes']} |"
        )
    lines.extend([
        "",
        "## Decoder Contract",
        "",
        "- Inputs: left anchor, right anchor, normalized time, pre-shared Stage65 adapter, policy keep fraction, policy adapter-delta scale.",
        "- Forbidden: target dense anchor, target residual, target RGB, oracle labels, transmitted selected indices, transmitted residual values, teacher residual side-info.",
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
    parser.add_argument("--stage132_policy", type=Path, default=DEFAULT_STAGE132_POLICY)
    parser.add_argument("--stage137_summary", type=Path, default=DEFAULT_STAGE137_SUMMARY)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_STAGE65_ADAPTER)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage132_policy = read_json(args.stage132_policy)
    stage137_summary = read_json(args.stage137_summary)
    summary_rows = stage137_summary["summary_rows"]
    primary = build_setting(select_setting(summary_rows, "primary"), "primary", previous_stage132_row(stage132_policy, "primary"))
    low_rate = build_setting(select_setting(summary_rows, "low_rate"), "low_rate", previous_stage132_row(stage132_policy, "low_rate"))
    settings = [primary, low_rate]
    policy = build_policy(settings, stage132_policy, args.stage132_policy, stage137_summary, args.stage137_summary, args.adapter)
    policy_json = args.summary_root / "stage138_render_aware_deployable_policy.json"
    settings_csv = args.summary_root / "stage138_render_aware_deployable_policy_settings.csv"
    package_json = args.summary_root / "stage138_render_aware_deployable_policy_package.json"
    report_md = args.summary_root / "stage138_render_aware_deployable_policy_report.md"
    write_csv(settings, settings_csv, SETTING_FIELDS)
    policy_json.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    package = {
        "stage": 138,
        "mode": "render-aware deployable predictor policy package",
        "policy_name": policy["policy_name"],
        "status": policy["status"],
        "replaces_policy": policy["replaces_policy"],
        "policy_json": str(policy_json),
        "settings_csv": str(settings_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "primary_setting": policy["selected_primary_setting"],
        "primary_adapter_delta_scale": policy["selected_primary_adapter_delta_scale"],
        "optional_low_rate_setting": policy["optional_low_rate_setting"],
        "optional_low_rate_adapter_delta_scale": policy["optional_low_rate_adapter_delta_scale"],
        "adapter_checkpoint_exists": policy["predictor"]["checkpoint_exists"],
        "notes": policy["limitations"],
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(policy, package, report_md)
    print(json.dumps({
        "package": str(package_json),
        "policy": policy["policy_name"],
        "primary": primary,
        "low_rate": low_rate,
        "adapter_checkpoint_exists": policy["predictor"]["checkpoint_exists"],
    }, indent=2))


if __name__ == "__main__":
    main()
