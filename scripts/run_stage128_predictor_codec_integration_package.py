import argparse
import csv
import json
import sys
from pathlib import Path

from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE123_POLICY = REPO_ROOT / "experiments/stage123_compressed_deterministic_codec_policy_package/stage123_compressed_deterministic_codec_policy.json"
DEFAULT_STAGE126_STATS = REPO_ROOT / "experiments/stage126_selected_residual_predictor_dataset_package/stage126_selected_residual_predictor_train_stats.json"
DEFAULT_STAGE127_PACKAGE = REPO_ROOT / "experiments/stage127_selected_residual_predictor_training_smoke/stage127_selected_residual_predictor_training_smoke_package.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage128_predictor_codec_integration_package"

SETTING_FIELDS = [
    "setting_label",
    "setting_role",
    "keep_fraction",
    "side_bits",
    "feature_dim",
    "residual_dim",
    "hidden_dim",
    "checkpoint_exists",
    "checkpoint_load_ok",
    "checkpoint_path",
    "eval_mse_reduction",
    "transmitted_residual_payload_bytes",
    "transmitted_selected_index_bytes",
]


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.residual_value_predictor import SelectedResidualValueMLP  # noqa: E402


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


def verify_checkpoint(metric):
    checkpoint_path = Path(metric["checkpoint_path"])
    exists = checkpoint_path.exists()
    load_ok = False
    if exists:
        model = SelectedResidualValueMLP(
            feature_dim=int(metric["feature_dim"]),
            residual_dim=int(metric["residual_dim"]),
            hidden_dim=int(metric["hidden_dim"]),
        )
        state = load_file(str(checkpoint_path), device="cpu")
        model.load_state_dict(state, strict=True)
        load_ok = True
    return exists, load_ok


def build_setting_rows(stage127_package):
    rows = []
    for metric in stage127_package["metrics"]:
        exists, load_ok = verify_checkpoint(metric)
        rows.append({
            "setting_label": metric["setting_label"],
            "setting_role": metric["setting_role"],
            "keep_fraction": metric["keep_fraction"],
            "side_bits": metric["side_bits"],
            "feature_dim": metric["feature_dim"],
            "residual_dim": metric["residual_dim"],
            "hidden_dim": metric["hidden_dim"],
            "checkpoint_exists": int(exists),
            "checkpoint_load_ok": int(load_ok),
            "checkpoint_path": metric["checkpoint_path"],
            "eval_mse_reduction": metric["eval_mse_reduction"],
            "transmitted_residual_payload_bytes": 0,
            "transmitted_selected_index_bytes": 0,
        })
    return rows


def build_policy(stage123_policy, stage126_stats, stage127_package, setting_rows):
    return {
        "stage": 128,
        "policy_name": "predictor_only_selected_residual_codec_v1",
        "policy_type": "predictor_codec_integration_manifest",
        "status": "predictor_integrated_pending_rendered_validation",
        "base_codec_policy": stage123_policy["policy_name"],
        "selector_policy": stage123_policy["selector_policy"],
        "index_selection": stage123_policy["index_selection"],
        "residual_value_predictor": {
            "predictor_name": "selected_residual_value_mlp_v1",
            "module": "mono_dfcgs.residual_value_predictor.SelectedResidualValueMLP",
            "feature_builder": "mono_dfcgs.residual_value_predictor.selected_residual_feature_matrix",
            "checkpoint_source": stage127_package["package_json"],
            "normalization_stats_source": str(DEFAULT_STAGE126_STATS),
            "settings": setting_rows,
        },
        "rate_accounting": {
            "transmitted_residual_payload_bytes": 0,
            "transmitted_selected_index_bytes": 0,
            "sideinfo_bytes_for_selected_residual_values": 0,
            "note": "Predictor-only residual values are decoder-generated; model weights are assumed pre-shared by the policy package and are not counted per frame.",
        },
        "decoder_contract": {
            "inputs": [
                "left_anchor",
                "right_anchor",
                "base linear anchor attrs",
                "normalized_time",
                "policy setting keep_fraction",
                "pre-shared selected residual MLP checkpoint and normalization stats",
            ],
            "steps": [
                "recompute endpoint-diff top-k selected indices",
                "build selected residual feature matrix from left/right/base attrs and time",
                "normalize features with Stage126 train stats",
                "predict normalized residual values with the Stage127 MLP",
                "denormalize residual values and add them at selected indices",
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
        "validation_sources": {
            "stage123_policy": str(DEFAULT_STAGE123_POLICY),
            "stage126_stats": str(DEFAULT_STAGE126_STATS),
            "stage127_package": str(DEFAULT_STAGE127_PACKAGE),
        },
        "stage126_stats_digest": {
            setting: {
                "feature_dim": row["feature_dim"],
                "residual_dim": row["residual_dim"],
                "sample_count": row["sample_count"],
                "feature_rms": row["feature_rms"],
                "label_rms": row["label_rms"],
            }
            for setting, row in stage126_stats["stats_by_setting"].items()
        },
        "limitations": [
            "Stage128 only packages the predictor codec integration; rendered validation follows in Stage129.",
            "Model checkpoints are external files and are not committed to git.",
            "Policy assumes checkpoint weights are pre-shared and not counted per-frame.",
        ],
    }


def format_float(value):
    return f"{float(value):.6f}"


def write_report(policy, setting_rows, path):
    lines = [
        "# Stage128 Predictor Codec Integration Package",
        "",
        "## Scope",
        "",
        "- Packages Stage127 selected residual MLP checkpoints into a predictor-only codec manifest.",
        "- Residual values and selected indices are not transmitted per frame.",
        "- Checkpoints remain outside git and are referenced by path.",
        "",
        "## Settings",
        "",
        "| setting | role | keep | hidden | eval reduction | ckpt exists | load ok | residual bytes | index bytes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in setting_rows:
        lines.append(
            f"| {row['setting_label']} | {row['setting_role']} | {row['keep_fraction']} | {row['hidden_dim']} | {format_float(row['eval_mse_reduction'])} | {row['checkpoint_exists']} | {row['checkpoint_load_ok']} | {row['transmitted_residual_payload_bytes']} | {row['transmitted_selected_index_bytes']} |"
        )
    lines.extend([
        "",
        "## Decoder Contract",
        "",
        "- Recompute endpoint-diff selected indices from left/right anchors.",
        "- Build MLP features from left/right/base attrs and normalized time.",
        "- Decode residual values with pre-shared MLP weights and Stage126 normalization stats.",
        "- Do not use target dense anchors, target residuals, target RGB, oracle labels, transmitted indices, or transmitted residual values.",
        "",
        "## Outputs",
        "",
        f"- policy JSON: `{policy['output_paths']['policy_json']}`",
        f"- settings CSV: `{policy['output_paths']['settings_csv']}`",
        f"- package JSON: `{policy['output_paths']['package_json']}`",
        f"- report Markdown: `{policy['output_paths']['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage123_policy", type=Path, default=DEFAULT_STAGE123_POLICY)
    parser.add_argument("--stage126_stats", type=Path, default=DEFAULT_STAGE126_STATS)
    parser.add_argument("--stage127_package", type=Path, default=DEFAULT_STAGE127_PACKAGE)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage123_policy = read_json(args.stage123_policy)
    stage126_stats = read_json(args.stage126_stats)
    stage127_package = read_json(args.stage127_package)
    setting_rows = build_setting_rows(stage127_package)
    policy = build_policy(stage123_policy, stage126_stats, stage127_package, setting_rows)

    policy_json = args.summary_root / "stage128_predictor_codec_integration_policy.json"
    settings_csv = args.summary_root / "stage128_predictor_codec_integration_settings.csv"
    package_json = args.summary_root / "stage128_predictor_codec_integration_package.json"
    report_md = args.summary_root / "stage128_predictor_codec_integration_report.md"
    policy["validation_sources"] = {
        "stage123_policy": str(args.stage123_policy),
        "stage126_stats": str(args.stage126_stats),
        "stage127_package": str(args.stage127_package),
    }
    policy["output_paths"] = {
        "policy_json": str(policy_json),
        "settings_csv": str(settings_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    write_csv(setting_rows, settings_csv, SETTING_FIELDS)
    policy_json.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    package = {
        "stage": 128,
        "mode": "predictor codec integration package",
        "policy_name": policy["policy_name"],
        "status": policy["status"],
        "stage123_policy": str(args.stage123_policy),
        "stage126_stats": str(args.stage126_stats),
        "stage127_package": str(args.stage127_package),
        "policy_json": str(policy_json),
        "settings_csv": str(settings_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "setting_count": len(setting_rows),
        "all_checkpoints_load_ok": int(all(int(row["checkpoint_load_ok"]) == 1 for row in setting_rows)),
        "notes": policy["limitations"],
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(policy, setting_rows, report_md)
    print(json.dumps({"package": str(package_json), "setting_count": len(setting_rows), "all_checkpoints_load_ok": package["all_checkpoints_load_ok"]}, indent=2))


if __name__ == "__main__":
    main()
