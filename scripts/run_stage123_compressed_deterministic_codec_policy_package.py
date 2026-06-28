import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE114_POLICY = REPO_ROOT / "experiments/stage114_strict_safe_selector_fallback_package/stage114_strict_safe_selector_fallback_policy.json"
DEFAULT_STAGE122_PACKAGE = REPO_ROOT / "experiments/stage122_compressed_deterministic_rd_package/stage122_compressed_deterministic_rd_package.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage123_compressed_deterministic_codec_policy_package"


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.residual_sideinfo_codec import (  # noqa: E402
    DETERMINISTIC_ENTROPY_HEADER_STRUCT,
    DETERMINISTIC_ENTROPY_MAGIC,
    VERSION,
)


ROLE_ORDER = {"primary": 0, "low_rate": 1, "near_anchor": 2, "anchor": 3}
SETTING_FIELDS = [
    "role",
    "setting_label",
    "keep_fraction",
    "side_bits",
    "mean_sideinfo_payload_bytes",
    "mean_sideinfo_mib_per_intermediate_frame",
    "mean_direct_total_mib_per_frame",
    "mean_amortized_total_mib_per_frame",
    "mean_psnr",
    "mean_delta_psnr_vs_q6_top10",
    "weighted_direct_rate_delta_vs_stage96_entropy",
    "weighted_psnr_delta_vs_stage96_entropy",
]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_float(value):
    return f"{float(value):.6f}"


def sorted_settings(rd_package):
    rows = list(rd_package["setting_summary_rows"])
    return sorted(rows, key=lambda row: ROLE_ORDER[row["role"]])


def build_policy(selector_policy, rd_package, settings, selector_policy_path, rd_package_path):
    if selector_policy["policy_name"] != "strict_safe_endpoint_selector_v1":
        raise ValueError("Stage123 expects the Stage114 strict-safe selector policy")
    if selector_policy["selected_candidate"] != "endpoint_diff_baseline":
        raise ValueError("Stage123 expects endpoint_diff_baseline as the selected candidate")

    return {
        "stage": 123,
        "policy_name": "compressed_deterministic_value_only_residual_codec_v1",
        "policy_type": "codec_policy_manifest",
        "status": "package_not_full_residual_predictor",
        "stage122_commit": "9972a6a",
        "selector_policy": {
            "policy_name": selector_policy["policy_name"],
            "policy_type": selector_policy["policy_type"],
            "selected_candidate": selector_policy["selected_candidate"],
            "source_policy_json": str(selector_policy_path),
            "validation_summary": selector_policy["validation_summary"],
        },
        "index_selection": {
            "rule_name": "endpoint_diff_topk_v1",
            "selected_candidate": "endpoint_diff_baseline",
            "decoder_reproducible": True,
            "transmitted_index_bytes": 0,
            "input_attrs": ["left_anchor_static_attrs", "right_anchor_static_attrs"],
            "shape_contract": "left_attrs and right_attrs must be matching [1, N, D] tensors with stable Gaussian ordering",
            "keep_count_rule": "min(max(round(N * keep_fraction), 0), N)",
            "score_rule": "sum((right_attrs[0].float() - left_attrs[0].float()) ** 2, dim=-1)",
            "selection_rule": "select top keep_count largest scores, then sort selected indices ascending",
        },
        "sideinfo_codec": {
            "codec_name": "compressed_deterministic_value_only_residual_sideinfo_v1",
            "encode_function": "mono_dfcgs.residual_sideinfo_codec.encode_selected_residual_values_sideinfo_entropy",
            "decode_function": "mono_dfcgs.residual_sideinfo_codec.decode_selected_residual_values_sideinfo_entropy",
            "payload_magic": DETERMINISTIC_ENTROPY_MAGIC.decode("ascii"),
            "version": VERSION,
            "header_struct": DETERMINISTIC_ENTROPY_HEADER_STRUCT.format,
            "header_bytes": DETERMINISTIC_ENTROPY_HEADER_STRUCT.size,
            "zlib_level": 9,
            "metadata_payload": "zlib(float16 residual min per attr + float16 residual max per attr)",
            "residual_payload": "zlib(bitpacked quantized residual values for deterministic selected indices)",
            "selected_index_payload": "omitted; decoder recomputes indices from endpoint-diff rule",
        },
        "recommended_settings": settings,
        "recommendation": rd_package["recommendation"],
        "rate_accounting": {
            "all_sideinfo_bytes_counted": True,
            "direct_total_rate": "q12_main_anchor_mib_per_frame + sideinfo_mib_per_intermediate_frame",
            "amortized_total_rate": "q12_main_anchor_mib_per_frame + sideinfo_mib_per_intermediate_frame * uniform_intermediate_frame_ratio",
            "stage96_entropy_reference": "q6/top10 entropy-coded index+value side-info",
        },
        "encoder_contract": {
            "inputs": [
                "left_anchor",
                "right_anchor",
                "base_attrs from selected base reconstruction method",
                "target_dense_anchor for teacher residual values",
            ],
            "steps": [
                "recompute deterministic endpoint-diff selected indices",
                "compute target residual values at selected indices",
                "encode compressed deterministic value-only residual side-info payload",
            ],
        },
        "decoder_contract": {
            "inputs": [
                "left_anchor",
                "right_anchor",
                "base_attrs from selected base reconstruction method",
                "compressed deterministic side-info payload",
                "keep_fraction and side_bits from policy setting",
            ],
            "forbidden_inputs": [
                "target_dense_anchor",
                "target_residual",
                "target_rgb",
                "oracle_task_label",
                "transmitted selected indices",
            ],
            "steps": [
                "recompute deterministic endpoint-diff selected indices",
                "decode value-only residual payload",
                "add decoded residuals to base_attrs at selected indices",
            ],
        },
        "validation_sources": {
            "selector_policy_json": str(selector_policy_path),
            "rd_package_json": str(rd_package_path),
            "rd_rows_csv": rd_package["rd_rows_csv"],
            "rd_points_csv": rd_package["rd_points_csv"],
            "rd_report_md": rd_package["report_md"],
        },
        "limitations": [
            "Residual values remain teacher-derived from dense target anchors at encoder side.",
            "This package is not yet a full deployable residual value predictor.",
            "Decoder deployability depends on stable Gaussian ordering and matching left/right anchor attributes.",
            "Stage96 entropy reference remains higher quality in the Stage122 comparison.",
        ],
        "next_step": "Train or package a feed-forward residual value predictor to replace teacher-derived residual values.",
    }


def write_report(policy, settings, path):
    lines = [
        "# Stage123 Compressed Deterministic Codec Policy Package",
        "",
        "## Scope",
        "",
        "- Packages selector, deterministic selected-index rule, side-info codec, and RD settings into one manifest.",
        "- Selected indices are decoder-reproducible and are not transmitted.",
        "- Side-info payload uses compressed deterministic value-only residuals.",
        "- Residual values remain teacher-derived; this is not a residual value predictor.",
        "",
        "## Policy",
        "",
        f"- policy: `{policy['policy_name']}`",
        f"- selector: `{policy['selector_policy']['policy_name']}`",
        f"- selected candidate: `{policy['selector_policy']['selected_candidate']}`",
        f"- index rule: `{policy['index_selection']['rule_name']}`",
        f"- side-info codec: `{policy['sideinfo_codec']['codec_name']}`",
        f"- payload magic: `{policy['sideinfo_codec']['payload_magic']}`",
        f"- header bytes: `{policy['sideinfo_codec']['header_bytes']}`",
        "",
        "## Settings",
        "",
        "| role | setting | keep | bits | payload bytes | direct | amortized | PSNR | delta q6 | dRate entropy | dPSNR entropy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in settings:
        lines.append(
            f"| {row['role']} | {row['setting_label']} | {row['keep_fraction']} | {row['side_bits']} | {format_float(row['mean_sideinfo_payload_bytes'])} | {format_float(row['mean_direct_total_mib_per_frame'])} | {format_float(row['mean_amortized_total_mib_per_frame'])} | {format_float(row['mean_psnr'])} | {format_float(row['mean_delta_psnr_vs_q6_top10'])} | {format_float(row['weighted_direct_rate_delta_vs_stage96_entropy'])} | {format_float(row['weighted_psnr_delta_vs_stage96_entropy'])} |"
        )
    lines.extend([
        "",
        "## Decoder Contract",
        "",
        f"- keep count: `{policy['index_selection']['keep_count_rule']}`",
        f"- score: `{policy['index_selection']['score_rule']}`",
        f"- selection: `{policy['index_selection']['selection_rule']}`",
        "- forbidden decoder inputs: target dense anchor, target residual, target RGB, oracle task label, transmitted selected indices.",
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
    parser.add_argument("--stage114_policy", type=Path, default=DEFAULT_STAGE114_POLICY)
    parser.add_argument("--stage122_package", type=Path, default=DEFAULT_STAGE122_PACKAGE)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    selector_policy = read_json(args.stage114_policy)
    rd_package = read_json(args.stage122_package)
    settings = sorted_settings(rd_package)
    policy = build_policy(selector_policy, rd_package, settings, args.stage114_policy, args.stage122_package)

    policy_json = args.summary_root / "stage123_compressed_deterministic_codec_policy.json"
    settings_csv = args.summary_root / "stage123_compressed_deterministic_codec_policy_settings.csv"
    package_json = args.summary_root / "stage123_compressed_deterministic_codec_policy_package.json"
    report_md = args.summary_root / "stage123_compressed_deterministic_codec_policy_report.md"
    policy["output_paths"] = {
        "policy_json": str(policy_json),
        "settings_csv": str(settings_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }

    write_csv(settings, settings_csv, SETTING_FIELDS)
    policy_json.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    package = {
        "stage": 123,
        "mode": "compressed deterministic codec policy package",
        "policy_name": policy["policy_name"],
        "stage114_policy": str(args.stage114_policy),
        "stage122_package": str(args.stage122_package),
        "policy_json": str(policy_json),
        "settings_csv": str(settings_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "setting_count": len(settings),
        "primary_setting": rd_package["recommendation"]["primary"],
        "low_rate_setting": rd_package["recommendation"]["low_rate"],
        "status": policy["status"],
        "notes": policy["limitations"],
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(policy, settings, report_md)
    print(json.dumps({"policy": str(policy_json), "setting_count": len(settings), "status": policy["status"]}, indent=2))


if __name__ == "__main__":
    main()
