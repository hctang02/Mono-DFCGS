import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE150_SUMMARY = REPO_ROOT / "experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_package_summary.json"
DEFAULT_STAGE150_PACKAGE = REPO_ROOT / "experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_package.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage151_middle_frame_recovery_policy_package"

EVIDENCE_FIELDS = [
    "reference_gap",
    "target_middle_psnr",
    "achieved_middle_psnr",
    "gap_to_target_psnr",
    "base_method",
    "codec",
    "keep_fraction",
    "side_bits",
    "task_count",
    "positive_delta_count",
    "positive_delta_fraction",
    "payload_bytes_per_intermediate_frame",
    "sideinfo_mib_per_intermediate_frame",
    "direct_total_mib_per_frame",
    "amortized_total_mib_per_frame",
    "decode_max_abs_diff",
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_evidence_rows(stage150_summary):
    rows = []
    for row in stage150_summary["rows"]:
        rows.append({
            "reference_gap": int(row["reference_gap"]),
            "target_middle_psnr": float(row["target_middle_psnr"]),
            "achieved_middle_psnr": float(row["mean_entropy_psnr"]),
            "gap_to_target_psnr": float(row["entropy_gap_to_target"]),
            "base_method": row["base_method"],
            "codec": row["codec"],
            "keep_fraction": float(row["keep_fraction"]),
            "side_bits": int(row["side_bits"]),
            "task_count": int(row["task_count"]),
            "positive_delta_count": int(row["positive_delta_count"]),
            "positive_delta_fraction": float(row["positive_delta_fraction"]),
            "payload_bytes_per_intermediate_frame": float(row["mean_entropy_payload_bytes"]),
            "sideinfo_mib_per_intermediate_frame": float(row["mean_entropy_mib_per_intermediate_frame"]),
            "direct_total_mib_per_frame": float(row["mean_direct_total_mib_per_frame"]),
            "amortized_total_mib_per_frame": float(row["mean_amortized_total_mib_per_frame"]),
            "decode_max_abs_diff": float(row["max_decoded_max_abs_diff_vs_fixed"]),
        })
    return sorted(rows, key=lambda row: row["reference_gap"])


def build_policy(evidence_rows, stage150_summary_path, stage150_package_path):
    return {
        "stage": 151,
        "policy_name": "middle_frame_recovery_linear_base_entropy_sideinfo_v1",
        "policy_type": "rate_counted_middle_frame_recovery_policy",
        "status": "target_recovered_on_full_q12_gap4_gap8_eval_rows",
        "base_reconstruction": {
            "method": "linear",
            "decoder_reproducible": True,
            "inputs": ["left_anchor", "right_anchor", "normalized_time"],
        },
        "anchor_codec": {
            "codec": "q12",
            "source": "Stage79/Stage150 q12 endpoint anchors",
        },
        "sideinfo_codec": {
            "setting_label": "q6_top10_entropy_index_value",
            "keep_fraction": 0.1,
            "side_bits": 6,
            "index_payload": "encoder selected residual-energy top10 indices, sorted/delta-coded, zlib-compressed",
            "value_payload": "q6 residual values, bitpacked and zlib-compressed",
            "metadata_payload": "float16 residual min/max per attr, zlib-compressed",
            "all_payload_bytes_counted": True,
        },
        "decoder_contract": {
            "allowed_inputs": ["left_anchor", "right_anchor", "normalized_time", "encoded_sideinfo_payload"],
            "forbidden_inputs": ["target_dense_anchor", "target_rgb", "unencoded_target_residual", "oracle labels not represented in payload"],
            "steps": [
                "compute linear base anchor from endpoints and normalized time",
                "decode side-info payload into selected indices and residual values",
                "apply decoded residual values to base anchor attributes",
                "render corrected anchor",
            ],
        },
        "evidence": evidence_rows,
        "target_recovery": {
            "all_gap_rows_above_corrected_targets": all(float(row["gap_to_target_psnr"]) >= 0.0 for row in evidence_rows),
            "min_gap_to_target_psnr": min(float(row["gap_to_target_psnr"]) for row in evidence_rows),
            "max_decode_abs_diff": max(float(row["decode_max_abs_diff"]) for row in evidence_rows),
            "min_positive_delta_fraction": min(float(row["positive_delta_fraction"]) for row in evidence_rows),
        },
        "source_paths": {
            "stage150_summary": str(stage150_summary_path),
            "stage150_package": str(stage150_package_path),
        },
        "next_optional_work": [
            "Build full-video RD plots around this policy.",
            "Evaluate lower-rate variants once the recovered target point is locked.",
            "Extend the same policy to gap16/all-gap packaging if needed.",
        ],
    }


def write_report(policy, evidence_rows, package, path):
    lines = [
        "# Stage151 Middle-Frame Recovery Policy Package",
        "",
        "## Policy",
        "",
        f"- policy: `{policy['policy_name']}`",
        f"- status: `{policy['status']}`",
        "- base: decoder-safe linear interpolation",
        "- side-info: q6/top10 entropy index+value residual payload",
        "- all side-info payload bytes are counted",
        "",
        "## Evidence",
        "",
        "| gap | target | achieved | achieved-target | tasks | positives | payload bytes | direct rate | amortized rate | decode diff |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evidence_rows:
        lines.append(
            f"| {row['reference_gap']} | {row['target_middle_psnr']:.6f} | {row['achieved_middle_psnr']:.6f} | {row['gap_to_target_psnr']:.6f} | {row['task_count']} | {row['positive_delta_count']}/{row['task_count']} | {row['payload_bytes_per_intermediate_frame']:.3f} | {row['direct_total_mib_per_frame']:.6f} | {row['amortized_total_mib_per_frame']:.6f} | {row['decode_max_abs_diff']:.6f} |"
        )
    lines.extend([
        "",
        "## Decoder Contract",
        "",
        "- Allowed decoder inputs: left anchor, right anchor, normalized time, encoded side-info payload.",
        "- Forbidden decoder inputs: target dense anchor, target RGB, unencoded target residual tensor, oracle labels not represented in payload.",
        "",
        "## Outputs",
        "",
        f"- policy JSON: `{package['policy_json']}`",
        f"- evidence CSV: `{package['evidence_csv']}`",
        f"- summary JSON: `{package['summary_json']}`",
        f"- package JSON: `{package['package_json']}`",
        f"- report Markdown: `{package['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage150_summary", type=Path, default=DEFAULT_STAGE150_SUMMARY)
    parser.add_argument("--stage150_package", type=Path, default=DEFAULT_STAGE150_PACKAGE)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage150_summary = read_json(args.stage150_summary)
    stage150_package = read_json(args.stage150_package)
    evidence_rows = build_evidence_rows(stage150_summary)
    policy = build_policy(evidence_rows, args.stage150_summary, args.stage150_package)
    if stage150_package["decision"] != "full_eval_validation_passed":
        raise RuntimeError(f"Stage151 expects Stage150 pass, got {stage150_package['decision']}")
    if not policy["target_recovery"]["all_gap_rows_above_corrected_targets"]:
        raise RuntimeError("Stage151 cannot package a target recovery policy if any gap is below target")

    policy_json = args.summary_root / "stage151_middle_frame_recovery_policy.json"
    evidence_csv = args.summary_root / "stage151_middle_frame_recovery_evidence.csv"
    summary_json = args.summary_root / "stage151_middle_frame_recovery_policy_summary.json"
    package_json = args.summary_root / "stage151_middle_frame_recovery_policy_package.json"
    report_md = args.summary_root / "stage151_middle_frame_recovery_policy_report.md"
    package = {
        "stage": 151,
        "mode": "middle-frame recovery policy package",
        "policy_name": policy["policy_name"],
        "status": policy["status"],
        "policy_json": str(policy_json),
        "evidence_csv": str(evidence_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "target_recovery": policy["target_recovery"],
    }
    summary = {
        "stage": 151,
        "mode": package["mode"],
        "policy": policy,
        "package": package,
    }
    write_csv(evidence_rows, evidence_csv, EVIDENCE_FIELDS)
    policy_json.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(policy, evidence_rows, package, report_md)
    print(json.dumps({"package": str(package_json), "status": package["status"], "target_recovery": package["target_recovery"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
