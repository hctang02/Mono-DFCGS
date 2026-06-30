import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage161_stage158_method_narrative_package"

STAGE153_SUMMARY = REPO_ROOT / "experiments/stage153_middle_multimetric_badcase_eval/stage153_middle_multimetric_summary.csv"
STAGE154_SUMMARY = REPO_ROOT / "experiments/stage154_original_streamsplat_middle_base_alignment/stage154_original_streamsplat_middle_summary.csv"
STAGE155_SUMMARY = REPO_ROOT / "experiments/stage155_streamsplat_base_sideinfo_upper_bound/stage155_streamsplat_base_sideinfo_summary.csv"
STAGE156_SUMMARY = REPO_ROOT / "experiments/stage156_streamsplat_half_anchor_gaussian_residual/stage156_streamsplat_half_anchor_summary.csv"
STAGE158_EVIDENCE = REPO_ROOT / "experiments/stage158_recovered_middle_policy_package/stage158_recovered_middle_policy_evidence.csv"
STAGE160_SUMMARY = REPO_ROOT / "experiments/stage160_stage158_extended_subjective_evidence/stage160_stage158_extended_subjective_evidence_summary.csv"
STAGE160_PACKAGE = REPO_ROOT / "experiments/stage160_stage158_extended_subjective_evidence/stage160_stage158_extended_subjective_evidence_package.json"

EVIDENCE_CHAIN_FIELDS = ["stage", "role", "artifact", "decision", "key_result"]
METHOD_COMPARISON_FIELDS = [
    "gap", "method", "task_count", "psnr", "ssim", "ms_ssim", "lpips", "payload_bytes", "direct_rate_ref", "status",
]
SUBJECTIVE_FIELDS = [
    "sequence", "task_count", "mean_key_avg_psnr", "mean_key_avg_lpips", "mean_middle_psnr", "mean_middle_lpips",
    "mean_original_middle_psnr", "mean_original_middle_lpips", "mean_delta_psnr_vs_original", "mean_delta_lpips_vs_original",
    "mean_payload_bytes", "mean_direct_total_mib_per_frame_ref",
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


def find(rows, **kwargs):
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in kwargs.items()):
            return row
    return None


def f(row, key, default=None):
    if row is None:
        return default
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return float(value)


def i(row, key, default=0):
    if row is None:
        return default
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return int(float(value))


def build_method_comparison(stage153, stage154, stage155, stage156, stage158):
    rows = []
    for gap in [4, 8]:
        linear_recovered = find(stage153, gap=gap, method="stage151_recovered_linear_base_sideinfo")
        original = find(stage154, gap=gap, method="original_streamsplat_middle_base")
        image_q4 = None
        for row in stage155:
            if int(row["gap"]) == gap and row["method"] == "image_residual_sideinfo_full_frame" and int(row["side_bits"]) == 4:
                image_q4 = row
                break
        sampled_half = None
        for row in stage156:
            if int(row["gap"]) == gap and row["method"] == "streamsplat_half_anchor_entropy_residual" and row["half_policy"] == "best_half_selector" and float(row["keep_fraction"]) == 1.0 and int(row["side_bits"]) == 6:
                sampled_half = row
                break
        broader_stage158 = find(stage158, gap=gap)
        rows.extend([
            {
                "gap": gap,
                "method": "stage151_linear_base_entropy_sideinfo_reference",
                "task_count": i(linear_recovered, "task_count"),
                "psnr": f(linear_recovered, "mean_psnr"),
                "ssim": f(linear_recovered, "mean_ssim"),
                "ms_ssim": f(linear_recovered, "mean_ms_ssim"),
                "lpips": f(linear_recovered, "mean_lpips"),
                "payload_bytes": f(linear_recovered, "mean_payload_bytes"),
                "direct_rate_ref": None,
                "status": "historical_psnr_recovery_reference_not_final_visual_base",
            },
            {
                "gap": gap,
                "method": "original_streamsplat_middle_base",
                "task_count": i(original, "task_count"),
                "psnr": f(original, "mean_psnr"),
                "ssim": f(original, "mean_ssim"),
                "ms_ssim": f(original, "mean_ms_ssim"),
                "lpips": f(original, "mean_lpips"),
                "payload_bytes": 0.0,
                "direct_rate_ref": None,
                "status": "perceptual_motion_geometry_base",
            },
            {
                "gap": gap,
                "method": "stage155_q4_image_residual_upper_bound",
                "task_count": i(image_q4, "task_count"),
                "psnr": f(image_q4, "mean_psnr"),
                "ssim": f(image_q4, "mean_ssim"),
                "ms_ssim": f(image_q4, "mean_ms_ssim"),
                "lpips": f(image_q4, "mean_lpips"),
                "payload_bytes": f(image_q4, "mean_payload_bytes"),
                "direct_rate_ref": f(image_q4, "mean_direct_total_mib_per_frame_ref"),
                "status": "upper_bound_only_not_final_gs_method",
            },
            {
                "gap": gap,
                "method": "stage156_sampled_half_anchor_keep1_q6",
                "task_count": i(sampled_half, "task_count"),
                "psnr": f(sampled_half, "mean_psnr"),
                "ssim": f(sampled_half, "mean_ssim"),
                "ms_ssim": f(sampled_half, "mean_ms_ssim"),
                "lpips": f(sampled_half, "mean_lpips"),
                "payload_bytes": f(sampled_half, "mean_payload_bytes"),
                "direct_rate_ref": f(sampled_half, "mean_direct_total_mib_per_frame_ref"),
                "status": "sampled_gs_domain_discovery",
            },
            {
                "gap": gap,
                "method": "stage158_streamsplat_guided_half_anchor_entropy_residual_v1",
                "task_count": i(broader_stage158, "task_count"),
                "psnr": f(broader_stage158, "mean_psnr"),
                "ssim": f(broader_stage158, "mean_ssim"),
                "ms_ssim": f(broader_stage158, "mean_ms_ssim"),
                "lpips": f(broader_stage158, "mean_lpips"),
                "payload_bytes": f(broader_stage158, "mean_payload_bytes"),
                "direct_rate_ref": f(broader_stage158, "mean_direct_total_mib_per_frame_ref"),
                "status": "current_quality_first_middle_recovery_policy",
            },
        ])
    return rows


def build_evidence_chain(stage160_package):
    video = stage160_package["video_path"]
    contact = stage160_package["contact_sheet_path"]
    return [
        {
            "stage": 151,
            "role": "historical_psnr_reference",
            "artifact": "experiments/stage151_middle_frame_recovery_policy_package/",
            "decision": "not_final_visual_base",
            "key_result": "Linear-base q6/top10 side-info recovered corrected PSNR targets but Stage153/152 showed visual/perceptual risk.",
        },
        {
            "stage": 153,
            "role": "multi_metric_diagnostic",
            "artifact": str(STAGE153_SUMMARY),
            "decision": "psnr_alone_insufficient",
            "key_result": "Stage151 improved PSNR/SSIM but LPIPS and bad cases motivated returning to original StreamSplat guidance.",
        },
        {
            "stage": 154,
            "role": "streamsplat_base_alignment",
            "artifact": str(STAGE154_SUMMARY),
            "decision": "use_original_streamsplat_as_base",
            "key_result": "Original StreamSplat had lower PSNR than Stage151 but better LPIPS, making it the preferred motion/geometry base.",
        },
        {
            "stage": 155,
            "role": "achievability_upper_bound",
            "artifact": str(STAGE155_SUMMARY),
            "decision": "image_residual_is_upper_bound_only",
            "key_result": "q4 image residual on StreamSplat base reached 31-32 dB but is not the final Gaussian-domain method.",
        },
        {
            "stage": 156,
            "role": "gs_domain_discovery",
            "artifact": str(STAGE156_SUMMARY),
            "decision": "select_best_half_keep1_q6",
            "key_result": "Half-anchor Gaussian residual keep1/q6 exceeded 29 dB sampled with improved LPIPS/SSIM over original StreamSplat.",
        },
        {
            "stage": 158,
            "role": "policy_contract",
            "artifact": str(STAGE158_EVIDENCE),
            "decision": "freeze_quality_first_policy",
            "key_result": "Broader 120-task validation passes: gap4 29.7805 dB, gap8 29.5787 dB, LPIPS improves on both gaps.",
        },
        {
            "stage": 160,
            "role": "subjective_evidence",
            "artifact": video,
            "decision": "subjective_examples_available",
            "key_result": f"Extended 24-frame gap4 video and contact sheet are available: {video}; {contact}.",
        },
    ]


def write_report(package, method_rows, sequence_rows, evidence_rows, path):
    lines = [
        "# Stage161 Stage158 Method Narrative Package",
        "",
        "## Claim",
        "",
        "`streamsplat_guided_half_anchor_entropy_residual_v1` is the current quality-first middle-frame recovery method.",
        "It keeps original StreamSplat as the target-time motion/geometry base and corrects one selected half-anchor with counted Gaussian-domain entropy residual side-info.",
        "Rate is explicitly counted but not aggressively optimized at this stage, per user direction.",
        "",
        "## Decoder Contract",
        "",
        "Allowed decoder inputs:",
        "",
        "- Original StreamSplat endpoint/base inputs.",
        "- Normalized time.",
        "- Encoded q6/keep1.0 entropy residual payload for the selected half-anchor.",
        "- Counted one-byte half selector.",
        "",
        "Forbidden decoder inputs:",
        "",
        "- Target dense anchor.",
        "- Target RGB.",
        "- Unencoded target residual tensors.",
        "- Oracle labels not represented in the transmitted payload.",
        "",
        "## Gap-Level Evidence",
        "",
        "| gap | method | tasks | PSNR | SSIM | MS-SSIM | LPIPS | payload bytes | direct rate ref | status |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in method_rows:
        lines.append(
            f"| {row['gap']} | {row['method']} | {row['task_count']} | {fmt(row['psnr'])} | {fmt(row['ssim'])} | {fmt(row['ms_ssim'])} | {fmt(row['lpips'])} | {fmt(row['payload_bytes'])} | {fmt(row['direct_rate_ref'])} | {row['status']} |"
        )
    lines.extend([
        "",
        "## Stage160 Subjective Evidence",
        "",
        f"- Video: `{package['stage160_video_path']}`",
        f"- Contact sheet: `{package['stage160_contact_sheet_path']}`",
        f"- Video bytes: `{package['stage160_video_file_bytes']}`",
        f"- Contact sheet bytes: `{package['stage160_contact_sheet_file_bytes']}`",
        "",
        "| sequence | tasks | key PSNR/LPIPS | Stage158 middle PSNR/LPIPS | original middle PSNR/LPIPS | delta PSNR/LPIPS | payload bytes | rate ref |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sequence_rows:
        lines.append(
            f"| {row['sequence']} | {row['task_count']} | {fmt(row['mean_key_avg_psnr'])}/{fmt(row['mean_key_avg_lpips'])} | "
            f"{fmt(row['mean_middle_psnr'])}/{fmt(row['mean_middle_lpips'])} | {fmt(row['mean_original_middle_psnr'])}/{fmt(row['mean_original_middle_lpips'])} | "
            f"{fmt(row['mean_delta_psnr_vs_original'])}/{fmt(row['mean_delta_lpips_vs_original'])} | {fmt(row['mean_payload_bytes'])} | {fmt(row['mean_direct_total_mib_per_frame_ref'])} |"
        )
    lines.extend([
        "",
        "## Evidence Chain",
        "",
        "| stage | role | decision | key result |",
        "|---:|---|---|---|",
    ])
    for row in evidence_rows:
        lines.append(f"| {row['stage']} | {row['role']} | {row['decision']} | {row['key_result']} |")
    lines.extend([
        "",
        "## Next Direction",
        "",
        "Proceed to keyframe selector work. Encoder-side RGB/motion cues are allowed for selecting keyframes if keyframe indices are transmitted and counted; Stage162 will audit feature sources and feed-forward validity.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value):
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage153 = read_csv(STAGE153_SUMMARY)
    stage154 = read_csv(STAGE154_SUMMARY)
    stage155 = read_csv(STAGE155_SUMMARY)
    stage156 = read_csv(STAGE156_SUMMARY)
    stage158 = read_csv(STAGE158_EVIDENCE)
    stage160_summary = read_csv(STAGE160_SUMMARY)
    stage160_package = read_json(STAGE160_PACKAGE)
    method_rows = build_method_comparison(stage153, stage154, stage155, stage156, stage158)
    evidence_rows = build_evidence_chain(stage160_package)
    evidence_csv = args.summary_root / "stage161_stage158_evidence_chain.csv"
    comparison_csv = args.summary_root / "stage161_stage158_method_comparison.csv"
    subjective_csv = args.summary_root / "stage161_stage158_subjective_sequence_summary.csv"
    package_json = args.summary_root / "stage161_stage158_method_narrative_package.json"
    report_md = args.summary_root / "stage161_stage158_method_narrative_report.md"
    write_csv(evidence_rows, evidence_csv, EVIDENCE_CHAIN_FIELDS)
    write_csv(method_rows, comparison_csv, METHOD_COMPARISON_FIELDS)
    write_csv(stage160_summary, subjective_csv, SUBJECTIVE_FIELDS)
    package = {
        "stage": 161,
        "policy": "streamsplat_guided_half_anchor_entropy_residual_v1",
        "status": "quality_first_middle_frame_recovery_method_packaged",
        "rate_position": "explicitly_counted_but_not_over_optimized",
        "innovation_claim": "Original StreamSplat target-time half-anchor is corrected in Gaussian space by counted entropy-coded residual side-info.",
        "decoder_allowed_inputs": [
            "original StreamSplat endpoint/base inputs",
            "normalized time",
            "encoded q6/keep1.0 entropy residual payload",
            "counted one-byte half selector",
        ],
        "decoder_forbidden_inputs": [
            "target dense anchor",
            "target RGB",
            "unencoded target residual tensors",
            "oracle labels not represented in transmitted payload",
        ],
        "method_comparison": method_rows,
        "subjective_sequence_summary": stage160_summary,
        "evidence_chain": evidence_rows,
        "stage160_video_path": stage160_package["video_path"],
        "stage160_contact_sheet_path": stage160_package["contact_sheet_path"],
        "stage160_video_file_bytes": stage160_package["video_file_bytes"],
        "stage160_contact_sheet_file_bytes": stage160_package["contact_sheet_file_bytes"],
        "evidence_csv": str(evidence_csv),
        "comparison_csv": str(comparison_csv),
        "subjective_csv": str(subjective_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, method_rows, stage160_summary, evidence_rows, report_md)
    print(json.dumps({"package": str(package_json), "report": str(report_md), "status": package["status"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
