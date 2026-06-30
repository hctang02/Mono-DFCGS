import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE157_PACKAGE = REPO_ROOT / "experiments/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_broader_validation_package.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage158_recovered_middle_policy_package"


EVIDENCE_FIELDS = [
    "gap",
    "task_count",
    "mean_psnr",
    "p10_psnr",
    "mean_ssim",
    "mean_ms_ssim",
    "mean_lpips",
    "p90_lpips",
    "mean_original_psnr",
    "mean_original_ssim",
    "mean_original_ms_ssim",
    "mean_original_lpips",
    "mean_payload_bytes",
    "mean_side_mib_per_intermediate",
    "mean_direct_total_mib_per_frame_ref",
    "mean_delta_psnr_vs_original",
    "mean_delta_ssim_vs_original",
    "mean_delta_ms_ssim_vs_original",
    "mean_delta_lpips_vs_original",
    "passes_quality_gate",
]

DECISION_FIELDS = ["item", "decision", "evidence"]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_evidence(summary_rows, psnr_target):
    out = []
    for row in summary_rows:
        passes = (
            float(row["mean_psnr"]) >= float(psnr_target)
            and float(row["mean_delta_ssim_vs_original"]) > 0.0
            and float(row["mean_delta_ms_ssim_vs_original"]) > 0.0
            and float(row["mean_delta_lpips_vs_original"]) < 0.0
        )
        out.append({
            "gap": int(row["gap"]),
            "task_count": int(row["task_count"]),
            "mean_psnr": float(row["mean_psnr"]),
            "p10_psnr": float(row["p10_psnr"]),
            "mean_ssim": float(row["mean_ssim"]),
            "mean_ms_ssim": float(row["mean_ms_ssim"]),
            "mean_lpips": float(row["mean_lpips"]),
            "p90_lpips": float(row["p90_lpips"]),
            "mean_original_psnr": float(row["mean_original_psnr"]),
            "mean_original_ssim": float(row["mean_original_ssim"]),
            "mean_original_ms_ssim": float(row["mean_original_ms_ssim"]),
            "mean_original_lpips": float(row["mean_original_lpips"]),
            "mean_payload_bytes": float(row["mean_payload_bytes"]),
            "mean_side_mib_per_intermediate": float(row["mean_side_mib_per_intermediate"]),
            "mean_direct_total_mib_per_frame_ref": float(row["mean_direct_total_mib_per_frame_ref"]),
            "mean_delta_psnr_vs_original": float(row["mean_delta_psnr_vs_original"]),
            "mean_delta_ssim_vs_original": float(row["mean_delta_ssim_vs_original"]),
            "mean_delta_ms_ssim_vs_original": float(row["mean_delta_ms_ssim_vs_original"]),
            "mean_delta_lpips_vs_original": float(row["mean_delta_lpips_vs_original"]),
            "passes_quality_gate": passes,
        })
    return out


def build_decisions(evidence, psnr_target):
    min_psnr = min(float(row["mean_psnr"]) for row in evidence)
    min_ssim_delta = min(float(row["mean_delta_ssim_vs_original"]) for row in evidence)
    min_msssim_delta = min(float(row["mean_delta_ms_ssim_vs_original"]) for row in evidence)
    max_lpips_delta = max(float(row["mean_delta_lpips_vs_original"]) for row in evidence)
    max_rate = max(float(row["mean_direct_total_mib_per_frame_ref"]) for row in evidence)
    return [
        {
            "item": "quality_target",
            "decision": "passes_sampled_middle_target" if min_psnr >= float(psnr_target) else "below_target",
            "evidence": f"minimum gap mean PSNR = {min_psnr}, target = {psnr_target}",
        },
        {
            "item": "structural_metrics",
            "decision": "ssim_and_msssim_improve" if min_ssim_delta > 0.0 and min_msssim_delta > 0.0 else "structural_regression",
            "evidence": f"min SSIM delta = {min_ssim_delta}, min MS-SSIM delta = {min_msssim_delta}",
        },
        {
            "item": "perceptual_metric",
            "decision": "lpips_improves" if max_lpips_delta < 0.0 else "lpips_regression",
            "evidence": f"max LPIPS delta vs original = {max_lpips_delta}",
        },
        {
            "item": "rate_accounting",
            "decision": "residual_and_selector_bytes_counted",
            "evidence": f"max reference direct total rate = {max_rate} MiB/frame",
        },
        {
            "item": "method_status",
            "decision": "current_quality_safe_gs_domain_candidate",
            "evidence": "Stage157 validates StreamSplat-guided half-anchor Gaussian residual side-info on 120 sampled q12 gap4/gap8 tasks.",
        },
    ]


def write_report(policy, evidence, decisions, path):
    lines = [
        "# Stage158 Recovered Middle-Frame Policy Package",
        "",
        "## Policy",
        "",
        f"- Name: `{policy['policy_name']}`",
        f"- Base: `{policy['base_method']}`",
        f"- Correction: `{policy['correction']}`",
        f"- Keep fraction: `{policy['keep_fraction']}`",
        f"- Side bits: `{policy['side_bits']}`",
        f"- Selector metadata: `{policy['half_selector_payload_bytes']} byte/intermediate`",
        "",
        "## Evidence",
        "",
        "| gap | tasks | PSNR | p10 PSNR | SSIM | MS-SSIM | LPIPS | original PSNR | original LPIPS | payload bytes | direct rate ref | delta PSNR | delta LPIPS | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(
            f"| {row['gap']} | {row['task_count']} | {row['mean_psnr']:.6f} | {row['p10_psnr']:.6f} | {row['mean_ssim']:.6f} | {row['mean_ms_ssim']:.6f} | {row['mean_lpips']:.6f} | {row['mean_original_psnr']:.6f} | {row['mean_original_lpips']:.6f} | {row['mean_payload_bytes']:.3f} | {row['mean_direct_total_mib_per_frame_ref']:.6f} | {row['mean_delta_psnr_vs_original']:.6f} | {row['mean_delta_lpips_vs_original']:.6f} | {row['passes_quality_gate']} |"
        )
    lines.extend([
        "",
        "## Decisions",
        "",
        "| item | decision | evidence |",
        "|---|---|---|",
    ])
    for row in decisions:
        lines.append(f"| {row['item']} | {row['decision']} | {row['evidence']} |")
    lines.extend([
        "",
        "## Decoder Contract",
        "",
        "Allowed decoder inputs:",
        "",
        "- Original StreamSplat endpoint inputs/base used to produce target-time half anchors.",
        "- Normalized time.",
        "- Encoded entropy residual payload for the selected half-anchor.",
        "- Counted half-selector metadata.",
        "",
        "Forbidden decoder inputs:",
        "",
        "- Unencoded target dense anchor.",
        "- Target RGB.",
        "- Unencoded target residual tensors.",
        "- Oracle labels not represented in the transmitted payload.",
        "",
        "## Notes",
        "",
        "- Stage155 image residual is retained only as an upper-bound diagnostic.",
        "- Stage157 is the current GS-domain quality-safe candidate.",
        "- The target dense anchor is used encoder-side to build residual payloads and for offline diagnostics only.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage157_package", type=Path, default=DEFAULT_STAGE157_PACKAGE)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--psnr_target", type=float, default=26.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage157 = read_json(args.stage157_package)
    evidence = build_evidence(stage157["summary_rows"], args.psnr_target)
    decisions = build_decisions(evidence, args.psnr_target)
    policy = {
        "stage": 158,
        "policy_name": "streamsplat_guided_half_anchor_entropy_residual_v1",
        "status": "current_quality_safe_middle_frame_candidate",
        "base_method": "original_streamsplat_target_time_half_anchor",
        "correction": "entropy_coded_residual_to_encoder_side_target_dense_anchor",
        "half_policy": "best_half_selector",
        "keep_fraction": 1.0,
        "side_bits": 6,
        "half_selector_payload_bytes": 1,
        "validation_scope": "120 sampled q12 gap4/gap8 eval tasks from Stage153/154 protocol",
        "evidence": evidence,
        "decisions": decisions,
        "decoder_allowed_inputs": [
            "original StreamSplat endpoint inputs/base",
            "normalized time",
            "encoded entropy residual payload for selected half-anchor",
            "counted half-selector metadata",
        ],
        "decoder_forbidden_inputs": [
            "unencoded target dense anchor",
            "target RGB",
            "unencoded target residual tensors",
            "oracle labels not represented in transmitted payload",
        ],
        "source_stage157_package": str(args.stage157_package),
    }
    evidence_csv = args.summary_root / "stage158_recovered_middle_policy_evidence.csv"
    decisions_csv = args.summary_root / "stage158_recovered_middle_policy_decisions.csv"
    policy_json = args.summary_root / "stage158_recovered_middle_policy.json"
    summary_json = args.summary_root / "stage158_recovered_middle_policy_summary.json"
    package_json = args.summary_root / "stage158_recovered_middle_policy_package.json"
    report_md = args.summary_root / "stage158_recovered_middle_policy_report.md"
    write_csv(evidence, evidence_csv, EVIDENCE_FIELDS)
    write_csv(decisions, decisions_csv, DECISION_FIELDS)
    policy["evidence_csv"] = str(evidence_csv)
    policy["decisions_csv"] = str(decisions_csv)
    policy["policy_json"] = str(policy_json)
    policy["summary_json"] = str(summary_json)
    policy["package_json"] = str(package_json)
    policy["report_md"] = str(report_md)
    summary = {
        "stage": 158,
        "policy_name": policy["policy_name"],
        "status": policy["status"],
        "evidence": evidence,
        "decisions": decisions,
        "outputs": {
            "evidence_csv": str(evidence_csv),
            "decisions_csv": str(decisions_csv),
            "policy_json": str(policy_json),
            "summary_json": str(summary_json),
            "package_json": str(package_json),
            "report_md": str(report_md),
        },
    }
    policy_json.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    write_report(policy, evidence, decisions, report_md)
    print(json.dumps({"policy": str(policy_json), "report": str(report_md), "decisions": decisions}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
