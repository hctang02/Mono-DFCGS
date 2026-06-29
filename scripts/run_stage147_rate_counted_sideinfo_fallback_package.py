import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE96_SUMMARY = REPO_ROOT / "experiments/stage96_broader_entropy_residual_sideinfo_rd_package/stage96_broader_entropy_residual_sideinfo_rd_summary.json"
DEFAULT_STAGE142_TARGETS = REPO_ROOT / "experiments/stage142_middle_frame_protocol_alignment_audit/stage142_middle_frame_targets.csv"
DEFAULT_STAGE144_PACKAGE = REPO_ROOT / "experiments/stage144_high_rate_middle_frame_upper_bound/stage144_high_rate_middle_frame_upper_bound_package.json"
DEFAULT_STAGE146_SUMMARY = REPO_ROOT / "experiments/stage146_gap_balanced_adapter_training_v2/stage146_gap_balanced_adapter_training_v2_summary.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage147_rate_counted_sideinfo_fallback_package"


ROW_FIELDS = [
    "reference_gap",
    "target_middle_psnr",
    "base_method",
    "codec",
    "setting_label",
    "keep_fraction",
    "side_bits",
    "task_count",
    "mean_base_psnr",
    "mean_sideinfo_psnr",
    "mean_delta_psnr_vs_base",
    "base_gap_to_target",
    "sideinfo_gap_to_target",
    "positive_delta_count",
    "positive_delta_fraction",
    "entropy_sideinfo_mib_per_intermediate_frame",
    "entropy_sideinfo_payload_bytes_per_intermediate_frame",
    "entropy_direct_total_mib_per_frame",
    "entropy_amortized_total_mib_per_frame",
    "uniform_intermediate_frame_ratio",
    "q12_main_anchor_mib_per_frame",
    "rate_counted",
    "transmitted_index_payload_counted",
    "decision",
]

DECISION_FIELDS = ["item", "decision", "evidence"]


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


def target_middle_by_gap(rows):
    out = {}
    for row in rows:
        setting = row["paper_setting"]
        if setting == "Middle-4 frames":
            out[4] = float(row["local_corrected_middle_psnr"])
        elif setting == "8-frame interval":
            out[8] = float(row["local_corrected_middle_psnr"])
    missing = {4, 8} - set(out)
    if missing:
        raise KeyError(f"missing Stage142 target gaps: {sorted(missing)}")
    return out


def select_stage96_rows(stage96_summary, targets, near_target_tolerance):
    out = []
    for row in stage96_summary["rows"]:
        if row["base_method"] != "stage65_adapter":
            continue
        if row["codec"] != "q12":
            continue
        gap = int(row["reference_gap"])
        if gap not in targets:
            continue
        target = targets[gap]
        base_psnr = float(row["mean_base_psnr"])
        side_psnr = float(row["mean_entropy_psnr"])
        task_count = int(row["task_count"])
        positive_count = int(row["positive_delta_count"])
        sideinfo_gap = side_psnr - target
        decision = "promote_for_stage148_revalidation" if sideinfo_gap >= -float(near_target_tolerance) else "insufficient_quality"
        out.append({
            "reference_gap": gap,
            "target_middle_psnr": target,
            "base_method": row["base_method"],
            "codec": row["codec"],
            "setting_label": "q6_top10_entropy_index_value",
            "keep_fraction": 0.1,
            "side_bits": 6,
            "task_count": task_count,
            "mean_base_psnr": base_psnr,
            "mean_sideinfo_psnr": side_psnr,
            "mean_delta_psnr_vs_base": float(row["mean_delta_psnr_vs_base"]),
            "base_gap_to_target": base_psnr - target,
            "sideinfo_gap_to_target": sideinfo_gap,
            "positive_delta_count": positive_count,
            "positive_delta_fraction": positive_count / max(task_count, 1),
            "entropy_sideinfo_mib_per_intermediate_frame": float(row["entropy_sideinfo_mib_per_intermediate_frame"]),
            "entropy_sideinfo_payload_bytes_per_intermediate_frame": float(row["entropy_sideinfo_mib_per_intermediate_frame"]) * 1024.0 * 1024.0,
            "entropy_direct_total_mib_per_frame": float(row["entropy_direct_total_mib_per_frame"]),
            "entropy_amortized_total_mib_per_frame": float(row["entropy_amortized_total_mib_per_frame"]),
            "uniform_intermediate_frame_ratio": float(row["uniform_intermediate_frame_ratio"]),
            "q12_main_anchor_mib_per_frame": float(row["q12_main_anchor_mib_per_frame"]),
            "rate_counted": 1,
            "transmitted_index_payload_counted": 1,
            "decision": decision,
        })
    if sorted(row["reference_gap"] for row in out) != [4, 8]:
        raise RuntimeError("Stage147 expects q12 stage65_adapter Stage96 rows for gap4 and gap8")
    return out


def build_policy(rows, args):
    return {
        "stage": 147,
        "policy_name": "rate_counted_entropy_index_value_residual_sideinfo_fallback_v1",
        "policy_type": "fallback_sideinfo_policy_package",
        "status": "candidate_for_stage148_full_rendered_revalidation",
        "setting_label": "q6_top10_entropy_index_value",
        "base_method": "stage65_adapter",
        "codec": "q12",
        "keep_fraction": 0.1,
        "side_bits": 6,
        "index_selection": {
            "rule_name": "encoder_residual_energy_top10_with_transmitted_indices_v1",
            "encoder_side": True,
            "decoder_reproducible_without_payload": False,
            "selected_indices_transmitted": True,
            "selected_index_payload_counted": True,
            "selection_rule": "select top 10 percent Gaussian residual-energy indices at encoder, sort/delta-code, and zlib-compress the index stream",
        },
        "sideinfo_codec": {
            "codec_family": "entropy-coded index+value residual side-info",
            "residual_values_quantized": True,
            "residual_side_bits": 6,
            "zlib_compressed_components": ["metadata", "index_deltas", "residual_values"],
            "all_payload_bytes_counted": True,
        },
        "encoder_contract": {
            "allowed_inputs": [
                "left_anchor",
                "right_anchor",
                "normalized_time",
                "intermediate frame or encoder-side target representation used only to form transmitted residual side-info",
            ],
            "steps": [
                "build base adapter prediction",
                "compute residual values for selected Gaussian attributes",
                "encode selected indices and quantized residual values into a payload",
                "include payload bytes in total rate",
            ],
        },
        "decoder_contract": {
            "allowed_inputs": [
                "left_anchor",
                "right_anchor",
                "normalized_time",
                "encoded residual side-info payload",
            ],
            "forbidden_inputs": [
                "target_dense_anchor",
                "target_rgb",
                "unencoded target residual tensor",
                "oracle labels not represented in the payload",
            ],
            "steps": [
                "build base adapter prediction",
                "decode selected indices and quantized residual values from payload",
                "apply decoded residual values to base anchor attributes",
                "render corrected anchor",
            ],
        },
        "rate_accounting": {
            "all_sideinfo_bytes_counted": True,
            "direct_total_rate": "q12_main_anchor_mib_per_frame + entropy_sideinfo_mib_per_intermediate_frame",
            "amortized_total_rate": "q12_main_anchor_mib_per_frame + entropy_sideinfo_mib_per_intermediate_frame * uniform_intermediate_frame_ratio",
            "rate_rows": rows,
        },
        "scope_limitations": [
            "Stage96 evidence is a 60-task broader side-info eval slice, not the full Stage75 paper-protocol video evaluation.",
            "This package promotes the fallback for Stage148 revalidation; it is not the final full-video RD claim.",
            "Payload transmission is required; without the payload this is not decoder-side feed-forward prediction.",
        ],
        "source_paths": {
            "stage96_summary": str(args.stage96_summary),
            "stage142_targets": str(args.stage142_targets),
            "stage144_package": str(args.stage144_package),
            "stage146_summary": str(args.stage146_summary),
        },
    }


def build_decisions(rows, stage144_package, stage146_summary, tolerance):
    worst_gap = min(float(row["sideinfo_gap_to_target"]) for row in rows)
    max_direct_rate = max(float(row["entropy_direct_total_mib_per_frame"]) for row in rows)
    min_positive_fraction = min(float(row["positive_delta_fraction"]) for row in rows)
    stage146_best_step = int(stage146_summary["best_step"])
    stage146_final_delta = float(stage146_summary["final_eval"]["model_psnr_avg"]) - float(stage146_summary["initial_eval"]["model_psnr_avg"])
    return [
        {
            "item": "feedforward_only_path",
            "decision": "not_enough_as_current_primary",
            "evidence": f"Stage146 best_step={stage146_best_step}; final mean PSNR delta vs init = {stage146_final_delta} dB",
        },
        {
            "item": "higher_qbit_path",
            "decision": "already_rejected",
            "evidence": stage144_package["first_phase_decision"],
        },
        {
            "item": "rate_counted_sideinfo_fallback",
            "decision": "promote_for_stage148_revalidation",
            "evidence": f"worst side-info gap to corrected target = {worst_gap} dB with tolerance {tolerance} dB",
        },
        {
            "item": "rate_accounting",
            "decision": "all_payload_bytes_counted",
            "evidence": f"max direct total rate among gap4/gap8 rows = {max_direct_rate} MiB/frame; index payload is transmitted and counted",
        },
        {
            "item": "task_positivity",
            "decision": "positive_on_all_stage96_gap4_gap8_adapter_tasks" if min_positive_fraction >= 1.0 else "mixed_task_deltas",
            "evidence": f"min positive delta fraction = {min_positive_fraction}",
        },
    ]


def write_report(rows, decisions, policy, package, path):
    lines = [
        "# Stage147 Rate-Counted Side-Info Fallback Package",
        "",
        "## Summary",
        "",
        "| gap | target | base PSNR | side PSNR | side-target | delta base | side MiB/frame | direct rate | amortized rate | positives | decision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['reference_gap']} | {float(row['target_middle_psnr']):.6f} | {float(row['mean_base_psnr']):.6f} | {float(row['mean_sideinfo_psnr']):.6f} | {float(row['sideinfo_gap_to_target']):.6f} | {float(row['mean_delta_psnr_vs_base']):.6f} | {float(row['entropy_sideinfo_mib_per_intermediate_frame']):.6f} | {float(row['entropy_direct_total_mib_per_frame']):.6f} | {float(row['entropy_amortized_total_mib_per_frame']):.6f} | {row['positive_delta_count']}/{row['task_count']} | {row['decision']} |"
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
        "## Contract",
        "",
        f"- policy: `{policy['policy_name']}`",
        "- side-info is a transmitted payload and all payload bytes are counted in total rate.",
        "- Encoder-side target/intermediate information may be used only to form the payload.",
        "- Decoder forbidden inputs: target dense anchor, target RGB, unencoded target residual tensor, oracle labels not represented in the payload.",
        "- This is not the previously rejected uncounted teacher side-info framing.",
        "",
        "## Outputs",
        "",
        f"- rows CSV: `{package['rows_csv']}`",
        f"- decisions CSV: `{package['decisions_csv']}`",
        f"- policy JSON: `{package['policy_json']}`",
        f"- summary JSON: `{package['summary_json']}`",
        f"- package JSON: `{package['package_json']}`",
        f"- report Markdown: `{package['report_md']}`",
        "",
        "## Limitation",
        "",
        "Stage96 evidence is not the final full-video paper-protocol evaluation. Stage148 must revalidate this fallback before a final quality claim.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage96_summary", type=Path, default=DEFAULT_STAGE96_SUMMARY)
    parser.add_argument("--stage142_targets", type=Path, default=DEFAULT_STAGE142_TARGETS)
    parser.add_argument("--stage144_package", type=Path, default=DEFAULT_STAGE144_PACKAGE)
    parser.add_argument("--stage146_summary", type=Path, default=DEFAULT_STAGE146_SUMMARY)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--near_target_tolerance", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    targets = target_middle_by_gap(read_csv(args.stage142_targets))
    stage96_summary = read_json(args.stage96_summary)
    stage144_package = read_json(args.stage144_package)
    stage146_summary = read_json(args.stage146_summary)
    rows = select_stage96_rows(stage96_summary, targets, args.near_target_tolerance)
    decisions = build_decisions(rows, stage144_package, stage146_summary, args.near_target_tolerance)
    policy = build_policy(rows, args)

    rows_csv = args.summary_root / "stage147_rate_counted_sideinfo_fallback_rows.csv"
    decisions_csv = args.summary_root / "stage147_rate_counted_sideinfo_fallback_decisions.csv"
    policy_json = args.summary_root / "stage147_rate_counted_sideinfo_fallback_policy.json"
    summary_json = args.summary_root / "stage147_rate_counted_sideinfo_fallback_summary.json"
    package_json = args.summary_root / "stage147_rate_counted_sideinfo_fallback_package.json"
    report_md = args.summary_root / "stage147_rate_counted_sideinfo_fallback_report.md"

    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(decisions, decisions_csv, DECISION_FIELDS)
    policy_json.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    summary = {
        "stage": 147,
        "mode": "rate-counted side-info fallback package",
        "stage96_scope_note": stage96_summary["scope_note"],
        "near_target_tolerance": args.near_target_tolerance,
        "rows": rows,
        "decisions": decisions,
        "policy_name": policy["policy_name"],
        "recommendation": "Run Stage148 full rendered revalidation of q6/top10 entropy index+value residual side-info fallback.",
        "rows_csv": str(rows_csv),
        "decisions_csv": str(decisions_csv),
        "policy_json": str(policy_json),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package = {
        "stage": 147,
        "mode": summary["mode"],
        "policy_name": policy["policy_name"],
        "decision": "promote_rate_counted_sideinfo_fallback_for_stage148_revalidation",
        "rows_csv": str(rows_csv),
        "decisions_csv": str(decisions_csv),
        "policy_json": str(policy_json),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "all_sideinfo_bytes_counted": True,
        "decoder_target_dense_anchor_input": False,
        "teacher_sideinfo_uncounted": False,
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(rows, decisions, policy, package, report_md)
    print(json.dumps({
        "package": str(package_json),
        "policy": policy["policy_name"],
        "decision": package["decision"],
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
