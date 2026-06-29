import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE148_SUMMARY = REPO_ROOT / "experiments/stage148_rate_counted_sideinfo_rendered_revalidation/stage148_rate_counted_sideinfo_rendered_revalidation_summary.json"
DEFAULT_STAGE142_TARGETS = REPO_ROOT / "experiments/stage142_middle_frame_protocol_alignment_audit/stage142_middle_frame_targets.csv"
DEFAULT_STAGE147_ROWS = REPO_ROOT / "experiments/stage147_rate_counted_sideinfo_fallback_package/stage147_rate_counted_sideinfo_fallback_rows.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage148_rate_counted_sideinfo_rendered_revalidation"


ROW_FIELDS = [
    "reference_gap",
    "target_middle_psnr",
    "base_method",
    "codec",
    "task_count",
    "keep_fraction",
    "side_bits",
    "mean_base_psnr",
    "mean_entropy_psnr",
    "mean_delta_psnr_vs_base",
    "base_gap_to_target",
    "entropy_gap_to_target",
    "positive_delta_count",
    "positive_delta_fraction",
    "max_decoded_max_abs_diff_vs_fixed",
    "mean_entropy_payload_bytes",
    "mean_entropy_mib_per_intermediate_frame",
    "q12_main_anchor_mib_per_frame",
    "uniform_intermediate_frame_ratio",
    "mean_direct_total_mib_per_frame",
    "mean_amortized_total_mib_per_frame",
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


def targets_by_gap(rows):
    out = {}
    for row in rows:
        if row["paper_setting"] == "Middle-4 frames":
            out[4] = float(row["local_corrected_middle_psnr"])
        elif row["paper_setting"] == "8-frame interval":
            out[8] = float(row["local_corrected_middle_psnr"])
    return out


def rate_reference_by_gap(rows):
    out = {}
    for row in rows:
        gap = int(row["reference_gap"])
        out[gap] = {
            "q12_main_anchor_mib_per_frame": float(row["q12_main_anchor_mib_per_frame"]),
            "uniform_intermediate_frame_ratio": float(row["uniform_intermediate_frame_ratio"]),
        }
    return out


def build_rows(stage148_summary, targets, rates, tolerance):
    out = []
    for row in stage148_summary["summary_rows"]:
        if row["base_method"] != "stage65_adapter":
            continue
        if row["codec"] != "q12":
            continue
        gap = int(row["reference_gap"])
        if gap not in targets:
            continue
        target = targets[gap]
        rate = rates[gap]
        side_mib = float(row["mean_entropy_mib_per_intermediate_frame"])
        direct_rate = rate["q12_main_anchor_mib_per_frame"] + side_mib
        amortized_rate = rate["q12_main_anchor_mib_per_frame"] + side_mib * rate["uniform_intermediate_frame_ratio"]
        base_psnr = float(row["mean_base_psnr"])
        entropy_psnr = float(row["mean_entropy_psnr"])
        entropy_gap = entropy_psnr - target
        task_count = int(row["task_count"])
        positive_count = int(row["positive_delta_count"])
        decode_diff = float(row["max_decoded_max_abs_diff_vs_fixed"])
        decision = "passes_sample_revalidation" if entropy_gap >= -float(tolerance) and decode_diff == 0.0 else "needs_followup"
        out.append({
            "reference_gap": gap,
            "target_middle_psnr": target,
            "base_method": row["base_method"],
            "codec": row["codec"],
            "task_count": task_count,
            "keep_fraction": float(row["keep_fraction"]),
            "side_bits": int(row["side_bits"]),
            "mean_base_psnr": base_psnr,
            "mean_entropy_psnr": entropy_psnr,
            "mean_delta_psnr_vs_base": float(row["mean_delta_psnr_vs_base"]),
            "base_gap_to_target": base_psnr - target,
            "entropy_gap_to_target": entropy_gap,
            "positive_delta_count": positive_count,
            "positive_delta_fraction": positive_count / max(task_count, 1),
            "max_decoded_max_abs_diff_vs_fixed": decode_diff,
            "mean_entropy_payload_bytes": float(row["mean_entropy_payload_bytes"]),
            "mean_entropy_mib_per_intermediate_frame": side_mib,
            "q12_main_anchor_mib_per_frame": rate["q12_main_anchor_mib_per_frame"],
            "uniform_intermediate_frame_ratio": rate["uniform_intermediate_frame_ratio"],
            "mean_direct_total_mib_per_frame": direct_rate,
            "mean_amortized_total_mib_per_frame": amortized_rate,
            "decision": decision,
        })
    if sorted(row["reference_gap"] for row in out) != [4, 8]:
        raise RuntimeError("Stage148 package expects stage65_adapter q12 gap4/gap8 summary rows")
    return out


def build_decisions(rows, stage148_summary, tolerance):
    worst_gap = min(float(row["entropy_gap_to_target"]) for row in rows)
    max_decode_diff = max(float(row["max_decoded_max_abs_diff_vs_fixed"]) for row in rows)
    min_positive_fraction = min(float(row["positive_delta_fraction"]) for row in rows)
    max_direct_rate = max(float(row["mean_direct_total_mib_per_frame"]) for row in rows)
    return [
        {
            "item": "entropy_decode",
            "decision": "matches_fixed_decode" if max_decode_diff == 0.0 else "decode_mismatch",
            "evidence": f"max decoded abs diff vs fixed = {max_decode_diff}",
        },
        {
            "item": "target_alignment",
            "decision": "passes_sample_revalidation" if worst_gap >= -float(tolerance) else "below_target_tolerance",
            "evidence": f"worst entropy gap to corrected target = {worst_gap} dB with tolerance {tolerance} dB",
        },
        {
            "item": "task_positivity",
            "decision": "positive_on_all_sampled_tasks" if min_positive_fraction >= 1.0 else "mixed_task_deltas",
            "evidence": f"min positive delta fraction = {min_positive_fraction}",
        },
        {
            "item": "rate_accounting",
            "decision": "all_sideinfo_bytes_counted",
            "evidence": f"max direct total rate = {max_direct_rate} MiB/frame; payload bytes from {stage148_summary['rows_csv']}",
        },
        {
            "item": "next_step",
            "decision": "run_full_all_row_or_full_video_rd" if worst_gap >= -float(tolerance) and max_decode_diff == 0.0 else "adjust_sideinfo_setting_or_model",
            "evidence": "Stage148 is sampled rendered revalidation, not full all-row eval.",
        },
    ]


def write_report(rows, decisions, package, path):
    lines = [
        "# Stage148 Rate-Counted Side-Info Rendered Revalidation Package",
        "",
        "## Summary",
        "",
        "| gap | target | base PSNR | entropy PSNR | entropy-target | delta base | payload bytes | side MiB | direct rate | amortized rate | decode diff | positives | decision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['reference_gap']} | {float(row['target_middle_psnr']):.6f} | {float(row['mean_base_psnr']):.6f} | {float(row['mean_entropy_psnr']):.6f} | {float(row['entropy_gap_to_target']):.6f} | {float(row['mean_delta_psnr_vs_base']):.6f} | {float(row['mean_entropy_payload_bytes']):.3f} | {float(row['mean_entropy_mib_per_intermediate_frame']):.6f} | {float(row['mean_direct_total_mib_per_frame']):.6f} | {float(row['mean_amortized_total_mib_per_frame']):.6f} | {float(row['max_decoded_max_abs_diff_vs_fixed']):.6f} | {row['positive_delta_count']}/{row['task_count']} | {row['decision']} |"
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
        "- This package is based on actual entropy payload encode/decode/render results.",
        "- Side-info payload bytes are counted in direct and amortized total rate.",
        "- Decoder inputs remain endpoint anchors, normalized time, and encoded payload only.",
        "- Decoder does not receive target dense anchors, target RGB, or unencoded residual tensors.",
        "",
        "## Outputs",
        "",
        f"- rows CSV: `{package['rows_csv']}`",
        f"- decisions CSV: `{package['decisions_csv']}`",
        f"- summary JSON: `{package['summary_json']}`",
        f"- package JSON: `{package['package_json']}`",
        f"- report Markdown: `{package['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage148_summary", type=Path, default=DEFAULT_STAGE148_SUMMARY)
    parser.add_argument("--stage142_targets", type=Path, default=DEFAULT_STAGE142_TARGETS)
    parser.add_argument("--stage147_rows", type=Path, default=DEFAULT_STAGE147_ROWS)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--near_target_tolerance", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage148_summary = read_json(args.stage148_summary)
    targets = targets_by_gap(read_csv(args.stage142_targets))
    rates = rate_reference_by_gap(read_csv(args.stage147_rows))
    rows = build_rows(stage148_summary, targets, rates, args.near_target_tolerance)
    decisions = build_decisions(rows, stage148_summary, args.near_target_tolerance)

    rows_csv = args.summary_root / "stage148_rate_counted_sideinfo_rendered_revalidation_package_rows.csv"
    decisions_csv = args.summary_root / "stage148_rate_counted_sideinfo_rendered_revalidation_package_decisions.csv"
    summary_json = args.summary_root / "stage148_rate_counted_sideinfo_rendered_revalidation_package_summary.json"
    package_json = args.summary_root / "stage148_rate_counted_sideinfo_rendered_revalidation_package.json"
    report_md = args.summary_root / "stage148_rate_counted_sideinfo_rendered_revalidation_package_report.md"

    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(decisions, decisions_csv, DECISION_FIELDS)
    summary = {
        "stage": 148,
        "mode": "rate-counted side-info rendered revalidation package",
        "render_summary": str(args.stage148_summary),
        "near_target_tolerance": args.near_target_tolerance,
        "rows": rows,
        "decisions": decisions,
        "rows_csv": str(rows_csv),
        "decisions_csv": str(decisions_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package = {
        "stage": 148,
        "mode": summary["mode"],
        "decision": "sample_revalidation_passed" if all(row["decision"] == "passes_sample_revalidation" for row in rows) else "needs_followup",
        "rows_csv": str(rows_csv),
        "decisions_csv": str(decisions_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "all_sideinfo_bytes_counted": True,
        "actual_entropy_encode_decode_render": True,
        "decoder_target_dense_anchor_input": False,
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(rows, decisions, package, report_md)
    print(json.dumps({"package": str(package_json), "decision": package["decision"], "rows": rows}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
