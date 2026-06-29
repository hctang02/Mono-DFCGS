import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE75_ROOT = REPO_ROOT / "experiments/stage75_corrected_streamsplat_paper_protocol_package"
DEFAULT_STAGE77_ROOT = REPO_ROOT / "experiments/stage77_qbit_full_video_anchor_only_rd_sweep"
DEFAULT_STAGE78_ROOT = REPO_ROOT / "experiments/stage78_integrated_davis_rd_package"
DEFAULT_STAGE141_PACKAGE = REPO_ROOT / "experiments/stage141_deployable_full_pipeline_manifest/stage141_deployable_full_pipeline_manifest_package.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage142_middle_frame_protocol_alignment_audit"

TARGET_FIELDS = [
    "target_name",
    "reference_source",
    "reference_scope",
    "local_gap",
    "paper_setting",
    "paper_psnr",
    "local_corrected_middle_psnr",
    "local_corrected_all_psnr",
    "local_corrected_given_psnr",
]

COMPARISON_FIELDS = [
    "comparison_name",
    "reference_scope",
    "ours_scope",
    "reference_gap",
    "ours_gap",
    "ours_codec",
    "ours_method",
    "ours_rate_mib_per_frame",
    "ours_middle_psnr",
    "reference_middle_psnr",
    "middle_gap_to_reference",
    "scope_matched",
    "metric_matched",
    "rate_matched",
    "use_as_final_claim",
]

FINDING_FIELDS = ["severity", "finding", "evidence", "required_action"]


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


def f(value):
    return float(value)


def build_target_rows(stage75_rows):
    rows = []
    for row in stage75_rows:
        rows.append({
            "target_name": "paper_level_middle_frame_target",
            "reference_source": "Stage75 corrected StreamSplat paper-protocol package",
            "reference_scope": "DAVIS val full 30 sequences, sliding fixed intervals, official_256_float",
            "local_gap": row["local_gap"],
            "paper_setting": row["paper_setting"],
            "paper_psnr": row["paper_psnr"],
            "local_corrected_middle_psnr": row["middle_psnr"],
            "local_corrected_all_psnr": row["all_psnr"],
            "local_corrected_given_psnr": row["given_psnr"],
        })
    return rows


def build_comparison_rows(stage78_reference_rows):
    rows = []
    for row in stage78_reference_rows:
        rows.append({
            "comparison_name": "Stage78_anchor_only_vs_Stage75_reference",
            "reference_scope": "DAVIS val full 30 sequences, sliding fixed intervals, official_256_float",
            "ours_scope": "Stage77 scoped 4 DAVIS val sequences: bmx-trees, car-shadow, goat, soapbox",
            "reference_gap": row["reference_local_gap"],
            "ours_gap": row["anchor_gap"],
            "ours_codec": row["anchor_codec"],
            "ours_method": row["anchor_method"],
            "ours_rate_mib_per_frame": row["anchor_rate_mib_per_frame"],
            "ours_middle_psnr": row["anchor_middle_psnr"],
            "reference_middle_psnr": row["reference_middle_psnr"],
            "middle_gap_to_reference": row["middle_psnr_gap_to_reference"],
            "scope_matched": 0,
            "metric_matched": 1,
            "rate_matched": "n/a_reference_has_no_anchor_codec_rate",
            "use_as_final_claim": 0,
        })
    return rows


def best_q12_adapter_gap(comparison_rows, ours_gap):
    candidates = [
        row for row in comparison_rows
        if row["ours_codec"] == "q12" and row["ours_method"] == "adapter" and int(float(row["ours_gap"])) == int(ours_gap)
    ]
    if not candidates:
        return None
    return candidates[0]


def build_findings(stage77_summary, comparison_rows, stage141_package):
    gap4 = best_q12_adapter_gap(comparison_rows, 4)
    gap8 = best_q12_adapter_gap(comparison_rows, 8)
    q12_method_rows = [row for row in stage77_summary if row["codec"] == "q12" and row["method"] == "adapter"]
    findings = [
        {
            "severity": "critical",
            "finding": "Stage78 reference comparison is diagnostic, not a final apples-to-apples paper-protocol claim.",
            "evidence": "Stage75 uses full DAVIS val 30 sequences; Stage77/78 anchor-only rows use 4 scoped DAVIS val sequences.",
            "required_action": "Rerun our anchor/adapter pipeline on the same full DAVIS val paper-style protocol before final claims.",
        },
        {
            "severity": "critical",
            "finding": "Current middle-frame quality is far below the corrected StreamSplat/paper-level target.",
            "evidence": f"q12 adapter gap4 middle gap={gap4['middle_gap_to_reference'] if gap4 else 'missing'}; q12 adapter gap8 middle gap={gap8['middle_gap_to_reference'] if gap8 else 'missing'}.",
            "required_action": "Run Stage143/144 to separate renderer/data/quantization/model causes, then train or add rate-counted side-info until middle PSNR recovers.",
        },
        {
            "severity": "high",
            "finding": "Stage141 final manifest is decoder-safe but is not a paper-level quality solution.",
            "evidence": f"Stage141 primary PSNR={stage141_package['primary_psnr']} at rate={stage141_package['primary_rate']}; this is a task-slice deployable residual metric, not full paper-protocol middle PSNR.",
            "required_action": "Keep Stage141 as a decoder-side accounting checkpoint only; do not treat it as final quality result.",
        },
        {
            "severity": "high",
            "finding": "q-bit increase from q8 to q12 has only small middle-frame gains, so quantization alone is unlikely to explain the full gap.",
            "evidence": "; ".join(
                f"gap{row['frame_gap']} q12 adapter middle={row['mean_middle_psnr']}" for row in q12_method_rows
            ),
            "required_action": "Stage144 must include uncompressed or high-rate anchors to prove the remaining ceiling; if ceiling stays low, prioritize dynamic model training.",
        },
    ]
    return findings


def write_report(target_rows, comparison_rows, findings, package, path):
    lines = [
        "# Stage142 Middle-Frame Protocol Alignment Audit",
        "",
        "## Target Protocol",
        "",
        "| setting | local gap | paper PSNR | corrected local middle | corrected all | corrected given |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in target_rows:
        lines.append(
            f"| {row['paper_setting']} | {row['local_gap']} | {row['paper_psnr']} | {row['local_corrected_middle_psnr']} | {row['local_corrected_all_psnr']} | {row['local_corrected_given_psnr']} |"
        )
    lines.extend([
        "",
        "## Current Diagnostic Gap",
        "",
        "| ours | gap | method | rate | ours middle | reference middle | gap to reference | final claim? |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in comparison_rows:
        if row["ours_codec"] != "q12" or row["ours_method"] != "adapter":
            continue
        lines.append(
            f"| {row['ours_codec']} | {row['ours_gap']} | {row['ours_method']} | {row['ours_rate_mib_per_frame']} | {row['ours_middle_psnr']} | {row['reference_middle_psnr']} | {row['middle_gap_to_reference']} | {row['use_as_final_claim']} |"
        )
    lines.extend([
        "",
        "## Findings",
        "",
        "| severity | finding | required action |",
        "|---|---|---|",
    ])
    for row in findings:
        lines.append(f"| {row['severity']} | {row['finding']} | {row['required_action']} |")
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- target CSV: `{package['target_csv']}`",
        f"- comparison CSV: `{package['comparison_csv']}`",
        f"- findings CSV: `{package['findings_csv']}`",
        f"- summary JSON: `{package['summary_json']}`",
        f"- package JSON: `{package['package_json']}`",
        f"- report Markdown: `{package['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage75_root", type=Path, default=DEFAULT_STAGE75_ROOT)
    parser.add_argument("--stage77_root", type=Path, default=DEFAULT_STAGE77_ROOT)
    parser.add_argument("--stage78_root", type=Path, default=DEFAULT_STAGE78_ROOT)
    parser.add_argument("--stage141_package", type=Path, default=DEFAULT_STAGE141_PACKAGE)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage75_rows = read_csv(args.stage75_root / "stage75_corrected_streamsplat_paper_protocol_summary.csv")
    stage77_summary = read_csv(args.stage77_root / "stage77_qbit_full_video_anchor_only_rd_summary.csv")
    stage78_reference_rows = read_csv(args.stage78_root / "stage78_reference_gap_table.csv")
    stage141_package = read_json(args.stage141_package)
    target_rows = build_target_rows(stage75_rows)
    comparison_rows = build_comparison_rows(stage78_reference_rows)
    findings = build_findings(stage77_summary, comparison_rows, stage141_package)
    target_csv = args.summary_root / "stage142_middle_frame_targets.csv"
    comparison_csv = args.summary_root / "stage142_protocol_comparison_rows.csv"
    findings_csv = args.summary_root / "stage142_protocol_alignment_findings.csv"
    summary_json = args.summary_root / "stage142_middle_frame_protocol_alignment_summary.json"
    package_json = args.summary_root / "stage142_middle_frame_protocol_alignment_package.json"
    report_md = args.summary_root / "stage142_middle_frame_protocol_alignment_report.md"
    write_csv(target_rows, target_csv, TARGET_FIELDS)
    write_csv(comparison_rows, comparison_csv, COMPARISON_FIELDS)
    write_csv(findings, findings_csv, FINDING_FIELDS)
    q12_gap4 = best_q12_adapter_gap(comparison_rows, 4)
    q12_gap8 = best_q12_adapter_gap(comparison_rows, 8)
    summary = {
        "stage": 142,
        "mode": "middle-frame protocol alignment audit",
        "stage75_root": str(args.stage75_root),
        "stage77_root": str(args.stage77_root),
        "stage78_root": str(args.stage78_root),
        "stage141_package": str(args.stage141_package),
        "target_csv": str(target_csv),
        "comparison_csv": str(comparison_csv),
        "findings_csv": str(findings_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "target_rows": target_rows,
        "q12_adapter_gap4_diagnostic": q12_gap4,
        "q12_adapter_gap8_diagnostic": q12_gap8,
        "scope_matched_for_current_stage78_gap": 0,
        "use_stage78_as_final_claim": 0,
        "next_required_stages": [
            "Stage143: PSNR collapse decomposition on matched diagnostics",
            "Stage144: high-rate/uncompressed anchor upper-bound evaluation",
            "Stage145+: large-scale render-loss adapter training if model ceiling is the bottleneck",
        ],
    }
    package = {
        "stage": 142,
        "mode": "middle-frame protocol alignment audit",
        "target_csv": str(target_csv),
        "comparison_csv": str(comparison_csv),
        "findings_csv": str(findings_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "critical_findings": [row for row in findings if row["severity"] == "critical"],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(target_rows, comparison_rows, findings, package, report_md)
    print(json.dumps({
        "package": str(package_json),
        "q12_adapter_gap4_middle_gap": q12_gap4["middle_gap_to_reference"] if q12_gap4 else None,
        "q12_adapter_gap8_middle_gap": q12_gap8["middle_gap_to_reference"] if q12_gap8 else None,
        "use_stage78_as_final_claim": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
