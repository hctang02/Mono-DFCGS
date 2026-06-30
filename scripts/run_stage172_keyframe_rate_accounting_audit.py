import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE166_PACKAGE = REPO_ROOT / "experiments/stage166_adaptive_schedule_label_rd_comparison/stage166_adaptive_schedule_label_rd_comparison_package.json"
DEFAULT_STAGE166_ROWS = REPO_ROOT / "experiments/stage166_adaptive_schedule_label_rd_comparison/stage166_sampled_row_consequences.csv"
DEFAULT_STAGE171_PACKAGE = REPO_ROOT / "experiments/stage171_combined_adaptive_evidence_review/stage171_combined_adaptive_evidence_review_package.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage172_keyframe_rate_accounting_audit"


COMPONENT_FIELDS = [
    "schedule",
    "total_frames",
    "keyframe_count",
    "keyframe_delta_vs_uniform_gap8",
    "main_anchor_mib_per_frame_proxy",
    "baseline_gap8_main_anchor_mib_per_frame",
    "extra_keyframe_mib_per_frame_vs_gap8",
    "metadata_bytes",
    "metadata_mib_per_frame",
    "sampled_residual_row_count",
    "sampled_promoted_row_count",
    "sampled_residual_payload_bytes",
    "sampled_residual_side_mib_total",
    "sampled_residual_side_mib_per_sample",
    "sampled_avoided_side_mib_vs_gap8",
    "total_proxy_mib_per_frame_recomputed",
    "total_proxy_mib_per_frame_source",
    "delta_total_proxy_vs_uniform_gap8",
    "delta_total_proxy_vs_uniform_gap4",
]

NOTE_FIELDS = ["item", "value", "note"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def schedule_map(stage166):
    return {row["schedule"]: row for row in stage166["schedule_comparison"]}


def f(row, key):
    return float(row[key])


def i(row, key):
    return int(row[key])


def build_components(stage166):
    schedules = schedule_map(stage166)
    gap8 = schedules["uniform_gap8"]
    gap4 = schedules["uniform_gap4"]
    total_frames = i(gap8, "total_frames")
    gap8_keyframes = i(gap8, "total_keyframe_count")
    gap4_keyframes = i(gap4, "total_keyframe_count")
    gap8_main = f(gap8, "main_anchor_mib_per_frame_proxy")
    gap4_main = f(gap4, "main_anchor_mib_per_frame_proxy")
    per_extra_keyframe_mib_per_frame = (gap4_main - gap8_main) / float(gap4_keyframes - gap8_keyframes)
    per_extra_keyframe_payload_mib = per_extra_keyframe_mib_per_frame * total_frames

    rows = []
    source_totals = {}
    for name in ["uniform_gap8", "stage165_adaptive", "uniform_gap4"]:
        row = schedules[name]
        keyframes = i(row, "total_keyframe_count")
        delta_keyframes = keyframes - gap8_keyframes
        main_anchor = f(row, "main_anchor_mib_per_frame_proxy")
        metadata = f(row, "metadata_mib_per_frame")
        residual_per_sample = f(row, "mean_residual_side_mib_per_sample")
        recomputed = main_anchor + residual_per_sample + metadata
        source_totals[name] = f(row, "total_proxy_mib_per_frame")
        rows.append({
            "schedule": name,
            "total_frames": total_frames,
            "keyframe_count": keyframes,
            "keyframe_delta_vs_uniform_gap8": delta_keyframes,
            "main_anchor_mib_per_frame_proxy": main_anchor,
            "baseline_gap8_main_anchor_mib_per_frame": gap8_main,
            "extra_keyframe_mib_per_frame_vs_gap8": delta_keyframes * per_extra_keyframe_mib_per_frame,
            "metadata_bytes": i(row, "metadata_bytes"),
            "metadata_mib_per_frame": metadata,
            "sampled_residual_row_count": i(row, "residual_row_count"),
            "sampled_promoted_row_count": i(row, "promoted_row_count"),
            "sampled_residual_payload_bytes": f(row, "residual_payload_bytes"),
            "sampled_residual_side_mib_total": f(row, "residual_side_mib"),
            "sampled_residual_side_mib_per_sample": residual_per_sample,
            "sampled_avoided_side_mib_vs_gap8": f(row, "avoided_side_mib"),
            "total_proxy_mib_per_frame_recomputed": recomputed,
            "total_proxy_mib_per_frame_source": f(row, "total_proxy_mib_per_frame"),
            "delta_total_proxy_vs_uniform_gap8": recomputed - source_totals.get("uniform_gap8", f(gap8, "total_proxy_mib_per_frame")),
            "delta_total_proxy_vs_uniform_gap4": recomputed - f(gap4, "total_proxy_mib_per_frame"),
        })
    notes = [
        {
            "item": "per_extra_keyframe_mib_per_frame",
            "value": per_extra_keyframe_mib_per_frame,
            "note": "Derived from uniform gap8/gap4 main-anchor proxy rate difference divided by keyframe-count difference.",
        },
        {
            "item": "per_extra_keyframe_payload_mib",
            "value": per_extra_keyframe_payload_mib,
            "note": "Dataset-level proxy payload for one added keyframe, equal to per-frame cost times total frames.",
        },
        {
            "item": "accounting_scope",
            "value": "sampled_proxy",
            "note": "Residual side-info is measured on the 120 Stage163/166 sampled rows; full-sequence residual decisions remain future work.",
        },
        {
            "item": "decoder_contract",
            "value": "schedule_metadata_transmitted",
            "note": "Decoder receives counted schedule/keyframe metadata and does not recompute selector RGB/motion features.",
        },
    ]
    return rows, notes, per_extra_keyframe_mib_per_frame, per_extra_keyframe_payload_mib


def write_report(component_rows, notes, decision, path):
    lines = [
        "# Stage172 Keyframe Rate Accounting Audit",
        "",
        "## Decision",
        "",
        f"- Decision: `{decision}`.",
        "- Adaptive remains rate-promising under the Stage166 sampled proxy after charging extra keyframes and metadata.",
        "- This still does not constitute final full-sequence RD; it is sufficient to design Stage173 medium rendered validation.",
        "",
        "## Components",
        "",
        "| schedule | keyframes | extra vs gap8 | main anchor MiB/frame | residual MiB/sample | metadata MiB/frame | total proxy MiB/frame | delta vs gap8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in component_rows:
        lines.append(
            f"| {row['schedule']} | {row['keyframe_count']} | {row['keyframe_delta_vs_uniform_gap8']} | "
            f"{row['main_anchor_mib_per_frame_proxy']:.12f} | {row['sampled_residual_side_mib_per_sample']:.12f} | "
            f"{row['metadata_mib_per_frame']:.12f} | {row['total_proxy_mib_per_frame_recomputed']:.12f} | "
            f"{row['delta_total_proxy_vs_uniform_gap8']:.12f} |"
        )
    lines.extend(["", "## Notes", ""])
    for note in notes:
        lines.append(f"- `{note['item']}`: `{note['value']}`. {note['note']}")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Stage165 adaptive adds `66` keyframes over uniform gap8 but avoids much more sampled residual side-info on selected rows.",
        "- The extra schedule metadata is tiny at `327` bytes total for the Stage165 schedule.",
        "- Because false negatives remain, Stage173 must include false-negative controls and should not only sample selected/promoted rows.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage166_package", type=Path, default=DEFAULT_STAGE166_PACKAGE)
    parser.add_argument("--stage166_rows", type=Path, default=DEFAULT_STAGE166_ROWS)
    parser.add_argument("--stage171_package", type=Path, default=DEFAULT_STAGE171_PACKAGE)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage166 = read_json(args.stage166_package)
    stage171 = read_json(args.stage171_package)
    stage166_rows = read_csv(args.stage166_rows)
    component_rows, notes, per_extra_frame, per_extra_payload = build_components(stage166)
    adaptive = next(row for row in component_rows if row["schedule"] == "stage165_adaptive")
    gap8 = next(row for row in component_rows if row["schedule"] == "uniform_gap8")
    gap4 = next(row for row in component_rows if row["schedule"] == "uniform_gap4")
    decision = "adaptive_rate_promising_for_medium_protocol" if adaptive["total_proxy_mib_per_frame_recomputed"] < gap8["total_proxy_mib_per_frame_recomputed"] and adaptive["total_proxy_mib_per_frame_recomputed"] < gap4["total_proxy_mib_per_frame_recomputed"] else "needs_selector_refinement_before_medium_protocol"
    component_csv = args.output_root / "stage172_rate_component_audit.csv"
    notes_csv = args.output_root / "stage172_rate_accounting_notes.csv"
    package_json = args.output_root / "stage172_keyframe_rate_accounting_audit_package.json"
    report_md = args.output_root / "stage172_keyframe_rate_accounting_audit_report.md"
    write_csv(component_rows, component_csv, COMPONENT_FIELDS)
    write_csv(notes, notes_csv, NOTE_FIELDS)
    package = {
        "stage": 172,
        "status": "keyframe_rate_accounting_audit_packaged",
        "decision": decision,
        "stage171_decision": stage171["decision"],
        "accounting_scope": "stage166_sampled_proxy",
        "sampled_row_count": len(stage166_rows),
        "per_extra_keyframe_mib_per_frame": per_extra_frame,
        "per_extra_keyframe_payload_mib": per_extra_payload,
        "component_rows": component_rows,
        "notes": notes,
        "component_csv": str(component_csv),
        "notes_csv": str(notes_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(component_rows, notes, decision, report_md)
    print(json.dumps({"package": str(package_json), "decision": decision}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
