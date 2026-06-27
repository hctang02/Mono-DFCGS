import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE74_ROOT = REPO_ROOT / "experiments/stage74_stage72_vs_actual_gap_diagnosis_full_val_sliding_per_frame"
DEFAULT_STAGE72_ROWS = REPO_ROOT / "experiments/stage72_original_davis_baseline/stage72_original_davis_baseline_rows.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage75_corrected_streamsplat_paper_protocol_package"

PAPER_SETTINGS = {
    5: {"paper_setting": "Middle-4 frames", "paper_psnr": 23.66},
    8: {"paper_setting": "8-frame interval", "paper_psnr": 22.10},
}

SUMMARY_FIELDS = [
    "paper_setting",
    "local_gap",
    "sequence_scope",
    "window_mode",
    "depth_norm",
    "metric_space",
    "pair_count",
    "all_count",
    "middle_count",
    "given_count",
    "all_psnr",
    "middle_psnr",
    "given_psnr",
    "paper_psnr",
    "local_minus_paper_middle_psnr",
]

SEQUENCE_FIELDS = [
    "sample",
    "paper_setting",
    "local_gap",
    "pair_count",
    "all_count",
    "middle_count",
    "given_count",
    "all_psnr",
    "middle_psnr",
    "given_psnr",
]

COMPARE_FIELDS = [
    "comparison",
    "stage72_gap",
    "stage72_scope",
    "stage72_metric_space",
    "stage72_all_psnr",
    "stage72_middle_psnr",
    "stage72_given_psnr",
    "stage75_gap",
    "stage75_scope",
    "stage75_metric_space",
    "stage75_all_psnr",
    "stage75_middle_psnr",
    "stage75_given_psnr",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def to_int(value):
    return int(float(value))


def to_float(value):
    return float(value) if value not in (None, "") else None


def build_summary(stage74_aggregate_rows):
    out = []
    for row in stage74_aggregate_rows:
        if row["sample"] != "MEAN_WEIGHTED" or row["metric_space"] != "official_256_float":
            continue
        gap = to_int(row["gap"])
        if gap not in PAPER_SETTINGS:
            continue
        paper = PAPER_SETTINGS[gap]
        middle_psnr = to_float(row["middle_psnr"])
        out.append({
            "paper_setting": paper["paper_setting"],
            "local_gap": gap,
            "sequence_scope": "DAVIS val full 30 sequences",
            "window_mode": row["mode"],
            "depth_norm": row["depth_norm"],
            "metric_space": row["metric_space"],
            "pair_count": to_int(row["pair_count"]),
            "all_count": to_int(row["all_count"]),
            "middle_count": to_int(row["middle_count"]),
            "given_count": to_int(row["given_count"]),
            "all_psnr": to_float(row["all_psnr"]),
            "middle_psnr": middle_psnr,
            "given_psnr": to_float(row["given_psnr"]),
            "paper_psnr": paper["paper_psnr"],
            "local_minus_paper_middle_psnr": middle_psnr - paper["paper_psnr"],
        })
    return out


def build_sequence_rows(stage74_rows):
    out = []
    for row in stage74_rows:
        if row["metric_space"] != "official_256_float":
            continue
        gap = to_int(row["gap"])
        if gap not in PAPER_SETTINGS:
            continue
        out.append({
            "sample": row["sample"],
            "paper_setting": PAPER_SETTINGS[gap]["paper_setting"],
            "local_gap": gap,
            "pair_count": to_int(row["pair_count"]),
            "all_count": to_int(row["all_count"]),
            "middle_count": to_int(row["middle_count"]),
            "given_count": to_int(row["given_count"]),
            "all_psnr": to_float(row["all_psnr"]),
            "middle_psnr": to_float(row["middle_psnr"]),
            "given_psnr": to_float(row["given_psnr"]),
        })
    return out


def build_stage72_comparison(stage72_rows, stage75_summary):
    stage72_by_gap = {to_int(row["frame_gap"]): row for row in stage72_rows if row["sample"] == "DAVIS/val/bmx-trees"}
    # Stage72 CSV has per-sequence rows; aggregate from its own gap summary is already in the report,
    # so use known mean rows from all Stage72 rows grouped by gap.
    grouped = {}
    for row in stage72_rows:
        gap = to_int(row["frame_gap"])
        grouped.setdefault(gap, []).append(row)
    stage72_mean = {}
    for gap, rows in grouped.items():
        stage72_mean[gap] = {
            "all": sum(to_float(row["all_psnr_avg"]) for row in rows) / len(rows),
            "middle": sum(to_float(row["middle_psnr_avg"]) for row in rows) / len(rows),
            "given": sum(to_float(row["given_psnr_avg"]) for row in rows) / len(rows),
        }
    stage75_by_gap = {to_int(row["local_gap"]): row for row in stage75_summary}
    pairs = [(4, 5, "Stage72 gap4 scoped vs corrected Middle-4"), (8, 8, "Stage72 gap8 scoped vs corrected 8-frame")]
    out = []
    for stage72_gap, stage75_gap, label in pairs:
        if stage72_gap not in stage72_mean or stage75_gap not in stage75_by_gap:
            continue
        local = stage72_mean[stage72_gap]
        corrected = stage75_by_gap[stage75_gap]
        out.append({
            "comparison": label,
            "stage72_gap": stage72_gap,
            "stage72_scope": "4 DAVIS val sequences, disjoint sparse-keyframe, 512x288 uint8",
            "stage72_metric_space": "stage72_512_uint8",
            "stage72_all_psnr": local["all"],
            "stage72_middle_psnr": local["middle"],
            "stage72_given_psnr": local["given"],
            "stage75_gap": stage75_gap,
            "stage75_scope": "full DAVIS val, sliding fixed windows, 256x256 float",
            "stage75_metric_space": "official_256_float",
            "stage75_all_psnr": corrected["all_psnr"],
            "stage75_middle_psnr": corrected["middle_psnr"],
            "stage75_given_psnr": corrected["given_psnr"],
        })
    return out


def write_report(summary_rows, compare_rows, path):
    lines = [
        "# Stage75 Corrected StreamSplat Paper-Protocol DAVIS Package",
        "",
        "## Scope",
        "",
        "- Full DAVIS val split: 30 sequences.",
        "- Sliding fixed intervals.",
        "- Paper-style `256x256` float metric space.",
        "- Main quality metric: middle/non-input PSNR.",
        "",
        "## Corrected Baseline",
        "",
        "| paper setting | local gap | pair count | all PSNR | middle PSNR | given PSNR | paper PSNR | local - paper |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['paper_setting']} | {row['local_gap']} | {row['pair_count']} | {row['all_psnr']} | {row['middle_psnr']} | {row['given_psnr']} | {row['paper_psnr']} | {row['local_minus_paper_middle_psnr']} |"
        )
    lines.extend([
        "",
        "## Stage72 Versus Corrected Stage75",
        "",
        "| comparison | Stage72 all | Stage72 middle | Stage75 all | Stage75 middle |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in compare_rows:
        lines.append(
            f"| {row['comparison']} | {row['stage72_all_psnr']} | {row['stage72_middle_psnr']} | {row['stage75_all_psnr']} | {row['stage75_middle_psnr']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Stage75 should be used when comparing local StreamSplat against paper-style DAVIS interpolation numbers. Stage72 remains useful only as a scoped Mono-DFCGS diagnostic baseline.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage74_root", type=Path, default=DEFAULT_STAGE74_ROOT)
    parser.add_argument("--stage72_rows", type=Path, default=DEFAULT_STAGE72_ROWS)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage74_aggregate = read_csv(args.stage74_root / "stage74_stage72_vs_actual_gap_diagnosis_aggregate.csv")
    stage74_rows = read_csv(args.stage74_root / "stage74_stage72_vs_actual_gap_diagnosis_rows.csv")
    stage72_rows = read_csv(args.stage72_rows)
    summary_rows = build_summary(stage74_aggregate)
    sequence_rows = build_sequence_rows(stage74_rows)
    compare_rows = build_stage72_comparison(stage72_rows, summary_rows)
    summary_csv = args.summary_root / "stage75_corrected_streamsplat_paper_protocol_summary.csv"
    sequence_csv = args.summary_root / "stage75_corrected_streamsplat_per_sequence.csv"
    compare_csv = args.summary_root / "stage75_stage72_vs_corrected_comparison.csv"
    report_md = args.summary_root / "stage75_corrected_streamsplat_paper_protocol_report.md"
    summary_json = args.summary_root / "stage75_corrected_streamsplat_paper_protocol_summary.json"
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(sequence_rows, sequence_csv, SEQUENCE_FIELDS)
    write_csv(compare_rows, compare_csv, COMPARE_FIELDS)
    write_report(summary_rows, compare_rows, report_md)
    summary = {
        "stage": 75,
        "mode": "corrected StreamSplat paper-protocol DAVIS package",
        "stage74_root": str(args.stage74_root),
        "stage72_rows": str(args.stage72_rows),
        "summary_csv": str(summary_csv),
        "sequence_csv": str(sequence_csv),
        "compare_csv": str(compare_csv),
        "report_md": str(report_md),
        "summary_json": str(summary_json),
        "summary_rows": summary_rows,
        "notes": [
            "Use this package for paper-style StreamSplat DAVIS references.",
            "Stage72 remains a scoped diagnostic baseline and should not be used as a paper-protocol reproduction.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_json),
        "summary_csv": str(summary_csv),
        "sequence_csv": str(sequence_csv),
        "compare_csv": str(compare_csv),
        "report_md": str(report_md),
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
