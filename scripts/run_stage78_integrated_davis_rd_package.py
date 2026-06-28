import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE75_ROOT = REPO_ROOT / "experiments/stage75_corrected_streamsplat_paper_protocol_package"
DEFAULT_STAGE77_ROOT = REPO_ROOT / "experiments/stage77_qbit_full_video_anchor_only_rd_sweep"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage78_integrated_davis_rd_package"

METHOD_LABELS = {
    "linear": "Linear Anchor",
    "adapter": "Stage65 Adapter",
}

CODEC_ORDER = {"q8": 0, "q10": 1, "q12": 2}
METHOD_ORDER = {"linear": 0, "adapter": 1}

RATE_FIELDS = [
    "codec",
    "bits",
    "frame_gap",
    "method",
    "mean_static_anchor_mib_per_frame_with_metadata",
]

PSNR_FIELDS = [
    "codec",
    "bits",
    "frame_gap",
    "method",
    "mean_all_psnr",
    "mean_middle_psnr",
    "mean_given_psnr",
    "mean_all_ssim",
    "mean_middle_ssim",
    "mean_given_ssim",
]

METHOD_SUMMARY_FIELDS = [
    "method",
    "codec",
    "bits",
    "point_count",
    "mean_rate_mib_per_frame",
    "mean_all_psnr",
    "mean_middle_psnr",
    "mean_given_psnr",
]

REFERENCE_FIELDS = [
    "reference_setting",
    "reference_local_gap",
    "reference_middle_psnr",
    "reference_all_psnr",
    "reference_given_psnr",
    "anchor_codec",
    "anchor_gap",
    "anchor_method",
    "anchor_rate_mib_per_frame",
    "anchor_all_psnr",
    "anchor_middle_psnr",
    "anchor_given_psnr",
    "middle_psnr_gap_to_reference",
    "comparison_note",
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


def normalized_stage77_rows(rows):
    out = []
    for row in rows:
        item = dict(row)
        for key in [
            "bits",
            "frame_gap",
            "sequence_count",
            "frame_count",
        ]:
            item[key] = to_int(item[key])
        for key in [
            "mean_static_anchor_mib_per_frame_with_metadata",
            "mean_all_psnr",
            "mean_middle_psnr",
            "mean_given_psnr",
            "mean_all_ssim",
            "mean_middle_ssim",
            "mean_given_ssim",
            "delta_all_vs_q8",
            "delta_middle_vs_q8",
        ]:
            item[key] = to_float(item[key])
        out.append(item)
    return sorted(out, key=lambda row: (CODEC_ORDER[row["codec"]], row["frame_gap"], METHOD_ORDER[row["method"]]))


def normalized_stage75_rows(rows):
    out = []
    for row in rows:
        item = dict(row)
        for key in ["local_gap", "pair_count", "all_count", "middle_count", "given_count"]:
            item[key] = to_int(item[key])
        for key in ["all_psnr", "middle_psnr", "given_psnr", "paper_psnr", "local_minus_paper_middle_psnr"]:
            item[key] = to_float(item[key])
        out.append(item)
    return out


def build_rate_rows(rows):
    return [{field: row[field] for field in RATE_FIELDS} for row in rows]


def build_psnr_rows(rows):
    return [{field: row[field] for field in PSNR_FIELDS} for row in rows]


def build_method_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["codec"])].append(row)
    out = []
    for (method, codec), items in sorted(grouped.items(), key=lambda item: (METHOD_ORDER[item[0][0]], CODEC_ORDER[item[0][1]])):
        out.append({
            "method": method,
            "codec": codec,
            "bits": items[0]["bits"],
            "point_count": len(items),
            "mean_rate_mib_per_frame": float(np.mean([row["mean_static_anchor_mib_per_frame_with_metadata"] for row in items])),
            "mean_all_psnr": float(np.mean([row["mean_all_psnr"] for row in items])),
            "mean_middle_psnr": float(np.mean([row["mean_middle_psnr"] for row in items])),
            "mean_given_psnr": float(np.mean([row["mean_given_psnr"] for row in items])),
        })
    return out


def build_reference_rows(anchor_rows, reference_rows):
    ref_by_gap = {row["local_gap"]: row for row in reference_rows}
    mapping = {
        4: (5, "diagnostic: Stage77 gap4 has 3 middle frames, Stage75 gap5 is paper Middle-4"),
        8: (8, "diagnostic: both are treated as local 8-frame interval references"),
    }
    out = []
    for anchor in anchor_rows:
        if anchor["frame_gap"] not in mapping:
            continue
        ref_gap, note = mapping[anchor["frame_gap"]]
        ref = ref_by_gap[ref_gap]
        out.append({
            "reference_setting": ref["paper_setting"],
            "reference_local_gap": ref["local_gap"],
            "reference_middle_psnr": ref["middle_psnr"],
            "reference_all_psnr": ref["all_psnr"],
            "reference_given_psnr": ref["given_psnr"],
            "anchor_codec": anchor["codec"],
            "anchor_gap": anchor["frame_gap"],
            "anchor_method": anchor["method"],
            "anchor_rate_mib_per_frame": anchor["mean_static_anchor_mib_per_frame_with_metadata"],
            "anchor_all_psnr": anchor["mean_all_psnr"],
            "anchor_middle_psnr": anchor["mean_middle_psnr"],
            "anchor_given_psnr": anchor["mean_given_psnr"],
            "middle_psnr_gap_to_reference": anchor["mean_middle_psnr"] - ref["middle_psnr"],
            "comparison_note": note,
        })
    return out


def plot_rd(rows, y_key, ylabel, path):
    plt.figure(figsize=(7, 5))
    for method in ["linear", "adapter"]:
        for codec in ["q8", "q10", "q12"]:
            items = [row for row in rows if row["method"] == method and row["codec"] == codec]
            items = sorted(items, key=lambda row: row["mean_static_anchor_mib_per_frame_with_metadata"])
            xs = [row["mean_static_anchor_mib_per_frame_with_metadata"] for row in items]
            ys = [row[y_key] for row in items]
            label = f"{METHOD_LABELS[method]} {codec}"
            marker = "o" if method == "adapter" else "s"
            linestyle = "-" if method == "adapter" else "--"
            plt.plot(xs, ys, marker=marker, linestyle=linestyle, label=label)
            for row, x, y in zip(items, xs, ys):
                plt.annotate(f"g{row['frame_gap']}", (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
    plt.xlabel("Static anchor payload (MiB/frame, metadata included)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_report(summary, rate_rows, psnr_rows, method_summary, reference_rows, path):
    lines = [
        "# Stage78 Integrated DAVIS RD Package",
        "",
        "## Scope",
        "",
        "- Anchor-only RD rows come from Stage77 scoped DAVIS val q8/q10/q12 sweep.",
        "- StreamSplat reference rows come from Stage75 corrected paper-protocol package.",
        "- FCGS/D-FCGS are intentionally not included in this package.",
        "- Primary anchor-only metric remains all-frame PSNR; middle/given are diagnostic.",
        "",
        "## Anchor-Only RD Summary",
        "",
        "| codec | gap | method | MiB/frame | all PSNR | middle PSNR | given PSNR |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in psnr_rows:
        matching_rate = next(
            item for item in rate_rows
            if item["codec"] == row["codec"] and item["frame_gap"] == row["frame_gap"] and item["method"] == row["method"]
        )
        lines.append(
            f"| {row['codec']} | {row['frame_gap']} | {row['method']} | {matching_rate['mean_static_anchor_mib_per_frame_with_metadata']} | {row['mean_all_psnr']} | {row['mean_middle_psnr']} | {row['mean_given_psnr']} |"
        )
    lines.extend([
        "",
        "## Method Averages",
        "",
        "| method | codec | mean rate | mean all PSNR | mean middle PSNR | mean given PSNR |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in method_summary:
        lines.append(
            f"| {row['method']} | {row['codec']} | {row['mean_rate_mib_per_frame']} | {row['mean_all_psnr']} | {row['mean_middle_psnr']} | {row['mean_given_psnr']} |"
        )
    lines.extend([
        "",
        "## StreamSplat Reference Gap",
        "",
        "| ref setting | anchor codec | anchor gap | method | anchor middle | ref middle | gap | note |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ])
    for row in reference_rows:
        lines.append(
            f"| {row['reference_setting']} | {row['anchor_codec']} | {row['anchor_gap']} | {row['anchor_method']} | {row['anchor_middle_psnr']} | {row['reference_middle_psnr']} | {row['middle_psnr_gap_to_reference']} | {row['comparison_note']} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Summary JSON: `{summary['summary_json']}`",
        f"- Rate table: `{summary['rate_table_csv']}`",
        f"- PSNR table: `{summary['psnr_table_csv']}`",
        f"- Method summary: `{summary['method_summary_csv']}`",
        f"- Reference gap table: `{summary['reference_gap_csv']}`",
        f"- All-frame RD plot: `{summary['all_rd_plot']}`",
        f"- Middle-frame RD plot: `{summary['middle_rd_plot']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage75_root", type=Path, default=DEFAULT_STAGE75_ROOT)
    parser.add_argument("--stage77_root", type=Path, default=DEFAULT_STAGE77_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage75_summary = normalized_stage75_rows(read_csv(args.stage75_root / "stage75_corrected_streamsplat_paper_protocol_summary.csv"))
    stage77_summary = normalized_stage77_rows(read_csv(args.stage77_root / "stage77_qbit_full_video_anchor_only_rd_summary.csv"))
    rate_rows = build_rate_rows(stage77_summary)
    psnr_rows = build_psnr_rows(stage77_summary)
    method_summary = build_method_summary(stage77_summary)
    reference_rows = build_reference_rows(stage77_summary, stage75_summary)

    rate_table_csv = args.summary_root / "stage78_anchor_only_rate_table.csv"
    psnr_table_csv = args.summary_root / "stage78_anchor_only_psnr_table.csv"
    method_summary_csv = args.summary_root / "stage78_method_summary.csv"
    reference_gap_csv = args.summary_root / "stage78_reference_gap_table.csv"
    all_rd_plot = args.summary_root / "stage78_anchor_only_all_rd_curve.png"
    middle_rd_plot = args.summary_root / "stage78_anchor_only_middle_rd_curve.png"
    report_md = args.summary_root / "stage78_integrated_davis_rd_report.md"
    summary_json = args.summary_root / "stage78_integrated_davis_rd_summary.json"

    write_csv(rate_rows, rate_table_csv, RATE_FIELDS)
    write_csv(psnr_rows, psnr_table_csv, PSNR_FIELDS)
    write_csv(method_summary, method_summary_csv, METHOD_SUMMARY_FIELDS)
    write_csv(reference_rows, reference_gap_csv, REFERENCE_FIELDS)
    plot_rd(stage77_summary, "mean_all_psnr", "All-frame PSNR", all_rd_plot)
    plot_rd(stage77_summary, "mean_middle_psnr", "Middle-frame PSNR", middle_rd_plot)

    summary = {
        "stage": 78,
        "mode": "integrated DAVIS RD package",
        "stage75_root": str(args.stage75_root),
        "stage77_root": str(args.stage77_root),
        "summary_root": str(args.summary_root),
        "rate_table_csv": str(rate_table_csv),
        "psnr_table_csv": str(psnr_table_csv),
        "method_summary_csv": str(method_summary_csv),
        "reference_gap_csv": str(reference_gap_csv),
        "all_rd_plot": str(all_rd_plot),
        "middle_rd_plot": str(middle_rd_plot),
        "report_md": str(report_md),
        "summary_json": str(summary_json),
        "method_summary": method_summary,
        "notes": [
            "Stage78 excludes FCGS/D-FCGS by user request.",
            "Stage75 StreamSplat reference is paper-protocol; Stage77 anchor-only rows are scoped DAVIS RD diagnostics.",
            "Use all-frame PSNR as primary anchor-only RD metric and middle/given PSNR as diagnostics.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, rate_rows, psnr_rows, method_summary, reference_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "all_rd_plot": str(all_rd_plot),
        "middle_rd_plot": str(middle_rd_plot),
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
