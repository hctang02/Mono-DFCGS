import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE78_RATE_TABLE = REPO_ROOT / "experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_rate_table.csv"
DEFAULT_STAGE93_SUMMARY = REPO_ROOT / "experiments/stage93_residual_sideinfo_entropy_codec_smoke/stage93_residual_sideinfo_entropy_codec_summary.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage94_entropy_residual_sideinfo_rd_package"
DEFAULT_OUTPUT_PREFIX = "stage94_entropy_residual_sideinfo_rd"
DEFAULT_REPORT_TITLE = "Stage94 Entropy-Coded Residual Side-Info RD Package"
DEFAULT_MODE = "entropy-coded residual side-info RD package"
DEFAULT_QUALITY_SOURCE = "Stage93 entropy codec smoke"
DEFAULT_SCOPE_NOTE = "This is a 12-task codec smoke package, not final full-video RD."

METHOD_TO_STAGE78 = {
    "linear": "linear",
    "stage65_adapter": "adapter",
}

METHOD_ORDER = {
    "linear": 0,
    "stage65_adapter": 1,
}

ROW_FIELDS = [
    "base_method",
    "stage78_rate_method",
    "codec",
    "reference_gap",
    "task_count",
    "q12_main_anchor_mib_per_frame",
    "fixed_sideinfo_mib_per_intermediate_frame",
    "entropy_sideinfo_mib_per_intermediate_frame",
    "uniform_intermediate_frame_ratio",
    "fixed_direct_total_mib_per_frame",
    "entropy_direct_total_mib_per_frame",
    "fixed_amortized_total_mib_per_frame",
    "entropy_amortized_total_mib_per_frame",
    "entropy_direct_savings_mib_per_frame",
    "entropy_amortized_savings_mib_per_frame",
    "entropy_ratio_vs_fixed_sideinfo",
    "mean_base_psnr",
    "mean_entropy_psnr",
    "mean_delta_psnr_vs_base",
    "positive_delta_count",
]

POINT_FIELDS = [
    "point_type",
    "base_method",
    "codec",
    "reference_gap",
    "rate_mode",
    "rate_mib_per_frame",
    "psnr",
    "delta_psnr_vs_base",
    "task_count",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value):
    return float(value) if value not in (None, "") else 0.0


def to_int(value):
    return int(float(value))


def uniform_intermediate_frame_ratio(gap):
    if gap <= 1:
        return 0.0
    return (gap - 1.0) / gap


def build_main_rate_lookup(rows):
    lookup = {}
    for row in rows:
        if row["codec"] != "q12":
            continue
        lookup[(row["method"], to_int(row["frame_gap"]))] = to_float(row["mean_static_anchor_mib_per_frame_with_metadata"])
    return lookup


def build_rows(stage93_rows, main_rate_lookup):
    out = []
    for row in stage93_rows:
        base_method = row["base_method"]
        if base_method not in METHOD_TO_STAGE78:
            raise KeyError(f"Unknown base method: {base_method}")
        gap = to_int(row["reference_gap"])
        stage78_method = METHOD_TO_STAGE78[base_method]
        main_rate = main_rate_lookup[(stage78_method, gap)]
        fixed_side = to_float(row["mean_fixed_mib_per_intermediate_frame"])
        entropy_side = to_float(row["mean_entropy_mib_per_intermediate_frame"])
        ratio = uniform_intermediate_frame_ratio(gap)
        fixed_direct = main_rate + fixed_side
        entropy_direct = main_rate + entropy_side
        fixed_amortized = main_rate + fixed_side * ratio
        entropy_amortized = main_rate + entropy_side * ratio
        out.append({
            "base_method": base_method,
            "stage78_rate_method": stage78_method,
            "codec": row["codec"],
            "reference_gap": gap,
            "task_count": to_int(row["task_count"]),
            "q12_main_anchor_mib_per_frame": main_rate,
            "fixed_sideinfo_mib_per_intermediate_frame": fixed_side,
            "entropy_sideinfo_mib_per_intermediate_frame": entropy_side,
            "uniform_intermediate_frame_ratio": ratio,
            "fixed_direct_total_mib_per_frame": fixed_direct,
            "entropy_direct_total_mib_per_frame": entropy_direct,
            "fixed_amortized_total_mib_per_frame": fixed_amortized,
            "entropy_amortized_total_mib_per_frame": entropy_amortized,
            "entropy_direct_savings_mib_per_frame": fixed_direct - entropy_direct,
            "entropy_amortized_savings_mib_per_frame": fixed_amortized - entropy_amortized,
            "entropy_ratio_vs_fixed_sideinfo": to_float(row["mean_entropy_ratio_vs_fixed"]),
            "mean_base_psnr": to_float(row["mean_base_psnr"]),
            "mean_entropy_psnr": to_float(row["mean_entropy_psnr"]),
            "mean_delta_psnr_vs_base": to_float(row["mean_delta_psnr_vs_base"]),
            "positive_delta_count": to_int(row["positive_delta_count"]),
        })
    return sorted(out, key=lambda item: (METHOD_ORDER[item["base_method"]], item["reference_gap"]))


def build_points(rows):
    out = []
    for row in rows:
        for rate_mode, fixed_key, entropy_key in [
            ("direct", "fixed_direct_total_mib_per_frame", "entropy_direct_total_mib_per_frame"),
            ("amortized", "fixed_amortized_total_mib_per_frame", "entropy_amortized_total_mib_per_frame"),
        ]:
            out.extend([
                {
                    "point_type": "fixed_sideinfo",
                    "base_method": row["base_method"],
                    "codec": row["codec"],
                    "reference_gap": row["reference_gap"],
                    "rate_mode": rate_mode,
                    "rate_mib_per_frame": row[fixed_key],
                    "psnr": row["mean_entropy_psnr"],
                    "delta_psnr_vs_base": row["mean_delta_psnr_vs_base"],
                    "task_count": row["task_count"],
                },
                {
                    "point_type": "entropy_sideinfo",
                    "base_method": row["base_method"],
                    "codec": row["codec"],
                    "reference_gap": row["reference_gap"],
                    "rate_mode": rate_mode,
                    "rate_mib_per_frame": row[entropy_key],
                    "psnr": row["mean_entropy_psnr"],
                    "delta_psnr_vs_base": row["mean_delta_psnr_vs_base"],
                    "task_count": row["task_count"],
                },
            ])
    return out


def format_float(value):
    return f"{value:.6f}"


def write_report(summary, rows, path):
    lines = [
        f"# {summary['report_title']}",
        "",
        "## Scope",
        "",
        "- Main rate comes from Stage78 q12 static-anchor rate table.",
        "- Side-info rate comes from Stage93 actual entropy-coded payload bytes.",
        f"- Rendered quality comes from {summary['quality_source']}.",
        f"- {summary['scope_note']}",
        "",
        "## RD Rows",
        "",
        "| base | gap | main | fixed side | entropy side | fixed direct | entropy direct | entropy amortized | entropy PSNR | delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['base_method']} | {row['reference_gap']} | {format_float(row['q12_main_anchor_mib_per_frame'])} | {format_float(row['fixed_sideinfo_mib_per_intermediate_frame'])} | {format_float(row['entropy_sideinfo_mib_per_intermediate_frame'])} | {format_float(row['fixed_direct_total_mib_per_frame'])} | {format_float(row['entropy_direct_total_mib_per_frame'])} | {format_float(row['entropy_amortized_total_mib_per_frame'])} | {format_float(row['mean_entropy_psnr'])} | {format_float(row['mean_delta_psnr_vs_base'])} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- rows CSV: `{summary['rows_csv']}`",
        f"- points CSV: `{summary['points_csv']}`",
        "",
        "## Conclusion",
        "",
        "- Entropy coding reduces side-info rate without changing decoded rendered quality relative to fixed q6 residual side-info.",
        "- Side-info remains transmitted information and is included in both direct and amortized total rates.",
        "- If this package uses a broader eval summary, it is the preferred RD reference over the 12-task smoke package.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage78_rate_table", type=Path, default=DEFAULT_STAGE78_RATE_TABLE)
    parser.add_argument("--stage93_summary", type=Path, default=DEFAULT_STAGE93_SUMMARY)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--stage", type=int, default=94)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--report_title", default=DEFAULT_REPORT_TITLE)
    parser.add_argument("--quality_source", default=DEFAULT_QUALITY_SOURCE)
    parser.add_argument("--scope_note", default=DEFAULT_SCOPE_NOTE)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    main_rate_lookup = build_main_rate_lookup(read_csv(args.stage78_rate_table))
    rows = build_rows(read_csv(args.stage93_summary), main_rate_lookup)
    points = build_points(rows)

    rows_csv = args.summary_root / f"{args.output_prefix}_rows.csv"
    points_csv = args.summary_root / f"{args.output_prefix}_points.csv"
    summary_json = args.summary_root / f"{args.output_prefix}_summary.json"
    report_md = args.summary_root / f"{args.output_prefix}_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(points, points_csv, POINT_FIELDS)
    summary = {
        "stage": args.stage,
        "mode": args.mode,
        "report_title": args.report_title,
        "quality_source": args.quality_source,
        "scope_note": args.scope_note,
        "stage78_rate_table": str(args.stage78_rate_table),
        "stage93_summary": str(args.stage93_summary),
        "rows_csv": str(rows_csv),
        "points_csv": str(points_csv),
        "report_md": str(report_md),
        "row_count": len(rows),
        "point_count": len(points),
        "rows": rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, rows, report_md)
    print(json.dumps({"summary": str(summary_json), "row_count": len(rows), "point_count": len(points)}, indent=2))


if __name__ == "__main__":
    main()
