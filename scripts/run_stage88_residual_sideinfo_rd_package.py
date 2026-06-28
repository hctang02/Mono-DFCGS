import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE78_RATE_TABLE = REPO_ROOT / "experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_rate_table.csv"
DEFAULT_STAGE87_SUMMARY = REPO_ROOT / "experiments/stage87_quantized_residual_sideinfo_smoke/stage87_quantized_residual_sideinfo_summary.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage88_residual_sideinfo_rd_package"
DEFAULT_OUTPUT_PREFIX = "stage88_residual_sideinfo_rd"
DEFAULT_REPORT_TITLE = "Stage88 Residual Side-Info RD Package"
DEFAULT_MODE = "residual side-info RD package"
DEFAULT_QUALITY_SOURCE = "Stage87 quantized residual side-info smoke"
DEFAULT_SCOPE_NOTE = "This is a 12-task smoke package, not a full-video RD benchmark."
DEFAULT_PLOT_QUALITY_LABEL = "Rendered PSNR (dB, Stage87 12-task smoke)"

METHOD_TO_STAGE78 = {
    "linear": "linear",
    "stage65_adapter": "adapter",
}

METHOD_LABELS = {
    "linear": "Linear Anchor",
    "stage65_adapter": "Stage65 Adapter",
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
    "keep_fraction",
    "side_bits",
    "task_count",
    "q12_main_anchor_mib_per_frame",
    "side_info_mib_per_intermediate_frame",
    "uniform_intermediate_frame_ratio",
    "direct_total_mib_per_frame",
    "amortized_total_mib_per_frame",
    "mean_base_psnr",
    "mean_sideinfo_psnr",
    "mean_delta_psnr_vs_base",
    "positive_delta_count",
    "extra_direct_mib_per_frame",
    "extra_amortized_mib_per_frame",
    "delta_psnr_per_direct_extra_mib",
    "delta_psnr_per_amortized_extra_mib",
]

POINT_FIELDS = [
    "point_type",
    "base_method",
    "codec",
    "reference_gap",
    "keep_fraction",
    "side_bits",
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
        key = (row["method"], to_int(row["frame_gap"]))
        lookup[key] = to_float(row["mean_static_anchor_mib_per_frame_with_metadata"])
    return lookup


def build_rd_rows(stage87_rows, main_rate_lookup):
    out = []
    for row in stage87_rows:
        codec = row["codec"]
        if codec != "q12":
            continue
        base_method = row["base_method"]
        if base_method not in METHOD_TO_STAGE78:
            raise KeyError(f"Unknown base method: {base_method}")
        gap = to_int(row["reference_gap"])
        stage78_method = METHOD_TO_STAGE78[base_method]
        main_key = (stage78_method, gap)
        if main_key not in main_rate_lookup:
            raise KeyError(f"Missing Stage78 main rate for {main_key}")

        main_rate = main_rate_lookup[main_key]
        side_rate = to_float(row["mean_side_info_mib_per_intermediate_frame"])
        ratio = uniform_intermediate_frame_ratio(gap)
        direct_total = main_rate + side_rate
        amortized_total = main_rate + side_rate * ratio
        delta = to_float(row["mean_delta_psnr_vs_base"])
        extra_direct = direct_total - main_rate
        extra_amortized = amortized_total - main_rate
        out.append({
            "base_method": base_method,
            "stage78_rate_method": stage78_method,
            "codec": codec,
            "reference_gap": gap,
            "keep_fraction": to_float(row["keep_fraction"]),
            "side_bits": to_int(row["side_bits"]),
            "task_count": to_int(row["task_count"]),
            "q12_main_anchor_mib_per_frame": main_rate,
            "side_info_mib_per_intermediate_frame": side_rate,
            "uniform_intermediate_frame_ratio": ratio,
            "direct_total_mib_per_frame": direct_total,
            "amortized_total_mib_per_frame": amortized_total,
            "mean_base_psnr": to_float(row["mean_base_psnr"]),
            "mean_sideinfo_psnr": to_float(row["mean_sideinfo_psnr"]),
            "mean_delta_psnr_vs_base": delta,
            "positive_delta_count": to_int(row["positive_delta_count"]),
            "extra_direct_mib_per_frame": extra_direct,
            "extra_amortized_mib_per_frame": extra_amortized,
            "delta_psnr_per_direct_extra_mib": delta / extra_direct if extra_direct > 0.0 else 0.0,
            "delta_psnr_per_amortized_extra_mib": delta / extra_amortized if extra_amortized > 0.0 else 0.0,
        })
    return sorted(out, key=lambda item: (METHOD_ORDER[item["base_method"]], item["reference_gap"], item["keep_fraction"], item["side_bits"]))


def build_point_rows(rd_rows):
    rows = []
    seen_base = set()
    for row in rd_rows:
        base_key = (row["base_method"], row["codec"], row["reference_gap"])
        if base_key not in seen_base:
            seen_base.add(base_key)
            for rate_mode in ["direct", "amortized"]:
                rows.append({
                    "point_type": "base",
                    "base_method": row["base_method"],
                    "codec": row["codec"],
                    "reference_gap": row["reference_gap"],
                    "keep_fraction": 0.0,
                    "side_bits": 0,
                    "rate_mode": rate_mode,
                    "rate_mib_per_frame": row["q12_main_anchor_mib_per_frame"],
                    "psnr": row["mean_base_psnr"],
                    "delta_psnr_vs_base": 0.0,
                    "task_count": row["task_count"],
                })
        for rate_mode, rate_key in [
            ("direct", "direct_total_mib_per_frame"),
            ("amortized", "amortized_total_mib_per_frame"),
        ]:
            rows.append({
                "point_type": "sideinfo",
                "base_method": row["base_method"],
                "codec": row["codec"],
                "reference_gap": row["reference_gap"],
                "keep_fraction": row["keep_fraction"],
                "side_bits": row["side_bits"],
                "rate_mode": rate_mode,
                "rate_mib_per_frame": row[rate_key],
                "psnr": row["mean_sideinfo_psnr"],
                "delta_psnr_vs_base": row["mean_delta_psnr_vs_base"],
                "task_count": row["task_count"],
            })
    return sorted(rows, key=lambda item: (item["rate_mode"], METHOD_ORDER[item["base_method"]], item["reference_gap"], item["keep_fraction"], item["side_bits"], item["point_type"]))


def plot_rd(rd_rows, rate_key, title, xlabel, ylabel, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for method in ["linear", "stage65_adapter"]:
        base_items = []
        seen_gap = set()
        for row in rd_rows:
            if row["base_method"] != method or row["reference_gap"] in seen_gap:
                continue
            seen_gap.add(row["reference_gap"])
            base_items.append(row)
        base_items = sorted(base_items, key=lambda item: item["q12_main_anchor_mib_per_frame"])
        plt.plot(
            [row["q12_main_anchor_mib_per_frame"] for row in base_items],
            [row["mean_base_psnr"] for row in base_items],
            marker="x",
            linestyle=":" if method == "linear" else "-.",
            label=f"{METHOD_LABELS[method]} base",
        )

        for keep, bits in [(0.1, 6), (0.1, 8), (0.25, 6), (0.25, 8)]:
            items = [
                row for row in rd_rows
                if row["base_method"] == method and row["keep_fraction"] == keep and row["side_bits"] == bits
            ]
            items = sorted(items, key=lambda item: item[rate_key])
            if not items:
                continue
            plt.plot(
                [row[rate_key] for row in items],
                [row["mean_sideinfo_psnr"] for row in items],
                marker="o" if method == "stage65_adapter" else "s",
                linestyle="-" if keep == 0.25 else "--",
                label=f"{METHOD_LABELS[method]} keep{keep:g} q{bits}",
            )
            for row in items:
                plt.annotate(
                    f"g{row['reference_gap']}",
                    (row[rate_key], row["mean_sideinfo_psnr"]),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def format_float(value):
    return f"{value:.6f}"


def write_report(summary, rd_rows, path):
    top10_q6 = [row for row in rd_rows if row["keep_fraction"] == 0.1 and row["side_bits"] == 6]
    top10_q8 = [row for row in rd_rows if row["keep_fraction"] == 0.1 and row["side_bits"] == 8]
    lines = [
        f"# {summary['report_title']}",
        "",
        "## Scope",
        "",
        "- Main rate comes from Stage78 q12 static-anchor rate table.",
        f"- Rendered quality comes from {summary['quality_source']}.",
        "- Side-info is treated as transmitted information and included in total rate.",
        f"- {summary['scope_note']}",
        "",
        "## Rate Definitions",
        "",
        "- `direct_total_mib_per_frame = q12_main_anchor_mib_per_frame + side_info_mib_per_intermediate_frame`.",
        "- `amortized_total_mib_per_frame = q12_main_anchor_mib_per_frame + side_info_mib_per_intermediate_frame * ((gap - 1) / gap)`.",
        "- The direct total is conservative for the per-intermediate-frame side-info eval; the amortized total is a uniform-gap full-video approximation.",
        "",
        "## Low-Rate Operating Point: keep 0.1, q6 residual",
        "",
        "| base | gap | main MiB/frame | side MiB/intermediate | direct total | amortized total | base PSNR | side PSNR | delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top10_q6:
        lines.append(
            f"| {row['base_method']} | {row['reference_gap']} | {format_float(row['q12_main_anchor_mib_per_frame'])} | {format_float(row['side_info_mib_per_intermediate_frame'])} | {format_float(row['direct_total_mib_per_frame'])} | {format_float(row['amortized_total_mib_per_frame'])} | {format_float(row['mean_base_psnr'])} | {format_float(row['mean_sideinfo_psnr'])} | {format_float(row['mean_delta_psnr_vs_base'])} |"
        )
    lines.extend([
        "",
        "## Full RD Rows",
        "",
        "| base | gap | keep | bits | direct total | amortized total | side PSNR | delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rd_rows:
        lines.append(
            f"| {row['base_method']} | {row['reference_gap']} | {row['keep_fraction']} | {row['side_bits']} | {format_float(row['direct_total_mib_per_frame'])} | {format_float(row['amortized_total_mib_per_frame'])} | {format_float(row['mean_sideinfo_psnr'])} | {format_float(row['mean_delta_psnr_vs_base'])} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- rows CSV: `{summary['rows_csv']}`",
        f"- point CSV: `{summary['points_csv']}`",
        f"- direct RD plot: `{summary['direct_plot']}`",
        f"- amortized RD plot: `{summary['amortized_plot']}`",
        "",
        "## Conclusion",
        "",
    ])
    if top10_q6:
        lines.append(
            f"- q6 top10 residual side-info adds `{format_float(top10_q6[0]['side_info_mib_per_intermediate_frame'])} MiB/intermediate-frame` and preserves multi-dB rendered PSNR gains in this package."
        )
    if top10_q8:
        lines.append("- q8 top10 side-info is included for cross-bit comparison; q6 should remain the low-rate default if quality is similar.")
    else:
        lines.append("- This package only includes the requested side-info operating points; it is not a cross-bit sweep.")
    lines.append("- The next step is a real bitstream/entropy-coded side-info implementation; this package does not yet claim final RD.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage78_rate_table", type=Path, default=DEFAULT_STAGE78_RATE_TABLE)
    parser.add_argument("--stage87_summary", type=Path, default=DEFAULT_STAGE87_SUMMARY)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--stage", type=int, default=88)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--report_title", default=DEFAULT_REPORT_TITLE)
    parser.add_argument("--quality_source", default=DEFAULT_QUALITY_SOURCE)
    parser.add_argument("--scope_note", default=DEFAULT_SCOPE_NOTE)
    parser.add_argument("--plot_quality_label", default=DEFAULT_PLOT_QUALITY_LABEL)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)

    main_rate_lookup = build_main_rate_lookup(read_csv(args.stage78_rate_table))
    rd_rows = build_rd_rows(read_csv(args.stage87_summary), main_rate_lookup)
    point_rows = build_point_rows(rd_rows)

    rows_csv = args.summary_root / f"{args.output_prefix}_rows.csv"
    points_csv = args.summary_root / f"{args.output_prefix}_points.csv"
    summary_json = args.summary_root / f"{args.output_prefix}_summary.json"
    report_md = args.summary_root / f"{args.output_prefix}_report.md"
    direct_plot = args.summary_root / f"{args.output_prefix}_direct.png"
    amortized_plot = args.summary_root / f"{args.output_prefix}_amortized.png"

    write_csv(rd_rows, rows_csv, ROW_FIELDS)
    write_csv(point_rows, points_csv, POINT_FIELDS)
    plot_rd(
        rd_rows,
        "direct_total_mib_per_frame",
        f"{args.report_title} (Direct Total Rate)",
        "q12 main anchor + side-info (MiB/frame)",
        args.plot_quality_label,
        direct_plot,
    )
    plot_rd(
        rd_rows,
        "amortized_total_mib_per_frame",
        f"{args.report_title} (Uniform-Gap Amortized Rate)",
        "q12 main anchor + amortized side-info (MiB/frame)",
        args.plot_quality_label,
        amortized_plot,
    )

    summary = {
        "stage": args.stage,
        "mode": args.mode,
        "report_title": args.report_title,
        "quality_source": args.quality_source,
        "scope_note": args.scope_note,
        "stage78_rate_table": str(args.stage78_rate_table),
        "stage87_summary": str(args.stage87_summary),
        "rows_csv": str(rows_csv),
        "points_csv": str(points_csv),
        "report_md": str(report_md),
        "direct_plot": str(direct_plot),
        "amortized_plot": str(amortized_plot),
        "row_count": len(rd_rows),
        "point_count": len(point_rows),
        "rate_definitions": {
            "direct_total_mib_per_frame": "q12_main_anchor_mib_per_frame + side_info_mib_per_intermediate_frame",
            "amortized_total_mib_per_frame": "q12_main_anchor_mib_per_frame + side_info_mib_per_intermediate_frame * ((gap - 1) / gap)",
        },
        "rows": rd_rows,
    }
    write_report(summary, rd_rows, report_md)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_json), "row_count": len(rd_rows)}, indent=2))


if __name__ == "__main__":
    main()
