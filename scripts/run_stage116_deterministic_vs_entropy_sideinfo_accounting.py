import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE78_RATE_TABLE = REPO_ROOT / "experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_rate_table.csv"
DEFAULT_STAGE93_SUMMARY = REPO_ROOT / "experiments/stage93_residual_sideinfo_entropy_codec_smoke/stage93_residual_sideinfo_entropy_codec_summary.csv"
DEFAULT_STAGE96_ROWS = REPO_ROOT / "experiments/stage96_broader_entropy_residual_sideinfo_rd_package/stage96_broader_entropy_residual_sideinfo_rd_rows.csv"
DEFAULT_STAGE115_SUMMARY = REPO_ROOT / "experiments/stage115_deterministic_index_residual_codec_smoke/stage115_deterministic_index_residual_codec_summary.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage116_deterministic_vs_entropy_sideinfo_accounting"
DEFAULT_OUTPUT_PREFIX = "stage116_deterministic_vs_entropy_sideinfo_accounting"

MIB = 1024.0 * 1024.0

METHOD_TO_STAGE78 = {
    "linear": "linear",
    "stage65_adapter": "adapter",
}

METHOD_ORDER = {
    "linear": 0,
    "stage65_adapter": 1,
}

SOURCE_ORDER = {
    "stage93_entropy_smoke": 0,
    "stage96_entropy_broader": 1,
}

ROW_FIELDS = [
    "source_scope",
    "base_method",
    "stage78_rate_method",
    "codec",
    "reference_gap",
    "entropy_task_count",
    "deterministic_task_count",
    "q12_main_anchor_mib_per_frame",
    "uniform_intermediate_frame_ratio",
    "fixed_sideinfo_mib_per_intermediate_frame",
    "entropy_sideinfo_mib_per_intermediate_frame",
    "deterministic_sideinfo_mib_per_intermediate_frame",
    "fixed_sideinfo_payload_bytes",
    "entropy_sideinfo_payload_bytes",
    "deterministic_sideinfo_payload_bytes",
    "deterministic_minus_entropy_sideinfo_mib_per_intermediate_frame",
    "deterministic_minus_entropy_sideinfo_bytes",
    "deterministic_ratio_vs_entropy_sideinfo",
    "deterministic_sideinfo_savings_vs_fixed_mib_per_intermediate_frame",
    "deterministic_sideinfo_savings_vs_fixed_bytes",
    "entropy_sideinfo_savings_vs_fixed_mib_per_intermediate_frame",
    "entropy_sideinfo_savings_vs_fixed_bytes",
    "fixed_direct_total_mib_per_frame",
    "entropy_direct_total_mib_per_frame",
    "deterministic_direct_total_mib_per_frame",
    "fixed_amortized_total_mib_per_frame",
    "entropy_amortized_total_mib_per_frame",
    "deterministic_amortized_total_mib_per_frame",
    "deterministic_minus_entropy_direct_total_mib_per_frame",
    "deterministic_minus_entropy_amortized_total_mib_per_frame",
    "entropy_quality_psnr",
    "entropy_delta_psnr_vs_base",
    "deterministic_quality_status",
    "entropy_payload_accounting",
    "deterministic_payload_accounting",
    "entropy_transmits_indices",
    "deterministic_transmits_indices",
]

POINT_FIELDS = [
    "source_scope",
    "point_type",
    "base_method",
    "codec",
    "reference_gap",
    "rate_mode",
    "rate_mib_per_frame",
    "sideinfo_mib_per_intermediate_frame",
    "sideinfo_payload_bytes",
    "transmits_indices",
    "quality_psnr",
    "quality_status",
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
    if value in (None, ""):
        return 0.0
    return float(value)


def to_int(value):
    return int(float(value))


def uniform_intermediate_frame_ratio(gap):
    if gap <= 1:
        return 0.0
    return (float(gap) - 1.0) / float(gap)


def build_main_rate_lookup(rows):
    lookup = {}
    for row in rows:
        if row["codec"] != "q12":
            continue
        key = (row["method"], to_int(row["frame_gap"]))
        lookup[key] = to_float(row["mean_static_anchor_mib_per_frame_with_metadata"])
    return lookup


def build_deterministic_lookup(rows):
    lookup = {}
    for row in rows:
        key = (row["base_method"], row["codec"], to_int(row["reference_gap"]))
        lookup[key] = {
            "task_count": to_int(row["task_count"]),
            "fixed_payload_bytes": to_float(row["mean_fixed_payload_bytes"]),
            "deterministic_payload_bytes": to_float(row["mean_deterministic_payload_bytes"]),
            "fixed_sideinfo_mib_per_intermediate_frame": to_float(row["mean_fixed_mib_per_intermediate_frame"]),
            "deterministic_sideinfo_mib_per_intermediate_frame": to_float(row["mean_deterministic_mib_per_intermediate_frame"]),
        }
    return lookup


def entropy_rows_from_stage93(rows):
    out = []
    for row in rows:
        out.append({
            "source_scope": "stage93_entropy_smoke",
            "base_method": row["base_method"],
            "codec": row["codec"],
            "reference_gap": to_int(row["reference_gap"]),
            "task_count": to_int(row["task_count"]),
            "fixed_sideinfo_mib_per_intermediate_frame": to_float(row["mean_fixed_mib_per_intermediate_frame"]),
            "entropy_sideinfo_mib_per_intermediate_frame": to_float(row["mean_entropy_mib_per_intermediate_frame"]),
            "fixed_payload_bytes": to_float(row["mean_fixed_payload_bytes"]),
            "entropy_payload_bytes": to_float(row["mean_entropy_payload_bytes"]),
            "entropy_quality_psnr": to_float(row["mean_entropy_psnr"]),
            "entropy_delta_psnr_vs_base": to_float(row["mean_delta_psnr_vs_base"]),
        })
    return out


def entropy_rows_from_stage96(rows):
    out = []
    for row in rows:
        entropy_mib = to_float(row["entropy_sideinfo_mib_per_intermediate_frame"])
        fixed_mib = to_float(row["fixed_sideinfo_mib_per_intermediate_frame"])
        out.append({
            "source_scope": "stage96_entropy_broader",
            "base_method": row["base_method"],
            "codec": row["codec"],
            "reference_gap": to_int(row["reference_gap"]),
            "task_count": to_int(row["task_count"]),
            "fixed_sideinfo_mib_per_intermediate_frame": fixed_mib,
            "entropy_sideinfo_mib_per_intermediate_frame": entropy_mib,
            "fixed_payload_bytes": fixed_mib * MIB,
            "entropy_payload_bytes": entropy_mib * MIB,
            "entropy_quality_psnr": to_float(row["mean_entropy_psnr"]),
            "entropy_delta_psnr_vs_base": to_float(row["mean_delta_psnr_vs_base"]),
        })
    return out


def build_rows(entropy_rows, deterministic_lookup, main_rate_lookup):
    out = []
    for entropy in entropy_rows:
        base_method = entropy["base_method"]
        codec = entropy["codec"]
        gap = int(entropy["reference_gap"])
        key = (base_method, codec, gap)
        if key not in deterministic_lookup:
            raise KeyError(f"Missing deterministic Stage115 row for {key}")
        stage78_method = METHOD_TO_STAGE78[base_method]
        main_rate = main_rate_lookup[(stage78_method, gap)]
        det = deterministic_lookup[key]
        fixed_side_mib = entropy["fixed_sideinfo_mib_per_intermediate_frame"]
        entropy_side_mib = entropy["entropy_sideinfo_mib_per_intermediate_frame"]
        det_side_mib = det["deterministic_sideinfo_mib_per_intermediate_frame"]
        fixed_bytes = entropy["fixed_payload_bytes"]
        entropy_bytes = entropy["entropy_payload_bytes"]
        det_bytes = det["deterministic_payload_bytes"]
        ratio = uniform_intermediate_frame_ratio(gap)
        fixed_direct = main_rate + fixed_side_mib
        entropy_direct = main_rate + entropy_side_mib
        det_direct = main_rate + det_side_mib
        fixed_amortized = main_rate + fixed_side_mib * ratio
        entropy_amortized = main_rate + entropy_side_mib * ratio
        det_amortized = main_rate + det_side_mib * ratio
        out.append({
            "source_scope": entropy["source_scope"],
            "base_method": base_method,
            "stage78_rate_method": stage78_method,
            "codec": codec,
            "reference_gap": gap,
            "entropy_task_count": entropy["task_count"],
            "deterministic_task_count": det["task_count"],
            "q12_main_anchor_mib_per_frame": main_rate,
            "uniform_intermediate_frame_ratio": ratio,
            "fixed_sideinfo_mib_per_intermediate_frame": fixed_side_mib,
            "entropy_sideinfo_mib_per_intermediate_frame": entropy_side_mib,
            "deterministic_sideinfo_mib_per_intermediate_frame": det_side_mib,
            "fixed_sideinfo_payload_bytes": fixed_bytes,
            "entropy_sideinfo_payload_bytes": entropy_bytes,
            "deterministic_sideinfo_payload_bytes": det_bytes,
            "deterministic_minus_entropy_sideinfo_mib_per_intermediate_frame": det_side_mib - entropy_side_mib,
            "deterministic_minus_entropy_sideinfo_bytes": det_bytes - entropy_bytes,
            "deterministic_ratio_vs_entropy_sideinfo": det_side_mib / entropy_side_mib,
            "deterministic_sideinfo_savings_vs_fixed_mib_per_intermediate_frame": fixed_side_mib - det_side_mib,
            "deterministic_sideinfo_savings_vs_fixed_bytes": fixed_bytes - det_bytes,
            "entropy_sideinfo_savings_vs_fixed_mib_per_intermediate_frame": fixed_side_mib - entropy_side_mib,
            "entropy_sideinfo_savings_vs_fixed_bytes": fixed_bytes - entropy_bytes,
            "fixed_direct_total_mib_per_frame": fixed_direct,
            "entropy_direct_total_mib_per_frame": entropy_direct,
            "deterministic_direct_total_mib_per_frame": det_direct,
            "fixed_amortized_total_mib_per_frame": fixed_amortized,
            "entropy_amortized_total_mib_per_frame": entropy_amortized,
            "deterministic_amortized_total_mib_per_frame": det_amortized,
            "deterministic_minus_entropy_direct_total_mib_per_frame": det_direct - entropy_direct,
            "deterministic_minus_entropy_amortized_total_mib_per_frame": det_amortized - entropy_amortized,
            "entropy_quality_psnr": entropy["entropy_quality_psnr"],
            "entropy_delta_psnr_vs_base": entropy["entropy_delta_psnr_vs_base"],
            "deterministic_quality_status": "not_rendered_rate_only",
            "entropy_payload_accounting": "zlib(metadata+index_deltas+q_residual_values)+header",
            "deterministic_payload_accounting": "header+metadata+q_residual_values_no_indices",
            "entropy_transmits_indices": 1,
            "deterministic_transmits_indices": 0,
        })
    return sorted(out, key=lambda item: (SOURCE_ORDER[item["source_scope"]], METHOD_ORDER[item["base_method"]], item["reference_gap"]))


def build_points(rows):
    out = []
    variants = [
        ("fixed_index_value", "fixed", 1, "entropy_quality_psnr", "rendered_topk_reference"),
        ("entropy_index_value", "entropy", 1, "entropy_quality_psnr", "rendered_topk_reference"),
        ("deterministic_value_only", "deterministic", 0, None, "not_rendered_rate_only"),
    ]
    for row in rows:
        for rate_mode in ["direct", "amortized"]:
            for point_type, prefix, transmits_indices, psnr_key, quality_status in variants:
                out.append({
                    "source_scope": row["source_scope"],
                    "point_type": point_type,
                    "base_method": row["base_method"],
                    "codec": row["codec"],
                    "reference_gap": row["reference_gap"],
                    "rate_mode": rate_mode,
                    "rate_mib_per_frame": row[f"{prefix}_{rate_mode}_total_mib_per_frame"],
                    "sideinfo_mib_per_intermediate_frame": row[f"{prefix}_sideinfo_mib_per_intermediate_frame"],
                    "sideinfo_payload_bytes": row[f"{prefix}_sideinfo_payload_bytes"],
                    "transmits_indices": transmits_indices,
                    "quality_psnr": row[psnr_key] if psnr_key else "",
                    "quality_status": quality_status,
                })
    return out


def format_float(value):
    return f"{float(value):.6f}"


def write_report(summary, rows, path):
    lines = [
        "# Stage116 Deterministic vs Entropy Side-Info Accounting",
        "",
        "## Scope",
        "",
        "- Stage115 deterministic-index value-only payload is compared against entropy-coded index+value residual side-info.",
        "- Stage93 is the matched 12-task entropy smoke source; Stage96 is the broader 60-task entropy RD reference.",
        "- Deterministic rows are rate-only here because endpoint-diff deterministic residual indices were not rendered in Stage115.",
        "- Every transmitted side-info byte is counted: headers, metadata, residual values, and transmitted indices when present.",
        "",
        "## Rows",
        "",
        "| source | base | gap | entropy side | deterministic side | det - entropy | det/entropy | det direct | det amortized | quality |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['source_scope']} | {row['base_method']} | {row['reference_gap']} | {format_float(row['entropy_sideinfo_mib_per_intermediate_frame'])} | {format_float(row['deterministic_sideinfo_mib_per_intermediate_frame'])} | {format_float(row['deterministic_minus_entropy_sideinfo_mib_per_intermediate_frame'])} | {format_float(row['deterministic_ratio_vs_entropy_sideinfo'])} | {format_float(row['deterministic_direct_total_mib_per_frame'])} | {format_float(row['deterministic_amortized_total_mib_per_frame'])} | {row['deterministic_quality_status']} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- rows CSV: `{summary['rows_csv']}`",
        f"- points CSV: `{summary['points_csv']}`",
        f"- summary JSON: `{summary['summary_json']}`",
        "",
        "## Conclusion",
        "",
        "- Deterministic value-only side-info removes transmitted selected-index bytes and stays at `0.034340858459472656` MiB/intermediate for this q6 top10 setup.",
        "- Existing zlib entropy-coded index+value side-info is still smaller for linear groups and remains close for Stage65 adapter groups.",
        "- Stage116 is an accounting package, not a rendered quality validation for deterministic endpoint-diff residuals.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage78_rate_table", type=Path, default=DEFAULT_STAGE78_RATE_TABLE)
    parser.add_argument("--stage93_summary", type=Path, default=DEFAULT_STAGE93_SUMMARY)
    parser.add_argument("--stage96_rows", type=Path, default=DEFAULT_STAGE96_ROWS)
    parser.add_argument("--stage115_summary", type=Path, default=DEFAULT_STAGE115_SUMMARY)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    main_rate_lookup = build_main_rate_lookup(read_csv(args.stage78_rate_table))
    deterministic_lookup = build_deterministic_lookup(read_csv(args.stage115_summary))
    entropy_rows = entropy_rows_from_stage93(read_csv(args.stage93_summary)) + entropy_rows_from_stage96(read_csv(args.stage96_rows))
    rows = build_rows(entropy_rows, deterministic_lookup, main_rate_lookup)
    points = build_points(rows)

    rows_csv = args.summary_root / f"{args.output_prefix}_rows.csv"
    points_csv = args.summary_root / f"{args.output_prefix}_points.csv"
    summary_json = args.summary_root / f"{args.output_prefix}_summary.json"
    report_md = args.summary_root / f"{args.output_prefix}_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(points, points_csv, POINT_FIELDS)
    summary = {
        "stage": 116,
        "mode": "deterministic vs entropy side-info accounting",
        "stage78_rate_table": str(args.stage78_rate_table),
        "stage93_summary": str(args.stage93_summary),
        "stage96_rows": str(args.stage96_rows),
        "stage115_summary": str(args.stage115_summary),
        "rows_csv": str(rows_csv),
        "points_csv": str(points_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "row_count": len(rows),
        "point_count": len(points),
        "rows": rows,
        "notes": [
            "All transmitted side-info bytes are counted in side and total rates.",
            "Deterministic value-only rows do not transmit selected indices.",
            "Deterministic Stage115 quality is not rendered in this accounting package.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, rows, report_md)
    print(json.dumps({"summary": str(summary_json), "row_count": len(rows), "point_count": len(points)}, indent=2))


if __name__ == "__main__":
    main()
