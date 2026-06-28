import argparse
import csv
import json
import math
import struct
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE115_SUMMARY = REPO_ROOT / "experiments/stage115_deterministic_index_residual_codec_smoke/stage115_deterministic_index_residual_codec_summary.csv"
DEFAULT_STAGE116_ROWS = REPO_ROOT / "experiments/stage116_deterministic_vs_entropy_sideinfo_accounting/stage116_deterministic_vs_entropy_sideinfo_accounting_rows.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage117_deterministic_sideinfo_sweep"
DEFAULT_OUTPUT_PREFIX = "stage117_deterministic_sideinfo_sweep"

MIB = 1024.0 * 1024.0
FIXED_HEADER_BYTES = struct.Struct("<4sBBBBHII").size
DETERMINISTIC_HEADER_BYTES = struct.Struct("<4sBBBBHII").size
FLOAT16_BYTES = 2

DEFAULT_KEEP_FRACTIONS = [0.025, 0.05, 0.1, 0.15, 0.2]
DEFAULT_SIDE_BITS = [2, 3, 4, 5, 6, 8]

METHOD_ORDER = {
    "linear": 0,
    "stage65_adapter": 1,
}

ROW_FIELDS = [
    "source_scope",
    "base_method",
    "codec",
    "reference_gap",
    "keep_fraction",
    "side_bits",
    "gaussian_count",
    "attr_dim",
    "keep_count",
    "fixed_payload_bytes",
    "deterministic_payload_bytes",
    "fixed_sideinfo_mib_per_intermediate_frame",
    "deterministic_sideinfo_mib_per_intermediate_frame",
    "saved_index_bytes_vs_fixed",
    "deterministic_ratio_vs_fixed",
    "entropy_reference_setting",
    "entropy_reference_mib_per_intermediate_frame",
    "entropy_reference_payload_bytes",
    "deterministic_minus_entropy_reference_mib_per_intermediate_frame",
    "deterministic_minus_entropy_reference_bytes",
    "deterministic_ratio_vs_entropy_reference",
    "below_entropy_reference_rate",
    "q12_main_anchor_mib_per_frame",
    "uniform_intermediate_frame_ratio",
    "deterministic_direct_total_mib_per_frame",
    "deterministic_amortized_total_mib_per_frame",
    "comparison_note",
]

SUMMARY_FIELDS = [
    "keep_fraction",
    "side_bits",
    "gaussian_count",
    "attr_dim",
    "keep_count",
    "fixed_payload_bytes",
    "deterministic_payload_bytes",
    "fixed_sideinfo_mib_per_intermediate_frame",
    "deterministic_sideinfo_mib_per_intermediate_frame",
    "saved_index_bytes_vs_fixed",
    "deterministic_ratio_vs_fixed",
    "group_count",
    "groups_below_entropy_reference_rate",
    "mean_ratio_vs_entropy_reference",
    "max_ratio_vs_entropy_reference",
    "mean_deterministic_minus_entropy_reference_bytes",
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


def payload_sizes(gaussian_count, attr_dim, keep_fraction, side_bits):
    keep_count = min(max(int(round(int(gaussian_count) * float(keep_fraction))), 0), int(gaussian_count))
    metadata_bytes = int(attr_dim) * 2 * FLOAT16_BYTES
    index_bits = math.ceil(math.log2(max(int(gaussian_count), 2)))
    index_bytes = (keep_count * index_bits + 7) // 8
    residual_bytes = (keep_count * int(attr_dim) * int(side_bits) + 7) // 8
    fixed_payload_bytes = FIXED_HEADER_BYTES + metadata_bytes + index_bytes + residual_bytes
    deterministic_payload_bytes = DETERMINISTIC_HEADER_BYTES + metadata_bytes + residual_bytes
    return {
        "keep_count": keep_count,
        "metadata_bytes": metadata_bytes,
        "index_bits": index_bits,
        "index_bytes": index_bytes,
        "residual_bytes": residual_bytes,
        "fixed_payload_bytes": fixed_payload_bytes,
        "deterministic_payload_bytes": deterministic_payload_bytes,
    }


def derive_geometry(stage115_rows):
    first = stage115_rows[0]
    keep_fraction = to_float(first["keep_fraction"])
    keep_count = to_int(first["mean_keep_gaussians"])
    gaussian_count = int(round(keep_count / keep_fraction))
    side_bits = to_int(first["side_bits"])
    target_bytes = int(round(to_float(first["mean_deterministic_payload_bytes"])))
    for attr_dim in range(1, 257):
        sizes = payload_sizes(gaussian_count, attr_dim, keep_fraction, side_bits)
        if sizes["deterministic_payload_bytes"] == target_bytes:
            return {
                "gaussian_count": gaussian_count,
                "attr_dim": attr_dim,
                "reference_keep_fraction": keep_fraction,
                "reference_side_bits": side_bits,
                "reference_keep_count": keep_count,
                "reference_deterministic_payload_bytes": target_bytes,
            }
    raise RuntimeError("Could not derive attr_dim from Stage115 deterministic payload")


def load_entropy_reference_rows(rows, source_scope):
    out = []
    for row in rows:
        if row["source_scope"] != source_scope:
            continue
        out.append({
            "source_scope": row["source_scope"],
            "base_method": row["base_method"],
            "codec": row["codec"],
            "reference_gap": to_int(row["reference_gap"]),
            "entropy_reference_mib_per_intermediate_frame": to_float(row["entropy_sideinfo_mib_per_intermediate_frame"]),
            "entropy_reference_payload_bytes": to_float(row["entropy_sideinfo_payload_bytes"]),
            "q12_main_anchor_mib_per_frame": to_float(row["q12_main_anchor_mib_per_frame"]),
            "uniform_intermediate_frame_ratio": to_float(row["uniform_intermediate_frame_ratio"]),
        })
    return sorted(out, key=lambda item: (METHOD_ORDER[item["base_method"]], item["reference_gap"]))


def build_rows(geometry, reference_rows, keep_fractions, side_bits_values):
    out = []
    for keep_fraction in keep_fractions:
        for side_bits in side_bits_values:
            sizes = payload_sizes(geometry["gaussian_count"], geometry["attr_dim"], keep_fraction, side_bits)
            fixed_mib = sizes["fixed_payload_bytes"] / MIB
            det_mib = sizes["deterministic_payload_bytes"] / MIB
            for ref in reference_rows:
                entropy_mib = ref["entropy_reference_mib_per_intermediate_frame"]
                entropy_bytes = ref["entropy_reference_payload_bytes"]
                ratio = ref["uniform_intermediate_frame_ratio"]
                det_direct = ref["q12_main_anchor_mib_per_frame"] + det_mib
                det_amortized = ref["q12_main_anchor_mib_per_frame"] + det_mib * ratio
                same_setting = (
                    abs(float(keep_fraction) - float(geometry["reference_keep_fraction"])) < 1e-12
                    and int(side_bits) == int(geometry["reference_side_bits"])
                )
                out.append({
                    "source_scope": ref["source_scope"],
                    "base_method": ref["base_method"],
                    "codec": ref["codec"],
                    "reference_gap": ref["reference_gap"],
                    "keep_fraction": keep_fraction,
                    "side_bits": side_bits,
                    "gaussian_count": geometry["gaussian_count"],
                    "attr_dim": geometry["attr_dim"],
                    "keep_count": sizes["keep_count"],
                    "fixed_payload_bytes": sizes["fixed_payload_bytes"],
                    "deterministic_payload_bytes": sizes["deterministic_payload_bytes"],
                    "fixed_sideinfo_mib_per_intermediate_frame": fixed_mib,
                    "deterministic_sideinfo_mib_per_intermediate_frame": det_mib,
                    "saved_index_bytes_vs_fixed": sizes["index_bytes"],
                    "deterministic_ratio_vs_fixed": sizes["deterministic_payload_bytes"] / sizes["fixed_payload_bytes"],
                    "entropy_reference_setting": "stage96_q6_top10_measured",
                    "entropy_reference_mib_per_intermediate_frame": entropy_mib,
                    "entropy_reference_payload_bytes": entropy_bytes,
                    "deterministic_minus_entropy_reference_mib_per_intermediate_frame": det_mib - entropy_mib,
                    "deterministic_minus_entropy_reference_bytes": sizes["deterministic_payload_bytes"] - entropy_bytes,
                    "deterministic_ratio_vs_entropy_reference": det_mib / entropy_mib,
                    "below_entropy_reference_rate": int(det_mib < entropy_mib),
                    "q12_main_anchor_mib_per_frame": ref["q12_main_anchor_mib_per_frame"],
                    "uniform_intermediate_frame_ratio": ratio,
                    "deterministic_direct_total_mib_per_frame": det_direct,
                    "deterministic_amortized_total_mib_per_frame": det_amortized,
                    "comparison_note": "same_setting_rate_only" if same_setting else "cross_setting_rate_only_quality_unknown",
                })
    return out


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(float(row["keep_fraction"]), int(row["side_bits"]))].append(row)
    out = []
    for (keep_fraction, side_bits), items in sorted(grouped.items()):
        first = items[0]
        ratios = [float(row["deterministic_ratio_vs_entropy_reference"]) for row in items]
        deltas = [float(row["deterministic_minus_entropy_reference_bytes"]) for row in items]
        out.append({
            "keep_fraction": keep_fraction,
            "side_bits": side_bits,
            "gaussian_count": first["gaussian_count"],
            "attr_dim": first["attr_dim"],
            "keep_count": first["keep_count"],
            "fixed_payload_bytes": first["fixed_payload_bytes"],
            "deterministic_payload_bytes": first["deterministic_payload_bytes"],
            "fixed_sideinfo_mib_per_intermediate_frame": first["fixed_sideinfo_mib_per_intermediate_frame"],
            "deterministic_sideinfo_mib_per_intermediate_frame": first["deterministic_sideinfo_mib_per_intermediate_frame"],
            "saved_index_bytes_vs_fixed": first["saved_index_bytes_vs_fixed"],
            "deterministic_ratio_vs_fixed": first["deterministic_ratio_vs_fixed"],
            "group_count": len(items),
            "groups_below_entropy_reference_rate": sum(int(row["below_entropy_reference_rate"]) for row in items),
            "mean_ratio_vs_entropy_reference": sum(ratios) / len(ratios),
            "max_ratio_vs_entropy_reference": max(ratios),
            "mean_deterministic_minus_entropy_reference_bytes": sum(deltas) / len(deltas),
        })
    return out


def format_float(value):
    return f"{float(value):.6f}"


def write_report(summary, summary_rows, path):
    lines = [
        "# Stage117 Deterministic Side-Info Sweep",
        "",
        "## Scope",
        "",
        "- Sweeps deterministic-index value-only payload size over keep fraction and side bits.",
        "- The entropy reference is Stage116 / Stage96 measured q6 top10 entropy-coded index+value side-info.",
        "- Non-q6/top10 rows are cross-setting rate-only comparisons; rendered quality is unknown.",
        "- Every deterministic transmitted byte is counted: header, metadata, and q residual values. Indices are not transmitted.",
        "",
        "## Geometry",
        "",
        f"- gaussian count: `{summary['geometry']['gaussian_count']}`",
        f"- attr dim: `{summary['geometry']['attr_dim']}`",
        f"- Stage115 reference keep fraction: `{summary['geometry']['reference_keep_fraction']}`",
        f"- Stage115 reference side bits: `{summary['geometry']['reference_side_bits']}`",
        "",
        "## Setting Summary",
        "",
        "| keep | bits | keep count | det bytes | det MiB | saved index bytes | groups below Stage96 entropy | max det/entropy | note |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        same_setting = row["keep_fraction"] == summary["geometry"]["reference_keep_fraction"] and row["side_bits"] == summary["geometry"]["reference_side_bits"]
        note = "Stage115 setting" if same_setting else "quality unknown"
        lines.append(
            f"| {row['keep_fraction']} | {row['side_bits']} | {row['keep_count']} | {row['deterministic_payload_bytes']} | {format_float(row['deterministic_sideinfo_mib_per_intermediate_frame'])} | {row['saved_index_bytes_vs_fixed']} | {row['groups_below_entropy_reference_rate']}/{row['group_count']} | {format_float(row['max_ratio_vs_entropy_reference'])} | {note} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- rows CSV: `{summary['rows_csv']}`",
        f"- summary CSV: `{summary['setting_summary_csv']}`",
        f"- summary JSON: `{summary['summary_json']}`",
        "",
        "## Conclusion",
        "",
        "- The Stage115 q6 top10 deterministic payload is `36009 bytes`, matching the derived geometry.",
        "- Lower keep fractions or lower side bits can beat the Stage96 q6 top10 entropy reference in rate, but their rendered quality is not validated here.",
        "- Stage118 should only package RD points after rendered quality exists for the selected deterministic settings.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage115_summary", type=Path, default=DEFAULT_STAGE115_SUMMARY)
    parser.add_argument("--stage116_rows", type=Path, default=DEFAULT_STAGE116_ROWS)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--entropy_source_scope", default="stage96_entropy_broader")
    parser.add_argument("--keep_fractions", nargs="+", type=float, default=DEFAULT_KEEP_FRACTIONS)
    parser.add_argument("--side_bits", nargs="+", type=int, default=DEFAULT_SIDE_BITS)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    geometry = derive_geometry(read_csv(args.stage115_summary))
    reference_rows = load_entropy_reference_rows(read_csv(args.stage116_rows), args.entropy_source_scope)
    if not reference_rows:
        raise RuntimeError(f"No Stage116 rows found for source scope {args.entropy_source_scope}")
    rows = build_rows(geometry, reference_rows, args.keep_fractions, args.side_bits)
    summary_rows = summarize(rows)

    rows_csv = args.summary_root / f"{args.output_prefix}_rows.csv"
    setting_summary_csv = args.summary_root / f"{args.output_prefix}_setting_summary.csv"
    summary_json = args.summary_root / f"{args.output_prefix}_summary.json"
    report_md = args.summary_root / f"{args.output_prefix}_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, setting_summary_csv, SUMMARY_FIELDS)
    summary = {
        "stage": 117,
        "mode": "deterministic side-info q-bit / keep-fraction sweep",
        "stage115_summary": str(args.stage115_summary),
        "stage116_rows": str(args.stage116_rows),
        "entropy_source_scope": args.entropy_source_scope,
        "keep_fractions": args.keep_fractions,
        "side_bits": args.side_bits,
        "geometry": geometry,
        "rows_csv": str(rows_csv),
        "setting_summary_csv": str(setting_summary_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "row_count": len(rows),
        "setting_count": len(summary_rows),
        "setting_summary_rows": summary_rows,
        "notes": [
            "The entropy reference is measured Stage96 q6 top10 entropy-coded index+value side-info.",
            "Non-q6/top10 deterministic settings are cross-setting rate-only comparisons with unknown rendered quality.",
            "Deterministic value-only side-info does not transmit selected indices.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, summary_rows, report_md)
    print(json.dumps({"summary": str(summary_json), "row_count": len(rows), "setting_count": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
