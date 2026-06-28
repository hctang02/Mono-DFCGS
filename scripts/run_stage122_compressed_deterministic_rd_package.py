import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE121_GROUP_SUMMARY = REPO_ROOT / "experiments/stage121_broader_rendered_compressed_deterministic_validation/stage121_broader_rendered_compressed_deterministic_validation_group_summary.csv"
DEFAULT_STAGE121_SETTING_SUMMARY = REPO_ROOT / "experiments/stage121_broader_rendered_compressed_deterministic_validation/stage121_broader_rendered_compressed_deterministic_validation_setting_summary.csv"
DEFAULT_STAGE96_ROWS = REPO_ROOT / "experiments/stage96_broader_entropy_residual_sideinfo_rd_package/stage96_broader_entropy_residual_sideinfo_rd_rows.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage122_compressed_deterministic_rd_package"

ROLE_BY_SETTING = {
    "q4_top20": "primary",
    "q4_top10": "low_rate",
    "q5_top10": "near_anchor",
    "q6_top10": "anchor",
}
SETTING_ORDER = {"q4_top20": 0, "q4_top10": 1, "q5_top10": 2, "q6_top10": 3}
METHOD_ORDER = {"linear": 0, "stage65_adapter": 1}

RD_ROW_FIELDS = [
    "source",
    "role",
    "setting_label",
    "base_method",
    "codec",
    "reference_gap",
    "task_count",
    "keep_fraction",
    "side_bits",
    "sideinfo_payload_bytes",
    "sideinfo_mib_per_intermediate_frame",
    "q12_main_anchor_mib_per_frame",
    "direct_total_mib_per_frame",
    "amortized_total_mib_per_frame",
    "base_psnr",
    "psnr",
    "delta_psnr_vs_base",
    "delta_psnr_vs_q6_top10",
    "stage96_entropy_direct_total_mib_per_frame",
    "stage96_entropy_amortized_total_mib_per_frame",
    "stage96_entropy_psnr",
    "direct_rate_delta_vs_stage96_entropy",
    "amortized_rate_delta_vs_stage96_entropy",
    "psnr_delta_vs_stage96_entropy",
]

SETTING_SUMMARY_FIELDS = [
    "role",
    "setting_label",
    "keep_fraction",
    "side_bits",
    "task_row_count",
    "mean_sideinfo_payload_bytes",
    "mean_sideinfo_mib_per_intermediate_frame",
    "mean_direct_total_mib_per_frame",
    "mean_amortized_total_mib_per_frame",
    "mean_base_psnr",
    "mean_psnr",
    "mean_delta_psnr_vs_base",
    "mean_delta_psnr_vs_q6_top10",
    "weighted_direct_rate_delta_vs_stage96_entropy",
    "weighted_amortized_rate_delta_vs_stage96_entropy",
    "weighted_psnr_delta_vs_stage96_entropy",
]

POINT_FIELDS = [
    "point_type",
    "role",
    "setting_label",
    "base_method",
    "codec",
    "reference_gap",
    "rate_mode",
    "rate_mib_per_frame",
    "sideinfo_mib_per_intermediate_frame",
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


def weighted_average(rows, value_key, weight_key="task_count"):
    total_weight = sum(to_float(row[weight_key]) for row in rows)
    if total_weight <= 0.0:
        return 0.0
    return sum(to_float(row[value_key]) * to_float(row[weight_key]) for row in rows) / total_weight


def build_entropy_lookup(rows):
    out = {}
    for row in rows:
        key = (row["base_method"], row["codec"], to_int(row["reference_gap"]))
        out[key] = row
    return out


def build_rd_rows(group_rows, entropy_lookup):
    out = []
    for row in group_rows:
        setting = row["setting_label"]
        role = ROLE_BY_SETTING[setting]
        gap = to_int(row["reference_gap"])
        key = (row["base_method"], row["codec"], gap)
        entropy = entropy_lookup[key]
        out.append({
            "source": "stage121_compressed_deterministic",
            "role": role,
            "setting_label": setting,
            "base_method": row["base_method"],
            "codec": row["codec"],
            "reference_gap": gap,
            "task_count": to_int(row["task_count"]),
            "keep_fraction": to_float(row["keep_fraction"]),
            "side_bits": to_int(row["side_bits"]),
            "sideinfo_payload_bytes": to_float(row["mean_compressed_payload_bytes"]),
            "sideinfo_mib_per_intermediate_frame": to_float(row["mean_compressed_mib_per_intermediate_frame"]),
            "q12_main_anchor_mib_per_frame": to_float(row["mean_q12_main_anchor_mib_per_frame"]),
            "direct_total_mib_per_frame": to_float(row["mean_direct_total_mib_per_frame"]),
            "amortized_total_mib_per_frame": to_float(row["mean_amortized_total_mib_per_frame"]),
            "base_psnr": to_float(row["mean_base_psnr"]),
            "psnr": to_float(row["mean_setting_psnr"]),
            "delta_psnr_vs_base": to_float(row["mean_delta_psnr_vs_base"]),
            "delta_psnr_vs_q6_top10": to_float(row["mean_delta_psnr_vs_q6_top10"]),
            "stage96_entropy_direct_total_mib_per_frame": to_float(entropy["entropy_direct_total_mib_per_frame"]),
            "stage96_entropy_amortized_total_mib_per_frame": to_float(entropy["entropy_amortized_total_mib_per_frame"]),
            "stage96_entropy_psnr": to_float(entropy["mean_entropy_psnr"]),
            "direct_rate_delta_vs_stage96_entropy": to_float(row["mean_direct_total_mib_per_frame"]) - to_float(entropy["entropy_direct_total_mib_per_frame"]),
            "amortized_rate_delta_vs_stage96_entropy": to_float(row["mean_amortized_total_mib_per_frame"]) - to_float(entropy["entropy_amortized_total_mib_per_frame"]),
            "psnr_delta_vs_stage96_entropy": to_float(row["mean_setting_psnr"]) - to_float(entropy["mean_entropy_psnr"]),
        })
    return sorted(out, key=lambda item: (SETTING_ORDER[item["setting_label"]], METHOD_ORDER[item["base_method"]], item["reference_gap"]))


def build_setting_summary(setting_rows, rd_rows):
    by_setting = {}
    for row in setting_rows:
        by_setting[row["setting_label"]] = row
    out = []
    for setting in sorted(by_setting, key=lambda label: SETTING_ORDER[label]):
        rows = [row for row in rd_rows if row["setting_label"] == setting]
        setting_row = by_setting[setting]
        out.append({
            "role": ROLE_BY_SETTING[setting],
            "setting_label": setting,
            "keep_fraction": to_float(setting_row["keep_fraction"]),
            "side_bits": to_int(setting_row["side_bits"]),
            "task_row_count": to_int(setting_row["task_row_count"]),
            "mean_sideinfo_payload_bytes": to_float(setting_row["mean_compressed_payload_bytes"]),
            "mean_sideinfo_mib_per_intermediate_frame": to_float(setting_row["mean_compressed_mib_per_intermediate_frame"]),
            "mean_direct_total_mib_per_frame": to_float(setting_row["mean_direct_total_mib_per_frame"]),
            "mean_amortized_total_mib_per_frame": to_float(setting_row["mean_amortized_total_mib_per_frame"]),
            "mean_base_psnr": to_float(setting_row["mean_base_psnr"]),
            "mean_psnr": to_float(setting_row["mean_setting_psnr"]),
            "mean_delta_psnr_vs_base": to_float(setting_row["mean_delta_psnr_vs_base"]),
            "mean_delta_psnr_vs_q6_top10": to_float(setting_row["mean_delta_psnr_vs_q6_top10"]),
            "weighted_direct_rate_delta_vs_stage96_entropy": weighted_average(rows, "direct_rate_delta_vs_stage96_entropy"),
            "weighted_amortized_rate_delta_vs_stage96_entropy": weighted_average(rows, "amortized_rate_delta_vs_stage96_entropy"),
            "weighted_psnr_delta_vs_stage96_entropy": weighted_average(rows, "psnr_delta_vs_stage96_entropy"),
        })
    return out


def build_points(rd_rows, entropy_rows):
    out = []
    for row in rd_rows:
        for rate_mode, rate_key in [("direct", "direct_total_mib_per_frame"), ("amortized", "amortized_total_mib_per_frame")]:
            out.append({
                "point_type": "compressed_deterministic",
                "role": row["role"],
                "setting_label": row["setting_label"],
                "base_method": row["base_method"],
                "codec": row["codec"],
                "reference_gap": row["reference_gap"],
                "rate_mode": rate_mode,
                "rate_mib_per_frame": row[rate_key],
                "sideinfo_mib_per_intermediate_frame": row["sideinfo_mib_per_intermediate_frame"],
                "psnr": row["psnr"],
                "delta_psnr_vs_base": row["delta_psnr_vs_base"],
                "task_count": row["task_count"],
            })
    for row in entropy_rows:
        for rate_mode, rate_key in [("direct", "entropy_direct_total_mib_per_frame"), ("amortized", "entropy_amortized_total_mib_per_frame")]:
            out.append({
                "point_type": "stage96_entropy_reference",
                "role": "entropy_reference",
                "setting_label": "q6_top10_entropy_index_value",
                "base_method": row["base_method"],
                "codec": row["codec"],
                "reference_gap": to_int(row["reference_gap"]),
                "rate_mode": rate_mode,
                "rate_mib_per_frame": to_float(row[rate_key]),
                "sideinfo_mib_per_intermediate_frame": to_float(row["entropy_sideinfo_mib_per_intermediate_frame"]),
                "psnr": to_float(row["mean_entropy_psnr"]),
                "delta_psnr_vs_base": to_float(row["mean_delta_psnr_vs_base"]),
                "task_count": to_int(row["task_count"]),
            })
    return out


def format_float(value):
    return f"{float(value):.6f}"


def write_report(summary, setting_rows, path):
    lines = [
        "# Stage122 Compressed Deterministic RD Package",
        "",
        "## Scope",
        "",
        "- Quality and deterministic side-info rates come from Stage121 broader rendered validation.",
        "- Stage96 entropy-coded q6/top10 index+value side-info is included as reference.",
        "- Every transmitted side-info byte is counted in direct and amortized total rates.",
        "- Residual values are teacher-derived; this is not residual value prediction.",
        "",
        "## Setting Summary",
        "",
        "| role | setting | keep | bits | side bytes | direct | amortized | PSNR | delta base | delta q6 | dRate entropy | dPSNR entropy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in setting_rows:
        lines.append(
            f"| {row['role']} | {row['setting_label']} | {row['keep_fraction']} | {row['side_bits']} | {format_float(row['mean_sideinfo_payload_bytes'])} | {format_float(row['mean_direct_total_mib_per_frame'])} | {format_float(row['mean_amortized_total_mib_per_frame'])} | {format_float(row['mean_psnr'])} | {format_float(row['mean_delta_psnr_vs_base'])} | {format_float(row['mean_delta_psnr_vs_q6_top10'])} | {format_float(row['weighted_direct_rate_delta_vs_stage96_entropy'])} | {format_float(row['weighted_psnr_delta_vs_stage96_entropy'])} |"
        )
    lines.extend([
        "",
        "## Package Recommendation",
        "",
        "- Primary candidate: `q4_top20`.",
        "- Low-rate candidate: `q4_top10`.",
        "- Near-anchor candidate: `q5_top10`.",
        "- Anchor candidate: `q6_top10`.",
        "",
        "## Outputs",
        "",
        f"- RD rows CSV: `{summary['rd_rows_csv']}`",
        f"- RD points CSV: `{summary['rd_points_csv']}`",
        f"- setting summary CSV: `{summary['setting_summary_csv']}`",
        f"- package JSON: `{summary['package_json']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage121_group_summary", type=Path, default=DEFAULT_STAGE121_GROUP_SUMMARY)
    parser.add_argument("--stage121_setting_summary", type=Path, default=DEFAULT_STAGE121_SETTING_SUMMARY)
    parser.add_argument("--stage96_rows", type=Path, default=DEFAULT_STAGE96_ROWS)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage121_group_rows = read_csv(args.stage121_group_summary)
    stage121_setting_rows = read_csv(args.stage121_setting_summary)
    stage96_rows = read_csv(args.stage96_rows)
    entropy_lookup = build_entropy_lookup(stage96_rows)
    rd_rows = build_rd_rows(stage121_group_rows, entropy_lookup)
    setting_summary = build_setting_summary(stage121_setting_rows, rd_rows)
    points = build_points(rd_rows, stage96_rows)

    rd_rows_csv = args.summary_root / "stage122_compressed_deterministic_rd_rows.csv"
    rd_points_csv = args.summary_root / "stage122_compressed_deterministic_rd_points.csv"
    setting_summary_csv = args.summary_root / "stage122_compressed_deterministic_rd_setting_summary.csv"
    package_json = args.summary_root / "stage122_compressed_deterministic_rd_package.json"
    report_md = args.summary_root / "stage122_compressed_deterministic_rd_report.md"
    write_csv(rd_rows, rd_rows_csv, RD_ROW_FIELDS)
    write_csv(points, rd_points_csv, POINT_FIELDS)
    write_csv(setting_summary, setting_summary_csv, SETTING_SUMMARY_FIELDS)
    package = {
        "stage": 122,
        "mode": "compressed deterministic RD package",
        "stage121_group_summary": str(args.stage121_group_summary),
        "stage121_setting_summary": str(args.stage121_setting_summary),
        "stage96_rows": str(args.stage96_rows),
        "rd_rows_csv": str(rd_rows_csv),
        "rd_points_csv": str(rd_points_csv),
        "setting_summary_csv": str(setting_summary_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "row_count": len(rd_rows),
        "point_count": len(points),
        "recommendation": {
            "primary": "q4_top20",
            "low_rate": "q4_top10",
            "near_anchor": "q5_top10",
            "anchor": "q6_top10",
        },
        "setting_summary_rows": setting_summary,
        "notes": [
            "All side-info bytes are counted in direct and amortized total rates.",
            "Selected indices are decoder-reproducible endpoint-diff indices and are not transmitted.",
            "Residual values remain teacher-derived from dense target anchors.",
        ],
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, setting_summary, report_md)
    print(json.dumps({"package": str(package_json), "row_count": len(rd_rows), "point_count": len(points)}, indent=2))


if __name__ == "__main__":
    main()
