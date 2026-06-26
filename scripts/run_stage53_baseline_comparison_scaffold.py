import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE51_CSV = REPO_ROOT / "experiments/stage51_high_rate_multibit_rd/stage51_high_rate_multibit_rd.csv"
DEFAULT_STAGE52_CSV = REPO_ROOT / "experiments/stage52_fcgs_dfcgs_baseline_preflight/stage52_baseline_summary_records.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage53_baseline_comparison_scaffold"


SCAFFOLD_FIELDS = [
    "family",
    "method",
    "variant",
    "sample",
    "sequence",
    "dataset_protocol",
    "source_stage",
    "source_group",
    "source_file",
    "literature_or_local",
    "fair_local_run",
    "comparison_status",
    "rate_mib_per_frame",
    "rate_unit",
    "rate_scope",
    "secondary_rate_value",
    "secondary_rate_unit",
    "quality_psnr",
    "quality_ssim",
    "quality_lpips",
    "quality_l1",
    "quality_middle_psnr",
    "diagnostic_codec_psnr",
    "diagnostic_codec_l1",
    "quality_scope",
    "quality_reliable_for_input_video",
    "frame_count",
    "keyframe_or_iframe_count",
    "pframe_count",
    "reference_gap",
    "bits",
    "max_points_per_frame",
    "lambda_or_q",
    "stage53_candidate",
    "notes",
]


AGG_FIELDS = [
    "family",
    "method",
    "sample",
    "bits",
    "lambda_or_q",
    "source_group",
    "rows",
    "mean_rate_mib_per_frame",
    "mean_quality_psnr",
    "mean_quality_ssim",
    "mean_quality_middle_psnr",
    "stage53_candidate_rows",
    "comparison_status",
    "rate_unit",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, fields, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def maybe_float(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def maybe_int(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    return int(float(value))


def csv_bool(value):
    return str(value).lower() == "true"


def bool_text(value):
    return "true" if value else "false"


def mean(values):
    values = [float(value) for value in values if value not in (None, "")]
    if not values:
        return None
    return sum(values) / len(values)


def normalize_ours(stage51_rows, source_path):
    rows = []
    for row in stage51_rows:
        method = row["method"]
        if method == "rendered_prior_0p1":
            method_name = "Mono-DFCGS adaptive"
            comparison_status = "our local oracle-calibrated selector; not final deployable selector"
            selector_note = "Adaptive layout uses rendered-oracle calibrated costs from Stage45b/49/51."
        else:
            method_name = "Mono-DFCGS uniform"
            comparison_status = "our local deployable uniform-keyframe baseline"
            selector_note = "Uniform keyframe layout is deployable; adapter weights are excluded from rate."
        bits = maybe_int(row, "bits")
        gap = maybe_int(row, "reference_gap")
        rows.append({
            "family": "ours",
            "method": method_name,
            "variant": f"{method} gap={gap} q{bits}",
            "sample": row["sample"],
            "sequence": row["sample"],
            "dataset_protocol": "local StreamSplat-derived monocular videos; anchor-only adapter full-video RD",
            "source_stage": "Stage51",
            "source_group": "stage51_high_rate_multibit_rd",
            "source_file": str(source_path),
            "literature_or_local": "local run",
            "fair_local_run": "true",
            "comparison_status": comparison_status,
            "rate_mib_per_frame": maybe_float(row, "zlib_mib_per_frame"),
            "rate_unit": "actual zlib Gaussian-anchor bitstream MiB/frame",
            "rate_scope": "transmitted keyframe Gaussian anchors only; decoder/model weights excluded",
            "secondary_rate_value": maybe_float(row, "raw_mib_per_frame"),
            "secondary_rate_unit": "actual raw Gaussian-anchor bitstream MiB/frame",
            "quality_psnr": maybe_float(row, "adapter_all_psnr"),
            "quality_ssim": maybe_float(row, "adapter_all_ssim"),
            "quality_lpips": None,
            "quality_l1": None,
            "quality_middle_psnr": maybe_float(row, "adapter_middle_psnr"),
            "diagnostic_codec_psnr": None,
            "diagnostic_codec_l1": None,
            "quality_scope": "all-frame rendered adapter PSNR/SSIM; middle-only PSNR included when available",
            "quality_reliable_for_input_video": "true",
            "frame_count": maybe_int(row, "total_frames"),
            "keyframe_or_iframe_count": maybe_int(row, "keyframe_count"),
            "pframe_count": None,
            "reference_gap": gap,
            "bits": bits,
            "max_points_per_frame": None,
            "lambda_or_q": None,
            "stage53_candidate": "true",
            "notes": selector_note + " Multi-bit q10/q12/q16 storage is the Stage50 prototype container, not bit-packed entropy coding.",
        })
    return rows


def external_method_name(row):
    codec_mode = row.get("codec_mode")
    if codec_mode == "fcgs_per_frame":
        return "FCGS"
    if codec_mode == "fcgs_i_plus_dfcgs_p_gop2":
        return "FCGS-I + D-FCGS-P"
    if codec_mode == "raw_i_plus_dfcgs_p_gop2":
        return "Raw-I + D-FCGS-P"
    return row.get("method") or "external baseline"


def normalize_external(stage52_rows, source_path):
    rows = []
    for row in stage52_rows:
        quality_ok = csv_bool(row.get("quality_for_stage53"))
        rate_ok = csv_bool(row.get("rate_for_stage53"))
        stage53_candidate = csv_bool(row.get("stage53_candidate"))
        dummy = csv_bool(row.get("dummy_reference_images"))
        method_name = external_method_name(row)
        if dummy:
            comparison_status = "local external diagnostic only; dummy references do not provide input-video quality"
        elif stage53_candidate:
            comparison_status = "local external protocol reference; rate/quality available but not apples-to-apples with Mono-DFCGS"
        else:
            comparison_status = "local external incomplete candidate"
        quality_psnr = maybe_float(row, "psnr_avg") if quality_ok else None
        quality_ssim = maybe_float(row, "ssim_avg") if quality_ok else None
        quality_l1 = maybe_float(row, "l1_avg") if quality_ok else None
        quality_lpips = maybe_float(row, "lpips_avg_nonnull") if quality_ok else None
        codec_mode = row.get("codec_mode")
        rows.append({
            "family": "external_baseline",
            "method": method_name,
            "variant": f"{codec_mode} lambda_or_q={row.get('lambda_or_q')} points={row.get('max_points_per_frame')}",
            "sample": row.get("sample"),
            "sequence": row.get("sequence"),
            "dataset_protocol": "local external FCGS/D-FCGS generated from CompactWorld/NeoVerse Gaussian streams",
            "source_stage": "Stage52",
            "source_group": row.get("source_group"),
            "source_file": row.get("path") or str(source_path),
            "literature_or_local": "local run",
            "fair_local_run": "false",
            "comparison_status": comparison_status,
            "rate_mib_per_frame": maybe_float(row, "avg_size_mib_per_frame") if rate_ok else None,
            "rate_unit": "full FCGS/D-FCGS codec MiB/frame",
            "rate_scope": "codec bitstream includes I-frame payloads plus D-FCGS P-frame payloads when present",
            "secondary_rate_value": maybe_float(row, "total_size_mib"),
            "secondary_rate_unit": "total sequence codec MiB",
            "quality_psnr": quality_psnr,
            "quality_ssim": quality_ssim,
            "quality_lpips": quality_lpips,
            "quality_l1": quality_l1,
            "quality_middle_psnr": None,
            "diagnostic_codec_psnr": maybe_float(row, "codec_psnr_avg"),
            "diagnostic_codec_l1": maybe_float(row, "codec_l1_avg"),
            "quality_scope": "input-video all-frame summary when dummy_reference_images=false; codec_psnr is P-frame/raw-render diagnostic",
            "quality_reliable_for_input_video": bool_text(quality_ok),
            "frame_count": maybe_int(row, "frame_count"),
            "keyframe_or_iframe_count": maybe_int(row, "iframe_count"),
            "pframe_count": maybe_int(row, "pframe_count"),
            "reference_gap": 2 if "gop2" in str(codec_mode) else None,
            "bits": None,
            "max_points_per_frame": maybe_int(row, "max_points_per_frame"),
            "lambda_or_q": row.get("lambda_or_q"),
            "stage53_candidate": bool_text(stage53_candidate),
            "notes": row.get("notes"),
        })
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["family"],
            row["method"],
            row["sample"],
            str(row.get("bits") or ""),
            str(row.get("lambda_or_q") or ""),
            row["source_group"],
            row["comparison_status"],
            row["rate_unit"],
        )
        grouped[key].append(row)
    out = []
    for key, group_rows in sorted(grouped.items()):
        family, method, sample, bits, lambda_or_q, source_group, comparison_status, rate_unit = key
        out.append({
            "family": family,
            "method": method,
            "sample": sample,
            "bits": bits,
            "lambda_or_q": lambda_or_q,
            "source_group": source_group,
            "rows": len(group_rows),
            "mean_rate_mib_per_frame": mean(row.get("rate_mib_per_frame") for row in group_rows),
            "mean_quality_psnr": mean(row.get("quality_psnr") for row in group_rows),
            "mean_quality_ssim": mean(row.get("quality_ssim") for row in group_rows),
            "mean_quality_middle_psnr": mean(row.get("quality_middle_psnr") for row in group_rows),
            "stage53_candidate_rows": sum(1 for row in group_rows if row.get("stage53_candidate") == "true"),
            "comparison_status": comparison_status,
            "rate_unit": rate_unit,
        })
    return out


def counts_by(rows, key):
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def write_report(summary, paths, path):
    lines = [
        "# Stage53 Baseline Comparison Scaffold",
        "",
        "## Outputs",
        f"- Unified scaffold CSV: `{paths['scaffold_csv']}`",
        f"- Mono-DFCGS rows CSV: `{paths['ours_csv']}`",
        f"- External baseline rows CSV: `{paths['external_csv']}`",
        f"- Method/sample aggregate CSV: `{paths['aggregate_csv']}`",
        f"- Summary JSON: `{paths['summary_json']}`",
        "",
        "## Counts",
        f"- Unified rows: {summary['rows_total']}",
        f"- Mono-DFCGS rows: {summary['ours_rows']}",
        f"- External baseline rows: {summary['external_rows']}",
        f"- External rows with input-video quality and rate: {summary['external_stage53_candidate_rows']}",
        f"- Fair external apples-to-apples rows: {summary['fair_external_rows']}",
        "",
        "## Methods",
        "| Method | Rows |",
        "|---|---:|",
    ]
    for method, count in summary["rows_by_method"].items():
        lines.append(f"| {method} | {count} |")
    lines.extend([
        "",
        "## Comparison Status",
        "| Status | Rows |",
        "|---|---:|",
    ])
    for status, count in summary["rows_by_comparison_status"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend([
        "",
        "## Usage Notes",
        "- Use Mono-DFCGS rows for our own RD curves; rate is transmitted Gaussian-anchor bitstream MiB/frame and excludes model weights.",
        "- Use external rows as local protocol-reference baselines only unless a future Stage adds matched inputs, matched frame sets, and matched rate accounting.",
        "- Do not mix `full FCGS/D-FCGS codec MiB/frame` with Mono-DFCGS anchor-only MiB/frame on an unlabeled primary plot.",
        "- Rows with `quality_reliable_for_input_video=false` are diagnostic only; their codec PSNR can be reported separately from input-video PSNR.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage51_csv", type=Path, default=DEFAULT_STAGE51_CSV)
    parser.add_argument("--stage52_csv", type=Path, default=DEFAULT_STAGE52_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    ours_rows = normalize_ours(read_csv(args.stage51_csv), args.stage51_csv)
    external_rows = normalize_external(read_csv(args.stage52_csv), args.stage52_csv)
    scaffold_rows = ours_rows + external_rows
    aggregate_rows = aggregate(scaffold_rows)

    ours_csv = args.summary_root / "stage53_mono_dfcgs_rows.csv"
    external_csv = args.summary_root / "stage53_external_baseline_rows.csv"
    scaffold_csv = args.summary_root / "stage53_baseline_comparison_scaffold.csv"
    aggregate_csv = args.summary_root / "stage53_method_sample_aggregate.csv"
    summary_json = args.summary_root / "stage53_baseline_comparison_scaffold_summary.json"
    report_md = args.summary_root / "stage53_baseline_comparison_scaffold_report.md"

    write_csv(ours_rows, SCAFFOLD_FIELDS, ours_csv)
    write_csv(external_rows, SCAFFOLD_FIELDS, external_csv)
    write_csv(scaffold_rows, SCAFFOLD_FIELDS, scaffold_csv)
    write_csv(aggregate_rows, AGG_FIELDS, aggregate_csv)

    summary = {
        "stage": 53,
        "mode": "baseline comparison scaffold",
        "stage51_csv": str(args.stage51_csv),
        "stage52_csv": str(args.stage52_csv),
        "rows_total": len(scaffold_rows),
        "ours_rows": len(ours_rows),
        "external_rows": len(external_rows),
        "external_stage53_candidate_rows": sum(1 for row in external_rows if row["stage53_candidate"] == "true"),
        "fair_external_rows": sum(1 for row in external_rows if row["fair_local_run"] == "true"),
        "rows_by_family": counts_by(scaffold_rows, "family"),
        "rows_by_method": counts_by(scaffold_rows, "method"),
        "rows_by_comparison_status": counts_by(scaffold_rows, "comparison_status"),
        "rows_by_rate_unit": counts_by(scaffold_rows, "rate_unit"),
        "outputs": {
            "ours_csv": str(ours_csv),
            "external_csv": str(external_csv),
            "scaffold_csv": str(scaffold_csv),
            "aggregate_csv": str(aggregate_csv),
            "summary_json": str(summary_json),
            "report_md": str(report_md),
        },
        "notes": [
            "This stage normalizes existing records into a comparison scaffold; it does not make FCGS/D-FCGS apples-to-apples with Mono-DFCGS.",
            "External fair_local_run remains false because rate scope, source Gaussian generation, and protocol differ from Mono-DFCGS.",
            "Use this scaffold to choose Stage54/Stage57 tables and to identify missing fair-run requirements.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, summary["outputs"], report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "rows_total": summary["rows_total"],
        "ours_rows": summary["ours_rows"],
        "external_rows": summary["external_rows"],
        "external_stage53_candidate_rows": summary["external_stage53_candidate_rows"],
        "fair_external_rows": summary["fair_external_rows"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
