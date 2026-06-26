import argparse
import ast
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage52_fcgs_dfcgs_baseline_preflight"


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
LOG_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2} .*? - [A-Za-z-]+ - [A-Z]+ - ")
EVAL_RE = re.compile(
    r"Evaluation results: PSNR: (?P<psnr>[-+0-9.eE]+), SSIM: (?P<ssim>[-+0-9.eE]+), "
    r"LPIPS: (?P<lpips>[-+0-9.eE]+), L1: (?P<l1>[-+0-9.eE]+)"
)
ZSTD_RE = re.compile(r"motion_xyz zstd size: (?P<bytes>\d+) bytes \((?P<mb>[-+0-9.eE]+) MB\)")
PAIR_RE = re.compile(r"pair_(?P<start>\d+)_(?P<end>\d+)")


SOURCE_PATTERNS = [
    (
        "legacy_driving_dfcgs_logs",
        Path("/mnt/hdd2tC/tmp/opencode/dfcgs_fcgs_i_model/driving"),
        "D-FCGS_20260620_*.log",
    ),
    (
        "multisequence_dfcgs_logs",
        Path("/mnt/hdd2tC/tmp/opencode/multisequence_rd/rd_sweep/dfcgs_model"),
        "**/D-FCGS_20260626_*.log",
    ),
    (
        "multisequence_dfcgs_fcgs_i_summaries",
        Path("/mnt/hdd2tC/tmp/opencode/multisequence_rd/rd_sweep/dfcgs_fcgs_i"),
        "**/dfcgs_gop2_target_topology_summary.json",
    ),
    (
        "compactworld_dfcgs_fcgs_i_lambda_sweep",
        Path("/mnt/hdd2tC/tmp/opencode/dfcgs_fcgs_i_lambda_sweep"),
        "**/dfcgs_gop2_target_topology_summary.json",
    ),
    (
        "compactworld_dfcgs_raw_i_rate_sweep",
        Path("/mnt/hdd2tC/tmp/opencode/dfcgs_codec_rd_sweep"),
        "**/dfcgs_gop2_target_topology_summary.json",
    ),
    (
        "compactworld_dfcgs_fcgs_i_gop2",
        Path("/mnt/hdd2tC/tmp/opencode/dfcgs_fcgs_i_gop2"),
        "**/dfcgs_gop2_target_topology_summary.json",
    ),
    (
        "lowrate_dfcgs_fcgs_i_summaries",
        Path("/mnt/hdd2tC/tmp/opencode/lowrate_rd_sweep/dfcgs_fcgs_i"),
        "**/dfcgs_gop2_target_topology_summary.json",
    ),
    (
        "compactworld_standalone_fcgs_summaries",
        Path("/mnt/hdd2tC/tmp/opencode"),
        "fcgs_*_out/fcgs_sequence_summary.json",
    ),
    (
        "multisequence_standalone_fcgs_summaries",
        Path("/mnt/hdd2tC/tmp/opencode/multisequence_rd/rd_sweep/fcgs"),
        "**/fcgs_sequence_summary.json",
    ),
    (
        "lowrate_standalone_fcgs_summaries",
        Path("/mnt/hdd2tC/tmp/opencode/lowrate_rd_sweep/fcgs"),
        "**/fcgs_sequence_summary.json",
    ),
]


LOG_FIELDS = [
    "source_group",
    "path",
    "parse_status",
    "sample",
    "sequence",
    "scene",
    "pair_start",
    "pair_end",
    "model_path",
    "dataset_path",
    "checkpoint_path",
    "lmd",
    "motion_xyz_quant_step",
    "bits_motion",
    "bits_prior_motion",
    "bits_total",
    "motion_zstd_bytes",
    "motion_zstd_mb",
    "psnr",
    "ssim",
    "lpips",
    "l1",
    "missing_required_fields",
]


SUMMARY_FIELDS = [
    "source_group",
    "path",
    "parse_status",
    "sample",
    "sequence",
    "method",
    "record_kind",
    "codec_mode",
    "lambda_or_q",
    "max_points_per_frame",
    "dummy_reference_images",
    "frame_count",
    "pframe_count",
    "iframe_count",
    "dfcgs_pframe_count",
    "fcgs_iframe_count",
    "raw_iframe_count",
    "total_size_mib",
    "avg_size_mib_per_frame",
    "psnr_avg",
    "psnr_min",
    "ssim_avg",
    "l1_avg",
    "lpips_avg_nonnull",
    "codec_psnr_avg",
    "codec_psnr_min",
    "codec_l1_avg",
    "quality_for_stage53",
    "rate_for_stage53",
    "stage53_candidate",
    "missing_required_fields",
    "notes",
]


def strip_ansi(text):
    return ANSI_RE.sub("", text)


def clean_log_text(text):
    lines = []
    for line in strip_ansi(text).splitlines():
        lines.append(LOG_PREFIX_RE.sub("", line))
    return "\n".join(lines)


def safe_float(value):
    if value is None or value == "":
        return None
    return float(value)


def infer_sample(path, payload=None):
    text = " ".join([str(path), json.dumps(payload or {}, sort_keys=True)[:2000]]).lower()
    for sample in ["meetroom", "driving", "robot", "n3dv"]:
        if sample in text:
            return sample
    parent = Path(path).parent.name
    if "_points" in parent:
        return parent.split("_points", 1)[0]
    return parent


def infer_sequence(path, payload=None):
    payload = payload or {}
    if payload.get("scene_list"):
        return str(payload["scene_list"][0])
    input_path = payload.get("input_path") or payload.get("cache_dir")
    if input_path:
        name = Path(str(input_path)).stem
        if name:
            return name
    parent = Path(path).parent.name
    return parent


def find_source_group(path):
    path = Path(path)
    matches = []
    for group, root, _pattern in SOURCE_PATTERNS:
        try:
            path.relative_to(root)
            matches.append((len(str(root)), group))
        except ValueError:
            continue
    if matches:
        return max(matches)[1]
    return "unknown"


def extract_balanced_json(text, marker):
    marker_pos = text.find(marker)
    if marker_pos < 0:
        return None
    start = text.find("{", marker_pos)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_pair(scene):
    if not scene:
        return None, None
    match = PAIR_RE.search(scene)
    if not match:
        return None, None
    return int(match.group("start")), int(match.group("end"))


def missing_fields(row, required_fields):
    missing = [field for field in required_fields if row.get(field) in (None, "")]
    return " ".join(missing)


def parse_dfcgs_log(path):
    row = {field: None for field in LOG_FIELDS}
    row.update({"source_group": find_source_group(path), "path": str(path), "parse_status": "ok"})
    try:
        text = clean_log_text(Path(path).read_text(encoding="utf-8", errors="replace"))
        args_blob = extract_balanced_json(text, "Training arguments:")
        args = json.loads(args_blob) if args_blob else {}
        scene = args.get("scene_list", [None])[0] if args.get("scene_list") else None
        pair_start, pair_end = parse_pair(scene)
        row.update({
            "sample": infer_sample(path, args),
            "sequence": infer_sequence(path, args),
            "scene": scene,
            "pair_start": pair_start,
            "pair_end": pair_end,
            "model_path": args.get("model_path"),
            "dataset_path": args.get("dataset_path"),
            "checkpoint_path": args.get("checkpoint_path"),
            "lmd": args.get("lmd"),
            "motion_xyz_quant_step": args.get("motion_xyz_quant_step"),
        })
        comp_match = re.search(r"Compression Result: (\{[^\n]+\})", text)
        if comp_match:
            comp = ast.literal_eval(comp_match.group(1))
            row.update({
                "bits_motion": safe_float(comp.get("bits_motion")),
                "bits_prior_motion": safe_float(comp.get("bits_prior_motion")),
                "bits_total": safe_float(comp.get("bits_total")),
            })
        zstd_match = ZSTD_RE.search(text)
        if zstd_match:
            row.update({
                "motion_zstd_bytes": int(zstd_match.group("bytes")),
                "motion_zstd_mb": safe_float(zstd_match.group("mb")),
            })
        eval_match = EVAL_RE.search(text)
        if eval_match:
            row.update({key: safe_float(eval_match.group(key)) for key in ["psnr", "ssim", "lpips", "l1"]})
    except Exception as exc:  # noqa: BLE001
        row["parse_status"] = f"error:{type(exc).__name__}:{exc}"
    row["missing_required_fields"] = missing_fields(row, ["sample", "sequence", "bits_total", "motion_zstd_bytes", "psnr", "ssim", "lpips", "l1"])
    return row


def nonnull_mean(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def classify_codec_mode(payload):
    method = str(payload.get("method", "")).lower()
    iframe_codec = payload.get("iframe_codec")
    if "fcgs per-frame" in method:
        return "fcgs_per_frame"
    if iframe_codec == "fcgs":
        return "fcgs_i_plus_dfcgs_p_gop2"
    if iframe_codec == "raw":
        return "raw_i_plus_dfcgs_p_gop2"
    if "d-fcgs" in method:
        return "dfcgs_gop2"
    return "unknown"


def stage53_flags(row):
    has_rate = row.get("avg_size_mib_per_frame") not in (None, "")
    has_quality = row.get("psnr_avg") not in (None, "") and row.get("dummy_reference_images") is False
    if row.get("record_kind") == "fcgs_lambda" and row.get("dummy_reference_images") is True:
        has_quality = False
    return {
        "quality_for_stage53": bool(has_quality),
        "rate_for_stage53": bool(has_rate),
        "stage53_candidate": bool(has_rate and has_quality),
    }


def summarize_frame_kinds(frames):
    counts = Counter(frame.get("kind") for frame in frames)
    return {
        "iframe_count": counts.get("fcgs_i", 0) + counts.get("raw_i", 0),
        "dfcgs_pframe_count": counts.get("dfcgs_p", 0),
        "fcgs_iframe_count": counts.get("fcgs_i", 0),
        "raw_iframe_count": counts.get("raw_i", 0),
        "lpips_avg_nonnull": nonnull_mean(frame.get("lpips") for frame in frames),
    }


def base_summary_row(path, payload):
    frames = payload.get("frames") or []
    kind_summary = summarize_frame_kinds(frames)
    return {
        "source_group": find_source_group(path),
        "path": str(path),
        "parse_status": "ok",
        "sample": infer_sample(path, payload),
        "sequence": infer_sequence(path, payload),
        "method": payload.get("method"),
        "codec_mode": classify_codec_mode(payload),
        "max_points_per_frame": payload.get("max_points_per_frame"),
        "dummy_reference_images": payload.get("dummy_reference_images"),
        **kind_summary,
    }


def parse_gop_summary(path, payload):
    row = {field: None for field in SUMMARY_FIELDS}
    row.update(base_summary_row(path, payload))
    summary = payload.get("summary") or {}
    codec_summary = payload.get("codec_summary") or {}
    row.update({
        "record_kind": "dfcgs_gop_summary",
        "lambda_or_q": payload.get("iframe_fcgs_lambda"),
        "frame_count": summary.get("frame_count"),
        "pframe_count": codec_summary.get("pframe_count"),
        "total_size_mib": summary.get("total_size_mib"),
        "avg_size_mib_per_frame": summary.get("avg_size_mib_per_frame"),
        "psnr_avg": summary.get("psnr_avg"),
        "psnr_min": summary.get("psnr_min"),
        "ssim_avg": summary.get("ssim_avg"),
        "l1_avg": summary.get("l1_avg"),
        "codec_psnr_avg": codec_summary.get("codec_pframe_psnr_avg"),
        "codec_psnr_min": codec_summary.get("codec_pframe_psnr_min"),
        "codec_l1_avg": codec_summary.get("codec_pframe_l1_avg"),
        "notes": "Full GOP=2 summary. Rate includes I-frame plus P-frame payloads, not Mono-DFCGS keyframe-anchor-only rate.",
    })
    row.update(stage53_flags(row))
    row["missing_required_fields"] = missing_fields(row, ["sample", "codec_mode", "frame_count", "avg_size_mib_per_frame", "psnr_avg"])
    return [row]


def parse_fcgs_summary(path, payload):
    rows = []
    lambda_results = payload.get("lambdas_results") or {}
    if not lambda_results:
        row = {field: None for field in SUMMARY_FIELDS}
        row.update(base_summary_row(path, payload))
        row.update({"record_kind": "fcgs_summary", "notes": "FCGS summary without lambdas_results."})
        row.update(stage53_flags(row))
        row["missing_required_fields"] = missing_fields(row, ["sample", "frame_count", "avg_size_mib_per_frame", "psnr_avg"])
        return [row]
    for lambda_value, result in sorted(lambda_results.items(), key=lambda item: float(item[0])):
        row = {field: None for field in SUMMARY_FIELDS}
        row.update(base_summary_row(path, payload))
        summary = result.get("summary") or {}
        frames = result.get("frames") or []
        frame_kinds = summarize_frame_kinds(frames)
        row.update(frame_kinds)
        row.update({
            "record_kind": "fcgs_lambda",
            "lambda_or_q": lambda_value,
            "frame_count": summary.get("frame_count") or payload.get("frame_count"),
            "pframe_count": 0,
            "total_size_mib": summary.get("total_size_mib"),
            "avg_size_mib_per_frame": summary.get("avg_size_mib_per_frame"),
            "psnr_avg": summary.get("psnr_avg"),
            "psnr_min": summary.get("psnr_min"),
            "ssim_avg": nonnull_mean(frame.get("ssim") for frame in frames),
            "l1_avg": nonnull_mean(frame.get("l1") for frame in frames),
            "codec_psnr_avg": summary.get("codec_psnr_avg"),
            "codec_psnr_min": summary.get("codec_psnr_min"),
            "codec_l1_avg": nonnull_mean(frame.get("codec_l1") for frame in frames),
            "notes": "FCGS per-frame summary. If dummy_reference_images=true, psnr_avg is not input-video quality; codec_psnr compares decoded render to raw uncompressed render.",
        })
        row.update(stage53_flags(row))
        row["missing_required_fields"] = missing_fields(row, ["sample", "frame_count", "avg_size_mib_per_frame", "psnr_avg"])
        rows.append(row)
    return rows


def parse_summary_json(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "lambdas_results" in payload:
            return parse_fcgs_summary(path, payload)
        return parse_gop_summary(path, payload)
    except Exception as exc:  # noqa: BLE001
        row = {field: None for field in SUMMARY_FIELDS}
        row.update({
            "source_group": find_source_group(path),
            "path": str(path),
            "parse_status": f"error:{type(exc).__name__}:{exc}",
            "sample": infer_sample(path),
            "missing_required_fields": "parse_error",
        })
        return [row]


def discover_sources():
    discovered = []
    availability = []
    for group, root, pattern in SOURCE_PATTERNS:
        exists = root.exists()
        paths = sorted(root.glob(pattern)) if exists else []
        availability.append({
            "source_group": group,
            "root": str(root),
            "pattern": pattern,
            "root_exists": exists,
            "file_count": len(paths),
        })
        for path in paths:
            discovered.append((group, path))
    return availability, discovered


def write_csv(rows, fields, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bool_count(rows, field):
    return sum(1 for row in rows if row.get(field) is True)


def counts_by(rows, field):
    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def sample_coverage(rows):
    coverage = defaultdict(lambda: defaultdict(int))
    for row in rows:
        coverage[str(row.get("source_group"))][str(row.get("sample"))] += 1
    return {group: dict(sorted(samples.items())) for group, samples in sorted(coverage.items())}


def build_summary(availability, log_rows, summary_rows):
    summary_candidates = [row for row in summary_rows if row.get("stage53_candidate") is True]
    return {
        "stage": 52,
        "mode": "FCGS/D-FCGS baseline ingestion preflight",
        "availability": availability,
        "log_records": len(log_rows),
        "log_parse_status": counts_by(log_rows, "parse_status"),
        "log_records_with_all_key_fields": sum(1 for row in log_rows if not row.get("missing_required_fields")),
        "summary_records": len(summary_rows),
        "summary_parse_status": counts_by(summary_rows, "parse_status"),
        "summary_records_with_rate": bool_count(summary_rows, "rate_for_stage53"),
        "summary_records_with_input_video_quality": bool_count(summary_rows, "quality_for_stage53"),
        "stage53_candidate_summary_records": len(summary_candidates),
        "summary_records_by_codec_mode": counts_by(summary_rows, "codec_mode"),
        "summary_records_by_source": counts_by(summary_rows, "source_group"),
        "log_records_by_source": counts_by(log_rows, "source_group"),
        "summary_sample_coverage": sample_coverage(summary_rows),
        "log_sample_coverage": sample_coverage(log_rows),
        "notes": [
            "D-FCGS log rows are single P-frame compression/decompression records; they are not complete video RD points without I-frame and aggregation accounting.",
            "GOP summary rows expose video-level MiB/frame and quality; their rate includes FCGS or raw I-frames plus D-FCGS P-frames, unlike Mono-DFCGS keyframe-anchor-only rate.",
            "Rows with dummy_reference_images=true are excluded from Stage53 input-video quality candidates; codec_psnr remains useful for codec fidelity diagnostics only.",
        ],
    }


def write_report(summary, log_csv, summary_csv, summary_json, path):
    lines = [
        "# Stage52 FCGS/D-FCGS Baseline Preflight",
        "",
        "## Outputs",
        f"- Log records CSV: `{log_csv}`",
        f"- Summary records CSV: `{summary_csv}`",
        f"- Summary JSON: `{summary_json}`",
        "",
        "## Availability",
        "| Source group | Root exists | File count | Pattern |",
        "|---|---:|---:|---|",
    ]
    for item in summary["availability"]:
        lines.append(
            f"| {item['source_group']} | {item['root_exists']} | {item['file_count']} | `{item['pattern']}` |"
        )
    lines.extend([
        "",
        "## Parse Summary",
        f"- D-FCGS log records: {summary['log_records']}",
        f"- D-FCGS log records with all key fields: {summary['log_records_with_all_key_fields']}",
        f"- Summary records: {summary['summary_records']}",
        f"- Summary records with rate: {summary['summary_records_with_rate']}",
        f"- Summary records with input-video quality: {summary['summary_records_with_input_video_quality']}",
        f"- Stage53 candidate summary records: {summary['stage53_candidate_summary_records']}",
        "",
        "## Codec Modes",
        "| Codec mode | Records |",
        "|---|---:|",
    ])
    for key, value in summary["summary_records_by_codec_mode"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "## Sample Coverage",
        "| Source group | Sample | Records |",
        "|---|---|---:|",
    ])
    for group, samples in summary["summary_sample_coverage"].items():
        for sample, count in samples.items():
            lines.append(f"| {group} | {sample} | {count} |")
    lines.extend([
        "",
        "## Stage53 Use Notes",
        "- Use GOP summary rows, not raw single-P-frame logs, for first baseline tables.",
        "- Treat rates as full FCGS/D-FCGS codec MiB/frame, not as Mono-DFCGS transmitted Gaussian-anchor MiB/frame.",
        "- Exclude `dummy_reference_images=true` rows from input-video PSNR/SSIM comparisons; keep their `codec_psnr` only as compression-fidelity diagnostics.",
        "- LPIPS is often per P-frame only or null for I-frames, so Stage53 should not assume a complete all-frame LPIPS curve.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    availability, discovered = discover_sources()
    log_rows = []
    summary_rows = []
    for _group, path in discovered:
        if path.suffix == ".log":
            log_rows.append(parse_dfcgs_log(path))
        elif path.suffix == ".json":
            summary_rows.extend(parse_summary_json(path))

    log_csv = args.summary_root / "stage52_dfcgs_log_records.csv"
    summary_csv = args.summary_root / "stage52_baseline_summary_records.csv"
    summary_json = args.summary_root / "stage52_fcgs_dfcgs_baseline_preflight_summary.json"
    report_md = args.summary_root / "stage52_fcgs_dfcgs_baseline_preflight_report.md"
    write_csv(log_rows, LOG_FIELDS, log_csv)
    write_csv(summary_rows, SUMMARY_FIELDS, summary_csv)
    summary = build_summary(availability, log_rows, summary_rows)
    summary.update({
        "log_csv": str(log_csv),
        "summary_csv": str(summary_csv),
        "report_md": str(report_md),
    })
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, log_csv, summary_csv, summary_json, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "log_records": summary["log_records"],
        "summary_records": summary["summary_records"],
        "stage53_candidate_summary_records": summary["stage53_candidate_summary_records"],
        "summary_records_by_codec_mode": summary["summary_records_by_codec_mode"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
