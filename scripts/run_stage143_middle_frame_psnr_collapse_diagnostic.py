import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments/stage61_davis_anchor_export_data_full/stage61_davis_anchor_export_manifest.csv"
DEFAULT_ADAPTER = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors")
DEFAULT_STAGE142_TARGETS = REPO_ROOT / "experiments/stage142_middle_frame_protocol_alignment_audit/stage142_middle_frame_targets.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage143_middle_frame_psnr_collapse_diagnostic"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, uniform_dequantize, uniform_quantize, unflatten_static_anchor  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import uniform_indices  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import render_anchor  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import (  # noqa: E402
    frame_psnr,
    frame_ssim,
    load_adapter,
    load_rgb_numpy,
    tensor_to_numpy_rgb,
)
from scripts.run_stage27_anchor_available_selector_rd import selected_records  # noqa: E402
from scripts.run_stage66_davis_feedforward_selector_dataset import load_sequence_anchors, read_gap1_manifest  # noqa: E402


ROW_FIELDS = [
    "sample",
    "codec",
    "bits",
    "reference_gap",
    "method",
    "total_frames",
    "keyframe_count",
    "middle_count",
    "anchor_mib_per_frame_with_metadata",
    "all_psnr",
    "middle_psnr",
    "given_psnr",
    "all_ssim",
    "middle_ssim",
    "given_ssim",
]

SUMMARY_FIELDS = [
    "codec",
    "bits",
    "reference_gap",
    "method",
    "sequence_count",
    "frame_count",
    "middle_frame_count",
    "mean_anchor_mib_per_frame_with_metadata",
    "mean_all_psnr",
    "mean_middle_psnr",
    "mean_given_psnr",
    "mean_all_ssim",
    "mean_middle_ssim",
    "mean_given_ssim",
    "delta_middle_vs_float32_same_method",
    "delta_middle_vs_linear_same_codec",
    "gap_to_stage75_middle_target",
]

FINDING_FIELDS = ["severity", "finding", "evidence", "required_action"]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def codec_bits(codec):
    if codec == "float32":
        return 32
    if codec.startswith("q"):
        return int(codec[1:])
    raise ValueError(codec)


def quantize_anchor(anchor, codec):
    if codec == "float32":
        return anchor
    bits = codec_bits(codec)
    attrs = flatten_static_anchor(anchor)
    q, mins, scales = uniform_quantize(attrs, bits=bits)
    return unflatten_static_anchor(uniform_dequantize(q, mins, scales))


def quantize_map(anchor_map, codec):
    return {idx: quantize_anchor(anchor, codec) for idx, anchor in anchor_map.items()}


def payload_mib_per_frame(anchor, keyframe_count, total_frames, codec):
    bits = codec_bits(codec)
    values = int(flatten_static_anchor(anchor).numel())
    metadata_bytes = 13 * 2 * 2
    anchor_bytes = values * bits / 8.0 + metadata_bytes
    return anchor_bytes * keyframe_count / max(total_frames, 1) / (1024.0 * 1024.0)


def summarize_records(records, indices):
    items = [records[idx] for idx in indices]
    if not items:
        return {"psnr": None, "ssim": None, "count": 0}
    return {
        "psnr": float(np.mean([row["psnr"] for row in items])),
        "ssim": float(np.mean([row["ssim"] for row in items])),
        "count": len(items),
    }


def dense_direct_metrics(qmap, frame_files, selected, opt, background):
    total_frames = len(frame_files)
    keyframes = set(selected)
    height, width = opt.image_height, opt.image_width
    records = {}
    with torch.no_grad():
        for idx in range(total_frames):
            ref = load_rgb_numpy(frame_files[idx], height, width)
            pred = tensor_to_numpy_rgb(render_anchor(qmap[idx], background, opt))
            records[idx] = {"psnr": frame_psnr(ref, pred), "ssim": frame_ssim(ref, pred)}
    all_indices = list(range(total_frames))
    middle_indices = [idx for idx in all_indices if idx not in keyframes]
    given_indices = [idx for idx in all_indices if idx in keyframes]
    all_summary = summarize_records(records, all_indices)
    middle_summary = summarize_records(records, middle_indices)
    given_summary = summarize_records(records, given_indices)
    return {
        "all": {"psnr_avg": all_summary["psnr"], "ssim_avg": all_summary["ssim"], "count": all_summary["count"]},
        "middle_only": {"psnr_avg": middle_summary["psnr"], "ssim_avg": middle_summary["ssim"], "count": middle_summary["count"]},
        "given_keyframes": {"psnr_avg": given_summary["psnr"], "ssim_avg": given_summary["ssim"], "count": given_summary["count"]},
    }


def add_metric_row(rows, sample, codec, gap, method, total_frames, keyframe_count, rate, metrics):
    middle_count = total_frames - keyframe_count
    rows.append({
        "sample": sample,
        "codec": codec,
        "bits": codec_bits(codec),
        "reference_gap": gap,
        "method": method,
        "total_frames": total_frames,
        "keyframe_count": keyframe_count,
        "middle_count": middle_count,
        "anchor_mib_per_frame_with_metadata": rate,
        "all_psnr": metrics["all"]["psnr_avg"],
        "middle_psnr": metrics["middle_only"]["psnr_avg"],
        "given_psnr": metrics["given_keyframes"]["psnr_avg"],
        "all_ssim": metrics["all"]["ssim_avg"],
        "middle_ssim": metrics["middle_only"]["ssim_avg"],
        "given_ssim": metrics["given_keyframes"]["ssim_avg"],
    })


def evaluate_sequence(sample, anchor_map, rgb_paths, codecs, gaps, model, opt, background):
    indices = sorted(anchor_map)
    frame_files = [rgb_paths[idx] for idx in indices]
    total_frames = len(indices)
    rows = []
    first_anchor = anchor_map[indices[0]]
    for codec in codecs:
        print(f"=== Stage143 {sample} codec={codec} ===", flush=True)
        qmap = quantize_map(anchor_map, codec)
        for gap in gaps:
            selected = uniform_indices(total_frames, gap)
            keyframe_count = len(selected)
            rate = payload_mib_per_frame(first_anchor, keyframe_count, total_frames, codec)
            dense_metrics = dense_direct_metrics(qmap, frame_files, selected, opt, background)
            add_metric_row(rows, sample, codec, gap, "dense_direct", total_frames, keyframe_count, rate, dense_metrics)
            pred_metrics = selected_records(selected, qmap, model, frame_files, opt, background)
            add_metric_row(rows, sample, codec, gap, "linear", total_frames, keyframe_count, rate, pred_metrics["linear"])
            add_metric_row(rows, sample, codec, gap, "adapter", total_frames, keyframe_count, rate, pred_metrics["adapter"])
            if background.device.type == "cuda":
                torch.cuda.empty_cache()
        del qmap
        if background.device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def weighted_mean(items, value_key, weight_key):
    weights = np.array([float(row[weight_key]) for row in items], dtype=np.float64)
    values = np.array([float(row[value_key]) for row in items], dtype=np.float64)
    return float(np.average(values, weights=weights))


def target_for_gap(target_rows, gap):
    # Stage78 maps gap4 to the Middle-4 local gap5 reference and gap8 to the local gap8 reference.
    if int(gap) == 4:
        ref_gap = 5
    elif int(gap) == 8:
        ref_gap = 8
    else:
        return None
    for row in target_rows:
        if int(float(row["local_gap"])) == ref_gap:
            return float(row["local_corrected_middle_psnr"])
    return None


def build_summary(rows, target_rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["codec"], int(row["reference_gap"]), row["method"])].append(row)
    out = []
    base = {}
    linear = {}
    for (codec, gap, method), items in sorted(grouped.items(), key=lambda item: (codec_bits(item[0][0]), item[0][1], item[0][2])):
        summary = {
            "codec": codec,
            "bits": codec_bits(codec),
            "reference_gap": gap,
            "method": method,
            "sequence_count": len(items),
            "frame_count": int(sum(int(row["total_frames"]) for row in items)),
            "middle_frame_count": int(sum(int(row["middle_count"]) for row in items)),
            "mean_anchor_mib_per_frame_with_metadata": weighted_mean(items, "anchor_mib_per_frame_with_metadata", "total_frames"),
            "mean_all_psnr": weighted_mean(items, "all_psnr", "total_frames"),
            "mean_middle_psnr": weighted_mean(items, "middle_psnr", "middle_count"),
            "mean_given_psnr": weighted_mean(items, "given_psnr", "keyframe_count"),
            "mean_all_ssim": weighted_mean(items, "all_ssim", "total_frames"),
            "mean_middle_ssim": weighted_mean(items, "middle_ssim", "middle_count"),
            "mean_given_ssim": weighted_mean(items, "given_ssim", "keyframe_count"),
        }
        if codec == "float32":
            base[(gap, method)] = summary
        if method == "linear":
            linear[(codec, gap)] = summary
        out.append(summary)
    for row in out:
        ref_float = base.get((row["reference_gap"], row["method"]))
        ref_linear = linear.get((row["codec"], row["reference_gap"]))
        target = target_for_gap(target_rows, row["reference_gap"])
        row["delta_middle_vs_float32_same_method"] = row["mean_middle_psnr"] - ref_float["mean_middle_psnr"] if ref_float else None
        row["delta_middle_vs_linear_same_codec"] = row["mean_middle_psnr"] - ref_linear["mean_middle_psnr"] if ref_linear else None
        row["gap_to_stage75_middle_target"] = row["mean_middle_psnr"] - target if target is not None else None
    return out


def lookup(summary_rows, codec, gap, method):
    for row in summary_rows:
        if row["codec"] == codec and int(row["reference_gap"]) == int(gap) and row["method"] == method:
            return row
    return None


def build_findings(summary_rows):
    findings = []
    for gap in [4, 8]:
        dense = lookup(summary_rows, "float32", gap, "dense_direct")
        adapter = lookup(summary_rows, "float32", gap, "adapter")
        q12_adapter = lookup(summary_rows, "q12", gap, "adapter")
        q16_adapter = lookup(summary_rows, "q16", gap, "adapter")
        if dense and adapter:
            model_gap = dense["mean_middle_psnr"] - adapter["mean_middle_psnr"]
            findings.append({
                "severity": "critical" if model_gap > 3.0 else "high",
                "finding": f"gap{gap} float32 dense-direct ceiling greatly exceeds adapter prediction" if model_gap > 3.0 else f"gap{gap} dense/direct ceiling is close to adapter",
                "evidence": f"dense_direct={dense['mean_middle_psnr']}, adapter={adapter['mean_middle_psnr']}, dense-adapter={model_gap}",
                "required_action": "Prioritize dynamic adapter training and/or rate-counted motion/residual side-info." if model_gap > 3.0 else "Quantization or protocol may be the dominant bottleneck; inspect high-rate rows.",
            })
        if q12_adapter and q16_adapter:
            q_gap = q16_adapter["mean_middle_psnr"] - q12_adapter["mean_middle_psnr"]
            findings.append({
                "severity": "high" if q_gap > 0.5 else "medium",
                "finding": f"gap{gap} q16-vs-q12 quantization sensitivity",
                "evidence": f"q16_adapter={q16_adapter['mean_middle_psnr']}, q12_adapter={q12_adapter['mean_middle_psnr']}, q16-q12={q_gap}",
                "required_action": "If q_gap is small, raising keyframe quantization alone will not recover paper-level middle PSNR.",
            })
    return findings


def write_report(summary_rows, findings, package, path):
    lines = [
        "# Stage143 Middle-Frame PSNR Collapse Diagnostic",
        "",
        "## Key Summary",
        "",
        "| codec | gap | method | rate | middle PSNR | given PSNR | delta vs linear | gap to Stage75 target |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["method"] not in {"dense_direct", "adapter", "linear"}:
            continue
        if row["codec"] not in {"float32", "q12", "q16"}:
            continue
        if int(row["reference_gap"]) not in {4, 8}:
            continue
        lines.append(
            f"| {row['codec']} | {row['reference_gap']} | {row['method']} | {row['mean_anchor_mib_per_frame_with_metadata']} | {row['mean_middle_psnr']} | {row['mean_given_psnr']} | {row['delta_middle_vs_linear_same_codec']} | {row['gap_to_stage75_middle_target']} |"
        )
    lines.extend([
        "",
        "## Findings",
        "",
        "| severity | finding | evidence | required action |",
        "|---|---|---|---|",
    ])
    for row in findings:
        lines.append(f"| {row['severity']} | {row['finding']} | {row['evidence']} | {row['required_action']} |")
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- rows CSV: `{package['rows_csv']}`",
        f"- summary CSV: `{package['summary_csv']}`",
        f"- findings CSV: `{package['findings_csv']}`",
        f"- summary JSON: `{package['summary_json']}`",
        f"- package JSON: `{package['package_json']}`",
        f"- report Markdown: `{package['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--stage142_targets", type=Path, default=DEFAULT_STAGE142_TARGETS)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sequences", nargs="+", default=["bmx-trees", "car-shadow", "goat", "soapbox"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--codecs", nargs="+", default=["float32", "q8", "q12", "q16"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    model = load_adapter(args.adapter, args.hidden_dim, device)
    rows_by_sequence = read_gap1_manifest(args.manifest, [args.split])
    rows = []
    for sequence in args.sequences:
        key = ("DAVIS", args.split, sequence)
        if key not in rows_by_sequence:
            raise RuntimeError(f"Missing gap1 anchors for {key}")
        sample = f"DAVIS/{args.split}/{sequence}"
        print(f"=== Stage143 loading {sample} ===", flush=True)
        _indices, anchor_map, _qmap, _attrs_map, rgb_paths = load_sequence_anchors(rows_by_sequence[key], device, quant_bits=0)
        rows.extend(evaluate_sequence(sample, anchor_map, rgb_paths, args.codecs, args.gaps, model, opt, background))
        del anchor_map
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    target_rows = read_csv(args.stage142_targets)
    summary_rows = build_summary(rows, target_rows)
    findings = build_findings(summary_rows)
    rows_csv = args.summary_root / "stage143_middle_frame_psnr_collapse_rows.csv"
    summary_csv = args.summary_root / "stage143_middle_frame_psnr_collapse_summary.csv"
    findings_csv = args.summary_root / "stage143_middle_frame_psnr_collapse_findings.csv"
    summary_json = args.summary_root / "stage143_middle_frame_psnr_collapse_summary.json"
    package_json = args.summary_root / "stage143_middle_frame_psnr_collapse_package.json"
    report_md = args.summary_root / "stage143_middle_frame_psnr_collapse_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(findings, findings_csv, FINDING_FIELDS)
    summary = {
        "stage": 143,
        "mode": "middle-frame PSNR collapse diagnostic",
        "manifest": str(args.manifest),
        "adapter": str(args.adapter),
        "stage142_targets": str(args.stage142_targets),
        "split": args.split,
        "sequences": args.sequences,
        "gaps": args.gaps,
        "codecs": args.codecs,
        "row_count": len(rows),
        "summary_row_count": len(summary_rows),
        "findings": findings,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "findings_csv": str(findings_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package = {
        "stage": 143,
        "mode": "middle-frame PSNR collapse diagnostic",
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "findings_csv": str(findings_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "row_count": len(rows),
        "critical_findings": [row for row in findings if row["severity"] == "critical"],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary_rows, findings, package, report_md)
    print(json.dumps({"package": str(package_json), "row_count": len(rows), "critical_findings": package["critical_findings"]}, indent=2))


if __name__ == "__main__":
    main()
