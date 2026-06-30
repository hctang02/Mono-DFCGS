import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE157_ROWS = REPO_ROOT / "experiments/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_rows.csv"
DEFAULT_TASK_MANIFEST = REPO_ROOT / "experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_tasks.csv"
DEFAULT_STAGE162_PACKAGE = REPO_ROOT / "experiments/stage162_keyframe_selector_protocol_source_audit/stage162_keyframe_selector_protocol_source_audit_package.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage163_davis_rgb_motion_selector_data"


ROW_FIELDS = [
    "task_id", "sequence", "gap", "codec", "left_index", "target_index", "right_index", "normalized_time",
    "segment_length", "feature_height", "feature_width",
    "rgb_mad_left_target", "rgb_mad_target_right", "rgb_mad_left_right", "rgb_mad_linear_interp_target",
    "rgb_mse_left_target", "rgb_mse_target_right", "rgb_mse_left_right", "rgb_mse_linear_interp_target",
    "edge_mad_left_target", "edge_mad_target_right", "edge_mad_left_right", "edge_mad_linear_interp_target",
    "hist_chi_left_target", "hist_chi_target_right", "hist_chi_left_right", "hist_chi_linear_interp_target",
    "temporal_mad_asymmetry", "temporal_edge_asymmetry", "rgb_motion_proxy_score",
    "stage158_psnr", "stage158_ssim", "stage158_ms_ssim", "stage158_lpips",
    "original_psnr", "original_ssim", "original_ms_ssim", "original_lpips",
    "delta_psnr_vs_original", "delta_ssim_vs_original", "delta_ms_ssim_vs_original", "delta_lpips_vs_original",
    "payload_bytes", "side_mib_per_intermediate", "direct_total_mib_per_frame_ref",
    "label_low_psnr_lt26", "label_high_lpips_gt022", "label_high_payload_gt220k",
    "inference_feature_columns", "offline_label_columns",
]

SUMMARY_FIELDS = [
    "sequence", "gap", "task_count", "mean_rgb_motion_proxy_score", "mean_rgb_mad_left_right", "mean_rgb_mad_linear_interp_target",
    "mean_edge_mad_left_right", "mean_hist_chi_left_right", "mean_stage158_psnr", "mean_stage158_lpips",
    "mean_payload_bytes", "low_psnr_count", "high_lpips_count", "high_payload_count",
]

COMPLIANCE_FIELDS = ["column_group", "columns", "source", "inference_allowed", "feedforward_status", "notes"]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_rgb(path, height, width, cache):
    key = (str(path), int(height), int(width))
    if key in cache:
        return cache[key]
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    out = image.astype(np.float32) / 255.0
    cache[key] = out
    return out


def gray(image):
    return cv2.cvtColor((image * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def edge_mag(image):
    g = gray(image)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(sx * sx + sy * sy)


def hist_gray(image, bins):
    hist = cv2.calcHist([(gray(image) * 255.0).round().astype(np.uint8)], [0], None, [int(bins)], [0, 256]).astype(np.float64).reshape(-1)
    total = float(hist.sum())
    if total > 0.0:
        hist /= total
    return hist


def pair_stats(a, b, bins):
    diff = a - b
    mse = float(np.mean(diff * diff))
    mad = float(np.mean(np.abs(diff)))
    ea = edge_mag(a)
    eb = edge_mag(b)
    edge_mad = float(np.mean(np.abs(ea - eb)))
    ha = hist_gray(a, bins)
    hb = hist_gray(b, bins)
    hist_chi = float(np.sum((ha - hb) ** 2 / (ha + hb + 1e-12)))
    return {
        "mad": mad,
        "mse": mse,
        "edge_mad": edge_mad,
        "hist_chi": hist_chi,
    }


def parse_task_manifest(path):
    out = {}
    for row in read_csv(path):
        out[row["task_id"]] = row
    return out


def mean(values):
    vals = [float(v) for v in values if v not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["sequence"], int(row["gap"]))].append(row)
    out = []
    for (sequence, gap), group in sorted(groups.items()):
        out.append({
            "sequence": sequence,
            "gap": gap,
            "task_count": len(group),
            "mean_rgb_motion_proxy_score": mean(row["rgb_motion_proxy_score"] for row in group),
            "mean_rgb_mad_left_right": mean(row["rgb_mad_left_right"] for row in group),
            "mean_rgb_mad_linear_interp_target": mean(row["rgb_mad_linear_interp_target"] for row in group),
            "mean_edge_mad_left_right": mean(row["edge_mad_left_right"] for row in group),
            "mean_hist_chi_left_right": mean(row["hist_chi_left_right"] for row in group),
            "mean_stage158_psnr": mean(row["stage158_psnr"] for row in group),
            "mean_stage158_lpips": mean(row["stage158_lpips"] for row in group),
            "mean_payload_bytes": mean(row["payload_bytes"] for row in group),
            "low_psnr_count": sum(int(row["label_low_psnr_lt26"]) for row in group),
            "high_lpips_count": sum(int(row["label_high_lpips_gt022"]) for row in group),
            "high_payload_count": sum(int(row["label_high_payload_gt220k"]) for row in group),
        })
    return out


def compliance_rows():
    return [
        {
            "column_group": "deployable_encoder_features",
            "columns": "rgb_*, edge_*, hist_*, temporal_*, normalized_time, segment_length",
            "source": "DAVIS/input RGB frames available at encoder",
            "inference_allowed": "yes",
            "feedforward_status": "feed-forward for offline encoding; online mode would need declared lookahead",
            "notes": "These are the only columns intended as selector inference features in the first heuristic selector.",
        },
        {
            "column_group": "offline_stage158_labels",
            "columns": "stage158_*, original_*, delta_*, payload_bytes, side_mib_per_intermediate, direct_total_mib_per_frame_ref, label_*",
            "source": "Stage157/158 rendered validation rows",
            "inference_allowed": "no",
            "feedforward_status": "offline labels only",
            "notes": "Use for training/evaluation or difficulty diagnostics, not as deployable selector inputs.",
        },
        {
            "column_group": "schedule_metadata_future",
            "columns": "keyframe indices or segment lengths, not present in this row-level package",
            "source": "selector output",
            "inference_allowed": "transmitted_output",
            "feedforward_status": "decoder consumes transmitted metadata only",
            "notes": "Stage164 should add candidate schedule rows and count metadata bits.",
        },
    ]


def build_rows(stage157_rows, task_by_id, args):
    cache = {}
    out = []
    inference_cols = [
        "gap", "normalized_time", "segment_length",
        "rgb_mad_left_target", "rgb_mad_target_right", "rgb_mad_left_right", "rgb_mad_linear_interp_target",
        "rgb_mse_left_target", "rgb_mse_target_right", "rgb_mse_left_right", "rgb_mse_linear_interp_target",
        "edge_mad_left_target", "edge_mad_target_right", "edge_mad_left_right", "edge_mad_linear_interp_target",
        "hist_chi_left_target", "hist_chi_target_right", "hist_chi_left_right", "hist_chi_linear_interp_target",
        "temporal_mad_asymmetry", "temporal_edge_asymmetry", "rgb_motion_proxy_score",
    ]
    label_cols = [
        "stage158_psnr", "stage158_ssim", "stage158_ms_ssim", "stage158_lpips",
        "original_psnr", "original_ssim", "original_ms_ssim", "original_lpips",
        "delta_psnr_vs_original", "delta_ssim_vs_original", "delta_ms_ssim_vs_original", "delta_lpips_vs_original",
        "payload_bytes", "side_mib_per_intermediate", "direct_total_mib_per_frame_ref",
        "label_low_psnr_lt26", "label_high_lpips_gt022", "label_high_payload_gt220k",
    ]
    for row in stage157_rows:
        task = task_by_id[row["task_id"]]
        left = load_rgb(task["left_rgb_path"], args.feature_height, args.feature_width, cache)
        target = load_rgb(task["target_rgb_path"], args.feature_height, args.feature_width, cache)
        right = load_rgb(task["right_rgb_path"], args.feature_height, args.feature_width, cache)
        t = float(row["normalized_time"])
        linear = left * (1.0 - t) + right * t
        lt = pair_stats(left, target, args.hist_bins)
        tr = pair_stats(target, right, args.hist_bins)
        lr = pair_stats(left, right, args.hist_bins)
        it = pair_stats(linear, target, args.hist_bins)
        temporal_mad_asymmetry = abs(lt["mad"] - tr["mad"])
        temporal_edge_asymmetry = abs(lt["edge_mad"] - tr["edge_mad"])
        # Cheap scalar for ranking likely hard segments; raw columns remain available for learning.
        proxy = lr["mad"] + it["mad"] + 0.25 * lr["edge_mad"] + 0.01 * lr["hist_chi"] + 0.5 * temporal_mad_asymmetry
        psnr = float(row["psnr"])
        lpips = float(row["lpips"])
        payload = float(row["payload_bytes"])
        item = {
            "task_id": row["task_id"],
            "sequence": row["sequence"],
            "gap": int(row["gap"]),
            "codec": row["codec"],
            "left_index": int(task["left_index"]),
            "target_index": int(task["target_index"]),
            "right_index": int(task["right_index"]),
            "normalized_time": t,
            "segment_length": int(task["right_index"]) - int(task["left_index"]),
            "feature_height": int(args.feature_height),
            "feature_width": int(args.feature_width),
            "rgb_mad_left_target": lt["mad"],
            "rgb_mad_target_right": tr["mad"],
            "rgb_mad_left_right": lr["mad"],
            "rgb_mad_linear_interp_target": it["mad"],
            "rgb_mse_left_target": lt["mse"],
            "rgb_mse_target_right": tr["mse"],
            "rgb_mse_left_right": lr["mse"],
            "rgb_mse_linear_interp_target": it["mse"],
            "edge_mad_left_target": lt["edge_mad"],
            "edge_mad_target_right": tr["edge_mad"],
            "edge_mad_left_right": lr["edge_mad"],
            "edge_mad_linear_interp_target": it["edge_mad"],
            "hist_chi_left_target": lt["hist_chi"],
            "hist_chi_target_right": tr["hist_chi"],
            "hist_chi_left_right": lr["hist_chi"],
            "hist_chi_linear_interp_target": it["hist_chi"],
            "temporal_mad_asymmetry": temporal_mad_asymmetry,
            "temporal_edge_asymmetry": temporal_edge_asymmetry,
            "rgb_motion_proxy_score": proxy,
            "stage158_psnr": psnr,
            "stage158_ssim": float(row["ssim"]),
            "stage158_ms_ssim": float(row["ms_ssim"]),
            "stage158_lpips": lpips,
            "original_psnr": float(row["original_psnr"]),
            "original_ssim": float(row["original_ssim"]),
            "original_ms_ssim": float(row["original_ms_ssim"]),
            "original_lpips": float(row["original_lpips"]),
            "delta_psnr_vs_original": float(row["delta_psnr_vs_original"]),
            "delta_ssim_vs_original": float(row["delta_ssim_vs_original"]),
            "delta_ms_ssim_vs_original": float(row["delta_ms_ssim_vs_original"]),
            "delta_lpips_vs_original": float(row["delta_lpips_vs_original"]),
            "payload_bytes": payload,
            "side_mib_per_intermediate": float(row["side_mib_per_intermediate"]),
            "direct_total_mib_per_frame_ref": float(row["direct_total_mib_per_frame_ref"]),
            "label_low_psnr_lt26": int(psnr < 26.0),
            "label_high_lpips_gt022": int(lpips > 0.22),
            "label_high_payload_gt220k": int(payload > 220000.0),
            "inference_feature_columns": " ".join(inference_cols),
            "offline_label_columns": " ".join(label_cols),
        }
        out.append(item)
    return out


def write_report(package, rows, summary_rows, path):
    lines = [
        "# Stage163 DAVIS RGB/Motion Selector Data",
        "",
        "## Scope",
        "",
        f"- Rows: `{len(rows)}` Stage157/158 sampled tasks.",
        f"- Feature resolution: `{package['feature_width']}x{package['feature_height']}`.",
        "- Inference features are derived from input RGB only.",
        "- Stage158 metrics/payloads are attached as offline labels, not selector inference inputs.",
        "",
        "## Outputs",
        "",
        f"- Rows CSV: `{package['rows_csv']}`",
        f"- Summary CSV: `{package['summary_csv']}`",
        f"- Compliance CSV: `{package['compliance_csv']}`",
        "",
        "## Compliance",
        "",
        "| group | source | inference | feed-forward | notes |",
        "|---|---|---|---|---|",
    ]
    for row in package["compliance_rows"]:
        lines.append(f"| {row['column_group']} | {row['source']} | {row['inference_allowed']} | {row['feedforward_status']} | {row['notes']} |")
    lines.extend([
        "",
        "## Sequence/Gap Summary",
        "",
        "| sequence | gap | tasks | proxy score | RGB LR MAD | interp-target MAD | Stage158 PSNR | Stage158 LPIPS | payload bytes | low PSNR | high LPIPS | high payload |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        lines.append(
            f"| {row['sequence']} | {row['gap']} | {row['task_count']} | {float(row['mean_rgb_motion_proxy_score']):.6f} | "
            f"{float(row['mean_rgb_mad_left_right']):.6f} | {float(row['mean_rgb_mad_linear_interp_target']):.6f} | "
            f"{float(row['mean_stage158_psnr']):.6f} | {float(row['mean_stage158_lpips']):.6f} | {float(row['mean_payload_bytes']):.3f} | "
            f"{row['low_psnr_count']} | {row['high_lpips_count']} | {row['high_payload_count']} |"
        )
    lines.extend([
        "",
        "## Next Stage",
        "",
        "Stage164 should turn these row-level features into candidate keyframe schedules and evaluate a first RGB/motion heuristic against uniform gap4/gap8 under Stage158 accounting.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage157_rows", type=Path, default=DEFAULT_STAGE157_ROWS)
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--stage162_package", type=Path, default=DEFAULT_STAGE162_PACKAGE)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--feature_height", type=int, default=256)
    parser.add_argument("--feature_width", type=int, default=448)
    parser.add_argument("--hist_bins", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage162 = read_json(args.stage162_package)
    stage157_rows = read_csv(args.stage157_rows)
    task_by_id = parse_task_manifest(args.task_manifest)
    rows = build_rows(stage157_rows, task_by_id, args)
    summary_rows = summarize(rows)
    compliance = compliance_rows()
    rows_csv = args.summary_root / "stage163_davis_rgb_motion_selector_feature_rows.csv"
    summary_csv = args.summary_root / "stage163_davis_rgb_motion_selector_summary.csv"
    compliance_csv = args.summary_root / "stage163_davis_rgb_motion_selector_compliance.csv"
    package_json = args.summary_root / "stage163_davis_rgb_motion_selector_data_package.json"
    report_md = args.summary_root / "stage163_davis_rgb_motion_selector_data_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(compliance, compliance_csv, COMPLIANCE_FIELDS)
    package = {
        "stage": 163,
        "status": "davis_rgb_motion_selector_data_packaged",
        "source_stage157_rows": str(args.stage157_rows),
        "source_stage162_protocol": str(args.stage162_package),
        "fixed_middle_recovery_policy": stage162["fixed_middle_recovery_policy"],
        "row_count": len(rows),
        "sequence_gap_count": len(summary_rows),
        "feature_height": int(args.feature_height),
        "feature_width": int(args.feature_width),
        "hist_bins": int(args.hist_bins),
        "inference_feature_source": "input RGB frames only",
        "offline_label_source": "Stage157/158 rendered validation rows",
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "compliance_csv": str(compliance_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "compliance_rows": compliance,
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, rows, summary_rows, report_md)
    print(json.dumps({"package": str(package_json), "report": str(report_md), "row_count": len(rows)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
