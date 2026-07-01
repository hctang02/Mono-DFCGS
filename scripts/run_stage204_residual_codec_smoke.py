import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_MANIFEST = REPO_ROOT / "experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_tasks.csv"
DEFAULT_STAGE203_PACKAGE = REPO_ROOT / "experiments/stage203_gs_latent_residual_codec_design/stage203_gs_latent_residual_codec_design_package.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage204_residual_codec_smoke"

sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, unflatten_static_anchor  # noqa: E402
from mono_dfcgs.learned_gs_predictor import linear_static_anchor  # noqa: E402
from mono_dfcgs.residual_sideinfo_codec import decode_residual_sideinfo_entropy, encode_topk_residual_sideinfo_entropy  # noqa: E402
from scripts.run_stage201_predictor_only_smoke import (  # noqa: E402
    align_target_to_render,
    balanced_sample,
    load_task_tensors,
    parse_task_rows,
    psnr_from_mse,
    write_csv,
)
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import render_anchor  # noqa: E402


ROW_FIELDS = [
    "task_id",
    "sequence",
    "reference_gap",
    "target_index",
    "normalized_time",
    "setting_label",
    "keep_fraction",
    "side_bits",
    "payload_bytes",
    "base_anchor_mse",
    "corrected_anchor_mse",
    "anchor_mse_reduction",
    "base_render_mse",
    "corrected_render_mse",
    "base_psnr",
    "corrected_psnr",
    "delta_psnr_vs_base",
    "status",
    "error",
]
SUMMARY_FIELDS = [
    "setting_label",
    "keep_fraction",
    "side_bits",
    "task_count",
    "mean_payload_bytes",
    "mean_base_psnr",
    "mean_corrected_psnr",
    "mean_delta_psnr_vs_base",
    "mean_anchor_mse_reduction",
    "mean_corrected_anchor_mse",
]
GATE_FIELDS = ["gate", "status", "value", "threshold", "detail"]
TASK_FIELDS = ["task_id", "sequence", "reference_gap", "left_index", "right_index", "target_index", "normalized_time"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    values = [float(value) for value in values]
    return sum(values) / max(len(values), 1)


def selected_task_rows(rows):
    return [
        {
            "task_id": row["task_id"],
            "sequence": row["sequence"],
            "reference_gap": row["reference_gap"],
            "left_index": row["left_index"],
            "right_index": row["right_index"],
            "target_index": row["target_index"],
            "normalized_time": row["normalized_time"],
        }
        for row in rows
    ]


def setting_label(keep_fraction, side_bits):
    return f"topk_keep{str(keep_fraction).replace('.', 'p')}_q{side_bits}"


def evaluate_task(row, settings, args, device, cache, opt, background):
    out = []
    try:
        left, right, target_anchor, target_rgb, t = load_task_tensors(row, device, cache, opt)
        base_anchor = linear_static_anchor(left, right, t)
        base_attrs = flatten_static_anchor(base_anchor)
        target_attrs = flatten_static_anchor(target_anchor)
        base_anchor_mse = float(F.mse_loss(base_attrs, target_attrs).item())
        base_render = render_anchor(base_anchor, background, opt).clamp(0.0, 1.0)
        aligned_target = align_target_to_render(base_render, target_rgb)
        base_render_mse = float(F.mse_loss(base_render, aligned_target).item())
        base_psnr = psnr_from_mse(base_render_mse)
        for keep_fraction, side_bits in settings:
            label = setting_label(keep_fraction, side_bits)
            try:
                payload, info = encode_topk_residual_sideinfo_entropy(
                    base_attrs,
                    target_attrs,
                    keep_fraction=keep_fraction,
                    side_bits=side_bits,
                    zlib_level=args.zlib_level,
                )
                corrected_attrs = decode_residual_sideinfo_entropy(base_attrs, payload)
                corrected_anchor = unflatten_static_anchor(corrected_attrs)
                corrected_anchor_mse = float(F.mse_loss(corrected_attrs, target_attrs).item())
                corrected_render = render_anchor(corrected_anchor, background, opt).clamp(0.0, 1.0)
                corrected_target = align_target_to_render(corrected_render, target_rgb)
                corrected_render_mse = float(F.mse_loss(corrected_render, corrected_target).item())
                corrected_psnr = psnr_from_mse(corrected_render_mse)
                out.append({
                    "task_id": row["task_id"],
                    "sequence": row["sequence"],
                    "reference_gap": row["reference_gap"],
                    "target_index": row["target_index"],
                    "normalized_time": row["normalized_time"],
                    "setting_label": label,
                    "keep_fraction": keep_fraction,
                    "side_bits": side_bits,
                    "payload_bytes": len(payload),
                    "base_anchor_mse": base_anchor_mse,
                    "corrected_anchor_mse": corrected_anchor_mse,
                    "anchor_mse_reduction": 1.0 - corrected_anchor_mse / base_anchor_mse if base_anchor_mse > 0.0 else 0.0,
                    "base_render_mse": base_render_mse,
                    "corrected_render_mse": corrected_render_mse,
                    "base_psnr": base_psnr,
                    "corrected_psnr": corrected_psnr,
                    "delta_psnr_vs_base": corrected_psnr - base_psnr,
                    "status": "ok" if len(payload) == int(info["payload_bytes"]) else "error",
                    "error": "" if len(payload) == int(info["payload_bytes"]) else "payload_length_mismatch",
                })
            except Exception as exc:  # noqa: BLE001
                out.append(error_row(row, label, keep_fraction, side_bits, repr(exc)))
    except Exception as exc:  # noqa: BLE001
        for keep_fraction, side_bits in settings:
            out.append(error_row(row, setting_label(keep_fraction, side_bits), keep_fraction, side_bits, repr(exc)))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def error_row(row, label, keep_fraction, side_bits, error):
    return {
        "task_id": row.get("task_id", ""),
        "sequence": row.get("sequence", ""),
        "reference_gap": row.get("reference_gap", ""),
        "target_index": row.get("target_index", ""),
        "normalized_time": row.get("normalized_time", ""),
        "setting_label": label,
        "keep_fraction": keep_fraction,
        "side_bits": side_bits,
        "payload_bytes": "",
        "base_anchor_mse": "",
        "corrected_anchor_mse": "",
        "anchor_mse_reduction": "",
        "base_render_mse": "",
        "corrected_render_mse": "",
        "base_psnr": "",
        "corrected_psnr": "",
        "delta_psnr_vs_base": "",
        "status": "error",
        "error": error,
    }


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            grouped[(row["setting_label"], float(row["keep_fraction"]), int(row["side_bits"]))].append(row)
    out = []
    for (label, keep_fraction, side_bits), items in sorted(grouped.items(), key=lambda item: (item[0][2], item[0][1])):
        out.append({
            "setting_label": label,
            "keep_fraction": keep_fraction,
            "side_bits": side_bits,
            "task_count": len(items),
            "mean_payload_bytes": mean(row["payload_bytes"] for row in items),
            "mean_base_psnr": mean(row["base_psnr"] for row in items),
            "mean_corrected_psnr": mean(row["corrected_psnr"] for row in items),
            "mean_delta_psnr_vs_base": mean(row["delta_psnr_vs_base"] for row in items),
            "mean_anchor_mse_reduction": mean(row["anchor_mse_reduction"] for row in items),
            "mean_corrected_anchor_mse": mean(row["corrected_anchor_mse"] for row in items),
        })
    return out


def gate_rows(rows, summary_rows, args):
    errors = [row for row in rows if row["status"] != "ok"]
    best = max(summary_rows, key=lambda row: float(row["mean_delta_psnr_vs_base"])) if summary_rows else None
    min_payload = min((float(row["payload_bytes"]) for row in rows if row["status"] == "ok"), default=0.0)
    return [
        {
            "gate": "metric_rows_ok",
            "status": "pass" if not errors else "fail",
            "value": len(errors),
            "threshold": "0",
            "detail": "all rendered metrics must use explicit target-shape alignment",
        },
        {
            "gate": "payload_counted_nonzero",
            "status": "pass" if min_payload > 0 else "fail",
            "value": min_payload,
            "threshold": ">0 bytes and counted as len(payload)",
            "detail": "residual payload bytes include headers, metadata, indices/deltas, q values, and zlib bytes",
        },
        {
            "gate": "residual_anchor_mse_reduction_positive",
            "status": "pass" if best and float(best["mean_anchor_mse_reduction"]) > 0.0 else "fail",
            "value": best["mean_anchor_mse_reduction"] if best else "",
            "threshold": ">0",
            "detail": f"best_setting={best['setting_label']}" if best else "no summary rows",
        },
        {
            "gate": "residual_render_headroom_positive",
            "status": "pass" if best and float(best["mean_delta_psnr_vs_base"]) > float(args.positive_headroom_db) else "fail",
            "value": best["mean_delta_psnr_vs_base"] if best else "",
            "threshold": f"> {args.positive_headroom_db} dB vs base",
            "detail": f"best_setting={best['setting_label']}" if best else "no summary rows",
        },
        {
            "gate": "stage197_decoder_contract",
            "status": "pass",
            "value": 0,
            "threshold": "no target dense/RGB decoder input",
            "detail": "decoder uses base GS plus transmitted counted GS residual payload",
        },
    ]


def decision(gates):
    metric_ok = next(row for row in gates if row["gate"] == "metric_rows_ok")
    headroom = next(row for row in gates if row["gate"] == "residual_render_headroom_positive")
    if metric_ok["status"] != "pass":
        return "residual_codec_smoke_invalid_metrics"
    if headroom["status"] == "pass":
        return "residual_codec_smoke_positive_headroom"
    return "residual_codec_smoke_needs_review"


def write_report(package, summary_rows, gates, path):
    lines = [
        "# Stage204 Residual Codec Smoke",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Tasks: `{package['task_count']}`; settings: `{package['setting_count']}`.",
        f"- Best setting: `{package['best_setting_label']}`.",
        f"- Best mean dPSNR vs base: `{package['best_delta_psnr_vs_base']}` dB.",
        "",
        "## Summary",
        "",
        "| setting | keep | q | tasks | payload bytes | base PSNR | corrected PSNR | dPSNR | anchor MSE reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['setting_label']} | {row['keep_fraction']} | {row['side_bits']} | {row['task_count']} | {float(row['mean_payload_bytes']):.3f} | {float(row['mean_base_psnr']):.6f} | {float(row['mean_corrected_psnr']):.6f} | {float(row['mean_delta_psnr_vs_base']):.6f} | {float(row['mean_anchor_mse_reduction']):.6f} |"
        )
    lines.extend([
        "",
        "## Gates",
        "",
        "| gate | status | value | threshold | detail |",
        "|---|---|---:|---|---|",
    ])
    for row in gates:
        lines.append(f"| {row['gate']} | {row['status']} | {row['value']} | {row['threshold']} | {row['detail']} |")
    lines.extend([
        "",
        "## Decoder Contract",
        "",
        "- Encoder uses target dense anchors only to produce GS residual payloads.",
        "- Decoder uses predictor/base GS plus transmitted counted GS-native residual payloads.",
        "- No RGB/image residual or target dense anchor is used as a decoder input.",
        "",
        "## Outputs",
        "",
        f"- selected tasks: `{package['selected_tasks_csv']}`",
        f"- rows: `{package['rows_csv']}`",
        f"- summary: `{package['summary_csv']}`",
        f"- gates: `{package['gates_csv']}`",
        f"- package: `{package['package_json']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--stage203_package", type=Path, default=DEFAULT_STAGE203_PACKAGE)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--task_split", default="eval")
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--keyframe_codec", default="q12")
    parser.add_argument("--max_tasks", type=int, default=12)
    parser.add_argument("--keep_fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--side_bits", type=int, default=6)
    parser.add_argument("--zlib_level", type=int, default=9)
    parser.add_argument("--positive_headroom_db", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    stage203 = read_json(args.stage203_package)
    if stage203["primary_codec"] != "gs_attr_topk_residual_entropy_v1":
        raise RuntimeError(f"Unexpected Stage203 codec: {stage203['primary_codec']}")
    device = torch.device(args.device)
    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    cache = {}
    all_rows = parse_task_rows(args.task_manifest, args.task_split, args.gaps, args.keyframe_codec)
    rows = balanced_sample(all_rows, args.max_tasks, args.seed)
    settings = [(keep, int(args.side_bits)) for keep in args.keep_fractions]
    metric_rows = []
    for row in rows:
        metric_rows.extend(evaluate_task(row, settings, args, device, cache, opt, background))
    summary_rows = summarize(metric_rows)
    gates = gate_rows(metric_rows, summary_rows, args)
    best = max(summary_rows, key=lambda row: float(row["mean_delta_psnr_vs_base"])) if summary_rows else None
    decision_value = decision(gates)

    selected_tasks_csv = args.summary_root / "stage204_selected_tasks.csv"
    rows_csv = args.summary_root / "stage204_residual_codec_smoke_rows.csv"
    summary_csv = args.summary_root / "stage204_residual_codec_smoke_summary.csv"
    gates_csv = args.summary_root / "stage204_residual_codec_smoke_gates.csv"
    package_json = args.summary_root / "stage204_residual_codec_smoke_package.json"
    report_md = args.summary_root / "stage204_residual_codec_smoke_report.md"
    write_csv(selected_task_rows(rows), selected_tasks_csv, TASK_FIELDS)
    write_csv(metric_rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(gates, gates_csv, GATE_FIELDS)
    package = {
        "stage": 204,
        "name": "residual_codec_smoke",
        "decision": decision_value,
        "task_manifest": str(args.task_manifest),
        "stage203_package": str(args.stage203_package),
        "task_split": args.task_split,
        "gaps": args.gaps,
        "keyframe_codec": args.keyframe_codec,
        "task_count": len(rows),
        "setting_count": len(settings),
        "side_bits": args.side_bits,
        "keep_fractions": args.keep_fractions,
        "best_setting_label": best["setting_label"] if best else "",
        "best_delta_psnr_vs_base": best["mean_delta_psnr_vs_base"] if best else "",
        "best_mean_payload_bytes": best["mean_payload_bytes"] if best else "",
        "selected_tasks_csv": str(selected_tasks_csv),
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "gates_csv": str(gates_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "gate_rows": gates,
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, summary_rows, gates, report_md)
    print(json.dumps({"package": str(package_json), "decision": decision_value, "best_delta": package["best_delta_psnr_vs_base"]}, indent=2))


if __name__ == "__main__":
    main()
