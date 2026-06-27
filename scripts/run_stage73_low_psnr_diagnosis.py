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
DEFAULT_STAGE72_ROWS = REPO_ROOT / "experiments/stage72_original_davis_baseline/stage72_original_davis_baseline_rows.csv"
DEFAULT_STAGE70_PSNR = REPO_ROOT / "experiments/stage70_scoped_davis_rd_package/stage70_all_psnr_table.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage73_low_psnr_diagnosis"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import uniform_indices  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import load_adapter  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import flatten_eval, selected_records  # noqa: E402
from scripts.run_stage66_davis_feedforward_selector_dataset import load_sequence_anchors, read_gap1_manifest  # noqa: E402


DIAG_FIELDS = [
    "sample",
    "frame_gap",
    "total_frames",
    "keyframe_count",
    "original_all_psnr",
    "original_middle_psnr",
    "original_given_psnr",
    "float_linear_all_psnr",
    "float_adapter_all_psnr",
    "float_linear_middle_psnr",
    "float_adapter_middle_psnr",
    "float_given_psnr",
    "q8_linear_all_psnr",
    "q8_adapter_all_psnr",
    "q8_linear_middle_psnr",
    "q8_adapter_middle_psnr",
    "q8_given_psnr",
    "stage70_linear_uniform_all_psnr",
    "stage70_adapter_uniform_all_psnr",
    "float_adapter_gap_to_original_all",
    "q8_adapter_gap_to_original_all",
    "q8_loss_adapter_all",
    "q8_loss_linear_all",
    "float_given_gap_to_original_given",
    "q8_given_gap_to_original_given",
]

SUMMARY_FIELDS = [
    "frame_gap",
    "point_count",
    "mean_original_all_psnr",
    "mean_float_adapter_all_psnr",
    "mean_q8_adapter_all_psnr",
    "mean_stage70_adapter_uniform_all_psnr",
    "mean_float_adapter_gap_to_original_all",
    "mean_q8_adapter_gap_to_original_all",
    "mean_q8_loss_adapter_all",
    "mean_float_given_gap_to_original_given",
    "mean_q8_given_gap_to_original_given",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_stage72(path):
    out = {}
    for row in read_csv(path):
        out[(row["sample"], int(row["frame_gap"]))] = {
            "all": float(row["all_psnr_avg"]),
            "middle": float(row["middle_psnr_avg"]),
            "given": float(row["given_psnr_avg"]),
        }
    return out


def load_stage70(path):
    out = {}
    for row in read_csv(path):
        if row["selector"] != "uniform":
            continue
        out[(row["sample"], int(row["reference_gap"]), row["method"])] = float(row["all_psnr"])
    return out


def metric(metrics, method, scope, key="psnr_avg"):
    return metrics[method][scope][key]


def average(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def build_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["frame_gap"])].append(row)
    out = []
    for gap, items in sorted(grouped.items()):
        out.append({
            "frame_gap": gap,
            "point_count": len(items),
            "mean_original_all_psnr": average(row["original_all_psnr"] for row in items),
            "mean_float_adapter_all_psnr": average(row["float_adapter_all_psnr"] for row in items),
            "mean_q8_adapter_all_psnr": average(row["q8_adapter_all_psnr"] for row in items),
            "mean_stage70_adapter_uniform_all_psnr": average(row["stage70_adapter_uniform_all_psnr"] for row in items),
            "mean_float_adapter_gap_to_original_all": average(row["float_adapter_gap_to_original_all"] for row in items),
            "mean_q8_adapter_gap_to_original_all": average(row["q8_adapter_gap_to_original_all"] for row in items),
            "mean_q8_loss_adapter_all": average(row["q8_loss_adapter_all"] for row in items),
            "mean_float_given_gap_to_original_given": average(row["float_given_gap_to_original_given"] for row in items),
            "mean_q8_given_gap_to_original_given": average(row["q8_given_gap_to_original_given"] for row in items),
        })
    return out


def write_report(summary, rows, gap_rows, path):
    lines = [
        "# Stage73 Low-PSNR Diagnosis",
        "",
        "## Main Diagnosis",
        "",
        "The Stage70 numbers are low mostly because the current anchor-only path renders static anchors and discards original StreamSplat dynamic Gaussian components, not because of a large q8 quantization loss.",
        "",
        "## Gap Summary",
        "",
        "| gap | original all | float adapter all | q8 adapter all | q8 loss | original - float adapter | original given - float given |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gap_rows:
        lines.append(
            f"| {row['frame_gap']} | {row['mean_original_all_psnr']} | {row['mean_float_adapter_all_psnr']} | {row['mean_q8_adapter_all_psnr']} | {row['mean_q8_loss_adapter_all']} | {row['mean_float_adapter_gap_to_original_all']} | {row['mean_float_given_gap_to_original_given']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `float_*` uses the saved fp16 static anchors without the additional q8 simulation.",
        "- `q8_*` matches the Stage70 q8 static-anchor protocol.",
        "- If `q8_loss` is small but `original - float adapter` is large, the main loss is static-anchor-only modeling, not quantization.",
        "- If `original given - float given` is large, even keyframe rendering loses quality when dynamic Gaussian fields are discarded.",
        "",
        "## Outputs",
        "",
        f"- Summary JSON: `{summary['summary_json']}`",
        f"- Diagnosis CSV: `{summary['diagnosis_csv']}`",
        f"- Gap summary CSV: `{summary['gap_summary_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--stage72_rows", type=Path, default=DEFAULT_STAGE72_ROWS)
    parser.add_argument("--stage70_psnr", type=Path, default=DEFAULT_STAGE70_PSNR)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--sequences", nargs="+", default=["bmx-trees", "car-shadow", "goat", "soapbox"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    stage72 = load_stage72(args.stage72_rows)
    stage70 = load_stage70(args.stage70_psnr)

    rows_by_sequence = read_gap1_manifest(args.manifest, [args.split])
    diagnosis_rows = []
    for sequence in args.sequences:
        key = ("DAVIS", args.split, sequence)
        if key not in rows_by_sequence:
            raise RuntimeError(f"Missing gap1 anchors for {key}")
        sample = f"DAVIS/{args.split}/{sequence}"
        print(f"=== Stage73 diagnosis {sample} ===", flush=True)
        indices, anchor_map, q8_map, _attrs_map, rgb_paths = load_sequence_anchors(rows_by_sequence[key], device, args.quant_bits)
        frame_files = [rgb_paths[idx] for idx in indices]
        for gap in args.gaps:
            selected = uniform_indices(len(indices), gap)
            float_metrics = selected_records(selected, anchor_map, model, frame_files, opt, background)
            q8_metrics = selected_records(selected, q8_map, model, frame_files, opt, background)
            original = stage72[(sample, gap)]
            stage70_linear = stage70.get((sample, gap, "linear_anchor"))
            stage70_adapter = stage70.get((sample, gap, "stage65_rgb_h256_adapter"))
            row = {
                "sample": sample,
                "frame_gap": gap,
                "total_frames": len(indices),
                "keyframe_count": len(selected),
                "original_all_psnr": original["all"],
                "original_middle_psnr": original["middle"],
                "original_given_psnr": original["given"],
                "float_linear_all_psnr": metric(float_metrics, "linear", "all"),
                "float_adapter_all_psnr": metric(float_metrics, "adapter", "all"),
                "float_linear_middle_psnr": metric(float_metrics, "linear", "middle_only"),
                "float_adapter_middle_psnr": metric(float_metrics, "adapter", "middle_only"),
                "float_given_psnr": metric(float_metrics, "adapter", "given_keyframes"),
                "q8_linear_all_psnr": metric(q8_metrics, "linear", "all"),
                "q8_adapter_all_psnr": metric(q8_metrics, "adapter", "all"),
                "q8_linear_middle_psnr": metric(q8_metrics, "linear", "middle_only"),
                "q8_adapter_middle_psnr": metric(q8_metrics, "adapter", "middle_only"),
                "q8_given_psnr": metric(q8_metrics, "adapter", "given_keyframes"),
                "stage70_linear_uniform_all_psnr": stage70_linear,
                "stage70_adapter_uniform_all_psnr": stage70_adapter,
            }
            row.update({
                "float_adapter_gap_to_original_all": row["original_all_psnr"] - row["float_adapter_all_psnr"],
                "q8_adapter_gap_to_original_all": row["original_all_psnr"] - row["q8_adapter_all_psnr"],
                "q8_loss_adapter_all": row["float_adapter_all_psnr"] - row["q8_adapter_all_psnr"],
                "q8_loss_linear_all": row["float_linear_all_psnr"] - row["q8_linear_all_psnr"],
                "float_given_gap_to_original_given": row["original_given_psnr"] - row["float_given_psnr"],
                "q8_given_gap_to_original_given": row["original_given_psnr"] - row["q8_given_psnr"],
            })
            diagnosis_rows.append(row)
        del anchor_map, q8_map
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gap_rows = build_summary(diagnosis_rows)
    diagnosis_csv = args.summary_root / "stage73_low_psnr_diagnosis.csv"
    gap_summary_csv = args.summary_root / "stage73_low_psnr_gap_summary.csv"
    summary_json = args.summary_root / "stage73_low_psnr_diagnosis_summary.json"
    report_md = args.summary_root / "stage73_low_psnr_diagnosis_report.md"
    write_csv(diagnosis_rows, diagnosis_csv, DIAG_FIELDS)
    write_csv(gap_rows, gap_summary_csv, SUMMARY_FIELDS)
    summary = {
        "stage": 73,
        "mode": "Stage70 low-PSNR diagnosis",
        "manifest": str(args.manifest),
        "adapter": str(args.adapter),
        "stage72_rows": str(args.stage72_rows),
        "stage70_psnr": str(args.stage70_psnr),
        "sequences": args.sequences,
        "gaps": args.gaps,
        "quant_bits": args.quant_bits,
        "diagnosis_csv": str(diagnosis_csv),
        "gap_summary_csv": str(gap_summary_csv),
        "report_md": str(report_md),
        "summary_json": str(summary_json),
        "gap_summary": gap_rows,
        "notes": [
            "Float static anchors remove the extra q8 simulation but still discard dynamic Gaussian fields.",
            "Small q8_loss with large original-vs-float gap means low PSNR is mainly modeling/information loss, not quantization.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, diagnosis_rows, gap_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "diagnosis_csv": str(diagnosis_csv),
        "gap_summary_csv": str(gap_summary_csv),
        "gap_summary": gap_rows,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
