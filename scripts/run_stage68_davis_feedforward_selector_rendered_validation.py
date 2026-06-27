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
DEFAULT_STAGE66_SUMMARY = REPO_ROOT / "experiments/stage66_davis_feedforward_selector_dataset/stage66_davis_selector_dataset_summary.json"
DEFAULT_STAGE67_PARAMS = REPO_ROOT / "experiments/stage67_davis_selector_predictor_training/stage67_selector_predictor_model_params.json"
DEFAULT_ADAPTER = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage68_davis_feedforward_selector_rendered_validation"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import q8_static_mib_per_frame, uniform_indices  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import load_adapter  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import flatten_eval, selected_records  # noqa: E402
from scripts.run_stage39_predicted_selector_rd import dp_select_from_costs, segment_fields  # noqa: E402
from scripts.run_stage66_davis_feedforward_selector_dataset import (  # noqa: E402
    anchor_features,
    load_sequence_anchors,
    load_small_frames,
    read_gap1_manifest,
    rgb_motion_features,
)


METHOD_UNIFORM = "uniform"
METHOD_PREDICTED = "predicted_full_feature_dp"

RESULT_FIELDS = [
    "sample", "method", "reference_gap", "selector_cost", "total_frames", "keyframe_count", "keyframe_ratio",
    "estimated_q8_static_mib_per_frame", "max_segment_length", "mean_segment_length", "segment_lengths", "indices",
    "linear_all_psnr", "adapter_all_psnr", "delta_all_psnr", "linear_middle_psnr", "adapter_middle_psnr",
    "delta_middle_psnr", "linear_given_psnr", "adapter_given_psnr", "delta_given_psnr", "linear_all_ssim",
    "adapter_all_ssim", "linear_middle_ssim", "adapter_middle_ssim", "checkpoint",
]

SELECTION_FIELDS = [
    "sample", "method", "reference_gap", "selector_cost", "total_frames", "keyframe_count",
    "max_segment_length", "mean_segment_length", "segment_lengths", "indices",
]

COMPARISON_FIELDS = [
    "sample", "reference_gap", "rate_mib_per_frame", "uniform_adapter_all_psnr", "predicted_adapter_all_psnr",
    "selector_delta_adapter_all_psnr", "uniform_linear_all_psnr", "predicted_linear_all_psnr",
    "selector_delta_linear_all_psnr", "uniform_indices", "predicted_indices",
]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_eval_sequence_keys(summary_path, limit):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    keys = []
    for row in summary["selected_sequences"]:
        if row["selector_split"] != "eval":
            continue
        keys.append((row["dataset"], row["split"], row["sequence"]))
    keys = sorted(keys)
    if limit > 0:
        keys = keys[:limit]
    return keys


def load_predictor(params_path, model_name):
    params = json.loads(params_path.read_text(encoding="utf-8"))
    if model_name not in params:
        raise KeyError(f"Missing model {model_name} in {params_path}")
    model = params[model_name]
    return {
        "features": model["features"],
        "intercept": float(model["intercept"]),
        "weights": np.asarray(model["weights"], dtype=np.float64),
        "mean": np.asarray(model["mean"], dtype=np.float64),
        "std": np.asarray(model["std"], dtype=np.float64),
    }


def predict_log_cost(feature_row, predictor):
    x = np.asarray([float(feature_row[name]) for name in predictor["features"]], dtype=np.float64)
    z = (x - predictor["mean"]) / predictor["std"]
    return float(predictor["intercept"] + np.dot(z, predictor["weights"]))


def segment_feature_row(indices, attrs_map, frames, a, b):
    return {
        "segment_length": b - a,
        "middle_count": max(b - a - 1, 0),
        "normalized_left": a / max(indices[-1], 1),
        "normalized_right": b / max(indices[-1], 1),
        **anchor_features(attrs_map, a, b),
        **rgb_motion_features(frames, a, b),
    }


def predicted_cost_matrix(indices, attrs_map, frames, predictor, max_segment_length):
    costs = {}
    index_set = set(indices)
    for a in indices[:-1]:
        for length in range(1, max_segment_length + 1):
            b = a + length
            if b not in index_set:
                continue
            if length <= 1:
                costs[(a, b)] = 0.0
                continue
            features = segment_feature_row(indices, attrs_map, frames, a, b)
            pred_mean = 10.0 ** predict_log_cost(features, predictor)
            costs[(a, b)] = float(pred_mean * features["middle_count"])
    return costs


def selection_row(sample, method, gap, indices, selector_cost, total_frames):
    fields = segment_fields(indices)
    return {
        "sample": sample,
        "method": method,
        "reference_gap": gap,
        "selector_cost": selector_cost,
        "total_frames": total_frames,
        "keyframe_count": len(indices),
        "indices": " ".join(str(idx) for idx in indices),
        **fields,
    }


def build_comparisons(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["sample"], row["reference_gap"])][row["method"]] = row
    out = []
    for (sample, gap), methods in sorted(grouped.items()):
        if METHOD_UNIFORM not in methods or METHOD_PREDICTED not in methods:
            continue
        uniform = methods[METHOD_UNIFORM]
        predicted = methods[METHOD_PREDICTED]
        out.append({
            "sample": sample,
            "reference_gap": gap,
            "rate_mib_per_frame": predicted["estimated_q8_static_mib_per_frame"],
            "uniform_adapter_all_psnr": uniform["adapter_all_psnr"],
            "predicted_adapter_all_psnr": predicted["adapter_all_psnr"],
            "selector_delta_adapter_all_psnr": predicted["adapter_all_psnr"] - uniform["adapter_all_psnr"],
            "uniform_linear_all_psnr": uniform["linear_all_psnr"],
            "predicted_linear_all_psnr": predicted["linear_all_psnr"],
            "selector_delta_linear_all_psnr": predicted["linear_all_psnr"] - uniform["linear_all_psnr"],
            "uniform_indices": uniform["indices"],
            "predicted_indices": predicted["indices"],
        })
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage66_summary", type=Path, default=DEFAULT_STAGE66_SUMMARY)
    parser.add_argument("--stage67_params", type=Path, default=DEFAULT_STAGE67_PARAMS)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--predictor_model", default="full_feature_ridge")
    parser.add_argument("--gaps", nargs="*", type=int, default=[4, 8, 16])
    parser.add_argument("--max_segment_length", type=int, default=16)
    parser.add_argument("--max_eval_sequences", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=64)
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

    eval_keys = read_eval_sequence_keys(args.stage66_summary, args.max_eval_sequences)
    if not eval_keys:
        raise RuntimeError("No Stage66 eval sequences found")
    splits = sorted({key[1] for key in eval_keys})
    rows_by_sequence = read_gap1_manifest(args.manifest, splits)
    predictor = load_predictor(args.stage67_params, args.predictor_model)
    model = load_adapter(args.adapter, args.hidden_dim, device)

    result_rows = []
    selection_rows = []
    coverage_rows = []
    eval_cache = {}
    for key in eval_keys:
        dataset, split, sequence = key
        sample = f"{dataset}/{split}/{sequence}"
        print(f"=== Stage68 {sample} ===", flush=True)
        if key not in rows_by_sequence:
            raise RuntimeError(f"Missing gap1 manifest rows for {sample}")
        indices, _anchor_map, quantized_map, attrs_map, rgb_paths = load_sequence_anchors(rows_by_sequence[key], device, args.quant_bits)
        frame_files = [rgb_paths[idx] for idx in indices]
        frames = load_small_frames(rgb_paths, args.image_size)
        costs = predicted_cost_matrix(indices, attrs_map, frames, predictor, args.max_segment_length)
        coverage_rows.append({
            "sample": sample,
            "total_frames": len(indices),
            "candidate_cost_count": len(costs),
            "max_segment_length": args.max_segment_length,
        })
        for gap in args.gaps:
            uniform = uniform_indices(len(indices), gap)
            budget = len(uniform)
            predicted, selector_cost = dp_select_from_costs(len(indices), costs, budget, args.max_segment_length)
            run_specs = [(METHOD_UNIFORM, uniform, None), (METHOD_PREDICTED, predicted, selector_cost)]
            for method, selected, cost in run_specs:
                cache_key = (sample, tuple(selected))
                if cache_key not in eval_cache:
                    print(f"=== Stage68 render {sample} method={method} gap={gap} ===", flush=True)
                    eval_cache[cache_key] = selected_records(selected, quantized_map, model, frame_files, opt, background)
                metrics = eval_cache[cache_key]
                row = flatten_eval(sample, method, gap, selected, metrics, args.adapter)
                row["selector_cost"] = cost
                row["estimated_q8_static_mib_per_frame"] = q8_static_mib_per_frame(len(selected), len(indices))
                result_rows.append(row)
                selection_rows.append(selection_row(sample, method, gap, selected, cost, len(indices)))
        del quantized_map, attrs_map, frames
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison_rows = build_comparisons(result_rows)
    result_csv = args.summary_root / "stage68_davis_selector_rendered_validation.csv"
    comparison_csv = args.summary_root / "stage68_davis_selector_comparison.csv"
    selection_csv = args.summary_root / "stage68_davis_selector_selections.csv"
    summary_path = args.summary_root / "stage68_davis_selector_rendered_validation_summary.json"
    write_csv(result_rows, result_csv, RESULT_FIELDS)
    write_csv(comparison_rows, comparison_csv, COMPARISON_FIELDS)
    write_csv(selection_rows, selection_csv, SELECTION_FIELDS)

    deltas = [row["selector_delta_adapter_all_psnr"] for row in comparison_rows]
    linear_deltas = [row["selector_delta_linear_all_psnr"] for row in comparison_rows]
    summary = {
        "stage": 68,
        "mode": "DAVIS feed-forward selector deterministic-DP rendered validation",
        "manifest": str(args.manifest),
        "stage66_summary": str(args.stage66_summary),
        "stage67_params": str(args.stage67_params),
        "adapter": str(args.adapter),
        "predictor_model": args.predictor_model,
        "gaps": args.gaps,
        "max_segment_length": args.max_segment_length,
        "quant_bits": args.quant_bits,
        "eval_sequences": [f"{dataset}/{split}/{sequence}" for dataset, split, sequence in eval_keys],
        "coverage": coverage_rows,
        "result_csv": str(result_csv),
        "comparison_csv": str(comparison_csv),
        "selection_csv": str(selection_csv),
        "comparison_count": len(comparison_rows),
        "mean_selector_delta_adapter_all_psnr": float(np.mean(deltas)) if deltas else None,
        "positive_selector_adapter_all_points": sum(1 for value in deltas if value > 0.0),
        "mean_selector_delta_linear_all_psnr": float(np.mean(linear_deltas)) if linear_deltas else None,
        "positive_selector_linear_all_points": sum(1 for value in linear_deltas if value > 0.0),
        "notes": "Selection uses feed-forward predicted costs plus deterministic DP. It does not use rendered oracle, PSNR labels, dense-anchor labels, or reconstruction lookahead at selection time.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "comparison_csv": str(comparison_csv),
        "comparison_count": summary["comparison_count"],
        "mean_selector_delta_adapter_all_psnr": summary["mean_selector_delta_adapter_all_psnr"],
        "positive_selector_adapter_all_points": summary["positive_selector_adapter_all_points"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
