import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_CACHE = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage1_streamsplat_fair_metrics/cache")
DEFAULT_STAGE25_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage25_leave_one_out_adapter_training")
DEFAULT_STAGE33_MANIFEST = REPO_ROOT / "experiments/stage33_dense_gap1_anchor_export/stage33_dense_gap1_anchor_manifest.csv"
DEFAULT_STAGE40_PREDICTIONS = REPO_ROOT / "experiments/stage40_normalized_cost_predictor_validation/stage40_predictor_predictions.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage42_calibrated_selector_prior_rd"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import q8_static_mib_per_frame, uniform_indices  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import load_adapter  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import build_anchor_index, flatten_eval, read_manifest_rows, selected_records  # noqa: E402
from scripts.run_stage39_predicted_selector_rd import dp_select_from_costs, segment_fields, selector_comparison, write_comparison_csv, write_csv  # noqa: E402
from scripts.run_stage41_normalized_predicted_selector_rd import read_predictions  # noqa: E402


def method_name(model_name, alpha):
    token = str(alpha).replace(".", "p")
    return f"{model_name}_prior_{token}"


def calibrated_costs(base_costs, target_gap, alpha):
    out = {}
    for (a, b), cost in base_costs.items():
        length = b - a
        prior = ((length - target_gap) / target_gap) ** 2
        out[(a, b)] = float(cost + alpha * prior)
    return out


def best_by_alpha(comparisons, model_name, alphas):
    out = {}
    for alpha in alphas:
        name = method_name(model_name, alpha)
        rows = [row for row in comparisons if row["method"] == name]
        out[name] = {
            "count": len(rows),
            "mean_delta_all_psnr": float(np.mean([row["selector_delta_adapter_all_psnr"] for row in rows])),
            "mean_delta_middle_psnr": float(np.mean([row["selector_delta_adapter_middle_psnr"] for row in rows])),
            "positive_all_points": sum(1 for row in rows if row["selector_delta_adapter_all_psnr"] > 0.0),
            "positive_middle_points": sum(1 for row in rows if row["selector_delta_adapter_middle_psnr"] > 0.0),
            "exact_uniform_points": sum(1 for row in rows if abs(row["selector_delta_adapter_all_psnr"]) < 1e-12),
        }
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--stage25_root", type=Path, default=DEFAULT_STAGE25_ROOT)
    parser.add_argument("--stage33_manifest", type=Path, default=DEFAULT_STAGE33_MANIFEST)
    parser.add_argument("--stage40_predictions", type=Path, default=DEFAULT_STAGE40_PREDICTIONS)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--model", default="full_sample_z_rank")
    parser.add_argument("--gaps", nargs="*", type=int, default=[4, 8, 16])
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.0, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0])
    parser.add_argument("--hidden_dim", type=int, default=128)
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
    pred_rows = read_predictions(args.stage40_predictions, args.samples, [args.model])
    pred_by_sample = defaultdict(dict)
    for row in pred_rows:
        pred_by_sample[row["sample"]][(row["left_index"], row["right_index"])] = row["pred_cost"]

    rows = []
    selections = []
    for sample in args.samples:
        frame_files = sorted((args.stage1_cache / sample / "frames").glob("*.png"))
        manifest_rows = read_manifest_rows(args.stage33_manifest, sample)
        anchor_map = build_anchor_index(manifest_rows, device, args.quant_bits)
        checkpoint_path = args.stage25_root / sample / "stage25_best_anchor_adapter.safetensors"
        model = load_adapter(checkpoint_path, args.hidden_dim, device)
        for gap in args.gaps:
            uniform = uniform_indices(len(frame_files), gap)
            run_specs = [("uniform", uniform, None)]
            budget = len(uniform)
            max_segment_length = gap * 2
            for alpha in args.alphas:
                costs = calibrated_costs(pred_by_sample[sample], gap, alpha)
                selected, cost = dp_select_from_costs(len(frame_files), costs, budget, max_segment_length)
                run_specs.append((method_name(args.model, alpha), selected, cost))
            seen = set()
            for method, indices, selector_cost in run_specs:
                key = (method, tuple(indices))
                if key in seen:
                    continue
                seen.add(key)
                print(f"=== Stage42 sample={sample} method={method} gap={gap} ===", flush=True)
                metrics = selected_records(indices, anchor_map, model, frame_files, opt, background)
                row = flatten_eval(sample, method, gap, indices, metrics, checkpoint_path)
                row.update(segment_fields(indices))
                row["estimated_q8_static_mib_per_frame"] = q8_static_mib_per_frame(len(indices), len(frame_files))
                row["selector_cost"] = selector_cost
                rows.append(row)
                selections.append({"sample": sample, "method": method, "reference_gap": gap, "indices": indices, "selector_cost": selector_cost})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = args.summary_root / "stage42_calibrated_selector_prior_rd.csv"
    comparison_csv = args.summary_root / "stage42_selector_comparison.csv"
    summary_path = args.summary_root / "stage42_calibrated_selector_prior_rd_summary.json"
    write_csv(rows, csv_path)
    comparisons = selector_comparison(rows)
    write_comparison_csv(comparisons, comparison_csv)
    aggregates = best_by_alpha(comparisons, args.model, args.alphas)
    best_method = max(aggregates, key=lambda key: aggregates[key]["mean_delta_all_psnr"])
    summary = {
        "stage": 42,
        "mode": "uniform-prior calibrated predicted selector RD",
        "stage40_predictions": str(args.stage40_predictions),
        "model": args.model,
        "alphas": args.alphas,
        "csv": str(csv_path),
        "comparison_csv": str(comparison_csv),
        "rows": rows,
        "selections": selections,
        "comparisons": comparisons,
        "aggregates": aggregates,
        "best_method_by_mean_all_psnr_delta": best_method,
        "notes": "Adds a uniform segment-length prior to Stage40 predicted relative costs. alpha=0 is equivalent to Stage41 for the selected model.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "csv": str(csv_path), "comparison_csv": str(comparison_csv), "best_method": best_method, "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
