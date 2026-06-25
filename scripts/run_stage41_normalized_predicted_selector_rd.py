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
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage41_normalized_predicted_selector_rd"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import q8_static_mib_per_frame, uniform_indices  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import load_adapter  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import build_anchor_index, flatten_eval, read_manifest_rows, selected_records  # noqa: E402
from scripts.run_stage39_predicted_selector_rd import (  # noqa: E402
    dp_select_from_costs,
    plot_delta,
    segment_fields,
    selector_comparison,
    write_comparison_csv,
    write_csv,
)


def read_predictions(path, samples, models):
    sample_set = set(samples)
    model_set = set(models)
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sample"] not in sample_set or row["model"] not in model_set:
                continue
            rows.append({
                "sample": row["sample"],
                "model": row["model"],
                "left_index": int(row["left_index"]),
                "right_index": int(row["right_index"]),
                "segment_length": int(row["segment_length"]),
                "pred_cost": float(row["pred_cost"]),
            })
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--stage25_root", type=Path, default=DEFAULT_STAGE25_ROOT)
    parser.add_argument("--stage33_manifest", type=Path, default=DEFAULT_STAGE33_MANIFEST)
    parser.add_argument("--stage40_predictions", type=Path, default=DEFAULT_STAGE40_PREDICTIONS)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--models", nargs="*", default=["length_sample_z_rank", "full_sample_z_rank", "full_sample_z_zlog"])
    parser.add_argument("--gaps", nargs="*", type=int, default=[4, 8, 16])
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
    pred_rows = read_predictions(args.stage40_predictions, args.samples, args.models)
    pred_by_sample_model = defaultdict(dict)
    for row in pred_rows:
        pred_by_sample_model[(row["sample"], row["model"])][(row["left_index"], row["right_index"])] = row["pred_cost"]

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
            for model_name in args.models:
                selected, cost = dp_select_from_costs(len(frame_files), pred_by_sample_model[(sample, model_name)], budget, max_segment_length)
                run_specs.append((model_name, selected, cost))
            for method, indices, selector_cost in run_specs:
                print(f"=== Stage41 sample={sample} method={method} gap={gap} ===", flush=True)
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

    csv_path = args.summary_root / "stage41_normalized_predicted_selector_rd.csv"
    comparison_csv = args.summary_root / "stage41_selector_comparison.csv"
    summary_path = args.summary_root / "stage41_normalized_predicted_selector_rd_summary.json"
    write_csv(rows, csv_path)
    comparisons = selector_comparison(rows)
    write_comparison_csv(comparisons, comparison_csv)
    plots = []
    for method in args.models:
        for key, ylabel, filename in [
            ("selector_delta_adapter_all_psnr", f"{method} - uniform all PSNR (dB)", f"stage41_{method}_delta_all_psnr.png"),
            ("selector_delta_adapter_middle_psnr", f"{method} - uniform middle PSNR (dB)", f"stage41_{method}_delta_middle_psnr.png"),
        ]:
            out_path = args.summary_root / filename
            plot_delta(comparisons, method, key, ylabel, f"Stage 41 Normalized Predicted Selector Gain: {method}", out_path)
            plots.append(str(out_path))
    aggregates = {}
    for method in args.models:
        method_rows = [row for row in comparisons if row["method"] == method]
        aggregates[method] = {
            "count": len(method_rows),
            "mean_delta_all_psnr": float(np.mean([row["selector_delta_adapter_all_psnr"] for row in method_rows])),
            "mean_delta_middle_psnr": float(np.mean([row["selector_delta_adapter_middle_psnr"] for row in method_rows])),
            "positive_all_points": sum(1 for row in method_rows if row["selector_delta_adapter_all_psnr"] > 0.0),
            "positive_middle_points": sum(1 for row in method_rows if row["selector_delta_adapter_middle_psnr"] > 0.0),
        }
    summary = {
        "stage": 41,
        "mode": "normalized predicted deployable selector RD",
        "stage40_predictions": str(args.stage40_predictions),
        "csv": str(csv_path),
        "comparison_csv": str(comparison_csv),
        "plots": plots,
        "rows": rows,
        "selections": selections,
        "comparisons": comparisons,
        "aggregates": aggregates,
        "notes": "Uses Stage40 sample-normalized predictor scores as relative DP costs. Features are deployable; labels are oracle/proxy only during training.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "csv": str(csv_path), "comparison_csv": str(comparison_csv), "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
