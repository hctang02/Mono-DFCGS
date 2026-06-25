import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_CACHE = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage1_streamsplat_fair_metrics/cache")
DEFAULT_STAGE25_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage25_leave_one_out_adapter_training")
DEFAULT_STAGE33_MANIFEST = REPO_ROOT / "experiments/stage33_dense_gap1_anchor_export/stage33_dense_gap1_anchor_manifest.csv"
DEFAULT_STAGE38_PREDICTIONS = REPO_ROOT / "experiments/stage38_deployable_cost_predictor_validation/stage38_predictor_predictions.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage39_predicted_selector_rd"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import q8_static_mib_per_frame, uniform_indices  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import load_adapter  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import build_anchor_index, flatten_eval, read_manifest_rows, selected_records  # noqa: E402


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
                "pred_log_cost": float(row["pred_log_cost"]),
            })
    return rows


def dp_select_from_costs(total_frames, costs, budget, max_segment_length):
    inf = float("inf")
    dp = [[inf] * total_frames for _ in range(budget)]
    prev = [[None] * total_frames for _ in range(budget)]
    dp[0][0] = 0.0
    for used in range(1, budget):
        for b in range(1, total_frames):
            start = max(0, b - max_segment_length)
            best = inf
            best_a = None
            for a in range(start, b):
                if dp[used - 1][a] == inf:
                    continue
                cost = costs.get((a, b))
                if cost is None:
                    continue
                value = dp[used - 1][a] + cost
                if value < best:
                    best = value
                    best_a = a
            dp[used][b] = best
            prev[used][b] = best_a
    if dp[budget - 1][total_frames - 1] == inf:
        raise RuntimeError(f"No feasible predicted selection for total_frames={total_frames} budget={budget}")
    indices = []
    used = budget - 1
    pos = total_frames - 1
    while pos is not None:
        indices.append(pos)
        pos = prev[used][pos]
        used -= 1
        if used < 0:
            break
    return list(reversed(indices)), float(dp[budget - 1][total_frames - 1])


def segment_fields(indices):
    lengths = [b - a for a, b in zip(indices[:-1], indices[1:])]
    return {
        "max_segment_length": int(max(lengths)),
        "mean_segment_length": float(np.mean(lengths)),
        "segment_lengths": " ".join(str(v) for v in lengths),
    }


def write_csv(rows, path):
    fields = [
        "sample", "method", "reference_gap", "selector_cost", "total_frames", "keyframe_count", "keyframe_ratio",
        "estimated_q8_static_mib_per_frame", "max_segment_length", "mean_segment_length", "segment_lengths", "indices",
        "linear_all_psnr", "adapter_all_psnr", "delta_all_psnr", "linear_middle_psnr", "adapter_middle_psnr",
        "delta_middle_psnr", "linear_given_psnr", "adapter_given_psnr", "delta_given_psnr", "linear_all_ssim",
        "adapter_all_ssim", "linear_middle_ssim", "adapter_middle_ssim", "checkpoint",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selector_comparison(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["sample"], row["reference_gap"])][row["method"]] = row
    out = []
    for (sample, gap), methods in sorted(grouped.items()):
        if "uniform" not in methods:
            continue
        uniform = methods["uniform"]
        for method, selected in methods.items():
            if method == "uniform":
                continue
            out.append({
                "sample": sample,
                "reference_gap": gap,
                "method": method,
                "selector_delta_adapter_all_psnr": selected["adapter_all_psnr"] - uniform["adapter_all_psnr"],
                "selector_delta_adapter_middle_psnr": selected["adapter_middle_psnr"] - uniform["adapter_middle_psnr"],
                "selector_delta_linear_all_psnr": selected["linear_all_psnr"] - uniform["linear_all_psnr"],
                "selector_delta_linear_middle_psnr": selected["linear_middle_psnr"] - uniform["linear_middle_psnr"],
            })
    return out


def write_comparison_csv(rows, path):
    fields = [
        "sample", "reference_gap", "method", "selector_delta_adapter_all_psnr", "selector_delta_adapter_middle_psnr",
        "selector_delta_linear_all_psnr", "selector_delta_linear_middle_psnr",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_delta(comparisons, method, key, ylabel, title, out_path):
    rows = [row for row in comparisons if row["method"] == method]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sample"]].append(row)
    samples = sorted(grouped)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharey=True)
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        points = sorted(grouped[sample], key=lambda row: row["reference_gap"])
        ax.bar([str(row["reference_gap"]) for row in points], [row[key] for row in points], color="#ff7f0e")
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(sample)
        ax.set_xlabel("Reference GOP gap")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    axes[0].set_ylabel(ylabel)
    axes[2].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--stage25_root", type=Path, default=DEFAULT_STAGE25_ROOT)
    parser.add_argument("--stage33_manifest", type=Path, default=DEFAULT_STAGE33_MANIFEST)
    parser.add_argument("--stage38_predictions", type=Path, default=DEFAULT_STAGE38_PREDICTIONS)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--models", nargs="*", default=["length_only_ridge", "full_feature_ridge"])
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
    pred_rows = read_predictions(args.stage38_predictions, args.samples, args.models)
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
                print(f"=== Stage39 sample={sample} method={method} gap={gap} ===", flush=True)
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

    csv_path = args.summary_root / "stage39_predicted_selector_rd.csv"
    comparison_csv = args.summary_root / "stage39_selector_comparison.csv"
    summary_path = args.summary_root / "stage39_predicted_selector_rd_summary.json"
    write_csv(rows, csv_path)
    comparisons = selector_comparison(rows)
    write_comparison_csv(comparisons, comparison_csv)
    plots = []
    for method in args.models:
        for key, ylabel, filename in [
            ("selector_delta_adapter_all_psnr", f"{method} - uniform all PSNR (dB)", f"stage39_{method}_delta_all_psnr.png"),
            ("selector_delta_adapter_middle_psnr", f"{method} - uniform middle PSNR (dB)", f"stage39_{method}_delta_middle_psnr.png"),
        ]:
            out_path = args.summary_root / filename
            plot_delta(comparisons, method, key, ylabel, f"Stage 39 Predicted Selector Gain: {method}", out_path)
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
        "stage": 39,
        "mode": "predicted deployable selector RD",
        "stage38_predictions": str(args.stage38_predictions),
        "csv": str(csv_path),
        "comparison_csv": str(comparison_csv),
        "plots": plots,
        "rows": rows,
        "selections": selections,
        "comparisons": comparisons,
        "aggregates": aggregates,
        "notes": "Uses Stage38 predicted segment costs. This is deployable with respect to features, but the predictor itself is a simple leave-one-out ridge baseline.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "csv": str(csv_path), "comparison_csv": str(comparison_csv), "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
