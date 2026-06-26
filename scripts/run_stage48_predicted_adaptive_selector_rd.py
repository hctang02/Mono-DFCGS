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
DEFAULT_STAGE47_PREDICTIONS = REPO_ROOT / "experiments/stage47_rendered_cost_predictor_validation/stage47_predictor_predictions.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage48_predicted_adaptive_selector_rd"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import q8_static_mib_per_frame, uniform_indices  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import load_adapter  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import build_anchor_index, flatten_eval, read_manifest_rows, selected_records  # noqa: E402
from scripts.run_stage39_predicted_selector_rd import dp_select_from_costs, segment_fields, selector_comparison, write_comparison_csv, write_csv  # noqa: E402


METHOD_SPECS = [
    ("length_raw_log_prior_0p1", "length_raw_log", 0.1),
    ("full_raw_log_prior_0p1", "full_raw_log", 0.1),
    ("length_sample_z_rank_prior_0p1", "length_sample_z_rank", 0.1),
    ("full_sample_z_rank", "full_sample_z_rank", 0.0),
    ("full_sample_z_rank_prior_0p1", "full_sample_z_rank", 0.1),
    ("full_sample_z_rank_prior_0p3", "full_sample_z_rank", 0.3),
]


def read_predictions(path, samples):
    sample_set = set(samples)
    pred = defaultdict(dict)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = row["sample"]
            if sample not in sample_set or row["heldout_fold"] != sample:
                continue
            model = row["model"]
            a = int(row["left_index"])
            b = int(row["right_index"])
            pred[(sample, model)][(a, b)] = float(row["pred_cost"])
    return pred


def calibrated_costs(costs, gap, alpha):
    out = {}
    for (a, b), cost in costs.items():
        length = b - a
        prior = ((length - gap) / gap) ** 2
        out[(a, b)] = float(cost + alpha * prior)
    return out


def aggregate_comparisons(comparisons):
    aggregates = {}
    for method in sorted({row["method"] for row in comparisons}):
        rows = [row for row in comparisons if row["method"] == method]
        aggregates[method] = {
            "count": len(rows),
            "positive_all_points": sum(1 for row in rows if row["selector_delta_adapter_all_psnr"] > 0.0),
            "positive_middle_points": sum(1 for row in rows if row["selector_delta_adapter_middle_psnr"] > 0.0),
            "mean_delta_all_psnr": float(np.mean([row["selector_delta_adapter_all_psnr"] for row in rows])),
            "mean_delta_middle_psnr": float(np.mean([row["selector_delta_adapter_middle_psnr"] for row in rows])),
            "min_delta_all_psnr": float(np.min([row["selector_delta_adapter_all_psnr"] for row in rows])),
            "exact_uniform_points": sum(1 for row in rows if abs(row["selector_delta_adapter_all_psnr"]) < 1e-12),
        }
    return aggregates


def write_aggregate_csv(aggregates, path):
    fields = [
        "method", "count", "positive_all_points", "positive_middle_points", "mean_delta_all_psnr",
        "mean_delta_middle_psnr", "min_delta_all_psnr", "exact_uniform_points",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method, values in aggregates.items():
            writer.writerow({"method": method, **values})


def plot_best(rows, best_method, metric_key, ylabel, out_path):
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] in {"uniform", best_method}:
            grouped[(row["sample"], row["method"])].append(row)
    samples = sorted({row["sample"] for row in rows})
    colors = {"uniform": "#1f77b4", best_method: "#2ca02c"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharey=False)
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        for method in ["uniform", best_method]:
            points = sorted(grouped[(sample, method)], key=lambda row: row["estimated_q8_static_mib_per_frame"])
            ax.plot(
                [row["estimated_q8_static_mib_per_frame"] for row in points],
                [row[metric_key] for row in points],
                marker="o",
                color=colors[method],
                linewidth=2.0,
                label=method,
            )
        ax.set_title(sample)
        ax.set_xlabel("Estimated q8 anchor MiB/frame")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--stage25_root", type=Path, default=DEFAULT_STAGE25_ROOT)
    parser.add_argument("--stage33_manifest", type=Path, default=DEFAULT_STAGE33_MANIFEST)
    parser.add_argument("--stage47_predictions", type=Path, default=DEFAULT_STAGE47_PREDICTIONS)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--samples", nargs="*", default=["n3dv", "meetroom", "driving", "robot"])
    parser.add_argument("--gaps", nargs="*", type=int, default=[4, 8, 16])
    parser.add_argument("--max_segment_multiplier", type=int, default=2)
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
    predictions = read_predictions(args.stage47_predictions, args.samples)
    rows = []
    selections = []
    eval_cache = {}
    for sample in args.samples:
        print(f"=== Stage48 sample={sample} ===", flush=True)
        frame_files = sorted((args.stage1_cache / sample / "frames").glob("*.png"))
        manifest_rows = read_manifest_rows(args.stage33_manifest, sample)
        anchor_map = build_anchor_index(manifest_rows, device, args.quant_bits)
        checkpoint_path = args.stage25_root / sample / "stage25_best_anchor_adapter.safetensors"
        model = load_adapter(checkpoint_path, args.hidden_dim, device)
        for gap in args.gaps:
            uniform = uniform_indices(len(frame_files), gap)
            run_specs = [("uniform", uniform, None)]
            budget = len(uniform)
            max_segment_length = gap * args.max_segment_multiplier
            for method, source_model, alpha in METHOD_SPECS:
                costs = calibrated_costs(predictions[(sample, source_model)], gap, alpha)
                selected, selector_cost = dp_select_from_costs(len(frame_files), costs, budget, max_segment_length)
                run_specs.append((method, selected, selector_cost))
            for method, indices, selector_cost in run_specs:
                cache_key = (sample, tuple(indices))
                if cache_key not in eval_cache:
                    print(f"=== Stage48 sample={sample} method={method} gap={gap} ===", flush=True)
                    eval_cache[cache_key] = selected_records(indices, anchor_map, model, frame_files, opt, background)
                metrics = eval_cache[cache_key]
                row = flatten_eval(sample, method, gap, indices, metrics, checkpoint_path)
                row.update(segment_fields(indices))
                row["estimated_q8_static_mib_per_frame"] = q8_static_mib_per_frame(len(indices), len(frame_files))
                row["selector_cost"] = selector_cost
                rows.append(row)
                selections.append({"sample": sample, "method": method, "reference_gap": gap, "indices": indices, "selector_cost": selector_cost})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = args.summary_root / "stage48_predicted_adaptive_selector_rd.csv"
    comparison_csv = args.summary_root / "stage48_selector_comparison.csv"
    aggregate_csv = args.summary_root / "stage48_selector_aggregates.csv"
    summary_path = args.summary_root / "stage48_predicted_adaptive_selector_rd_summary.json"
    write_csv(rows, csv_path)
    comparisons = selector_comparison(rows)
    write_comparison_csv(comparisons, comparison_csv)
    aggregates = aggregate_comparisons(comparisons)
    write_aggregate_csv(aggregates, aggregate_csv)
    best_method = max(aggregates, key=lambda key: (aggregates[key]["positive_all_points"], aggregates[key]["mean_delta_all_psnr"]))
    plots = []
    for metric_key, ylabel, filename in [
        ("adapter_all_psnr", "Adapter all-frame PSNR (dB)", "stage48_best_adapter_all_psnr_rd.png"),
        ("adapter_middle_psnr", "Adapter middle-only PSNR (dB)", "stage48_best_adapter_middle_psnr_rd.png"),
    ]:
        out_path = args.summary_root / filename
        plot_best(rows, best_method, metric_key, ylabel, out_path)
        plots.append(str(out_path))
    summary = {
        "stage": 48,
        "mode": "feed-forward predicted adaptive selector RD",
        "stage47_predictions": str(args.stage47_predictions),
        "csv": str(csv_path),
        "comparison_csv": str(comparison_csv),
        "aggregate_csv": str(aggregate_csv),
        "plots": plots,
        "rows": rows,
        "selections": selections,
        "comparisons": comparisons,
        "aggregates": aggregates,
        "best_method_by_positive_then_mean_all_psnr": best_method,
        "best_aggregate": aggregates[best_method],
        "notes": "This is the first fully feed-forward predicted selector evaluation from Stage47 rendered-cost predictions. It uses encoder-side features only plus deterministic DP; no rendered oracle cost is used at selection time.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "comparison_csv": str(comparison_csv),
        "aggregate_csv": str(aggregate_csv),
        "plots": plots,
        "best_method": best_method,
        "best_aggregate": aggregates[best_method],
        "aggregates": aggregates,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
