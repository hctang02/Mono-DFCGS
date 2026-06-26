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
DEFAULT_STAGE44_CSV = REPO_ROOT / "experiments/stage44_rendered_segment_distortion_dataset/stage44_rendered_segment_distortion_dataset.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage45b_calibrated_rendered_oracle_selector_rd"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import q8_static_mib_per_frame, uniform_indices  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import load_adapter  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import build_anchor_index, flatten_eval, read_manifest_rows, selected_records  # noqa: E402
from scripts.run_stage39_predicted_selector_rd import segment_fields, selector_comparison, write_comparison_csv, write_csv  # noqa: E402
from scripts.run_stage45_rendered_oracle_adaptive_selector_rd import read_stage44_costs  # noqa: E402


CALIBRATIONS = [
    ("rendered_raw", 0.0, "none"),
    ("rendered_prior_0p1", 0.1, "none"),
    ("rendered_prior_0p3", 0.3, "none"),
    ("rendered_prior_1p0", 1.0, "none"),
    ("rendered_min2", 0.0, "two"),
    ("rendered_min2_prior_0p3", 0.3, "two"),
    ("rendered_minhalf", 0.0, "half"),
    ("rendered_minhalf_prior_0p3", 0.3, "half"),
]


def min_segment_length(gap, mode):
    if mode == "none":
        return 1
    if mode == "two":
        return 2
    if mode == "half":
        return max(2, gap // 2)
    raise ValueError(mode)


def calibrated_costs(costs, gap, alpha):
    out = {}
    for (a, b), cost in costs.items():
        length = b - a
        prior = ((length - gap) / gap) ** 2
        out[(a, b)] = float(cost + alpha * prior)
    return out


def dp_select_from_costs_minlen(total_frames, costs, budget, max_segment_length, min_segment):
    inf = float("inf")
    dp = [[inf] * total_frames for _ in range(budget)]
    prev = [[None] * total_frames for _ in range(budget)]
    dp[0][0] = 0.0
    for used in range(1, budget):
        for b in range(1, total_frames):
            start = max(0, b - max_segment_length)
            end = b - min_segment
            if end < start:
                continue
            best = inf
            best_a = None
            for a in range(start, end + 1):
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
        raise RuntimeError(f"No feasible selection total_frames={total_frames} budget={budget} min_segment={min_segment}")
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
            "min_delta_middle_psnr": float(np.min([row["selector_delta_adapter_middle_psnr"] for row in rows])),
            "exact_uniform_points": sum(1 for row in rows if abs(row["selector_delta_adapter_all_psnr"]) < 1e-12),
        }
    return aggregates


def write_aggregate_csv(aggregates, path):
    fields = [
        "method", "count", "positive_all_points", "positive_middle_points", "mean_delta_all_psnr",
        "mean_delta_middle_psnr", "min_delta_all_psnr", "min_delta_middle_psnr", "exact_uniform_points",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method, values in aggregates.items():
            writer.writerow({"method": method, **values})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--stage25_root", type=Path, default=DEFAULT_STAGE25_ROOT)
    parser.add_argument("--stage33_manifest", type=Path, default=DEFAULT_STAGE33_MANIFEST)
    parser.add_argument("--stage44_csv", type=Path, default=DEFAULT_STAGE44_CSV)
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
    stage44_costs = read_stage44_costs(args.stage44_csv, args.samples)
    rows = []
    selections = []
    eval_cache = {}
    for sample in args.samples:
        print(f"=== Stage45b sample={sample} ===", flush=True)
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
            for method, alpha, min_mode in CALIBRATIONS:
                min_len = min_segment_length(gap, min_mode)
                costs = calibrated_costs(stage44_costs[sample], gap, alpha)
                selected, selector_cost = dp_select_from_costs_minlen(len(frame_files), costs, budget, max_segment_length, min_len)
                run_specs.append((method, selected, selector_cost))
            for method, indices, selector_cost in run_specs:
                cache_key = (sample, tuple(indices))
                if cache_key not in eval_cache:
                    print(f"=== Stage45b sample={sample} method={method} gap={gap} ===", flush=True)
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

    csv_path = args.summary_root / "stage45b_calibrated_rendered_oracle_selector_rd.csv"
    comparison_csv = args.summary_root / "stage45b_selector_comparison.csv"
    aggregate_csv = args.summary_root / "stage45b_selector_aggregates.csv"
    summary_path = args.summary_root / "stage45b_calibrated_rendered_oracle_selector_rd_summary.json"
    write_csv(rows, csv_path)
    comparisons = selector_comparison(rows)
    write_comparison_csv(comparisons, comparison_csv)
    aggregates = aggregate_comparisons(comparisons)
    write_aggregate_csv(aggregates, aggregate_csv)
    best_method = max(aggregates, key=lambda key: (aggregates[key]["positive_all_points"], aggregates[key]["mean_delta_all_psnr"]))
    summary = {
        "stage": "45b",
        "mode": "calibrated rendered-distortion oracle selector RD",
        "stage44_csv": str(args.stage44_csv),
        "csv": str(csv_path),
        "comparison_csv": str(comparison_csv),
        "aggregate_csv": str(aggregate_csv),
        "rows": rows,
        "selections": selections,
        "comparisons": comparisons,
        "aggregates": aggregates,
        "best_method_by_positive_then_mean_all_psnr": best_method,
        "notes": "Scans uniform-prior and minimum-segment-length calibration over Stage44 rendered oracle costs. Still oracle analysis, not deployable predictor.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "comparison_csv": str(comparison_csv),
        "aggregate_csv": str(aggregate_csv),
        "best_method": best_method,
        "best_aggregate": aggregates[best_method],
        "aggregates": aggregates,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
