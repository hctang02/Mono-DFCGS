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
DEFAULT_STAGE6_MANIFEST = REPO_ROOT / "experiments/stage6_real_anchor_dataset/stage6_real_anchor_dataset_manifest.csv"
DEFAULT_STAGE25_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage25_leave_one_out_adapter_training")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage29_anchor_attribute_oracle_selector_rd"
SAMPLES = ["n3dv", "meetroom", "driving", "robot"]


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import q8_static_mib_per_frame, segment_stats, uniform_indices  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import load_adapter  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import (  # noqa: E402
    build_anchor_index,
    flatten_eval,
    read_manifest_rows,
    selected_records,
)


def segment_anchor_mse(model, anchor_map, attrs_map, a, b):
    mids = [idx for idx in sorted(anchor_map) if a < idx < b]
    if not mids:
        return 0.0
    left = anchor_map[a]
    right = anchor_map[b]
    length = b - a
    costs = []
    with torch.no_grad():
        for idx in mids:
            t = torch.tensor([(idx - a) / length], dtype=torch.float32, device=next(iter(left.values())).device)
            pred = model(left, right, t, apply_output_constraints=False)
            pred_attrs = flatten_static_anchor(pred)
            costs.append(float(torch.mean((pred_attrs - attrs_map[idx]) ** 2).item()))
    return float(sum(costs))


def compute_cost_matrix(model, anchor_map, max_segment_length):
    indices = sorted(anchor_map)
    attrs_map = {idx: flatten_static_anchor(anchor_map[idx]) for idx in indices}
    costs = {}
    for i, a in enumerate(indices[:-1]):
        for j in range(i + 1, len(indices)):
            b = indices[j]
            if max_segment_length > 0 and b - a > max_segment_length:
                break
            costs[(i, j)] = segment_anchor_mse(model, anchor_map, attrs_map, a, b)
    return indices, costs


def dp_select(indices, costs, keyframe_count):
    n = len(indices)
    inf = float("inf")
    dp = [[inf] * n for _ in range(keyframe_count)]
    prev = [[None] * n for _ in range(keyframe_count)]
    dp[0][0] = 0.0
    for used in range(1, keyframe_count):
        for j in range(1, n):
            best = inf
            best_i = None
            for i in range(j):
                if (i, j) not in costs or dp[used - 1][i] == inf:
                    continue
                value = dp[used - 1][i] + costs[(i, j)]
                if value < best:
                    best = value
                    best_i = i
            dp[used][j] = best
            prev[used][j] = best_i
    if dp[keyframe_count - 1][n - 1] == inf:
        raise RuntimeError(f"No feasible selection for keyframe_count={keyframe_count}")
    selected_positions = []
    used = keyframe_count - 1
    pos = n - 1
    while pos is not None:
        selected_positions.append(pos)
        pos = prev[used][pos]
        used -= 1
        if used < 0:
            break
    selected_positions.reverse()
    return [indices[pos] for pos in selected_positions], float(dp[keyframe_count - 1][n - 1])


def build_selections_for_sample(sample, args, model, anchor_map, total_frames):
    selections = []
    cost_cache = {}
    for gap in args.gaps:
        uniform = uniform_indices(total_frames, gap)
        budget = len(uniform)
        max_segment_length = gap * args.max_segment_multiplier
        if max_segment_length not in cost_cache:
            cost_cache[max_segment_length] = compute_cost_matrix(model, anchor_map, max_segment_length)
        candidate_indices, costs = cost_cache[max_segment_length]
        oracle_indices, oracle_cost = dp_select(candidate_indices, costs, budget)
        selections.append({
            "sample": sample,
            "method": "uniform",
            "reference_gap": gap,
            "indices": uniform,
            "selector_anchor_mse_cost": None,
            **segment_stats(uniform),
        })
        selections.append({
            "sample": sample,
            "method": "anchor_attr_oracle",
            "reference_gap": gap,
            "indices": oracle_indices,
            "selector_anchor_mse_cost": oracle_cost,
            **segment_stats(oracle_indices),
        })
    return selections


def selector_comparison(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["sample"], row["reference_gap"])][row["method"]] = row
    out = []
    for (sample, gap), methods in sorted(grouped.items()):
        if "uniform" not in methods or "anchor_attr_oracle" not in methods:
            continue
        uniform = methods["uniform"]
        oracle = methods["anchor_attr_oracle"]
        out.append({
            "sample": sample,
            "reference_gap": gap,
            "rate_mib_per_frame": oracle["estimated_q8_static_mib_per_frame"],
            "selector_delta_adapter_all_psnr": oracle["adapter_all_psnr"] - uniform["adapter_all_psnr"],
            "selector_delta_adapter_middle_psnr": oracle["adapter_middle_psnr"] - uniform["adapter_middle_psnr"],
            "selector_delta_linear_all_psnr": oracle["linear_all_psnr"] - uniform["linear_all_psnr"],
            "selector_delta_linear_middle_psnr": oracle["linear_middle_psnr"] - uniform["linear_middle_psnr"],
        })
    return out


def write_result_csv(rows, path):
    fields = [
        "sample", "method", "reference_gap", "total_frames", "keyframe_count", "keyframe_ratio",
        "estimated_q8_static_mib_per_frame", "max_segment_length", "mean_segment_length", "segment_lengths", "indices",
        "selector_anchor_mse_cost", "linear_all_psnr", "adapter_all_psnr", "delta_all_psnr", "linear_middle_psnr",
        "adapter_middle_psnr", "delta_middle_psnr", "linear_given_psnr", "adapter_given_psnr", "delta_given_psnr",
        "linear_all_ssim", "adapter_all_ssim", "linear_middle_ssim", "adapter_middle_ssim", "checkpoint",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_csv(rows, path):
    fields = [
        "sample", "reference_gap", "rate_mib_per_frame", "selector_delta_adapter_all_psnr",
        "selector_delta_adapter_middle_psnr", "selector_delta_linear_all_psnr", "selector_delta_linear_middle_psnr",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_delta(comparisons, key, ylabel, title, out_path):
    samples = sorted({row["sample"] for row in comparisons})
    grouped = defaultdict(list)
    for row in comparisons:
        grouped[row["sample"]].append(row)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharey=True)
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        rows = sorted(grouped[sample], key=lambda row: row["reference_gap"])
        gaps = [str(row["reference_gap"]) for row in rows]
        values = [row[key] for row in rows]
        ax.bar(gaps, values, color="#9467bd")
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
    parser.add_argument("--stage6_manifest", type=Path, default=DEFAULT_STAGE6_MANIFEST)
    parser.add_argument("--stage25_root", type=Path, default=DEFAULT_STAGE25_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--samples", nargs="*", default=SAMPLES)
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

    rows = []
    selections = []
    for sample in args.samples:
        frame_files = sorted((args.stage1_cache / sample / "frames").glob("*.png"))
        if not frame_files:
            raise RuntimeError(f"Missing frame cache for {sample}")
        manifest_rows = read_manifest_rows(args.stage6_manifest, sample)
        anchor_map = build_anchor_index(manifest_rows, device, args.quant_bits)
        checkpoint_path = args.stage25_root / sample / "stage25_best_anchor_adapter.safetensors"
        model = load_adapter(checkpoint_path, args.hidden_dim, device)
        sample_selections = build_selections_for_sample(sample, args, model, anchor_map, len(frame_files))
        selections.extend(sample_selections)
        for selection in sample_selections:
            print(f"=== Stage29 sample={sample} method={selection['method']} gap={selection['reference_gap']} ===", flush=True)
            metrics = selected_records(selection["indices"], anchor_map, model, frame_files, opt, background)
            row = flatten_eval(sample, selection["method"], selection["reference_gap"], selection["indices"], metrics, checkpoint_path)
            row["selector_anchor_mse_cost"] = selection["selector_anchor_mse_cost"]
            rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = args.summary_root / "stage29_anchor_attribute_oracle_selector_rd.csv"
    comparison_csv = args.summary_root / "stage29_selector_comparison.csv"
    summary_path = args.summary_root / "stage29_anchor_attribute_oracle_selector_rd_summary.json"
    write_result_csv(rows, csv_path)
    comparisons = selector_comparison(rows)
    write_comparison_csv(comparisons, comparison_csv)
    plots = []
    for key, ylabel, title, filename in [
        ("selector_delta_adapter_all_psnr", "Oracle selector - uniform all PSNR (dB)", "Stage 29 Anchor-Attribute Oracle Selector Gain: All PSNR", "stage29_delta_all_psnr.png"),
        ("selector_delta_adapter_middle_psnr", "Oracle selector - uniform middle PSNR (dB)", "Stage 29 Anchor-Attribute Oracle Selector Gain: Middle PSNR", "stage29_delta_middle_psnr.png"),
    ]:
        out_path = args.summary_root / filename
        plot_delta(comparisons, key, ylabel, title, out_path)
        plots.append(str(out_path))

    all_values = [row["selector_delta_adapter_all_psnr"] for row in comparisons]
    middle_values = [row["selector_delta_adapter_middle_psnr"] for row in comparisons]
    summary = {
        "stage": 29,
        "mode": "anchor-attribute oracle/proxy selected-keyframe anchor-only RD",
        "samples": args.samples,
        "gaps": args.gaps,
        "methods": ["uniform", "anchor_attr_oracle"],
        "max_segment_multiplier": args.max_segment_multiplier,
        "csv": str(csv_path),
        "comparison_csv": str(comparison_csv),
        "plots": plots,
        "rows": rows,
        "selections": selections,
        "comparisons": comparisons,
        "mean_selector_delta_adapter_all_psnr": float(np.mean(all_values)),
        "mean_selector_delta_adapter_middle_psnr": float(np.mean(middle_values)),
        "positive_selector_all_points": sum(1 for value in all_values if value > 0.0),
        "positive_selector_middle_points": sum(1 for value in middle_values if value > 0.0),
        "notes": "Selector uses held-out sample intermediate q8 anchors to minimize adapter-predicted anchor-attribute MSE. This is an oracle/proxy upper-bound, not a causal deployed selector.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "comparison_csv": str(comparison_csv),
        "rows": len(rows),
        "mean_selector_delta_adapter_all_psnr": summary["mean_selector_delta_adapter_all_psnr"],
        "mean_selector_delta_adapter_middle_psnr": summary["mean_selector_delta_adapter_middle_psnr"],
        "positive_selector_all_points": summary["positive_selector_all_points"],
        "positive_selector_middle_points": summary["positive_selector_middle_points"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
