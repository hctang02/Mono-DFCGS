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
DEFAULT_STAGE44_CSV = REPO_ROOT / "experiments/stage44_rendered_segment_distortion_dataset/stage44_rendered_segment_distortion_dataset.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage45_rendered_oracle_adaptive_selector_rd"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import q8_static_mib_per_frame, uniform_indices  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import load_adapter  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import build_anchor_index, flatten_eval, read_manifest_rows, selected_records  # noqa: E402
from scripts.run_stage39_predicted_selector_rd import dp_select_from_costs, segment_fields, selector_comparison, write_comparison_csv, write_csv  # noqa: E402


def read_stage44_costs(path, samples):
    sample_set = set(samples)
    costs = defaultdict(dict)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = row["sample"]
            if sample not in sample_set:
                continue
            if row["heldout_fold"] != sample or row["split"] != "eval":
                continue
            a = int(row["left_index"])
            b = int(row["right_index"])
            costs[sample][(a, b)] = float(row["adapter_mse_sum_est"])
    return costs


def plot_rd(rows, metric_key, ylabel, out_path):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sample"]].append(row)
    samples = sorted(grouped)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=False, sharey=False)
    axes = axes.flatten()
    colors = {"uniform": "#1f77b4", "rendered_distortion_oracle": "#d62728"}
    for ax, sample in zip(axes, samples):
        sample_rows = grouped[sample]
        for method in ["uniform", "rendered_distortion_oracle"]:
            pts = sorted([row for row in sample_rows if row["method"] == method], key=lambda row: row["estimated_q8_static_mib_per_frame"])
            ax.plot(
                [row["estimated_q8_static_mib_per_frame"] for row in pts],
                [row[metric_key] for row in pts],
                marker="o",
                label=method,
                color=colors[method],
            )
        ax.set_title(sample)
        ax.set_xlabel("Estimated q8 anchor MiB/frame")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


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
    for sample in args.samples:
        print(f"=== Stage45 sample={sample} ===", flush=True)
        frame_files = sorted((args.stage1_cache / sample / "frames").glob("*.png"))
        manifest_rows = read_manifest_rows(args.stage33_manifest, sample)
        anchor_map = build_anchor_index(manifest_rows, device, args.quant_bits)
        checkpoint_path = args.stage25_root / sample / "stage25_best_anchor_adapter.safetensors"
        model = load_adapter(checkpoint_path, args.hidden_dim, device)
        for gap in args.gaps:
            uniform = uniform_indices(len(frame_files), gap)
            budget = len(uniform)
            max_segment_length = gap * args.max_segment_multiplier
            selected, selector_cost = dp_select_from_costs(len(frame_files), stage44_costs[sample], budget, max_segment_length)
            for method, indices, cost in [("uniform", uniform, None), ("rendered_distortion_oracle", selected, selector_cost)]:
                print(f"=== Stage45 sample={sample} method={method} gap={gap} ===", flush=True)
                metrics = selected_records(indices, anchor_map, model, frame_files, opt, background)
                row = flatten_eval(sample, method, gap, indices, metrics, checkpoint_path)
                row.update(segment_fields(indices))
                row["estimated_q8_static_mib_per_frame"] = q8_static_mib_per_frame(len(indices), len(frame_files))
                row["selector_cost"] = cost
                rows.append(row)
                selections.append({"sample": sample, "method": method, "reference_gap": gap, "indices": indices, "selector_cost": cost})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = args.summary_root / "stage45_rendered_oracle_adaptive_selector_rd.csv"
    comparison_csv = args.summary_root / "stage45_selector_comparison.csv"
    summary_path = args.summary_root / "stage45_rendered_oracle_adaptive_selector_rd_summary.json"
    write_csv(rows, csv_path)
    comparisons = selector_comparison(rows)
    write_comparison_csv(comparisons, comparison_csv)
    plots = []
    for metric_key, ylabel, filename in [
        ("adapter_all_psnr", "Adapter all-frame PSNR (dB)", "stage45_adapter_all_psnr_rd.png"),
        ("adapter_middle_psnr", "Adapter middle-only PSNR (dB)", "stage45_adapter_middle_psnr_rd.png"),
        ("adapter_all_ssim", "Adapter all-frame SSIM", "stage45_adapter_all_ssim_rd.png"),
        ("adapter_middle_ssim", "Adapter middle-only SSIM", "stage45_adapter_middle_ssim_rd.png"),
    ]:
        out_path = args.summary_root / filename
        plot_rd(rows, metric_key, ylabel, out_path)
        plots.append(str(out_path))
    oracle_rows = [row for row in comparisons if row["method"] == "rendered_distortion_oracle"]
    aggregates = {
        "count": len(oracle_rows),
        "mean_delta_all_psnr": float(np.mean([row["selector_delta_adapter_all_psnr"] for row in oracle_rows])),
        "mean_delta_middle_psnr": float(np.mean([row["selector_delta_adapter_middle_psnr"] for row in oracle_rows])),
        "positive_all_points": sum(1 for row in oracle_rows if row["selector_delta_adapter_all_psnr"] > 0.0),
        "positive_middle_points": sum(1 for row in oracle_rows if row["selector_delta_adapter_middle_psnr"] > 0.0),
    }
    summary = {
        "stage": 45,
        "mode": "rendered-distortion oracle adaptive selector RD",
        "stage44_csv": str(args.stage44_csv),
        "csv": str(csv_path),
        "comparison_csv": str(comparison_csv),
        "plots": plots,
        "rows": rows,
        "selections": selections,
        "comparisons": comparisons,
        "aggregates": aggregates,
        "notes": "Uses Stage44 sampled rendered segment distortion labels as oracle DP costs. This is an upper-bound analysis, not the final feed-forward deployable selector.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "csv": str(csv_path), "comparison_csv": str(comparison_csv), "plots": plots, "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
