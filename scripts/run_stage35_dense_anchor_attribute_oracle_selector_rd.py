import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_CACHE = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage1_streamsplat_fair_metrics/cache")
DEFAULT_STAGE25_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage25_leave_one_out_adapter_training")
DEFAULT_STAGE33_MANIFEST = REPO_ROOT / "experiments/stage33_dense_gap1_anchor_export/stage33_dense_gap1_anchor_manifest.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage35_dense_anchor_attribute_oracle_selector_rd"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import uniform_indices  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import load_adapter  # noqa: E402
from scripts.run_stage27_anchor_available_selector_rd import build_anchor_index, flatten_eval, read_manifest_rows, selected_records  # noqa: E402
from scripts.run_stage29_anchor_attribute_oracle_selector_rd import (  # noqa: E402
    compute_cost_matrix,
    dp_select,
    plot_delta,
    selector_comparison,
    write_comparison_csv,
    write_result_csv,
)


def segment_stats(indices):
    lengths = [b - a for a, b in zip(indices[:-1], indices[1:])]
    return {
        "segment_lengths": " ".join(str(v) for v in lengths),
        "max_segment_length": int(max(lengths)),
        "mean_segment_length": float(np.mean(lengths)),
    }


def build_dense_oracle_selections(sample, args, model, anchor_map, total_frames):
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
            "method": "dense_anchor_attr_oracle",
            "reference_gap": gap,
            "indices": oracle_indices,
            "selector_anchor_mse_cost": oracle_cost,
            **segment_stats(oracle_indices),
        })
    return selections


def normalize_comparison_methods(rows):
    normalized = []
    for row in rows:
        row = dict(row)
        if row["method"] == "dense_anchor_attr_oracle":
            row["method"] = "anchor_attr_oracle"
        normalized.append(row)
    return selector_comparison(normalized)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--stage25_root", type=Path, default=DEFAULT_STAGE25_ROOT)
    parser.add_argument("--stage33_manifest", type=Path, default=DEFAULT_STAGE33_MANIFEST)
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

    rows = []
    selections = []
    coverage = []
    for sample in args.samples:
        frame_files = sorted((args.stage1_cache / sample / "frames").glob("*.png"))
        if not frame_files:
            raise RuntimeError(f"Missing frame cache for {sample}")
        manifest_rows = read_manifest_rows(args.stage33_manifest, sample)
        anchor_map = build_anchor_index(manifest_rows, device, args.quant_bits)
        missing = sorted(set(range(len(frame_files))) - set(anchor_map))
        if missing:
            raise RuntimeError(f"Stage33 dense anchors missing frames for {sample}: {missing[:10]}")
        coverage.append({"sample": sample, "total_frames": len(frame_files), "anchor_count": len(anchor_map)})
        checkpoint_path = args.stage25_root / sample / "stage25_best_anchor_adapter.safetensors"
        model = load_adapter(checkpoint_path, args.hidden_dim, device)
        sample_selections = build_dense_oracle_selections(sample, args, model, anchor_map, len(frame_files))
        selections.extend(sample_selections)
        for selection in sample_selections:
            print(f"=== Stage35 sample={sample} method={selection['method']} gap={selection['reference_gap']} ===", flush=True)
            metrics = selected_records(selection["indices"], anchor_map, model, frame_files, opt, background)
            row = flatten_eval(sample, selection["method"], selection["reference_gap"], selection["indices"], metrics, checkpoint_path)
            row["selector_anchor_mse_cost"] = selection["selector_anchor_mse_cost"]
            rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = args.summary_root / "stage35_dense_anchor_attribute_oracle_selector_rd.csv"
    comparison_csv = args.summary_root / "stage35_selector_comparison.csv"
    summary_path = args.summary_root / "stage35_dense_anchor_attribute_oracle_selector_rd_summary.json"
    write_result_csv(rows, csv_path)
    comparisons = normalize_comparison_methods(rows)
    write_comparison_csv(comparisons, comparison_csv)
    plots = []
    for key, ylabel, title, filename in [
        ("selector_delta_adapter_all_psnr", "Dense oracle - uniform all PSNR (dB)", "Stage 35 Dense Anchor-Attribute Oracle Selector Gain: All PSNR", "stage35_delta_all_psnr.png"),
        ("selector_delta_adapter_middle_psnr", "Dense oracle - uniform middle PSNR (dB)", "Stage 35 Dense Anchor-Attribute Oracle Selector Gain: Middle PSNR", "stage35_delta_middle_psnr.png"),
    ]:
        out_path = args.summary_root / filename
        plot_delta(comparisons, key, ylabel, title, out_path)
        plots.append(str(out_path))

    all_values = [row["selector_delta_adapter_all_psnr"] for row in comparisons]
    middle_values = [row["selector_delta_adapter_middle_psnr"] for row in comparisons]
    summary = {
        "stage": 35,
        "mode": "dense anchor-attribute oracle/proxy selected-keyframe anchor-only RD",
        "stage33_manifest": str(args.stage33_manifest),
        "samples": args.samples,
        "gaps": args.gaps,
        "methods": ["uniform", "dense_anchor_attr_oracle"],
        "max_segment_multiplier": args.max_segment_multiplier,
        "csv": str(csv_path),
        "comparison_csv": str(comparison_csv),
        "plots": plots,
        "coverage": coverage,
        "rows": rows,
        "selections": selections,
        "comparisons": comparisons,
        "mean_selector_delta_adapter_all_psnr": float(np.mean(all_values)),
        "mean_selector_delta_adapter_middle_psnr": float(np.mean(middle_values)),
        "positive_selector_all_points": sum(1 for value in all_values if value > 0.0),
        "positive_selector_middle_points": sum(1 for value in middle_values if value > 0.0),
        "notes": "Uses dense Stage33 anchors as oracle/proxy targets. This is an upper-bound selector, not a deployable encoder-side selector.",
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
