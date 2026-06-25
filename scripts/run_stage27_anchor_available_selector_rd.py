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
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage27_anchor_available_selector_rd"
SAMPLES = ["n3dv", "meetroom", "driving", "robot"]


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage16_segment_error_keyframe_selection import (  # noqa: E402
    gaussian_point_scores,
    motion_edge_scores,
    normalize,
    point_to_edge_scores,
    q8_static_mib_per_frame,
    read_stage6_rows,
    segment_cost,
    segment_stats,
    uniform_indices,
)
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import linear_anchor, render_anchor, render_prediction  # noqa: E402
from scripts.run_stage23_full_video_anchor_only_evaluator import (  # noqa: E402
    frame_psnr,
    frame_ssim,
    load_adapter,
    load_rgb_numpy,
    maybe_quantize_anchor,
    summarize_metrics,
    tensor_to_numpy_rgb,
)


def read_manifest_rows(path, sample):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sample"] != sample:
                continue
            if not Path(row["dataset_item"]).exists():
                continue
            row["frame_gap"] = int(row["frame_gap"])
            row["left_index"] = int(row["left_index"])
            row["right_index"] = int(row["right_index"])
            rows.append(row)
    return rows


def anchor_to_device(anchor, device):
    return {key: value.unsqueeze(0).float().to(device) for key, value in anchor.items()}


def build_anchor_index(manifest_rows, device, quant_bits):
    anchor_map = {}
    for row in sorted(manifest_rows, key=lambda r: (r["frame_gap"], r["left_index"])):
        item = torch.load(row["dataset_item"], map_location="cpu", weights_only=True)
        for side in ["left", "right"]:
            idx = int(item[f"{side}_index"])
            if idx in anchor_map:
                continue
            anchor = anchor_to_device(item[f"{side}_anchor"], device)
            anchor_map[idx] = maybe_quantize_anchor(anchor, quant_bits)
    return anchor_map


def best_split_constrained(edge_scores, a, b, candidates, max_segment_length):
    base = segment_cost(edge_scores, a, b, max_segment_length)
    best_idx = None
    best_gain = -float("inf")
    for idx in candidates:
        if idx <= a or idx >= b:
            continue
        gain = base - segment_cost(edge_scores, a, idx, max_segment_length) - segment_cost(edge_scores, idx, b, max_segment_length)
        if gain > best_gain:
            best_idx = idx
            best_gain = gain
    return best_idx, best_gain


def constrained_segment_indices(edge_scores, candidate_indices, budget, max_segment_length):
    selected = {candidate_indices[0], candidate_indices[-1]}
    candidates = sorted(set(candidate_indices))
    while len(selected) < budget:
        ordered = sorted(selected)
        best_idx = None
        best_gain = -float("inf")
        best_length = -1
        for a, b in zip(ordered[:-1], ordered[1:]):
            if b - a <= 1:
                continue
            idx, gain = best_split_constrained(edge_scores, a, b, candidates, max_segment_length)
            if idx is None or idx in selected:
                continue
            length = b - a
            if gain > best_gain or (gain == best_gain and length > best_length):
                best_idx = idx
                best_gain = gain
                best_length = length
        if best_idx is None:
            break
        selected.add(best_idx)
    return sorted(selected)


def selected_records(indices, anchor_map, model, frame_files, opt, background):
    height, width = opt.image_height, opt.image_width
    keyframes = set(indices)
    linear_records = {}
    adapter_records = {}

    with torch.no_grad():
        for idx in indices:
            ref = load_rgb_numpy(frame_files[idx], height, width)
            pred = tensor_to_numpy_rgb(render_anchor(anchor_map[idx], background, opt))
            metric = {"frame_index": idx, "is_keyframe": True, "psnr": frame_psnr(ref, pred), "ssim": frame_ssim(ref, pred)}
            linear_records[idx] = metric
            adapter_records[idx] = dict(metric)

        for a, b in zip(indices[:-1], indices[1:]):
            left = anchor_map[a]
            right = anchor_map[b]
            length = b - a
            for frame_idx in range(a + 1, b):
                ref = load_rgb_numpy(frame_files[frame_idx], height, width)
                t = (frame_idx - a) / length
                linear_pred = tensor_to_numpy_rgb(render_anchor(linear_anchor(left, right, t), background, opt))
                adapter_pred = tensor_to_numpy_rgb(render_prediction(model, {
                    "left": left,
                    "right": right,
                    "normalized_time": t,
                }, background, opt))
                linear_records[frame_idx] = {
                    "frame_index": frame_idx,
                    "is_keyframe": False,
                    "psnr": frame_psnr(ref, linear_pred),
                    "ssim": frame_ssim(ref, linear_pred),
                }
                adapter_records[frame_idx] = {
                    "frame_index": frame_idx,
                    "is_keyframe": False,
                    "psnr": frame_psnr(ref, adapter_pred),
                    "ssim": frame_ssim(ref, adapter_pred),
                }

    total_frames = len(frame_files)
    missing = sorted(set(range(total_frames)) - set(linear_records))
    if missing:
        raise RuntimeError(f"Selected indices do not cover full video: missing={missing[:10]}")

    def method_summary(records):
        all_records = [records[idx] for idx in range(total_frames)]
        middle_records = [records[idx] for idx in range(total_frames) if idx not in keyframes]
        given_records = [records[idx] for idx in range(total_frames) if idx in keyframes]
        return {
            "all": summarize_metrics(all_records),
            "middle_only": summarize_metrics(middle_records),
            "given_keyframes": summarize_metrics(given_records),
        }

    linear_summary = method_summary(linear_records)
    adapter_summary = method_summary(adapter_records)
    return {
        "linear": linear_summary,
        "adapter": adapter_summary,
        "delta_all_psnr": adapter_summary["all"]["psnr_avg"] - linear_summary["all"]["psnr_avg"],
        "delta_middle_psnr": adapter_summary["middle_only"]["psnr_avg"] - linear_summary["middle_only"]["psnr_avg"],
        "delta_given_psnr": adapter_summary["given_keyframes"]["psnr_avg"] - linear_summary["given_keyframes"]["psnr_avg"],
    }


def flatten_eval(sample, method, reference_gap, indices, metrics, checkpoint_path):
    total_frames = metrics["linear"]["all"]["count"]
    return {
        "sample": sample,
        "method": method,
        "reference_gap": reference_gap,
        "total_frames": total_frames,
        "keyframe_count": len(indices),
        "keyframe_ratio": len(indices) / total_frames,
        "estimated_q8_static_mib_per_frame": q8_static_mib_per_frame(len(indices), total_frames),
        "max_segment_length": max(b - a for a, b in zip(indices[:-1], indices[1:])),
        "mean_segment_length": float(np.mean([b - a for a, b in zip(indices[:-1], indices[1:])])),
        "segment_lengths": " ".join(str(b - a) for a, b in zip(indices[:-1], indices[1:])),
        "indices": " ".join(str(idx) for idx in indices),
        "linear_all_psnr": metrics["linear"]["all"]["psnr_avg"],
        "adapter_all_psnr": metrics["adapter"]["all"]["psnr_avg"],
        "delta_all_psnr": metrics["delta_all_psnr"],
        "linear_middle_psnr": metrics["linear"]["middle_only"]["psnr_avg"],
        "adapter_middle_psnr": metrics["adapter"]["middle_only"]["psnr_avg"],
        "delta_middle_psnr": metrics["delta_middle_psnr"],
        "linear_given_psnr": metrics["linear"]["given_keyframes"]["psnr_avg"],
        "adapter_given_psnr": metrics["adapter"]["given_keyframes"]["psnr_avg"],
        "delta_given_psnr": metrics["delta_given_psnr"],
        "linear_all_ssim": metrics["linear"]["all"]["ssim_avg"],
        "adapter_all_ssim": metrics["adapter"]["all"]["ssim_avg"],
        "linear_middle_ssim": metrics["linear"]["middle_only"]["ssim_avg"],
        "adapter_middle_ssim": metrics["adapter"]["middle_only"]["ssim_avg"],
        "checkpoint": str(checkpoint_path),
    }


def build_selections(sample, args, frame_files, anchor_indices):
    total_frames = len(frame_files)
    motion_edges = normalize(motion_edge_scores(frame_files, args.score_image_size))
    g_rows = read_stage6_rows(args.stage6_manifest, sample, frame_gap=2)
    gaussian_edges = normalize(point_to_edge_scores(gaussian_point_scores(total_frames, g_rows))) if g_rows else np.zeros(total_frames - 1)
    rd_edges = args.motion_weight * motion_edges + (1.0 - args.motion_weight) * gaussian_edges

    selections = []
    anchor_set = set(anchor_indices)
    for gap in args.gaps:
        uniform = uniform_indices(total_frames, gap)
        missing = [idx for idx in uniform if idx not in anchor_set]
        if missing:
            raise RuntimeError(f"Uniform gap={gap} selected unavailable anchors for {sample}: {missing[:10]}")
        budget = len(uniform)
        constrained = constrained_segment_indices(rd_edges, anchor_indices, budget, gap * args.max_segment_multiplier)
        selections.append({"sample": sample, "method": "uniform", "reference_gap": gap, "indices": uniform, **segment_stats(uniform)})
        selections.append({"sample": sample, "method": "anchor_segment_rd", "reference_gap": gap, "indices": constrained, **segment_stats(constrained)})
    return selections


def write_csv(rows, path):
    fields = [
        "sample", "method", "reference_gap", "total_frames", "keyframe_count", "keyframe_ratio",
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
        if "uniform" not in methods or "anchor_segment_rd" not in methods:
            continue
        uniform = methods["uniform"]
        selected = methods["anchor_segment_rd"]
        out.append({
            "sample": sample,
            "reference_gap": gap,
            "rate_mib_per_frame": selected["estimated_q8_static_mib_per_frame"],
            "selector_delta_adapter_all_psnr": selected["adapter_all_psnr"] - uniform["adapter_all_psnr"],
            "selector_delta_adapter_middle_psnr": selected["adapter_middle_psnr"] - uniform["adapter_middle_psnr"],
            "selector_delta_linear_all_psnr": selected["linear_all_psnr"] - uniform["linear_all_psnr"],
            "selector_delta_linear_middle_psnr": selected["linear_middle_psnr"] - uniform["linear_middle_psnr"],
        })
    return out


def write_comparison_csv(rows, path):
    fields = [
        "sample", "reference_gap", "rate_mib_per_frame", "selector_delta_adapter_all_psnr",
        "selector_delta_adapter_middle_psnr", "selector_delta_linear_all_psnr", "selector_delta_linear_middle_psnr",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_method_rd(rows, metric_key, ylabel, title, out_path):
    samples = sorted({row["sample"] for row in rows})
    methods = ["uniform", "anchor_segment_rd"]
    colors = {"uniform": "#1f77b4", "anchor_segment_rd": "#2ca02c"}
    markers = {"uniform": "o", "anchor_segment_rd": "^"}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["sample"], row["method"])].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharey=False)
    axes = axes.flatten()
    for ax, sample in zip(axes, samples):
        for method in methods:
            points = sorted(grouped[(sample, method)], key=lambda row: row["estimated_q8_static_mib_per_frame"])
            xs = [row["estimated_q8_static_mib_per_frame"] for row in points]
            ys = [row[metric_key] for row in points]
            labels = [f"g{row['reference_gap']}" for row in points]
            ax.plot(xs, ys, marker=markers[method], color=colors[method], linewidth=2.0, label=method)
            for x, y, label in zip(xs, ys, labels):
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_title(sample)
        ax.set_xlabel("Transmitted q8 static Gaussian MiB/frame")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(frameon=False)
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
    parser.add_argument("--score_image_size", type=int, default=128)
    parser.add_argument("--max_segment_multiplier", type=int, default=2)
    parser.add_argument("--motion_weight", type=float, default=0.7)
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
    anchor_coverage = []
    for sample in args.samples:
        frame_files = sorted((args.stage1_cache / sample / "frames").glob("*.png"))
        if not frame_files:
            raise RuntimeError(f"Missing frame cache for {sample}")
        manifest_rows = read_manifest_rows(args.stage6_manifest, sample)
        anchor_map = build_anchor_index(manifest_rows, device, args.quant_bits)
        anchor_indices = sorted(anchor_map)
        if anchor_indices[0] != 0 or anchor_indices[-1] != len(frame_files) - 1:
            raise RuntimeError(f"Anchor endpoints do not cover full video for {sample}: {anchor_indices[0]}..{anchor_indices[-1]} total={len(frame_files)}")
        checkpoint_path = args.stage25_root / sample / "stage25_best_anchor_adapter.safetensors"
        model = load_adapter(checkpoint_path, args.hidden_dim, device)
        sample_selections = build_selections(sample, args, frame_files, anchor_indices)
        selections.extend(sample_selections)
        anchor_coverage.append({
            "sample": sample,
            "total_frames": len(frame_files),
            "available_anchor_count": len(anchor_indices),
            "available_anchor_indices": " ".join(str(idx) for idx in anchor_indices),
        })
        for selection in sample_selections:
            print(f"=== Stage27 sample={sample} method={selection['method']} gap={selection['reference_gap']} ===", flush=True)
            metrics = selected_records(selection["indices"], anchor_map, model, frame_files, opt, background)
            rows.append(flatten_eval(sample, selection["method"], selection["reference_gap"], selection["indices"], metrics, checkpoint_path))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = args.summary_root / "stage27_anchor_available_selector_rd.csv"
    comparison_csv = args.summary_root / "stage27_selector_comparison.csv"
    summary_path = args.summary_root / "stage27_anchor_available_selector_rd_summary.json"
    write_csv(rows, csv_path)
    comparisons = selector_comparison(rows)
    write_comparison_csv(comparisons, comparison_csv)
    plots = []
    for metric_key, ylabel, title, filename in [
        ("adapter_all_psnr", "Adapter all-frame PSNR (dB)", "Stage 27 Anchor-Available Selector RD: All PSNR", "stage27_adapter_all_psnr_rd.png"),
        ("adapter_middle_psnr", "Adapter middle-only PSNR (dB)", "Stage 27 Anchor-Available Selector RD: Middle PSNR", "stage27_adapter_middle_psnr_rd.png"),
    ]:
        out_path = args.summary_root / filename
        plot_method_rd(rows, metric_key, ylabel, title, out_path)
        plots.append(str(out_path))

    selector_all = [row["selector_delta_adapter_all_psnr"] for row in comparisons]
    selector_middle = [row["selector_delta_adapter_middle_psnr"] for row in comparisons]
    summary = {
        "stage": 27,
        "mode": "anchor-available constrained selected-keyframe anchor-only RD",
        "samples": args.samples,
        "gaps": args.gaps,
        "methods": ["uniform", "anchor_segment_rd"],
        "csv": str(csv_path),
        "comparison_csv": str(comparison_csv),
        "plots": plots,
        "rows": rows,
        "comparisons": comparisons,
        "anchor_coverage": anchor_coverage,
        "mean_selector_delta_adapter_all_psnr": float(np.mean(selector_all)),
        "mean_selector_delta_adapter_middle_psnr": float(np.mean(selector_middle)),
        "positive_selector_all_points": sum(1 for value in selector_all if value > 0.0),
        "positive_selector_middle_points": sum(1 for value in selector_middle if value > 0.0),
        "notes": "Stage6 currently provides q8 anchors only at even endpoint frames. The selector is therefore constrained to available anchor indices instead of using unconstrained Stage16 odd-frame selections.",
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
