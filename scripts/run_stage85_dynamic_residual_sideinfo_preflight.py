import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_MANIFEST = REPO_ROOT / "experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_tasks.csv"
DEFAULT_DENSE_MANIFEST = REPO_ROOT / "experiments/stage61_davis_anchor_export_data_full/stage61_davis_anchor_export_manifest.csv"
DEFAULT_ADAPTER = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage85_dynamic_residual_sideinfo_preflight"


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.anchor_predictor import GaussianAnchorDynamicPredictor  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, uniform_dequantize, uniform_quantize, unflatten_static_anchor  # noqa: E402


ROW_FIELDS = [
    "task_id",
    "sequence",
    "codec",
    "reference_gap",
    "target_index",
    "method",
    "side_bits",
    "keep_fraction",
    "keep_gaussians",
    "side_info_mib_per_intermediate_frame",
    "baseline_anchor_mse",
    "residual_anchor_mse_after_sideinfo",
    "captured_energy_fraction",
    "residual_mse_reduction_fraction",
]

SUMMARY_FIELDS = [
    "method",
    "codec",
    "reference_gap",
    "side_bits",
    "keep_fraction",
    "task_count",
    "mean_side_info_mib_per_intermediate_frame",
    "mean_baseline_anchor_mse",
    "mean_residual_anchor_mse_after_sideinfo",
    "mean_captured_energy_fraction",
    "mean_residual_mse_reduction_fraction",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_task_rows(path, task_split, codecs, gaps):
    codec_set = set(codecs)
    gap_set = set(gaps)
    rows = []
    for row in read_csv(path):
        if row["task_split"] != task_split:
            continue
        if codec_set and row["codec"] not in codec_set:
            continue
        gap = int(row["reference_gap"])
        if gap_set and gap not in gap_set:
            continue
        row = dict(row)
        row["bits"] = int(row["bits"])
        row["reference_gap"] = gap
        row["left_index"] = int(row["left_index"])
        row["right_index"] = int(row["right_index"])
        row["target_index"] = int(row["target_index"])
        row["normalized_time"] = float(row["normalized_time"])
        rows.append(row)
    return rows


def select_balanced(rows, max_tasks, seed):
    if max_tasks <= 0 or len(rows) <= max_tasks:
        return rows
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["sequence"], row["codec"], row["reference_gap"])].append(row)
    for items in groups.values():
        rng.shuffle(items)
    keys = sorted(groups)
    rng.shuffle(keys)
    selected = []
    offset = 0
    while len(selected) < max_tasks:
        progressed = False
        for key in keys:
            items = groups[key]
            if offset >= len(items):
                continue
            selected.append(items[offset])
            progressed = True
            if len(selected) >= max_tasks:
                break
        if not progressed:
            break
        offset += 1
    return selected


def build_dense_index(path, split_filter):
    split_set = set(split_filter)
    index = {}
    for row in read_csv(path):
        if row["split"] not in split_set or int(row["frame_gap"]) != 1:
            continue
        item = row["dataset_item"]
        if not Path(item).exists():
            continue
        key_base = (row["dataset"], row["split"], row["sequence"])
        left_key = (*key_base, int(row["left_index"]))
        right_key = (*key_base, int(row["right_index"]))
        index.setdefault(left_key, (item, "left_anchor"))
        index.setdefault(right_key, (item, "right_anchor"))
    return index


def anchor_to_device(anchor, device):
    return {key: value.unsqueeze(0).float().to(device) for key, value in anchor.items()}


def maybe_quantize_anchor(anchor, bits):
    if bits <= 0:
        return anchor
    attrs = flatten_static_anchor(anchor)
    q, mins, scales = uniform_quantize(attrs, bits=bits)
    return unflatten_static_anchor(uniform_dequantize(q, mins, scales))


def load_anchor(source_item, source_side, device, bits=None, cache=None):
    key = (source_item, source_side, bits, str(device))
    if cache is not None and key in cache:
        return cache[key]
    item = torch.load(source_item, map_location="cpu", weights_only=True)
    anchor = anchor_to_device(item[source_side], device)
    if bits is not None:
        anchor = maybe_quantize_anchor(anchor, bits)
    if cache is not None:
        cache[key] = anchor
    return anchor


def linear_anchor(left, right, t):
    left_attrs = flatten_static_anchor(left)
    right_attrs = flatten_static_anchor(right)
    t_scalar = torch.tensor([t], dtype=left_attrs.dtype, device=left_attrs.device).reshape(1, 1, 1)
    return unflatten_static_anchor(left_attrs * (1.0 - t_scalar) + right_attrs * t_scalar)


def safetensors_device(device):
    if device.type == "cuda" and device.index is None:
        return "cuda:0"
    return str(device)


def load_adapter(path, hidden_dim, device):
    model = GaussianAnchorDynamicPredictor(hidden_dim=hidden_dim, apply_output_constraints=False, zero_init_residual=True).to(device)
    state = load_file(str(path), device=safetensors_device(device))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def side_info_mib(gaussian_count, keep_count, attr_dim, side_bits):
    if keep_count <= 0:
        return 0.0
    index_bits = math.ceil(math.log2(max(gaussian_count, 2)))
    bits = keep_count * (index_bits + attr_dim * side_bits)
    return bits / 8.0 / (1024.0 * 1024.0)


def residual_rows_for_method(task, method, pred_attrs, target_attrs, side_bits_list, keep_fractions):
    residual = target_attrs - pred_attrs
    energy_per_gaussian = torch.sum(residual.reshape(residual.shape[1], residual.shape[2]) ** 2, dim=-1)
    total_energy = float(torch.sum(energy_per_gaussian).item())
    value_count = int(residual.numel())
    gaussian_count = int(energy_per_gaussian.numel())
    attr_dim = int(residual.shape[-1])
    baseline_mse = total_energy / max(value_count, 1)
    sorted_energy = torch.sort(energy_per_gaussian, descending=True).values
    rows = []
    for side_bits in side_bits_list:
        for keep_fraction in keep_fractions:
            keep_count = int(round(gaussian_count * keep_fraction))
            keep_count = min(max(keep_count, 0), gaussian_count)
            kept_energy = float(torch.sum(sorted_energy[:keep_count]).item()) if keep_count > 0 else 0.0
            remaining_energy = max(total_energy - kept_energy, 0.0)
            residual_mse = remaining_energy / max(value_count, 1)
            captured = kept_energy / total_energy if total_energy > 0.0 else 0.0
            reduction = 1.0 - residual_mse / baseline_mse if baseline_mse > 0.0 else 0.0
            rows.append({
                "task_id": task["task_id"],
                "sequence": task["sequence"],
                "codec": task["codec"],
                "reference_gap": task["reference_gap"],
                "target_index": task["target_index"],
                "method": method,
                "side_bits": side_bits,
                "keep_fraction": keep_fraction,
                "keep_gaussians": keep_count,
                "side_info_mib_per_intermediate_frame": side_info_mib(gaussian_count, keep_count, attr_dim, side_bits),
                "baseline_anchor_mse": baseline_mse,
                "residual_anchor_mse_after_sideinfo": residual_mse,
                "captured_energy_fraction": captured,
                "residual_mse_reduction_fraction": reduction,
            })
    return rows


def average(rows, key):
    values = [float(row[key]) for row in rows]
    return sum(values) / max(len(values), 1)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["method"], row["codec"], row["reference_gap"], row["side_bits"], row["keep_fraction"])
        grouped[key].append(row)
    out = []
    for (method, codec, gap, side_bits, keep_fraction), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3], item[0][4])):
        out.append({
            "method": method,
            "codec": codec,
            "reference_gap": gap,
            "side_bits": side_bits,
            "keep_fraction": keep_fraction,
            "task_count": len(items),
            "mean_side_info_mib_per_intermediate_frame": average(items, "side_info_mib_per_intermediate_frame"),
            "mean_baseline_anchor_mse": average(items, "baseline_anchor_mse"),
            "mean_residual_anchor_mse_after_sideinfo": average(items, "residual_anchor_mse_after_sideinfo"),
            "mean_captured_energy_fraction": average(items, "captured_energy_fraction"),
            "mean_residual_mse_reduction_fraction": average(items, "residual_mse_reduction_fraction"),
        })
    return out


def write_report(summary, summary_rows, path):
    lines = [
        "# Stage85 Dynamic Residual Side-Info Preflight",
        "",
        "## Configuration",
        "",
        f"- task count: `{summary['task_count']}`",
        f"- codecs: `{summary['codecs']}`",
        f"- gaps: `{summary['gaps']}`",
        f"- keep fractions: `{summary['keep_fractions']}`",
        f"- side bits: `{summary['side_bits']}`",
        "",
        "## Key Rows",
        "",
        "| method | codec | gap | bits | keep | side MiB/frame | captured energy | residual reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    key_rows = [row for row in summary_rows if float(row["keep_fraction"]) in {0.0, 0.05, 0.1, 0.25} and int(row["side_bits"]) == 8]
    for row in key_rows:
        lines.append(
            f"| {row['method']} | {row['codec']} | {row['reference_gap']} | {row['side_bits']} | {row['keep_fraction']} | {row['mean_side_info_mib_per_intermediate_frame']} | {row['mean_captured_energy_fraction']} | {row['mean_residual_mse_reduction_fraction']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- This is an optimistic anchor-space estimate, not rendered RD.",
        "- Any transmitted residual side-info must be counted as side-info rate and total rate.",
        "- Top-k estimates include index bits plus quantized residual attribute bits; quantization distortion is not modeled yet.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--task_split", default="eval")
    parser.add_argument("--codecs", nargs="+", default=["q10", "q12"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--max_tasks", type=int, default=60)
    parser.add_argument("--keep_fractions", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.1, 0.25, 1.0])
    parser.add_argument("--side_bits", nargs="+", type=int, default=[6, 8])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = select_balanced(parse_task_rows(args.task_manifest, args.task_split, args.codecs, args.gaps), args.max_tasks, args.seed)
    if not rows:
        raise RuntimeError("No tasks selected")
    dense_index = build_dense_index(args.dense_manifest, sorted({row["split"] for row in rows}))
    model = load_adapter(args.adapter, args.hidden_dim, device)
    cache = {}
    result_rows = []
    with torch.no_grad():
        for task in rows:
            left = load_anchor(task["left_anchor_source_item"], task["left_anchor_source_side"], device, bits=task["bits"], cache=cache)
            right = load_anchor(task["right_anchor_source_item"], task["right_anchor_source_side"], device, bits=task["bits"], cache=cache)
            dense_key = (task["dataset"], task["split"], task["sequence"], task["target_index"])
            if dense_key not in dense_index:
                raise KeyError(f"Missing dense target anchor for {dense_key}")
            target_item, target_side = dense_index[dense_key]
            target = load_anchor(target_item, target_side, device, bits=None, cache=cache)
            target_attrs = flatten_static_anchor(target)
            linear_attrs = flatten_static_anchor(linear_anchor(left, right, task["normalized_time"]))
            t = torch.tensor([task["normalized_time"]], dtype=torch.float32, device=device)
            adapter_attrs = flatten_static_anchor(model(left, right, t, apply_output_constraints=False))
            result_rows.extend(residual_rows_for_method(task, "linear", linear_attrs, target_attrs, args.side_bits, args.keep_fractions))
            result_rows.extend(residual_rows_for_method(task, "stage65_adapter", adapter_attrs, target_attrs, args.side_bits, args.keep_fractions))

    summary_rows = summarize(result_rows)
    rows_csv = args.summary_root / "stage85_dynamic_residual_sideinfo_rows.csv"
    summary_csv = args.summary_root / "stage85_dynamic_residual_sideinfo_summary.csv"
    summary_json = args.summary_root / "stage85_dynamic_residual_sideinfo_preflight_summary.json"
    report_md = args.summary_root / "stage85_dynamic_residual_sideinfo_preflight_report.md"
    write_csv(result_rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    summary = {
        "stage": 85,
        "mode": "dynamic residual side-info anchor-space preflight",
        "task_manifest": str(args.task_manifest),
        "dense_manifest": str(args.dense_manifest),
        "adapter": str(args.adapter),
        "task_split": args.task_split,
        "codecs": args.codecs,
        "gaps": args.gaps,
        "task_count": len(rows),
        "keep_fractions": args.keep_fractions,
        "side_bits": args.side_bits,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "notes": [
            "Anchor-space only; no rendering or RD claim.",
            "Residual side-info would be transmitted information and must be added to total rate.",
            "Top-k side-info estimate includes index bits and residual attribute bits but not quantization distortion.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, summary_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "task_count": len(rows),
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
