import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage87_quantized_residual_sideinfo_smoke"
DEFAULT_OUTPUT_PREFIX = "stage87_quantized_residual_sideinfo"
DEFAULT_SUMMARY_PREFIX = "stage87_quantized_residual_sideinfo_smoke"


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.gaussian_codec import flatten_static_anchor, unflatten_static_anchor  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import (  # noqa: E402
    DEFAULT_ADAPTER,
    DEFAULT_DENSE_MANIFEST,
    DEFAULT_TASK_MANIFEST,
    build_dense_index,
    linear_anchor,
    load_adapter,
    load_anchor,
    parse_task_rows,
    select_balanced,
)
from scripts.run_stage86_rendered_residual_sideinfo_smoke import render_psnr  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import load_rgb  # noqa: E402
from configs.options_inference import Options  # noqa: E402


ROW_FIELDS = [
    "task_id",
    "sequence",
    "codec",
    "reference_gap",
    "target_index",
    "base_method",
    "keep_fraction",
    "keep_gaussians",
    "side_bits",
    "metadata_bits_per_attr_value",
    "side_info_mib_per_intermediate_frame",
    "base_psnr",
    "sideinfo_psnr",
    "delta_psnr_vs_base",
]

SUMMARY_FIELDS = [
    "base_method",
    "codec",
    "reference_gap",
    "keep_fraction",
    "side_bits",
    "task_count",
    "mean_side_info_mib_per_intermediate_frame",
    "mean_base_psnr",
    "mean_sideinfo_psnr",
    "mean_delta_psnr_vs_base",
    "positive_delta_count",
]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def quantize_values(values, bits, eps=1e-8):
    if values.numel() == 0:
        return values
    qmax = (1 << bits) - 1
    mins = values.amin(dim=0, keepdim=True)
    maxs = values.amax(dim=0, keepdim=True)
    scales = (maxs - mins).clamp_min(eps) / qmax
    q = torch.round((values - mins) / scales).clamp(0, qmax)
    return q * scales + mins


def side_info_mib_with_metadata(gaussian_count, keep_count, attr_dim, side_bits, metadata_bits_per_attr_value):
    if keep_count <= 0:
        return 0.0
    index_bits = math.ceil(math.log2(max(gaussian_count, 2)))
    payload_bits = keep_count * (index_bits + attr_dim * side_bits)
    metadata_bits = attr_dim * 2 * metadata_bits_per_attr_value
    return (payload_bits + metadata_bits) / 8.0 / (1024.0 * 1024.0)


def apply_quantized_topk_residual(base_attrs, target_attrs, keep_fraction, side_bits):
    residual = target_attrs - base_attrs
    gaussian_count = int(residual.shape[1])
    keep_count = min(max(int(round(gaussian_count * keep_fraction)), 0), gaussian_count)
    if keep_count <= 0:
        return unflatten_static_anchor(base_attrs), keep_count
    energy = torch.sum(residual[0] ** 2, dim=-1)
    keep_idx = torch.topk(energy, k=keep_count, largest=True).indices
    kept = residual[0, keep_idx, :]
    dequantized = quantize_values(kept, side_bits)
    masked = torch.zeros_like(residual)
    masked[0, keep_idx, :] = dequantized
    return unflatten_static_anchor(base_attrs + masked), keep_count


def average(rows, key):
    values = [float(row[key]) for row in rows]
    return sum(values) / max(len(values), 1)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["base_method"], row["codec"], row["reference_gap"], row["keep_fraction"], row["side_bits"])].append(row)
    out = []
    for (method, codec, gap, keep, bits), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3], item[0][4])):
        out.append({
            "base_method": method,
            "codec": codec,
            "reference_gap": gap,
            "keep_fraction": keep,
            "side_bits": bits,
            "task_count": len(items),
            "mean_side_info_mib_per_intermediate_frame": average(items, "side_info_mib_per_intermediate_frame"),
            "mean_base_psnr": average(items, "base_psnr"),
            "mean_sideinfo_psnr": average(items, "sideinfo_psnr"),
            "mean_delta_psnr_vs_base": average(items, "delta_psnr_vs_base"),
            "positive_delta_count": sum(1 for row in items if float(row["delta_psnr_vs_base"]) > 0.0),
        })
    return out


def write_report(summary, summary_rows, path):
    lines = [
        f"# {summary['report_title']}",
        "",
        "## Configuration",
        "",
        f"- task count: `{summary['task_count']}`",
        f"- codecs: `{summary['codecs']}`",
        f"- gaps: `{summary['gaps']}`",
        f"- keep fractions: `{summary['keep_fractions']}`",
        f"- side bits: `{summary['side_bits']}`",
        f"- metadata bits per min/max value: `{summary['metadata_bits_per_attr_value']}`",
        "",
        "## Summary",
        "",
        "| base | codec | gap | keep | bits | side MiB/intermediate | base PSNR | side PSNR | delta | positives |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['base_method']} | {row['codec']} | {row['reference_gap']} | {row['keep_fraction']} | {row['side_bits']} | {row['mean_side_info_mib_per_intermediate_frame']} | {row['mean_base_psnr']} | {row['mean_sideinfo_psnr']} | {row['mean_delta_psnr_vs_base']} | {row['positive_delta_count']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Residual values are quantized per frame and per attribute over the kept Gaussian set.",
        "- Rate includes Gaussian indices, quantized residual attrs, and per-attribute min/max metadata.",
        "- This is still a smoke test; entropy coding and full-video RD are not implemented here.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--stage", type=int, default=87)
    parser.add_argument("--mode", default="quantized rendered residual side-info smoke")
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--summary_prefix", default=DEFAULT_SUMMARY_PREFIX)
    parser.add_argument("--report_title", default="Stage87 Quantized Residual Side-Info Smoke")
    parser.add_argument("--task_split", default="eval")
    parser.add_argument("--codecs", nargs="+", default=["q12"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--max_tasks", type=int, default=12)
    parser.add_argument("--keep_fractions", nargs="+", type=float, default=[0.1, 0.25])
    parser.add_argument("--side_bits", nargs="+", type=int, default=[6, 8])
    parser.add_argument("--metadata_bits_per_attr_value", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260628)
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
    tasks = select_balanced(parse_task_rows(args.task_manifest, args.task_split, args.codecs, args.gaps), args.max_tasks, args.seed)
    if not tasks:
        raise RuntimeError("No tasks selected")
    dense_index = build_dense_index(args.dense_manifest, sorted({row["split"] for row in tasks}))
    model = load_adapter(args.adapter, args.hidden_dim, device)
    cache = {}
    rows = []
    with torch.no_grad():
        for task in tasks:
            left = load_anchor(task["left_anchor_source_item"], task["left_anchor_source_side"], device, bits=task["bits"], cache=cache)
            right = load_anchor(task["right_anchor_source_item"], task["right_anchor_source_side"], device, bits=task["bits"], cache=cache)
            dense_key = (task["dataset"], task["split"], task["sequence"], task["target_index"])
            if dense_key not in dense_index:
                raise KeyError(f"Missing dense target anchor for {dense_key}")
            target_item, target_side = dense_index[dense_key]
            target_anchor = load_anchor(target_item, target_side, device, bits=None, cache=cache)
            target_attrs = flatten_static_anchor(target_anchor)
            target_rgb = load_rgb(task["target_rgb_path"], opt.image_height, opt.image_width, device)
            t = torch.tensor([task["normalized_time"]], dtype=torch.float32, device=device)
            base_attrs_by_method = {
                "linear": flatten_static_anchor(linear_anchor(left, right, task["normalized_time"])),
                "stage65_adapter": flatten_static_anchor(model(left, right, t, apply_output_constraints=False)),
            }
            for method, base_attrs in base_attrs_by_method.items():
                base_anchor = unflatten_static_anchor(base_attrs)
                base_psnr = render_psnr(base_anchor, target_rgb, background, opt)
                for keep_fraction in args.keep_fractions:
                    for side_bits in args.side_bits:
                        side_anchor, keep_count = apply_quantized_topk_residual(base_attrs, target_attrs, keep_fraction, side_bits)
                        side_psnr = render_psnr(side_anchor, target_rgb, background, opt)
                        rows.append({
                            "task_id": task["task_id"],
                            "sequence": task["sequence"],
                            "codec": task["codec"],
                            "reference_gap": task["reference_gap"],
                            "target_index": task["target_index"],
                            "base_method": method,
                            "keep_fraction": keep_fraction,
                            "keep_gaussians": keep_count,
                            "side_bits": side_bits,
                            "metadata_bits_per_attr_value": args.metadata_bits_per_attr_value,
                            "side_info_mib_per_intermediate_frame": side_info_mib_with_metadata(
                                int(base_attrs.shape[1]), keep_count, int(base_attrs.shape[-1]), side_bits, args.metadata_bits_per_attr_value
                            ),
                            "base_psnr": base_psnr,
                            "sideinfo_psnr": side_psnr,
                            "delta_psnr_vs_base": side_psnr - base_psnr,
                        })
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_rows = summarize(rows)
    rows_csv = args.summary_root / f"{args.output_prefix}_rows.csv"
    summary_csv = args.summary_root / f"{args.output_prefix}_summary.csv"
    summary_json = args.summary_root / f"{args.summary_prefix}_summary.json"
    report_md = args.summary_root / f"{args.summary_prefix}_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    summary = {
        "stage": args.stage,
        "mode": args.mode,
        "report_title": args.report_title,
        "task_manifest": str(args.task_manifest),
        "dense_manifest": str(args.dense_manifest),
        "adapter": str(args.adapter),
        "task_split": args.task_split,
        "codecs": args.codecs,
        "gaps": args.gaps,
        "task_count": len(tasks),
        "keep_fractions": args.keep_fractions,
        "side_bits": args.side_bits,
        "metadata_bits_per_attr_value": args.metadata_bits_per_attr_value,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "notes": [
            "Residual values are quantized per frame and per attribute over kept Gaussians.",
            "Rate includes indices, residual values, and min/max metadata.",
            "Entropy coding and full-video RD are not implemented in this smoke.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, summary_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "task_count": len(tasks),
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
