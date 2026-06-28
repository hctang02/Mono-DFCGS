import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage93_residual_sideinfo_entropy_codec_smoke"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, unflatten_static_anchor  # noqa: E402
from mono_dfcgs.residual_sideinfo_codec import (  # noqa: E402
    decode_residual_sideinfo,
    decode_residual_sideinfo_entropy,
    encode_topk_residual_sideinfo,
    encode_topk_residual_sideinfo_entropy,
)
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import load_rgb  # noqa: E402
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


ROW_FIELDS = [
    "task_id",
    "sequence",
    "codec",
    "reference_gap",
    "target_index",
    "base_method",
    "keep_fraction",
    "side_bits",
    "keep_gaussians",
    "fixed_payload_bytes",
    "entropy_payload_bytes",
    "fixed_mib_per_intermediate_frame",
    "entropy_mib_per_intermediate_frame",
    "entropy_ratio_vs_fixed",
    "entropy_savings_bytes_vs_fixed",
    "entropy_header_bytes",
    "metadata_zlib_bytes",
    "delta_zlib_bytes",
    "residual_zlib_bytes",
    "decoded_max_abs_diff_vs_fixed",
    "base_psnr",
    "entropy_psnr",
    "delta_psnr_vs_base",
]

SUMMARY_FIELDS = [
    "base_method",
    "codec",
    "reference_gap",
    "task_count",
    "keep_fraction",
    "side_bits",
    "mean_fixed_payload_bytes",
    "mean_entropy_payload_bytes",
    "mean_fixed_mib_per_intermediate_frame",
    "mean_entropy_mib_per_intermediate_frame",
    "mean_entropy_ratio_vs_fixed",
    "mean_entropy_savings_bytes_vs_fixed",
    "max_decoded_max_abs_diff_vs_fixed",
    "mean_base_psnr",
    "mean_entropy_psnr",
    "mean_delta_psnr_vs_base",
    "positive_delta_count",
]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def average(rows, key):
    values = [float(row[key]) for row in rows]
    return sum(values) / max(len(values), 1)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["base_method"], row["codec"], row["reference_gap"])].append(row)
    out = []
    for (method, codec, gap), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        out.append({
            "base_method": method,
            "codec": codec,
            "reference_gap": gap,
            "task_count": len(items),
            "keep_fraction": items[0]["keep_fraction"],
            "side_bits": items[0]["side_bits"],
            "mean_fixed_payload_bytes": average(items, "fixed_payload_bytes"),
            "mean_entropy_payload_bytes": average(items, "entropy_payload_bytes"),
            "mean_fixed_mib_per_intermediate_frame": average(items, "fixed_mib_per_intermediate_frame"),
            "mean_entropy_mib_per_intermediate_frame": average(items, "entropy_mib_per_intermediate_frame"),
            "mean_entropy_ratio_vs_fixed": average(items, "entropy_ratio_vs_fixed"),
            "mean_entropy_savings_bytes_vs_fixed": average(items, "entropy_savings_bytes_vs_fixed"),
            "max_decoded_max_abs_diff_vs_fixed": max(float(row["decoded_max_abs_diff_vs_fixed"]) for row in items),
            "mean_base_psnr": average(items, "base_psnr"),
            "mean_entropy_psnr": average(items, "entropy_psnr"),
            "mean_delta_psnr_vs_base": average(items, "delta_psnr_vs_base"),
            "positive_delta_count": sum(1 for row in items if float(row["delta_psnr_vs_base"]) > 0.0),
        })
    return out


def write_report(summary, summary_rows, path):
    lines = [
        "# Stage93 Entropy-Coded Residual Side-Info Codec Smoke",
        "",
        "## Configuration",
        "",
        f"- task count: `{summary['task_count']}`",
        f"- codecs: `{summary['codecs']}`",
        f"- gaps: `{summary['gaps']}`",
        f"- keep fraction: `{summary['keep_fraction']}`",
        f"- side bits: `{summary['side_bits']}`",
        f"- zlib level: `{summary['zlib_level']}`",
        "",
        "## Summary",
        "",
        "| base | codec | gap | tasks | fixed bytes | entropy bytes | ratio | entropy MiB/intermediate | max decode diff | entropy PSNR | delta | positives |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['base_method']} | {row['codec']} | {row['reference_gap']} | {row['task_count']} | {row['mean_fixed_payload_bytes']} | {row['mean_entropy_payload_bytes']} | {row['mean_entropy_ratio_vs_fixed']} | {row['mean_entropy_mib_per_intermediate_frame']} | {row['max_decoded_max_abs_diff_vs_fixed']} | {row['mean_entropy_psnr']} | {row['mean_delta_psnr_vs_base']} | {row['positive_delta_count']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Entropy payload uses sorted-index deltas and zlib-compressed metadata/index/residual components.",
        "- Decode is compared against Stage91 fixed decode before rendering.",
        "- Residuals are still teacher-derived; this is a codec smoke, not a deployable residual predictor.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--task_split", default="eval")
    parser.add_argument("--codecs", nargs="+", default=["q12"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--max_tasks", type=int, default=12)
    parser.add_argument("--keep_fraction", type=float, default=0.1)
    parser.add_argument("--side_bits", type=int, default=6)
    parser.add_argument("--zlib_level", type=int, default=9)
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
                fixed_payload, fixed_info = encode_topk_residual_sideinfo(base_attrs, target_attrs, args.keep_fraction, args.side_bits)
                fixed_decoded_attrs = decode_residual_sideinfo(base_attrs, fixed_payload)
                entropy_payload, entropy_info = encode_topk_residual_sideinfo_entropy(
                    base_attrs,
                    target_attrs,
                    args.keep_fraction,
                    args.side_bits,
                    zlib_level=args.zlib_level,
                )
                entropy_decoded_attrs = decode_residual_sideinfo_entropy(base_attrs, entropy_payload)
                decoded_diff = torch.max(torch.abs(fixed_decoded_attrs - entropy_decoded_attrs)).item()
                entropy_anchor = unflatten_static_anchor(entropy_decoded_attrs)
                entropy_psnr = render_psnr(entropy_anchor, target_rgb, background, opt)
                rows.append({
                    "task_id": task["task_id"],
                    "sequence": task["sequence"],
                    "codec": task["codec"],
                    "reference_gap": task["reference_gap"],
                    "target_index": task["target_index"],
                    "base_method": method,
                    "keep_fraction": args.keep_fraction,
                    "side_bits": args.side_bits,
                    "keep_gaussians": entropy_info["keep_count"],
                    "fixed_payload_bytes": fixed_info["payload_bytes"],
                    "entropy_payload_bytes": entropy_info["payload_bytes"],
                    "fixed_mib_per_intermediate_frame": fixed_info["payload_bytes"] / (1024.0 * 1024.0),
                    "entropy_mib_per_intermediate_frame": entropy_info["payload_bytes"] / (1024.0 * 1024.0),
                    "entropy_ratio_vs_fixed": entropy_info["payload_bytes"] / fixed_info["payload_bytes"],
                    "entropy_savings_bytes_vs_fixed": fixed_info["payload_bytes"] - entropy_info["payload_bytes"],
                    "entropy_header_bytes": entropy_info["header_bytes"],
                    "metadata_zlib_bytes": entropy_info["metadata_zlib_bytes"],
                    "delta_zlib_bytes": entropy_info["delta_zlib_bytes"],
                    "residual_zlib_bytes": entropy_info["residual_zlib_bytes"],
                    "decoded_max_abs_diff_vs_fixed": decoded_diff,
                    "base_psnr": base_psnr,
                    "entropy_psnr": entropy_psnr,
                    "delta_psnr_vs_base": entropy_psnr - base_psnr,
                })
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_rows = summarize(rows)
    rows_csv = args.summary_root / "stage93_residual_sideinfo_entropy_codec_rows.csv"
    summary_csv = args.summary_root / "stage93_residual_sideinfo_entropy_codec_summary.csv"
    summary_json = args.summary_root / "stage93_residual_sideinfo_entropy_codec_summary.json"
    report_md = args.summary_root / "stage93_residual_sideinfo_entropy_codec_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    summary = {
        "stage": 93,
        "mode": "entropy-coded residual side-info codec smoke",
        "task_manifest": str(args.task_manifest),
        "dense_manifest": str(args.dense_manifest),
        "adapter": str(args.adapter),
        "task_split": args.task_split,
        "codecs": args.codecs,
        "gaps": args.gaps,
        "task_count": len(tasks),
        "keep_fraction": args.keep_fraction,
        "side_bits": args.side_bits,
        "zlib_level": args.zlib_level,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "notes": [
            "Entropy payload uses sorted index deltas and zlib-compressed components.",
            "Entropy decode is compared to fixed decode before rendering.",
            "Residuals are teacher-derived from dense target anchors.",
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
