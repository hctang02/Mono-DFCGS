import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE114_POLICY = REPO_ROOT / "experiments/stage114_strict_safe_selector_fallback_package/stage114_strict_safe_selector_fallback_policy.json"
DEFAULT_STAGE116_ROWS = REPO_ROOT / "experiments/stage116_deterministic_vs_entropy_sideinfo_accounting/stage116_deterministic_vs_entropy_sideinfo_accounting_rows.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage119_actual_compressed_deterministic_sweep"

DEFAULT_KEEP_FRACTIONS = [0.025, 0.05, 0.1, 0.15, 0.2]
DEFAULT_SIDE_BITS = [2, 3, 4, 5, 6, 8]


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402
from mono_dfcgs.residual_sideinfo_codec import (  # noqa: E402
    decode_selected_residual_values_sideinfo,
    decode_selected_residual_values_sideinfo_entropy,
    encode_selected_residual_values_sideinfo,
    encode_selected_residual_values_sideinfo_entropy,
)
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


ROW_FIELDS = [
    "task_id",
    "sequence",
    "codec",
    "reference_gap",
    "target_index",
    "base_method",
    "selector_policy",
    "selected_candidate",
    "keep_fraction",
    "side_bits",
    "zlib_level",
    "keep_gaussians",
    "raw_deterministic_payload_bytes",
    "compressed_deterministic_payload_bytes",
    "raw_deterministic_mib_per_intermediate_frame",
    "compressed_deterministic_mib_per_intermediate_frame",
    "compressed_savings_bytes_vs_raw",
    "compressed_ratio_vs_raw",
    "metadata_zlib_bytes",
    "residual_zlib_bytes",
    "omitted_index_bytes_vs_fixed_aligned",
    "stage96_entropy_reference_bytes",
    "stage96_entropy_reference_mib_per_intermediate_frame",
    "compressed_minus_stage96_entropy_reference_bytes",
    "compressed_ratio_vs_stage96_entropy_reference",
    "compressed_below_stage96_entropy_reference",
    "decoded_max_abs_diff_vs_raw_deterministic",
    "comparison_note",
]

GROUP_SUMMARY_FIELDS = [
    "base_method",
    "codec",
    "reference_gap",
    "keep_fraction",
    "side_bits",
    "task_count",
    "selector_policy",
    "selected_candidate",
    "zlib_level",
    "mean_keep_gaussians",
    "mean_raw_deterministic_payload_bytes",
    "mean_compressed_deterministic_payload_bytes",
    "mean_raw_deterministic_mib_per_intermediate_frame",
    "mean_compressed_deterministic_mib_per_intermediate_frame",
    "mean_compressed_savings_bytes_vs_raw",
    "mean_compressed_ratio_vs_raw",
    "mean_stage96_entropy_reference_bytes",
    "mean_stage96_entropy_reference_mib_per_intermediate_frame",
    "mean_compressed_minus_stage96_entropy_reference_bytes",
    "mean_compressed_ratio_vs_stage96_entropy_reference",
    "compressed_below_stage96_entropy_reference_count",
    "max_decoded_max_abs_diff_vs_raw_deterministic",
    "comparison_note",
]

SETTING_SUMMARY_FIELDS = [
    "keep_fraction",
    "side_bits",
    "group_count",
    "mean_keep_gaussians",
    "mean_raw_deterministic_payload_bytes",
    "mean_compressed_deterministic_payload_bytes",
    "mean_compressed_deterministic_mib_per_intermediate_frame",
    "mean_compressed_ratio_vs_raw",
    "groups_below_stage96_entropy_reference",
    "mean_ratio_vs_stage96_entropy_reference",
    "max_ratio_vs_stage96_entropy_reference",
    "mean_compressed_minus_stage96_entropy_reference_bytes",
    "max_decoded_max_abs_diff_vs_raw_deterministic",
    "comparison_note",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value):
    return float(value) if value not in (None, "") else 0.0


def to_int(value):
    return int(float(value))


def average(rows, key):
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def endpoint_diff_indices(left_attrs, right_attrs, keep_fraction):
    if left_attrs.shape != right_attrs.shape or left_attrs.ndim != 3 or left_attrs.shape[0] != 1:
        raise ValueError(f"expected matching [1, N, D] attrs, got {left_attrs.shape} and {right_attrs.shape}")
    gaussian_count = int(left_attrs.shape[1])
    keep_count = min(max(int(round(gaussian_count * float(keep_fraction))), 0), gaussian_count)
    if keep_count <= 0:
        return torch.empty((0,), dtype=torch.int64)
    scores = torch.sum((right_attrs[0].float() - left_attrs[0].float()) ** 2, dim=-1)
    return torch.sort(torch.topk(scores, k=keep_count, largest=True).indices.to(torch.int64)).values.detach().cpu()


def build_stage96_entropy_reference(rows):
    out = {}
    for row in rows:
        if row["source_scope"] != "stage96_entropy_broader":
            continue
        key = (row["base_method"], row["codec"], to_int(row["reference_gap"]))
        out[key] = {
            "bytes": to_float(row["entropy_sideinfo_payload_bytes"]),
            "mib": to_float(row["entropy_sideinfo_mib_per_intermediate_frame"]),
        }
    return out


def comparison_note(keep_fraction, side_bits):
    if abs(float(keep_fraction) - 0.1) < 1e-12 and int(side_bits) == 6:
        return "stage115_q6_top10_setting"
    return "cross_setting_rate_only_quality_unknown"


def summarize_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["base_method"], row["codec"], int(row["reference_gap"]), float(row["keep_fraction"]), int(row["side_bits"]))
        grouped[key].append(row)
    out = []
    for (method, codec, gap, keep_fraction, side_bits), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3], item[0][4])):
        out.append({
            "base_method": method,
            "codec": codec,
            "reference_gap": gap,
            "keep_fraction": keep_fraction,
            "side_bits": side_bits,
            "task_count": len(items),
            "selector_policy": items[0]["selector_policy"],
            "selected_candidate": items[0]["selected_candidate"],
            "zlib_level": items[0]["zlib_level"],
            "mean_keep_gaussians": average(items, "keep_gaussians"),
            "mean_raw_deterministic_payload_bytes": average(items, "raw_deterministic_payload_bytes"),
            "mean_compressed_deterministic_payload_bytes": average(items, "compressed_deterministic_payload_bytes"),
            "mean_raw_deterministic_mib_per_intermediate_frame": average(items, "raw_deterministic_mib_per_intermediate_frame"),
            "mean_compressed_deterministic_mib_per_intermediate_frame": average(items, "compressed_deterministic_mib_per_intermediate_frame"),
            "mean_compressed_savings_bytes_vs_raw": average(items, "compressed_savings_bytes_vs_raw"),
            "mean_compressed_ratio_vs_raw": average(items, "compressed_ratio_vs_raw"),
            "mean_stage96_entropy_reference_bytes": average(items, "stage96_entropy_reference_bytes"),
            "mean_stage96_entropy_reference_mib_per_intermediate_frame": average(items, "stage96_entropy_reference_mib_per_intermediate_frame"),
            "mean_compressed_minus_stage96_entropy_reference_bytes": average(items, "compressed_minus_stage96_entropy_reference_bytes"),
            "mean_compressed_ratio_vs_stage96_entropy_reference": average(items, "compressed_ratio_vs_stage96_entropy_reference"),
            "compressed_below_stage96_entropy_reference_count": sum(int(row["compressed_below_stage96_entropy_reference"]) for row in items),
            "max_decoded_max_abs_diff_vs_raw_deterministic": max(float(row["decoded_max_abs_diff_vs_raw_deterministic"]) for row in items),
            "comparison_note": items[0]["comparison_note"],
        })
    return out


def summarize_settings(group_rows):
    grouped = defaultdict(list)
    for row in group_rows:
        grouped[(float(row["keep_fraction"]), int(row["side_bits"]))].append(row)
    out = []
    for (keep_fraction, side_bits), items in sorted(grouped.items()):
        out.append({
            "keep_fraction": keep_fraction,
            "side_bits": side_bits,
            "group_count": len(items),
            "mean_keep_gaussians": average(items, "mean_keep_gaussians"),
            "mean_raw_deterministic_payload_bytes": average(items, "mean_raw_deterministic_payload_bytes"),
            "mean_compressed_deterministic_payload_bytes": average(items, "mean_compressed_deterministic_payload_bytes"),
            "mean_compressed_deterministic_mib_per_intermediate_frame": average(items, "mean_compressed_deterministic_mib_per_intermediate_frame"),
            "mean_compressed_ratio_vs_raw": average(items, "mean_compressed_ratio_vs_raw"),
            "groups_below_stage96_entropy_reference": sum(1 for row in items if float(row["mean_compressed_minus_stage96_entropy_reference_bytes"]) < 0.0),
            "mean_ratio_vs_stage96_entropy_reference": average(items, "mean_compressed_ratio_vs_stage96_entropy_reference"),
            "max_ratio_vs_stage96_entropy_reference": max(float(row["mean_compressed_ratio_vs_stage96_entropy_reference"]) for row in items),
            "mean_compressed_minus_stage96_entropy_reference_bytes": average(items, "mean_compressed_minus_stage96_entropy_reference_bytes"),
            "max_decoded_max_abs_diff_vs_raw_deterministic": max(float(row["max_decoded_max_abs_diff_vs_raw_deterministic"]) for row in items),
            "comparison_note": comparison_note(keep_fraction, side_bits),
        })
    return out


def format_float(value):
    return f"{float(value):.6f}"


def write_report(summary, setting_rows, path):
    lines = [
        "# Stage119 Actual Compressed Deterministic Sweep",
        "",
        "## Configuration",
        "",
        f"- task count: `{summary['task_count']}`",
        f"- keep fractions: `{summary['keep_fractions']}`",
        f"- side bits: `{summary['side_bits']}`",
        f"- zlib level: `{summary['zlib_level']}`",
        "- no rendering, no training, no checkpoint, no heavy tensor output",
        "- Stage96 q6/top10 entropy side-info is used as rate reference only",
        "",
        "## Setting Summary",
        "",
        "| keep | bits | mean comp bytes | mean comp MiB | comp/raw | groups below Stage96 entropy | max comp/entropy | max decode diff | note |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in setting_rows:
        lines.append(
            f"| {row['keep_fraction']} | {row['side_bits']} | {row['mean_compressed_deterministic_payload_bytes']} | {format_float(row['mean_compressed_deterministic_mib_per_intermediate_frame'])} | {format_float(row['mean_compressed_ratio_vs_raw'])} | {row['groups_below_stage96_entropy_reference']}/{row['group_count']} | {format_float(row['max_ratio_vs_stage96_entropy_reference'])} | {row['max_decoded_max_abs_diff_vs_raw_deterministic']} | {row['comparison_note']} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- rows CSV: `{summary['rows_csv']}`",
        f"- group summary CSV: `{summary['group_summary_csv']}`",
        f"- setting summary CSV: `{summary['setting_summary_csv']}`",
        f"- summary JSON: `{summary['summary_json']}`",
        "",
        "## Notes",
        "",
        "- q6/top10 is the Stage115/118 setting and is decode-equivalent to previous deterministic q6/top10 payloads.",
        "- Non-q6/top10 settings are rate-only until rendered validation is run.",
        "- Residual values remain teacher-derived from dense target anchors.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--stage114_policy", type=Path, default=DEFAULT_STAGE114_POLICY)
    parser.add_argument("--stage116_rows", type=Path, default=DEFAULT_STAGE116_ROWS)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--task_split", default="eval")
    parser.add_argument("--codecs", nargs="+", default=["q12"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--max_tasks", type=int, default=12)
    parser.add_argument("--keep_fractions", nargs="+", type=float, default=DEFAULT_KEEP_FRACTIONS)
    parser.add_argument("--side_bits", nargs="+", type=int, default=DEFAULT_SIDE_BITS)
    parser.add_argument("--zlib_level", type=int, default=9)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    policy = json.loads(args.stage114_policy.read_text(encoding="utf-8"))
    if policy["selected_candidate"] != "endpoint_diff_baseline":
        raise ValueError("Stage119 sweep expects the Stage114 endpoint-diff selector")
    entropy_reference = build_stage96_entropy_reference(read_csv(args.stage116_rows))
    device = torch.device(args.device)
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
            left_attrs = flatten_static_anchor(left)
            right_attrs = flatten_static_anchor(right)
            dense_key = (task["dataset"], task["split"], task["sequence"], task["target_index"])
            if dense_key not in dense_index:
                raise KeyError(f"Missing dense target anchor for {dense_key}")
            target_item, target_side = dense_index[dense_key]
            target_anchor = load_anchor(target_item, target_side, device, bits=None, cache=cache)
            target_attrs = flatten_static_anchor(target_anchor)
            t = torch.tensor([task["normalized_time"]], dtype=torch.float32, device=device)
            base_attrs_by_method = {
                "linear": flatten_static_anchor(linear_anchor(left, right, task["normalized_time"])),
                "stage65_adapter": flatten_static_anchor(model(left, right, t, apply_output_constraints=False)),
            }
            selected_by_keep = {keep_fraction: endpoint_diff_indices(left_attrs, right_attrs, keep_fraction) for keep_fraction in args.keep_fractions}
            for method, base_attrs in base_attrs_by_method.items():
                ref_key = (method, task["codec"], int(task["reference_gap"]))
                if ref_key not in entropy_reference:
                    raise KeyError(f"Missing Stage96 entropy reference for {ref_key}")
                ref = entropy_reference[ref_key]
                for keep_fraction, selected_indices in selected_by_keep.items():
                    for side_bits in args.side_bits:
                        raw_payload, raw_info = encode_selected_residual_values_sideinfo(base_attrs, target_attrs, selected_indices, side_bits)
                        raw_decoded = decode_selected_residual_values_sideinfo(base_attrs, raw_payload, selected_indices)
                        compressed_payload, compressed_info = encode_selected_residual_values_sideinfo_entropy(
                            base_attrs,
                            target_attrs,
                            selected_indices,
                            side_bits,
                            zlib_level=args.zlib_level,
                        )
                        compressed_decoded = decode_selected_residual_values_sideinfo_entropy(base_attrs, compressed_payload, selected_indices)
                        decoded_diff = torch.max(torch.abs(compressed_decoded - raw_decoded)).item()
                        rows.append({
                            "task_id": task["task_id"],
                            "sequence": task["sequence"],
                            "codec": task["codec"],
                            "reference_gap": task["reference_gap"],
                            "target_index": task["target_index"],
                            "base_method": method,
                            "selector_policy": policy["policy_name"],
                            "selected_candidate": policy["selected_candidate"],
                            "keep_fraction": keep_fraction,
                            "side_bits": side_bits,
                            "zlib_level": args.zlib_level,
                            "keep_gaussians": raw_info["keep_count"],
                            "raw_deterministic_payload_bytes": raw_info["payload_bytes"],
                            "compressed_deterministic_payload_bytes": compressed_info["payload_bytes"],
                            "raw_deterministic_mib_per_intermediate_frame": raw_info["payload_bytes"] / (1024.0 * 1024.0),
                            "compressed_deterministic_mib_per_intermediate_frame": compressed_info["payload_bytes"] / (1024.0 * 1024.0),
                            "compressed_savings_bytes_vs_raw": raw_info["payload_bytes"] - compressed_info["payload_bytes"],
                            "compressed_ratio_vs_raw": compressed_info["payload_bytes"] / raw_info["payload_bytes"],
                            "metadata_zlib_bytes": compressed_info["metadata_zlib_bytes"],
                            "residual_zlib_bytes": compressed_info["residual_zlib_bytes"],
                            "omitted_index_bytes_vs_fixed_aligned": raw_info["omitted_index_bytes_vs_fixed_aligned"],
                            "stage96_entropy_reference_bytes": ref["bytes"],
                            "stage96_entropy_reference_mib_per_intermediate_frame": ref["mib"],
                            "compressed_minus_stage96_entropy_reference_bytes": compressed_info["payload_bytes"] - ref["bytes"],
                            "compressed_ratio_vs_stage96_entropy_reference": compressed_info["payload_bytes"] / ref["bytes"],
                            "compressed_below_stage96_entropy_reference": int(compressed_info["payload_bytes"] < ref["bytes"]),
                            "decoded_max_abs_diff_vs_raw_deterministic": decoded_diff,
                            "comparison_note": comparison_note(keep_fraction, side_bits),
                        })
            if device.type == "cuda":
                torch.cuda.empty_cache()

    group_rows = summarize_groups(rows)
    setting_rows = summarize_settings(group_rows)
    rows_csv = args.summary_root / "stage119_actual_compressed_deterministic_sweep_rows.csv"
    group_summary_csv = args.summary_root / "stage119_actual_compressed_deterministic_sweep_group_summary.csv"
    setting_summary_csv = args.summary_root / "stage119_actual_compressed_deterministic_sweep_setting_summary.csv"
    summary_json = args.summary_root / "stage119_actual_compressed_deterministic_sweep_summary.json"
    report_md = args.summary_root / "stage119_actual_compressed_deterministic_sweep_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(group_rows, group_summary_csv, GROUP_SUMMARY_FIELDS)
    write_csv(setting_rows, setting_summary_csv, SETTING_SUMMARY_FIELDS)
    summary = {
        "stage": 119,
        "mode": "actual compressed deterministic side-info q-bit / keep-fraction sweep",
        "stage114_policy": str(args.stage114_policy),
        "stage116_rows": str(args.stage116_rows),
        "selector_policy": policy["policy_name"],
        "selected_candidate": policy["selected_candidate"],
        "task_manifest": str(args.task_manifest),
        "dense_manifest": str(args.dense_manifest),
        "adapter": str(args.adapter),
        "task_split": args.task_split,
        "codecs": args.codecs,
        "gaps": args.gaps,
        "task_count": len(tasks),
        "keep_fractions": args.keep_fractions,
        "side_bits": args.side_bits,
        "zlib_level": args.zlib_level,
        "rows_csv": str(rows_csv),
        "group_summary_csv": str(group_summary_csv),
        "setting_summary_csv": str(setting_summary_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "row_count": len(rows),
        "group_count": len(group_rows),
        "setting_count": len(setting_rows),
        "setting_summary_rows": setting_rows,
        "notes": [
            "Compressed deterministic payloads are measured from actual task residual values.",
            "Stage96 q6/top10 entropy-coded index+value side-info is used as a rate reference only.",
            "Non-q6/top10 settings require rendered validation before RD packaging.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, setting_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "row_count": len(rows),
        "setting_count": len(setting_rows),
        "max_decode_diff": max(float(row["decoded_max_abs_diff_vs_raw_deterministic"]) for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
