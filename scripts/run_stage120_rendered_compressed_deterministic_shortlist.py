import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE78_RATE_TABLE = REPO_ROOT / "experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_rate_table.csv"
DEFAULT_STAGE114_POLICY = REPO_ROOT / "experiments/stage114_strict_safe_selector_fallback_package/stage114_strict_safe_selector_fallback_policy.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage120_rendered_compressed_deterministic_shortlist"

SHORTLIST = [
    {"label": "q6_top10", "keep_fraction": 0.1, "side_bits": 6},
    {"label": "q5_top10", "keep_fraction": 0.1, "side_bits": 5},
    {"label": "q4_top10", "keep_fraction": 0.1, "side_bits": 4},
    {"label": "q6_top5", "keep_fraction": 0.05, "side_bits": 6},
    {"label": "q4_top20", "keep_fraction": 0.2, "side_bits": 4},
]
SETTING_ORDER = {setting["label"]: index for index, setting in enumerate(SHORTLIST)}
METHOD_TO_STAGE78 = {"linear": "linear", "stage65_adapter": "adapter"}
METHOD_ORDER = {"linear": 0, "stage65_adapter": 1}


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, unflatten_static_anchor  # noqa: E402
from mono_dfcgs.residual_sideinfo_codec import (  # noqa: E402
    decode_selected_residual_values_sideinfo,
    decode_selected_residual_values_sideinfo_entropy,
    encode_selected_residual_values_sideinfo,
    encode_selected_residual_values_sideinfo_entropy,
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
    "setting_label",
    "selector_policy",
    "selected_candidate",
    "keep_fraction",
    "side_bits",
    "zlib_level",
    "keep_gaussians",
    "compressed_payload_bytes",
    "compressed_mib_per_intermediate_frame",
    "q12_main_anchor_mib_per_frame",
    "uniform_intermediate_frame_ratio",
    "direct_total_mib_per_frame",
    "amortized_total_mib_per_frame",
    "base_psnr",
    "setting_psnr",
    "delta_psnr_vs_base",
    "q6_top10_psnr",
    "delta_psnr_vs_q6_top10",
    "q6_top10_payload_bytes",
    "decoded_max_abs_diff_vs_raw_deterministic",
]

GROUP_SUMMARY_FIELDS = [
    "base_method",
    "codec",
    "reference_gap",
    "setting_label",
    "task_count",
    "keep_fraction",
    "side_bits",
    "zlib_level",
    "mean_keep_gaussians",
    "mean_compressed_payload_bytes",
    "mean_compressed_mib_per_intermediate_frame",
    "mean_q12_main_anchor_mib_per_frame",
    "mean_direct_total_mib_per_frame",
    "mean_amortized_total_mib_per_frame",
    "mean_base_psnr",
    "mean_setting_psnr",
    "mean_delta_psnr_vs_base",
    "positive_delta_vs_base_count",
    "mean_q6_top10_psnr",
    "mean_delta_psnr_vs_q6_top10",
    "near_q6_within_0p10db_count",
    "max_decoded_max_abs_diff_vs_raw_deterministic",
]

SETTING_SUMMARY_FIELDS = [
    "setting_label",
    "keep_fraction",
    "side_bits",
    "group_count",
    "task_row_count",
    "mean_keep_gaussians",
    "mean_compressed_payload_bytes",
    "mean_compressed_mib_per_intermediate_frame",
    "mean_direct_total_mib_per_frame",
    "mean_amortized_total_mib_per_frame",
    "mean_base_psnr",
    "mean_setting_psnr",
    "mean_delta_psnr_vs_base",
    "positive_delta_vs_base_count",
    "mean_delta_psnr_vs_q6_top10",
    "near_q6_within_0p10db_count",
    "max_decoded_max_abs_diff_vs_raw_deterministic",
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


def uniform_intermediate_frame_ratio(gap):
    if gap <= 1:
        return 0.0
    return (float(gap) - 1.0) / float(gap)


def build_main_rate_lookup(rows):
    lookup = {}
    for row in rows:
        if row["codec"] != "q12":
            continue
        lookup[(row["method"], to_int(row["frame_gap"]))] = to_float(row["mean_static_anchor_mib_per_frame_with_metadata"])
    return lookup


def summarize_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["base_method"], row["codec"], int(row["reference_gap"]), row["setting_label"])].append(row)
    out = []
    for (method, codec, gap, label), items in sorted(grouped.items(), key=lambda item: (METHOD_ORDER[item[0][0]], item[0][2], SETTING_ORDER[item[0][3]])):
        out.append({
            "base_method": method,
            "codec": codec,
            "reference_gap": gap,
            "setting_label": label,
            "task_count": len(items),
            "keep_fraction": items[0]["keep_fraction"],
            "side_bits": items[0]["side_bits"],
            "zlib_level": items[0]["zlib_level"],
            "mean_keep_gaussians": average(items, "keep_gaussians"),
            "mean_compressed_payload_bytes": average(items, "compressed_payload_bytes"),
            "mean_compressed_mib_per_intermediate_frame": average(items, "compressed_mib_per_intermediate_frame"),
            "mean_q12_main_anchor_mib_per_frame": average(items, "q12_main_anchor_mib_per_frame"),
            "mean_direct_total_mib_per_frame": average(items, "direct_total_mib_per_frame"),
            "mean_amortized_total_mib_per_frame": average(items, "amortized_total_mib_per_frame"),
            "mean_base_psnr": average(items, "base_psnr"),
            "mean_setting_psnr": average(items, "setting_psnr"),
            "mean_delta_psnr_vs_base": average(items, "delta_psnr_vs_base"),
            "positive_delta_vs_base_count": sum(1 for row in items if float(row["delta_psnr_vs_base"]) > 0.0),
            "mean_q6_top10_psnr": average(items, "q6_top10_psnr"),
            "mean_delta_psnr_vs_q6_top10": average(items, "delta_psnr_vs_q6_top10"),
            "near_q6_within_0p10db_count": sum(1 for row in items if float(row["delta_psnr_vs_q6_top10"]) >= -0.10),
            "max_decoded_max_abs_diff_vs_raw_deterministic": max(float(row["decoded_max_abs_diff_vs_raw_deterministic"]) for row in items),
        })
    return out


def summarize_settings(rows, group_rows):
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[row["setting_label"]].append(row)
    grouped_groups = defaultdict(list)
    for row in group_rows:
        grouped_groups[row["setting_label"]].append(row)
    out = []
    for setting in SHORTLIST:
        label = setting["label"]
        items = grouped_rows[label]
        groups = grouped_groups[label]
        out.append({
            "setting_label": label,
            "keep_fraction": setting["keep_fraction"],
            "side_bits": setting["side_bits"],
            "group_count": len(groups),
            "task_row_count": len(items),
            "mean_keep_gaussians": average(items, "keep_gaussians"),
            "mean_compressed_payload_bytes": average(items, "compressed_payload_bytes"),
            "mean_compressed_mib_per_intermediate_frame": average(items, "compressed_mib_per_intermediate_frame"),
            "mean_direct_total_mib_per_frame": average(items, "direct_total_mib_per_frame"),
            "mean_amortized_total_mib_per_frame": average(items, "amortized_total_mib_per_frame"),
            "mean_base_psnr": average(items, "base_psnr"),
            "mean_setting_psnr": average(items, "setting_psnr"),
            "mean_delta_psnr_vs_base": average(items, "delta_psnr_vs_base"),
            "positive_delta_vs_base_count": sum(1 for row in items if float(row["delta_psnr_vs_base"]) > 0.0),
            "mean_delta_psnr_vs_q6_top10": average(items, "delta_psnr_vs_q6_top10"),
            "near_q6_within_0p10db_count": sum(1 for row in items if float(row["delta_psnr_vs_q6_top10"]) >= -0.10),
            "max_decoded_max_abs_diff_vs_raw_deterministic": max(float(row["decoded_max_abs_diff_vs_raw_deterministic"]) for row in items),
        })
    return out


def format_float(value):
    return f"{float(value):.6f}"


def write_report(summary, setting_rows, path):
    lines = [
        "# Stage120 Rendered Compressed Deterministic Shortlist",
        "",
        "## Configuration",
        "",
        f"- task count: `{summary['task_count']}`",
        f"- shortlist: `{summary['shortlist']}`",
        f"- zlib level: `{summary['zlib_level']}`",
        "- all side-info bytes are counted in direct/amortized total rates",
        "- residual values are teacher-derived; no residual value predictor is trained",
        "",
        "## Setting Summary",
        "",
        "| setting | keep | bits | payload bytes | side MiB | direct rate | amortized rate | PSNR | delta base | delta q6 | near q6 | positives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in setting_rows:
        lines.append(
            f"| {row['setting_label']} | {row['keep_fraction']} | {row['side_bits']} | {format_float(row['mean_compressed_payload_bytes'])} | {format_float(row['mean_compressed_mib_per_intermediate_frame'])} | {format_float(row['mean_direct_total_mib_per_frame'])} | {format_float(row['mean_amortized_total_mib_per_frame'])} | {format_float(row['mean_setting_psnr'])} | {format_float(row['mean_delta_psnr_vs_base'])} | {format_float(row['mean_delta_psnr_vs_q6_top10'])} | {row['near_q6_within_0p10db_count']}/{row['task_row_count']} | {row['positive_delta_vs_base_count']}/{row['task_row_count']} |"
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
        "- q6_top10 is the quality anchor for this deterministic endpoint-diff selector line.",
        "- `near q6` counts rows with PSNR no worse than 0.10 dB below q6_top10 for the same task/base.",
        "- If a lower-rate setting is close to q6_top10, it should be broadened in Stage121.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--stage78_rate_table", type=Path, default=DEFAULT_STAGE78_RATE_TABLE)
    parser.add_argument("--stage114_policy", type=Path, default=DEFAULT_STAGE114_POLICY)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--task_split", default="eval")
    parser.add_argument("--codecs", nargs="+", default=["q12"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--max_tasks", type=int, default=12)
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
        raise ValueError("Stage120 rendered shortlist expects the Stage114 endpoint-diff selector")
    main_rate_lookup = build_main_rate_lookup(read_csv(args.stage78_rate_table))
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
            left_attrs = flatten_static_anchor(left)
            right_attrs = flatten_static_anchor(right)
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
            keep_fractions = sorted({setting["keep_fraction"] for setting in SHORTLIST})
            selected_by_keep = {keep_fraction: endpoint_diff_indices(left_attrs, right_attrs, keep_fraction) for keep_fraction in keep_fractions}
            for method, base_attrs in base_attrs_by_method.items():
                stage78_method = METHOD_TO_STAGE78[method]
                gap = int(task["reference_gap"])
                main_rate = main_rate_lookup[(stage78_method, gap)]
                intermediate_ratio = uniform_intermediate_frame_ratio(gap)
                base_psnr = render_psnr(unflatten_static_anchor(base_attrs), target_rgb, background, opt)
                per_setting_rows = []
                for setting in SHORTLIST:
                    selected_indices = selected_by_keep[setting["keep_fraction"]]
                    raw_payload, _raw_info = encode_selected_residual_values_sideinfo(
                        base_attrs,
                        target_attrs,
                        selected_indices,
                        setting["side_bits"],
                    )
                    raw_decoded = decode_selected_residual_values_sideinfo(base_attrs, raw_payload, selected_indices)
                    compressed_payload, compressed_info = encode_selected_residual_values_sideinfo_entropy(
                        base_attrs,
                        target_attrs,
                        selected_indices,
                        setting["side_bits"],
                        zlib_level=args.zlib_level,
                    )
                    decoded_attrs = decode_selected_residual_values_sideinfo_entropy(base_attrs, compressed_payload, selected_indices)
                    decoded_diff = torch.max(torch.abs(decoded_attrs - raw_decoded)).item()
                    decoded_psnr = render_psnr(unflatten_static_anchor(decoded_attrs), target_rgb, background, opt)
                    side_mib = compressed_info["payload_bytes"] / (1024.0 * 1024.0)
                    per_setting_rows.append({
                        "task_id": task["task_id"],
                        "sequence": task["sequence"],
                        "codec": task["codec"],
                        "reference_gap": gap,
                        "target_index": task["target_index"],
                        "base_method": method,
                        "setting_label": setting["label"],
                        "selector_policy": policy["policy_name"],
                        "selected_candidate": policy["selected_candidate"],
                        "keep_fraction": setting["keep_fraction"],
                        "side_bits": setting["side_bits"],
                        "zlib_level": args.zlib_level,
                        "keep_gaussians": compressed_info["keep_count"],
                        "compressed_payload_bytes": compressed_info["payload_bytes"],
                        "compressed_mib_per_intermediate_frame": side_mib,
                        "q12_main_anchor_mib_per_frame": main_rate,
                        "uniform_intermediate_frame_ratio": intermediate_ratio,
                        "direct_total_mib_per_frame": main_rate + side_mib,
                        "amortized_total_mib_per_frame": main_rate + side_mib * intermediate_ratio,
                        "base_psnr": base_psnr,
                        "setting_psnr": decoded_psnr,
                        "delta_psnr_vs_base": decoded_psnr - base_psnr,
                        "decoded_max_abs_diff_vs_raw_deterministic": decoded_diff,
                    })
                q6_row = next(row for row in per_setting_rows if row["setting_label"] == "q6_top10")
                for row in per_setting_rows:
                    row["q6_top10_psnr"] = q6_row["setting_psnr"]
                    row["delta_psnr_vs_q6_top10"] = row["setting_psnr"] - q6_row["setting_psnr"]
                    row["q6_top10_payload_bytes"] = q6_row["compressed_payload_bytes"]
                    rows.append(row)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    group_rows = summarize_groups(rows)
    setting_rows = summarize_settings(rows, group_rows)
    rows_csv = args.summary_root / "stage120_rendered_compressed_deterministic_shortlist_rows.csv"
    group_summary_csv = args.summary_root / "stage120_rendered_compressed_deterministic_shortlist_group_summary.csv"
    setting_summary_csv = args.summary_root / "stage120_rendered_compressed_deterministic_shortlist_setting_summary.csv"
    summary_json = args.summary_root / "stage120_rendered_compressed_deterministic_shortlist_summary.json"
    report_md = args.summary_root / "stage120_rendered_compressed_deterministic_shortlist_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(group_rows, group_summary_csv, GROUP_SUMMARY_FIELDS)
    write_csv(setting_rows, setting_summary_csv, SETTING_SUMMARY_FIELDS)
    summary = {
        "stage": 120,
        "mode": "rendered compressed deterministic shortlist smoke",
        "stage114_policy": str(args.stage114_policy),
        "stage78_rate_table": str(args.stage78_rate_table),
        "selector_policy": policy["policy_name"],
        "selected_candidate": policy["selected_candidate"],
        "task_manifest": str(args.task_manifest),
        "dense_manifest": str(args.dense_manifest),
        "adapter": str(args.adapter),
        "task_split": args.task_split,
        "codecs": args.codecs,
        "gaps": args.gaps,
        "task_count": len(tasks),
        "shortlist": SHORTLIST,
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
            "Rendered smoke uses teacher-derived residual values from dense target anchors.",
            "All compressed deterministic side-info bytes are counted in direct and amortized total rates.",
            "q6_top10 is the deterministic quality anchor for same-task deltas.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, setting_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "row_count": len(rows),
        "max_decode_diff": max(float(row["decoded_max_abs_diff_vs_raw_deterministic"]) for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
