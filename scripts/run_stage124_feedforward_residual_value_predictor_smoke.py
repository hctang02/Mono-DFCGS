import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE78_RATE_TABLE = REPO_ROOT / "experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_rate_table.csv"
DEFAULT_STAGE123_POLICY = REPO_ROOT / "experiments/stage123_compressed_deterministic_codec_policy_package/stage123_compressed_deterministic_codec_policy.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage124_feedforward_residual_value_predictor_smoke"

POLICY_SETTINGS = [
    {"label": "q4_top20", "role": "primary", "keep_fraction": 0.2, "side_bits": 4},
    {"label": "q4_top10", "role": "low_rate", "keep_fraction": 0.1, "side_bits": 4},
]

ROW_FIELDS = [
    "task_id",
    "sequence",
    "codec",
    "reference_gap",
    "target_index",
    "predictor_name",
    "base_method",
    "full_predictor_method",
    "setting_label",
    "setting_role",
    "keep_fraction",
    "side_bits",
    "keep_gaussians",
    "residual_value_source",
    "transmitted_residual_payload_bytes",
    "transmitted_selected_index_bytes",
    "q12_main_anchor_mib_per_frame",
    "direct_total_mib_per_frame",
    "amortized_total_mib_per_frame",
    "base_psnr",
    "full_predictor_psnr",
    "selected_predicted_psnr",
    "delta_psnr_vs_base",
    "delta_psnr_vs_full_predictor",
]

SUMMARY_FIELDS = [
    "predictor_name",
    "setting_label",
    "setting_role",
    "keep_fraction",
    "side_bits",
    "task_count",
    "mean_keep_gaussians",
    "mean_direct_total_mib_per_frame",
    "mean_amortized_total_mib_per_frame",
    "mean_base_psnr",
    "mean_full_predictor_psnr",
    "mean_selected_predicted_psnr",
    "mean_delta_psnr_vs_base",
    "mean_delta_psnr_vs_full_predictor",
    "positive_delta_vs_base_count",
    "near_full_within_0p10db_count",
]


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, unflatten_static_anchor  # noqa: E402
from mono_dfcgs.residual_value_predictor import (  # noqa: E402
    apply_selected_residual_values,
    endpoint_diff_topk_indices,
    selected_residual_values_from_prediction,
)
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import load_rgb  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import (  # noqa: E402
    DEFAULT_ADAPTER,
    DEFAULT_TASK_MANIFEST,
    linear_anchor,
    load_adapter,
    load_anchor,
    parse_task_rows,
    select_balanced,
)
from scripts.run_stage86_rendered_residual_sideinfo_smoke import render_psnr  # noqa: E402


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def average(rows, key):
    values = [float(row[key]) for row in rows]
    return sum(values) / max(len(values), 1)


def build_main_rate_lookup(rows):
    lookup = {}
    for row in rows:
        if row.get("codec") != "q12":
            continue
        gap = int(row.get("reference_gap", row.get("frame_gap")))
        rate = float(row.get("transmitted_mib_per_frame", row.get("mean_static_anchor_mib_per_frame_with_metadata")))
        lookup[(row["method"], gap)] = rate
    return lookup


def uniform_intermediate_frame_ratio(gap):
    if gap <= 1:
        return 0.0
    return (float(gap) - 1.0) / float(gap)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["predictor_name"], row["setting_label"])].append(row)
    out = []
    for (predictor, setting), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        first = items[0]
        out.append({
            "predictor_name": predictor,
            "setting_label": setting,
            "setting_role": first["setting_role"],
            "keep_fraction": first["keep_fraction"],
            "side_bits": first["side_bits"],
            "task_count": len(items),
            "mean_keep_gaussians": average(items, "keep_gaussians"),
            "mean_direct_total_mib_per_frame": average(items, "direct_total_mib_per_frame"),
            "mean_amortized_total_mib_per_frame": average(items, "amortized_total_mib_per_frame"),
            "mean_base_psnr": average(items, "base_psnr"),
            "mean_full_predictor_psnr": average(items, "full_predictor_psnr"),
            "mean_selected_predicted_psnr": average(items, "selected_predicted_psnr"),
            "mean_delta_psnr_vs_base": average(items, "delta_psnr_vs_base"),
            "mean_delta_psnr_vs_full_predictor": average(items, "delta_psnr_vs_full_predictor"),
            "positive_delta_vs_base_count": sum(1 for row in items if float(row["delta_psnr_vs_base"]) > 0.0),
            "near_full_within_0p10db_count": sum(1 for row in items if float(row["delta_psnr_vs_full_predictor"]) >= -0.10),
        })
    return out


def format_float(value):
    return f"{float(value):.6f}"


def write_report(summary, summary_rows, path):
    lines = [
        "# Stage124 Feed-Forward Residual Value Predictor Smoke",
        "",
        "## Scope",
        "",
        "- Uses the Stage65 adapter as a feed-forward residual value predictor over a linear base.",
        "- Predicted residual values are `adapter_attrs - linear_attrs` at deterministic endpoint-diff selected indices.",
        "- No residual payload bytes and no selected-index bytes are transmitted in this smoke.",
        "- Target dense anchors are not used; target RGB is used only for offline rendered metrics.",
        "",
        "## Summary",
        "",
        "| predictor | setting | role | keep | tasks | rate | base PSNR | full predictor PSNR | selected PSNR | delta base | delta full | near full |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['predictor_name']} | {row['setting_label']} | {row['setting_role']} | {row['keep_fraction']} | {row['task_count']} | {format_float(row['mean_direct_total_mib_per_frame'])} | {format_float(row['mean_base_psnr'])} | {format_float(row['mean_full_predictor_psnr'])} | {format_float(row['mean_selected_predicted_psnr'])} | {format_float(row['mean_delta_psnr_vs_base'])} | {format_float(row['mean_delta_psnr_vs_full_predictor'])} | {row['near_full_within_0p10db_count']}/{row['task_count']} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- rows CSV: `{summary['rows_csv']}`",
        f"- summary CSV: `{summary['summary_csv']}`",
        f"- summary JSON: `{summary['summary_json']}`",
        f"- report Markdown: `{summary['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--stage78_rate_table", type=Path, default=DEFAULT_STAGE78_RATE_TABLE)
    parser.add_argument("--stage123_policy", type=Path, default=DEFAULT_STAGE123_POLICY)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--task_split", default="eval")
    parser.add_argument("--codecs", nargs="+", default=["q12"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--max_tasks", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    policy = json.loads(args.stage123_policy.read_text(encoding="utf-8"))
    if policy["policy_name"] != "compressed_deterministic_value_only_residual_codec_v1":
        raise ValueError("Stage124 expects the Stage123 compressed deterministic codec policy")
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
    adapter = load_adapter(args.adapter, args.hidden_dim, device)
    cache = {}
    rows = []
    with torch.no_grad():
        for task in tasks:
            left = load_anchor(task["left_anchor_source_item"], task["left_anchor_source_side"], device, bits=task["bits"], cache=cache)
            right = load_anchor(task["right_anchor_source_item"], task["right_anchor_source_side"], device, bits=task["bits"], cache=cache)
            left_attrs = flatten_static_anchor(left)
            right_attrs = flatten_static_anchor(right)
            target_rgb = load_rgb(task["target_rgb_path"], opt.image_height, opt.image_width, device)
            t = torch.tensor([task["normalized_time"]], dtype=torch.float32, device=device)
            linear_attrs = flatten_static_anchor(linear_anchor(left, right, task["normalized_time"]))
            adapter_attrs = flatten_static_anchor(adapter(left, right, t, apply_output_constraints=False))
            base_psnr = render_psnr(unflatten_static_anchor(linear_attrs), target_rgb, background, opt)
            full_predictor_psnr = render_psnr(unflatten_static_anchor(adapter_attrs), target_rgb, background, opt)
            gap = int(task["reference_gap"])
            main_rate = main_rate_lookup[("linear", gap)]
            intermediate_ratio = uniform_intermediate_frame_ratio(gap)
            for setting in POLICY_SETTINGS:
                selected_indices = endpoint_diff_topk_indices(left_attrs, right_attrs, setting["keep_fraction"])
                residual_values = selected_residual_values_from_prediction(linear_attrs, adapter_attrs, selected_indices)
                selected_attrs = apply_selected_residual_values(linear_attrs, selected_indices, residual_values)
                selected_psnr = render_psnr(unflatten_static_anchor(selected_attrs), target_rgb, background, opt)
                rows.append({
                    "task_id": task["task_id"],
                    "sequence": task["sequence"],
                    "codec": task["codec"],
                    "reference_gap": gap,
                    "target_index": task["target_index"],
                    "predictor_name": "adapter_delta_selected_v1",
                    "base_method": "linear",
                    "full_predictor_method": "stage65_adapter",
                    "setting_label": setting["label"],
                    "setting_role": setting["role"],
                    "keep_fraction": setting["keep_fraction"],
                    "side_bits": setting["side_bits"],
                    "keep_gaussians": int(selected_indices.numel()),
                    "residual_value_source": "feed_forward_stage65_adapter_delta",
                    "transmitted_residual_payload_bytes": 0,
                    "transmitted_selected_index_bytes": 0,
                    "q12_main_anchor_mib_per_frame": main_rate,
                    "direct_total_mib_per_frame": main_rate,
                    "amortized_total_mib_per_frame": main_rate,
                    "base_psnr": base_psnr,
                    "full_predictor_psnr": full_predictor_psnr,
                    "selected_predicted_psnr": selected_psnr,
                    "delta_psnr_vs_base": selected_psnr - base_psnr,
                    "delta_psnr_vs_full_predictor": selected_psnr - full_predictor_psnr,
                })
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_rows = summarize(rows)
    rows_csv = args.summary_root / "stage124_feedforward_residual_value_predictor_smoke_rows.csv"
    summary_csv = args.summary_root / "stage124_feedforward_residual_value_predictor_smoke_summary.csv"
    summary_json = args.summary_root / "stage124_feedforward_residual_value_predictor_smoke_summary.json"
    report_md = args.summary_root / "stage124_feedforward_residual_value_predictor_smoke_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    summary = {
        "stage": 124,
        "mode": "feed-forward residual value predictor smoke",
        "stage123_policy": str(args.stage123_policy),
        "task_manifest": str(args.task_manifest),
        "adapter": str(args.adapter),
        "task_split": args.task_split,
        "codecs": args.codecs,
        "gaps": args.gaps,
        "task_count": len(tasks),
        "row_count": len(rows),
        "predictor_name": "adapter_delta_selected_v1",
        "settings": POLICY_SETTINGS,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "notes": [
            "No residual payload bytes are transmitted in this predictor smoke.",
            "No selected-index bytes are transmitted; indices are reproduced from endpoint-diff top-k.",
            "Target dense anchors are not loaded or used.",
            "Target RGB is used only for offline rendered metrics.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, summary_rows, report_md)
    print(json.dumps({"summary": str(summary_json), "row_count": len(rows), "summary_rows": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
