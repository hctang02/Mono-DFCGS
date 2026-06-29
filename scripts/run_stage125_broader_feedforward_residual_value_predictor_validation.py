import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage125_broader_feedforward_residual_value_predictor_validation"


sys.path.insert(0, str(REPO_ROOT))
import scripts.run_stage124_feedforward_residual_value_predictor_smoke as stage124  # noqa: E402


def format_float(value):
    return f"{float(value):.6f}"


def write_report(summary, summary_rows, path):
    lines = [
        "# Stage125 Broader Feed-Forward Residual Value Predictor Validation",
        "",
        "## Scope",
        "",
        "- Broadens Stage124 from 12 eval tasks to a 60-task rendered validation slice.",
        "- Uses `adapter_delta_selected_v1`: `adapter_attrs - linear_attrs` at deterministic endpoint-diff selected indices.",
        "- Transmits no residual payload bytes and no selected-index bytes.",
        "- Does not load target dense anchors; target RGB is used only for offline rendered metrics.",
        "",
        "## Summary",
        "",
        "| predictor | setting | role | keep | tasks | rate | base PSNR | full predictor PSNR | selected PSNR | delta base | delta full | positives | near full |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['predictor_name']} | {row['setting_label']} | {row['setting_role']} | {row['keep_fraction']} | {row['task_count']} | {format_float(row['mean_direct_total_mib_per_frame'])} | {format_float(row['mean_base_psnr'])} | {format_float(row['mean_full_predictor_psnr'])} | {format_float(row['mean_selected_predicted_psnr'])} | {format_float(row['mean_delta_psnr_vs_base'])} | {format_float(row['mean_delta_psnr_vs_full_predictor'])} | {row['positive_delta_vs_base_count']}/{row['task_count']} | {row['near_full_within_0p10db_count']}/{row['task_count']} |"
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
    parser.add_argument("--task_manifest", type=Path, default=stage124.DEFAULT_TASK_MANIFEST)
    parser.add_argument("--adapter", type=Path, default=stage124.DEFAULT_ADAPTER)
    parser.add_argument("--stage78_rate_table", type=Path, default=stage124.DEFAULT_STAGE78_RATE_TABLE)
    parser.add_argument("--stage123_policy", type=Path, default=stage124.DEFAULT_STAGE123_POLICY)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--task_split", default="eval")
    parser.add_argument("--codecs", nargs="+", default=["q12"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--max_tasks", type=int, default=60)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    policy = json.loads(args.stage123_policy.read_text(encoding="utf-8"))
    if policy["policy_name"] != "compressed_deterministic_value_only_residual_codec_v1":
        raise ValueError("Stage125 expects the Stage123 compressed deterministic codec policy")
    main_rate_lookup = stage124.build_main_rate_lookup(stage124.read_csv(args.stage78_rate_table))
    device = torch.device(args.device)
    opt = stage124.Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    tasks = stage124.select_balanced(stage124.parse_task_rows(args.task_manifest, args.task_split, args.codecs, args.gaps), args.max_tasks, args.seed)
    if not tasks:
        raise RuntimeError("No tasks selected")
    adapter = stage124.load_adapter(args.adapter, args.hidden_dim, device)
    cache = {}
    rows = []
    with torch.no_grad():
        for task in tasks:
            left = stage124.load_anchor(task["left_anchor_source_item"], task["left_anchor_source_side"], device, bits=task["bits"], cache=cache)
            right = stage124.load_anchor(task["right_anchor_source_item"], task["right_anchor_source_side"], device, bits=task["bits"], cache=cache)
            left_attrs = stage124.flatten_static_anchor(left)
            right_attrs = stage124.flatten_static_anchor(right)
            target_rgb = stage124.load_rgb(task["target_rgb_path"], opt.image_height, opt.image_width, device)
            t = torch.tensor([task["normalized_time"]], dtype=torch.float32, device=device)
            linear_attrs = stage124.flatten_static_anchor(stage124.linear_anchor(left, right, task["normalized_time"]))
            adapter_attrs = stage124.flatten_static_anchor(adapter(left, right, t, apply_output_constraints=False))
            base_psnr = stage124.render_psnr(stage124.unflatten_static_anchor(linear_attrs), target_rgb, background, opt)
            full_predictor_psnr = stage124.render_psnr(stage124.unflatten_static_anchor(adapter_attrs), target_rgb, background, opt)
            gap = int(task["reference_gap"])
            main_rate = main_rate_lookup[("linear", gap)]
            for setting in stage124.POLICY_SETTINGS:
                selected_indices = stage124.endpoint_diff_topk_indices(left_attrs, right_attrs, setting["keep_fraction"])
                residual_values = stage124.selected_residual_values_from_prediction(linear_attrs, adapter_attrs, selected_indices)
                selected_attrs = stage124.apply_selected_residual_values(linear_attrs, selected_indices, residual_values)
                selected_psnr = stage124.render_psnr(stage124.unflatten_static_anchor(selected_attrs), target_rgb, background, opt)
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

    summary_rows = stage124.summarize(rows)
    rows_csv = args.summary_root / "stage125_broader_feedforward_residual_value_predictor_validation_rows.csv"
    summary_csv = args.summary_root / "stage125_broader_feedforward_residual_value_predictor_validation_summary.csv"
    summary_json = args.summary_root / "stage125_broader_feedforward_residual_value_predictor_validation_summary.json"
    report_md = args.summary_root / "stage125_broader_feedforward_residual_value_predictor_validation_report.md"
    stage124.write_csv(rows, rows_csv, stage124.ROW_FIELDS)
    stage124.write_csv(summary_rows, summary_csv, stage124.SUMMARY_FIELDS)
    summary = {
        "stage": 125,
        "mode": "broader feed-forward residual value predictor validation",
        "stage123_policy": str(args.stage123_policy),
        "task_manifest": str(args.task_manifest),
        "adapter": str(args.adapter),
        "task_split": args.task_split,
        "codecs": args.codecs,
        "gaps": args.gaps,
        "task_count": len(tasks),
        "row_count": len(rows),
        "predictor_name": "adapter_delta_selected_v1",
        "settings": stage124.POLICY_SETTINGS,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "notes": [
            "No residual payload bytes are transmitted in this predictor validation.",
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
