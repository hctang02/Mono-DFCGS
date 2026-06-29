import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE136_SUMMARY = REPO_ROOT / "experiments/stage136_render_aware_scale_sweep_smoke/stage136_render_aware_scale_sweep_smoke_summary.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage137_broader_render_aware_scale_validation"


sys.path.insert(0, str(REPO_ROOT))
import scripts.run_stage124_feedforward_residual_value_predictor_smoke as stage124  # noqa: E402
import scripts.run_stage136_render_aware_scale_sweep_smoke as stage136  # noqa: E402


def matching_summary_row(summary_rows, setting_label, adapter_delta_scale):
    for row in summary_rows:
        if row["setting_label"] == setting_label and abs(float(row["adapter_delta_scale"]) - float(adapter_delta_scale)) < 1e-12:
            return row
    raise KeyError(f"missing summary row for {setting_label} scale {adapter_delta_scale}")


def write_report(summary, summary_rows, path):
    smoke = summary["smoke_selected_summary_row"]
    best = summary["broader_best_summary_row"]
    lines = [
        "# Stage137 Broader Render-Aware Scale Validation",
        "",
        "## Scope",
        "",
        "- Broadens Stage136 from 12 tasks to a 60-task rendered validation slice.",
        "- Sweeps the Stage135 scale candidates for q4/top20 and q4/top10.",
        "- Uses only decoder-side endpoint anchors, normalized time, and the pre-shared Stage65 adapter.",
        "- Transmits no residual payload bytes and no selected-index bytes.",
        "- Does not load target dense anchors or teacher residual side-info.",
        "- Target RGB is used only for offline rendered PSNR metrics.",
        "",
        "## Stage136 Smoke-Selected Candidate",
        "",
        f"- setting: `{smoke['setting_label']}`",
        f"- adapter delta scale: `{smoke['adapter_delta_scale']}`",
        f"- mean selected PSNR: `{stage136.format_float(smoke['mean_selected_predicted_psnr'])}`",
        f"- mean delta vs base: `{stage136.format_float(smoke['mean_delta_psnr_vs_base'])}`",
        f"- positives: `{smoke['positive_delta_vs_base_count']}/{smoke['task_count']}`",
        "",
        "## Broader Best Candidate",
        "",
        f"- setting: `{best['setting_label']}`",
        f"- adapter delta scale: `{best['adapter_delta_scale']}`",
        f"- mean selected PSNR: `{stage136.format_float(best['mean_selected_predicted_psnr'])}`",
        f"- mean delta vs base: `{stage136.format_float(best['mean_delta_psnr_vs_base'])}`",
        f"- positives: `{best['positive_delta_vs_base_count']}/{best['task_count']}`",
        "",
        "## Summary",
        "",
        "| setting | role | keep | scale | tasks | rate | base PSNR | full PSNR | selected PSNR | delta base | delta full | positives | near full |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['setting_label']} | {row['setting_role']} | {row['keep_fraction']} | {row['adapter_delta_scale']} | {row['task_count']} | {stage136.format_float(row['mean_direct_total_mib_per_frame'])} | {stage136.format_float(row['mean_base_psnr'])} | {stage136.format_float(row['mean_full_predictor_psnr'])} | {stage136.format_float(row['mean_selected_predicted_psnr'])} | {stage136.format_float(row['mean_delta_psnr_vs_base'])} | {stage136.format_float(row['mean_delta_psnr_vs_full_predictor'])} | {row['positive_delta_vs_base_count']}/{row['task_count']} | {row['near_full_within_0p10db_count']}/{row['task_count']} |"
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
    parser.add_argument("--stage136_summary", type=Path, default=DEFAULT_STAGE136_SUMMARY)
    parser.add_argument("--stage135_protocol", type=Path, default=stage136.DEFAULT_STAGE135_PROTOCOL)
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
    stage136_summary = json.loads(args.stage136_summary.read_text(encoding="utf-8"))
    protocol = json.loads(args.stage135_protocol.read_text(encoding="utf-8"))
    policy = json.loads(args.stage123_policy.read_text(encoding="utf-8"))
    if protocol["protocol_name"] != "render_aware_adapter_delta_scale_calibration_v1":
        raise ValueError("Stage137 expects the Stage135 render-aware adapter-delta scale protocol")
    if policy["policy_name"] != "compressed_deterministic_value_only_residual_codec_v1":
        raise ValueError("Stage137 expects the Stage123 compressed deterministic codec policy")
    settings = protocol["settings"]
    scale_candidates = [float(value) for value in protocol["scale_candidates"]]
    main_rate_lookup = stage124.build_main_rate_lookup(stage124.read_csv(args.stage78_rate_table))
    device = torch.device(args.device)
    opt = stage124.Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    tasks = stage124.select_balanced(
        stage124.parse_task_rows(args.task_manifest, args.task_split, args.codecs, args.gaps),
        args.max_tasks,
        args.seed,
    )
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
            for setting in settings:
                selected_indices = stage124.endpoint_diff_topk_indices(left_attrs, right_attrs, setting["keep_fraction"])
                adapter_delta = stage124.selected_residual_values_from_prediction(linear_attrs, adapter_attrs, selected_indices)
                for scale in scale_candidates:
                    residual_values = adapter_delta * scale
                    selected_attrs = stage124.apply_selected_residual_values(linear_attrs, selected_indices, residual_values)
                    selected_psnr = stage124.render_psnr(stage124.unflatten_static_anchor(selected_attrs), target_rgb, background, opt)
                    rows.append({
                        "task_id": task["task_id"],
                        "sequence": task["sequence"],
                        "codec": task["codec"],
                        "reference_gap": gap,
                        "target_index": task["target_index"],
                        "predictor_name": "adapter_delta_selected_scaled_v1",
                        "base_method": "linear",
                        "full_predictor_method": "stage65_adapter",
                        "setting_label": setting["label"],
                        "setting_role": setting["role"],
                        "keep_fraction": setting["keep_fraction"],
                        "side_bits": setting["side_bits"],
                        "adapter_delta_scale": scale,
                        "keep_gaussians": int(selected_indices.numel()),
                        "residual_value_source": "feed_forward_stage65_adapter_delta_scaled",
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

    summary_rows = stage136.summarize(rows)
    smoke_selected = stage136_summary["best_summary_row"]
    smoke_selected_broader = matching_summary_row(
        summary_rows,
        smoke_selected["setting_label"],
        smoke_selected["adapter_delta_scale"],
    )
    broader_best = stage136.best_summary_row(summary_rows)
    rows_csv = args.summary_root / "stage137_broader_render_aware_scale_validation_rows.csv"
    summary_csv = args.summary_root / "stage137_broader_render_aware_scale_validation_summary.csv"
    summary_json = args.summary_root / "stage137_broader_render_aware_scale_validation_summary.json"
    report_md = args.summary_root / "stage137_broader_render_aware_scale_validation_report.md"
    stage136.write_csv(rows, rows_csv, stage136.ROW_FIELDS)
    stage136.write_csv(summary_rows, summary_csv, stage136.SUMMARY_FIELDS)
    summary = {
        "stage": 137,
        "mode": "broader render-aware adapter-delta scale validation",
        "stage135_protocol": str(args.stage135_protocol),
        "stage136_summary": str(args.stage136_summary),
        "stage123_policy": str(args.stage123_policy),
        "task_manifest": str(args.task_manifest),
        "adapter": str(args.adapter),
        "task_split": args.task_split,
        "codecs": args.codecs,
        "gaps": args.gaps,
        "task_count": len(tasks),
        "row_count": len(rows),
        "predictor_name": "adapter_delta_selected_scaled_v1",
        "scale_candidates": scale_candidates,
        "settings": settings,
        "smoke_selected_summary_row": smoke_selected_broader,
        "broader_best_summary_row": broader_best,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "notes": [
            "No residual payload bytes are transmitted in this predictor validation.",
            "No selected-index bytes are transmitted; indices are reproduced from endpoint-diff top-k.",
            "Target dense anchors and teacher residual side-info are not loaded or used.",
            "Target RGB is used only for offline rendered metrics.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, summary_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "row_count": len(rows),
        "smoke_selected_summary_row": smoke_selected_broader,
        "broader_best_summary_row": broader_best,
    }, indent=2))


if __name__ == "__main__":
    main()
