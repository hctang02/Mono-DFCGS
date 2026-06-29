import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE128_SETTINGS = REPO_ROOT / "experiments/stage128_predictor_codec_integration_package/stage128_predictor_codec_integration_settings.csv"
DEFAULT_STAGE126_STATS = REPO_ROOT / "experiments/stage126_selected_residual_predictor_dataset_package/stage126_selected_residual_predictor_train_stats.json"
DEFAULT_STAGE78_RATE_TABLE = REPO_ROOT / "experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_rate_table.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage129_broader_predictor_codec_rendered_validation"

ROW_FIELDS = [
    "task_id",
    "sequence",
    "codec",
    "reference_gap",
    "target_index",
    "setting_label",
    "setting_role",
    "keep_fraction",
    "side_bits",
    "keep_gaussians",
    "transmitted_residual_payload_bytes",
    "transmitted_selected_index_bytes",
    "q12_main_anchor_mib_per_frame",
    "direct_total_mib_per_frame",
    "amortized_total_mib_per_frame",
    "base_psnr",
    "full_adapter_psnr",
    "predictor_psnr",
    "delta_psnr_vs_base",
    "delta_psnr_vs_full_adapter",
]

SUMMARY_FIELDS = [
    "setting_label",
    "setting_role",
    "keep_fraction",
    "side_bits",
    "task_count",
    "mean_keep_gaussians",
    "mean_direct_total_mib_per_frame",
    "mean_amortized_total_mib_per_frame",
    "mean_base_psnr",
    "mean_full_adapter_psnr",
    "mean_predictor_psnr",
    "mean_delta_psnr_vs_base",
    "mean_delta_psnr_vs_full_adapter",
    "positive_delta_vs_base_count",
    "near_full_within_0p10db_count",
]


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, unflatten_static_anchor  # noqa: E402
from mono_dfcgs.residual_value_predictor import (  # noqa: E402
    SelectedResidualValueMLP,
    apply_selected_residual_values,
    endpoint_diff_topk_indices,
    selected_residual_feature_matrix,
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
from scripts.run_stage124_feedforward_residual_value_predictor_smoke import build_main_rate_lookup  # noqa: E402


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


def load_stats(path):
    return json.loads(path.read_text(encoding="utf-8"))["stats_by_setting"]


def load_predictor(setting_row, device):
    model = SelectedResidualValueMLP(
        feature_dim=int(setting_row["feature_dim"]),
        residual_dim=int(setting_row["residual_dim"]),
        hidden_dim=int(setting_row["hidden_dim"]),
    ).to(device)
    state = load_file(setting_row["checkpoint_path"], device=str(device) if device.type == "cpu" else "cuda:0")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_residual_values(model, features, stats, device):
    feature_mean = torch.tensor(stats["feature_mean"], dtype=torch.float32, device=device)
    feature_std = torch.tensor(stats["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    label_mean = torch.tensor(stats["label_mean"], dtype=torch.float32, device=device)
    label_std = torch.tensor(stats["label_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    with torch.no_grad():
        pred_norm = model((features.float() - feature_mean) / feature_std)
    return pred_norm * label_std + label_mean


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["setting_label"]].append(row)
    out = []
    for setting, items in sorted(grouped.items()):
        first = items[0]
        out.append({
            "setting_label": setting,
            "setting_role": first["setting_role"],
            "keep_fraction": first["keep_fraction"],
            "side_bits": first["side_bits"],
            "task_count": len(items),
            "mean_keep_gaussians": average(items, "keep_gaussians"),
            "mean_direct_total_mib_per_frame": average(items, "direct_total_mib_per_frame"),
            "mean_amortized_total_mib_per_frame": average(items, "amortized_total_mib_per_frame"),
            "mean_base_psnr": average(items, "base_psnr"),
            "mean_full_adapter_psnr": average(items, "full_adapter_psnr"),
            "mean_predictor_psnr": average(items, "predictor_psnr"),
            "mean_delta_psnr_vs_base": average(items, "delta_psnr_vs_base"),
            "mean_delta_psnr_vs_full_adapter": average(items, "delta_psnr_vs_full_adapter"),
            "positive_delta_vs_base_count": sum(1 for row in items if float(row["delta_psnr_vs_base"]) > 0.0),
            "near_full_within_0p10db_count": sum(1 for row in items if float(row["delta_psnr_vs_full_adapter"]) >= -0.10),
        })
    return out


def format_float(value):
    return f"{float(value):.6f}"


def write_report(summary, summary_rows, path):
    lines = [
        "# Stage129 Broader Predictor Codec Rendered Validation",
        "",
        "## Scope",
        "",
        "- Render-validates Stage128 predictor-only selected residual codec on 60 eval tasks.",
        "- Uses no residual payload and no selected-index payload.",
        "- Target dense anchors are not loaded; target RGB is used only for offline rendered metrics.",
        "",
        "## Summary",
        "",
        "| setting | role | keep | tasks | rate | base PSNR | full adapter PSNR | predictor PSNR | delta base | delta full | positives | near full |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['setting_label']} | {row['setting_role']} | {row['keep_fraction']} | {row['task_count']} | {format_float(row['mean_direct_total_mib_per_frame'])} | {format_float(row['mean_base_psnr'])} | {format_float(row['mean_full_adapter_psnr'])} | {format_float(row['mean_predictor_psnr'])} | {format_float(row['mean_delta_psnr_vs_base'])} | {format_float(row['mean_delta_psnr_vs_full_adapter'])} | {row['positive_delta_vs_base_count']}/{row['task_count']} | {row['near_full_within_0p10db_count']}/{row['task_count']} |"
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
    parser.add_argument("--stage128_settings", type=Path, default=DEFAULT_STAGE128_SETTINGS)
    parser.add_argument("--stage126_stats", type=Path, default=DEFAULT_STAGE126_STATS)
    parser.add_argument("--stage78_rate_table", type=Path, default=DEFAULT_STAGE78_RATE_TABLE)
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
    device = torch.device(args.device)
    settings = read_csv(args.stage128_settings)
    stats_by_setting = load_stats(args.stage126_stats)
    predictors = {row["setting_label"]: load_predictor(row, device) for row in settings}
    main_rate_lookup = build_main_rate_lookup(read_csv(args.stage78_rate_table))
    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    tasks = select_balanced(parse_task_rows(args.task_manifest, args.task_split, args.codecs, args.gaps), args.max_tasks, args.seed)
    if not tasks:
        raise RuntimeError("No tasks selected")
    adapter = load_adapter(args.adapter, args.hidden_dim, device)
    rows = []
    with torch.no_grad():
        for task in tasks:
            left = load_anchor(task["left_anchor_source_item"], task["left_anchor_source_side"], device, bits=task["bits"], cache=None)
            right = load_anchor(task["right_anchor_source_item"], task["right_anchor_source_side"], device, bits=task["bits"], cache=None)
            left_attrs = flatten_static_anchor(left)
            right_attrs = flatten_static_anchor(right)
            target_rgb = load_rgb(task["target_rgb_path"], opt.image_height, opt.image_width, device)
            t = torch.tensor([task["normalized_time"]], dtype=torch.float32, device=device)
            linear_attrs = flatten_static_anchor(linear_anchor(left, right, task["normalized_time"]))
            adapter_attrs = flatten_static_anchor(adapter(left, right, t, apply_output_constraints=False))
            base_psnr = render_psnr(unflatten_static_anchor(linear_attrs), target_rgb, background, opt)
            full_adapter_psnr = render_psnr(unflatten_static_anchor(adapter_attrs), target_rgb, background, opt)
            gap = int(task["reference_gap"])
            main_rate = main_rate_lookup[("linear", gap)]
            for setting in settings:
                selected_indices = endpoint_diff_topk_indices(left_attrs, right_attrs, float(setting["keep_fraction"]))
                features = selected_residual_feature_matrix(left_attrs, right_attrs, linear_attrs, selected_indices, task["normalized_time"])
                residual_values = predict_residual_values(predictors[setting["setting_label"]], features, stats_by_setting[setting["setting_label"]], device)
                predictor_attrs = apply_selected_residual_values(linear_attrs, selected_indices, residual_values)
                predictor_psnr = render_psnr(unflatten_static_anchor(predictor_attrs), target_rgb, background, opt)
                rows.append({
                    "task_id": task["task_id"],
                    "sequence": task["sequence"],
                    "codec": task["codec"],
                    "reference_gap": gap,
                    "target_index": task["target_index"],
                    "setting_label": setting["setting_label"],
                    "setting_role": setting["setting_role"],
                    "keep_fraction": setting["keep_fraction"],
                    "side_bits": setting["side_bits"],
                    "keep_gaussians": int(selected_indices.numel()),
                    "transmitted_residual_payload_bytes": 0,
                    "transmitted_selected_index_bytes": 0,
                    "q12_main_anchor_mib_per_frame": main_rate,
                    "direct_total_mib_per_frame": main_rate,
                    "amortized_total_mib_per_frame": main_rate,
                    "base_psnr": base_psnr,
                    "full_adapter_psnr": full_adapter_psnr,
                    "predictor_psnr": predictor_psnr,
                    "delta_psnr_vs_base": predictor_psnr - base_psnr,
                    "delta_psnr_vs_full_adapter": predictor_psnr - full_adapter_psnr,
                })
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_rows = summarize(rows)
    rows_csv = args.summary_root / "stage129_broader_predictor_codec_rendered_validation_rows.csv"
    summary_csv = args.summary_root / "stage129_broader_predictor_codec_rendered_validation_summary.csv"
    summary_json = args.summary_root / "stage129_broader_predictor_codec_rendered_validation_summary.json"
    report_md = args.summary_root / "stage129_broader_predictor_codec_rendered_validation_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    summary = {
        "stage": 129,
        "mode": "broader predictor codec rendered validation",
        "stage128_settings": str(args.stage128_settings),
        "stage126_stats": str(args.stage126_stats),
        "task_manifest": str(args.task_manifest),
        "adapter": str(args.adapter),
        "task_count": len(tasks),
        "row_count": len(rows),
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "notes": [
            "Target dense anchors are not loaded or used.",
            "No residual payload bytes are transmitted.",
            "No selected-index bytes are transmitted.",
            "Target RGB is used only for offline rendered metrics.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, summary_rows, report_md)
    print(json.dumps({"summary": str(summary_json), "row_count": len(rows), "summary_rows": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
