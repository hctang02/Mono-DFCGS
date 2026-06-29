import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage126_selected_residual_predictor_dataset_package"

SETTINGS = [
    {"label": "q4_top20", "role": "primary", "keep_fraction": 0.2, "side_bits": 4},
    {"label": "q4_top10", "role": "low_rate", "keep_fraction": 0.1, "side_bits": 4},
]

DATASET_FIELDS = [
    "task_split",
    "setting_label",
    "setting_role",
    "keep_fraction",
    "side_bits",
    "task_id",
    "dataset",
    "split",
    "sequence",
    "codec",
    "bits",
    "reference_gap",
    "target_index",
    "normalized_time",
    "left_anchor_source_item",
    "left_anchor_source_side",
    "right_anchor_source_item",
    "right_anchor_source_side",
    "target_dense_anchor_source_item",
    "target_dense_anchor_source_side",
    "keep_gaussians",
    "feature_dim",
    "residual_dim",
    "feature_rms",
    "label_rms",
]

SUMMARY_FIELDS = [
    "task_split",
    "setting_label",
    "setting_role",
    "keep_fraction",
    "side_bits",
    "task_count",
    "sample_count",
    "mean_keep_gaussians",
    "feature_dim",
    "residual_dim",
    "mean_feature_rms",
    "mean_label_rms",
]

STATS_FIELDS = [
    "setting_label",
    "setting_role",
    "keep_fraction",
    "side_bits",
    "train_task_count",
    "train_sample_count",
    "feature_dim",
    "residual_dim",
    "feature_rms",
    "label_rms",
]


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402
from mono_dfcgs.residual_value_predictor import (  # noqa: E402
    endpoint_diff_topk_indices,
    selected_residual_feature_matrix,
)
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import (  # noqa: E402
    DEFAULT_DENSE_MANIFEST,
    DEFAULT_TASK_MANIFEST,
    build_dense_index,
    linear_anchor,
    load_anchor,
    parse_task_rows,
    select_balanced,
)


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def average(rows, key):
    values = [float(row[key]) for row in rows]
    return sum(values) / max(len(values), 1)


def init_accum(feature_dim, residual_dim):
    return {
        "count": 0,
        "feature_sum": torch.zeros(feature_dim, dtype=torch.float64),
        "feature_sumsq": torch.zeros(feature_dim, dtype=torch.float64),
        "label_sum": torch.zeros(residual_dim, dtype=torch.float64),
        "label_sumsq": torch.zeros(residual_dim, dtype=torch.float64),
    }


def update_accum(accum, features, labels):
    features = features.detach().cpu().double()
    labels = labels.detach().cpu().double()
    accum["count"] += int(features.shape[0])
    accum["feature_sum"] += features.sum(dim=0)
    accum["feature_sumsq"] += (features * features).sum(dim=0)
    accum["label_sum"] += labels.sum(dim=0)
    accum["label_sumsq"] += (labels * labels).sum(dim=0)


def finalize_accum(accum):
    count = max(int(accum["count"]), 1)
    feature_mean = accum["feature_sum"] / count
    feature_var = (accum["feature_sumsq"] / count - feature_mean * feature_mean).clamp_min(0.0)
    label_mean = accum["label_sum"] / count
    label_var = (accum["label_sumsq"] / count - label_mean * label_mean).clamp_min(0.0)
    feature_rms = torch.sqrt(accum["feature_sumsq"].sum() / max(count * int(feature_mean.numel()), 1)).item()
    label_rms = torch.sqrt(accum["label_sumsq"].sum() / max(count * int(label_mean.numel()), 1)).item()
    return {
        "sample_count": int(accum["count"]),
        "feature_dim": int(feature_mean.numel()),
        "residual_dim": int(label_mean.numel()),
        "feature_mean": feature_mean.tolist(),
        "feature_std": torch.sqrt(feature_var + 1e-12).tolist(),
        "label_mean": label_mean.tolist(),
        "label_std": torch.sqrt(label_var + 1e-12).tolist(),
        "feature_rms": feature_rms,
        "label_rms": label_rms,
    }


def summarize_dataset_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task_split"], row["setting_label"])].append(row)
    out = []
    for (task_split, setting), items in sorted(grouped.items()):
        first = items[0]
        out.append({
            "task_split": task_split,
            "setting_label": setting,
            "setting_role": first["setting_role"],
            "keep_fraction": first["keep_fraction"],
            "side_bits": first["side_bits"],
            "task_count": len(items),
            "sample_count": sum(int(row["keep_gaussians"]) for row in items),
            "mean_keep_gaussians": average(items, "keep_gaussians"),
            "feature_dim": first["feature_dim"],
            "residual_dim": first["residual_dim"],
            "mean_feature_rms": average(items, "feature_rms"),
            "mean_label_rms": average(items, "label_rms"),
        })
    return out


def format_float(value):
    return f"{float(value):.6f}"


def write_report(summary, summary_rows, stats_rows, path):
    lines = [
        "# Stage126 Selected Residual Predictor Dataset Package",
        "",
        "## Scope",
        "",
        "- Builds task-level manifests and normalization stats for a dedicated selected residual value predictor.",
        "- Target dense anchors are used only for train/eval labels and aggregate stats.",
        "- No per-Gaussian tensors, anchors, or checkpoints are saved.",
        "",
        "## Dataset Summary",
        "",
        "| split | setting | role | keep | tasks | samples | feature dim | residual dim | feature rms | label rms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['task_split']} | {row['setting_label']} | {row['setting_role']} | {row['keep_fraction']} | {row['task_count']} | {row['sample_count']} | {row['feature_dim']} | {row['residual_dim']} | {format_float(row['mean_feature_rms'])} | {format_float(row['mean_label_rms'])} |"
        )
    lines.extend([
        "",
        "## Train Normalization Stats",
        "",
        "| setting | role | train tasks | train samples | feature rms | label rms |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in stats_rows:
        lines.append(
            f"| {row['setting_label']} | {row['setting_role']} | {row['train_task_count']} | {row['train_sample_count']} | {format_float(row['feature_rms'])} | {format_float(row['label_rms'])} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- dataset rows CSV: `{summary['dataset_rows_csv']}`",
        f"- dataset summary CSV: `{summary['dataset_summary_csv']}`",
        f"- stats CSV: `{summary['stats_csv']}`",
        f"- stats JSON: `{summary['stats_json']}`",
        f"- package JSON: `{summary['package_json']}`",
        f"- report Markdown: `{summary['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--codecs", nargs="+", default=["q12"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--max_train_tasks", type=int, default=120)
    parser.add_argument("--max_eval_tasks", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def task_rows_for_split(args, task_split):
    max_tasks = args.max_train_tasks if task_split == "train" else args.max_eval_tasks
    rows = parse_task_rows(args.task_manifest, task_split, args.codecs, args.gaps)
    return select_balanced(rows, max_tasks, args.seed + (0 if task_split == "train" else 1000))


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    selected_tasks = {
        "train": task_rows_for_split(args, "train"),
        "eval": task_rows_for_split(args, "eval"),
    }
    all_tasks = selected_tasks["train"] + selected_tasks["eval"]
    if not all_tasks:
        raise RuntimeError("No tasks selected")
    dense_index = build_dense_index(args.dense_manifest, sorted({row["split"] for row in all_tasks}))
    dataset_rows = []
    accum_by_setting = {}

    with torch.no_grad():
        for task_split, tasks in selected_tasks.items():
            for task in tasks:
                dense_key = (task["dataset"], task["split"], task["sequence"], task["target_index"])
                if dense_key not in dense_index:
                    raise KeyError(f"Missing dense target anchor for {dense_key}")
                target_item, target_side = dense_index[dense_key]
                left = load_anchor(task["left_anchor_source_item"], task["left_anchor_source_side"], device, bits=task["bits"], cache=None)
                right = load_anchor(task["right_anchor_source_item"], task["right_anchor_source_side"], device, bits=task["bits"], cache=None)
                target = load_anchor(target_item, target_side, device, bits=None, cache=None)
                left_attrs = flatten_static_anchor(left)
                right_attrs = flatten_static_anchor(right)
                base_attrs = flatten_static_anchor(linear_anchor(left, right, task["normalized_time"]))
                target_attrs = flatten_static_anchor(target)
                for setting in SETTINGS:
                    selected_indices = endpoint_diff_topk_indices(left_attrs, right_attrs, setting["keep_fraction"])
                    features = selected_residual_feature_matrix(left_attrs, right_attrs, base_attrs, selected_indices, task["normalized_time"])
                    keep_idx = torch.as_tensor(selected_indices, dtype=torch.int64, device=device).reshape(-1)
                    labels = target_attrs[0, keep_idx, :] - base_attrs[0, keep_idx, :]
                    feature_rms = torch.sqrt(torch.mean(features.float() * features.float())).item() if features.numel() > 0 else 0.0
                    label_rms = torch.sqrt(torch.mean(labels.float() * labels.float())).item() if labels.numel() > 0 else 0.0
                    if task_split == "train" and features.numel() > 0:
                        if setting["label"] not in accum_by_setting:
                            accum_by_setting[setting["label"]] = init_accum(int(features.shape[1]), int(labels.shape[1]))
                        update_accum(accum_by_setting[setting["label"]], features, labels)
                    dataset_rows.append({
                        "task_split": task_split,
                        "setting_label": setting["label"],
                        "setting_role": setting["role"],
                        "keep_fraction": setting["keep_fraction"],
                        "side_bits": setting["side_bits"],
                        "task_id": task["task_id"],
                        "dataset": task["dataset"],
                        "split": task["split"],
                        "sequence": task["sequence"],
                        "codec": task["codec"],
                        "bits": task["bits"],
                        "reference_gap": task["reference_gap"],
                        "target_index": task["target_index"],
                        "normalized_time": task["normalized_time"],
                        "left_anchor_source_item": task["left_anchor_source_item"],
                        "left_anchor_source_side": task["left_anchor_source_side"],
                        "right_anchor_source_item": task["right_anchor_source_item"],
                        "right_anchor_source_side": task["right_anchor_source_side"],
                        "target_dense_anchor_source_item": target_item,
                        "target_dense_anchor_source_side": target_side,
                        "keep_gaussians": int(selected_indices.numel()),
                        "feature_dim": int(features.shape[1]),
                        "residual_dim": int(labels.shape[1]),
                        "feature_rms": feature_rms,
                        "label_rms": label_rms,
                    })
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary_rows = summarize_dataset_rows(dataset_rows)
    stats = {}
    stats_rows = []
    train_task_count = sum(1 for row in dataset_rows if row["task_split"] == "train" and row["setting_label"] == SETTINGS[0]["label"])
    for setting in SETTINGS:
        setting_stats = finalize_accum(accum_by_setting[setting["label"]])
        stats[setting["label"]] = {**setting, **setting_stats}
        stats_rows.append({
            "setting_label": setting["label"],
            "setting_role": setting["role"],
            "keep_fraction": setting["keep_fraction"],
            "side_bits": setting["side_bits"],
            "train_task_count": train_task_count,
            "train_sample_count": setting_stats["sample_count"],
            "feature_dim": setting_stats["feature_dim"],
            "residual_dim": setting_stats["residual_dim"],
            "feature_rms": setting_stats["feature_rms"],
            "label_rms": setting_stats["label_rms"],
        })

    dataset_rows_csv = args.summary_root / "stage126_selected_residual_predictor_dataset_rows.csv"
    dataset_summary_csv = args.summary_root / "stage126_selected_residual_predictor_dataset_summary.csv"
    stats_csv = args.summary_root / "stage126_selected_residual_predictor_train_stats.csv"
    stats_json = args.summary_root / "stage126_selected_residual_predictor_train_stats.json"
    package_json = args.summary_root / "stage126_selected_residual_predictor_dataset_package.json"
    report_md = args.summary_root / "stage126_selected_residual_predictor_dataset_report.md"
    write_csv(dataset_rows, dataset_rows_csv, DATASET_FIELDS)
    write_csv(summary_rows, dataset_summary_csv, SUMMARY_FIELDS)
    write_csv(stats_rows, stats_csv, STATS_FIELDS)
    stats_json.write_text(json.dumps({"stage": 126, "stats_by_setting": stats}, indent=2) + "\n", encoding="utf-8")
    package = {
        "stage": 126,
        "mode": "selected residual predictor dataset package",
        "task_manifest": str(args.task_manifest),
        "dense_manifest": str(args.dense_manifest),
        "codecs": args.codecs,
        "gaps": args.gaps,
        "train_task_count": len(selected_tasks["train"]),
        "eval_task_count": len(selected_tasks["eval"]),
        "dataset_row_count": len(dataset_rows),
        "settings": SETTINGS,
        "dataset_rows_csv": str(dataset_rows_csv),
        "dataset_summary_csv": str(dataset_summary_csv),
        "stats_csv": str(stats_csv),
        "stats_json": str(stats_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "notes": [
            "Target dense anchors are used only for label/stat computation.",
            "No per-Gaussian tensor data is saved in this package.",
            "Decoder-side features use left/right/base attrs and normalized time only.",
        ],
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, summary_rows, stats_rows, report_md)
    print(json.dumps({"package": str(package_json), "dataset_rows": len(dataset_rows), "stats": stats_rows}, indent=2))


if __name__ == "__main__":
    main()
