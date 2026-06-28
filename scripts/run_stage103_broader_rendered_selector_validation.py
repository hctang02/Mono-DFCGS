import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage103_broader_rendered_selector_validation"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import unflatten_static_anchor  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import load_rgb  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_ADAPTER, load_adapter  # noqa: E402
from scripts.run_stage86_rendered_residual_sideinfo_smoke import render_psnr  # noqa: E402
from scripts.run_stage98_residual_importance_predictor_smoke import (  # noqa: E402
    DEFAULT_STAGE97_TASKS,
    build_task_tensors,
    parse_rows,
    topk_metrics,
)
from scripts.run_stage99_predictor_selected_sideinfo_render_smoke import quantize_selected_residual_anchor  # noqa: E402
from scripts.run_stage101_enhanced_selector_feature_sweep import BASE_FEATURE_DIM, make_full_features  # noqa: E402
from scripts.run_stage102_group_specific_selector_heads import collect_training_examples, train_selector  # noqa: E402


ROW_FIELDS = [
    "stage97_task_id",
    "source_task_id",
    "sequence",
    "codec",
    "reference_gap",
    "target_index",
    "base_method",
    "candidate",
    "keep_fraction",
    "keep_gaussians",
    "precision_at_keep",
    "energy_recall_total",
    "oracle_energy_recall_total",
    "relative_energy_recall_vs_oracle",
    "base_psnr",
    "sideinfo_psnr",
    "teacher_oracle_sideinfo_psnr",
    "delta_psnr_vs_base",
    "gap_to_teacher_oracle_psnr",
]

SUMMARY_FIELDS = [
    "candidate",
    "base_method",
    "reference_gap",
    "task_count",
    "mean_precision_at_keep",
    "mean_energy_recall_total",
    "mean_relative_energy_recall_vs_oracle",
    "mean_base_psnr",
    "mean_sideinfo_psnr",
    "mean_teacher_oracle_sideinfo_psnr",
    "mean_delta_psnr_vs_base",
    "mean_gap_to_teacher_oracle_psnr",
]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["candidate"], row["base_method"], row["reference_gap"])].append(row)
    out = []
    for (candidate, method, gap), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], int(item[0][2]))):
        def avg(key):
            return sum(float(row[key]) for row in items) / len(items)
        out.append({
            "candidate": candidate,
            "base_method": method,
            "reference_gap": int(gap),
            "task_count": len(items),
            "mean_precision_at_keep": avg("precision_at_keep"),
            "mean_energy_recall_total": avg("energy_recall_total"),
            "mean_relative_energy_recall_vs_oracle": avg("relative_energy_recall_vs_oracle"),
            "mean_base_psnr": avg("base_psnr"),
            "mean_sideinfo_psnr": avg("sideinfo_psnr"),
            "mean_teacher_oracle_sideinfo_psnr": avg("teacher_oracle_sideinfo_psnr"),
            "mean_delta_psnr_vs_base": avg("delta_psnr_vs_base"),
            "mean_gap_to_teacher_oracle_psnr": avg("gap_to_teacher_oracle_psnr"),
        })
    return out


def best_learned_by_group(summary_rows):
    best = {}
    for row in summary_rows:
        if not row["candidate"].startswith("shared_"):
            continue
        key = (row["base_method"], row["reference_gap"])
        if key not in best or row["mean_sideinfo_psnr"] > best[key]["mean_sideinfo_psnr"]:
            best[key] = row
    return [best[key] for key in sorted(best, key=lambda item: (item[0], int(item[1])))]


def write_report(summary, summary_rows, path):
    best_rows = best_learned_by_group(summary_rows)
    lines = [
        "# Stage103 Broader Rendered Selector Validation",
        "",
        "## Configuration",
        "",
        f"- train tasks: `{summary['train_task_count']}`",
        f"- eval tasks: `{summary['eval_task_count']}`",
        f"- train examples: `{summary['train_example_count']}`",
        f"- keep fraction: `{summary['keep_fraction']}`",
        f"- side bits: `{summary['side_bits']}`",
        f"- selector objectives: `{', '.join(summary['objectives'])}`",
        "- predicted indices use teacher residual values to isolate selection error",
        "- no checkpoint or heavy tensor output",
        "",
        "## Best Learned Rendered Selector By Group",
        "",
        "| base | gap | candidate | side PSNR | teacher PSNR | delta | gap to teacher | energy recall |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            f"| {row['base_method']} | {row['reference_gap']} | {row['candidate']} | {row['mean_sideinfo_psnr']} | {row['mean_teacher_oracle_sideinfo_psnr']} | {row['mean_delta_psnr_vs_base']} | {row['mean_gap_to_teacher_oracle_psnr']} | {row['mean_energy_recall_total']} |"
        )
    lines.extend([
        "",
        "## Summary",
        "",
        "| candidate | base | gap | tasks | precision | energy recall | relative recall | side PSNR | teacher PSNR | delta | gap to teacher |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        lines.append(
            f"| {row['candidate']} | {row['base_method']} | {row['reference_gap']} | {row['task_count']} | {row['mean_precision_at_keep']} | {row['mean_energy_recall_total']} | {row['mean_relative_energy_recall_vs_oracle']} | {row['mean_sideinfo_psnr']} | {row['mean_teacher_oracle_sideinfo_psnr']} | {row['mean_delta_psnr_vs_base']} | {row['mean_gap_to_teacher_oracle_psnr']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `teacher_oracle_topk` is not deployable and only bounds top10 index selection.",
        "- Shared learned selectors use decoder-available Stage100 base features.",
        "- Residual values are still teacher values at selected indices; residual value prediction remains unresolved.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_csv", type=Path, default=DEFAULT_STAGE97_TASKS)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--base_methods", nargs="+", default=["linear", "stage65_adapter"])
    parser.add_argument("--objectives", nargs="+", default=["topk_bce", "energy_regression"])
    parser.add_argument("--max_train_tasks", type=int, default=96)
    parser.add_argument("--max_eval_tasks", type=int, default=60)
    parser.add_argument("--train_gaussians_per_task", type=int, default=3072)
    parser.add_argument("--keep_fraction", type=float, default=0.1)
    parser.add_argument("--side_bits", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--train_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--eval_batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    adapter = load_adapter(args.adapter, hidden_dim=256, device=device)
    train_tasks = parse_rows(args.tasks_csv, "train", args.gaps, args.max_train_tasks, args.seed)
    eval_tasks = parse_rows(args.tasks_csv, "eval", args.gaps, args.max_eval_tasks, args.seed + 1)
    train_x, train_y, train_energy, train_group, group_to_id = collect_training_examples(train_tasks, adapter, args, device)

    models = {}
    train_logs = []
    for model_index, objective in enumerate(args.objectives):
        model, logs = train_selector(train_x, train_y, train_energy, train_group, args, device, objective, "shared", "all_groups", group_to_id, model_index)
        models[f"shared_{objective}"] = model
        train_logs.extend(logs)

    rows = []
    cache = {}
    method_ids = {"linear": 0.0, "stage65_adapter": 1.0}
    with torch.no_grad():
        for task in eval_tasks:
            left_attrs, right_attrs, target_attrs, base_attrs_by_method = build_task_tensors(task, adapter, device, cache)
            target_rgb = load_rgb(task["target_rgb_path"], opt.image_height, opt.image_width, device)
            endpoint_score = torch.sum((right_attrs[0] - left_attrs[0]) ** 2, dim=-1)
            for method in args.base_methods:
                base_attrs = base_attrs_by_method[method]
                base_psnr = render_psnr(unflatten_static_anchor(base_attrs), target_rgb, background, opt)
                residual = target_attrs - base_attrs
                energy = torch.sum(residual[0] ** 2, dim=-1)
                keep_count = max(1, int(round(energy.numel() * args.keep_fraction)))
                teacher_idx = torch.topk(energy, k=keep_count, largest=True).indices
                features = make_full_features(left_attrs, right_attrs, base_attrs, task, method_ids[method])[:, :BASE_FEATURE_DIM]
                candidate_scores = {
                    "teacher_oracle_topk": energy,
                    "endpoint_diff_baseline": endpoint_score,
                }
                for candidate_name, model in models.items():
                    logits = []
                    for start in range(0, features.shape[0], args.eval_batch_size):
                        chunk = features[start:start + args.eval_batch_size]
                        logits.append(model["predictor"]((chunk - model["mean"]) / model["std"]))
                    candidate_scores[candidate_name] = torch.cat(logits, dim=0)

                candidate_indices = {
                    candidate: torch.topk(scores, k=keep_count, largest=True).indices
                    for candidate, scores in candidate_scores.items()
                }
                teacher_anchor = quantize_selected_residual_anchor(base_attrs, target_attrs, teacher_idx, args.side_bits)
                teacher_psnr = render_psnr(teacher_anchor, target_rgb, background, opt)
                for candidate, selected_idx in candidate_indices.items():
                    if candidate == "teacher_oracle_topk":
                        side_psnr = teacher_psnr
                    else:
                        side_anchor = quantize_selected_residual_anchor(base_attrs, target_attrs, selected_idx, args.side_bits)
                        side_psnr = render_psnr(side_anchor, target_rgb, background, opt)
                    metrics = topk_metrics(candidate_scores[candidate], energy, keep_count)
                    rows.append({
                        "stage97_task_id": task["stage97_task_id"],
                        "source_task_id": task["source_task_id"],
                        "sequence": task["sequence"],
                        "codec": task["codec"],
                        "reference_gap": task["reference_gap"],
                        "target_index": task["target_index"],
                        "base_method": method,
                        "candidate": candidate,
                        "keep_fraction": args.keep_fraction,
                        "keep_gaussians": keep_count,
                        "precision_at_keep": metrics["precision_at_keep"],
                        "energy_recall_total": metrics["energy_recall_total"],
                        "oracle_energy_recall_total": metrics["oracle_energy_recall_total"],
                        "relative_energy_recall_vs_oracle": metrics["relative_energy_recall_vs_oracle"],
                        "base_psnr": base_psnr,
                        "sideinfo_psnr": side_psnr,
                        "teacher_oracle_sideinfo_psnr": teacher_psnr,
                        "delta_psnr_vs_base": side_psnr - base_psnr,
                        "gap_to_teacher_oracle_psnr": side_psnr - teacher_psnr,
                    })
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_rows = summarize(rows)
    rows_csv = args.summary_root / "stage103_broader_rendered_selector_rows.csv"
    summary_csv = args.summary_root / "stage103_broader_rendered_selector_summary.csv"
    train_log_csv = args.summary_root / "stage103_broader_rendered_selector_train_log.csv"
    summary_json = args.summary_root / "stage103_broader_rendered_selector_summary.json"
    report_md = args.summary_root / "stage103_broader_rendered_selector_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(train_logs, train_log_csv, [
        "model_name",
        "selector_scope",
        "group_key",
        "objective",
        "step",
        "loss",
        "batch_positive_rate",
        "batch_energy_target_mean",
        "train_example_count",
    ])
    summary = {
        "stage": 103,
        "mode": "broader rendered selector validation",
        "tasks_csv": str(args.tasks_csv),
        "adapter": str(args.adapter),
        "gaps": args.gaps,
        "base_methods": args.base_methods,
        "objectives": args.objectives,
        "train_task_count": len(train_tasks),
        "eval_task_count": len(eval_tasks),
        "train_example_count": int(train_x.shape[0]),
        "feature_dim": BASE_FEATURE_DIM,
        "keep_fraction": args.keep_fraction,
        "side_bits": args.side_bits,
        "train_steps": args.train_steps,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "train_log_csv": str(train_log_csv),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "best_learned_rows": best_learned_by_group(summary_rows),
        "notes": [
            "No model checkpoint or heavy tensor is saved.",
            "Predicted indices use teacher residual values to isolate selection error.",
            "Residual value prediction remains unresolved.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, summary_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "train_task_count": len(train_tasks),
        "eval_task_count": len(eval_tasks),
        "train_example_count": int(train_x.shape[0]),
    }, indent=2))


if __name__ == "__main__":
    main()
