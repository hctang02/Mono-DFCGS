import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage102_group_specific_selector_heads"


sys.path.insert(0, str(REPO_ROOT))
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_ADAPTER, load_adapter  # noqa: E402
from scripts.run_stage98_residual_importance_predictor_smoke import (  # noqa: E402
    DEFAULT_STAGE97_TASKS,
    ImportanceMLP,
    build_task_tensors,
    parse_rows,
    topk_metrics,
)
from scripts.run_stage101_enhanced_selector_feature_sweep import (  # noqa: E402
    BASE_FEATURE_DIM,
    make_full_features,
    normalized_energy_target,
)


TRAIN_LOG_FIELDS = [
    "model_name",
    "selector_scope",
    "group_key",
    "objective",
    "step",
    "loss",
    "batch_positive_rate",
    "batch_energy_target_mean",
    "train_example_count",
]

ROW_FIELDS = [
    "stage97_task_id",
    "source_task_id",
    "task_split",
    "sequence",
    "codec",
    "reference_gap",
    "target_index",
    "base_method",
    "selector_scope",
    "group_key",
    "objective",
    "candidate",
    "keep_fraction",
    "keep_gaussians",
    "precision_at_keep",
    "energy_recall_total",
    "oracle_energy_recall_total",
    "relative_energy_recall_vs_oracle",
]

SUMMARY_FIELDS = [
    "selector_scope",
    "group_key",
    "objective",
    "candidate",
    "base_method",
    "reference_gap",
    "task_count",
    "mean_precision_at_keep",
    "mean_energy_recall_total",
    "mean_oracle_energy_recall_total",
    "mean_relative_energy_recall_vs_oracle",
]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_group_key(base_method, reference_gap):
    return f"{base_method}__gap{int(reference_gap)}"


def objective_loss(logits, labels, energy_targets, pos_weight, objective):
    if objective == "topk_bce":
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    if objective == "energy_regression":
        return F.mse_loss(torch.sigmoid(logits), energy_targets)
    raise ValueError(f"Unknown objective: {objective}")


def collect_training_examples(tasks, adapter, args, device):
    group_keys = [make_group_key(method, gap) for method in args.base_methods for gap in args.gaps]
    group_to_id = {key: idx for idx, key in enumerate(group_keys)}
    cache = {}
    xs = []
    ys = []
    energy_targets = []
    group_ids = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    method_ids = {"linear": 0.0, "stage65_adapter": 1.0}
    with torch.no_grad():
        for task in tasks:
            left_attrs, right_attrs, target_attrs, base_attrs_by_method = build_task_tensors(task, adapter, device, cache)
            for method in args.base_methods:
                group_key = make_group_key(method, task["reference_gap"])
                base_attrs = base_attrs_by_method[method]
                residual = target_attrs - base_attrs
                energy = torch.sum(residual[0] ** 2, dim=-1)
                keep_count = max(1, int(round(energy.numel() * args.keep_fraction)))
                teacher_idx = torch.topk(energy, k=keep_count, largest=True).indices
                labels = torch.zeros_like(energy)
                labels[teacher_idx] = 1.0
                energy_target = normalized_energy_target(energy)
                features = make_full_features(left_attrs, right_attrs, base_attrs, task, method_ids[method])[:, :BASE_FEATURE_DIM]
                sample_count = min(args.train_gaussians_per_task, features.shape[0])
                sample_idx = torch.randperm(features.shape[0], generator=generator, device=device)[:sample_count]
                xs.append(features[sample_idx].detach().cpu())
                ys.append(labels[sample_idx].detach().cpu())
                energy_targets.append(energy_target[sample_idx].detach().cpu())
                group_ids.append(torch.full((sample_count,), group_to_id[group_key], dtype=torch.long))
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return (
        torch.cat(xs, dim=0),
        torch.cat(ys, dim=0),
        torch.cat(energy_targets, dim=0),
        torch.cat(group_ids, dim=0),
        group_to_id,
    )


def train_selector(train_x, train_y, train_energy, train_group, args, device, objective, selector_scope, group_key, group_to_id, model_index):
    if selector_scope == "shared":
        train_mask = torch.ones(train_group.shape[0], dtype=torch.bool)
    elif selector_scope == "group_specific":
        train_mask = train_group == group_to_id[group_key]
    else:
        raise ValueError(f"Unknown selector scope: {selector_scope}")
    model_name = f"{selector_scope}__{group_key}__{objective}"
    selected_count = int(train_mask.sum().item())
    if selected_count <= 0:
        raise ValueError(f"No training examples for {model_name}")
    torch.manual_seed(args.seed + 100 + model_index)
    selected_x = train_x[train_mask].to(device)
    selected_y = train_y[train_mask].to(device)
    selected_energy = train_energy[train_mask].to(device)
    mean = selected_x.mean(dim=0, keepdim=True)
    std = selected_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    selected_x = ((selected_x - mean) / std).detach()
    predictor = ImportanceMLP(selected_x.shape[1], args.hidden_dim).to(device)
    pos = selected_y.sum().clamp_min(1.0)
    neg = (selected_y.numel() - selected_y.sum()).clamp_min(1.0)
    pos_weight = neg / pos
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000 + model_index)
    logs = []
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, selected_x.shape[0], (args.batch_size,), generator=generator, device=device)
        batch_x = selected_x[idx]
        batch_y = selected_y[idx]
        batch_energy = selected_energy[idx]
        logits = predictor(batch_x)
        loss = objective_loss(logits, batch_y, batch_energy, pos_weight, objective)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            logs.append({
                "model_name": model_name,
                "selector_scope": selector_scope,
                "group_key": group_key,
                "objective": objective,
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "batch_positive_rate": float(batch_y.mean().detach().cpu().item()),
                "batch_energy_target_mean": float(batch_energy.mean().detach().cpu().item()),
                "train_example_count": selected_count,
            })
    predictor.eval()
    return {
        "model_name": model_name,
        "selector_scope": selector_scope,
        "group_key": group_key,
        "objective": objective,
        "predictor": predictor,
        "mean": mean,
        "std": std,
    }, logs


def evaluate(tasks, adapter, models, args, device):
    cache = {}
    rows = []
    method_ids = {"linear": 0.0, "stage65_adapter": 1.0}
    shared_models = [model for model in models if model["selector_scope"] == "shared"]
    group_models = defaultdict(list)
    for model in models:
        if model["selector_scope"] == "group_specific":
            group_models[model["group_key"]].append(model)
    with torch.no_grad():
        for task in tasks:
            left_attrs, right_attrs, target_attrs, base_attrs_by_method = build_task_tensors(task, adapter, device, cache)
            endpoint_score = torch.sum((right_attrs[0] - left_attrs[0]) ** 2, dim=-1)
            for method in args.base_methods:
                group_key = make_group_key(method, task["reference_gap"])
                base_attrs = base_attrs_by_method[method]
                residual = target_attrs - base_attrs
                energy = torch.sum(residual[0] ** 2, dim=-1)
                keep_count = max(1, int(round(energy.numel() * args.keep_fraction)))
                features = make_full_features(left_attrs, right_attrs, base_attrs, task, method_ids[method])[:, :BASE_FEATURE_DIM]
                candidate_scores = [("endpoint_only", group_key, "endpoint_diff_baseline", "endpoint_diff_baseline", endpoint_score)]
                for model in shared_models + group_models[group_key]:
                    logits = []
                    for start in range(0, features.shape[0], args.eval_batch_size):
                        chunk = features[start:start + args.eval_batch_size]
                        logits.append(model["predictor"]((chunk - model["mean"]) / model["std"]))
                    candidate_scores.append((model["selector_scope"], model["group_key"], model["objective"], "mlp_selector", torch.cat(logits, dim=0)))
                for selector_scope, row_group_key, objective, candidate, scores in candidate_scores:
                    metrics = topk_metrics(scores, energy, keep_count)
                    rows.append({
                        "stage97_task_id": task["stage97_task_id"],
                        "source_task_id": task["source_task_id"],
                        "task_split": task["task_split"],
                        "sequence": task["sequence"],
                        "codec": task["codec"],
                        "reference_gap": task["reference_gap"],
                        "target_index": task["target_index"],
                        "base_method": method,
                        "selector_scope": selector_scope,
                        "group_key": row_group_key,
                        "objective": objective,
                        "candidate": candidate,
                        "keep_fraction": args.keep_fraction,
                        "keep_gaussians": keep_count,
                        "precision_at_keep": metrics["precision_at_keep"],
                        "energy_recall_total": metrics["energy_recall_total"],
                        "oracle_energy_recall_total": metrics["oracle_energy_recall_total"],
                        "relative_energy_recall_vs_oracle": metrics["relative_energy_recall_vs_oracle"],
                    })
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["selector_scope"], row["group_key"], row["objective"], row["candidate"], row["base_method"], row["reference_gap"])].append(row)
    out = []
    for (scope, group_key, objective, candidate, method, gap), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][4], int(item[0][5]), item[0][2], item[0][1])):
        out.append({
            "selector_scope": scope,
            "group_key": group_key,
            "objective": objective,
            "candidate": candidate,
            "base_method": method,
            "reference_gap": int(gap),
            "task_count": len(items),
            "mean_precision_at_keep": sum(float(row["precision_at_keep"]) for row in items) / len(items),
            "mean_energy_recall_total": sum(float(row["energy_recall_total"]) for row in items) / len(items),
            "mean_oracle_energy_recall_total": sum(float(row["oracle_energy_recall_total"]) for row in items) / len(items),
            "mean_relative_energy_recall_vs_oracle": sum(float(row["relative_energy_recall_vs_oracle"]) for row in items) / len(items),
        })
    return out


def group_comparison(summary_rows):
    by_group = defaultdict(lambda: {"endpoint": None, "shared": None, "group_specific": None})
    for row in summary_rows:
        key = (row["base_method"], row["reference_gap"])
        if row["candidate"] == "endpoint_diff_baseline":
            by_group[key]["endpoint"] = row
            continue
        if row["candidate"] != "mlp_selector":
            continue
        scope = row["selector_scope"]
        current = by_group[key].get(scope)
        if current is None or row["mean_energy_recall_total"] > current["mean_energy_recall_total"]:
            by_group[key][scope] = row
    out = []
    for key in sorted(by_group, key=lambda item: (item[0], int(item[1]))):
        endpoint = by_group[key]["endpoint"]
        shared = by_group[key]["shared"]
        group_specific = by_group[key]["group_specific"]
        if not endpoint or not shared or not group_specific:
            continue
        out.append({
            "base_method": key[0],
            "reference_gap": int(key[1]),
            "endpoint_energy_recall": endpoint["mean_energy_recall_total"],
            "shared_objective": shared["objective"],
            "shared_energy_recall": shared["mean_energy_recall_total"],
            "shared_relative_recall": shared["mean_relative_energy_recall_vs_oracle"],
            "group_objective": group_specific["objective"],
            "group_energy_recall": group_specific["mean_energy_recall_total"],
            "group_relative_recall": group_specific["mean_relative_energy_recall_vs_oracle"],
            "group_minus_shared_energy_recall": group_specific["mean_energy_recall_total"] - shared["mean_energy_recall_total"],
            "group_precision_at_keep": group_specific["mean_precision_at_keep"],
        })
    return out


def write_report(summary, summary_rows, path):
    comparison_rows = group_comparison(summary_rows)
    lines = [
        "# Stage102 Group-Specific Selector Heads",
        "",
        "## Configuration",
        "",
        f"- train tasks: `{summary['train_task_count']}`",
        f"- eval tasks: `{summary['eval_task_count']}`",
        f"- train examples: `{summary['train_example_count']}`",
        f"- base feature dim: `{summary['feature_dim']}`",
        f"- keep fraction: `{summary['keep_fraction']}`",
        f"- objectives: `{', '.join(summary['objectives'])}`",
        "- no rendering, checkpoint, or heavy tensor output",
        "",
        "## Group Comparison",
        "",
        "| base | gap | endpoint recall | best shared | shared recall | best group | group recall | group-shared | group relative | group precision |",
        "|---|---:|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['base_method']} | {row['reference_gap']} | {row['endpoint_energy_recall']} | {row['shared_objective']} | {row['shared_energy_recall']} | {row['group_objective']} | {row['group_energy_recall']} | {row['group_minus_shared_energy_recall']} | {row['group_relative_recall']} | {row['group_precision_at_keep']} |"
        )
    lines.extend([
        "",
        "## Summary",
        "",
        "| scope | group | objective | candidate | base | gap | tasks | precision@keep | energy recall | oracle recall | relative recall |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        lines.append(
            f"| {row['selector_scope']} | {row['group_key']} | {row['objective']} | {row['candidate']} | {row['base_method']} | {row['reference_gap']} | {row['task_count']} | {row['mean_precision_at_keep']} | {row['mean_energy_recall_total']} | {row['mean_oracle_energy_recall_total']} | {row['mean_relative_energy_recall_vs_oracle']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Group-specific heads are trained per base method and reference gap.",
        "- Features are the Stage100 base decoder-available features only.",
        "- Target dense anchors are used only for offline labels/metrics.",
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
    adapter = load_adapter(args.adapter, hidden_dim=256, device=device)
    train_tasks = parse_rows(args.tasks_csv, "train", args.gaps, args.max_train_tasks, args.seed)
    eval_tasks = parse_rows(args.tasks_csv, "eval", args.gaps, args.max_eval_tasks, args.seed + 1)
    train_x, train_y, train_energy, train_group, group_to_id = collect_training_examples(train_tasks, adapter, args, device)

    models = []
    train_logs = []
    model_index = 0
    for objective in args.objectives:
        model, logs = train_selector(train_x, train_y, train_energy, train_group, args, device, objective, "shared", "all_groups", group_to_id, model_index)
        models.append(model)
        train_logs.extend(logs)
        model_index += 1
    for group_key in sorted(group_to_id):
        for objective in args.objectives:
            model, logs = train_selector(train_x, train_y, train_energy, train_group, args, device, objective, "group_specific", group_key, group_to_id, model_index)
            models.append(model)
            train_logs.extend(logs)
            model_index += 1

    eval_rows = evaluate(eval_tasks, adapter, models, args, device)
    summary_rows = summarize(eval_rows)
    comparison_rows = group_comparison(summary_rows)

    rows_csv = args.summary_root / "stage102_group_specific_selector_rows.csv"
    summary_csv = args.summary_root / "stage102_group_specific_selector_summary.csv"
    comparison_csv = args.summary_root / "stage102_group_specific_selector_comparison.csv"
    train_log_csv = args.summary_root / "stage102_group_specific_selector_train_log.csv"
    summary_json = args.summary_root / "stage102_group_specific_selector_summary.json"
    report_md = args.summary_root / "stage102_group_specific_selector_report.md"
    write_csv(eval_rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(comparison_rows, comparison_csv, [
        "base_method",
        "reference_gap",
        "endpoint_energy_recall",
        "shared_objective",
        "shared_energy_recall",
        "shared_relative_recall",
        "group_objective",
        "group_energy_recall",
        "group_relative_recall",
        "group_minus_shared_energy_recall",
        "group_precision_at_keep",
    ])
    write_csv(train_logs, train_log_csv, TRAIN_LOG_FIELDS)
    summary = {
        "stage": 102,
        "mode": "group-specific residual selector heads",
        "tasks_csv": str(args.tasks_csv),
        "adapter": str(args.adapter),
        "gaps": args.gaps,
        "base_methods": args.base_methods,
        "objectives": args.objectives,
        "train_task_count": len(train_tasks),
        "eval_task_count": len(eval_tasks),
        "train_example_count": int(train_x.shape[0]),
        "feature_dim": BASE_FEATURE_DIM,
        "positive_rate": float(train_y.mean().item()),
        "energy_target_mean": float(train_energy.mean().item()),
        "keep_fraction": args.keep_fraction,
        "train_steps": args.train_steps,
        "group_to_id": group_to_id,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "comparison_csv": str(comparison_csv),
        "train_log_csv": str(train_log_csv),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "comparison_rows": comparison_rows,
        "notes": [
            "No model checkpoint or heavy tensor is saved.",
            "Group-specific heads are trained per base_method and reference_gap.",
            "Features are Stage100 base decoder-available features only.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, summary_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "comparison": str(comparison_csv),
        "train_task_count": len(train_tasks),
        "eval_task_count": len(eval_tasks),
        "train_example_count": int(train_x.shape[0]),
    }, indent=2))


if __name__ == "__main__":
    main()
