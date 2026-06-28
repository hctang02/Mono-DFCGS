import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage101_enhanced_selector_feature_sweep"
BASE_FEATURE_DIM = 67
NORM_FEATURE_DIM = BASE_FEATURE_DIM + 7
RANK_FEATURE_DIM = NORM_FEATURE_DIM + 1


sys.path.insert(0, str(REPO_ROOT))
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_ADAPTER, load_adapter  # noqa: E402
from scripts.run_stage98_residual_importance_predictor_smoke import (  # noqa: E402
    DEFAULT_STAGE97_TASKS,
    ImportanceMLP,
    build_task_tensors,
    make_features,
    parse_rows,
    topk_metrics,
)


TRAIN_LOG_FIELDS = [
    "model_name",
    "feature_mode",
    "objective",
    "step",
    "loss",
    "batch_positive_rate",
    "batch_energy_target_mean",
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
    "feature_mode",
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
    "feature_mode",
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


def feature_dim(feature_mode):
    if feature_mode == "stage100_base":
        return BASE_FEATURE_DIM
    if feature_mode == "gap_endpoint_norms":
        return NORM_FEATURE_DIM
    if feature_mode == "gap_endpoint_rank":
        return RANK_FEATURE_DIM
    raise ValueError(f"Unknown feature mode: {feature_mode}")


def normalize_per_task(values):
    values = values.clamp_min(0.0)
    scaled = torch.log1p(values / values.mean().clamp_min(1e-12))
    return scaled / scaled.amax().clamp_min(1e-12)


def rank_percentile(scores):
    flat = scores.squeeze(-1)
    n = flat.numel()
    order = torch.argsort(flat, descending=True)
    ranks = torch.empty(n, dtype=flat.dtype, device=flat.device)
    ranks[order] = torch.arange(n, dtype=flat.dtype, device=flat.device)
    if n <= 1:
        return torch.ones((n, 1), dtype=flat.dtype, device=flat.device)
    return (1.0 - ranks / float(n - 1)).unsqueeze(-1)


def make_full_features(left_attrs, right_attrs, base_attrs, task, method_id):
    base_features = make_features(left_attrs, right_attrs, base_attrs, task["normalized_time"], method_id)
    if base_features.shape[-1] != BASE_FEATURE_DIM:
        raise ValueError(f"Unexpected base feature dim: {base_features.shape[-1]}")
    diff = right_attrs[0] - left_attrs[0]
    abs_diff = torch.abs(diff)
    endpoint_l2 = torch.sum(diff ** 2, dim=-1, keepdim=True)
    endpoint_l1 = torch.mean(abs_diff, dim=-1, keepdim=True)
    endpoint_max = torch.amax(abs_diff, dim=-1, keepdim=True)
    endpoint_l2_norm = normalize_per_task(endpoint_l2)
    endpoint_l1_norm = normalize_per_task(endpoint_l1)
    endpoint_max_norm = normalize_per_task(endpoint_max)
    n = base_features.shape[0]
    gap = float(task["reference_gap"])
    gap_norm = torch.full((n, 1), gap / 16.0, dtype=base_features.dtype, device=base_features.device)
    log_gap_norm = torch.full((n, 1), math.log2(max(gap, 1.0)) / 4.0, dtype=base_features.dtype, device=base_features.device)
    time_from_mid = torch.full((n, 1), abs(float(task["normalized_time"]) - 0.5), dtype=base_features.dtype, device=base_features.device)
    endpoint_gap_interaction = endpoint_l2_norm * gap_norm
    endpoint_rank = rank_percentile(endpoint_l2)
    return torch.cat([
        base_features,
        endpoint_l2_norm,
        endpoint_l1_norm,
        endpoint_max_norm,
        gap_norm,
        log_gap_norm,
        time_from_mid,
        endpoint_gap_interaction,
        endpoint_rank,
    ], dim=-1)


def normalized_energy_target(energy):
    mean_energy = energy.mean().clamp_min(1e-12)
    target = torch.log1p(energy / mean_energy)
    return target / target.amax().clamp_min(1e-12)


def collect_training_examples(tasks, adapter, args, device):
    cache = {}
    xs = []
    ys = []
    energy_targets = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    method_ids = {"linear": 0.0, "stage65_adapter": 1.0}
    with torch.no_grad():
        for task in tasks:
            left_attrs, right_attrs, target_attrs, base_attrs_by_method = build_task_tensors(task, adapter, device, cache)
            for method in args.base_methods:
                base_attrs = base_attrs_by_method[method]
                residual = target_attrs - base_attrs
                energy = torch.sum(residual[0] ** 2, dim=-1)
                keep_count = max(1, int(round(energy.numel() * args.keep_fraction)))
                teacher_idx = torch.topk(energy, k=keep_count, largest=True).indices
                labels = torch.zeros_like(energy)
                labels[teacher_idx] = 1.0
                energy_target = normalized_energy_target(energy)
                features = make_full_features(left_attrs, right_attrs, base_attrs, task, method_ids[method])
                sample_count = min(args.train_gaussians_per_task, features.shape[0])
                sample_idx = torch.randperm(features.shape[0], generator=generator, device=device)[:sample_count]
                xs.append(features[sample_idx].detach().cpu())
                ys.append(labels[sample_idx].detach().cpu())
                energy_targets.append(energy_target[sample_idx].detach().cpu())
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0), torch.cat(energy_targets, dim=0)


def objective_loss(logits, labels, energy_targets, pos_weight, objective):
    if objective == "topk_bce":
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    if objective == "energy_regression":
        return F.mse_loss(torch.sigmoid(logits), energy_targets)
    raise ValueError(f"Unknown objective: {objective}")


def train_selector(train_x, train_y, train_energy, feature_mode, objective, args, device, model_index):
    dim = feature_dim(feature_mode)
    model_name = f"{feature_mode}__{objective}"
    torch.manual_seed(args.seed + 100 + model_index)
    selected_x = train_x[:, :dim].to(device)
    mean = selected_x.mean(dim=0, keepdim=True)
    std = selected_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    selected_x = ((selected_x - mean) / std).detach()
    train_y = train_y.to(device)
    train_energy = train_energy.to(device)
    predictor = ImportanceMLP(dim, args.hidden_dim).to(device)
    pos = train_y.sum().clamp_min(1.0)
    neg = (train_y.numel() - train_y.sum()).clamp_min(1.0)
    pos_weight = neg / pos
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000 + model_index)
    logs = []
    n = selected_x.shape[0]
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), generator=generator, device=device)
        batch_x = selected_x[idx]
        batch_y = train_y[idx]
        batch_energy = train_energy[idx]
        logits = predictor(batch_x)
        loss = objective_loss(logits, batch_y, batch_energy, pos_weight, objective)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            logs.append({
                "model_name": model_name,
                "feature_mode": feature_mode,
                "objective": objective,
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "batch_positive_rate": float(batch_y.mean().detach().cpu().item()),
                "batch_energy_target_mean": float(batch_energy.mean().detach().cpu().item()),
            })
    predictor.eval()
    return {
        "model_name": model_name,
        "feature_mode": feature_mode,
        "objective": objective,
        "feature_dim": dim,
        "predictor": predictor,
        "mean": mean,
        "std": std,
    }, logs


def evaluate(tasks, adapter, models, args, device):
    cache = {}
    rows = []
    method_ids = {"linear": 0.0, "stage65_adapter": 1.0}
    with torch.no_grad():
        for task in tasks:
            left_attrs, right_attrs, target_attrs, base_attrs_by_method = build_task_tensors(task, adapter, device, cache)
            endpoint_score = torch.sum((right_attrs[0] - left_attrs[0]) ** 2, dim=-1)
            for method in args.base_methods:
                base_attrs = base_attrs_by_method[method]
                residual = target_attrs - base_attrs
                energy = torch.sum(residual[0] ** 2, dim=-1)
                keep_count = max(1, int(round(energy.numel() * args.keep_fraction)))
                full_features = make_full_features(left_attrs, right_attrs, base_attrs, task, method_ids[method])
                candidate_scores = [("endpoint_only", "endpoint_diff_baseline", "endpoint_diff_baseline", endpoint_score)]
                for model in models:
                    logits = []
                    dim = model["feature_dim"]
                    for start in range(0, full_features.shape[0], args.eval_batch_size):
                        chunk = full_features[start:start + args.eval_batch_size, :dim]
                        logits.append(model["predictor"]((chunk - model["mean"]) / model["std"]))
                    candidate_scores.append((model["feature_mode"], model["objective"], "mlp_selector", torch.cat(logits, dim=0)))
                for feature_mode, objective, candidate, scores in candidate_scores:
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
                        "feature_mode": feature_mode,
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
        grouped[(row["feature_mode"], row["objective"], row["candidate"], row["base_method"], row["reference_gap"])].append(row)
    out = []
    for (feature_mode, objective, candidate, method, gap), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][3], int(item[0][4]))):
        out.append({
            "feature_mode": feature_mode,
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


def best_learned_by_group(summary_rows):
    best = {}
    for row in summary_rows:
        if row["candidate"] != "mlp_selector":
            continue
        key = (row["base_method"], row["reference_gap"])
        if key not in best or row["mean_energy_recall_total"] > best[key]["mean_energy_recall_total"]:
            best[key] = row
    return [best[key] for key in sorted(best, key=lambda item: (item[0], int(item[1])))]


def write_report(summary, summary_rows, path):
    best_rows = best_learned_by_group(summary_rows)
    lines = [
        "# Stage101 Enhanced Selector Feature Sweep",
        "",
        "## Configuration",
        "",
        f"- train tasks: `{summary['train_task_count']}`",
        f"- eval tasks: `{summary['eval_task_count']}`",
        f"- train examples: `{summary['train_example_count']}`",
        f"- full feature dim: `{summary['full_feature_dim']}`",
        f"- keep fraction: `{summary['keep_fraction']}`",
        f"- feature modes: `{', '.join(summary['feature_modes'])}`",
        f"- objectives: `{', '.join(summary['objectives'])}`",
        "- no rendering, checkpoint, or heavy tensor output",
        "",
        "## Summary",
        "",
        "| feature mode | objective | candidate | base | gap | tasks | precision@keep | energy recall | oracle recall | relative recall |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['feature_mode']} | {row['objective']} | {row['candidate']} | {row['base_method']} | {row['reference_gap']} | {row['task_count']} | {row['mean_precision_at_keep']} | {row['mean_energy_recall_total']} | {row['mean_oracle_energy_recall_total']} | {row['mean_relative_energy_recall_vs_oracle']} |"
        )
    lines.extend([
        "",
        "## Best Learned Feature By Group",
        "",
        "| base | gap | feature mode | objective | energy recall | relative recall | precision@keep |",
        "|---|---:|---|---|---:|---:|---:|",
    ])
    for row in best_rows:
        lines.append(
            f"| {row['base_method']} | {row['reference_gap']} | {row['feature_mode']} | {row['objective']} | {row['mean_energy_recall_total']} | {row['mean_relative_energy_recall_vs_oracle']} | {row['mean_precision_at_keep']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Extra features are decoder-available only: reference gap, endpoint motion norms, and endpoint rank.",
        "- Target dense anchors are used only for offline labels/metrics.",
        "- This stage tests feature quality before any residual-value prediction or rendered validation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_csv", type=Path, default=DEFAULT_STAGE97_TASKS)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--base_methods", nargs="+", default=["linear", "stage65_adapter"])
    parser.add_argument("--feature_modes", nargs="+", default=["stage100_base", "gap_endpoint_norms", "gap_endpoint_rank"])
    parser.add_argument("--objectives", nargs="+", default=["topk_bce", "energy_regression"])
    parser.add_argument("--max_train_tasks", type=int, default=96)
    parser.add_argument("--max_eval_tasks", type=int, default=60)
    parser.add_argument("--train_gaussians_per_task", type=int, default=3072)
    parser.add_argument("--keep_fraction", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--train_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8192)
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
    for mode in args.feature_modes:
        feature_dim(mode)
    adapter = load_adapter(args.adapter, hidden_dim=256, device=device)
    train_tasks = parse_rows(args.tasks_csv, "train", args.gaps, args.max_train_tasks, args.seed)
    eval_tasks = parse_rows(args.tasks_csv, "eval", args.gaps, args.max_eval_tasks, args.seed + 1)
    train_x, train_y, train_energy = collect_training_examples(train_tasks, adapter, args, device)

    models = []
    train_logs = []
    model_index = 0
    for feature_mode in args.feature_modes:
        for objective in args.objectives:
            model, logs = train_selector(train_x, train_y, train_energy, feature_mode, objective, args, device, model_index)
            models.append(model)
            train_logs.extend(logs)
            model_index += 1
    eval_rows = evaluate(eval_tasks, adapter, models, args, device)
    summary_rows = summarize(eval_rows)

    rows_csv = args.summary_root / "stage101_enhanced_selector_feature_rows.csv"
    summary_csv = args.summary_root / "stage101_enhanced_selector_feature_summary.csv"
    train_log_csv = args.summary_root / "stage101_enhanced_selector_feature_train_log.csv"
    summary_json = args.summary_root / "stage101_enhanced_selector_feature_summary.json"
    report_md = args.summary_root / "stage101_enhanced_selector_feature_report.md"
    write_csv(eval_rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(train_logs, train_log_csv, TRAIN_LOG_FIELDS)
    summary = {
        "stage": 101,
        "mode": "enhanced residual selector feature sweep",
        "tasks_csv": str(args.tasks_csv),
        "adapter": str(args.adapter),
        "gaps": args.gaps,
        "base_methods": args.base_methods,
        "feature_modes": args.feature_modes,
        "objectives": args.objectives,
        "train_task_count": len(train_tasks),
        "eval_task_count": len(eval_tasks),
        "train_example_count": int(train_x.shape[0]),
        "full_feature_dim": int(train_x.shape[1]),
        "positive_rate": float(train_y.mean().item()),
        "energy_target_mean": float(train_energy.mean().item()),
        "keep_fraction": args.keep_fraction,
        "train_steps": args.train_steps,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "train_log_csv": str(train_log_csv),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "best_learned_rows": best_learned_by_group(summary_rows),
        "notes": [
            "No model checkpoint or heavy tensor is saved.",
            "Extra features are decoder-available only: reference gap, endpoint motion norms, and endpoint rank.",
            "Teacher labels and energy targets are generated from target dense anchors for training/evaluation only.",
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
        "full_feature_dim": int(train_x.shape[1]),
    }, indent=2))


if __name__ == "__main__":
    main()
