import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE97_TASKS = REPO_ROOT / "experiments/stage97_residual_predictor_task_manifest/stage97_residual_predictor_tasks.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage98_residual_importance_predictor_smoke"


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import (  # noqa: E402
    DEFAULT_ADAPTER,
    linear_anchor,
    load_adapter,
    load_anchor,
)


TRAIN_LOG_FIELDS = [
    "step",
    "loss",
    "batch_positive_rate",
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
    "candidate",
    "keep_fraction",
    "keep_gaussians",
    "precision_at_keep",
    "energy_recall_total",
    "oracle_energy_recall_total",
    "relative_energy_recall_vs_oracle",
    "random_expected_precision",
]

SUMMARY_FIELDS = [
    "candidate",
    "base_method",
    "reference_gap",
    "task_count",
    "mean_precision_at_keep",
    "mean_energy_recall_total",
    "mean_oracle_energy_recall_total",
    "mean_relative_energy_recall_vs_oracle",
]


class ImportanceMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_rows(path, task_split, gaps, max_tasks, seed):
    gap_set = set(gaps)
    rows = []
    for row in read_csv(path):
        if row["task_split"] != task_split:
            continue
        gap = int(row["reference_gap"])
        if gap_set and gap not in gap_set:
            continue
        item = dict(row)
        item["bits"] = int(item["bits"])
        item["reference_gap"] = gap
        item["left_index"] = int(item["left_index"])
        item["right_index"] = int(item["right_index"])
        item["target_index"] = int(item["target_index"])
        item["normalized_time"] = float(item["normalized_time"])
        rows.append(item)
    if max_tasks <= 0 or len(rows) <= max_tasks:
        return rows
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["sequence"], row["reference_gap"])].append(row)
    for items in groups.values():
        rng.shuffle(items)
    keys = sorted(groups)
    rng.shuffle(keys)
    selected = []
    offset = 0
    while len(selected) < max_tasks:
        progressed = False
        for key in keys:
            items = groups[key]
            if offset >= len(items):
                continue
            selected.append(items[offset])
            progressed = True
            if len(selected) >= max_tasks:
                break
        if not progressed:
            break
        offset += 1
    return selected


def make_features(left_attrs, right_attrs, base_attrs, normalized_time, method_id):
    left = left_attrs[0]
    right = right_attrs[0]
    base = base_attrs[0]
    diff = right - left
    abs_diff = torch.abs(diff)
    n = base.shape[0]
    t = torch.full((n, 1), float(normalized_time), dtype=base.dtype, device=base.device)
    method = torch.full((n, 1), float(method_id), dtype=base.dtype, device=base.device)
    return torch.cat([base, left, right, diff, abs_diff, t, method], dim=-1)


def topk_metrics(scores, residual_energy, keep_count):
    total_energy = torch.sum(residual_energy).clamp_min(1e-12)
    teacher_idx = torch.topk(residual_energy, k=keep_count, largest=True).indices
    pred_idx = torch.topk(scores, k=keep_count, largest=True).indices
    oracle_energy = torch.sum(residual_energy[teacher_idx]) / total_energy
    pred_energy = torch.sum(residual_energy[pred_idx]) / total_energy
    teacher_mask = torch.zeros_like(residual_energy, dtype=torch.bool)
    teacher_mask[teacher_idx] = True
    precision = torch.mean(teacher_mask[pred_idx].float())
    relative = pred_energy / oracle_energy.clamp_min(1e-12)
    return {
        "precision_at_keep": float(precision.item()),
        "energy_recall_total": float(pred_energy.item()),
        "oracle_energy_recall_total": float(oracle_energy.item()),
        "relative_energy_recall_vs_oracle": float(relative.item()),
    }


def build_task_tensors(task, model, device, cache):
    left = load_anchor(task["left_anchor_source_item"], task["left_anchor_source_side"], device, bits=task["bits"], cache=cache)
    right = load_anchor(task["right_anchor_source_item"], task["right_anchor_source_side"], device, bits=task["bits"], cache=cache)
    target = load_anchor(task["target_anchor_source_item"], task["target_anchor_source_side"], device, bits=None, cache=cache)
    left_attrs = flatten_static_anchor(left)
    right_attrs = flatten_static_anchor(right)
    target_attrs = flatten_static_anchor(target)
    t = torch.tensor([task["normalized_time"]], dtype=torch.float32, device=device)
    base_attrs = {
        "linear": flatten_static_anchor(linear_anchor(left, right, task["normalized_time"])),
        "stage65_adapter": flatten_static_anchor(model(left, right, t, apply_output_constraints=False)),
    }
    return left_attrs, right_attrs, target_attrs, base_attrs


def collect_training_examples(tasks, adapter, args, device):
    cache = {}
    xs = []
    ys = []
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
                features = make_features(left_attrs, right_attrs, base_attrs, task["normalized_time"], method_ids[method])
                sample_count = min(args.train_gaussians_per_task, features.shape[0])
                sample_idx = torch.randperm(features.shape[0], generator=generator, device=device)[:sample_count]
                xs.append(features[sample_idx].detach().cpu())
                ys.append(labels[sample_idx].detach().cpu())
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def train_predictor(train_x, train_y, args, device):
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    predictor = ImportanceMLP(train_x.shape[1], args.hidden_dim).to(device)
    pos = train_y.sum().clamp_min(1.0)
    neg = (train_y.numel() - train_y.sum()).clamp_min(1.0)
    pos_weight = neg / pos
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device=device).manual_seed(args.seed + 17)
    logs = []
    n = train_x.shape[0]
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), generator=generator, device=device)
        batch_x = train_x[idx]
        batch_y = train_y[idx]
        logits = predictor(batch_x)
        loss = F.binary_cross_entropy_with_logits(logits, batch_y, pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            logs.append({
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "batch_positive_rate": float(batch_y.mean().detach().cpu().item()),
            })
    return predictor, mean, std, logs


def evaluate(tasks, adapter, predictor, mean, std, args, device):
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
                features = make_features(left_attrs, right_attrs, base_attrs, task["normalized_time"], method_ids[method])
                logits = []
                for start in range(0, features.shape[0], args.eval_batch_size):
                    chunk = features[start:start + args.eval_batch_size]
                    logits.append(predictor((chunk - mean) / std))
                mlp_score = torch.cat(logits, dim=0)
                for candidate, scores in [
                    ("mlp_importance", mlp_score),
                    ("endpoint_diff_baseline", endpoint_score),
                ]:
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
                        "candidate": candidate,
                        "keep_fraction": args.keep_fraction,
                        "keep_gaussians": keep_count,
                        "precision_at_keep": metrics["precision_at_keep"],
                        "energy_recall_total": metrics["energy_recall_total"],
                        "oracle_energy_recall_total": metrics["oracle_energy_recall_total"],
                        "relative_energy_recall_vs_oracle": metrics["relative_energy_recall_vs_oracle"],
                        "random_expected_precision": args.keep_fraction,
                    })
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["candidate"], row["base_method"], row["reference_gap"])].append(row)
    out = []
    for (candidate, method, gap), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], int(item[0][2]))):
        out.append({
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


def write_report(summary, summary_rows, path):
    lines = [
        "# Stage98 Residual Importance Predictor Smoke",
        "",
        "## Configuration",
        "",
        f"- train tasks: `{summary['train_task_count']}`",
        f"- eval tasks: `{summary['eval_task_count']}`",
        f"- train examples: `{summary['train_example_count']}`",
        f"- keep fraction: `{summary['keep_fraction']}`",
        f"- train steps: `{summary['train_steps']}`",
        "- labels are teacher top10 residual-energy masks generated from target dense anchors",
        "- predictor inputs are decoder-available anchor features plus time and base method id",
        "",
        "## Summary",
        "",
        "| candidate | base | gap | tasks | precision@keep | energy recall | oracle recall | relative recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['candidate']} | {row['base_method']} | {row['reference_gap']} | {row['task_count']} | {row['mean_precision_at_keep']} | {row['mean_energy_recall_total']} | {row['mean_oracle_energy_recall_total']} | {row['mean_relative_energy_recall_vs_oracle']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- This stage does not render or transmit predicted residual values.",
        "- It measures whether a feed-forward predictor can localize high-energy residual Gaussians.",
        "- Target dense anchors are used only for offline labels and metrics, not as predictor inputs.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_csv", type=Path, default=DEFAULT_STAGE97_TASKS)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--base_methods", nargs="+", default=["linear", "stage65_adapter"])
    parser.add_argument("--max_train_tasks", type=int, default=24)
    parser.add_argument("--max_eval_tasks", type=int, default=12)
    parser.add_argument("--train_gaussians_per_task", type=int, default=4096)
    parser.add_argument("--keep_fraction", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--train_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--eval_batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=25)
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
    train_x, train_y = collect_training_examples(train_tasks, adapter, args, device)
    predictor, mean, std, train_logs = train_predictor(train_x, train_y, args, device)
    eval_rows = evaluate(eval_tasks, adapter, predictor, mean, std, args, device)
    summary_rows = summarize(eval_rows)

    rows_csv = args.summary_root / "stage98_residual_importance_predictor_rows.csv"
    summary_csv = args.summary_root / "stage98_residual_importance_predictor_summary.csv"
    train_log_csv = args.summary_root / "stage98_residual_importance_predictor_train_log.csv"
    summary_json = args.summary_root / "stage98_residual_importance_predictor_summary.json"
    report_md = args.summary_root / "stage98_residual_importance_predictor_report.md"
    write_csv(eval_rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(train_logs, train_log_csv, TRAIN_LOG_FIELDS)
    summary = {
        "stage": 98,
        "mode": "residual importance predictor smoke",
        "tasks_csv": str(args.tasks_csv),
        "adapter": str(args.adapter),
        "gaps": args.gaps,
        "base_methods": args.base_methods,
        "train_task_count": len(train_tasks),
        "eval_task_count": len(eval_tasks),
        "train_example_count": int(train_x.shape[0]),
        "feature_dim": int(train_x.shape[1]),
        "positive_rate": float(train_y.mean().item()),
        "keep_fraction": args.keep_fraction,
        "train_steps": args.train_steps,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "train_log_csv": str(train_log_csv),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "notes": [
            "No model checkpoint or heavy tensor is saved.",
            "Teacher labels are generated from target dense anchors for training/evaluation only.",
            "Predictor inputs are decoder-available anchor features, time, and base method id.",
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
    raise SystemExit(main())
