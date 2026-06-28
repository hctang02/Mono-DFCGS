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
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage108_anchor_stat_task_switch_predictor_preflight"


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import (  # noqa: E402
    DEFAULT_ADAPTER,
    linear_anchor,
    load_adapter,
    load_anchor,
)
from scripts.run_stage107_metadata_task_switch_predictor_preflight import (  # noqa: E402
    DEFAULT_STAGE97_TASKS,
    DEFAULT_STAGE103_ROWS,
    DEFAULT_STAGE106_POLICY,
    DEPLOYABLE_CANDIDATES,
    GROUP_SUMMARY_FIELDS,
    ROW_FIELDS,
    SUMMARY_FIELDS,
    TRAIN_LOG_FIELDS,
    build_train_group_policy,
    best_candidate_by_mean,
    evaluate_selection,
    load_candidate_tasks,
    load_manifest,
    make_features as make_metadata_features,
    predict_metadata_mlp,
    read_csv,
    select_stage106_policy,
    select_train_group_policy,
    split_folds,
    summarize,
    summarize_groups,
    train_metadata_mlp,
    write_csv,
)


class SwitchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def tensor_stats(values):
    flat = values.reshape(-1).float()
    if flat.numel() <= 0:
        return [0.0] * 6
    top_count = max(1, int(round(flat.numel() * 0.1)))
    top_mean = torch.topk(flat, k=top_count, largest=True).values.mean()
    return [
        float(flat.mean().detach().cpu().item()),
        float(flat.std(unbiased=False).detach().cpu().item()),
        float(flat.amax().detach().cpu().item()),
        float(torch.quantile(flat, 0.90).detach().cpu().item()),
        float(torch.quantile(flat, 0.99).detach().cpu().item()),
        float(top_mean.detach().cpu().item()),
    ]


def group_abs_means(diff):
    groups = [(0, 3), (3, 4), (4, 7), (7, 10), (10, 13)]
    return [float(torch.mean(torch.abs(diff[:, start:end])).detach().cpu().item()) for start, end in groups]


def build_anchor_features(task, manifest, adapter, device, cache):
    row = manifest[task["stage97_task_id"]]
    bits = int(row["bits"])
    left = load_anchor(row["left_anchor_source_item"], row["left_anchor_source_side"], device, bits=bits, cache=cache)
    right = load_anchor(row["right_anchor_source_item"], row["right_anchor_source_side"], device, bits=bits, cache=cache)
    left_attrs = flatten_static_anchor(left)[0]
    right_attrs = flatten_static_anchor(right)[0]
    if task["base_method"] == "linear":
        base = linear_anchor(left, right, task["normalized_time"])
    else:
        t = torch.tensor([task["normalized_time"]], dtype=torch.float32, device=device)
        base = adapter(left, right, t, apply_output_constraints=False)
    base_attrs = flatten_static_anchor(base)[0]
    endpoint_diff = right_attrs - left_attrs
    endpoint_energy = torch.sum(endpoint_diff ** 2, dim=-1)
    endpoint_abs_mean = torch.mean(torch.abs(endpoint_diff), dim=-1)
    base_left = base_attrs - left_attrs
    base_right = right_attrs - base_attrs
    base_left_energy = torch.sum(base_left ** 2, dim=-1)
    base_right_energy = torch.sum(base_right ** 2, dim=-1)
    base_endpoint_balance = torch.abs(base_left_energy - base_right_energy)
    base_attr_abs = torch.mean(torch.abs(base_attrs), dim=-1)
    features = []
    for values in [endpoint_energy, endpoint_abs_mean, base_left_energy, base_right_energy, base_endpoint_balance, base_attr_abs]:
        features.extend(tensor_stats(values))
    features.extend(group_abs_means(endpoint_diff))
    features.extend(group_abs_means(base_left))
    features.extend(group_abs_means(base_right))
    return features


def build_feature_table(tasks, manifest, adapter, device):
    cache = {}
    feature_table = {}
    with torch.no_grad():
        for task in tasks:
            key = (task["stage97_task_id"], task["base_method"])
            feature_table[key] = make_metadata_features(task) + build_anchor_features(task, manifest, adapter, device, cache)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return feature_table


def train_mlp(train_tasks, feature_table, args, fold):
    x = torch.tensor([feature_table[(task["stage97_task_id"], task["base_method"])] for task in train_tasks], dtype=torch.float32)
    y = torch.tensor([DEPLOYABLE_CANDIDATES.index(task["oracle_candidate"]) for task in train_tasks], dtype=torch.long)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_norm = (x - mean) / std
    torch.manual_seed(args.seed + 100 + fold)
    model = SwitchMLP(x.shape[1], args.hidden_dim, len(DEPLOYABLE_CANDIDATES))
    counts = torch.bincount(y, minlength=len(DEPLOYABLE_CANDIDATES)).float().clamp_min(1.0)
    weights = (counts.sum() / counts).clamp(max=10.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logs = []
    for step in range(1, args.train_steps + 1):
        logits = model(x_norm)
        loss = F.cross_entropy(logits, y, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            pred = torch.argmax(model(x_norm), dim=-1)
            logs.append({
                "fold": fold,
                "step": step,
                "loss": float(loss.detach().item()),
                "train_accuracy": float((pred == y).float().mean().item()),
            })
    return model.eval(), mean, std, logs


def predict_mlp(task, feature_table, model_bundle):
    model, mean, std = model_bundle
    x = torch.tensor([feature_table[(task["stage97_task_id"], task["base_method"])]], dtype=torch.float32)
    with torch.no_grad():
        pred = torch.argmax(model((x - mean) / std), dim=-1).item()
    return DEPLOYABLE_CANDIDATES[pred]


def run_cv(tasks, feature_table, args, stage106_table, fallback):
    fold_map = split_folds(tasks, args.fold_count)
    rows = []
    train_logs = []
    metadata_logs = []
    for fold in range(args.fold_count):
        train_tasks = [task for task in tasks if fold_map[task["stage97_task_id"]] != fold]
        test_tasks = [task for task in tasks if fold_map[task["stage97_task_id"]] == fold]
        global_best = best_candidate_by_mean(train_tasks, DEPLOYABLE_CANDIDATES)
        train_group_policy = build_train_group_policy(train_tasks, global_best)
        metadata_model, metadata_mean, metadata_std, metadata_fold_logs = train_metadata_mlp(train_tasks, args, fold)
        anchor_model, anchor_mean, anchor_std, anchor_fold_logs = train_mlp(train_tasks, feature_table, args, fold)
        metadata_logs.extend({**row, "model": "metadata_mlp_cv"} for row in metadata_fold_logs)
        train_logs.extend({**row, "model": "anchor_stat_mlp_cv"} for row in anchor_fold_logs)
        for task in test_tasks:
            policy_candidates = {
                "endpoint_only": "endpoint_diff_baseline",
                "train_fold_group_policy": select_train_group_policy(task, train_group_policy),
                "stage106_fixed_group_policy": select_stage106_policy(task, stage106_table, fallback),
                "metadata_mlp_cv": predict_metadata_mlp(task, (metadata_model, metadata_mean, metadata_std)),
                "anchor_stat_mlp_cv": predict_mlp(task, feature_table, (anchor_model, anchor_mean, anchor_std)),
                "oracle_task_best": task["oracle_candidate"],
            }
            for policy, selected_candidate in policy_candidates.items():
                rows.append(evaluate_selection(task, policy, fold, selected_candidate))
    return rows, metadata_logs + train_logs


def write_report(summary, summary_rows, group_summary_rows, path):
    lines = [
        "# Stage108 Anchor-Stat Task-Level Switch Predictor Preflight",
        "",
        "## Configuration",
        "",
        f"- task rows: `{summary['task_count']}`",
        f"- folds: `{summary['fold_count']}`",
        f"- feature dim: `{summary['feature_dim']}`",
        "- features are decoder-side metadata plus left/right/base anchor aggregate statistics",
        "- no target dense anchor, target residual, rendering, checkpoint, or heavy tensor output",
        "",
        "## Overall Summary",
        "",
        "| policy | tasks | selected PSNR | endpoint PSNR | gain vs endpoint | oracle task PSNR | gap to oracle task | accuracy | selections |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['policy']} | {row['task_count']} | {row['mean_selected_sideinfo_psnr']} | {row['mean_endpoint_sideinfo_psnr']} | {row['mean_delta_psnr_vs_endpoint']} | {row['mean_oracle_task_best_sideinfo_psnr']} | {row['mean_gap_to_oracle_task_best']} | {row['selection_accuracy']} | {row['selected_candidate_counts']} |"
        )
    lines.extend([
        "",
        "## Group Summary",
        "",
        "| policy | base | gap | tasks | selected PSNR | gain vs endpoint | accuracy | selections |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in group_summary_rows:
        lines.append(
            f"| {row['policy']} | {row['base_method']} | {row['reference_gap']} | {row['task_count']} | {row['mean_selected_sideinfo_psnr']} | {row['mean_delta_psnr_vs_endpoint']} | {row['selection_accuracy']} | {row['selected_candidate_counts']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Anchor-stat features are decoder-available because they use only transmitted left/right anchors and predicted base anchors.",
        "- Rendered PSNR labels are used only for offline training/evaluation.",
        "- If anchor-stat MLP does not beat Stage106, switch learning should wait for more data or selector-score features.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage97_tasks", type=Path, default=DEFAULT_STAGE97_TASKS)
    parser.add_argument("--stage103_rows", type=Path, default=DEFAULT_STAGE103_ROWS)
    parser.add_argument("--stage106_policy", type=Path, default=DEFAULT_STAGE106_POLICY)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    manifest = load_manifest(args.stage97_tasks)
    tasks = load_candidate_tasks(read_csv(args.stage103_rows), manifest)
    adapter = load_adapter(args.adapter, hidden_dim=256, device=device)
    feature_table = build_feature_table(tasks, manifest, adapter, device)
    package = json.loads(args.stage106_policy.read_text(encoding="utf-8"))
    rows, train_logs = run_cv(tasks, feature_table, args, package["selection_table"], package["fallback_candidate"])
    summary_rows = summarize(rows)
    group_summary_rows = summarize_groups(rows)

    rows_csv = args.summary_root / "stage108_anchor_stat_switch_rows.csv"
    summary_csv = args.summary_root / "stage108_anchor_stat_switch_summary.csv"
    group_summary_csv = args.summary_root / "stage108_anchor_stat_switch_group_summary.csv"
    train_log_csv = args.summary_root / "stage108_anchor_stat_switch_train_log.csv"
    summary_json = args.summary_root / "stage108_anchor_stat_switch_summary.json"
    report_md = args.summary_root / "stage108_anchor_stat_switch_report.md"
    train_log_fields = ["model", *TRAIN_LOG_FIELDS]
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(group_summary_rows, group_summary_csv, GROUP_SUMMARY_FIELDS)
    write_csv(train_logs, train_log_csv, train_log_fields)
    feature_dim = len(next(iter(feature_table.values()))) if feature_table else 0
    summary = {
        "stage": 108,
        "mode": "anchor-stat task-level switch predictor preflight",
        "task_count": len(tasks),
        "fold_count": args.fold_count,
        "feature_dim": feature_dim,
        "stage97_tasks": str(args.stage97_tasks),
        "stage103_rows": str(args.stage103_rows),
        "stage106_policy": str(args.stage106_policy),
        "adapter": str(args.adapter),
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "group_summary_csv": str(group_summary_csv),
        "train_log_csv": str(train_log_csv),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "notes": [
            "Feature construction loads only left/right q12 anchors and predicted base anchors.",
            "Target dense anchors and target residuals are not loaded for features.",
            "Rendered PSNR labels are used only for offline training/evaluation.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, summary_rows, group_summary_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "task_count": len(tasks),
        "feature_dim": feature_dim,
    }, indent=2))


if __name__ == "__main__":
    main()
