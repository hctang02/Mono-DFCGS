import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE97_TASKS = REPO_ROOT / "experiments/stage97_residual_predictor_task_manifest/stage97_residual_predictor_tasks.csv"
DEFAULT_STAGE103_ROWS = REPO_ROOT / "experiments/stage103_broader_rendered_selector_validation/stage103_broader_rendered_selector_rows.csv"
DEFAULT_STAGE106_POLICY = REPO_ROOT / "experiments/stage106_render_aware_group_policy_package/stage106_render_aware_group_policy.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage107_metadata_task_switch_predictor_preflight"
DEPLOYABLE_CANDIDATES = ["endpoint_diff_baseline", "shared_energy_regression", "shared_topk_bce"]


ROW_FIELDS = [
    "fold",
    "policy",
    "stage97_task_id",
    "source_task_id",
    "sequence",
    "base_method",
    "reference_gap",
    "target_index",
    "oracle_candidate",
    "selected_candidate",
    "selected_sideinfo_psnr",
    "endpoint_sideinfo_psnr",
    "oracle_task_best_sideinfo_psnr",
    "teacher_oracle_sideinfo_psnr",
    "delta_psnr_vs_endpoint",
    "gap_to_oracle_task_best",
    "gap_to_teacher_oracle",
    "selection_correct",
]

SUMMARY_FIELDS = [
    "policy",
    "task_count",
    "mean_selected_sideinfo_psnr",
    "mean_endpoint_sideinfo_psnr",
    "mean_oracle_task_best_sideinfo_psnr",
    "mean_teacher_oracle_sideinfo_psnr",
    "mean_delta_psnr_vs_endpoint",
    "mean_gap_to_oracle_task_best",
    "mean_gap_to_teacher_oracle",
    "selection_accuracy",
    "selected_candidate_counts",
]

GROUP_SUMMARY_FIELDS = [
    "policy",
    "base_method",
    "reference_gap",
    "task_count",
    "mean_selected_sideinfo_psnr",
    "mean_endpoint_sideinfo_psnr",
    "mean_delta_psnr_vs_endpoint",
    "selection_accuracy",
    "selected_candidate_counts",
]

TRAIN_LOG_FIELDS = ["fold", "step", "loss", "train_accuracy"]


class MetadataSwitchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def f(row, key):
    return float(row[key])


def task_key(row):
    return (
        row["stage97_task_id"],
        row["source_task_id"],
        row["base_method"],
        row["reference_gap"],
        row["target_index"],
    )


def load_manifest(path):
    out = {}
    for row in read_csv(path):
        out[row["stage97_task_id"]] = row
    return out


def load_candidate_tasks(stage103_rows, manifest):
    grouped = defaultdict(dict)
    for row in stage103_rows:
        grouped[task_key(row)][row["candidate"]] = row
    tasks = []
    for _, candidates in sorted(grouped.items()):
        if not all(candidate in candidates for candidate in DEPLOYABLE_CANDIDATES):
            continue
        any_row = candidates["endpoint_diff_baseline"]
        task_meta = manifest[any_row["stage97_task_id"]]
        oracle_candidate = max(DEPLOYABLE_CANDIDATES, key=lambda candidate: f(candidates[candidate], "sideinfo_psnr"))
        tasks.append({
            "stage97_task_id": any_row["stage97_task_id"],
            "source_task_id": any_row["source_task_id"],
            "sequence": any_row["sequence"],
            "base_method": any_row["base_method"],
            "reference_gap": int(any_row["reference_gap"]),
            "left_index": int(task_meta["left_index"]),
            "right_index": int(task_meta["right_index"]),
            "target_index": int(any_row["target_index"]),
            "normalized_time": float(task_meta["normalized_time"]),
            "candidates": candidates,
            "oracle_candidate": oracle_candidate,
        })
    return tasks


def make_features(task):
    gap = int(task["reference_gap"])
    method_id = 0.0 if task["base_method"] == "linear" else 1.0
    left = float(task["left_index"])
    right = float(task["right_index"])
    target = float(task["target_index"])
    span = max(1.0, right - left)
    t = float(task["normalized_time"])
    return [
        method_id,
        1.0 - method_id,
        gap / 16.0,
        1.0 if gap == 4 else 0.0,
        1.0 if gap == 8 else 0.0,
        1.0 if gap == 16 else 0.0,
        t,
        abs(t - 0.5),
        left / 100.0,
        right / 100.0,
        target / 100.0,
        (target - left) / span,
        (right - target) / span,
    ]


def split_folds(tasks, fold_count):
    ids = sorted({task["stage97_task_id"] for task in tasks})
    return {task_id: idx % fold_count for idx, task_id in enumerate(ids)}


def best_candidate_by_mean(tasks, candidates):
    means = {}
    for candidate in candidates:
        means[candidate] = sum(f(task["candidates"][candidate], "sideinfo_psnr") for task in tasks) / len(tasks)
    return max(candidates, key=lambda candidate: means[candidate])


def build_train_group_policy(train_tasks, fallback):
    grouped = defaultdict(list)
    for task in train_tasks:
        grouped[(task["base_method"], task["reference_gap"])].append(task)
    table = {}
    for key, items in grouped.items():
        table[key] = best_candidate_by_mean(items, DEPLOYABLE_CANDIDATES)
    return table, fallback


def select_train_group_policy(task, policy):
    table, fallback = policy
    return table.get((task["base_method"], task["reference_gap"]), fallback)


def select_stage106_policy(task, selection_table, fallback):
    return selection_table.get(task["base_method"], {}).get(str(task["reference_gap"]), fallback)


def train_metadata_mlp(train_tasks, args, fold):
    x = torch.tensor([make_features(task) for task in train_tasks], dtype=torch.float32)
    y = torch.tensor([DEPLOYABLE_CANDIDATES.index(task["oracle_candidate"]) for task in train_tasks], dtype=torch.long)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_norm = (x - mean) / std
    torch.manual_seed(args.seed + fold)
    model = MetadataSwitchMLP(x.shape[1], args.hidden_dim, len(DEPLOYABLE_CANDIDATES))
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


def predict_metadata_mlp(task, model_bundle):
    model, mean, std = model_bundle
    x = torch.tensor([make_features(task)], dtype=torch.float32)
    with torch.no_grad():
        pred = torch.argmax(model((x - mean) / std), dim=-1).item()
    return DEPLOYABLE_CANDIDATES[pred]


def evaluate_selection(task, policy, fold, selected_candidate):
    selected = task["candidates"][selected_candidate]
    endpoint = task["candidates"]["endpoint_diff_baseline"]
    oracle = task["candidates"][task["oracle_candidate"]]
    return {
        "fold": fold,
        "policy": policy,
        "stage97_task_id": task["stage97_task_id"],
        "source_task_id": task["source_task_id"],
        "sequence": task["sequence"],
        "base_method": task["base_method"],
        "reference_gap": task["reference_gap"],
        "target_index": task["target_index"],
        "oracle_candidate": task["oracle_candidate"],
        "selected_candidate": selected_candidate,
        "selected_sideinfo_psnr": f(selected, "sideinfo_psnr"),
        "endpoint_sideinfo_psnr": f(endpoint, "sideinfo_psnr"),
        "oracle_task_best_sideinfo_psnr": f(oracle, "sideinfo_psnr"),
        "teacher_oracle_sideinfo_psnr": f(selected, "teacher_oracle_sideinfo_psnr"),
        "delta_psnr_vs_endpoint": f(selected, "sideinfo_psnr") - f(endpoint, "sideinfo_psnr"),
        "gap_to_oracle_task_best": f(selected, "sideinfo_psnr") - f(oracle, "sideinfo_psnr"),
        "gap_to_teacher_oracle": f(selected, "sideinfo_psnr") - f(selected, "teacher_oracle_sideinfo_psnr"),
        "selection_correct": int(selected_candidate == task["oracle_candidate"]),
    }


def candidate_counts(rows):
    counts = Counter(row["selected_candidate"] for row in rows)
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    out = []
    for policy, items in sorted(grouped.items()):
        def avg(key):
            return sum(float(row[key]) for row in items) / len(items)
        out.append({
            "policy": policy,
            "task_count": len(items),
            "mean_selected_sideinfo_psnr": avg("selected_sideinfo_psnr"),
            "mean_endpoint_sideinfo_psnr": avg("endpoint_sideinfo_psnr"),
            "mean_oracle_task_best_sideinfo_psnr": avg("oracle_task_best_sideinfo_psnr"),
            "mean_teacher_oracle_sideinfo_psnr": avg("teacher_oracle_sideinfo_psnr"),
            "mean_delta_psnr_vs_endpoint": avg("delta_psnr_vs_endpoint"),
            "mean_gap_to_oracle_task_best": avg("gap_to_oracle_task_best"),
            "mean_gap_to_teacher_oracle": avg("gap_to_teacher_oracle"),
            "selection_accuracy": avg("selection_correct"),
            "selected_candidate_counts": candidate_counts(items),
        })
    return out


def summarize_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row["base_method"], row["reference_gap"])].append(row)
    out = []
    for (policy, base_method, gap), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], int(item[0][2]))):
        def avg(key):
            return sum(float(row[key]) for row in items) / len(items)
        out.append({
            "policy": policy,
            "base_method": base_method,
            "reference_gap": int(gap),
            "task_count": len(items),
            "mean_selected_sideinfo_psnr": avg("selected_sideinfo_psnr"),
            "mean_endpoint_sideinfo_psnr": avg("endpoint_sideinfo_psnr"),
            "mean_delta_psnr_vs_endpoint": avg("delta_psnr_vs_endpoint"),
            "selection_accuracy": avg("selection_correct"),
            "selected_candidate_counts": candidate_counts(items),
        })
    return out


def write_report(summary, summary_rows, group_rows, path):
    lines = [
        "# Stage107 Metadata Task-Level Switch Predictor Preflight",
        "",
        "## Configuration",
        "",
        f"- task rows: `{summary['task_count']}`",
        f"- folds: `{summary['fold_count']}`",
        f"- features: `{', '.join(summary['features'])}`",
        "- labels are per-task best rendered deployable candidate from Stage103 rows",
        "- no anchors, rendering, checkpoints, or heavy tensor output",
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
    for row in group_rows:
        lines.append(
            f"| {row['policy']} | {row['base_method']} | {row['reference_gap']} | {row['task_count']} | {row['mean_selected_sideinfo_psnr']} | {row['mean_delta_psnr_vs_endpoint']} | {row['selection_accuracy']} | {row['selected_candidate_counts']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `metadata_mlp_cv` uses only coarse task metadata and is evaluated out-of-fold.",
        "- `stage106_fixed_group_policy` is the packaged metadata group switch from Stage106.",
        "- If metadata MLP does not exceed Stage106, richer decoder-side anchor statistics are needed before task-level switching.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage97_tasks", type=Path, default=DEFAULT_STAGE97_TASKS)
    parser.add_argument("--stage103_rows", type=Path, default=DEFAULT_STAGE103_ROWS)
    parser.add_argument("--stage106_policy", type=Path, default=DEFAULT_STAGE106_POLICY)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--train_steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260628)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    manifest = load_manifest(args.stage97_tasks)
    tasks = load_candidate_tasks(read_csv(args.stage103_rows), manifest)
    fold_map = split_folds(tasks, args.fold_count)
    package = json.loads(args.stage106_policy.read_text(encoding="utf-8"))
    stage106_table = package["selection_table"]
    fallback = package["fallback_candidate"]
    rows = []
    train_logs = []
    for fold in range(args.fold_count):
        train_tasks = [task for task in tasks if fold_map[task["stage97_task_id"]] != fold]
        test_tasks = [task for task in tasks if fold_map[task["stage97_task_id"]] == fold]
        global_best = best_candidate_by_mean(train_tasks, DEPLOYABLE_CANDIDATES)
        train_group_policy = build_train_group_policy(train_tasks, global_best)
        model_bundle = train_metadata_mlp(train_tasks, args, fold)
        model, mean, std, logs = model_bundle
        train_logs.extend(logs)
        for task in test_tasks:
            policy_candidates = {
                "endpoint_only": "endpoint_diff_baseline",
                "global_train_best_policy": global_best,
                "train_fold_group_policy": select_train_group_policy(task, train_group_policy),
                "stage106_fixed_group_policy": select_stage106_policy(task, stage106_table, fallback),
                "metadata_mlp_cv": predict_metadata_mlp(task, (model, mean, std)),
                "oracle_task_best": task["oracle_candidate"],
            }
            for policy, selected_candidate in policy_candidates.items():
                rows.append(evaluate_selection(task, policy, fold, selected_candidate))
    summary_rows = summarize(rows)
    group_summary_rows = summarize_groups(rows)
    rows_csv = args.summary_root / "stage107_metadata_switch_rows.csv"
    summary_csv = args.summary_root / "stage107_metadata_switch_summary.csv"
    group_summary_csv = args.summary_root / "stage107_metadata_switch_group_summary.csv"
    train_log_csv = args.summary_root / "stage107_metadata_switch_train_log.csv"
    summary_json = args.summary_root / "stage107_metadata_switch_summary.json"
    report_md = args.summary_root / "stage107_metadata_switch_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(group_summary_rows, group_summary_csv, GROUP_SUMMARY_FIELDS)
    write_csv(train_logs, train_log_csv, TRAIN_LOG_FIELDS)
    feature_names = [
        "method_id",
        "is_linear",
        "gap_norm",
        "is_gap4",
        "is_gap8",
        "is_gap16",
        "normalized_time",
        "distance_to_mid_time",
        "left_index_norm",
        "right_index_norm",
        "target_index_norm",
        "target_relative_position",
        "right_relative_position",
    ]
    summary = {
        "stage": 107,
        "mode": "metadata task-level switch predictor preflight",
        "task_count": len(tasks),
        "fold_count": args.fold_count,
        "features": feature_names,
        "deployable_candidates": DEPLOYABLE_CANDIDATES,
        "stage97_tasks": str(args.stage97_tasks),
        "stage103_rows": str(args.stage103_rows),
        "stage106_policy": str(args.stage106_policy),
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "group_summary_csv": str(group_summary_csv),
        "train_log_csv": str(train_log_csv),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "notes": [
            "Metadata MLP is evaluated out-of-fold by stage97_task_id.",
            "Rendered PSNR labels are used only for training/evaluation, not as predictor inputs.",
            "No anchors, rendering, checkpoints, or heavy tensors are produced.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, summary_rows, group_summary_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "task_count": len(tasks),
    }, indent=2))


if __name__ == "__main__":
    main()
