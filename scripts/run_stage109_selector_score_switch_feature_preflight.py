import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage109_selector_score_switch_feature_preflight"


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import (  # noqa: E402
    DEFAULT_ADAPTER,
    linear_anchor,
    load_adapter,
    load_anchor,
)
from scripts.run_stage98_residual_importance_predictor_smoke import parse_rows  # noqa: E402
from scripts.run_stage101_enhanced_selector_feature_sweep import BASE_FEATURE_DIM, make_full_features  # noqa: E402
from scripts.run_stage102_group_specific_selector_heads import (  # noqa: E402
    TRAIN_LOG_FIELDS as SELECTOR_TRAIN_LOG_FIELDS,
    collect_training_examples,
    train_selector,
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
from scripts.run_stage108_anchor_stat_task_switch_predictor_preflight import (  # noqa: E402
    SwitchMLP,
    build_anchor_features,
)


SWITCH_POLICIES = [
    "endpoint_only",
    "global_train_best_policy",
    "train_fold_group_policy",
    "stage106_fixed_group_policy",
    "metadata_mlp_cv",
    "anchor_stat_mlp_cv",
    "score_stat_mlp_cv",
    "anchor_score_mlp_cv",
    "oracle_task_best",
]


def write_plain_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def score_distribution_stats(scores, keep_fraction):
    flat = scores.reshape(-1).float()
    n = int(flat.numel())
    if n <= 0:
        return [0.0] * 18
    keep = min(max(1, int(round(n * keep_fraction))), n)
    mean = flat.mean()
    std = flat.std(unbiased=False).clamp_min(1e-12)
    quantiles = torch.quantile(flat, torch.tensor([0.10, 0.50, 0.90, 0.99], device=flat.device))
    top_values = torch.topk(flat, k=keep, largest=True).values
    top_mean = top_values.mean()
    top_std = top_values.std(unbiased=False) if top_values.numel() > 1 else torch.zeros_like(top_mean)
    top_min = top_values.amin()
    if keep < n:
        next_score = torch.topk(flat, k=keep + 1, largest=True).values[-1]
        rest_mean = (flat.sum() - top_values.sum()) / float(n - keep)
    else:
        next_score = top_min
        rest_mean = top_mean
    margin = top_min - next_score
    centered = (flat - mean) / std
    probs = torch.softmax(centered, dim=0)
    entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-12))) / math.log(max(n, 2))
    top_indices = torch.topk(flat, k=keep, largest=True).indices
    top_prob_mass = probs[top_indices].sum()
    return [
        float(mean.detach().cpu().item()),
        float(std.detach().cpu().item()),
        float(flat.amin().detach().cpu().item()),
        float(flat.amax().detach().cpu().item()),
        float(quantiles[0].detach().cpu().item()),
        float(quantiles[1].detach().cpu().item()),
        float(quantiles[2].detach().cpu().item()),
        float(quantiles[3].detach().cpu().item()),
        float(top_mean.detach().cpu().item()),
        float(top_std.detach().cpu().item()),
        float(top_min.detach().cpu().item()),
        float(next_score.detach().cpu().item()),
        float(margin.detach().cpu().item()),
        float(rest_mean.detach().cpu().item()),
        float((top_mean - rest_mean).detach().cpu().item()),
        float(entropy.detach().cpu().item()),
        float(top_prob_mass.detach().cpu().item()),
        float((quantiles[2] - quantiles[0]).detach().cpu().item()),
    ]


def pairwise_score_stats(scores_a, scores_b, keep_fraction):
    a = scores_a.reshape(-1).float()
    b = scores_b.reshape(-1).float()
    n = int(min(a.numel(), b.numel()))
    if n <= 0:
        return [0.0] * 5
    a = a[:n]
    b = b[:n]
    keep = min(max(1, int(round(n * keep_fraction))), n)
    top_a = torch.topk(a, k=keep, largest=True).indices
    top_b = torch.topk(b, k=keep, largest=True).indices
    mask_a = torch.zeros(n, dtype=torch.bool, device=a.device)
    mask_b = torch.zeros(n, dtype=torch.bool, device=a.device)
    mask_a[top_a] = True
    mask_b[top_b] = True
    intersection = torch.sum(mask_a & mask_b).float()
    union = torch.sum(mask_a | mask_b).float().clamp_min(1.0)
    a_norm = (a - a.mean()) / a.std(unbiased=False).clamp_min(1e-12)
    b_norm = (b - b.mean()) / b.std(unbiased=False).clamp_min(1e-12)
    corr = torch.mean(a_norm * b_norm)
    return [
        float((intersection / float(keep)).detach().cpu().item()),
        float((intersection / union).detach().cpu().item()),
        float((1.0 - intersection / float(keep)).detach().cpu().item()),
        float(corr.detach().cpu().item()),
        float(torch.mean(torch.abs(a_norm - b_norm)).detach().cpu().item()),
    ]


def predict_scores(model, features, eval_batch_size):
    logits = []
    for start in range(0, features.shape[0], eval_batch_size):
        chunk = features[start:start + eval_batch_size]
        logits.append(model["predictor"]((chunk - model["mean"]) / model["std"]))
    return torch.cat(logits, dim=0)


def build_score_features(task, manifest, adapter, selector_models, device, cache, args):
    row = manifest[task["stage97_task_id"]]
    bits = int(row["bits"])
    left = load_anchor(row["left_anchor_source_item"], row["left_anchor_source_side"], device, bits=bits, cache=cache)
    right = load_anchor(row["right_anchor_source_item"], row["right_anchor_source_side"], device, bits=bits, cache=cache)
    left_attrs = flatten_static_anchor(left)
    right_attrs = flatten_static_anchor(right)
    if task["base_method"] == "linear":
        base = linear_anchor(left, right, task["normalized_time"])
    else:
        t = torch.tensor([task["normalized_time"]], dtype=torch.float32, device=device)
        base = adapter(left, right, t, apply_output_constraints=False)
    base_attrs = flatten_static_anchor(base)
    endpoint_score = torch.sum((right_attrs[0] - left_attrs[0]) ** 2, dim=-1)
    method_id = 0.0 if task["base_method"] == "linear" else 1.0
    gaussian_features = make_full_features(left_attrs, right_attrs, base_attrs, task, method_id)[:, :BASE_FEATURE_DIM]
    score_map = {"endpoint_diff_baseline": endpoint_score}
    for candidate in ["shared_energy_regression", "shared_topk_bce"]:
        score_map[candidate] = predict_scores(selector_models[candidate], gaussian_features, args.eval_batch_size)
    features = []
    for candidate in DEPLOYABLE_CANDIDATES:
        features.extend(score_distribution_stats(score_map[candidate], args.keep_fraction))
    pair_names = [
        ("endpoint_diff_baseline", "shared_energy_regression"),
        ("endpoint_diff_baseline", "shared_topk_bce"),
        ("shared_energy_regression", "shared_topk_bce"),
    ]
    for first, second in pair_names:
        features.extend(pairwise_score_stats(score_map[first], score_map[second], args.keep_fraction))
    return features


def train_score_selectors(args, adapter, device):
    selector_args = argparse.Namespace(**vars(args))
    selector_args.hidden_dim = args.selector_hidden_dim
    train_tasks = parse_rows(args.stage97_tasks, "train", args.gaps, args.max_train_tasks, args.seed)
    train_x, train_y, train_energy, train_group, group_to_id = collect_training_examples(train_tasks, adapter, selector_args, device)
    models = {}
    logs = []
    for model_index, objective in enumerate(args.objectives):
        model, model_logs = train_selector(
            train_x,
            train_y,
            train_energy,
            train_group,
            selector_args,
            device,
            objective,
            "shared",
            "all_groups",
            group_to_id,
            model_index,
        )
        models[f"shared_{objective}"] = model
        logs.extend(model_logs)
    return models, logs, len(train_tasks), int(train_x.shape[0])


def build_feature_tables(tasks, manifest, adapter, selector_models, device, args):
    cache = {}
    tables = {
        "anchor_stat_mlp_cv": {},
        "score_stat_mlp_cv": {},
        "anchor_score_mlp_cv": {},
    }
    with torch.no_grad():
        for task in tasks:
            key = (task["stage97_task_id"], task["base_method"])
            metadata = make_metadata_features(task)
            anchor = build_anchor_features(task, manifest, adapter, device, cache)
            score = build_score_features(task, manifest, adapter, selector_models, device, cache, args)
            tables["anchor_stat_mlp_cv"][key] = metadata + anchor
            tables["score_stat_mlp_cv"][key] = metadata + score
            tables["anchor_score_mlp_cv"][key] = metadata + anchor + score
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return tables


def train_feature_mlp(train_tasks, feature_table, args, fold, seed_offset):
    x = torch.tensor([feature_table[(task["stage97_task_id"], task["base_method"])] for task in train_tasks], dtype=torch.float32)
    y = torch.tensor([DEPLOYABLE_CANDIDATES.index(task["oracle_candidate"]) for task in train_tasks], dtype=torch.long)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_norm = (x - mean) / std
    torch.manual_seed(args.seed + seed_offset + fold)
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


def predict_feature_mlp(task, feature_table, model_bundle):
    model, mean, std = model_bundle
    x = torch.tensor([feature_table[(task["stage97_task_id"], task["base_method"])]], dtype=torch.float32)
    with torch.no_grad():
        pred = torch.argmax(model((x - mean) / std), dim=-1).item()
    return DEPLOYABLE_CANDIDATES[pred]


def run_cv(tasks, feature_tables, args, stage106_table, fallback):
    fold_map = split_folds(tasks, args.fold_count)
    rows = []
    train_logs = []
    for fold in range(args.fold_count):
        train_tasks = [task for task in tasks if fold_map[task["stage97_task_id"]] != fold]
        test_tasks = [task for task in tasks if fold_map[task["stage97_task_id"]] == fold]
        global_best = best_candidate_by_mean(train_tasks, DEPLOYABLE_CANDIDATES)
        train_group_policy = build_train_group_policy(train_tasks, global_best)
        metadata_model, metadata_mean, metadata_std, metadata_logs = train_metadata_mlp(train_tasks, args, fold)
        train_logs.extend({**row, "model": "metadata_mlp_cv"} for row in metadata_logs)
        model_bundles = {
            "metadata_mlp_cv": (metadata_model, metadata_mean, metadata_std),
        }
        for offset, policy in enumerate(["anchor_stat_mlp_cv", "score_stat_mlp_cv", "anchor_score_mlp_cv"], start=1):
            model, mean, std, logs = train_feature_mlp(train_tasks, feature_tables[policy], args, fold, 100 * offset)
            model_bundles[policy] = (model, mean, std)
            train_logs.extend({**row, "model": policy} for row in logs)
        for task in test_tasks:
            policy_candidates = {
                "endpoint_only": "endpoint_diff_baseline",
                "global_train_best_policy": global_best,
                "train_fold_group_policy": select_train_group_policy(task, train_group_policy),
                "stage106_fixed_group_policy": select_stage106_policy(task, stage106_table, fallback),
                "metadata_mlp_cv": predict_metadata_mlp(task, model_bundles["metadata_mlp_cv"]),
                "anchor_stat_mlp_cv": predict_feature_mlp(task, feature_tables["anchor_stat_mlp_cv"], model_bundles["anchor_stat_mlp_cv"]),
                "score_stat_mlp_cv": predict_feature_mlp(task, feature_tables["score_stat_mlp_cv"], model_bundles["score_stat_mlp_cv"]),
                "anchor_score_mlp_cv": predict_feature_mlp(task, feature_tables["anchor_score_mlp_cv"], model_bundles["anchor_score_mlp_cv"]),
                "oracle_task_best": task["oracle_candidate"],
            }
            for policy in SWITCH_POLICIES:
                rows.append(evaluate_selection(task, policy, fold, policy_candidates[policy]))
    return rows, train_logs


def write_report(summary, summary_rows, group_summary_rows, path):
    lines = [
        "# Stage109 Selector-Score Switch Feature Preflight",
        "",
        "## Configuration",
        "",
        f"- task rows: `{summary['task_count']}`",
        f"- folds: `{summary['fold_count']}`",
        f"- selector train tasks: `{summary['selector_train_task_count']}`",
        f"- selector train examples: `{summary['selector_train_example_count']}`",
        f"- feature dims: `{summary['feature_dims']}`",
        "- features are decoder-side metadata, left/right/base anchor aggregate statistics, and selector score/logit statistics",
        "- switch labels come from Stage103 rendered rows and are used only for offline train/eval",
        "- no target dense anchor, target residual, rendered PSNR, checkpoint, or heavy tensor is used as switch predictor input/output",
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
        "- `score_stat_mlp_cv` uses metadata plus selector score/logit statistics only.",
        "- `anchor_score_mlp_cv` combines metadata, anchor aggregate statistics, and selector score/logit statistics.",
        "- If score features do not beat Stage106, Stage106 remains the safe switch baseline until broader rendered labels are available.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage97_tasks", type=Path, default=DEFAULT_STAGE97_TASKS)
    parser.add_argument("--stage103_rows", type=Path, default=DEFAULT_STAGE103_ROWS)
    parser.add_argument("--stage106_policy", type=Path, default=DEFAULT_STAGE106_POLICY)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--base_methods", nargs="+", default=["linear", "stage65_adapter"])
    parser.add_argument("--objectives", nargs="+", default=["topk_bce", "energy_regression"])
    parser.add_argument("--max_train_tasks", type=int, default=96)
    parser.add_argument("--train_gaussians_per_task", type=int, default=3072)
    parser.add_argument("--keep_fraction", type=float, default=0.1)
    parser.add_argument("--selector_hidden_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--eval_batch_size", type=int, default=8192)
    parser.add_argument("--fold_count", type=int, default=5)
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
    selector_models, selector_logs, selector_train_task_count, selector_train_example_count = train_score_selectors(args, adapter, device)
    feature_tables = build_feature_tables(tasks, manifest, adapter, selector_models, device, args)
    package = json.loads(args.stage106_policy.read_text(encoding="utf-8"))
    rows, switch_logs = run_cv(tasks, feature_tables, args, package["selection_table"], package["fallback_candidate"])
    summary_rows = summarize(rows)
    group_summary_rows = summarize_groups(rows)

    rows_csv = args.summary_root / "stage109_selector_score_switch_rows.csv"
    summary_csv = args.summary_root / "stage109_selector_score_switch_summary.csv"
    group_summary_csv = args.summary_root / "stage109_selector_score_switch_group_summary.csv"
    selector_train_log_csv = args.summary_root / "stage109_selector_score_selector_train_log.csv"
    switch_train_log_csv = args.summary_root / "stage109_selector_score_switch_train_log.csv"
    summary_json = args.summary_root / "stage109_selector_score_switch_summary.json"
    report_md = args.summary_root / "stage109_selector_score_switch_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(group_summary_rows, group_summary_csv, GROUP_SUMMARY_FIELDS)
    write_plain_csv(selector_logs, selector_train_log_csv, SELECTOR_TRAIN_LOG_FIELDS)
    write_plain_csv(switch_logs, switch_train_log_csv, ["model", *TRAIN_LOG_FIELDS])
    feature_dims = {name: len(next(iter(table.values()))) if table else 0 for name, table in feature_tables.items()}
    summary = {
        "stage": 109,
        "mode": "selector-score task-level switch feature preflight",
        "task_count": len(tasks),
        "fold_count": args.fold_count,
        "selector_train_task_count": selector_train_task_count,
        "selector_train_example_count": selector_train_example_count,
        "feature_dims": feature_dims,
        "stage97_tasks": str(args.stage97_tasks),
        "stage103_rows": str(args.stage103_rows),
        "stage106_policy": str(args.stage106_policy),
        "adapter": str(args.adapter),
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "group_summary_csv": str(group_summary_csv),
        "selector_train_log_csv": str(selector_train_log_csv),
        "switch_train_log_csv": str(switch_train_log_csv),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "notes": [
            "Selector models are trained with encoder-side target labels, but switch features use only decoder-side anchors/base predictions and selector logits.",
            "Rendered PSNR labels from Stage103 are used only for switch train/eval labels.",
            "No checkpoint or heavy tensor is saved.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, summary_rows, group_summary_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "task_count": len(tasks),
        "feature_dims": feature_dims,
    }, indent=2))


if __name__ == "__main__":
    main()
