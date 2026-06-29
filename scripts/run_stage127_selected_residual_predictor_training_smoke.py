import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE126_ROOT = REPO_ROOT / "experiments/stage126_selected_residual_predictor_dataset_package"
DEFAULT_DATASET_ROWS = DEFAULT_STAGE126_ROOT / "stage126_selected_residual_predictor_dataset_rows.csv"
DEFAULT_STATS_JSON = DEFAULT_STAGE126_ROOT / "stage126_selected_residual_predictor_train_stats.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage127_selected_residual_predictor_training_smoke"
DEFAULT_HEAVY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage127_selected_residual_predictor_training_smoke")

METRIC_FIELDS = [
    "setting_label",
    "setting_role",
    "keep_fraction",
    "side_bits",
    "train_task_count",
    "eval_task_count",
    "train_sample_count",
    "eval_sample_count",
    "feature_dim",
    "residual_dim",
    "hidden_dim",
    "train_steps",
    "batch_size",
    "train_zero_mse",
    "train_pred_mse",
    "train_mse_reduction",
    "eval_zero_mse",
    "eval_pred_mse",
    "eval_mse_reduction",
    "checkpoint_path",
]

TRAIN_LOG_FIELDS = ["setting_label", "step", "train_loss_norm"]


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402
from mono_dfcgs.residual_value_predictor import (  # noqa: E402
    SelectedResidualValueMLP,
    endpoint_diff_topk_indices,
    selected_residual_feature_matrix,
)
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import linear_anchor, load_anchor  # noqa: E402


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_stats(path):
    raw = json.loads(path.read_text(encoding="utf-8"))["stats_by_setting"]
    return raw


def rows_for_setting(rows, setting_label, task_split):
    return [row for row in rows if row["setting_label"] == setting_label and row["task_split"] == task_split]


def sample_rows(rows, max_tasks, seed):
    if max_tasks <= 0 or len(rows) <= max_tasks:
        return rows
    rng = random.Random(seed)
    out = list(rows)
    rng.shuffle(out)
    return out[:max_tasks]


def build_samples(rows, max_gaussians_per_task, seed, device):
    feature_chunks = []
    label_chunks = []
    for row_index, row in enumerate(rows):
        left = load_anchor(row["left_anchor_source_item"], row["left_anchor_source_side"], device, bits=int(float(row["bits"])), cache=None)
        right = load_anchor(row["right_anchor_source_item"], row["right_anchor_source_side"], device, bits=int(float(row["bits"])), cache=None)
        target = load_anchor(row["target_dense_anchor_source_item"], row["target_dense_anchor_source_side"], device, bits=None, cache=None)
        left_attrs = flatten_static_anchor(left)
        right_attrs = flatten_static_anchor(right)
        base_attrs = flatten_static_anchor(linear_anchor(left, right, float(row["normalized_time"])))
        target_attrs = flatten_static_anchor(target)
        selected_indices = endpoint_diff_topk_indices(left_attrs, right_attrs, float(row["keep_fraction"]))
        if max_gaussians_per_task > 0 and int(selected_indices.numel()) > max_gaussians_per_task:
            generator = torch.Generator(device="cpu").manual_seed(int(seed) + row_index)
            choice = torch.randperm(int(selected_indices.numel()), generator=generator)[:max_gaussians_per_task]
            selected_indices = selected_indices[choice]
        features = selected_residual_feature_matrix(left_attrs, right_attrs, base_attrs, selected_indices, float(row["normalized_time"]))
        keep_idx = torch.as_tensor(selected_indices, dtype=torch.int64, device=device).reshape(-1)
        labels = target_attrs[0, keep_idx, :] - base_attrs[0, keep_idx, :]
        feature_chunks.append(features.detach())
        label_chunks.append(labels.detach())
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(feature_chunks, dim=0), torch.cat(label_chunks, dim=0)


def normalized_tensors(features, labels, stats, device):
    feature_mean = torch.tensor(stats["feature_mean"], dtype=torch.float32, device=device)
    feature_std = torch.tensor(stats["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    label_mean = torch.tensor(stats["label_mean"], dtype=torch.float32, device=device)
    label_std = torch.tensor(stats["label_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    return (features - feature_mean) / feature_std, (labels - label_mean) / label_std, label_mean, label_std


def evaluate_model(model, features, labels, stats, device, chunk_size=65536):
    model.eval()
    feature_norm, _label_norm, label_mean, label_std = normalized_tensors(features, labels, stats, device)
    preds = []
    with torch.no_grad():
        for start in range(0, int(feature_norm.shape[0]), chunk_size):
            pred_norm = model(feature_norm[start:start + chunk_size])
            preds.append(pred_norm * label_std + label_mean)
    pred = torch.cat(preds, dim=0)
    pred_mse = float(F.mse_loss(pred, labels).item())
    zero_mse = float(torch.mean(labels * labels).item())
    reduction = 1.0 - pred_mse / zero_mse if zero_mse > 0.0 else 0.0
    return zero_mse, pred_mse, reduction


def train_one_setting(setting_label, setting_rows, setting_stats, args, device):
    train_rows = sample_rows(rows_for_setting(setting_rows, setting_label, "train"), args.max_train_tasks, args.seed)
    eval_rows = sample_rows(rows_for_setting(setting_rows, setting_label, "eval"), args.max_eval_tasks, args.seed + 1000)
    train_x, train_y = build_samples(train_rows, args.gaussians_per_task, args.seed, device)
    eval_x, eval_y = build_samples(eval_rows, args.eval_gaussians_per_task, args.seed + 2000, device)
    train_xn, train_yn, _label_mean, _label_std = normalized_tensors(train_x, train_y, setting_stats, device)
    model = SelectedResidualValueMLP(
        feature_dim=int(setting_stats["feature_dim"]),
        residual_dim=int(setting_stats["residual_dim"]),
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logs = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for step in range(1, args.train_steps + 1):
        indices = torch.randint(0, int(train_xn.shape[0]), (args.batch_size,), generator=generator, device=device)
        pred = model(train_xn[indices])
        loss = F.mse_loss(pred, train_yn[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            logs.append({"setting_label": setting_label, "step": step, "train_loss_norm": float(loss.detach().item())})
    train_zero, train_pred, train_reduction = evaluate_model(model, train_x, train_y, setting_stats, device)
    eval_zero, eval_pred, eval_reduction = evaluate_model(model, eval_x, eval_y, setting_stats, device)
    setting_dir = args.heavy_root / setting_label
    setting_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = setting_dir / "selected_residual_value_mlp.safetensors"
    save_file({key: value.detach().cpu() for key, value in model.state_dict().items()}, str(checkpoint_path))
    metric = {
        "setting_label": setting_label,
        "setting_role": setting_stats["role"],
        "keep_fraction": setting_stats["keep_fraction"],
        "side_bits": setting_stats["side_bits"],
        "train_task_count": len(train_rows),
        "eval_task_count": len(eval_rows),
        "train_sample_count": int(train_x.shape[0]),
        "eval_sample_count": int(eval_x.shape[0]),
        "feature_dim": int(setting_stats["feature_dim"]),
        "residual_dim": int(setting_stats["residual_dim"]),
        "hidden_dim": args.hidden_dim,
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "train_zero_mse": train_zero,
        "train_pred_mse": train_pred,
        "train_mse_reduction": train_reduction,
        "eval_zero_mse": eval_zero,
        "eval_pred_mse": eval_pred,
        "eval_mse_reduction": eval_reduction,
        "checkpoint_path": str(checkpoint_path),
    }
    return metric, logs


def format_float(value):
    return f"{float(value):.6f}"


def write_report(summary, metrics, path):
    lines = [
        "# Stage127 Selected Residual Predictor Training Smoke",
        "",
        "## Scope",
        "",
        "- Trains small per-Gaussian MLP predictors for selected residual values.",
        "- Checkpoints are saved outside git; repo stores only metrics and manifests.",
        "- Target dense anchors are used only for training/eval labels.",
        "",
        "## Metrics",
        "",
        "| setting | role | train samples | eval samples | train reduction | eval reduction | eval zero MSE | eval pred MSE | checkpoint |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['setting_label']} | {row['setting_role']} | {row['train_sample_count']} | {row['eval_sample_count']} | {format_float(row['train_mse_reduction'])} | {format_float(row['eval_mse_reduction'])} | {format_float(row['eval_zero_mse'])} | {format_float(row['eval_pred_mse'])} | `{row['checkpoint_path']}` |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- metrics CSV: `{summary['metrics_csv']}`",
        f"- train log CSV: `{summary['train_log_csv']}`",
        f"- package JSON: `{summary['package_json']}`",
        f"- report Markdown: `{summary['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_rows", type=Path, default=DEFAULT_DATASET_ROWS)
    parser.add_argument("--stats_json", type=Path, default=DEFAULT_STATS_JSON)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--settings", nargs="+", default=["q4_top20", "q4_top10"])
    parser.add_argument("--max_train_tasks", type=int, default=120)
    parser.add_argument("--max_eval_tasks", type=int, default=60)
    parser.add_argument("--gaussians_per_task", type=int, default=512)
    parser.add_argument("--eval_gaussians_per_task", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--train_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = read_csv(args.dataset_rows)
    stats_by_setting = load_stats(args.stats_json)
    metrics = []
    train_logs = []
    for setting_label in args.settings:
        metric, logs = train_one_setting(setting_label, rows, stats_by_setting[setting_label], args, device)
        metrics.append(metric)
        train_logs.extend(logs)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics_csv = args.summary_root / "stage127_selected_residual_predictor_training_smoke_metrics.csv"
    train_log_csv = args.summary_root / "stage127_selected_residual_predictor_training_smoke_train_log.csv"
    package_json = args.summary_root / "stage127_selected_residual_predictor_training_smoke_package.json"
    report_md = args.summary_root / "stage127_selected_residual_predictor_training_smoke_report.md"
    write_csv(metrics, metrics_csv, METRIC_FIELDS)
    write_csv(train_logs, train_log_csv, TRAIN_LOG_FIELDS)
    package = {
        "stage": 127,
        "mode": "selected residual predictor training smoke",
        "dataset_rows": str(args.dataset_rows),
        "stats_json": str(args.stats_json),
        "summary_root": str(args.summary_root),
        "heavy_root": str(args.heavy_root),
        "settings": args.settings,
        "metrics_csv": str(metrics_csv),
        "train_log_csv": str(train_log_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "metrics": metrics,
        "notes": [
            "Model checkpoints are saved outside git.",
            "Target dense anchors are used only for training/eval labels.",
            "Committed outputs contain metrics and checkpoint paths only.",
        ],
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, metrics, report_md)
    print(json.dumps({"package": str(package_json), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
