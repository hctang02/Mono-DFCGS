import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE37_CSV = REPO_ROOT / "experiments/stage37_deployable_selector_cost_dataset/stage37_deployable_selector_cost_dataset.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage40_normalized_cost_predictor_validation"


sys.path.insert(0, str(REPO_ROOT))
from scripts.run_stage38_deployable_cost_predictor_validation import FULL_FEATURES, LENGTH_FEATURES, load_rows, pearson, rmse, spearman  # noqa: E402


def grouped(rows, key):
    out = defaultdict(list)
    for row in rows:
        out[row[key]].append(row)
    return out


def ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(values.size, dtype=np.float64)
    if values.size <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return rank / float(values.size - 1)


def sample_feature_matrix(rows, features):
    matrix = np.asarray([[row[key] for key in features] for row in rows], dtype=np.float64)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (matrix - mean) / std


def transform_rows_by_sample(rows, features, target_mode):
    xs = []
    ys = []
    originals = []
    for sample, sample_rows in sorted(grouped(rows, "sample").items()):
        del sample
        x = sample_feature_matrix(sample_rows, features)
        log_y = np.asarray([row["log_label_anchor_attr_mse"] for row in sample_rows], dtype=np.float64)
        if target_mode == "zlog":
            y_std = log_y.std()
            if y_std < 1e-12:
                y = np.zeros_like(log_y)
            else:
                y = (log_y - log_y.mean()) / y_std
        elif target_mode == "rank":
            y = ranks(log_y)
        else:
            raise ValueError(target_mode)
        xs.append(x)
        ys.append(y)
        originals.extend(sample_rows)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), originals


def fit_ridge(train_x, train_y, l2):
    x = np.concatenate([np.ones((train_x.shape[0], 1), dtype=np.float64), train_x], axis=1)
    reg = np.eye(x.shape[1], dtype=np.float64) * l2
    reg[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ train_y)


def predict_ridge(eval_x, weights):
    x = np.concatenate([np.ones((eval_x.shape[0], 1), dtype=np.float64), eval_x], axis=1)
    return x @ weights


def evaluate_model(rows, heldout, features, model_name, target_mode, l2):
    train_rows = [row for row in rows if row["heldout_fold"] == heldout and row["split"] == "train"]
    eval_rows = [row for row in rows if row["heldout_fold"] == heldout and row["split"] == "eval"]
    train_x, train_y, _ = transform_rows_by_sample(train_rows, features, target_mode)
    eval_x, eval_y, ordered_eval_rows = transform_rows_by_sample(eval_rows, features, target_mode)
    weights = fit_ridge(train_x, train_y, l2)
    pred = predict_ridge(eval_x, weights)
    raw_eval_y = np.asarray([row["log_label_anchor_attr_mse"] for row in ordered_eval_rows], dtype=np.float64)
    pred_min = float(pred.min())
    pred_rows = []
    for row, pred_value, target_value in zip(ordered_eval_rows, pred, eval_y):
        pred_selector_cost = float(pred_value - pred_min + 1e-9)
        pred_rows.append({
            "heldout_fold": heldout,
            "model": model_name,
            "sample": row["sample"],
            "left_index": row["left_index"],
            "right_index": row["right_index"],
            "segment_length": row["segment_length"],
            "target_log_cost": row["log_label_anchor_attr_mse"],
            "target_transformed_cost": float(target_value),
            "pred_score": float(pred_value),
            "pred_cost": pred_selector_cost,
            "target_cost": row["label_anchor_attr_mse"],
        })
    metrics = {
        "heldout_fold": heldout,
        "model": model_name,
        "target_mode": target_mode,
        "feature_count": len(features),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "pearson_transformed": pearson(pred, eval_y),
        "spearman_transformed": spearman(pred, eval_y),
        "rmse_transformed": rmse(pred, eval_y),
        "pearson_log": pearson(pred, raw_eval_y),
        "spearman_log": spearman(pred, raw_eval_y),
    }
    return metrics, pred_rows, {"features": features, "target_mode": target_mode, "weights": [float(v) for v in weights.tolist()]}


def write_metrics(rows, path):
    fields = [
        "heldout_fold", "model", "target_mode", "feature_count", "train_rows", "eval_rows", "pearson_transformed",
        "spearman_transformed", "rmse_transformed", "pearson_log", "spearman_log",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(rows, path):
    fields = [
        "heldout_fold", "model", "sample", "left_index", "right_index", "segment_length", "target_log_cost",
        "target_transformed_cost", "pred_score", "pred_cost", "target_cost",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage37_csv", type=Path, default=DEFAULT_STAGE37_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--l2", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.stage37_csv)
    heldouts = sorted({row["heldout_fold"] for row in rows})
    specs = [
        ("length_sample_z_zlog", LENGTH_FEATURES, "zlog"),
        ("full_sample_z_zlog", FULL_FEATURES, "zlog"),
        ("length_sample_z_rank", LENGTH_FEATURES, "rank"),
        ("full_sample_z_rank", FULL_FEATURES, "rank"),
    ]
    metrics_rows = []
    prediction_rows = []
    model_params = {}
    for heldout in heldouts:
        for model_name, features, target_mode in specs:
            metrics, preds, params = evaluate_model(rows, heldout, features, model_name, target_mode, args.l2)
            metrics_rows.append(metrics)
            prediction_rows.extend(preds)
            model_params[f"{heldout}/{model_name}"] = params

    metrics_csv = args.summary_root / "stage40_predictor_metrics.csv"
    predictions_csv = args.summary_root / "stage40_predictor_predictions.csv"
    summary_path = args.summary_root / "stage40_normalized_cost_predictor_validation_summary.json"
    write_metrics(metrics_rows, metrics_csv)
    write_predictions(prediction_rows, predictions_csv)
    aggregates = {}
    for model_name, _, _ in specs:
        model_rows = [row for row in metrics_rows if row["model"] == model_name]
        aggregates[model_name] = {
            "mean_spearman_log": float(np.mean([row["spearman_log"] for row in model_rows])),
            "mean_spearman_transformed": float(np.mean([row["spearman_transformed"] for row in model_rows])),
            "mean_rmse_transformed": float(np.mean([row["rmse_transformed"] for row in model_rows])),
        }
    summary = {
        "stage": 40,
        "mode": "sample-normalized deployable cost predictor validation",
        "stage37_csv": str(args.stage37_csv),
        "l2": args.l2,
        "metrics_csv": str(metrics_csv),
        "predictions_csv": str(predictions_csv),
        "metrics": metrics_rows,
        "aggregates": aggregates,
        "model_params": model_params,
        "notes": "Feature normalization uses only same-sample candidate features and is deployable. Target normalization/ranking is used only during training/evaluation; the emitted pred_cost is a per-heldout shifted score suitable for fixed-budget DP.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "metrics_csv": str(metrics_csv),
        "predictions_csv": str(predictions_csv),
        "aggregates": aggregates,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
