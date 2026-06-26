import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE44_CSV = REPO_ROOT / "experiments/stage44_rendered_segment_distortion_dataset/stage44_rendered_segment_distortion_dataset.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage47_rendered_cost_predictor_validation"


FULL_FEATURES = [
    "segment_length",
    "middle_count",
    "normalized_left",
    "normalized_right",
    "endpoint_anchor_mse",
    "endpoint_anchor_l1",
    "endpoint_rgb_mse",
    "endpoint_opacity_mse",
    "endpoint_scale_mse",
    "endpoint_xyz_mse",
    "endpoint_rot_mse",
    "left_opacity_mean",
    "right_opacity_mean",
    "left_scale_mean",
    "right_scale_mean",
    "rgb_motion_mean",
    "rgb_motion_max",
    "rgb_endpoint_mse",
]
LENGTH_FEATURES = ["segment_length", "middle_count"]


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {"sample": row["sample"], "heldout_fold": row["heldout_fold"], "split": row["split"]}
            for key in ["left_index", "right_index", "segment_length", "middle_count"]:
                parsed[key] = int(row[key])
            for key, value in row.items():
                if key in parsed or key in {"sample", "heldout_fold", "split", "target_indices"}:
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def make_matrix(rows, features):
    return np.asarray([[row[key] for key in features] for row in rows], dtype=np.float64)


def standardize(train_x, eval_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (train_x - mean) / std, (eval_x - mean) / std


def fit_ridge(train_x, train_y, l2):
    x = np.concatenate([np.ones((train_x.shape[0], 1), dtype=np.float64), train_x], axis=1)
    reg = np.eye(x.shape[1], dtype=np.float64) * l2
    reg[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ train_y)


def predict_ridge(eval_x, weights):
    x = np.concatenate([np.ones((eval_x.shape[0], 1), dtype=np.float64), eval_x], axis=1)
    return x @ weights


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() <= 0 or b.std() <= 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return pearson(ar, br)


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


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
    matrix = make_matrix(rows, features)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (matrix - mean) / std


def transform_by_sample(rows, features, target_mode):
    xs = []
    ys = []
    ordered = []
    for sample, sample_rows in sorted(grouped(rows, "sample").items()):
        del sample
        x = sample_feature_matrix(sample_rows, features)
        log_y = np.asarray([row["log_adapter_mse_sum_est"] for row in sample_rows], dtype=np.float64)
        if target_mode == "rank":
            y = ranks(log_y)
        elif target_mode == "zlog":
            std = log_y.std()
            y = np.zeros_like(log_y) if std < 1e-12 else (log_y - log_y.mean()) / std
        else:
            raise ValueError(target_mode)
        xs.append(x)
        ys.append(y)
        ordered.extend(sample_rows)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), ordered


def evaluate_raw_log(rows, heldout, features, model_name, l2):
    train_rows = [row for row in rows if row["heldout_fold"] == heldout and row["split"] == "train"]
    eval_rows = [row for row in rows if row["heldout_fold"] == heldout and row["split"] == "eval"]
    train_x = make_matrix(train_rows, features)
    eval_x = make_matrix(eval_rows, features)
    train_xz, eval_xz = standardize(train_x, eval_x)
    train_y = np.asarray([row["log_adapter_mse_sum_est"] for row in train_rows], dtype=np.float64)
    eval_y = np.asarray([row["log_adapter_mse_sum_est"] for row in eval_rows], dtype=np.float64)
    weights = fit_ridge(train_xz, train_y, l2)
    pred = predict_ridge(eval_xz, weights)
    pred_rows = prediction_rows(heldout, model_name, eval_rows, pred, np.power(10.0, pred))
    return metrics_row(heldout, model_name, len(features), len(train_rows), len(eval_rows), pred, eval_y, pred_rows), pred_rows


def evaluate_sample_norm(rows, heldout, features, model_name, target_mode, l2):
    train_rows = [row for row in rows if row["heldout_fold"] == heldout and row["split"] == "train"]
    eval_rows = [row for row in rows if row["heldout_fold"] == heldout and row["split"] == "eval"]
    train_x, train_y, _ = transform_by_sample(train_rows, features, target_mode)
    eval_x, eval_y, ordered_eval_rows = transform_by_sample(eval_rows, features, target_mode)
    weights = fit_ridge(train_x, train_y, l2)
    pred = predict_ridge(eval_x, weights)
    pred_cost = pred - pred.min() + 1e-9
    raw_eval_y = np.asarray([row["log_adapter_mse_sum_est"] for row in ordered_eval_rows], dtype=np.float64)
    pred_rows = prediction_rows(heldout, model_name, ordered_eval_rows, pred, pred_cost)
    metric = metrics_row(heldout, model_name, len(features), len(train_rows), len(eval_rows), pred, raw_eval_y, pred_rows)
    metric["target_mode"] = target_mode
    metric["spearman_transformed"] = spearman(pred, eval_y)
    metric["rmse_transformed"] = rmse(pred, eval_y)
    return metric, pred_rows


def metrics_row(heldout, model_name, feature_count, train_count, eval_count, pred, target_log, pred_rows):
    return {
        "heldout_fold": heldout,
        "model": model_name,
        "target_mode": "raw_log",
        "feature_count": feature_count,
        "train_rows": train_count,
        "eval_rows": eval_count,
        "pearson_log": pearson(pred, target_log),
        "spearman_log": spearman(pred, target_log),
        "rmse_log_or_score": rmse(pred, target_log),
        "pearson_cost": pearson([row["pred_cost"] for row in pred_rows], [row["target_cost"] for row in pred_rows]),
        "spearman_transformed": None,
        "rmse_transformed": None,
    }


def prediction_rows(heldout, model_name, rows, pred_score, pred_cost):
    out = []
    for row, score, cost in zip(rows, pred_score, pred_cost):
        out.append({
            "heldout_fold": heldout,
            "model": model_name,
            "sample": row["sample"],
            "left_index": row["left_index"],
            "right_index": row["right_index"],
            "segment_length": row["segment_length"],
            "middle_count": row["middle_count"],
            "target_log_cost": row["log_adapter_mse_sum_est"],
            "target_cost": row["adapter_mse_sum_est"],
            "pred_score": float(score),
            "pred_cost": float(max(cost, 0.0)),
        })
    return out


def write_metrics(rows, path):
    fields = [
        "heldout_fold", "model", "target_mode", "feature_count", "train_rows", "eval_rows", "pearson_log",
        "spearman_log", "rmse_log_or_score", "pearson_cost", "spearman_transformed", "rmse_transformed",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(rows, path):
    fields = [
        "heldout_fold", "model", "sample", "left_index", "right_index", "segment_length", "middle_count",
        "target_log_cost", "target_cost", "pred_score", "pred_cost",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage44_csv", type=Path, default=DEFAULT_STAGE44_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--l2", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.stage44_csv)
    heldouts = sorted({row["heldout_fold"] for row in rows})
    specs = [
        ("length_raw_log", "raw", LENGTH_FEATURES, None),
        ("full_raw_log", "raw", FULL_FEATURES, None),
        ("length_sample_z_rank", "sample", LENGTH_FEATURES, "rank"),
        ("full_sample_z_rank", "sample", FULL_FEATURES, "rank"),
        ("full_sample_z_zlog", "sample", FULL_FEATURES, "zlog"),
    ]
    metrics_rows = []
    prediction_rows_all = []
    for heldout in heldouts:
        for model_name, mode, features, target_mode in specs:
            if mode == "raw":
                metrics, preds = evaluate_raw_log(rows, heldout, features, model_name, args.l2)
            else:
                metrics, preds = evaluate_sample_norm(rows, heldout, features, model_name, target_mode, args.l2)
            metrics_rows.append(metrics)
            prediction_rows_all.extend(preds)

    metrics_csv = args.summary_root / "stage47_predictor_metrics.csv"
    predictions_csv = args.summary_root / "stage47_predictor_predictions.csv"
    summary_path = args.summary_root / "stage47_rendered_cost_predictor_validation_summary.json"
    write_metrics(metrics_rows, metrics_csv)
    write_predictions(prediction_rows_all, predictions_csv)
    aggregates = {}
    for model_name, _, _, _ in specs:
        model_rows = [row for row in metrics_rows if row["model"] == model_name]
        aggregates[model_name] = {
            "mean_spearman_log": float(np.mean([row["spearman_log"] for row in model_rows])),
            "mean_pearson_log": float(np.mean([row["pearson_log"] for row in model_rows])),
            "mean_pearson_cost": float(np.mean([row["pearson_cost"] for row in model_rows])),
        }
    summary = {
        "stage": 47,
        "mode": "feed-forward rendered segment cost predictor validation",
        "stage44_csv": str(args.stage44_csv),
        "l2": args.l2,
        "metrics_csv": str(metrics_csv),
        "predictions_csv": str(predictions_csv),
        "metrics": metrics_rows,
        "aggregates": aggregates,
        "notes": "Features are encoder-side only; rendered cost is offline supervision from Stage44. Predictions are intended for Stage48 DP selector evaluation.",
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
