import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE66_CSV = REPO_ROOT / "experiments/stage66_davis_feedforward_selector_dataset/stage66_davis_selector_dataset.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage67_davis_selector_predictor_training"


LENGTH_FEATURES = ["segment_length", "middle_count"]
RGB_MOTION_FEATURES = [
    "segment_length",
    "middle_count",
    "normalized_left",
    "normalized_right",
    "rgb_motion_mean",
    "rgb_motion_max",
    "rgb_endpoint_mse",
]
ANCHOR_ENDPOINT_FEATURES = [
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
]
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

MODEL_SPECS = [
    ("length_only_ridge", LENGTH_FEATURES),
    ("rgb_motion_ridge", RGB_MOTION_FEATURES),
    ("anchor_endpoint_ridge", ANCHOR_ENDPOINT_FEATURES),
    ("full_feature_ridge", FULL_FEATURES),
]

METRIC_FIELDS = [
    "model",
    "feature_count",
    "train_rows",
    "eval_rows",
    "train_rmse_log",
    "eval_rmse_log",
    "train_mae_log",
    "eval_mae_log",
    "train_pearson_log",
    "eval_pearson_log",
    "train_spearman_log",
    "eval_spearman_log",
]

PREDICTION_FIELDS = [
    "model",
    "selector_split",
    "dataset",
    "split",
    "sequence",
    "sample",
    "left_index",
    "right_index",
    "segment_length",
    "target_log_cost",
    "pred_log_cost",
    "target_cost",
    "pred_cost",
    "abs_log_error",
]


def parse_float(value):
    if value in {None, ""}:
        return 0.0
    return float(value)


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {
                "dataset": row["dataset"],
                "split": row["split"],
                "selector_split": row["selector_split"],
                "sequence": row["sequence"],
                "sample": row["sample"],
                "left_index": int(row["left_index"]),
                "right_index": int(row["right_index"]),
            }
            numeric_keys = set(FULL_FEATURES) | {
                "label_adapter_mse_mean",
                "label_log_adapter_mse_mean",
            }
            for key in numeric_keys:
                parsed[key] = parse_float(row.get(key))
            rows.append(parsed)
    return rows


def matrix(rows, features):
    return np.asarray([[row[key] for key in features] for row in rows], dtype=np.float64)


def target(rows):
    return np.asarray([row["label_log_adapter_mse_mean"] for row in rows], dtype=np.float64)


def standardize(train_x, eval_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (train_x - mean) / std, (eval_x - mean) / std, mean.reshape(-1), std.reshape(-1)


def fit_ridge(train_x, train_y, l2):
    x = np.concatenate([np.ones((train_x.shape[0], 1), dtype=np.float64), train_x], axis=1)
    reg = np.eye(x.shape[1], dtype=np.float64) * l2
    reg[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ train_y)


def predict(x, weights):
    design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    return design @ weights


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() <= 0 or b.std() <= 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return None
    ar = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort")
    br = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort")
    return pearson(ar, br)


def rmse(pred, truth):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2)))


def mae(pred, truth):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(truth))))


def metric_row(model, features, train_rows, eval_rows, train_pred, eval_pred, train_y, eval_y):
    return {
        "model": model,
        "feature_count": len(features),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_rmse_log": rmse(train_pred, train_y),
        "eval_rmse_log": rmse(eval_pred, eval_y),
        "train_mae_log": mae(train_pred, train_y),
        "eval_mae_log": mae(eval_pred, eval_y),
        "train_pearson_log": pearson(train_pred, train_y),
        "eval_pearson_log": pearson(eval_pred, eval_y),
        "train_spearman_log": spearman(train_pred, train_y),
        "eval_spearman_log": spearman(eval_pred, eval_y),
    }


def prediction_rows(model, rows, pred):
    out = []
    for row, pred_log in zip(rows, pred):
        target_log = row["label_log_adapter_mse_mean"]
        out.append({
            "model": model,
            "selector_split": row["selector_split"],
            "dataset": row["dataset"],
            "split": row["split"],
            "sequence": row["sequence"],
            "sample": row["sample"],
            "left_index": row["left_index"],
            "right_index": row["right_index"],
            "segment_length": row["segment_length"],
            "target_log_cost": target_log,
            "pred_log_cost": float(pred_log),
            "target_cost": row["label_adapter_mse_mean"],
            "pred_cost": float(10.0 ** pred_log),
            "abs_log_error": float(abs(pred_log - target_log)),
        })
    return out


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage66_csv", type=Path, default=DEFAULT_STAGE66_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--l2", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.stage66_csv)
    train_rows = [row for row in rows if row["selector_split"] == "train"]
    eval_rows = [row for row in rows if row["selector_split"] == "eval"]
    if not train_rows or not eval_rows:
        raise RuntimeError(f"Need train/eval rows, got train={len(train_rows)} eval={len(eval_rows)}")

    metrics = []
    predictions = []
    params = {}
    for model_name, features in MODEL_SPECS:
        train_x = matrix(train_rows, features)
        eval_x = matrix(eval_rows, features)
        train_y = target(train_rows)
        eval_y = target(eval_rows)
        train_xz, eval_xz, mean, std = standardize(train_x, eval_x)
        weights = fit_ridge(train_xz, train_y, args.l2)
        train_pred = predict(train_xz, weights)
        eval_pred = predict(eval_xz, weights)
        metrics.append(metric_row(model_name, features, train_rows, eval_rows, train_pred, eval_pred, train_y, eval_y))
        predictions.extend(prediction_rows(model_name, train_rows, train_pred))
        predictions.extend(prediction_rows(model_name, eval_rows, eval_pred))
        params[model_name] = {
            "features": features,
            "l2": args.l2,
            "intercept": float(weights[0]),
            "weights": [float(value) for value in weights[1:].tolist()],
            "mean": [float(value) for value in mean.tolist()],
            "std": [float(value) for value in std.tolist()],
        }

    metrics_csv = args.summary_root / "stage67_selector_predictor_metrics.csv"
    predictions_csv = args.summary_root / "stage67_selector_predictor_predictions.csv"
    params_json = args.summary_root / "stage67_selector_predictor_model_params.json"
    summary_path = args.summary_root / "stage67_selector_predictor_training_summary.json"
    write_csv(metrics, metrics_csv, METRIC_FIELDS)
    write_csv(predictions, predictions_csv, PREDICTION_FIELDS)
    params_json.write_text(json.dumps(params, indent=2), encoding="utf-8")
    best = max(metrics, key=lambda row: (row["eval_spearman_log"] if row["eval_spearman_log"] is not None else -1.0, -row["eval_rmse_log"]))
    summary = {
        "stage": 67,
        "mode": "DAVIS feed-forward selector proxy-cost predictor training",
        "stage66_csv": str(args.stage66_csv),
        "l2": args.l2,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "metrics_csv": str(metrics_csv),
        "predictions_csv": str(predictions_csv),
        "model_params_json": str(params_json),
        "metrics": metrics,
        "best_model_by_eval_spearman_then_rmse": best["model"],
        "best_metrics": best,
        "notes": "Predictors use feed-forward encoder-side features only. The target is the Stage66 anchor-space proxy label, not rendered all-frame PSNR.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "metrics_csv": str(metrics_csv),
        "predictions_csv": str(predictions_csv),
        "best_model": summary["best_model_by_eval_spearman_then_rmse"],
        "best_metrics": best,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
