import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE37_CSV = REPO_ROOT / "experiments/stage37_deployable_selector_cost_dataset/stage37_deployable_selector_cost_dataset.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage38_deployable_cost_predictor_validation"


FULL_FEATURES = [
    "segment_length",
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
LENGTH_FEATURES = ["segment_length"]


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {"sample": row["sample"], "heldout_fold": row["heldout_fold"], "split": row["split"]}
            for key in ["left_index", "right_index", "segment_length"]:
                parsed[key] = int(row[key])
            for key, value in row.items():
                if key in parsed or key in {"sample", "heldout_fold", "split"}:
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
    return (train_x - mean) / std, (eval_x - mean) / std, mean.reshape(-1), std.reshape(-1)


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


def evaluate_model(rows, heldout, features, model_name, l2):
    train_rows = [row for row in rows if row["heldout_fold"] == heldout and row["split"] == "train"]
    eval_rows = [row for row in rows if row["heldout_fold"] == heldout and row["split"] == "eval"]
    train_x = make_matrix(train_rows, features)
    eval_x = make_matrix(eval_rows, features)
    train_y = np.asarray([row["log_label_anchor_attr_mse"] for row in train_rows], dtype=np.float64)
    eval_y = np.asarray([row["log_label_anchor_attr_mse"] for row in eval_rows], dtype=np.float64)
    train_xz, eval_xz, mean, std = standardize(train_x, eval_x)
    weights = fit_ridge(train_xz, train_y, l2)
    pred = predict_ridge(eval_xz, weights)
    pred_rows = []
    for row, pred_value in zip(eval_rows, pred):
        pred_rows.append({
            "heldout_fold": heldout,
            "model": model_name,
            "sample": row["sample"],
            "left_index": row["left_index"],
            "right_index": row["right_index"],
            "segment_length": row["segment_length"],
            "target_log_cost": row["log_label_anchor_attr_mse"],
            "pred_log_cost": float(pred_value),
            "target_cost": row["label_anchor_attr_mse"],
            "pred_cost": float(10.0 ** pred_value),
        })
    metrics = {
        "heldout_fold": heldout,
        "model": model_name,
        "feature_count": len(features),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "pearson_log": pearson(pred, eval_y),
        "spearman_log": spearman(pred, eval_y),
        "rmse_log": rmse(pred, eval_y),
        "pearson_cost": pearson([r["pred_cost"] for r in pred_rows], [r["target_cost"] for r in pred_rows]),
    }
    return metrics, pred_rows, {"features": features, "weights": [float(v) for v in weights.tolist()], "mean": [float(v) for v in mean], "std": [float(v) for v in std]}


def write_metrics(rows, path):
    fields = ["heldout_fold", "model", "feature_count", "train_rows", "eval_rows", "pearson_log", "spearman_log", "rmse_log", "pearson_cost"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(rows, path):
    fields = ["heldout_fold", "model", "sample", "left_index", "right_index", "segment_length", "target_log_cost", "pred_log_cost", "target_cost", "pred_cost"]
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
    metrics_rows = []
    prediction_rows = []
    model_params = {}
    for heldout in heldouts:
        for model_name, features in [("length_only_ridge", LENGTH_FEATURES), ("full_feature_ridge", FULL_FEATURES)]:
            metrics, preds, params = evaluate_model(rows, heldout, features, model_name, args.l2)
            metrics_rows.append(metrics)
            prediction_rows.extend(preds)
            model_params[f"{heldout}/{model_name}"] = params

    metrics_csv = args.summary_root / "stage38_predictor_metrics.csv"
    predictions_csv = args.summary_root / "stage38_predictor_predictions.csv"
    summary_path = args.summary_root / "stage38_deployable_cost_predictor_validation_summary.json"
    write_metrics(metrics_rows, metrics_csv)
    write_predictions(prediction_rows, predictions_csv)
    full = [row for row in metrics_rows if row["model"] == "full_feature_ridge"]
    length = [row for row in metrics_rows if row["model"] == "length_only_ridge"]
    summary = {
        "stage": 38,
        "mode": "deployable selector cost predictor validation",
        "stage37_csv": str(args.stage37_csv),
        "l2": args.l2,
        "metrics_csv": str(metrics_csv),
        "predictions_csv": str(predictions_csv),
        "metrics": metrics_rows,
        "model_params": model_params,
        "mean_full_feature_spearman_log": float(np.mean([row["spearman_log"] for row in full])),
        "mean_length_only_spearman_log": float(np.mean([row["spearman_log"] for row in length])),
        "mean_full_feature_rmse_log": float(np.mean([row["rmse_log"] for row in full])),
        "mean_length_only_rmse_log": float(np.mean([row["rmse_log"] for row in length])),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "metrics_csv": str(metrics_csv),
        "predictions_csv": str(predictions_csv),
        "mean_full_feature_spearman_log": summary["mean_full_feature_spearman_log"],
        "mean_length_only_spearman_log": summary["mean_length_only_spearman_log"],
        "mean_full_feature_rmse_log": summary["mean_full_feature_rmse_log"],
        "mean_length_only_rmse_log": summary["mean_length_only_rmse_log"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
