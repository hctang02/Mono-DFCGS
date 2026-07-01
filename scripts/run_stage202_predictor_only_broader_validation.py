import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_MANIFEST = REPO_ROOT / "experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_tasks.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage202_predictor_only_broader_validation"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage202_predictor_only_broader_validation")
DEFAULT_STAGE78 = REPO_ROOT / "experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_psnr_table.csv"

sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.learned_gs_predictor import TemporalBasisGSRefiner  # noqa: E402
from scripts.run_stage201_predictor_only_smoke import (  # noqa: E402
    METRIC_FIELDS,
    align_target_to_render,
    balanced_sample,
    endpoint_identity_check,
    evaluate_rows,
    mean,
    parse_task_rows,
    stage78_old_adapter_reference,
    train_model,
    write_csv,
)


CONFIG_FIELDS = [
    "config_label",
    "hidden_dim",
    "global_dim",
    "train_steps",
    "eval_every",
    "lr",
    "render_loss_weight",
    "train_task_count",
    "eval_task_count",
    "linear_eval_psnr",
    "initial_eval_psnr",
    "best_eval_psnr",
    "final_eval_psnr",
    "best_delta_psnr_vs_linear",
    "final_delta_psnr_vs_linear",
    "best_step",
    "endpoint_error",
    "best_checkpoint_path",
    "final_checkpoint_path",
]
SUMMARY_FIELDS = [
    "config_label",
    "phase",
    "task_split",
    "task_count",
    "mean_anchor_mse",
    "mean_render_mse",
    "mean_psnr",
    "delta_psnr_vs_linear",
    "delta_psnr_vs_stage78_old_adapter_reference",
]
METRIC_FIELDS_WITH_CONFIG = ["config_label", *METRIC_FIELDS]
TRAIN_LOG_FIELDS = ["config_label", "step", "task_id", "sequence", "reference_gap", "anchor_loss", "render_loss", "total_loss", "train_psnr"]
GATE_FIELDS = ["gate", "status", "value", "threshold", "detail"]
SELECTED_TASK_FIELDS = ["selection_split", "task_id", "sequence", "reference_gap", "left_index", "right_index", "target_index", "normalized_time"]


CONFIGS = [
    {
        "config_label": "anchor_render_lr2e4",
        "hidden_dim": 192,
        "global_dim": 64,
        "train_steps": 32,
        "eval_every": 16,
        "lr": 2e-4,
        "render_loss_weight": 0.02,
    },
    {
        "config_label": "anchor_only_lr1e3",
        "hidden_dim": 192,
        "global_dim": 64,
        "train_steps": 32,
        "eval_every": 16,
        "lr": 1e-3,
        "render_loss_weight": 0.0,
    },
    {
        "config_label": "render_heavy_lr1e4",
        "hidden_dim": 192,
        "global_dim": 64,
        "train_steps": 24,
        "eval_every": 12,
        "lr": 1e-4,
        "render_loss_weight": 0.1,
    },
]


class ConfigArgs:
    def __init__(self, base_args, config):
        self.lr = float(config["lr"])
        self.weight_decay = float(base_args.weight_decay)
        self.render_loss_weight = float(config["render_loss_weight"])
        self.train_steps = int(config["train_steps"])
        self.eval_every = int(config["eval_every"])
        self.no_regression_tolerance_db = float(base_args.no_regression_tolerance_db)


def with_config(rows, config_label):
    out = []
    for row in rows:
        item = dict(row)
        item["config_label"] = config_label
        out.append(item)
    return out


def selected_task_rows(train_rows, eval_rows):
    out = []
    for selection_split, rows in (("train", train_rows), ("eval", eval_rows)):
        for row in rows:
            out.append({
                "selection_split": selection_split,
                "task_id": row["task_id"],
                "sequence": row["sequence"],
                "reference_gap": row["reference_gap"],
                "left_index": row["left_index"],
                "right_index": row["right_index"],
                "target_index": row["target_index"],
                "normalized_time": row["normalized_time"],
            })
    return out


def summarize_config(config_label, metric_rows, old_adapter_reference):
    linear_by_split = {}
    for split in sorted({row["task_split"] for row in metric_rows}):
        linear_rows = [row for row in metric_rows if row["phase"] == "linear" and row["task_split"] == split and row["status"] == "ok"]
        if linear_rows:
            linear_by_split[split] = mean(row["psnr"] for row in linear_rows)
    grouped = {}
    for row in metric_rows:
        if row["status"] != "ok":
            continue
        grouped.setdefault((row["phase"], row["task_split"]), []).append(row)
    out = []
    for (phase, task_split), rows in sorted(grouped.items()):
        mean_psnr = mean(row["psnr"] for row in rows)
        ref = old_adapter_reference.get(task_split)
        out.append({
            "config_label": config_label,
            "phase": phase,
            "task_split": task_split,
            "task_count": len(rows),
            "mean_anchor_mse": mean(row["anchor_mse"] for row in rows),
            "mean_render_mse": mean(row["render_mse"] for row in rows),
            "mean_psnr": mean_psnr,
            "delta_psnr_vs_linear": mean_psnr - linear_by_split.get(task_split, mean_psnr),
            "delta_psnr_vs_stage78_old_adapter_reference": mean_psnr - ref if ref is not None else "",
        })
    return out


def phase_psnr(summary_rows, phase, task_split="eval"):
    row = next(item for item in summary_rows if item["phase"] == phase and item["task_split"] == task_split)
    return float(row["mean_psnr"])


def run_config(config, train_rows, eval_rows, args, opt, background, device, old_adapter_reference):
    config_label = config["config_label"]
    cache = {}
    torch.manual_seed(args.seed)
    model = TemporalBasisGSRefiner(
        hidden_dim=int(config["hidden_dim"]),
        global_dim=int(config["global_dim"]),
        zero_init_residual=True,
        apply_output_constraints=False,
    ).to(device)
    linear_train = evaluate_rows(train_rows, model, "linear", "train", device, cache, opt, background)
    linear_eval = evaluate_rows(eval_rows, model, "linear", "eval", device, cache, opt, background)
    logs, initial_eval, best_state, best_step, best_eval_psnr = train_model(
        model,
        train_rows,
        eval_rows,
        ConfigArgs(args, config),
        device,
        cache,
        opt,
        background,
    )
    final_train = evaluate_rows(train_rows, model, "predictor_final", "train", device, cache, opt, background)
    final_eval = evaluate_rows(eval_rows, model, "predictor_final", "eval", device, cache, opt, background)
    config_dir = args.heavy_root / config_label
    config_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint_path = config_dir / "temporal_basis_gs_refiner_final.safetensors"
    best_checkpoint_path = config_dir / "temporal_basis_gs_refiner_best.safetensors"
    save_file({key: value.detach().cpu() for key, value in model.state_dict().items()}, str(final_checkpoint_path))
    save_file(best_state, str(best_checkpoint_path))
    best_model = TemporalBasisGSRefiner(
        hidden_dim=int(config["hidden_dim"]),
        global_dim=int(config["global_dim"]),
        zero_init_residual=True,
        apply_output_constraints=False,
    ).to(device)
    best_model.load_state_dict({key: value.to(device) for key, value in best_state.items()}, strict=True)
    best_train = evaluate_rows(train_rows, best_model, "predictor_best", "train", device, cache, opt, background)
    best_eval = evaluate_rows(eval_rows, best_model, "predictor_best", "eval", device, cache, opt, background)
    endpoint_error, _endpoint_t0, _endpoint_t1 = endpoint_identity_check(best_model, eval_rows[0], device, cache, opt)
    metric_rows = linear_train + linear_eval + initial_eval + final_train + final_eval + best_train + best_eval
    summary_rows = summarize_config(config_label, metric_rows, old_adapter_reference)
    linear_eval_psnr = phase_psnr(summary_rows, "linear")
    initial_eval_psnr = phase_psnr(summary_rows, "predictor_initial")
    best_eval_summary = phase_psnr(summary_rows, "predictor_best")
    final_eval_psnr = phase_psnr(summary_rows, "predictor_final")
    result = {
        "config_label": config_label,
        "hidden_dim": config["hidden_dim"],
        "global_dim": config["global_dim"],
        "train_steps": config["train_steps"],
        "eval_every": config["eval_every"],
        "lr": config["lr"],
        "render_loss_weight": config["render_loss_weight"],
        "train_task_count": len(train_rows),
        "eval_task_count": len(eval_rows),
        "linear_eval_psnr": linear_eval_psnr,
        "initial_eval_psnr": initial_eval_psnr,
        "best_eval_psnr": best_eval_summary,
        "final_eval_psnr": final_eval_psnr,
        "best_delta_psnr_vs_linear": best_eval_summary - linear_eval_psnr,
        "final_delta_psnr_vs_linear": final_eval_psnr - linear_eval_psnr,
        "best_step": best_step,
        "endpoint_error": endpoint_error,
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, summary_rows, with_config(metric_rows, config_label), with_config(logs, config_label)


def gate_rows(config_rows, metric_rows, args):
    errors = [row for row in metric_rows if row["status"] != "ok"]
    best_delta = max(float(row["best_delta_psnr_vs_linear"]) for row in config_rows)
    best_config = max(config_rows, key=lambda row: float(row["best_delta_psnr_vs_linear"]))
    max_endpoint = max(float(row["endpoint_error"]) for row in config_rows)
    any_no_regression = any(float(row["best_delta_psnr_vs_linear"]) >= -float(args.no_regression_tolerance_db) for row in config_rows)
    return [
        {
            "gate": "metric_rows_ok",
            "status": "pass" if not errors else "fail",
            "value": len(errors),
            "threshold": "0",
            "detail": "all rendered metrics must use explicit target-shape alignment",
        },
        {
            "gate": "endpoint_identity_all_configs",
            "status": "pass" if max_endpoint <= 1e-6 else "fail",
            "value": max_endpoint,
            "threshold": "<=1e-6",
            "detail": "t=0/t=1 output equals decoded endpoints",
        },
        {
            "gate": "predictor_only_payload",
            "status": "pass",
            "value": 0,
            "threshold": "0 per-frame payload bytes",
            "detail": "Stage202 still sends no residual or latent payload",
        },
        {
            "gate": "any_config_no_regression",
            "status": "pass" if any_no_regression else "fail",
            "value": max(float(row["best_delta_psnr_vs_linear"]) for row in config_rows),
            "threshold": f">= -{args.no_regression_tolerance_db} dB",
            "detail": "best checkpoint can fall back to zero-init linear",
        },
        {
            "gate": "predictor_headroom_positive",
            "status": "pass" if best_delta > float(args.positive_headroom_db) else "fail",
            "value": best_delta,
            "threshold": f"> {args.positive_headroom_db} dB vs linear",
            "detail": f"best_config={best_config['config_label']}; best_step={best_config['best_step']}",
        },
    ]


def decision_from_gates(gates):
    headroom = next(row for row in gates if row["gate"] == "predictor_headroom_positive")
    metric_ok = next(row for row in gates if row["gate"] == "metric_rows_ok")
    if metric_ok["status"] != "pass":
        return "predictor_only_broader_validation_invalid_metrics"
    if headroom["status"] == "pass":
        return "predictor_only_broader_headroom_positive"
    return "predictor_only_broader_training_headroom_not_observed"


def write_report(package, config_rows, gates, path):
    lines = [
        "# Stage202 Predictor-Only Broader Validation",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Best config: `{package['best_config_label']}`.",
        f"- Best eval delta vs linear: `{package['best_delta_psnr_vs_linear']}` dB.",
        f"- Per-frame payload bytes: `0`.",
        "",
        "## Config Results",
        "",
        "| config | steps | lr | render weight | linear PSNR | best PSNR | final PSNR | best dPSNR | final dPSNR | best step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in config_rows:
        lines.append(
            f"| {row['config_label']} | {row['train_steps']} | {row['lr']} | {row['render_loss_weight']} | {float(row['linear_eval_psnr']):.6f} | {float(row['best_eval_psnr']):.6f} | {float(row['final_eval_psnr']):.6f} | {float(row['best_delta_psnr_vs_linear']):.6f} | {float(row['final_delta_psnr_vs_linear']):.6f} | {row['best_step']} |"
        )
    lines.extend([
        "",
        "## Gates",
        "",
        "| gate | status | value | threshold | detail |",
        "|---|---|---:|---|---|",
    ])
    for row in gates:
        lines.append(f"| {row['gate']} | {row['status']} | {row['value']} | {row['threshold']} | {row['detail']} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Stage202 remains predictor-only and transmits no residual/latent payload.",
        "- If `predictor_headroom_positive` fails, Stage203 should prioritize GS-native residual/latent side-info rather than spending more effort on selector training.",
        "- Target dense anchors and RGB remain training/evaluation labels only.",
        "",
        "## Outputs",
        "",
        f"- selected tasks: `{package['selected_tasks_csv']}`",
        f"- config results: `{package['config_csv']}`",
        f"- summary: `{package['summary_csv']}`",
        f"- per-task metrics: `{package['metrics_csv']}`",
        f"- train log: `{package['train_log_csv']}`",
        f"- gates: `{package['gates_csv']}`",
        f"- package: `{package['package_json']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--stage78_table", type=Path, default=DEFAULT_STAGE78)
    parser.add_argument("--gaps", nargs="+", type=int, default=[2, 4, 8, 12])
    parser.add_argument("--keyframe_codec", default="q12")
    parser.add_argument("--max_train_tasks", type=int, default=16)
    parser.add_argument("--max_eval_tasks", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--no_regression_tolerance_db", type=float, default=0.05)
    parser.add_argument("--positive_headroom_db", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    train_all = parse_task_rows(args.task_manifest, "train", args.gaps, args.keyframe_codec)
    eval_all = parse_task_rows(args.task_manifest, "eval", args.gaps, args.keyframe_codec)
    train_rows = balanced_sample(train_all, args.max_train_tasks, args.seed)
    eval_rows = balanced_sample(eval_all, args.max_eval_tasks, args.seed + 1000)
    if not train_rows or not eval_rows:
        raise RuntimeError(f"Need non-empty train/eval rows, got train={len(train_rows)} eval={len(eval_rows)}")
    old_adapter_reference = stage78_old_adapter_reference(args.stage78_table, [gap for gap in args.gaps if gap in {4, 8}])

    config_rows = []
    summary_rows = []
    metric_rows = []
    train_logs = []
    for config in CONFIGS:
        result, summaries, metrics, logs = run_config(config, train_rows, eval_rows, args, opt, background, device, old_adapter_reference)
        config_rows.append(result)
        summary_rows.extend(summaries)
        metric_rows.extend(metrics)
        train_logs.extend(logs)
    gates = gate_rows(config_rows, metric_rows, args)
    decision = decision_from_gates(gates)
    best = max(config_rows, key=lambda row: float(row["best_delta_psnr_vs_linear"]))

    selected_tasks_csv = args.summary_root / "stage202_selected_tasks.csv"
    config_csv = args.summary_root / "stage202_predictor_only_config_results.csv"
    summary_csv = args.summary_root / "stage202_predictor_only_summary.csv"
    metrics_csv = args.summary_root / "stage202_predictor_only_metrics.csv"
    train_log_csv = args.summary_root / "stage202_predictor_only_train_log.csv"
    gates_csv = args.summary_root / "stage202_predictor_only_gates.csv"
    package_json = args.summary_root / "stage202_predictor_only_broader_validation_package.json"
    report_md = args.summary_root / "stage202_predictor_only_broader_validation_report.md"
    write_csv(selected_task_rows(train_rows, eval_rows), selected_tasks_csv, SELECTED_TASK_FIELDS)
    write_csv(config_rows, config_csv, CONFIG_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(metric_rows, metrics_csv, METRIC_FIELDS_WITH_CONFIG)
    write_csv(train_logs, train_log_csv, TRAIN_LOG_FIELDS)
    write_csv(gates, gates_csv, GATE_FIELDS)
    package = {
        "stage": 202,
        "name": "predictor_only_broader_validation",
        "decision": decision,
        "task_manifest": str(args.task_manifest),
        "gaps": args.gaps,
        "keyframe_codec": args.keyframe_codec,
        "train_task_count": len(train_rows),
        "eval_task_count": len(eval_rows),
        "config_count": len(CONFIGS),
        "best_config_label": best["config_label"],
        "best_delta_psnr_vs_linear": best["best_delta_psnr_vs_linear"],
        "per_frame_payload_bytes": 0,
        "selected_tasks_csv": str(selected_tasks_csv),
        "config_csv": str(config_csv),
        "summary_csv": str(summary_csv),
        "metrics_csv": str(metrics_csv),
        "train_log_csv": str(train_log_csv),
        "gates_csv": str(gates_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "config_rows": config_rows,
        "gate_rows": gates,
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, config_rows, gates, report_md)
    print(json.dumps({"package": str(package_json), "decision": decision, "best_delta": best["best_delta_psnr_vs_linear"]}, indent=2))


if __name__ == "__main__":
    main()
