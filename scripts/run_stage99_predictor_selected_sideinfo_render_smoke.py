import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage99_predictor_selected_sideinfo_render_smoke"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import unflatten_static_anchor  # noqa: E402
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import load_rgb  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import DEFAULT_ADAPTER, load_adapter  # noqa: E402
from scripts.run_stage86_rendered_residual_sideinfo_smoke import render_psnr  # noqa: E402
from scripts.run_stage98_residual_importance_predictor_smoke import (  # noqa: E402
    DEFAULT_STAGE97_TASKS,
    build_task_tensors,
    collect_training_examples,
    evaluate as _unused_evaluate,
    make_features,
    parse_rows,
    topk_metrics,
    train_predictor,
)


ROW_FIELDS = [
    "stage97_task_id",
    "source_task_id",
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
    "base_psnr",
    "sideinfo_psnr",
    "teacher_oracle_sideinfo_psnr",
    "delta_psnr_vs_base",
    "gap_to_teacher_oracle_psnr",
]

SUMMARY_FIELDS = [
    "candidate",
    "base_method",
    "reference_gap",
    "task_count",
    "mean_precision_at_keep",
    "mean_energy_recall_total",
    "mean_relative_energy_recall_vs_oracle",
    "mean_base_psnr",
    "mean_sideinfo_psnr",
    "mean_teacher_oracle_sideinfo_psnr",
    "mean_delta_psnr_vs_base",
    "mean_gap_to_teacher_oracle_psnr",
]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def quantize_selected_residual_anchor(base_attrs, target_attrs, selected_idx, side_bits, eps=1e-8):
    residual = target_attrs - base_attrs
    selected_idx = selected_idx.to(device=base_attrs.device, dtype=torch.long)
    if selected_idx.numel() <= 0:
        return unflatten_static_anchor(base_attrs)
    kept = residual[0, selected_idx, :].detach().cpu().float()
    mins = kept.amin(dim=0)
    maxs = kept.amax(dim=0)
    mins_half = mins.numpy().astype("<f2")
    maxs_half = maxs.numpy().astype("<f2")
    mins_codec = torch.from_numpy(mins_half.astype("<f4")).float()
    maxs_codec = torch.from_numpy(maxs_half.astype("<f4")).float()
    qmax = (1 << int(side_bits)) - 1
    scales = (maxs_codec - mins_codec).clamp_min(eps) / qmax
    q = torch.round((kept - mins_codec) / scales).clamp(0, qmax)
    dequant = (q * scales + mins_codec).to(device=base_attrs.device, dtype=base_attrs.dtype)
    masked = torch.zeros_like(residual)
    masked[0, selected_idx, :] = dequant
    return unflatten_static_anchor(base_attrs + masked)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["candidate"], row["base_method"], row["reference_gap"])].append(row)
    out = []
    for (candidate, method, gap), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], int(item[0][2]))):
        def avg(key):
            return sum(float(row[key]) for row in items) / len(items)
        out.append({
            "candidate": candidate,
            "base_method": method,
            "reference_gap": int(gap),
            "task_count": len(items),
            "mean_precision_at_keep": avg("precision_at_keep"),
            "mean_energy_recall_total": avg("energy_recall_total"),
            "mean_relative_energy_recall_vs_oracle": avg("relative_energy_recall_vs_oracle"),
            "mean_base_psnr": avg("base_psnr"),
            "mean_sideinfo_psnr": avg("sideinfo_psnr"),
            "mean_teacher_oracle_sideinfo_psnr": avg("teacher_oracle_sideinfo_psnr"),
            "mean_delta_psnr_vs_base": avg("delta_psnr_vs_base"),
            "mean_gap_to_teacher_oracle_psnr": avg("gap_to_teacher_oracle_psnr"),
        })
    return out


def write_report(summary, summary_rows, path):
    lines = [
        "# Stage99 Predictor-Selected Side-Info Rendered Smoke",
        "",
        "## Configuration",
        "",
        f"- train tasks: `{summary['train_task_count']}`",
        f"- eval tasks: `{summary['eval_task_count']}`",
        f"- train examples: `{summary['train_example_count']}`",
        f"- keep fraction: `{summary['keep_fraction']}`",
        f"- side bits: `{summary['side_bits']}`",
        "- predicted indices use teacher residual values to isolate selection error",
        "",
        "## Summary",
        "",
        "| candidate | base | gap | tasks | precision | energy recall | relative recall | side PSNR | teacher PSNR | delta | gap to teacher |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['candidate']} | {row['base_method']} | {row['reference_gap']} | {row['task_count']} | {row['mean_precision_at_keep']} | {row['mean_energy_recall_total']} | {row['mean_relative_energy_recall_vs_oracle']} | {row['mean_sideinfo_psnr']} | {row['mean_teacher_oracle_sideinfo_psnr']} | {row['mean_delta_psnr_vs_base']} | {row['mean_gap_to_teacher_oracle_psnr']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `teacher_oracle_topk` is not deployable and only bounds top10 index selection.",
        "- `mlp_importance` uses decoder-available features for indices, but still uses teacher residual values at those indices.",
        "- This stage isolates selection error; residual value prediction remains unresolved.",
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
    parser.add_argument("--side_bits", type=int, default=6)
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
    device = torch.device(args.device)
    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    adapter = load_adapter(args.adapter, hidden_dim=256, device=device)
    train_tasks = parse_rows(args.tasks_csv, "train", args.gaps, args.max_train_tasks, args.seed)
    eval_tasks = parse_rows(args.tasks_csv, "eval", args.gaps, args.max_eval_tasks, args.seed + 1)
    train_x, train_y = collect_training_examples(train_tasks, adapter, args, device)
    predictor, mean, std, train_logs = train_predictor(train_x, train_y, args, device)

    rows = []
    cache = {}
    method_ids = {"linear": 0.0, "stage65_adapter": 1.0}
    with torch.no_grad():
        for task in eval_tasks:
            left_attrs, right_attrs, target_attrs, base_attrs_by_method = build_task_tensors(task, adapter, device, cache)
            target_rgb = load_rgb(task["target_rgb_path"], opt.image_height, opt.image_width, device)
            endpoint_score = torch.sum((right_attrs[0] - left_attrs[0]) ** 2, dim=-1)
            for method in args.base_methods:
                base_attrs = base_attrs_by_method[method]
                base_psnr = render_psnr(unflatten_static_anchor(base_attrs), target_rgb, background, opt)
                residual = target_attrs - base_attrs
                energy = torch.sum(residual[0] ** 2, dim=-1)
                keep_count = max(1, int(round(energy.numel() * args.keep_fraction)))
                teacher_idx = torch.topk(energy, k=keep_count, largest=True).indices
                features = make_features(left_attrs, right_attrs, base_attrs, task["normalized_time"], method_ids[method])
                logits = []
                for start in range(0, features.shape[0], args.eval_batch_size):
                    chunk = features[start:start + args.eval_batch_size]
                    logits.append(predictor((chunk - mean) / std))
                mlp_score = torch.cat(logits, dim=0)
                candidate_indices = {
                    "teacher_oracle_topk": teacher_idx,
                    "mlp_importance": torch.topk(mlp_score, k=keep_count, largest=True).indices,
                    "endpoint_diff_baseline": torch.topk(endpoint_score, k=keep_count, largest=True).indices,
                }
                teacher_anchor = quantize_selected_residual_anchor(base_attrs, target_attrs, teacher_idx, args.side_bits)
                teacher_psnr = render_psnr(teacher_anchor, target_rgb, background, opt)
                for candidate, selected_idx in candidate_indices.items():
                    side_anchor = teacher_anchor if candidate == "teacher_oracle_topk" else quantize_selected_residual_anchor(base_attrs, target_attrs, selected_idx, args.side_bits)
                    side_psnr = teacher_psnr if candidate == "teacher_oracle_topk" else render_psnr(side_anchor, target_rgb, background, opt)
                    score_for_metrics = energy if candidate == "teacher_oracle_topk" else (mlp_score if candidate == "mlp_importance" else endpoint_score)
                    metrics = topk_metrics(score_for_metrics, energy, keep_count)
                    rows.append({
                        "stage97_task_id": task["stage97_task_id"],
                        "source_task_id": task["source_task_id"],
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
                        "base_psnr": base_psnr,
                        "sideinfo_psnr": side_psnr,
                        "teacher_oracle_sideinfo_psnr": teacher_psnr,
                        "delta_psnr_vs_base": side_psnr - base_psnr,
                        "gap_to_teacher_oracle_psnr": side_psnr - teacher_psnr,
                    })
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_rows = summarize(rows)
    rows_csv = args.summary_root / "stage99_predictor_selected_sideinfo_rows.csv"
    summary_csv = args.summary_root / "stage99_predictor_selected_sideinfo_summary.csv"
    train_log_csv = args.summary_root / "stage99_predictor_selected_sideinfo_train_log.csv"
    summary_json = args.summary_root / "stage99_predictor_selected_sideinfo_summary.json"
    report_md = args.summary_root / "stage99_predictor_selected_sideinfo_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(train_logs, train_log_csv, ["step", "loss", "batch_positive_rate"])
    summary = {
        "stage": 99,
        "mode": "predictor-selected side-info rendered smoke",
        "tasks_csv": str(args.tasks_csv),
        "adapter": str(args.adapter),
        "train_task_count": len(train_tasks),
        "eval_task_count": len(eval_tasks),
        "train_example_count": int(train_x.shape[0]),
        "keep_fraction": args.keep_fraction,
        "side_bits": args.side_bits,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "train_log_csv": str(train_log_csv),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "notes": [
            "Predicted indices are feed-forward, but residual values are teacher-derived in this smoke.",
            "This isolates selection error from residual value prediction error.",
            "No model checkpoint or heavy tensor is saved.",
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
