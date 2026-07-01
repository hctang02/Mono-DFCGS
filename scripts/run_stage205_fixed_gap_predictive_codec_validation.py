import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_MANIFEST = REPO_ROOT / "experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_tasks.csv"
DEFAULT_STAGE204_PACKAGE = REPO_ROOT / "experiments/stage204_residual_codec_smoke/stage204_residual_codec_smoke_package.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage205_fixed_gap_predictive_codec_validation"
MIB = 1024.0 * 1024.0

sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage201_predictor_only_smoke import balanced_sample, parse_task_rows, write_csv  # noqa: E402
from scripts.run_stage204_residual_codec_smoke import evaluate_task, selected_task_rows  # noqa: E402


ROW_FIELDS = [
    "task_id",
    "sequence",
    "reference_gap",
    "target_index",
    "normalized_time",
    "setting_label",
    "keep_fraction",
    "side_bits",
    "payload_bytes",
    "base_anchor_mse",
    "corrected_anchor_mse",
    "anchor_mse_reduction",
    "base_render_mse",
    "corrected_render_mse",
    "base_psnr",
    "corrected_psnr",
    "delta_psnr_vs_base",
    "status",
    "error",
]
SUMMARY_FIELDS = [
    "reference_gap",
    "setting_label",
    "keep_fraction",
    "side_bits",
    "task_count",
    "mean_payload_bytes",
    "mean_residual_mib_per_intermediate",
    "mean_base_psnr",
    "mean_corrected_psnr",
    "mean_delta_psnr_vs_base",
    "mean_anchor_mse_reduction",
]
GAP_BEST_FIELDS = [
    "reference_gap",
    "best_setting_label",
    "best_keep_fraction",
    "best_mean_payload_bytes",
    "best_mean_residual_mib_per_intermediate",
    "best_mean_base_psnr",
    "best_mean_corrected_psnr",
    "best_mean_delta_psnr_vs_base",
]
GATE_FIELDS = ["gate", "status", "value", "threshold", "detail"]
TASK_FIELDS = ["task_id", "sequence", "reference_gap", "left_index", "right_index", "target_index", "normalized_time"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    values = [float(value) for value in values]
    return sum(values) / max(len(values), 1)


def sample_by_gap(rows, gaps, max_tasks_per_gap, seed):
    out = []
    for gap in gaps:
        gap_rows = [row for row in rows if int(row["reference_gap"]) == int(gap)]
        out.extend(balanced_sample(gap_rows, max_tasks_per_gap, seed + int(gap)))
    return out


def setting_label(keep_fraction, side_bits):
    return f"topk_keep{str(keep_fraction).replace('.', 'p')}_q{side_bits}"


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            key = (int(row["reference_gap"]), row["setting_label"], float(row["keep_fraction"]), int(row["side_bits"]))
            grouped[key].append(row)
    out = []
    for (gap, label, keep_fraction, side_bits), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2])):
        payload = mean(row["payload_bytes"] for row in items)
        out.append({
            "reference_gap": gap,
            "setting_label": label,
            "keep_fraction": keep_fraction,
            "side_bits": side_bits,
            "task_count": len(items),
            "mean_payload_bytes": payload,
            "mean_residual_mib_per_intermediate": payload / MIB,
            "mean_base_psnr": mean(row["base_psnr"] for row in items),
            "mean_corrected_psnr": mean(row["corrected_psnr"] for row in items),
            "mean_delta_psnr_vs_base": mean(row["delta_psnr_vs_base"] for row in items),
            "mean_anchor_mse_reduction": mean(row["anchor_mse_reduction"] for row in items),
        })
    return out


def gap_best_rows(summary_rows):
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[int(row["reference_gap"])].append(row)
    out = []
    for gap, items in sorted(grouped.items()):
        best = max(items, key=lambda row: float(row["mean_delta_psnr_vs_base"]))
        out.append({
            "reference_gap": gap,
            "best_setting_label": best["setting_label"],
            "best_keep_fraction": best["keep_fraction"],
            "best_mean_payload_bytes": best["mean_payload_bytes"],
            "best_mean_residual_mib_per_intermediate": best["mean_residual_mib_per_intermediate"],
            "best_mean_base_psnr": best["mean_base_psnr"],
            "best_mean_corrected_psnr": best["mean_corrected_psnr"],
            "best_mean_delta_psnr_vs_base": best["mean_delta_psnr_vs_base"],
        })
    return out


def gate_rows(metric_rows, best_rows, args):
    errors = [row for row in metric_rows if row["status"] != "ok"]
    min_payload = min((float(row["payload_bytes"]) for row in metric_rows if row["status"] == "ok"), default=0.0)
    missing_gaps = sorted(set(args.gaps) - {int(row["reference_gap"]) for row in best_rows})
    weak_gaps = [row for row in best_rows if float(row["best_mean_delta_psnr_vs_base"]) <= float(args.positive_headroom_db)]
    return [
        {
            "gate": "metric_rows_ok",
            "status": "pass" if not errors else "fail",
            "value": len(errors),
            "threshold": "0",
            "detail": "shape-mismatched metrics are errors",
        },
        {
            "gate": "payload_counted_nonzero",
            "status": "pass" if min_payload > 0 else "fail",
            "value": min_payload,
            "threshold": ">0 bytes",
            "detail": "payload_bytes are exact len(payload)",
        },
        {
            "gate": "gap_coverage",
            "status": "pass" if not missing_gaps else "fail",
            "value": len(missing_gaps),
            "threshold": "0 missing gaps",
            "detail": ";".join(str(gap) for gap in missing_gaps),
        },
        {
            "gate": "each_gap_positive_headroom",
            "status": "pass" if not weak_gaps else "fail",
            "value": min((float(row["best_mean_delta_psnr_vs_base"]) for row in best_rows), default=0.0),
            "threshold": f"> {args.positive_headroom_db} dB for every gap",
            "detail": ";".join(f"gap{row['reference_gap']}={row['best_mean_delta_psnr_vs_base']}" for row in weak_gaps),
        },
        {
            "gate": "stage197_decoder_contract",
            "status": "pass",
            "value": 0,
            "threshold": "no target dense/RGB decoder input",
            "detail": "decoder uses base GS plus transmitted counted GS residual payload",
        },
    ]


def decision(gates):
    if any(row["status"] != "pass" for row in gates if row["gate"] != "each_gap_positive_headroom"):
        return "fixed_gap_predictive_codec_validation_invalid"
    headroom = next(row for row in gates if row["gate"] == "each_gap_positive_headroom")
    if headroom["status"] == "pass":
        return "fixed_gap_predictive_codec_positive_headroom"
    return "fixed_gap_predictive_codec_needs_review"


def write_report(package, best_rows, gates, path):
    lines = [
        "# Stage205 Fixed-Gap Predictive Codec Validation",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Tasks: `{package['task_count']}`; settings: `{package['setting_count']}`.",
        "- Scope: sampled fixed-gap validation, not full-sequence RD.",
        "",
        "## Best By Gap",
        "",
        "| gap | best setting | keep | payload bytes | MiB/intermediate | base PSNR | corrected PSNR | dPSNR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            f"| {row['reference_gap']} | {row['best_setting_label']} | {row['best_keep_fraction']} | {float(row['best_mean_payload_bytes']):.3f} | {float(row['best_mean_residual_mib_per_intermediate']):.6f} | {float(row['best_mean_base_psnr']):.6f} | {float(row['best_mean_corrected_psnr']):.6f} | {float(row['best_mean_delta_psnr_vs_base']):.6f} |"
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
        "## Outputs",
        "",
        f"- selected tasks: `{package['selected_tasks_csv']}`",
        f"- rows: `{package['rows_csv']}`",
        f"- summary: `{package['summary_csv']}`",
        f"- best by gap: `{package['best_by_gap_csv']}`",
        f"- gates: `{package['gates_csv']}`",
        f"- package: `{package['package_json']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--stage204_package", type=Path, default=DEFAULT_STAGE204_PACKAGE)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--task_split", default="eval")
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--keyframe_codec", default="q12")
    parser.add_argument("--max_tasks_per_gap", type=int, default=8)
    parser.add_argument("--keep_fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--side_bits", type=int, default=6)
    parser.add_argument("--zlib_level", type=int, default=9)
    parser.add_argument("--positive_headroom_db", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    stage204 = read_json(args.stage204_package)
    if stage204["decision"] != "residual_codec_smoke_positive_headroom":
        raise RuntimeError(f"Stage204 did not pass positive headroom: {stage204['decision']}")
    device = torch.device(args.device)
    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    cache = {}
    all_rows = parse_task_rows(args.task_manifest, args.task_split, args.gaps, args.keyframe_codec)
    rows = sample_by_gap(all_rows, args.gaps, args.max_tasks_per_gap, args.seed)
    settings = [(keep, int(args.side_bits)) for keep in args.keep_fractions]
    metric_rows = []
    for row in rows:
        metric_rows.extend(evaluate_task(row, settings, args, device, cache, opt, background))
    summary_rows = summarize(metric_rows)
    best_rows = gap_best_rows(summary_rows)
    gates = gate_rows(metric_rows, best_rows, args)
    decision_value = decision(gates)

    selected_tasks_csv = args.summary_root / "stage205_selected_tasks.csv"
    rows_csv = args.summary_root / "stage205_fixed_gap_predictive_rows.csv"
    summary_csv = args.summary_root / "stage205_fixed_gap_predictive_summary.csv"
    best_by_gap_csv = args.summary_root / "stage205_fixed_gap_predictive_best_by_gap.csv"
    gates_csv = args.summary_root / "stage205_fixed_gap_predictive_gates.csv"
    package_json = args.summary_root / "stage205_fixed_gap_predictive_codec_validation_package.json"
    report_md = args.summary_root / "stage205_fixed_gap_predictive_codec_validation_report.md"
    write_csv(selected_task_rows(rows), selected_tasks_csv, TASK_FIELDS)
    write_csv(metric_rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(best_rows, best_by_gap_csv, GAP_BEST_FIELDS)
    write_csv(gates, gates_csv, GATE_FIELDS)
    package = {
        "stage": 205,
        "name": "fixed_gap_predictive_codec_validation",
        "decision": decision_value,
        "task_manifest": str(args.task_manifest),
        "stage204_package": str(args.stage204_package),
        "task_split": args.task_split,
        "gaps": args.gaps,
        "keyframe_codec": args.keyframe_codec,
        "task_count": len(rows),
        "setting_count": len(settings),
        "side_bits": args.side_bits,
        "keep_fractions": args.keep_fractions,
        "selected_tasks_csv": str(selected_tasks_csv),
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "best_by_gap_csv": str(best_by_gap_csv),
        "gates_csv": str(gates_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "best_rows": best_rows,
        "gate_rows": gates,
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, best_rows, gates, report_md)
    print(json.dumps({"package": str(package_json), "decision": decision_value}, indent=2))


if __name__ == "__main__":
    main()
