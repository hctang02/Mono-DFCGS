import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE143_SUMMARY = REPO_ROOT / "experiments/stage143_middle_frame_psnr_collapse_diagnostic/stage143_middle_frame_psnr_collapse_summary.csv"
DEFAULT_STAGE143_PACKAGE = REPO_ROOT / "experiments/stage143_middle_frame_psnr_collapse_diagnostic/stage143_middle_frame_psnr_collapse_package.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage144_high_rate_middle_frame_upper_bound"

UPPER_FIELDS = [
    "reference_gap",
    "target_middle_psnr",
    "q12_adapter_middle_psnr",
    "q16_adapter_middle_psnr",
    "float32_adapter_middle_psnr",
    "q12_dense_direct_middle_psnr",
    "q16_dense_direct_middle_psnr",
    "float32_dense_direct_middle_psnr",
    "q16_minus_q12_adapter_middle_psnr",
    "float32_minus_q12_adapter_middle_psnr",
    "float32_dense_minus_float32_adapter_middle_psnr",
    "float32_adapter_gap_to_target",
    "float32_dense_gap_to_target",
    "decision",
]

DECISION_FIELDS = ["item", "decision", "evidence"]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lookup(rows, codec, gap, method):
    for row in rows:
        if row["codec"] == codec and int(float(row["reference_gap"])) == int(gap) and row["method"] == method:
            return row
    raise KeyError(f"missing {codec} gap{gap} {method}")


def maybe_float(value):
    return None if value in (None, "") else float(value)


def build_upper_rows(rows):
    out = []
    for gap in [4, 8]:
        q12_adapter = lookup(rows, "q12", gap, "adapter")
        q16_adapter = lookup(rows, "q16", gap, "adapter")
        f32_adapter = lookup(rows, "float32", gap, "adapter")
        q12_dense = lookup(rows, "q12", gap, "dense_direct")
        q16_dense = lookup(rows, "q16", gap, "dense_direct")
        f32_dense = lookup(rows, "float32", gap, "dense_direct")
        target = maybe_float(f32_adapter["gap_to_stage75_middle_target"])
        if target is None:
            target_middle = None
        else:
            target_middle = float(f32_adapter["mean_middle_psnr"]) - target
        q16_minus_q12 = float(q16_adapter["mean_middle_psnr"]) - float(q12_adapter["mean_middle_psnr"])
        f32_minus_q12 = float(f32_adapter["mean_middle_psnr"]) - float(q12_adapter["mean_middle_psnr"])
        dense_minus_adapter = float(f32_dense["mean_middle_psnr"]) - float(f32_adapter["mean_middle_psnr"])
        adapter_gap_to_target = maybe_float(f32_adapter["gap_to_stage75_middle_target"])
        dense_gap_to_target = maybe_float(f32_dense["gap_to_stage75_middle_target"])
        decision = "model_training_or_sideinfo_required"
        if abs(f32_minus_q12) > 0.5:
            decision = "quantization_may_be_material"
        out.append({
            "reference_gap": gap,
            "target_middle_psnr": target_middle,
            "q12_adapter_middle_psnr": float(q12_adapter["mean_middle_psnr"]),
            "q16_adapter_middle_psnr": float(q16_adapter["mean_middle_psnr"]),
            "float32_adapter_middle_psnr": float(f32_adapter["mean_middle_psnr"]),
            "q12_dense_direct_middle_psnr": float(q12_dense["mean_middle_psnr"]),
            "q16_dense_direct_middle_psnr": float(q16_dense["mean_middle_psnr"]),
            "float32_dense_direct_middle_psnr": float(f32_dense["mean_middle_psnr"]),
            "q16_minus_q12_adapter_middle_psnr": q16_minus_q12,
            "float32_minus_q12_adapter_middle_psnr": f32_minus_q12,
            "float32_dense_minus_float32_adapter_middle_psnr": dense_minus_adapter,
            "float32_adapter_gap_to_target": adapter_gap_to_target,
            "float32_dense_gap_to_target": dense_gap_to_target,
            "decision": decision,
        })
    return out


def build_decisions(upper_rows):
    max_quant_gain = max(abs(row["float32_minus_q12_adapter_middle_psnr"]) for row in upper_rows)
    min_adapter_gap = min(row["float32_adapter_gap_to_target"] for row in upper_rows)
    min_dense_margin = min(row["float32_dense_gap_to_target"] for row in upper_rows)
    min_model_gap = min(row["float32_dense_minus_float32_adapter_middle_psnr"] for row in upper_rows)
    return [
        {
            "item": "raise_anchor_qbit",
            "decision": "reject_as_primary_fix",
            "evidence": f"max abs(float32-q12 adapter middle) = {max_quant_gain} dB",
        },
        {
            "item": "renderer_data_ceiling",
            "decision": "not_bottleneck",
            "evidence": f"min float32 dense gap to target = {min_dense_margin} dB",
        },
        {
            "item": "dynamic_model",
            "decision": "primary_bottleneck",
            "evidence": f"min dense-direct minus adapter middle = {min_model_gap} dB; best float32 adapter target gap = {min_adapter_gap} dB",
        },
        {
            "item": "next_stage",
            "decision": "start_large_scale_adapter_training_and_prepare_rate_counted_sideinfo_fallback",
            "evidence": "High-rate anchors do not move adapter middle PSNR; dense-direct has ample ceiling.",
        },
    ]


def format_float(value):
    return "" if value is None else f"{float(value):.6f}"


def write_report(upper_rows, decisions, package, path):
    lines = [
        "# Stage144 High-Rate Middle-Frame Upper Bound",
        "",
        "## Upper-Bound Table",
        "",
        "| gap | target | q12 adapter | q16 adapter | float32 adapter | float32 dense | float32-q12 adapter | dense-adapter | adapter-target | decision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in upper_rows:
        lines.append(
            f"| {row['reference_gap']} | {format_float(row['target_middle_psnr'])} | {format_float(row['q12_adapter_middle_psnr'])} | {format_float(row['q16_adapter_middle_psnr'])} | {format_float(row['float32_adapter_middle_psnr'])} | {format_float(row['float32_dense_direct_middle_psnr'])} | {format_float(row['float32_minus_q12_adapter_middle_psnr'])} | {format_float(row['float32_dense_minus_float32_adapter_middle_psnr'])} | {format_float(row['float32_adapter_gap_to_target'])} | {row['decision']} |"
        )
    lines.extend([
        "",
        "## Decisions",
        "",
        "| item | decision | evidence |",
        "|---|---|---|",
    ])
    for row in decisions:
        lines.append(f"| {row['item']} | {row['decision']} | {row['evidence']} |")
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- upper-bound CSV: `{package['upper_bound_csv']}`",
        f"- decisions CSV: `{package['decisions_csv']}`",
        f"- summary JSON: `{package['summary_json']}`",
        f"- package JSON: `{package['package_json']}`",
        f"- report Markdown: `{package['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage143_summary", type=Path, default=DEFAULT_STAGE143_SUMMARY)
    parser.add_argument("--stage143_package", type=Path, default=DEFAULT_STAGE143_PACKAGE)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage143_package = read_json(args.stage143_package)
    upper_rows = build_upper_rows(read_csv(args.stage143_summary))
    decisions = build_decisions(upper_rows)
    upper_csv = args.summary_root / "stage144_high_rate_middle_frame_upper_bound_rows.csv"
    decisions_csv = args.summary_root / "stage144_high_rate_middle_frame_upper_bound_decisions.csv"
    summary_json = args.summary_root / "stage144_high_rate_middle_frame_upper_bound_summary.json"
    package_json = args.summary_root / "stage144_high_rate_middle_frame_upper_bound_package.json"
    report_md = args.summary_root / "stage144_high_rate_middle_frame_upper_bound_report.md"
    write_csv(upper_rows, upper_csv, UPPER_FIELDS)
    write_csv(decisions, decisions_csv, DECISION_FIELDS)
    summary = {
        "stage": 144,
        "mode": "high-rate middle-frame upper-bound decision package",
        "stage143_summary": str(args.stage143_summary),
        "stage143_package": str(args.stage143_package),
        "stage143_row_count": stage143_package["row_count"],
        "upper_bound_rows": upper_rows,
        "decisions": decisions,
        "upper_bound_csv": str(upper_csv),
        "decisions_csv": str(decisions_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "first_phase_decision": "quality collapse is dynamic-model-side; proceed to large-scale adapter training and keep side-info fallback",
    }
    package = {
        "stage": 144,
        "mode": "high-rate middle-frame upper-bound decision package",
        "upper_bound_csv": str(upper_csv),
        "decisions_csv": str(decisions_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "first_phase_decision": summary["first_phase_decision"],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(upper_rows, decisions, package, report_md)
    print(json.dumps({"package": str(package_json), "first_phase_decision": summary["first_phase_decision"]}, indent=2))


if __name__ == "__main__":
    main()
