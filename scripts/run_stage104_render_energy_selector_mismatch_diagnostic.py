import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE103_ROWS = REPO_ROOT / "experiments/stage103_broader_rendered_selector_validation/stage103_broader_rendered_selector_rows.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage104_render_energy_selector_mismatch_diagnostic"


ROW_FIELDS = [
    "stage97_task_id",
    "source_task_id",
    "sequence",
    "reference_gap",
    "target_index",
    "base_method",
    "candidate",
    "endpoint_energy_recall",
    "candidate_energy_recall",
    "energy_recall_delta",
    "endpoint_precision",
    "candidate_precision",
    "precision_delta",
    "endpoint_sideinfo_psnr",
    "candidate_sideinfo_psnr",
    "sideinfo_psnr_delta",
    "endpoint_delta_psnr_vs_base",
    "candidate_delta_psnr_vs_base",
    "delta_psnr_vs_base_delta",
    "endpoint_gap_to_teacher",
    "candidate_gap_to_teacher",
    "gap_to_teacher_delta",
    "teacher_oracle_sideinfo_psnr",
    "energy_up",
    "psnr_up",
    "energy_up_psnr_down",
]

SUMMARY_FIELDS = [
    "candidate",
    "base_method",
    "reference_gap",
    "task_count",
    "mean_energy_recall_delta",
    "mean_precision_delta",
    "mean_sideinfo_psnr_delta",
    "mean_delta_psnr_vs_base_delta",
    "mean_gap_to_teacher_delta",
    "energy_up_count",
    "psnr_up_count",
    "both_up_count",
    "energy_up_psnr_down_count",
    "energy_down_psnr_up_count",
    "mean_psnr_delta_when_energy_up",
    "corr_energy_delta_psnr_delta",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def f(row, key):
    return float(row[key])


def row_key(row):
    return (
        row["stage97_task_id"],
        row["source_task_id"],
        row["base_method"],
        row["reference_gap"],
        row["target_index"],
    )


def corr(xs, ys):
    if len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-24 or vy <= 1e-24:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def build_rows(stage103_rows, candidates):
    grouped = defaultdict(dict)
    for row in stage103_rows:
        grouped[row_key(row)][row["candidate"]] = row
    out = []
    for _, items in sorted(grouped.items()):
        endpoint = items.get("endpoint_diff_baseline")
        if endpoint is None:
            continue
        for candidate in candidates:
            cand = items.get(candidate)
            if cand is None:
                continue
            energy_delta = f(cand, "energy_recall_total") - f(endpoint, "energy_recall_total")
            precision_delta = f(cand, "precision_at_keep") - f(endpoint, "precision_at_keep")
            psnr_delta = f(cand, "sideinfo_psnr") - f(endpoint, "sideinfo_psnr")
            delta_delta = f(cand, "delta_psnr_vs_base") - f(endpoint, "delta_psnr_vs_base")
            gap_delta = f(cand, "gap_to_teacher_oracle_psnr") - f(endpoint, "gap_to_teacher_oracle_psnr")
            energy_up = energy_delta > 0.0
            psnr_up = psnr_delta > 0.0
            out.append({
                "stage97_task_id": cand["stage97_task_id"],
                "source_task_id": cand["source_task_id"],
                "sequence": cand["sequence"],
                "reference_gap": int(cand["reference_gap"]),
                "target_index": int(cand["target_index"]),
                "base_method": cand["base_method"],
                "candidate": candidate,
                "endpoint_energy_recall": f(endpoint, "energy_recall_total"),
                "candidate_energy_recall": f(cand, "energy_recall_total"),
                "energy_recall_delta": energy_delta,
                "endpoint_precision": f(endpoint, "precision_at_keep"),
                "candidate_precision": f(cand, "precision_at_keep"),
                "precision_delta": precision_delta,
                "endpoint_sideinfo_psnr": f(endpoint, "sideinfo_psnr"),
                "candidate_sideinfo_psnr": f(cand, "sideinfo_psnr"),
                "sideinfo_psnr_delta": psnr_delta,
                "endpoint_delta_psnr_vs_base": f(endpoint, "delta_psnr_vs_base"),
                "candidate_delta_psnr_vs_base": f(cand, "delta_psnr_vs_base"),
                "delta_psnr_vs_base_delta": delta_delta,
                "endpoint_gap_to_teacher": f(endpoint, "gap_to_teacher_oracle_psnr"),
                "candidate_gap_to_teacher": f(cand, "gap_to_teacher_oracle_psnr"),
                "gap_to_teacher_delta": gap_delta,
                "teacher_oracle_sideinfo_psnr": f(cand, "teacher_oracle_sideinfo_psnr"),
                "energy_up": int(energy_up),
                "psnr_up": int(psnr_up),
                "energy_up_psnr_down": int(energy_up and not psnr_up),
            })
    return out


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["candidate"], row["base_method"], row["reference_gap"])].append(row)
    out = []
    for (candidate, base_method, gap), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], int(item[0][2]))):
        def avg(key):
            return sum(float(row[key]) for row in items) / len(items)
        energy_up_items = [row for row in items if int(row["energy_up"]) == 1]
        out.append({
            "candidate": candidate,
            "base_method": base_method,
            "reference_gap": int(gap),
            "task_count": len(items),
            "mean_energy_recall_delta": avg("energy_recall_delta"),
            "mean_precision_delta": avg("precision_delta"),
            "mean_sideinfo_psnr_delta": avg("sideinfo_psnr_delta"),
            "mean_delta_psnr_vs_base_delta": avg("delta_psnr_vs_base_delta"),
            "mean_gap_to_teacher_delta": avg("gap_to_teacher_delta"),
            "energy_up_count": sum(int(row["energy_up"]) for row in items),
            "psnr_up_count": sum(int(row["psnr_up"]) for row in items),
            "both_up_count": sum(1 for row in items if int(row["energy_up"]) == 1 and int(row["psnr_up"]) == 1),
            "energy_up_psnr_down_count": sum(int(row["energy_up_psnr_down"]) for row in items),
            "energy_down_psnr_up_count": sum(1 for row in items if int(row["energy_up"]) == 0 and int(row["psnr_up"]) == 1),
            "mean_psnr_delta_when_energy_up": sum(float(row["sideinfo_psnr_delta"]) for row in energy_up_items) / len(energy_up_items) if energy_up_items else 0.0,
            "corr_energy_delta_psnr_delta": corr([float(row["energy_recall_delta"]) for row in items], [float(row["sideinfo_psnr_delta"]) for row in items]),
        })
    return out


def write_report(summary, summary_rows, path):
    lines = [
        f"# {summary.get('report_title', 'Stage104 Render-Energy Selector Mismatch Diagnostic')}",
        "",
        "## Configuration",
        "",
        f"- input rows: `{summary['input_rows']}`",
        f"- compared candidates: `{', '.join(summary['candidates'])}`",
        "- reference candidate: `endpoint_diff_baseline`",
        "- no rendering or heavy tensor output",
        "",
        "## Summary",
        "",
        "| candidate | base | gap | tasks | energy delta | psnr delta | energy up | psnr up | both up | energy up psnr down | corr |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['candidate']} | {row['base_method']} | {row['reference_gap']} | {row['task_count']} | {row['mean_energy_recall_delta']} | {row['mean_sideinfo_psnr_delta']} | {row['energy_up_count']} | {row['psnr_up_count']} | {row['both_up_count']} | {row['energy_up_psnr_down_count']} | {row['corr_energy_delta_psnr_delta']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Positive `energy delta` means the learned selector captured more residual energy than endpoint selection.",
        "- Positive `psnr delta` means rendered PSNR improved over endpoint selection.",
        "- `energy up psnr down` is the key mismatch count for render-aware selector diagnosis.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage103_rows", type=Path, default=DEFAULT_STAGE103_ROWS)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--stage", type=int, default=104)
    parser.add_argument("--mode", default="render-energy selector mismatch diagnostic")
    parser.add_argument("--output_prefix", default="stage104_render_energy_mismatch")
    parser.add_argument("--report_title", default="Stage104 Render-Energy Selector Mismatch Diagnostic")
    parser.add_argument("--candidates", nargs="+", default=["shared_energy_regression", "shared_topk_bce"])
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage103_rows = read_csv(args.stage103_rows)
    diagnostic_rows = build_rows(stage103_rows, args.candidates)
    summary_rows = summarize(diagnostic_rows)
    rows_csv = args.summary_root / f"{args.output_prefix}_rows.csv"
    summary_csv = args.summary_root / f"{args.output_prefix}_summary.csv"
    summary_json = args.summary_root / f"{args.output_prefix}_summary.json"
    report_md = args.summary_root / f"{args.output_prefix}_report.md"
    write_csv(diagnostic_rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    summary = {
        "stage": args.stage,
        "mode": args.mode,
        "report_title": args.report_title,
        "input_rows": str(args.stage103_rows),
        "candidates": args.candidates,
        "diagnostic_row_count": len(diagnostic_rows),
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "notes": [
            "No rendering or heavy tensor output is produced.",
            "Reference candidate is endpoint_diff_baseline.",
            "Energy-up PSNR-down rows indicate that residual-energy recall is not render-aligned.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, summary_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "diagnostic_row_count": len(diagnostic_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
