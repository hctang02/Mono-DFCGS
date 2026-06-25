import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage43_selector_evidence_synthesis"
STAGE26_SUMMARY = REPO_ROOT / "experiments/stage26_leave_one_out_full_video_rd/stage26_leave_one_out_full_video_rd_summary.json"
STAGE36_ZLIB_COMPARISON = REPO_ROOT / "experiments/stage36_dense_oracle_actual_bitstream_rd/stage36_zlib_selector_comparison.csv"
STAGE39_SUMMARY = REPO_ROOT / "experiments/stage39_predicted_selector_rd/stage39_predicted_selector_rd_summary.json"
STAGE41_SUMMARY = REPO_ROOT / "experiments/stage41_normalized_predicted_selector_rd/stage41_normalized_predicted_selector_rd_summary.json"
STAGE42_SUMMARY = REPO_ROOT / "experiments/stage42_calibrated_selector_prior_rd/stage42_calibrated_selector_prior_rd_summary.json"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean_float(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def positives(rows, key):
    return sum(1 for row in rows if float(row[key]) > 0.0)


def write_csv(rows, path):
    fields = [
        "stage", "evidence", "deployable_selector", "actual_bitstream", "points", "positive_all_points",
        "positive_middle_points", "mean_delta_all_psnr", "mean_delta_middle_psnr", "mean_rate_delta_mib_per_frame", "interpretation",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    lines = [
        "| stage | evidence | deployable selector | actual bitstream | points | +all | mean Δall PSNR | mean Δmiddle PSNR | interpretation |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['evidence']} | {row['deployable_selector']} | {row['actual_bitstream']} | "
            f"{row['points']} | {row['positive_all_points']} | {row['mean_delta_all_psnr']:.6f} | "
            f"{row['mean_delta_middle_psnr']:.6f} | {row['interpretation']} |"
        )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    rows = []

    stage26 = load_json(STAGE26_SUMMARY)
    s26_rows = stage26["rows"]
    rows.append({
        "stage": 26,
        "evidence": "leave-one-out anchor adapter vs linear interpolation",
        "deployable_selector": "uniform only",
        "actual_bitstream": "estimated q8 rate",
        "points": len(s26_rows),
        "positive_all_points": sum(1 for row in s26_rows if row["delta_all_psnr"] > 0.0),
        "positive_middle_points": sum(1 for row in s26_rows if row["delta_middle_psnr"] > 0.0),
        "mean_delta_all_psnr": float(np.mean([row["delta_all_psnr"] for row in s26_rows])),
        "mean_delta_middle_psnr": float(np.mean([row["delta_middle_psnr"] for row in s26_rows])),
        "mean_rate_delta_mib_per_frame": None,
        "interpretation": "best current adapter generalization evidence; not a selector gain",
    })

    s36_rows = read_csv(STAGE36_ZLIB_COMPARISON)
    rows.append({
        "stage": 36,
        "evidence": "dense anchor-attribute oracle selector vs uniform",
        "deployable_selector": "no",
        "actual_bitstream": "yes, zlib q8 anchors",
        "points": len(s36_rows),
        "positive_all_points": positives(s36_rows, "selector_delta_adapter_all_psnr"),
        "positive_middle_points": positives(s36_rows, "selector_delta_adapter_middle_psnr"),
        "mean_delta_all_psnr": mean_float(s36_rows, "selector_delta_adapter_all_psnr"),
        "mean_delta_middle_psnr": mean_float(s36_rows, "selector_delta_adapter_middle_psnr"),
        "mean_rate_delta_mib_per_frame": mean_float(s36_rows, "rate_delta_mib_per_frame"),
        "interpretation": "strong selector upper bound; uses non-deployable dense intermediate anchors",
    })

    for stage, summary_path, prefix in [
        (39, STAGE39_SUMMARY, "raw ridge predicted selector"),
        (41, STAGE41_SUMMARY, "normalized/rank predicted selector"),
    ]:
        summary = load_json(summary_path)
        for method, agg in summary["aggregates"].items():
            rows.append({
                "stage": stage,
                "evidence": f"{prefix}: {method}",
                "deployable_selector": "yes",
                "actual_bitstream": "estimated q8 rate",
                "points": agg["count"],
                "positive_all_points": agg["positive_all_points"],
                "positive_middle_points": agg["positive_middle_points"],
                "mean_delta_all_psnr": agg["mean_delta_all_psnr"],
                "mean_delta_middle_psnr": agg["mean_delta_middle_psnr"],
                "mean_rate_delta_mib_per_frame": None,
                "interpretation": "negative selector result against uniform",
            })

    stage42 = load_json(STAGE42_SUMMARY)
    for method, agg in stage42["aggregates"].items():
        rows.append({
            "stage": 42,
            "evidence": f"uniform-prior calibrated selector: {method}",
            "deployable_selector": "yes",
            "actual_bitstream": "estimated q8 rate",
            "points": agg["count"],
            "positive_all_points": agg["positive_all_points"],
            "positive_middle_points": agg["positive_middle_points"],
            "mean_delta_all_psnr": agg["mean_delta_all_psnr"],
            "mean_delta_middle_psnr": agg["mean_delta_middle_psnr"],
            "mean_rate_delta_mib_per_frame": None,
            "interpretation": f"calibration fallback; exact uniform points={agg['exact_uniform_points']}",
        })

    csv_path = args.summary_root / "stage43_selector_evidence_synthesis.csv"
    json_path = args.summary_root / "stage43_selector_evidence_synthesis_summary.json"
    md_path = args.summary_root / "stage43_selector_evidence_synthesis.md"
    write_csv(rows, csv_path)
    best_deployable = max(
        [row for row in rows if row["deployable_selector"] == "yes"],
        key=lambda row: row["mean_delta_all_psnr"],
    )
    summary = {
        "stage": 43,
        "mode": "selector evidence synthesis",
        "csv": str(csv_path),
        "markdown": str(md_path),
        "rows": rows,
        "best_deployable_selector_by_mean_all_psnr_delta": best_deployable,
        "takeaway": "Current deployable predicted selectors do not beat uniform. Dense oracle/proxy selector remains a useful upper bound, while Stage26 leave-one-out adapter gains remain the strongest deployable codec evidence.",
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = "# Stage 43 Selector Evidence Synthesis\n\n" + markdown_table(rows) + "\n\n"
    md += "## Takeaway\n\n"
    md += summary["takeaway"] + "\n"
    md_path.write_text(md, encoding="utf-8")
    print(json.dumps({"summary": str(json_path), "csv": str(csv_path), "markdown": str(md_path), "best_deployable": best_deployable}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
