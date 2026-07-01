import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE192_ROOT = REPO_ROOT / "experiments/stage192_expanded_fixed_gap_measurement"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage193_oracle_upper_bound_analysis"

DEFAULT_FINAL_ROWS = DEFAULT_STAGE192_ROOT / "stage192_full_sequence_quality_by_schedule.csv"
DEFAULT_RD_QUALITY = DEFAULT_STAGE192_ROOT / "stage192_expanded_fixed_gap_rd_quality_points.csv"
DEFAULT_KEYFRAME_QUALITY = DEFAULT_STAGE192_ROOT / "stage192_unique_keyframe_quality_metrics.csv"
DEFAULT_RESIDUAL_QUALITY = DEFAULT_STAGE192_ROOT / "stage192_unique_stage158_residual_quality_metrics.csv"
DEFAULT_KEYFRAME_PAYLOAD = DEFAULT_STAGE192_ROOT / "stage192_unique_keyframe_payload_measurements.csv"
DEFAULT_RESIDUAL_PAYLOAD = DEFAULT_STAGE192_ROOT / "stage192_unique_stage158_residual_payload_measurements.csv"

METRICS = ["psnr", "ssim", "ms_ssim", "lpips"]

ORACLE_SUMMARY_FIELDS = [
    "oracle",
    "schedule_consistent",
    "frame_count",
    "mean_psnr",
    "mean_ssim",
    "mean_ms_ssim",
    "mean_lpips",
    "total_payload_bytes_additive_proxy",
    "mib_per_frame_additive_proxy",
    "best_fixed_by_psnr",
    "best_fixed_psnr",
    "delta_psnr_vs_best_fixed",
    "delta_ssim_vs_best_fixed",
    "delta_ms_ssim_vs_best_fixed",
    "delta_lpips_vs_best_fixed",
    "beats_best_fixed_by_1db_no_metric_regression",
    "note",
]

FRAMEWISE_FIELDS = [
    "sequence",
    "frame_index",
    "chosen_schedule",
    "final_type",
    "measurement_key",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
]

PATH_FRAME_FIELDS = [
    "sequence",
    "frame_index",
    "oracle_role",
    "left_index",
    "right_index",
    "measurement_key",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "payload_bytes",
]

SEQUENCE_FIELDS = [
    "oracle",
    "sequence",
    "frame_count",
    "mean_psnr",
    "mean_ssim",
    "mean_ms_ssim",
    "mean_lpips",
    "chosen_edge_count",
    "keyframe_count",
    "additive_payload_bytes",
]


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def numeric(row, key, default=0.0):
    value = row.get(key) if row else None
    if value in (None, "", "NA"):
        return default
    return float(value)


def mean(values):
    vals = [float(v) for v in values if v not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def group_by(rows, keys):
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[key] for key in keys)].append(row)
    return out


def best_fixed_row(rd_rows):
    fixed = [row for row in rd_rows if row["schedule_family"] == "fixed_gap"]
    return max(fixed, key=lambda row: numeric(row, "mean_psnr"))


def summarize_rows(oracle_name, rows, best_fixed, schedule_consistent, payload_bytes, note):
    frame_count = len(rows)
    row = {
        "oracle": oracle_name,
        "schedule_consistent": int(schedule_consistent),
        "frame_count": frame_count,
        "mean_psnr": mean(row["psnr"] for row in rows),
        "mean_ssim": mean(row["ssim"] for row in rows),
        "mean_ms_ssim": mean(row["ms_ssim"] for row in rows),
        "mean_lpips": mean(row["lpips"] for row in rows),
        "total_payload_bytes_additive_proxy": payload_bytes,
        "mib_per_frame_additive_proxy": payload_bytes / (1024.0 * 1024.0) / frame_count if frame_count else 0.0,
        "best_fixed_by_psnr": best_fixed["schedule"],
        "best_fixed_psnr": numeric(best_fixed, "mean_psnr"),
        "note": note,
    }
    row["delta_psnr_vs_best_fixed"] = row["mean_psnr"] - numeric(best_fixed, "mean_psnr")
    row["delta_ssim_vs_best_fixed"] = row["mean_ssim"] - numeric(best_fixed, "mean_ssim")
    row["delta_ms_ssim_vs_best_fixed"] = row["mean_ms_ssim"] - numeric(best_fixed, "mean_ms_ssim")
    row["delta_lpips_vs_best_fixed"] = row["mean_lpips"] - numeric(best_fixed, "mean_lpips")
    row["beats_best_fixed_by_1db_no_metric_regression"] = int(
        row["delta_psnr_vs_best_fixed"] >= 1.0
        and row["delta_ssim_vs_best_fixed"] >= 0.0
        and row["delta_ms_ssim_vs_best_fixed"] >= 0.0
        and row["delta_lpips_vs_best_fixed"] <= 0.0
    )
    return row


def build_framewise_oracle(final_rows):
    out = []
    for (sequence, frame_index), group in sorted(group_by(final_rows, ["sequence", "frame_index"]).items()):
        chosen = max(group, key=lambda row: numeric(row, "psnr"))
        out.append(
            {
                "sequence": sequence,
                "frame_index": int(frame_index),
                "chosen_schedule": chosen["schedule"],
                "final_type": chosen["final_type"],
                "measurement_key": chosen["measurement_key"],
                "psnr": numeric(chosen, "psnr"),
                "ssim": numeric(chosen, "ssim"),
                "ms_ssim": numeric(chosen, "ms_ssim"),
                "lpips": numeric(chosen, "lpips"),
            }
        )
    return out


def keyframe_key(sequence, frame_index):
    return f"keyframe::{sequence}:{int(frame_index):05d}"


def residual_key(sequence, target_index, left_index, right_index):
    return f"residual::{sequence}:{int(target_index):05d}:{int(left_index):05d}:{int(right_index):05d}"


def build_edges(residual_quality_rows, residual_payload_by_key):
    grouped = defaultdict(list)
    for row in residual_quality_rows:
        grouped[(row["sequence"], int(row["left_index"]), int(row["right_index"]))].append(row)
    edges = defaultdict(list)
    for (sequence, left, right), rows in grouped.items():
        expected_targets = set(range(left + 1, right))
        targets = {int(row["target_index"]) for row in rows}
        if targets != expected_targets:
            continue
        sorted_rows = sorted(rows, key=lambda row: int(row["target_index"]))
        payload = 0
        for row in sorted_rows:
            payload += int(numeric(residual_payload_by_key.get(row["measurement_key"], row), "payload_bytes"))
        edges[(sequence, left)].append(
            {
                "sequence": sequence,
                "left": left,
                "right": right,
                "interior_rows": sorted_rows,
                "payload_bytes": payload,
                "psnr_sum": sum(numeric(row, "psnr") for row in sorted_rows),
            }
        )
    return edges


def add_consecutive_edges(edges, keyframe_quality_by_node):
    by_sequence = defaultdict(list)
    for sequence, frame_index in keyframe_quality_by_node:
        by_sequence[sequence].append(frame_index)
    existing = {(seq, edge["left"], edge["right"]) for (seq, _left), group in edges.items() for edge in group}
    for sequence, frames in by_sequence.items():
        frame_set = set(frames)
        for left in frames:
            right = left + 1
            if right in frame_set and (sequence, left, right) not in existing:
                edges[(sequence, left)].append(
                    {
                        "sequence": sequence,
                        "left": left,
                        "right": right,
                        "interior_rows": [],
                        "payload_bytes": 0,
                        "psnr_sum": 0.0,
                    }
                )
    for key in edges:
        edges[key].sort(key=lambda edge: edge["right"])


def build_schedule_path_oracle(final_rows, keyframe_quality_rows, keyframe_payload_by_key, residual_quality_rows, residual_payload_by_key):
    keyframe_quality = {
        (row["sequence"], int(row["frame_index"])): row
        for row in keyframe_quality_rows
        if row.get("status") == "ok"
    }
    edges = build_edges([row for row in residual_quality_rows if row.get("status") == "ok"], residual_payload_by_key)
    add_consecutive_edges(edges, keyframe_quality)

    total_frames_by_sequence = {}
    for row in final_rows:
        total_frames_by_sequence[row["sequence"]] = max(total_frames_by_sequence.get(row["sequence"], 0), int(row["frame_index"]) + 1)

    oracle_rows = []
    sequence_rows = []
    for sequence, total_frames in sorted(total_frames_by_sequence.items()):
        start = 0
        end = total_frames - 1
        if (sequence, start) not in keyframe_quality or (sequence, end) not in keyframe_quality:
            raise ValueError(f"missing endpoint keyframe quality for {sequence}")
        nodes = sorted(frame for seq, frame in keyframe_quality if seq == sequence)
        dp = {node: -math.inf for node in nodes}
        prev = {}
        dp[start] = numeric(keyframe_quality[(sequence, start)], "psnr")
        for left in nodes:
            if dp[left] == -math.inf:
                continue
            for edge in edges.get((sequence, left), []):
                right = edge["right"]
                if (sequence, right) not in keyframe_quality:
                    continue
                score = dp[left] + edge["psnr_sum"] + numeric(keyframe_quality[(sequence, right)], "psnr")
                if score > dp.get(right, -math.inf):
                    dp[right] = score
                    prev[right] = edge
        if dp[end] == -math.inf:
            raise ValueError(f"no oracle path reaches {sequence} frame {end}")

        path_edges = []
        cur = end
        while cur != start:
            edge = prev[cur]
            path_edges.append(edge)
            cur = edge["left"]
        path_edges.reverse()

        chosen_frames = []
        start_key = keyframe_key(sequence, start)
        start_payload = int(numeric(keyframe_payload_by_key.get(start_key, {}), "bitstream_bytes"))
        chosen_frames.append(
            {
                "sequence": sequence,
                "frame_index": start,
                "oracle_role": "keyframe",
                "left_index": start,
                "right_index": start,
                "measurement_key": start_key,
                "psnr": numeric(keyframe_quality[(sequence, start)], "psnr"),
                "ssim": numeric(keyframe_quality[(sequence, start)], "ssim"),
                "ms_ssim": numeric(keyframe_quality[(sequence, start)], "ms_ssim"),
                "lpips": numeric(keyframe_quality[(sequence, start)], "lpips"),
                "payload_bytes": start_payload,
            }
        )
        additive_payload = start_payload
        keyframe_count = 1
        for edge in path_edges:
            for row in edge["interior_rows"]:
                payload = int(numeric(residual_payload_by_key.get(row["measurement_key"], row), "payload_bytes"))
                chosen_frames.append(
                    {
                        "sequence": sequence,
                        "frame_index": int(row["target_index"]),
                        "oracle_role": "residual",
                        "left_index": edge["left"],
                        "right_index": edge["right"],
                        "measurement_key": row["measurement_key"],
                        "psnr": numeric(row, "psnr"),
                        "ssim": numeric(row, "ssim"),
                        "ms_ssim": numeric(row, "ms_ssim"),
                        "lpips": numeric(row, "lpips"),
                        "payload_bytes": payload,
                    }
                )
                additive_payload += payload
            right = edge["right"]
            k = keyframe_key(sequence, right)
            payload = int(numeric(keyframe_payload_by_key.get(k, {}), "bitstream_bytes"))
            chosen_frames.append(
                {
                    "sequence": sequence,
                    "frame_index": right,
                    "oracle_role": "keyframe",
                    "left_index": right,
                    "right_index": right,
                    "measurement_key": k,
                    "psnr": numeric(keyframe_quality[(sequence, right)], "psnr"),
                    "ssim": numeric(keyframe_quality[(sequence, right)], "ssim"),
                    "ms_ssim": numeric(keyframe_quality[(sequence, right)], "ms_ssim"),
                    "lpips": numeric(keyframe_quality[(sequence, right)], "lpips"),
                    "payload_bytes": payload,
                }
            )
            additive_payload += payload
            keyframe_count += 1
        chosen_frames.sort(key=lambda row: row["frame_index"])
        if len(chosen_frames) != total_frames:
            raise ValueError(f"oracle path frame coverage mismatch for {sequence}: {len(chosen_frames)} vs {total_frames}")
        oracle_rows.extend(chosen_frames)
        sequence_rows.append(
            {
                "oracle": "schedule_path_psnr_oracle",
                "sequence": sequence,
                "frame_count": total_frames,
                "mean_psnr": mean(row["psnr"] for row in chosen_frames),
                "mean_ssim": mean(row["ssim"] for row in chosen_frames),
                "mean_ms_ssim": mean(row["ms_ssim"] for row in chosen_frames),
                "mean_lpips": mean(row["lpips"] for row in chosen_frames),
                "chosen_edge_count": len(path_edges),
                "keyframe_count": keyframe_count,
                "additive_payload_bytes": additive_payload,
            }
        )
    return oracle_rows, sequence_rows


def framewise_sequence_rows(framewise_rows):
    out = []
    for (sequence,), rows in sorted(group_by(framewise_rows, ["sequence"]).items()):
        out.append(
            {
                "oracle": "framewise_psnr_oracle",
                "sequence": sequence,
                "frame_count": len(rows),
                "mean_psnr": mean(row["psnr"] for row in rows),
                "mean_ssim": mean(row["ssim"] for row in rows),
                "mean_ms_ssim": mean(row["ms_ssim"] for row in rows),
                "mean_lpips": mean(row["lpips"] for row in rows),
                "chosen_edge_count": "",
                "keyframe_count": sum(1 for row in rows if row["final_type"] == "q12_keyframe"),
                "additive_payload_bytes": "",
            }
        )
    return out


def decide(summary_rows):
    framewise = next(row for row in summary_rows if row["oracle"] == "framewise_psnr_oracle")
    path = next(row for row in summary_rows if row["oracle"] == "schedule_path_psnr_oracle")
    if int(framewise["beats_best_fixed_by_1db_no_metric_regression"]) == 0:
        return "framewise_oracle_upper_bound_below_target_margin"
    if int(path["beats_best_fixed_by_1db_no_metric_regression"]) == 0:
        return "framewise_oracle_has_headroom_but_schedule_path_oracle_below_target"
    return "schedule_path_oracle_target_margin_achievable"


def write_report(summary_rows, package, path):
    lines = [
        "# Stage193 Oracle Upper-Bound Analysis",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Best fixed gap by PSNR: `{package['best_fixed_by_psnr']}` with PSNR `{package['best_fixed_psnr']}`.",
        "",
        "## Oracle Summary",
        "",
        "| oracle | schedule consistent | PSNR | SSIM | MS-SSIM | LPIPS | dPSNR vs best fixed | dLPIPS vs best fixed | +1dB pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['oracle']} | {row['schedule_consistent']} | {float(row['mean_psnr']):.6f} | {float(row['mean_ssim']):.6f} | "
            f"{float(row['mean_ms_ssim']):.6f} | {float(row['mean_lpips']):.6f} | {float(row['delta_psnr_vs_best_fixed']):.6f} | "
            f"{float(row['delta_lpips_vs_best_fixed']):.6f} | {row['beats_best_fixed_by_1db_no_metric_regression']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `framewise_psnr_oracle` is an optimistic non-schedule-consistent upper bound that picks the best measured schedule output independently per frame.",
        "- `schedule_path_psnr_oracle` is schedule-consistent, but limited to measured keyframe nodes and measured Stage158 residual edges from Stage192.",
        "- If the framewise oracle is below the +1 dB target, selector tuning over the current measured representation cannot plausibly satisfy the requested strong claim.",
        "",
        "## Outputs",
        "",
        f"- Summary CSV: `{package['summary_csv']}`",
        f"- Framewise oracle rows: `{package['framewise_oracle_csv']}`",
        f"- Schedule path oracle rows: `{package['schedule_path_oracle_csv']}`",
        f"- Sequence summary: `{package['sequence_summary_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_rows", type=Path, default=DEFAULT_FINAL_ROWS)
    parser.add_argument("--rd_quality", type=Path, default=DEFAULT_RD_QUALITY)
    parser.add_argument("--keyframe_quality", type=Path, default=DEFAULT_KEYFRAME_QUALITY)
    parser.add_argument("--residual_quality", type=Path, default=DEFAULT_RESIDUAL_QUALITY)
    parser.add_argument("--keyframe_payload", type=Path, default=DEFAULT_KEYFRAME_PAYLOAD)
    parser.add_argument("--residual_payload", type=Path, default=DEFAULT_RESIDUAL_PAYLOAD)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    final_rows = read_csv(args.final_rows)
    rd_rows = read_csv(args.rd_quality)
    keyframe_quality = read_csv(args.keyframe_quality)
    residual_quality = read_csv(args.residual_quality)
    keyframe_payload_by_key = {row["measurement_key"]: row for row in read_csv(args.keyframe_payload) if row.get("status") == "ok"}
    residual_payload_by_key = {row["measurement_key"]: row for row in read_csv(args.residual_payload) if row.get("status") == "ok"}
    best_fixed = best_fixed_row(rd_rows)

    framewise_rows = build_framewise_oracle(final_rows)
    path_rows, path_sequence_rows = build_schedule_path_oracle(final_rows, keyframe_quality, keyframe_payload_by_key, residual_quality, residual_payload_by_key)
    framewise_payload = 0
    path_payload = sum(int(row["payload_bytes"]) for row in path_rows)
    summary_rows = [
        summarize_rows(
            "framewise_psnr_oracle",
            framewise_rows,
            best_fixed,
            False,
            framewise_payload,
            "optimistic per-frame best measured schedule output; not schedule-consistent or rate-valid",
        ),
        summarize_rows(
            "schedule_path_psnr_oracle",
            path_rows,
            best_fixed,
            True,
            path_payload,
            "dynamic-programming path over measured keyframe nodes and measured residual edges; additive payload proxy only",
        ),
    ]
    sequence_rows = framewise_sequence_rows(framewise_rows) + path_sequence_rows
    decision = decide(summary_rows)

    summary_csv = args.output_root / "stage193_oracle_summary.csv"
    framewise_csv = args.output_root / "stage193_framewise_psnr_oracle_rows.csv"
    path_csv = args.output_root / "stage193_schedule_path_psnr_oracle_rows.csv"
    sequence_csv = args.output_root / "stage193_oracle_sequence_summary.csv"
    package_json = args.output_root / "stage193_oracle_upper_bound_analysis_package.json"
    report_md = args.output_root / "stage193_oracle_upper_bound_analysis_report.md"
    write_csv(summary_rows, summary_csv, ORACLE_SUMMARY_FIELDS)
    write_csv(framewise_rows, framewise_csv, FRAMEWISE_FIELDS)
    write_csv(path_rows, path_csv, PATH_FRAME_FIELDS)
    write_csv(sequence_rows, sequence_csv, SEQUENCE_FIELDS)
    package = {
        "stage": 193,
        "status": "oracle_upper_bound_analysis_complete",
        "decision": decision,
        "best_fixed_by_psnr": best_fixed["schedule"],
        "best_fixed_psnr": numeric(best_fixed, "mean_psnr"),
        "summary_rows": summary_rows,
        "summary_csv": str(summary_csv.relative_to(REPO_ROOT)),
        "framewise_oracle_csv": str(framewise_csv.relative_to(REPO_ROOT)),
        "schedule_path_oracle_csv": str(path_csv.relative_to(REPO_ROOT)),
        "sequence_summary_csv": str(sequence_csv.relative_to(REPO_ROOT)),
        "package_json": str(package_json.relative_to(REPO_ROOT)),
        "report_md": str(report_md.relative_to(REPO_ROOT)),
        "next": "If oracle headroom is below target, design changes must alter representation/payload policy rather than only selector thresholds.",
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary_rows, package, report_md)
    print(json.dumps({"package": str(package_json), "decision": decision, "summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
