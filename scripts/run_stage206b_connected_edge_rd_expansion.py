import argparse
import json
import random
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_MANIFEST = REPO_ROOT / "experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_tasks.csv"
DEFAULT_STAGE205_PACKAGE = REPO_ROOT / "experiments/stage205_fixed_gap_predictive_codec_validation/stage205_fixed_gap_predictive_codec_validation_package.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage206b_connected_edge_rd_expansion"

sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage201_predictor_only_smoke import parse_task_rows, write_csv  # noqa: E402
from scripts.run_stage204_residual_codec_smoke import evaluate_task  # noqa: E402
import scripts.run_stage206_edge_rd_table as stage206  # noqa: E402


WINDOW_FIELDS = [
    "window_id",
    "sequence",
    "start_index",
    "end_index",
    "window_frames",
    "edge_count",
    "target_count",
    "gap4_chain_edges",
    "connected_transition_count",
]


def actual_gap(row):
    return int(row["right_index"]) - int(row["left_index"])


def complete_edge_map(rows):
    out = {}
    for _, items in stage206.group_by_edge(rows).items():
        first = items[0]
        gap = int(first["reference_gap"])
        if actual_gap(first) != gap:
            continue
        if len(items) != stage206.expected_intermediates(first):
            continue
        out[(first["sequence"], int(first["left_index"]), int(first["right_index"]), gap)] = items
    return out


def connected_transitions(edge_items):
    by_sequence = {}
    for items in edge_items:
        first = items[0]
        by_sequence.setdefault(first["sequence"], []).append((int(first["left_index"]), int(first["right_index"])))
    total = 0
    for edges in by_sequence.values():
        starts = {}
        for left, right in edges:
            starts.setdefault(left, []).append(right)
        total += sum(1 for _left, right in edges for _next_right in starts.get(right, []))
    return total


def select_windows(edge_map, args):
    selected_windows = []
    selected_edge_keys = set()
    sequences = list(args.sequences)
    if not sequences:
        sequences = sorted({key[0] for key in edge_map})
    rng = random.Random(args.seed)
    if args.shuffle_sequences:
        rng.shuffle(sequences)
    min_gap = min(args.gaps)
    for sequence in sequences:
        starts = sorted({left for seq, left, _right, gap in edge_map if seq == sequence and gap == min_gap})
        for start in starts:
            end = start + int(args.window_frames)
            chain = [(sequence, idx, idx + min_gap, min_gap) for idx in range(start, end, min_gap)]
            if any(key not in edge_map for key in chain):
                continue
            window_keys = []
            for key in sorted(edge_map):
                seq, left, right, gap = key
                if seq != sequence or gap not in args.gaps:
                    continue
                if left < start or right > end:
                    continue
                window_keys.append(key)
            if not window_keys:
                continue
            edge_items = [edge_map[key] for key in window_keys]
            transitions = connected_transitions(edge_items)
            if transitions <= 0:
                continue
            window_id = f"{sequence}:{start:05d}:{end:05d}"
            selected_windows.append(
                {
                    "window_id": window_id,
                    "sequence": sequence,
                    "start_index": start,
                    "end_index": end,
                    "window_frames": args.window_frames,
                    "edge_count": len(window_keys),
                    "target_count": sum(len(edge_map[key]) for key in window_keys),
                    "gap4_chain_edges": len(chain),
                    "connected_transition_count": transitions,
                    "edge_keys": window_keys,
                }
            )
            selected_edge_keys.update(window_keys)
            if len(selected_windows) >= args.max_windows:
                return selected_windows, selected_edge_keys
    return selected_windows, selected_edge_keys


def window_rows(windows):
    return [
        {
            "window_id": row["window_id"],
            "sequence": row["sequence"],
            "start_index": row["start_index"],
            "end_index": row["end_index"],
            "window_frames": row["window_frames"],
            "edge_count": row["edge_count"],
            "target_count": row["target_count"],
            "gap4_chain_edges": row["gap4_chain_edges"],
            "connected_transition_count": row["connected_transition_count"],
        }
        for row in windows
    ]


def write_report(package, windows, best_rows, gates, path):
    lines = [
        "# Stage206b Connected Edge RD Expansion",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Windows: `{len(windows)}`; edges: `{package['edge_count']}`; target rows: `{package['target_count']}`; settings: `{package['setting_count']}`.",
        "- Scope: small connected-window expansion for Stage207 rerun, not final full-sequence RD.",
        "",
        "## Windows",
        "",
        "| window | sequence | start | end | edges | targets | connected transitions |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in windows:
        lines.append(
            f"| {row['window_id']} | {row['sequence']} | {row['start_index']} | {row['end_index']} | "
            f"{row['edge_count']} | {row['target_count']} | {row['connected_transition_count']} |"
        )
    lines.extend([
        "",
        "## Best By Gap",
        "",
        "| gap | best setting | keep | edge total bytes | DP incremental bytes | residual bytes | corrected PSNR | dPSNR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in best_rows:
        lines.append(
            f"| {row['reference_gap']} | {row['best_setting_label']} | {row['best_keep_fraction']} | "
            f"{float(row['best_mean_edge_total_bytes_once']):.3f} | {float(row['best_mean_dp_incremental_bytes']):.3f} | "
            f"{float(row['best_mean_residual_payload_bytes']):.3f} | {float(row['best_mean_corrected_psnr']):.6f} | "
            f"{float(row['best_mean_delta_psnr_vs_base']):.6f} |"
        )
    lines.extend([
        "",
        "## Gates",
        "",
        "| gate | status | value | threshold | detail |",
        "|---|---|---|---|---|",
    ])
    for row in gates:
        lines.append(f"| {row['gate']} | {row['status']} | {row['value']} | {row['threshold']} | {row['detail']} |")
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- windows: `{package['windows_csv']}`",
        f"- selected edges: `{package['selected_edges_csv']}`",
        f"- target rows: `{package['target_rows_csv']}`",
        f"- edge RD rows: `{package['edge_rows_csv']}`",
        f"- summary: `{package['summary_csv']}`",
        f"- best by gap: `{package['best_by_gap_csv']}`",
        f"- gates: `{package['gates_csv']}`",
        f"- package: `{package['package_json']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--stage205_package", type=Path, default=DEFAULT_STAGE205_PACKAGE)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--task_split", default="eval")
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--keyframe_codec", default="q12")
    parser.add_argument("--keyframe_bits", type=int, default=12)
    parser.add_argument("--keyframe_compression", default="none")
    parser.add_argument("--keyframe_payload_encoding", default="bitpack")
    parser.add_argument("--window_frames", type=int, default=24)
    parser.add_argument("--max_windows", type=int, default=1)
    parser.add_argument("--sequences", nargs="*", default=["bike-packing", "parkour"])
    parser.add_argument("--shuffle_sequences", action="store_true")
    parser.add_argument("--keep_fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--side_bits", type=int, default=6)
    parser.add_argument("--zlib_level", type=int, default=9)
    parser.add_argument("--edge_metadata_bytes", type=int, default=2)
    parser.add_argument("--positive_headroom_db", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    stage205 = stage206.read_json(args.stage205_package)
    if stage205["decision"] != "fixed_gap_predictive_codec_positive_headroom":
        raise RuntimeError(f"Stage205 prerequisite failed: {stage205['decision']}")
    all_rows = parse_task_rows(args.task_manifest, args.task_split, args.gaps, args.keyframe_codec)
    edge_map = complete_edge_map(all_rows)
    windows, selected_edge_keys = select_windows(edge_map, args)
    if not selected_edge_keys:
        raise RuntimeError("no connected windows found")
    selected_rows = []
    for key in sorted(selected_edge_keys):
        selected_rows.extend(edge_map[key])
    selected_rows.sort(key=lambda row: (row["sequence"], int(row["left_index"]), int(row["right_index"]), int(row["target_index"])))

    device = torch.device(args.device)
    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    cache = {}
    settings = [(keep, int(args.side_bits)) for keep in args.keep_fractions]
    raw_metric_rows = []
    for row in selected_rows:
        raw_metric_rows.extend(evaluate_task(row, settings, args, device, cache, opt, background))
    tasks_by_id = {row["task_id"]: row for row in selected_rows}
    target_rows = stage206.annotate_metric_rows(raw_metric_rows, tasks_by_id)
    edge_rows = stage206.build_edge_rows(selected_rows, target_rows, settings, args)
    summary_rows = stage206.summarize(edge_rows)
    best_rows = stage206.best_rows(summary_rows)
    gates = stage206.gate_rows(stage205, target_rows, edge_rows, best_rows, args)
    decision_value = stage206.decision(gates)

    windows_csv = args.output_root / "stage206b_connected_windows.csv"
    selected_edges_csv = args.output_root / "stage206b_selected_edges.csv"
    target_rows_csv = args.output_root / "stage206b_target_metric_rows.csv"
    edge_rows_csv = args.output_root / "stage206b_edge_rd_rows.csv"
    summary_csv = args.output_root / "stage206b_edge_rd_summary.csv"
    best_by_gap_csv = args.output_root / "stage206b_edge_rd_best_by_gap.csv"
    gates_csv = args.output_root / "stage206b_edge_rd_gates.csv"
    package_json = args.output_root / "stage206b_connected_edge_rd_expansion_package.json"
    report_md = args.output_root / "stage206b_connected_edge_rd_expansion_report.md"
    write_csv(window_rows(windows), windows_csv, WINDOW_FIELDS)
    write_csv(stage206.selected_edge_rows(selected_rows), selected_edges_csv, stage206.SELECTED_EDGE_FIELDS)
    write_csv(target_rows, target_rows_csv, stage206.TARGET_FIELDS)
    write_csv(edge_rows, edge_rows_csv, stage206.EDGE_FIELDS)
    write_csv(summary_rows, summary_csv, stage206.SUMMARY_FIELDS)
    write_csv(best_rows, best_by_gap_csv, stage206.BEST_FIELDS)
    write_csv(gates, gates_csv, stage206.GATE_FIELDS)
    package = {
        "stage": "206b",
        "name": "connected_edge_rd_expansion",
        "decision": decision_value,
        "task_manifest": str(args.task_manifest),
        "stage205_package": str(args.stage205_package),
        "task_split": args.task_split,
        "gaps": args.gaps,
        "keyframe_codec": args.keyframe_codec,
        "keyframe_bits": args.keyframe_bits,
        "keyframe_compression": args.keyframe_compression,
        "keyframe_payload_encoding": args.keyframe_payload_encoding,
        "edge_metadata_bytes": args.edge_metadata_bytes,
        "window_frames": args.window_frames,
        "window_count": len(windows),
        "edge_count": len(stage206.selected_edge_rows(selected_rows)),
        "target_count": len(selected_rows),
        "setting_count": len(settings),
        "side_bits": args.side_bits,
        "keep_fractions": args.keep_fractions,
        "windows_csv": str(windows_csv),
        "selected_edges_csv": str(selected_edges_csv),
        "target_rows_csv": str(target_rows_csv),
        "edge_rows_csv": str(edge_rows_csv),
        "summary_csv": str(summary_csv),
        "best_by_gap_csv": str(best_by_gap_csv),
        "gates_csv": str(gates_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "best_rows": best_rows,
        "gate_rows": gates,
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, windows, best_rows, gates, report_md)
    print(json.dumps({"package": str(package_json), "decision": decision_value}, indent=2))


if __name__ == "__main__":
    main()
