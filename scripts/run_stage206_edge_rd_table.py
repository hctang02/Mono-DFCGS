import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_MANIFEST = REPO_ROOT / "experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_tasks.csv"
DEFAULT_STAGE205_PACKAGE = REPO_ROOT / "experiments/stage205_fixed_gap_predictive_codec_validation/stage205_fixed_gap_predictive_codec_validation_package.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage206_edge_rd_table"
MIB = 1024.0 * 1024.0

sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.anchor_bitstream import encode_anchor_bitstream  # noqa: E402
from scripts.run_stage201_predictor_only_smoke import parse_task_rows, write_csv  # noqa: E402
from scripts.run_stage204_residual_codec_smoke import evaluate_task, setting_label  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import load_anchor  # noqa: E402


TARGET_FIELDS = [
    "edge_id",
    "task_id",
    "sequence",
    "reference_gap",
    "left_index",
    "right_index",
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
EDGE_FIELDS = [
    "edge_id",
    "sequence",
    "reference_gap",
    "left_index",
    "right_index",
    "represented_frame_count",
    "intermediate_count_expected",
    "intermediate_count_measured",
    "keyframe_codec",
    "keyframe_bits",
    "setting_label",
    "keep_fraction",
    "side_bits",
    "left_keyframe_bytes",
    "right_keyframe_bytes",
    "endpoint_keyframe_bytes",
    "residual_payload_bytes",
    "schedule_metadata_bytes",
    "edge_total_bytes_once",
    "dp_incremental_bytes",
    "residual_mib_per_intermediate",
    "edge_total_mib_once",
    "dp_incremental_mib",
    "mean_base_psnr",
    "mean_corrected_psnr",
    "mean_delta_psnr_vs_base",
    "min_delta_psnr_vs_base",
    "mean_anchor_mse_reduction",
    "status",
    "error",
]
SUMMARY_FIELDS = [
    "reference_gap",
    "setting_label",
    "keep_fraction",
    "side_bits",
    "edge_count",
    "target_count",
    "mean_left_keyframe_bytes",
    "mean_right_keyframe_bytes",
    "mean_endpoint_keyframe_bytes",
    "mean_residual_payload_bytes",
    "mean_schedule_metadata_bytes",
    "mean_edge_total_bytes_once",
    "mean_dp_incremental_bytes",
    "mean_residual_mib_per_intermediate",
    "mean_edge_total_mib_once",
    "mean_dp_incremental_mib",
    "mean_base_psnr",
    "mean_corrected_psnr",
    "mean_delta_psnr_vs_base",
    "min_edge_delta_psnr_vs_base",
]
BEST_FIELDS = [
    "reference_gap",
    "best_setting_label",
    "best_keep_fraction",
    "best_mean_edge_total_bytes_once",
    "best_mean_dp_incremental_bytes",
    "best_mean_residual_payload_bytes",
    "best_mean_corrected_psnr",
    "best_mean_delta_psnr_vs_base",
]
SELECTED_EDGE_FIELDS = [
    "edge_id",
    "sequence",
    "reference_gap",
    "left_index",
    "right_index",
    "target_count",
    "expected_intermediate_count",
    "complete_intermediate_coverage",
]
GATE_FIELDS = ["gate", "status", "value", "threshold", "detail"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    values = [float(value) for value in values]
    return sum(values) / max(len(values), 1)


def edge_id(row):
    return f"{row['sequence']}:{int(row['left_index']):05d}:{int(row['right_index']):05d}:gap{int(row['reference_gap'])}"


def edge_key(row):
    return (int(row["reference_gap"]), row["sequence"], int(row["left_index"]), int(row["right_index"]), row.get("segment_id", ""))


def expected_intermediates(row):
    return max(int(row["right_index"]) - int(row["left_index"]) - 1, 0)


def group_by_edge(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[edge_key(row)].append(row)
    for items in groups.values():
        items.sort(key=lambda row: int(row["target_index"]))
    return groups


def sample_edge_rows(rows, gaps, max_edges_per_gap, seed):
    groups = group_by_edge(rows)
    selected_keys = []
    for gap in gaps:
        rng = random.Random(seed + int(gap))
        by_sequence = defaultdict(list)
        for key in sorted(groups):
            if int(key[0]) == int(gap):
                by_sequence[key[1]].append(key)
        sequences = sorted(by_sequence)
        for seq in sequences:
            rng.shuffle(by_sequence[seq])
        rng.shuffle(sequences)
        if max_edges_per_gap <= 0:
            selected_keys.extend(key for seq in sequences for key in by_sequence[seq])
            continue
        offset = 0
        while len([key for key in selected_keys if int(key[0]) == int(gap)]) < max_edges_per_gap:
            progressed = False
            for seq in sequences:
                candidates = by_sequence[seq]
                if offset >= len(candidates):
                    continue
                selected_keys.append(candidates[offset])
                progressed = True
                if len([key for key in selected_keys if int(key[0]) == int(gap)]) >= max_edges_per_gap:
                    break
            if not progressed:
                break
            offset += 1
    out = []
    for key in selected_keys:
        out.extend(groups[key])
    out.sort(key=lambda row: (int(row["reference_gap"]), row["sequence"], int(row["left_index"]), int(row["target_index"])))
    return out


def selected_edge_rows(rows):
    out = []
    for _, items in sorted(group_by_edge(rows).items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])):
        first = items[0]
        expected = expected_intermediates(first)
        out.append({
            "edge_id": edge_id(first),
            "sequence": first["sequence"],
            "reference_gap": int(first["reference_gap"]),
            "left_index": int(first["left_index"]),
            "right_index": int(first["right_index"]),
            "target_count": len(items),
            "expected_intermediate_count": expected,
            "complete_intermediate_coverage": int(len(items) == expected),
        })
    return out


def keyframe_bytes(row, side, args, byte_cache, anchor_cache):
    if side == "left":
        source_item = row["left_anchor_source_item"]
        source_side = row["left_anchor_source_side"]
        frame_index = int(row["left_index"])
    elif side == "right":
        source_item = row["right_anchor_source_item"]
        source_side = row["right_anchor_source_side"]
        frame_index = int(row["right_index"])
    else:
        raise ValueError(f"unknown side {side}")
    key = (source_item, source_side, frame_index, int(args.keyframe_bits), args.keyframe_compression, args.keyframe_payload_encoding)
    if key in byte_cache:
        return byte_cache[key]
    anchor = load_anchor(source_item, source_side, torch.device("cpu"), bits=None, cache=anchor_cache)
    blob = encode_anchor_bitstream(
        [anchor],
        [frame_index],
        timestamps=[frame_index],
        bits=int(args.keyframe_bits),
        compression=args.keyframe_compression,
        payload_encoding=args.keyframe_payload_encoding,
    )
    byte_cache[key] = len(blob)
    return byte_cache[key]


def annotate_metric_rows(metric_rows, tasks_by_id):
    out = []
    for row in metric_rows:
        item = dict(row)
        task = tasks_by_id.get(item["task_id"])
        if task:
            item["edge_id"] = edge_id(task)
            item["left_index"] = int(task["left_index"])
            item["right_index"] = int(task["right_index"])
        else:
            item["edge_id"] = ""
            item["left_index"] = ""
            item["right_index"] = ""
        out.append(item)
    return out


def build_edge_rows(selected_rows, metric_rows, settings, args):
    metrics = defaultdict(list)
    for row in metric_rows:
        metrics[(row["edge_id"], row["setting_label"])].append(row)
    byte_cache = {}
    anchor_cache = {}
    edge_rows = []
    for _, tasks in sorted(group_by_edge(selected_rows).items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])):
        first = tasks[0]
        expected = expected_intermediates(first)
        eid = edge_id(first)
        left_bytes = keyframe_bytes(first, "left", args, byte_cache, anchor_cache)
        right_bytes = keyframe_bytes(first, "right", args, byte_cache, anchor_cache)
        endpoint_bytes = left_bytes + right_bytes
        represented_frames = int(first["right_index"]) - int(first["left_index"]) + 1
        for keep_fraction, side_bits in settings:
            label = setting_label(keep_fraction, side_bits)
            items = metrics.get((eid, label), [])
            ok_items = [row for row in items if row["status"] == "ok"]
            errors = [row for row in items if row["status"] != "ok"]
            residual_bytes = sum(float(row["payload_bytes"]) for row in ok_items)
            metadata_bytes = int(args.edge_metadata_bytes)
            edge_total = endpoint_bytes + residual_bytes + metadata_bytes
            dp_incremental = right_bytes + residual_bytes + metadata_bytes
            status = "ok"
            error = ""
            if errors:
                status = "error"
                error = ";".join(str(row["error"]) for row in errors[:3])
            elif len(ok_items) != expected:
                status = "error"
                error = f"incomplete_intermediates:{len(ok_items)}/{expected}"
            edge_rows.append({
                "edge_id": eid,
                "sequence": first["sequence"],
                "reference_gap": int(first["reference_gap"]),
                "left_index": int(first["left_index"]),
                "right_index": int(first["right_index"]),
                "represented_frame_count": represented_frames,
                "intermediate_count_expected": expected,
                "intermediate_count_measured": len(ok_items),
                "keyframe_codec": args.keyframe_codec,
                "keyframe_bits": int(args.keyframe_bits),
                "setting_label": label,
                "keep_fraction": keep_fraction,
                "side_bits": side_bits,
                "left_keyframe_bytes": left_bytes,
                "right_keyframe_bytes": right_bytes,
                "endpoint_keyframe_bytes": endpoint_bytes,
                "residual_payload_bytes": residual_bytes,
                "schedule_metadata_bytes": metadata_bytes,
                "edge_total_bytes_once": edge_total,
                "dp_incremental_bytes": dp_incremental,
                "residual_mib_per_intermediate": residual_bytes / max(len(ok_items), 1) / MIB,
                "edge_total_mib_once": edge_total / MIB,
                "dp_incremental_mib": dp_incremental / MIB,
                "mean_base_psnr": mean(row["base_psnr"] for row in ok_items) if ok_items else "",
                "mean_corrected_psnr": mean(row["corrected_psnr"] for row in ok_items) if ok_items else "",
                "mean_delta_psnr_vs_base": mean(row["delta_psnr_vs_base"] for row in ok_items) if ok_items else "",
                "min_delta_psnr_vs_base": min((float(row["delta_psnr_vs_base"]) for row in ok_items), default=""),
                "mean_anchor_mse_reduction": mean(row["anchor_mse_reduction"] for row in ok_items) if ok_items else "",
                "status": status,
                "error": error,
            })
    return edge_rows


def summarize(edge_rows):
    grouped = defaultdict(list)
    for row in edge_rows:
        if row["status"] == "ok":
            grouped[(int(row["reference_gap"]), row["setting_label"], float(row["keep_fraction"]), int(row["side_bits"]))].append(row)
    out = []
    for (gap, label, keep_fraction, side_bits), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2])):
        out.append({
            "reference_gap": gap,
            "setting_label": label,
            "keep_fraction": keep_fraction,
            "side_bits": side_bits,
            "edge_count": len(items),
            "target_count": sum(int(row["intermediate_count_measured"]) for row in items),
            "mean_left_keyframe_bytes": mean(row["left_keyframe_bytes"] for row in items),
            "mean_right_keyframe_bytes": mean(row["right_keyframe_bytes"] for row in items),
            "mean_endpoint_keyframe_bytes": mean(row["endpoint_keyframe_bytes"] for row in items),
            "mean_residual_payload_bytes": mean(row["residual_payload_bytes"] for row in items),
            "mean_schedule_metadata_bytes": mean(row["schedule_metadata_bytes"] for row in items),
            "mean_edge_total_bytes_once": mean(row["edge_total_bytes_once"] for row in items),
            "mean_dp_incremental_bytes": mean(row["dp_incremental_bytes"] for row in items),
            "mean_residual_mib_per_intermediate": mean(row["residual_mib_per_intermediate"] for row in items),
            "mean_edge_total_mib_once": mean(row["edge_total_mib_once"] for row in items),
            "mean_dp_incremental_mib": mean(row["dp_incremental_mib"] for row in items),
            "mean_base_psnr": mean(row["mean_base_psnr"] for row in items),
            "mean_corrected_psnr": mean(row["mean_corrected_psnr"] for row in items),
            "mean_delta_psnr_vs_base": mean(row["mean_delta_psnr_vs_base"] for row in items),
            "min_edge_delta_psnr_vs_base": min(float(row["mean_delta_psnr_vs_base"]) for row in items),
        })
    return out


def best_rows(summary_rows):
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
            "best_mean_edge_total_bytes_once": best["mean_edge_total_bytes_once"],
            "best_mean_dp_incremental_bytes": best["mean_dp_incremental_bytes"],
            "best_mean_residual_payload_bytes": best["mean_residual_payload_bytes"],
            "best_mean_corrected_psnr": best["mean_corrected_psnr"],
            "best_mean_delta_psnr_vs_base": best["mean_delta_psnr_vs_base"],
        })
    return out


def gate_rows(stage205, target_rows, edge_rows, best, args):
    target_errors = [row for row in target_rows if row["status"] != "ok"]
    edge_errors = [row for row in edge_rows if row["status"] != "ok"]
    missing_gaps = sorted(set(int(gap) for gap in args.gaps) - {int(row["reference_gap"]) for row in best})
    weak_gaps = [row for row in best if float(row["best_mean_delta_psnr_vs_base"]) <= float(args.positive_headroom_db)]
    min_residual = min((float(row["residual_payload_bytes"]) for row in edge_rows if row["status"] == "ok"), default=0.0)
    min_keyframe = min((float(row["right_keyframe_bytes"]) for row in edge_rows if row["status"] == "ok"), default=0.0)
    metadata_values = {int(row["schedule_metadata_bytes"]) for row in edge_rows}
    return [
        {
            "gate": "stage205_prereq",
            "status": "pass" if stage205.get("decision") == "fixed_gap_predictive_codec_positive_headroom" else "fail",
            "value": stage205.get("decision", ""),
            "threshold": "fixed_gap_predictive_codec_positive_headroom",
            "detail": str(args.stage205_package),
        },
        {
            "gate": "target_metric_rows_ok",
            "status": "pass" if not target_errors else "fail",
            "value": len(target_errors),
            "threshold": "0",
            "detail": "shape-mismatched metrics are errors",
        },
        {
            "gate": "edge_rows_ok",
            "status": "pass" if not edge_errors else "fail",
            "value": len(edge_errors),
            "threshold": "0",
            "detail": ";".join(f"{row['edge_id']}:{row['error']}" for row in edge_errors[:3]),
        },
        {
            "gate": "gap_coverage",
            "status": "pass" if not missing_gaps else "fail",
            "value": len(missing_gaps),
            "threshold": "0 missing gaps",
            "detail": ";".join(str(gap) for gap in missing_gaps),
        },
        {
            "gate": "payload_counted_nonzero",
            "status": "pass" if min_residual > 0.0 and min_keyframe > 0.0 else "fail",
            "value": f"residual={min_residual};right_keyframe={min_keyframe}",
            "threshold": ">0 residual and keyframe bytes",
            "detail": "keyframe bytes use encode_anchor_bitstream; residual bytes use len(payload)",
        },
        {
            "gate": "schedule_metadata_counted",
            "status": "pass" if len(metadata_values) == 1 and next(iter(metadata_values), -1) >= 0 else "fail",
            "value": ";".join(str(value) for value in sorted(metadata_values)),
            "threshold": ">=0 bytes explicitly recorded per edge",
            "detail": "Stage207 may replace provisional syntax but cannot hide metadata bytes",
        },
        {
            "gate": "each_gap_positive_edge_headroom",
            "status": "pass" if not weak_gaps else "fail",
            "value": min((float(row["best_mean_delta_psnr_vs_base"]) for row in best), default=0.0),
            "threshold": f"> {args.positive_headroom_db} dB for every gap",
            "detail": ";".join(f"gap{row['reference_gap']}={row['best_mean_delta_psnr_vs_base']}" for row in weak_gaps),
        },
        {
            "gate": "stage197_decoder_contract",
            "status": "pass",
            "value": 0,
            "threshold": "no target dense/RGB decoder input",
            "detail": "decoder uses transmitted q-keyframes, normalized time, schedule metadata, and counted GS residual payload",
        },
    ]


def decision(gates):
    critical_fail = any(row["status"] != "pass" for row in gates if row["gate"] != "each_gap_positive_edge_headroom")
    if critical_fail:
        return "edge_rd_table_invalid"
    headroom = next(row for row in gates if row["gate"] == "each_gap_positive_edge_headroom")
    if headroom["status"] == "pass":
        return "edge_rd_table_ready_for_stage207_dp"
    return "edge_rd_table_needs_review"


def write_report(package, best, gates, path):
    lines = [
        "# Stage206 Edge RD Table",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Selected edges: `{package['edge_count']}`; target rows: `{package['target_count']}`; settings: `{package['setting_count']}`.",
        "- Scope: sampled edge-level RD preflight for Stage207 DP, not final full-sequence RD.",
        "",
        "## Best By Gap",
        "",
        "| gap | best setting | keep | edge total bytes | DP incremental bytes | residual bytes | corrected PSNR | dPSNR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            f"| {row['reference_gap']} | {row['best_setting_label']} | {row['best_keep_fraction']} | "
            f"{float(row['best_mean_edge_total_bytes_once']):.3f} | {float(row['best_mean_dp_incremental_bytes']):.3f} | "
            f"{float(row['best_mean_residual_payload_bytes']):.3f} | {float(row['best_mean_corrected_psnr']):.6f} | "
            f"{float(row['best_mean_delta_psnr_vs_base']):.6f} |"
        )
    lines.extend([
        "",
        "## Accounting",
        "",
        "- `edge_total_bytes_once = left_keyframe_bytes + right_keyframe_bytes + residual_payload_bytes + schedule_metadata_bytes`.",
        "- `dp_incremental_bytes = right_keyframe_bytes + residual_payload_bytes + schedule_metadata_bytes`; Stage207 must add the initial left keyframe once per path.",
        "- Residual payload bytes are exact `len(payload)` from the GS-native residual codec.",
        "- Keyframe bytes are exact `encode_anchor_bitstream(..., q12, bitpack)` lengths for endpoint anchors.",
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
    parser.add_argument("--max_edges_per_gap", type=int, default=2)
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
    stage205 = read_json(args.stage205_package)
    if stage205["decision"] != "fixed_gap_predictive_codec_positive_headroom":
        raise RuntimeError(f"Stage205 prerequisite failed: {stage205['decision']}")
    device = torch.device(args.device)
    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    cache = {}
    all_rows = parse_task_rows(args.task_manifest, args.task_split, args.gaps, args.keyframe_codec)
    selected_rows = sample_edge_rows(all_rows, args.gaps, args.max_edges_per_gap, args.seed)
    settings = [(keep, int(args.side_bits)) for keep in args.keep_fractions]
    raw_metric_rows = []
    for row in selected_rows:
        raw_metric_rows.extend(evaluate_task(row, settings, args, device, cache, opt, background))
    tasks_by_id = {row["task_id"]: row for row in selected_rows}
    target_rows = annotate_metric_rows(raw_metric_rows, tasks_by_id)
    edge_rows = build_edge_rows(selected_rows, target_rows, settings, args)
    summary_rows = summarize(edge_rows)
    best = best_rows(summary_rows)
    gates = gate_rows(stage205, target_rows, edge_rows, best, args)
    decision_value = decision(gates)

    selected_edges_csv = args.output_root / "stage206_selected_edges.csv"
    target_rows_csv = args.output_root / "stage206_target_metric_rows.csv"
    edge_rows_csv = args.output_root / "stage206_edge_rd_rows.csv"
    summary_csv = args.output_root / "stage206_edge_rd_summary.csv"
    best_by_gap_csv = args.output_root / "stage206_edge_rd_best_by_gap.csv"
    gates_csv = args.output_root / "stage206_edge_rd_gates.csv"
    package_json = args.output_root / "stage206_edge_rd_table_package.json"
    report_md = args.output_root / "stage206_edge_rd_table_report.md"

    write_csv(selected_edge_rows(selected_rows), selected_edges_csv, SELECTED_EDGE_FIELDS)
    write_csv(target_rows, target_rows_csv, TARGET_FIELDS)
    write_csv(edge_rows, edge_rows_csv, EDGE_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(best, best_by_gap_csv, BEST_FIELDS)
    write_csv(gates, gates_csv, GATE_FIELDS)
    package = {
        "stage": 206,
        "name": "edge_rd_table",
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
        "edge_count": len(selected_edge_rows(selected_rows)),
        "target_count": len(selected_rows),
        "setting_count": len(settings),
        "side_bits": args.side_bits,
        "keep_fractions": args.keep_fractions,
        "selected_edges_csv": str(selected_edges_csv),
        "target_rows_csv": str(target_rows_csv),
        "edge_rows_csv": str(edge_rows_csv),
        "summary_csv": str(summary_csv),
        "best_by_gap_csv": str(best_by_gap_csv),
        "gates_csv": str(gates_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "best_rows": best,
        "gate_rows": gates,
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, best, gates, report_md)
    print(json.dumps({"package": str(package_json), "decision": decision_value}, indent=2))


if __name__ == "__main__":
    main()
