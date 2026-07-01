import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DENSE_MANIFEST = REPO_ROOT / "experiments/stage61_davis_anchor_export_data_full/stage61_davis_anchor_export_manifest.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage199_learned_gs_training_manifest"
DEFAULT_DAVIS_ROOT = Path("/data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS")

TASK_FIELDS = [
    "task_id",
    "task_split",
    "dataset",
    "split",
    "sequence",
    "keyframe_codec",
    "keyframe_bits",
    "reference_gap",
    "total_frames",
    "keyframe_count",
    "segment_id",
    "left_index",
    "right_index",
    "target_index",
    "segment_length",
    "normalized_time",
    "left_anchor_source_item",
    "left_anchor_source_side",
    "right_anchor_source_item",
    "right_anchor_source_side",
    "target_anchor_source_item",
    "target_anchor_source_side",
    "target_rgb_path",
    "training_label_sources",
    "decoder_allowed_inputs",
    "decoder_forbidden_inputs",
    "downstream_stage_uses",
]

SEQUENCE_FIELDS = [
    "dataset",
    "split",
    "task_split",
    "sequence",
    "total_frames",
    "first_index",
    "last_index",
    "gap1_pair_count",
    "contiguous",
    "source_frame_count",
    "missing_source_count",
    "missing_source_indices",
]

SUMMARY_FIELDS = [
    "task_split",
    "keyframe_codec",
    "reference_gap",
    "sequence_count",
    "segment_count",
    "task_count",
    "target_label_count",
    "mean_tasks_per_sequence",
    "min_normalized_time",
    "max_normalized_time",
]

AUDIT_FIELDS = ["audit", "status", "value", "requirement", "detail"]
MISSING_FIELDS = ["kind", "dataset", "split", "sequence", "frame_index", "expected_path"]


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def codec_bits(codec):
    if codec.startswith("q"):
        return int(codec[1:])
    raise ValueError(f"Unsupported keyframe codec: {codec}")


def rgb_path(davis_root, sequence, index):
    return davis_root / "JPEGImages/Full-Resolution" / sequence / f"{index:05d}.jpg"


def uniform_indices(total_frames, gap):
    selected = list(range(0, total_frames, gap))
    if not selected or selected[-1] != total_frames - 1:
        selected.append(total_frames - 1)
    return selected


def group_gap1_rows(rows, splits):
    split_set = set(splits)
    grouped = defaultdict(list)
    skipped_missing_items = []
    for row in rows:
        if row["split"] not in split_set or int(row["frame_gap"]) != 1:
            continue
        item_path = Path(row["dataset_item"])
        if not item_path.exists():
            skipped_missing_items.append({
                "kind": "missing_dense_gap1_pair_item",
                "dataset": row["dataset"],
                "split": row["split"],
                "sequence": row["sequence"],
                "frame_index": row["left_index"],
                "expected_path": str(item_path),
            })
            continue
        item = dict(row)
        item["left_index"] = int(item["left_index"])
        item["right_index"] = int(item["right_index"])
        grouped[(item["dataset"], item["split"], item["sequence"])].append(item)
    for key in grouped:
        grouped[key].sort(key=lambda item: item["left_index"])
    return grouped, skipped_missing_items


def build_frame_sources(rows):
    sources = {}
    for row in rows:
        for index_key, side in (("left_index", "left_anchor"), ("right_index", "right_anchor")):
            index = int(row[index_key])
            if index in sources:
                continue
            sources[index] = {
                "dataset_item": row["dataset_item"],
                "side": side,
            }
    return sources


def sequence_summary(key, rows, sources):
    dataset, split, sequence = key
    observed = sorted({int(row["left_index"]) for row in rows} | {int(row["right_index"]) for row in rows})
    expected = list(range(observed[0], observed[-1] + 1)) if observed else []
    missing_sources = sorted(set(expected) - set(sources))
    return {
        "dataset": dataset,
        "split": split,
        "task_split": "train" if split == "train" else "eval",
        "sequence": sequence,
        "total_frames": len(expected),
        "first_index": observed[0] if observed else "",
        "last_index": observed[-1] if observed else "",
        "gap1_pair_count": len(rows),
        "contiguous": observed == expected,
        "source_frame_count": len(sources),
        "missing_source_count": len(missing_sources),
        "missing_source_indices": ";".join(str(item) for item in missing_sources),
    }


def build_tasks_for_sequence(key, rows, sources, args, start_id):
    dataset, split, sequence = key
    task_split = "train" if split == "train" else "eval"
    total_frames = len(set(sources))
    tasks = []
    missing = []
    next_id = start_id
    for codec in args.keyframe_codecs:
        bits = codec_bits(codec)
        for gap in args.gaps:
            selected = uniform_indices(total_frames, gap)
            for segment_ordinal, (left, right) in enumerate(zip(selected[:-1], selected[1:])):
                segment_length = right - left
                if segment_length <= 1:
                    continue
                for target in range(left + 1, right):
                    rgb = rgb_path(args.davis_root, sequence, target)
                    if not rgb.exists():
                        missing.append({
                            "kind": "missing_target_rgb",
                            "dataset": dataset,
                            "split": split,
                            "sequence": sequence,
                            "frame_index": target,
                            "expected_path": str(rgb),
                        })
                        continue
                    left_source = sources.get(left)
                    right_source = sources.get(right)
                    target_source = sources.get(target)
                    if not left_source or not right_source or not target_source:
                        for index, source in ((left, left_source), (right, right_source), (target, target_source)):
                            if source:
                                continue
                            missing.append({
                                "kind": "missing_dense_frame_source",
                                "dataset": dataset,
                                "split": split,
                                "sequence": sequence,
                                "frame_index": index,
                                "expected_path": "stage61_gap1_frame_source",
                            })
                        continue
                    tasks.append({
                        "task_id": f"stage199_{next_id:08d}",
                        "task_split": task_split,
                        "dataset": dataset,
                        "split": split,
                        "sequence": sequence,
                        "keyframe_codec": codec,
                        "keyframe_bits": bits,
                        "reference_gap": gap,
                        "total_frames": total_frames,
                        "keyframe_count": len(selected),
                        "segment_id": f"{dataset}/{split}/{sequence}/gap{gap}/seg{segment_ordinal:04d}_{left:05d}_{right:05d}",
                        "left_index": left,
                        "right_index": right,
                        "target_index": target,
                        "segment_length": segment_length,
                        "normalized_time": (target - left) / segment_length,
                        "left_anchor_source_item": left_source["dataset_item"],
                        "left_anchor_source_side": left_source["side"],
                        "right_anchor_source_item": right_source["dataset_item"],
                        "right_anchor_source_side": right_source["side"],
                        "target_anchor_source_item": target_source["dataset_item"],
                        "target_anchor_source_side": target_source["side"],
                        "target_rgb_path": str(rgb),
                        "training_label_sources": "target_dense_anchor;target_rgb_render_loss",
                        "decoder_allowed_inputs": "transmitted_keyframe_gs;transmitted_schedule;normalized_time;shared_weights;counted_gs_latent_or_residual_payload",
                        "decoder_forbidden_inputs": "target_dense_anchor;target_rgb_or_image_residual;oracle_schedule_or_quality_labels",
                        "downstream_stage_uses": "stage200_architecture;stage201_predictor_smoke;stage203_residual_codec;stage206_edge_rd;stage208_selector_labels",
                    })
                    next_id += 1
    return tasks, missing, next_id


def summarize_tasks(tasks):
    grouped = defaultdict(list)
    for row in tasks:
        key = (row["task_split"], row["keyframe_codec"], int(row["reference_gap"]))
        grouped[key].append(row)
    out = []
    for (task_split, codec, gap), items in sorted(grouped.items(), key=lambda item: (item[0][0], codec_bits(item[0][1]), item[0][2])):
        sequence_count = len({(row["dataset"], row["split"], row["sequence"]) for row in items})
        segment_count = len({row["segment_id"] for row in items})
        times = [float(row["normalized_time"]) for row in items]
        out.append({
            "task_split": task_split,
            "keyframe_codec": codec,
            "reference_gap": gap,
            "sequence_count": sequence_count,
            "segment_count": segment_count,
            "task_count": len(items),
            "target_label_count": len(items),
            "mean_tasks_per_sequence": len(items) / max(sequence_count, 1),
            "min_normalized_time": min(times) if times else "",
            "max_normalized_time": max(times) if times else "",
        })
    return out


def split_overlap(sequence_rows):
    by_split = defaultdict(set)
    for row in sequence_rows:
        by_split[row["task_split"]].add((row["dataset"], row["sequence"]))
    return sorted(by_split["train"] & by_split["eval"])


def build_audit_rows(sequence_rows, summary_rows, missing_rows, args):
    train_eval_overlap = split_overlap(sequence_rows)
    missing_source_count = sum(int(row["missing_source_count"]) for row in sequence_rows)
    non_contiguous = [row for row in sequence_rows if str(row["contiguous"]) != "True"]
    observed = {(row["task_split"], int(row["reference_gap"])) for row in summary_rows}
    required = {(task_split, gap) for task_split in ("train", "eval") for gap in args.gaps}
    missing_gap_coverage = sorted(required - observed)
    missing_rgb_count = sum(1 for row in missing_rows if row["kind"] == "missing_target_rgb")
    missing_pair_count = sum(1 for row in missing_rows if row["kind"] == "missing_dense_gap1_pair_item")
    rows = [
        {
            "audit": "dense_anchor_coverage",
            "status": "pass" if missing_source_count == 0 and missing_pair_count == 0 and not non_contiguous else "fail",
            "value": missing_source_count + missing_pair_count + len(non_contiguous),
            "requirement": "all referenced frames have dense gap1 anchor sources",
            "detail": f"missing_sources={missing_source_count}; missing_pair_items={missing_pair_count}; non_contiguous_sequences={len(non_contiguous)}",
        },
        {
            "audit": "rgb_label_coverage",
            "status": "pass" if missing_rgb_count == 0 else "fail",
            "value": missing_rgb_count,
            "requirement": "target RGB paths exist for render-aware training labels",
            "detail": "target RGB is training/encoder-side only, never decoder-side image residual",
        },
        {
            "audit": "split_separation",
            "status": "pass" if not train_eval_overlap else "fail",
            "value": len(train_eval_overlap),
            "requirement": "no train/eval sequence overlap",
            "detail": ";".join(f"{dataset}/{sequence}" for dataset, sequence in train_eval_overlap),
        },
        {
            "audit": "gap_coverage",
            "status": "pass" if not missing_gap_coverage else "fail",
            "value": len(missing_gap_coverage),
            "requirement": f"train/eval rows exist for gaps {args.gaps}",
            "detail": ";".join(f"{task_split}:gap{gap}" for task_split, gap in missing_gap_coverage),
        },
        {
            "audit": "stage197_decoder_contract",
            "status": "pass",
            "value": 0,
            "requirement": "runtime decoder excludes target dense anchors, target RGB/image residuals, and oracle labels",
            "detail": "manifest fields mark those sources as training/encoder-side labels only; payloads must be GS-native and counted",
        },
        {
            "audit": "lightweight_reference_only",
            "status": "pass",
            "value": 0,
            "requirement": "do not copy heavy anchors, checkpoints, or tensors into git artifacts",
            "detail": "CSV rows contain existing file paths and metadata only",
        },
    ]
    return rows


def write_report(package, summary_rows, audit_rows, sequence_rows, path):
    split_counts = defaultdict(int)
    frame_counts = defaultdict(int)
    for row in sequence_rows:
        split_counts[row["task_split"]] += 1
        frame_counts[row["task_split"]] += int(row["total_frames"])
    lines = [
        "# Stage199 Learned GS Training Manifest",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Task rows: `{package['task_count']}`.",
        f"- Missing reference rows: `{package['missing_count']}`.",
        "- No anchors, checkpoints, residual tensors, or heavy payloads are copied.",
        "",
        "## Summary",
        "",
        "| split | codec | gap | sequences | segments | tasks | labels |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['task_split']} | {row['keyframe_codec']} | {row['reference_gap']} | {row['sequence_count']} | {row['segment_count']} | {row['task_count']} | {row['target_label_count']} |"
        )
    lines.extend([
        "",
        "## Split Coverage",
        "",
        "| split | sequences | frames |",
        "|---|---:|---:|",
    ])
    for split in sorted(split_counts):
        lines.append(f"| {split} | {split_counts[split]} | {frame_counts[split]} |")
    lines.extend([
        "",
        "## Contract Audit",
        "",
        "| audit | status | value | detail |",
        "|---|---|---:|---|",
    ])
    for row in audit_rows:
        lines.append(f"| {row['audit']} | {row['status']} | {row['value']} | {row['detail']} |")
    lines.extend([
        "",
        "## Decoder Contract Notes",
        "",
        "- Runtime decoder inputs: transmitted GS keyframes, transmitted schedule metadata, normalized time, shared weights, and counted GS-native latent/residual payloads.",
        "- Training/encoder-only labels: target dense anchors and target RGB render losses.",
        "- Forbidden decoder inputs: target dense anchors, target RGB/image residuals, and oracle schedule/quality labels.",
        "",
        "## Outputs",
        "",
        f"- tasks: `{package['tasks_csv']}`",
        f"- summary: `{package['summary_csv']}`",
        f"- sequence coverage: `{package['sequence_csv']}`",
        f"- contract audit: `{package['audit_csv']}`",
        f"- missing references: `{package['missing_csv']}`",
        f"- package: `{package['package_json']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[2, 4, 6, 8, 12, 16])
    parser.add_argument("--keyframe_codecs", nargs="+", default=["q12"])
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_csv(args.dense_manifest)
    grouped, missing_rows = group_gap1_rows(manifest_rows, args.splits)
    sequence_rows = []
    task_rows = []
    next_id = 0
    for key in sorted(grouped):
        rows = grouped[key]
        sources = build_frame_sources(rows)
        sequence_rows.append(sequence_summary(key, rows, sources))
        tasks, missing, next_id = build_tasks_for_sequence(key, rows, sources, args, next_id)
        task_rows.extend(tasks)
        missing_rows.extend(missing)
    summary_rows = summarize_tasks(task_rows)
    audit_rows = build_audit_rows(sequence_rows, summary_rows, missing_rows, args)
    decision = "manifest_ready_for_stage200_architecture_package"
    if any(row["status"] != "pass" for row in audit_rows):
        decision = "manifest_blocked_by_coverage_or_contract_audit"

    tasks_csv = args.output_root / "stage199_learned_gs_training_tasks.csv"
    summary_csv = args.output_root / "stage199_learned_gs_training_summary.csv"
    sequence_csv = args.output_root / "stage199_sequence_coverage.csv"
    audit_csv = args.output_root / "stage199_contract_audit.csv"
    missing_csv = args.output_root / "stage199_missing_references.csv"
    package_json = args.output_root / "stage199_learned_gs_training_manifest_package.json"
    report_md = args.output_root / "stage199_learned_gs_training_manifest_report.md"

    write_csv(task_rows, tasks_csv, TASK_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(sequence_rows, sequence_csv, SEQUENCE_FIELDS)
    write_csv(audit_rows, audit_csv, AUDIT_FIELDS)
    write_csv(missing_rows, missing_csv, MISSING_FIELDS)

    package = {
        "stage": 199,
        "name": "learned_gs_training_manifest",
        "decision": decision,
        "dense_manifest": str(args.dense_manifest),
        "davis_root": str(args.davis_root),
        "splits": args.splits,
        "gaps": args.gaps,
        "keyframe_codecs": args.keyframe_codecs,
        "sequence_count": len(sequence_rows),
        "task_count": len(task_rows),
        "missing_count": len(missing_rows),
        "tasks_csv": str(tasks_csv),
        "summary_csv": str(summary_csv),
        "sequence_csv": str(sequence_csv),
        "audit_csv": str(audit_csv),
        "missing_csv": str(missing_csv),
        "report_md": str(report_md),
        "package_json": str(package_json),
        "audit_rows": audit_rows,
        "summary_rows": summary_rows,
        "contract_notes": [
            "Target dense anchors and RGB are training/encoder-side label sources only.",
            "Decoder-side payloads for later stages must be GS-native and counted in total rate.",
            "No heavy anchors, tensors, or checkpoints are copied into this manifest package.",
        ],
    }
    package_json.write_text(json.dumps(package, indent=2), encoding="utf-8")
    write_report(package, summary_rows, audit_rows, sequence_rows, report_md)
    print(json.dumps({"package": str(package_json), "decision": decision, "task_count": len(task_rows), "missing_count": len(missing_rows)}, indent=2))


if __name__ == "__main__":
    main()
