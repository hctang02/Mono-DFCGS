import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage97_residual_predictor_task_manifest"
DEFAULT_TASK_MANIFEST = REPO_ROOT / "experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_tasks.csv"
DEFAULT_DENSE_MANIFEST = REPO_ROOT / "experiments/stage61_davis_anchor_export_data_full/stage61_davis_anchor_export_manifest.csv"


ROW_FIELDS = [
    "stage97_task_id",
    "source_task_id",
    "task_split",
    "dataset",
    "split",
    "sequence",
    "codec",
    "bits",
    "reference_gap",
    "left_index",
    "right_index",
    "target_index",
    "normalized_time",
    "target_rgb_path",
    "left_anchor_source_item",
    "left_anchor_source_side",
    "right_anchor_source_item",
    "right_anchor_source_side",
    "target_anchor_source_item",
    "target_anchor_source_side",
    "base_methods",
    "keep_fraction",
    "side_bits",
    "residual_codec",
    "label_generation",
    "decoder_inputs",
]

SUMMARY_FIELDS = [
    "task_split",
    "codec",
    "reference_gap",
    "task_count",
    "sequence_count",
    "potential_base_method_label_count",
]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_task_rows(path, task_split, codecs, gaps):
    codec_set = set(codecs)
    gap_set = set(gaps)
    rows = []
    for row in read_csv(path):
        if row["task_split"] != task_split:
            continue
        if codec_set and row["codec"] not in codec_set:
            continue
        gap = int(row["reference_gap"])
        if gap_set and gap not in gap_set:
            continue
        item = dict(row)
        item["bits"] = int(item["bits"])
        item["reference_gap"] = gap
        item["left_index"] = int(item["left_index"])
        item["right_index"] = int(item["right_index"])
        item["target_index"] = int(item["target_index"])
        item["normalized_time"] = float(item["normalized_time"])
        rows.append(item)
    return rows


def build_dense_index(path, split_filter):
    split_set = set(split_filter)
    index = {}
    for row in read_csv(path):
        if row["split"] not in split_set or int(row["frame_gap"]) != 1:
            continue
        item = row["dataset_item"]
        if not Path(item).exists():
            continue
        key_base = (row["dataset"], row["split"], row["sequence"])
        index.setdefault((*key_base, int(row["left_index"])), (item, "left_anchor"))
        index.setdefault((*key_base, int(row["right_index"])), (item, "right_anchor"))
    return index


def summarize(rows, base_method_count):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task_split"], row["codec"], row["reference_gap"])].append(row)
    out = []
    for (task_split, codec, gap), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], int(item[0][2]))):
        out.append({
            "task_split": task_split,
            "codec": codec,
            "reference_gap": int(gap),
            "task_count": len(items),
            "sequence_count": len({row["sequence"] for row in items}),
            "potential_base_method_label_count": len(items) * base_method_count,
        })
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--task_splits", nargs="+", default=["train", "eval"])
    parser.add_argument("--codecs", nargs="+", default=["q12"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--base_methods", nargs="+", default=["linear", "stage65_adapter"])
    parser.add_argument("--keep_fraction", type=float, default=0.1)
    parser.add_argument("--side_bits", type=int, default=6)
    parser.add_argument("--residual_codec", default="entropy_q6_top10_sorted_delta_zlib")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    task_rows = []
    for task_split in args.task_splits:
        task_rows.extend(parse_task_rows(args.task_manifest, task_split, args.codecs, args.gaps))
    data_splits = sorted({row["split"] for row in task_rows})
    dense_index = build_dense_index(args.dense_manifest, data_splits)
    base_methods = ";".join(args.base_methods)
    rows = []
    missing_dense = []
    for row in task_rows:
        dense_key = (row["dataset"], row["split"], row["sequence"], row["target_index"])
        if dense_key not in dense_index:
            missing_dense.append({
                "source_task_id": row["task_id"],
                "dataset": row["dataset"],
                "split": row["split"],
                "sequence": row["sequence"],
                "target_index": row["target_index"],
            })
            continue
        target_item, target_side = dense_index[dense_key]
        rows.append({
            "stage97_task_id": f"stage97_{len(rows):08d}",
            "source_task_id": row["task_id"],
            "task_split": row["task_split"],
            "dataset": row["dataset"],
            "split": row["split"],
            "sequence": row["sequence"],
            "codec": row["codec"],
            "bits": row["bits"],
            "reference_gap": row["reference_gap"],
            "left_index": row["left_index"],
            "right_index": row["right_index"],
            "target_index": row["target_index"],
            "normalized_time": row["normalized_time"],
            "target_rgb_path": row["target_rgb_path"],
            "left_anchor_source_item": row["left_anchor_source_item"],
            "left_anchor_source_side": row["left_anchor_source_side"],
            "right_anchor_source_item": row["right_anchor_source_item"],
            "right_anchor_source_side": row["right_anchor_source_side"],
            "target_anchor_source_item": target_item,
            "target_anchor_source_side": target_side,
            "base_methods": base_methods,
            "keep_fraction": args.keep_fraction,
            "side_bits": args.side_bits,
            "residual_codec": args.residual_codec,
            "label_generation": "regenerate_teacher_residual_from_target_anchor_at_training_or_encoder_side",
            "decoder_inputs": "left_right_keyframe_anchors_time_and_transmitted_residual_sideinfo",
        })

    summary_rows = summarize(rows, len(args.base_methods))
    rows_csv = args.summary_root / "stage97_residual_predictor_tasks.csv"
    summary_csv = args.summary_root / "stage97_residual_predictor_task_summary.csv"
    missing_csv = args.summary_root / "stage97_residual_predictor_missing_dense_targets.csv"
    summary_json = args.summary_root / "stage97_residual_predictor_task_manifest_summary.json"
    report_md = args.summary_root / "stage97_residual_predictor_task_manifest_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(missing_dense, missing_csv, ["source_task_id", "dataset", "split", "sequence", "target_index"])

    summary = {
        "stage": 97,
        "mode": "residual predictor task manifest",
        "task_manifest": str(args.task_manifest),
        "dense_manifest": str(args.dense_manifest),
        "task_splits": args.task_splits,
        "codecs": args.codecs,
        "gaps": args.gaps,
        "base_methods": args.base_methods,
        "keep_fraction": args.keep_fraction,
        "side_bits": args.side_bits,
        "residual_codec": args.residual_codec,
        "task_count": len(rows),
        "missing_dense_count": len(missing_dense),
        "potential_base_method_label_count": len(rows) * len(args.base_methods),
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "missing_csv": str(missing_csv),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "notes": [
            "No heavy residual labels, payloads, anchors, or tensors are copied into this package.",
            "Teacher labels can be regenerated from the target dense anchor during training or encoder-side analysis.",
            "Decoder-side inputs must not include target dense anchors; transmitted side-info must be counted in rate.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Stage97 Residual Predictor Task Manifest",
        "",
        "## Scope",
        "",
        f"- task count: `{len(rows)}`",
        f"- missing dense target count: `{len(missing_dense)}`",
        f"- potential base-method labels: `{len(rows) * len(args.base_methods)}`",
        f"- codecs: `{args.codecs}`",
        f"- gaps: `{args.gaps}`",
        f"- base methods: `{args.base_methods}`",
        f"- residual codec: `{args.residual_codec}`",
        "- no heavy labels or tensors are stored",
        "",
        "## Summary",
        "",
        "| split | codec | gap | tasks | sequences | potential labels |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in summary_rows:
        lines.append(
            f"| {item['task_split']} | {item['codec']} | {item['reference_gap']} | {item['task_count']} | {item['sequence_count']} | {item['potential_base_method_label_count']} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- tasks CSV: `{rows_csv}`",
        f"- summary CSV: `{summary_csv}`",
        f"- missing dense targets CSV: `{missing_csv}`",
        "",
        "## Notes",
        "",
        "- This package prepares supervised residual predictor data but does not train a model.",
        "- Target dense anchors are training/encoder-side label sources, not decoder-side inputs.",
        "- Any transmitted residual side-info remains part of total RD rate.",
    ])
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "task_count": len(rows),
        "missing_dense_count": len(missing_dense),
    }, indent=2))


if __name__ == "__main__":
    main()
