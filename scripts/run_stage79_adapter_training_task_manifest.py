import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments/stage61_davis_anchor_export_data_full/stage61_davis_anchor_export_manifest.csv"
DEFAULT_DAVIS_ROOT = Path("/data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage79_adapter_training_task_manifest"

TASK_FIELDS = [
    "task_id",
    "task_split",
    "dataset",
    "split",
    "sequence",
    "codec",
    "bits",
    "reference_gap",
    "total_frames",
    "keyframe_count",
    "left_index",
    "right_index",
    "target_index",
    "segment_length",
    "normalized_time",
    "left_rgb_path",
    "right_rgb_path",
    "target_rgb_path",
    "left_anchor_source_item",
    "left_anchor_source_side",
    "right_anchor_source_item",
    "right_anchor_source_side",
]

SEQUENCE_FIELDS = [
    "dataset",
    "split",
    "sequence",
    "total_frames",
    "contiguous",
    "first_index",
    "last_index",
    "gap1_pair_count",
]

SUMMARY_FIELDS = [
    "task_split",
    "codec",
    "bits",
    "reference_gap",
    "sequence_count",
    "task_count",
    "mean_tasks_per_sequence",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def codec_bits(codec):
    if codec.startswith("q"):
        return int(codec[1:])
    raise ValueError(codec)


def uniform_indices(total_frames, gap):
    selected = list(range(0, total_frames, gap))
    if not selected or selected[-1] != total_frames - 1:
        selected.append(total_frames - 1)
    return selected


def rgb_path(davis_root, split, sequence, index):
    path = Path(davis_root) / "JPEGImages/Full-Resolution" / sequence / f"{index:05d}.jpg"
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def group_gap1_rows(manifest_rows, splits):
    split_set = set(splits)
    grouped = defaultdict(list)
    for row in manifest_rows:
        if row["split"] not in split_set:
            continue
        if int(row["frame_gap"]) != 1:
            continue
        if not Path(row["dataset_item"]).exists():
            continue
        row = dict(row)
        row["left_index"] = int(row["left_index"])
        row["right_index"] = int(row["right_index"])
        grouped[(row["dataset"], row["split"], row["sequence"])].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda item: item["left_index"])
    return grouped


def build_anchor_sources(rows):
    sources = {}
    for row in rows:
        for index_key, side in [("left_index", "left_anchor"), ("right_index", "right_anchor")]:
            index = int(row[index_key])
            if index in sources:
                continue
            sources[index] = {
                "dataset_item": row["dataset_item"],
                "side": side,
            }
    return sources


def sequence_summary(key, rows):
    dataset, split, sequence = key
    indices = sorted({row["left_index"] for row in rows} | {row["right_index"] for row in rows})
    expected = list(range(indices[0], indices[-1] + 1)) if indices else []
    return {
        "dataset": dataset,
        "split": split,
        "sequence": sequence,
        "total_frames": len(indices),
        "contiguous": indices == expected,
        "first_index": indices[0] if indices else None,
        "last_index": indices[-1] if indices else None,
        "gap1_pair_count": len(rows),
    }


def build_tasks_for_sequence(key, rows, args, task_id_start):
    dataset, split, sequence = key
    summary = sequence_summary(key, rows)
    if not summary["contiguous"]:
        raise RuntimeError(f"Non-contiguous gap1 anchors for {key}")
    total_frames = int(summary["total_frames"])
    sources = build_anchor_sources(rows)
    missing_sources = sorted(set(range(total_frames)) - set(sources))
    if missing_sources:
        raise RuntimeError(f"Missing anchor sources for {key}: {missing_sources[:10]}")
    task_split = "train" if split == "train" else "eval"
    tasks = []
    task_id = task_id_start
    for codec in args.codecs:
        bits = codec_bits(codec)
        for gap in args.gaps:
            selected = uniform_indices(total_frames, gap)
            for left, right in zip(selected[:-1], selected[1:]):
                segment_length = right - left
                if segment_length <= 1:
                    continue
                for target in range(left + 1, right):
                    left_source = sources[left]
                    right_source = sources[right]
                    tasks.append({
                        "task_id": f"stage79_{task_id:08d}",
                        "task_split": task_split,
                        "dataset": dataset,
                        "split": split,
                        "sequence": sequence,
                        "codec": codec,
                        "bits": bits,
                        "reference_gap": gap,
                        "total_frames": total_frames,
                        "keyframe_count": len(selected),
                        "left_index": left,
                        "right_index": right,
                        "target_index": target,
                        "segment_length": segment_length,
                        "normalized_time": (target - left) / segment_length,
                        "left_rgb_path": rgb_path(args.davis_root, split, sequence, left),
                        "right_rgb_path": rgb_path(args.davis_root, split, sequence, right),
                        "target_rgb_path": rgb_path(args.davis_root, split, sequence, target),
                        "left_anchor_source_item": left_source["dataset_item"],
                        "left_anchor_source_side": left_source["side"],
                        "right_anchor_source_item": right_source["dataset_item"],
                        "right_anchor_source_side": right_source["side"],
                    })
                    task_id += 1
    return tasks, task_id


def build_summary(tasks):
    grouped = defaultdict(list)
    sequences = defaultdict(set)
    for task in tasks:
        key = (task["task_split"], task["codec"], int(task["reference_gap"]))
        grouped[key].append(task)
        sequences[key].add((task["dataset"], task["split"], task["sequence"]))
    out = []
    for (task_split, codec, gap), items in sorted(grouped.items(), key=lambda item: (item[0][0], codec_bits(item[0][1]), item[0][2])):
        seq_count = len(sequences[(task_split, codec, gap)])
        out.append({
            "task_split": task_split,
            "codec": codec,
            "bits": codec_bits(codec),
            "reference_gap": gap,
            "sequence_count": seq_count,
            "task_count": len(items),
            "mean_tasks_per_sequence": len(items) / max(seq_count, 1),
        })
    return out


def write_report(summary_rows, sequence_rows, path):
    lines = [
        "# Stage79 Adapter Training Task Manifest",
        "",
        "## Summary",
        "",
        "| split | codec | gap | sequences | tasks | tasks / sequence |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['task_split']} | {row['codec']} | {row['reference_gap']} | {row['sequence_count']} | {row['task_count']} | {row['mean_tasks_per_sequence']} |"
        )
    split_counts = defaultdict(int)
    frame_counts = defaultdict(int)
    for row in sequence_rows:
        split_counts[row["split"]] += 1
        frame_counts[row["split"]] += int(row["total_frames"])
    lines.extend([
        "",
        "## Sequence Coverage",
        "",
        "| split | sequences | frames |",
        "|---|---:|---:|",
    ])
    for split in sorted(split_counts):
        lines.append(f"| {split} | {split_counts[split]} | {frame_counts[split]} |")
    lines.extend([
        "",
        "## Notes",
        "",
        "- This manifest does not copy anchor tensors; it references Stage61 gap1 `.pt` items and source sides.",
        "- `task_split=train` maps to DAVIS train, and `task_split=eval` maps to DAVIS val.",
        "- Codecs are planned input quantization settings for Stage80 training/evaluation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--codecs", nargs="+", default=["q10", "q12"])
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_csv(args.manifest)
    grouped = group_gap1_rows(manifest_rows, args.splits)
    tasks = []
    sequence_rows = []
    task_id = 0
    for key in sorted(grouped):
        seq_summary = sequence_summary(key, grouped[key])
        sequence_rows.append(seq_summary)
        seq_tasks, task_id = build_tasks_for_sequence(key, grouped[key], args, task_id)
        tasks.extend(seq_tasks)
    summary_rows = build_summary(tasks)
    tasks_csv = args.summary_root / "stage79_adapter_training_tasks.csv"
    sequence_csv = args.summary_root / "stage79_adapter_sequence_summary.csv"
    summary_csv = args.summary_root / "stage79_adapter_task_summary.csv"
    report_md = args.summary_root / "stage79_adapter_training_task_manifest_report.md"
    summary_json = args.summary_root / "stage79_adapter_training_task_manifest_summary.json"
    write_csv(tasks, tasks_csv, TASK_FIELDS)
    write_csv(sequence_rows, sequence_csv, SEQUENCE_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_report(summary_rows, sequence_rows, report_md)
    summary = {
        "stage": 79,
        "mode": "adapter training task manifest",
        "manifest": str(args.manifest),
        "davis_root": str(args.davis_root),
        "splits": args.splits,
        "gaps": args.gaps,
        "codecs": args.codecs,
        "sequence_count": len(sequence_rows),
        "task_count": len(tasks),
        "tasks_csv": str(tasks_csv),
        "sequence_csv": str(sequence_csv),
        "summary_csv": str(summary_csv),
        "report_md": str(report_md),
        "summary_json": str(summary_json),
        "task_summary": summary_rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_json),
        "tasks_csv": str(tasks_csv),
        "task_count": len(tasks),
        "sequence_count": len(sequence_rows),
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
