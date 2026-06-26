import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAVIS_ROOT = Path("/mnt/hdd2tC/tmp/opencode/datasets/DAVIS")
DEFAULT_HEAVY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage61_davis_anchor_export")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage61_davis_anchor_export"

sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage6_export_real_anchor_dataset import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    get_depth,
    get_image,
    load_model,
    normalize_depths,
    save_pair_dataset_item,
    split_pair_anchors,
    static_anchor_from_half,
)


MANIFEST_FIELDS = [
    "dataset",
    "split",
    "sequence",
    "frame_gap",
    "left_index",
    "right_index",
    "segment_length",
    "middle_frame_count",
    "dataset_item",
    "dataset_item_relative",
    "anchor_mib",
    "gaussians_per_anchor",
]

SUMMARY_FIELDS = [
    "dataset",
    "split",
    "sequence",
    "frame_gap",
    "total_frames",
    "exported_pair_count",
    "total_middle_frames",
    "total_anchor_mib",
    "avg_anchor_mib_per_pair",
    "heavy_root",
]


def read_split(path):
    path = Path(path)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sorted_images(path):
    path = Path(path)
    if not path.exists():
        return []
    return sorted(path.glob("*.jpg"), key=lambda p: p.name)


def davis_sequences(root, splits):
    root = Path(root)
    image_root = root / "JPEGImages/Full-Resolution"
    rows = []
    for split in splits:
        for sequence in read_split(root / "ImageSets/2017" / f"{split}.txt"):
            image_dir = image_root / sequence
            if image_dir.exists():
                rows.append((split, sequence))
    return rows


def depth_path_for_frame(image_path):
    return Path(str(image_path).replace("JPEGImages", "depthImages")).with_name(f"{image_path.stem}_pred.png")


def build_pairs(total_frames, gap):
    selected = list(range(0, total_frames, gap))
    if not selected:
        return [], []
    if selected[-1] != total_frames - 1:
        selected.append(total_frames - 1)
    return selected, [(selected[i], selected[i + 1]) for i in range(len(selected) - 1)]


def load_sequence_arrays(root, split, sequence, opt):
    image_dir = Path(root) / "JPEGImages/Full-Resolution" / sequence
    frame_files = sorted_images(image_dir)
    depth_files = [depth_path_for_frame(path) for path in frame_files]
    missing_depth = [path for path in depth_files if not path.exists()]
    if missing_depth:
        raise FileNotFoundError(f"Missing {len(missing_depth)} depth files for DAVIS {split}/{sequence}")
    frames = [get_image(path, opt.image_height, opt.image_width) for path in frame_files]
    depths = [get_depth(path, opt.image_height, opt.image_width) for path in depth_files]
    return frame_files, depth_files, frames, depths


def run_sequence_gap(root, split, sequence, gap, args, model, opt, device):
    frame_files, depth_files, frames_all, depths_all = load_sequence_arrays(root, split, sequence, opt)
    _selected, pairs = build_pairs(len(frame_files), gap)
    if args.max_pairs_per_sequence > 0:
        pairs = pairs[: args.max_pairs_per_sequence]

    pairs_by_len = defaultdict(list)
    for pair in pairs:
        pairs_by_len[pair[1] - pair[0]].append(pair)

    manifest_rows = []
    total_mib = 0.0
    sample_id = f"DAVIS/{split}/{sequence}"
    with torch.no_grad():
        for _seg_len, seg_pairs in sorted(pairs_by_len.items()):
            for start in range(0, len(seg_pairs), args.batch_size):
                batch_pairs = seg_pairs[start:start + args.batch_size]
                batch_frames = []
                batch_depths = []
                for a, b in batch_pairs:
                    batch_frames.append(np.stack([frames_all[a], frames_all[b]], axis=0))
                    batch_depths.append(np.stack([depths_all[a], depths_all[b]], axis=0))
                frames = torch.from_numpy(np.stack(batch_frames, axis=0)).float().to(device) / 255.0
                frames = frames.permute(0, 1, 4, 2, 3)
                depths = torch.from_numpy(np.stack(batch_depths, axis=0)).float().to(device).unsqueeze(2)
                depths = normalize_depths(depths)
                timestamps = torch.tensor([[0.0, 1.0]] * len(batch_pairs), dtype=torch.float32, device=device)
                gs_out = model.forward_gaussians(frames, depths, timestamps)
                first_half, second_half = split_pair_anchors(gs_out["pred_gs"])
                for local_idx, (a, b) in enumerate(batch_pairs):
                    left_anchor = static_anchor_from_half(first_half, local_idx)
                    right_anchor = static_anchor_from_half(second_half, local_idx)
                    rel_path = Path("DAVIS") / split / sequence / f"gap{gap}" / f"pair_{a:06d}_{b:06d}.pt"
                    out_path = args.heavy_root / rel_path
                    anchor_mib = save_pair_dataset_item(
                        out_path, sample_id, gap, a, b, left_anchor, right_anchor, frame_files, depth_files
                    )
                    total_mib += anchor_mib
                    manifest_rows.append({
                        "dataset": "DAVIS",
                        "split": split,
                        "sequence": sequence,
                        "frame_gap": gap,
                        "left_index": a,
                        "right_index": b,
                        "segment_length": b - a,
                        "middle_frame_count": max(b - a - 1, 0),
                        "dataset_item": str(out_path),
                        "dataset_item_relative": str(rel_path),
                        "anchor_mib": anchor_mib,
                        "gaussians_per_anchor": int(left_anchor["rgb"].shape[0]),
                    })
                del gs_out, frames, depths, timestamps
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return len(frame_files), manifest_rows, total_mib


def write_csv(rows, fields, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--sequences", nargs="*", default=[])
    parser.add_argument("--gaps", nargs="+", type=int, default=[16])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_sequences", type=int, default=1, help="0 means all selected sequences")
    parser.add_argument("--max_pairs_per_sequence", type=int, default=1, help="0 means all pairs")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    args.summary_root.mkdir(parents=True, exist_ok=True)

    selected_sequences = davis_sequences(args.davis_root, args.splits)
    if args.sequences:
        allowed = set(args.sequences)
        selected_sequences = [(split, sequence) for split, sequence in selected_sequences if sequence in allowed]
    if args.max_sequences > 0:
        selected_sequences = selected_sequences[: args.max_sequences]
    if not selected_sequences:
        raise RuntimeError("No DAVIS sequences selected for Stage61 export")

    device = torch.device(args.device)
    opt = Options()
    opt.resume = str(args.checkpoint)
    opt.compile = False
    opt.input_frames = 1
    opt.output_frames = 2
    opt.epoch = 0
    model = load_model(deepcopy(opt), args.checkpoint, str(device))

    all_rows = []
    summaries = []
    for gap in args.gaps:
        for split, sequence in selected_sequences:
            print(f"=== Stage61 export DAVIS {split}/{sequence} gap={gap} ===", flush=True)
            total_frames, rows, total_mib = run_sequence_gap(args.davis_root, split, sequence, gap, args, model, opt, device)
            all_rows.extend(rows)
            summaries.append({
                "dataset": "DAVIS",
                "split": split,
                "sequence": sequence,
                "frame_gap": gap,
                "total_frames": total_frames,
                "exported_pair_count": len(rows),
                "total_middle_frames": sum(int(row["middle_frame_count"]) for row in rows),
                "total_anchor_mib": total_mib,
                "avg_anchor_mib_per_pair": total_mib / max(len(rows), 1),
                "heavy_root": str(args.heavy_root),
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    manifest_csv = args.summary_root / "stage61_davis_anchor_export_manifest.csv"
    manifest_json = args.summary_root / "stage61_davis_anchor_export_manifest.json"
    summary_csv = args.summary_root / "stage61_davis_anchor_export_summary.csv"
    summary_json = args.summary_root / "stage61_davis_anchor_export_summary.json"
    write_csv(all_rows, MANIFEST_FIELDS, manifest_csv)
    manifest_json.write_text(json.dumps({"rows": all_rows}, indent=2), encoding="utf-8")
    write_csv(summaries, SUMMARY_FIELDS, summary_csv)
    summary = {
        "stage": 61,
        "mode": "DAVIS anchor export smoke/full-capable script",
        "davis_root": str(args.davis_root),
        "checkpoint": str(args.checkpoint),
        "heavy_root": str(args.heavy_root),
        "summary_root": str(args.summary_root),
        "splits": args.splits,
        "sequences": [sequence for _split, sequence in selected_sequences],
        "gaps": args.gaps,
        "batch_size": args.batch_size,
        "max_sequences": args.max_sequences,
        "max_pairs_per_sequence": args.max_pairs_per_sequence,
        "manifest_csv": str(manifest_csv),
        "summary_csv": str(summary_csv),
        "total_rows": len(all_rows),
        "total_anchor_mib": sum(float(row["total_anchor_mib"]) for row in summaries),
        "summaries": summaries,
        "notes": "Default args intentionally run a one-sequence one-pair smoke export. Use explicit max_sequences=0 and max_pairs_per_sequence=0 only when storage is sufficient.",
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_json),
        "manifest_csv": str(manifest_csv),
        "summary_csv": str(summary_csv),
        "rows": len(all_rows),
        "total_anchor_mib": summary["total_anchor_mib"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
