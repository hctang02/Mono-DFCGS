import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEAVY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage33_dense_gap1_anchor_export")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage33_dense_gap1_anchor_export"


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from scripts.run_stage6_export_real_anchor_dataset import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_STAGE1_CACHE,
    SAMPLES,
    load_model,
    run_sample_gap,
    write_csv,
)


def unique_anchor_indices(rows):
    out = set()
    for row in rows:
        out.add(int(row["left_index"]))
        out.add(int(row["right_index"]))
    return sorted(out)


def write_summary_csv(rows, path):
    fields = [
        "sample", "total_frames", "exported_pair_count", "unique_anchor_count", "unique_anchor_ratio",
        "total_anchor_mib", "avg_anchor_mib_per_pair", "heavy_root",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache_root", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_pairs_per_sample", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    args.summary_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt = Options()
    opt.resume = str(args.checkpoint)
    opt.compile = False
    opt.input_frames = 1
    opt.output_frames = 2
    opt.epoch = 0
    model = load_model(deepcopy(opt), args.checkpoint, device)

    all_rows = []
    summaries = []
    gap = 1
    for sample in args.samples:
        print(f"=== Stage33 export {sample} gap=1 ===", flush=True)
        selected, total_frames, rows, total_mib = run_sample_gap(sample, gap, args, model, opt, device)
        unique_indices = unique_anchor_indices(rows)
        all_rows.extend(rows)
        summaries.append({
            "sample": sample,
            "total_frames": total_frames,
            "selected_keyframes": selected,
            "exported_pair_count": len(rows),
            "unique_anchor_count": len(unique_indices),
            "unique_anchor_indices": unique_indices,
            "unique_anchor_ratio": len(unique_indices) / max(total_frames, 1),
            "total_anchor_mib": total_mib,
            "avg_anchor_mib_per_pair": total_mib / max(len(rows), 1),
            "heavy_root": str(args.heavy_root),
        })
        torch.cuda.empty_cache()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    manifest_csv = args.summary_root / "stage33_dense_gap1_anchor_manifest.csv"
    manifest_json = args.summary_root / "stage33_dense_gap1_anchor_manifest.json"
    summary_csv = args.summary_root / "stage33_dense_gap1_anchor_summary.csv"
    summary_json = args.summary_root / "stage33_dense_gap1_anchor_summary.json"
    write_csv(all_rows, manifest_csv)
    manifest_json.write_text(json.dumps({"rows": all_rows}, indent=2), encoding="utf-8")
    write_summary_csv([{k: v for k, v in row.items() if k != "selected_keyframes" and k != "unique_anchor_indices"} for row in summaries], summary_csv)
    summary = {
        "stage": 33,
        "mode": "dense gap1 anchor export",
        "checkpoint": str(args.checkpoint),
        "cache_root": str(args.cache_root),
        "heavy_root": str(args.heavy_root),
        "manifest_csv": str(manifest_csv),
        "manifest_json": str(manifest_json),
        "summary_csv": str(summary_csv),
        "summaries": summaries,
        "total_rows": len(all_rows),
        "total_anchor_mib": sum(row["total_anchor_mib"] for row in summaries),
        "notes": "Gap1 pair export covers every frame as a keyframe anchor candidate, but adjacent pair files duplicate anchors across neighboring pairs.",
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
