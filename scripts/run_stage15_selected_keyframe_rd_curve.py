import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_stage12_selected_keyframe_reconstruction import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_STAGE1_CACHE,
    run_selected,
    write_csv,
)


DEFAULT_SELECTION_CSV = REPO_ROOT / "experiments/stage13_spaced_keyframe_selection/stage13_spaced_keyframe_selection_summary.csv"
DEFAULT_HEAVY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage15_selected_keyframe_rd_curve")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage15_selected_keyframe_rd_curve"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="*", default=["n3dv", "robot"])
    parser.add_argument("--methods", nargs="*", default=["uniform", "rd_spaced"])
    parser.add_argument("--gaps", nargs="*", type=int, default=[4, 8, 16])
    parser.add_argument("--selection_csv", type=Path, default=DEFAULT_SELECTION_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_segment_length", type=int, default=40)
    parser.add_argument("--save_per_frame", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCH_HOME", "/mnt/hdd2tC/tmp/opencode/torch_home")
    args.summary_root.mkdir(parents=True, exist_ok=True)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for sample in args.samples:
        for method in args.methods:
            for gap in args.gaps:
                print(f"=== Stage15 sample={sample} method={method} gap={gap} ===", flush=True)
                run_args = SimpleNamespace(
                    sample=sample,
                    method=method,
                    reference_gap=gap,
                    selection_csv=args.selection_csv,
                    checkpoint=args.checkpoint,
                    stage1_cache=args.stage1_cache,
                    heavy_root=args.heavy_root,
                    summary_root=args.summary_root,
                    batch_size=args.batch_size,
                    max_segment_length=args.max_segment_length,
                    save_per_frame=args.save_per_frame,
                )
                summary = run_selected(run_args, device)
                summary["stage"] = 15
                summary["method"] = "selected-keyframe RD curve batch"
                rows.append(summary)
                single_path = args.summary_root / f"{sample}_{method}_gap{gap}_summary.json"
                single_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = args.summary_root / "stage15_selected_keyframe_rd_curve_summary.csv"
    summary_path = args.summary_root / "stage15_selected_keyframe_rd_curve_summary.json"
    write_csv(rows, csv_path)
    summary = {
        "stage": 15,
        "mode": "selected-keyframe RD curve batch",
        "selection_csv": str(args.selection_csv),
        "samples": args.samples,
        "methods": args.methods,
        "gaps": args.gaps,
        "rows": rows,
        "csv": str(csv_path),
        "notes": "Uses StreamSplat RGB/depth-conditioned selected-pair inference. This is a keyframe-selection RD curve smoke, not final Gaussian-anchor-only decoding.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "csv": str(csv_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
