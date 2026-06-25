import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import gaussian_renderer_dynamic  # noqa: E402
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, unflatten_static_anchor  # noqa: E402
from mono_dfcgs.render_adapter import static_anchor_to_single_frame_gaussians  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "experiments/stage6_real_anchor_dataset/stage6_real_anchor_dataset_manifest.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage10_renderer_smoke"


def read_manifest(path, sample, frame_gap):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["frame_gap"] = int(row["frame_gap"])
            row["middle_frame_count"] = int(row["middle_frame_count"])
            if sample and row["sample"] != sample:
                continue
            if frame_gap is not None and row["frame_gap"] != frame_gap:
                continue
            if row["middle_frame_count"] <= 0:
                continue
            if Path(row["dataset_item"]).exists():
                rows.append(row)
    return rows


def load_rgb(path, height, width):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(image).float() / 255.0


def linear_anchor(left_anchor, right_anchor, t, device):
    left = {key: value.unsqueeze(0).float().to(device) for key, value in left_anchor.items()}
    right = {key: value.unsqueeze(0).float().to(device) for key, value in right_anchor.items()}
    left_attrs = flatten_static_anchor(left)
    right_attrs = flatten_static_anchor(right)
    attrs = left_attrs * (1.0 - t) + right_attrs * t
    return unflatten_static_anchor(attrs)


def psnr_from_mse(mse):
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--sample", default="n3dv")
    parser.add_argument("--frame_gap", type=int, default=4)
    parser.add_argument("--row_index", type=int, default=0)
    parser.add_argument("--middle_index", type=int, default=-1, help="Intermediate index inside the selected pair; -1 selects the center.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--training_renderer", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = read_manifest(args.manifest, args.sample, args.frame_gap)
    if not rows:
        raise RuntimeError(f"No stage6 rows found for sample={args.sample} frame_gap={args.frame_gap}")
    row = rows[args.row_index % len(rows)]
    item = torch.load(row["dataset_item"], map_location="cpu", weights_only=True)
    mids = item["intermediate_frames"]
    mid_idx = len(mids) // 2 if args.middle_index < 0 else args.middle_index % len(mids)
    target_record = mids[mid_idx]
    t = float(target_record["normalized_time"])

    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    height, width = opt.image_height, opt.image_width

    anchor = linear_anchor(item["left_anchor"], item["right_anchor"], t, device)
    gaussians = static_anchor_to_single_frame_gaussians(anchor)
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    with torch.no_grad(), autocast("cuda", enabled=False):
        render_pkg = gaussian_renderer_dynamic.render(
            gaussians,
            background,
            timestamps=None,
            opt=opt,
            anchor_time=None,
            training=args.training_renderer,
        )
    rendered = render_pkg["render"][0, 0].detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0)
    alpha = render_pkg["alpha"][0, 0].detach().cpu()
    target = load_rgb(target_record["rgb_path"], height, width)
    mse = float(torch.mean((rendered - target) ** 2).item())
    summary = {
        "stage": 10,
        "mode": "renderer smoke for linear static intermediate anchor",
        "manifest": str(args.manifest),
        "dataset_item": row["dataset_item"],
        "sample": item["sample"],
        "frame_gap": item["frame_gap"],
        "left_index": item["left_index"],
        "right_index": item["right_index"],
        "target_frame_index": target_record["frame_index"],
        "normalized_time": t,
        "target_rgb_path": target_record["rgb_path"],
        "device": str(device),
        "resolution": [width, height],
        "gaussian_count": int(anchor["rgb"].shape[1]),
        "render_shape": list(render_pkg["render"].shape),
        "depth_shape": list(render_pkg["depth"].shape),
        "alpha_shape": list(render_pkg["alpha"].shape),
        "rgb_mse": mse,
        "rgb_psnr": psnr_from_mse(mse),
        "alpha_mean": float(alpha.mean().item()),
        "alpha_max": float(alpha.max().item()),
        "notes": "Smoke only: linear raw static anchor wrapped as zero-dynamic gaussians. Not trained RGB reconstruction quality.",
    }
    out = args.summary_root / "stage10_renderer_smoke_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(out), **summary}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
