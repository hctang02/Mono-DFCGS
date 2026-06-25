import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import gaussian_renderer_dynamic  # noqa: E402
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.anchor_predictor import GaussianAnchorDynamicPredictor  # noqa: E402
from mono_dfcgs.gaussian_codec import (  # noqa: E402
    flatten_static_anchor,
    uniform_dequantize,
    uniform_quantize,
    unflatten_static_anchor,
)
from mono_dfcgs.render_adapter import static_anchor_to_single_frame_gaussians  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "experiments/stage6_real_anchor_dataset/stage6_real_anchor_dataset_manifest.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage10b_predictor_renderer_rgb_smoke"


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
            if row["middle_frame_count"] > 0 and Path(row["dataset_item"]).exists():
                rows.append(row)
    return rows


def load_rgb(path, height, width, device):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.0
    return image.to(device)


def anchor_to_device(anchor, device):
    return {key: value.unsqueeze(0).float().to(device) for key, value in anchor.items()}


def maybe_quantize_anchor(anchor, bits):
    if bits <= 0:
        return anchor
    attrs = flatten_static_anchor(anchor)
    q, mins, scales = uniform_quantize(attrs, bits=bits)
    return unflatten_static_anchor(uniform_dequantize(q, mins, scales))


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
    parser.add_argument("--middle_index", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    t = torch.tensor([float(target_record["normalized_time"])], dtype=torch.float32, device=device)

    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    height, width = opt.image_height, opt.image_width
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    target = load_rgb(target_record["rgb_path"], height, width, device)

    left = maybe_quantize_anchor(anchor_to_device(item["left_anchor"], device), args.quant_bits)
    right = maybe_quantize_anchor(anchor_to_device(item["right_anchor"], device), args.quant_bits)
    model = GaussianAnchorDynamicPredictor(
        hidden_dim=args.hidden_dim,
        apply_output_constraints=False,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    losses = []
    psnrs = []
    for step in range(1, args.steps + 1):
        pred_anchor = model(left, right, t, apply_output_constraints=False)
        gaussians = static_anchor_to_single_frame_gaussians(pred_anchor)
        with autocast("cuda", enabled=False):
            render_pkg = gaussian_renderer_dynamic.render(
                gaussians,
                background,
                timestamps=None,
                opt=opt,
                anchor_time=None,
                training=False,
            )
            pred_rgb = render_pkg["render"]
            loss = F.mse_loss(pred_rgb, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        mse = float(loss.detach().item())
        losses.append(mse)
        psnrs.append(psnr_from_mse(mse))
        print(f"step {step}/{args.steps} rgb_mse={mse:.6f} psnr={psnrs[-1]:.3f}", flush=True)

    summary = {
        "stage": "10b",
        "mode": "predictor renderer RGB loss smoke",
        "manifest": str(args.manifest),
        "dataset_item": row["dataset_item"],
        "sample": item["sample"],
        "frame_gap": item["frame_gap"],
        "left_index": item["left_index"],
        "right_index": item["right_index"],
        "target_frame_index": target_record["frame_index"],
        "normalized_time": float(t.item()),
        "target_rgb_path": target_record["rgb_path"],
        "device": str(device),
        "resolution": [width, height],
        "gaussian_count": int(left["rgb"].shape[1]),
        "steps": args.steps,
        "lr": args.lr,
        "quant_bits": args.quant_bits,
        "hidden_dim": args.hidden_dim,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_rgb_mse": losses[0],
        "final_rgb_mse": losses[-1],
        "rgb_mse_ratio": losses[-1] / losses[0] if losses[0] > 0 else None,
        "initial_rgb_psnr": psnrs[0],
        "final_rgb_psnr": psnrs[-1],
        "losses": losses,
        "psnrs": psnrs,
        "notes": "Differentiability smoke only: one pair, one intermediate target, no checkpoint saved.",
    }
    out = args.summary_root / "stage10b_predictor_renderer_rgb_smoke_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(out), **summary}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
