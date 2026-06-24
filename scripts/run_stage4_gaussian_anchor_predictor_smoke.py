import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mono_dfcgs.anchor_predictor import GaussianAnchorDynamicPredictor  # noqa: E402
from mono_dfcgs.gaussian_codec import estimate_static_anchor_payload, flatten_static_anchor  # noqa: E402


DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage4_gaussian_anchor_predictor_smoke"


def make_anchor(batch, gaussians, device):
    return {
        "rgb": torch.rand(batch, gaussians, 3, device=device),
        "opacity": torch.rand(batch, gaussians, 1, device=device) * 0.8 + 0.1,
        "scale": torch.rand(batch, gaussians, 2, device=device) * 0.05 + 0.005,
        "xyz": torch.rand(batch, gaussians, 3, device=device) * 2.0 - 1.0,
        "rot": F.normalize(torch.randn(batch, gaussians, 4, device=device), dim=-1),
    }


def linear_target(left, right, t):
    left_attrs = flatten_static_anchor(left)
    right_attrs = flatten_static_anchor(right)
    t_scalar = t.reshape(left_attrs.shape[0], 1, 1).to(left_attrs.device, left_attrs.dtype)
    return left_attrs * (1.0 - t_scalar) + right_attrs * t_scalar


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--gaussians", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260625)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    left = make_anchor(args.batch, args.gaussians, device)
    right = make_anchor(args.batch, args.gaussians, device)
    model = GaussianAnchorDynamicPredictor(hidden_dim=args.hidden_dim).to(device)
    model.eval()
    times = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
    losses = []
    output_shapes = {}
    with torch.no_grad():
        for t_value in times:
            t = t_value.repeat(args.batch)
            pred = model(left, right, t)
            pred_attrs = flatten_static_anchor(pred)
            target_attrs = linear_target(left, right, t)
            losses.append(float(F.mse_loss(pred_attrs, target_attrs).item()))
            output_shapes[str(float(t_value.item()))] = {key: list(value.shape) for key, value in pred.items()}
    params = sum(p.numel() for p in model.parameters())
    payload = estimate_static_anchor_payload(left, codec="q8", opacity_threshold=0.0)
    summary = {
        "stage": 4,
        "model": "GaussianAnchorDynamicPredictor smoke",
        "batch": args.batch,
        "gaussians": args.gaussians,
        "hidden_dim": args.hidden_dim,
        "device": str(device),
        "parameter_count": params,
        "times": [float(x.item()) for x in times],
        "mse_to_linear_target": losses,
        "mean_mse_to_linear_target": sum(losses) / len(losses),
        "output_shapes": output_shapes,
        "q8_payload_mib_for_left_anchor": payload.mib,
    }
    out = args.summary_root / "stage4_gaussian_anchor_predictor_smoke_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(out), **summary}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
