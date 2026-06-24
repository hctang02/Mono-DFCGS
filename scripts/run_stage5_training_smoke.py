import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mono_dfcgs.anchor_predictor import GaussianAnchorDynamicPredictor  # noqa: E402
from mono_dfcgs.gaussian_codec import (  # noqa: E402
    flatten_static_anchor,
    uniform_dequantize,
    uniform_quantize,
)


DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage5_training_smoke"


def make_anchor(batch, gaussians, device):
    return {
        "rgb": torch.rand(batch, gaussians, 3, device=device),
        "opacity": torch.rand(batch, gaussians, 1, device=device) * 0.8 + 0.1,
        "scale": torch.rand(batch, gaussians, 2, device=device) * 0.05 + 0.005,
        "xyz": torch.rand(batch, gaussians, 3, device=device) * 2.0 - 1.0,
        "rot": F.normalize(torch.randn(batch, gaussians, 4, device=device), dim=-1),
    }


def attrs_to_anchor(attrs):
    return {
        "rgb": attrs[..., 0:3],
        "opacity": attrs[..., 3:4],
        "scale": attrs[..., 4:6],
        "xyz": attrs[..., 6:9],
        "rot": attrs[..., 9:13],
    }


def maybe_quantize_anchor(anchor, bits):
    if bits <= 0:
        return anchor
    attrs = flatten_static_anchor(anchor)
    q, mins, scales = uniform_quantize(attrs, bits=bits)
    return attrs_to_anchor(uniform_dequantize(q, mins, scales))


def linear_target(left, right, t):
    left_attrs = flatten_static_anchor(left)
    right_attrs = flatten_static_anchor(right)
    t_scalar = t.reshape(left_attrs.shape[0], 1, 1).to(left_attrs.device, left_attrs.dtype)
    return left_attrs * (1.0 - t_scalar) + right_attrs * t_scalar


def make_batch(args, device):
    left_clean = make_anchor(args.batch, args.gaussians, device)
    right_clean = make_anchor(args.batch, args.gaussians, device)
    t = torch.rand(args.batch, device=device)
    target = linear_target(left_clean, right_clean, t)
    left = maybe_quantize_anchor(left_clean, args.quant_bits)
    right = maybe_quantize_anchor(right_clean, args.quant_bits)
    return left, right, t, target


def evaluate(model, args, device, batches=8):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            left, right, t, target = make_batch(args, device)
            pred = model(left, right, t)
            losses.append(float(F.mse_loss(flatten_static_anchor(pred), target).item()))
    return sum(losses) / len(losses)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--gaussians", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260625)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.summary_root.mkdir(parents=True, exist_ok=True)

    model = GaussianAnchorDynamicPredictor(hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    initial_eval = evaluate(model, args, device)
    train_losses = []
    model.train()
    for step in range(1, args.steps + 1):
        left, right, t, target = make_batch(args, device)
        pred = model(left, right, t)
        loss = F.mse_loss(flatten_static_anchor(pred), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_losses.append(float(loss.item()))
        if step % max(args.steps // 4, 1) == 0:
            print(f"step {step}/{args.steps} loss={loss.item():.6f}", flush=True)
    final_eval = evaluate(model, args, device)
    summary = {
        "stage": 5,
        "model": "GaussianAnchorDynamicPredictor training smoke",
        "device": str(device),
        "batch": args.batch,
        "gaussians": args.gaussians,
        "hidden_dim": args.hidden_dim,
        "steps": args.steps,
        "lr": args.lr,
        "quant_bits": args.quant_bits,
        "initial_eval_loss": initial_eval,
        "final_eval_loss": final_eval,
        "eval_loss_ratio": final_eval / initial_eval if initial_eval > 0 else None,
        "first_train_loss": train_losses[0],
        "last_train_loss": train_losses[-1],
        "train_loss_ratio": train_losses[-1] / train_losses[0] if train_losses[0] > 0 else None,
        "train_losses": train_losses,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    out = args.summary_root / "stage5_training_smoke_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(out), **summary}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
