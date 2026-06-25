import argparse
import csv
import json
import random
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
    unflatten_static_anchor,
)


DEFAULT_MANIFEST = REPO_ROOT / "experiments/stage6_real_anchor_dataset/stage6_real_anchor_dataset_manifest.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage9_real_anchor_proxy_training"


def read_manifest(path, frame_gap=None, samples=None):
    sample_set = set(samples) if samples else None
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["frame_gap"] = int(row["frame_gap"])
            row["left_index"] = int(row["left_index"])
            row["right_index"] = int(row["right_index"])
            row["segment_length"] = int(row["segment_length"])
            row["middle_frame_count"] = int(row["middle_frame_count"])
            row["gaussians_per_anchor"] = int(row["gaussians_per_anchor"])
            if frame_gap is not None and row["frame_gap"] != frame_gap:
                continue
            if sample_set is not None and row["sample"] not in sample_set:
                continue
            if Path(row["dataset_item"]).exists():
                rows.append(row)
    return rows


def split_rows(rows, eval_samples):
    eval_set = set(eval_samples or [])
    if eval_set:
        train_rows = [row for row in rows if row["sample"] not in eval_set]
        eval_rows = [row for row in rows if row["sample"] in eval_set]
        if train_rows and eval_rows:
            return train_rows, eval_rows
    train_rows = [row for i, row in enumerate(rows) if i % 5 != 0]
    eval_rows = [row for i, row in enumerate(rows) if i % 5 == 0]
    return train_rows, eval_rows


def attrs_to_anchor(attrs):
    return unflatten_static_anchor(attrs)


def maybe_quantize_anchor(anchor, bits):
    if bits <= 0:
        return anchor
    attrs = flatten_static_anchor(anchor)
    q, mins, scales = uniform_quantize(attrs, bits=bits)
    return attrs_to_anchor(uniform_dequantize(q, mins, scales))


def sample_gaussians(anchor, gaussians, rng):
    n = anchor["rgb"].shape[0]
    if gaussians <= 0 or gaussians >= n:
        idx = torch.arange(n)
    else:
        idx = torch.randperm(n, generator=rng)[:gaussians]
    return {key: value[idx].float() for key, value in anchor.items()}


def move_batch(anchors, device):
    out = {}
    for key in anchors[0]:
        out[key] = torch.stack([anchor[key] for anchor in anchors], dim=0).to(device, non_blocking=True)
    return out


def choose_time(item, rng):
    mids = item.get("intermediate_frames", [])
    if mids:
        selected = mids[int(torch.randint(len(mids), (1,), generator=rng).item())]
        return float(selected["normalized_time"])
    return float(torch.rand((), generator=rng).item())


def linear_target(left, right, t):
    left_attrs = flatten_static_anchor(left)
    right_attrs = flatten_static_anchor(right)
    t_scalar = t.reshape(left_attrs.shape[0], 1, 1).to(left_attrs.device, left_attrs.dtype)
    return left_attrs * (1.0 - t_scalar) + right_attrs * t_scalar


def load_batch(rows, batch_items, gaussians, quant_bits, device, rng):
    left_clean, right_clean, left_input, right_input, times = [], [], [], [], []
    selected_rows = random.choices(rows, k=batch_items)
    for row in selected_rows:
        item = torch.load(row["dataset_item"], map_location="cpu", weights_only=True)
        left = sample_gaussians(item["left_anchor"], gaussians, rng)
        right = sample_gaussians(item["right_anchor"], gaussians, rng)
        left_b = {key: value.unsqueeze(0) for key, value in left.items()}
        right_b = {key: value.unsqueeze(0) for key, value in right.items()}
        left_clean.append(left)
        right_clean.append(right)
        left_input.append({key: value.squeeze(0) for key, value in maybe_quantize_anchor(left_b, quant_bits).items()})
        right_input.append({key: value.squeeze(0) for key, value in maybe_quantize_anchor(right_b, quant_bits).items()})
        times.append(choose_time(item, rng))
    left_clean = move_batch(left_clean, device)
    right_clean = move_batch(right_clean, device)
    left_input = move_batch(left_input, device)
    right_input = move_batch(right_input, device)
    t = torch.tensor(times, dtype=torch.float32, device=device)
    target = linear_target(left_clean, right_clean, t)
    baseline = linear_target(left_input, right_input, t)
    return left_input, right_input, t, target, baseline


def evaluate(model, rows, args, device, rng, batches):
    model.eval()
    model_losses = []
    baseline_losses = []
    with torch.no_grad():
        for _ in range(batches):
            left, right, t, target, baseline = load_batch(
                rows, args.batch_items, args.gaussians, args.quant_bits, device, rng
            )
            pred = model(left, right, t, apply_output_constraints=args.apply_output_constraints)
            pred_attrs = flatten_static_anchor(pred)
            model_losses.append(float(F.mse_loss(pred_attrs, target).item()))
            baseline_losses.append(float(F.mse_loss(baseline, target).item()))
    return {
        "model_loss": sum(model_losses) / len(model_losses),
        "baseline_loss": sum(baseline_losses) / len(baseline_losses),
    }


def write_loss_csv(losses, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss"])
        writer.writeheader()
        writer.writerows({"step": i + 1, "train_loss": loss} for i, loss in enumerate(losses))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--frame_gap", type=int, default=4)
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--eval_samples", nargs="*", default=["robot"])
    parser.add_argument("--batch_items", type=int, default=2)
    parser.add_argument("--gaussians", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--eval_batches", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--apply_output_constraints", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260625)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)
    device = torch.device(args.device)

    rows = read_manifest(args.manifest, frame_gap=args.frame_gap, samples=args.samples)
    if not rows:
        raise RuntimeError(f"No stage6 manifest rows found for frame_gap={args.frame_gap}: {args.manifest}")
    train_rows, eval_rows = split_rows(rows, args.eval_samples)
    if not train_rows or not eval_rows:
        raise RuntimeError(f"Need non-empty train/eval split, got train={len(train_rows)} eval={len(eval_rows)}")

    model = GaussianAnchorDynamicPredictor(
        hidden_dim=args.hidden_dim,
        apply_output_constraints=args.apply_output_constraints,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    initial_eval = evaluate(model, eval_rows, args, device, rng, args.eval_batches)
    train_losses = []
    model.train()
    for step in range(1, args.steps + 1):
        left, right, t, target, _baseline = load_batch(
            train_rows, args.batch_items, args.gaussians, args.quant_bits, device, rng
        )
        pred = model(left, right, t, apply_output_constraints=args.apply_output_constraints)
        loss = F.mse_loss(flatten_static_anchor(pred), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_losses.append(float(loss.item()))
        if step % max(args.steps // 4, 1) == 0:
            print(f"step {step}/{args.steps} loss={loss.item():.6f}", flush=True)

    final_eval = evaluate(model, eval_rows, args, device, rng, args.eval_batches)
    loss_csv = args.summary_root / "stage9_train_losses.csv"
    write_loss_csv(train_losses, loss_csv)
    summary = {
        "stage": 9,
        "model": "GaussianAnchorDynamicPredictor real-anchor proxy training",
        "device": str(device),
        "manifest": str(args.manifest),
        "frame_gap": args.frame_gap,
        "samples": args.samples,
        "eval_samples": args.eval_samples,
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "batch_items": args.batch_items,
        "gaussians_per_item": args.gaussians,
        "hidden_dim": args.hidden_dim,
        "steps": args.steps,
        "eval_batches": args.eval_batches,
        "lr": args.lr,
        "quant_bits": args.quant_bits,
        "apply_output_constraints": args.apply_output_constraints,
        "initial_eval_loss": initial_eval["model_loss"],
        "final_eval_loss": final_eval["model_loss"],
        "eval_loss_ratio": final_eval["model_loss"] / initial_eval["model_loss"] if initial_eval["model_loss"] > 0 else None,
        "initial_baseline_loss": initial_eval["baseline_loss"],
        "final_baseline_loss": final_eval["baseline_loss"],
        "first_train_loss": train_losses[0],
        "last_train_loss": train_losses[-1],
        "train_loss_ratio": train_losses[-1] / train_losses[0] if train_losses[0] > 0 else None,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "loss_csv": str(loss_csv),
        "notes": "Proxy loss only: clean linear interpolation of real keyframe anchors in raw attribute space. No renderer/RGB supervision yet.",
    }
    summary_path = args.summary_root / "stage9_real_anchor_proxy_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
