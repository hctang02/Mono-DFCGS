import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.amp import autocast


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments/stage6_real_anchor_dataset/stage6_real_anchor_dataset_manifest.csv"
DEFAULT_HEAVY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage21_gaussian_anchor_only_adapter_smoke")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage21_gaussian_anchor_only_adapter_smoke"


sys.path.insert(0, str(REPO_ROOT))
import gaussian_renderer_dynamic  # noqa: E402
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.anchor_predictor import GaussianAnchorDynamicPredictor  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, uniform_dequantize, uniform_quantize, unflatten_static_anchor  # noqa: E402
from mono_dfcgs.render_adapter import static_anchor_to_single_frame_gaussians  # noqa: E402


def read_manifest(path, frame_gap, samples):
    sample_set = set(samples) if samples else None
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["frame_gap"] = int(row["frame_gap"])
            row["middle_frame_count"] = int(row["middle_frame_count"])
            if frame_gap is not None and row["frame_gap"] != frame_gap:
                continue
            if sample_set is not None and row["sample"] not in sample_set:
                continue
            if row["middle_frame_count"] <= 0:
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
    return [row for i, row in enumerate(rows) if i % 4 != 0], [row for i, row in enumerate(rows) if i % 4 == 0]


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
    if mse <= 1e-12:
        return 100.0
    return -10.0 * math.log10(mse)


def make_tasks(rows, max_rows, targets_per_row, height, width, device, quant_bits):
    tasks = []
    for row in rows[:max_rows]:
        item = torch.load(row["dataset_item"], map_location="cpu", weights_only=True)
        mids = item["intermediate_frames"]
        if targets_per_row >= len(mids):
            selected_mids = mids
        else:
            if targets_per_row == 1:
                selected_mids = [mids[len(mids) // 2]]
            else:
                idxs = np.linspace(0, len(mids) - 1, targets_per_row).round().astype(int).tolist()
                selected_mids = [mids[idx] for idx in idxs]
        left = maybe_quantize_anchor(anchor_to_device(item["left_anchor"], device), quant_bits)
        right = maybe_quantize_anchor(anchor_to_device(item["right_anchor"], device), quant_bits)
        for mid in selected_mids:
            tasks.append({
                "sample": item["sample"],
                "frame_gap": item["frame_gap"],
                "left_index": item["left_index"],
                "right_index": item["right_index"],
                "target_frame_index": mid["frame_index"],
                "normalized_time": float(mid["normalized_time"]),
                "target_rgb_path": mid["rgb_path"],
                "left": left,
                "right": right,
                "target": load_rgb(mid["rgb_path"], height, width, device),
            })
    return tasks


def render_anchor(anchor, background, opt):
    gaussians = static_anchor_to_single_frame_gaussians(anchor)
    with autocast("cuda", enabled=False):
        render_pkg = gaussian_renderer_dynamic.render(
            gaussians,
            background,
            timestamps=None,
            opt=opt,
            anchor_time=None,
            training=False,
        )
    return render_pkg["render"]


def render_prediction(model, task, background, opt):
    t = torch.tensor([task["normalized_time"]], dtype=torch.float32, device=background.device)
    pred_anchor = model(task["left"], task["right"], t, apply_output_constraints=False)
    return render_anchor(pred_anchor, background, opt)


def linear_anchor(left, right, t):
    left_attrs = flatten_static_anchor(left)
    right_attrs = flatten_static_anchor(right)
    t_scalar = torch.tensor([t], dtype=left_attrs.dtype, device=left_attrs.device).reshape(1, 1, 1)
    return unflatten_static_anchor(left_attrs * (1.0 - t_scalar) + right_attrs * t_scalar)


def evaluate(model, tasks, background, opt):
    model.eval()
    model_rows = []
    with torch.no_grad():
        for task in tasks:
            pred_rgb = render_prediction(model, task, background, opt).clamp(0.0, 1.0)
            model_mse = float(F.mse_loss(pred_rgb, task["target"]).item())
            baseline_rgb = render_anchor(linear_anchor(task["left"], task["right"], task["normalized_time"]), background, opt).clamp(0.0, 1.0)
            baseline_mse = float(F.mse_loss(baseline_rgb, task["target"]).item())
            model_rows.append({
                "sample": task["sample"],
                "frame_gap": task["frame_gap"],
                "target_frame_index": task["target_frame_index"],
                "normalized_time": task["normalized_time"],
                "model_mse": model_mse,
                "model_psnr": psnr_from_mse(model_mse),
                "linear_mse": baseline_mse,
                "linear_psnr": psnr_from_mse(baseline_mse),
            })
    return model_rows


def average(rows, key):
    return sum(row[key] for row in rows) / max(len(rows), 1)


def write_train_csv(rows, path):
    fields = ["step", "sample", "frame_gap", "target_frame_index", "normalized_time", "rgb_mse", "rgb_psnr"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_model(model, path):
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    save_file(state, str(path))
    return {"path": str(path), "tensor_count": len(state), "parameter_count": sum(t.numel() for t in state.values())}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples", nargs="*", default=["n3dv", "robot"])
    parser.add_argument("--eval_samples", nargs="*", default=["robot"])
    parser.add_argument("--frame_gap", type=int, default=4)
    parser.add_argument("--max_train_rows", type=int, default=2)
    parser.add_argument("--max_eval_rows", type=int, default=2)
    parser.add_argument("--targets_per_row", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    rows = read_manifest(args.manifest, args.frame_gap, args.samples)
    if not rows:
        raise RuntimeError(f"No Stage6 rows found for samples={args.samples} frame_gap={args.frame_gap}")
    train_rows, eval_rows = split_rows(rows, args.eval_samples)
    if not train_rows or not eval_rows:
        raise RuntimeError(f"Need non-empty train/eval rows, got train={len(train_rows)} eval={len(eval_rows)}")

    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    height, width = opt.image_height, opt.image_width
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)

    train_tasks = make_tasks(train_rows, args.max_train_rows, args.targets_per_row, height, width, device, args.quant_bits)
    eval_tasks = make_tasks(eval_rows, args.max_eval_rows, args.targets_per_row, height, width, device, args.quant_bits)
    model = GaussianAnchorDynamicPredictor(hidden_dim=args.hidden_dim, apply_output_constraints=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    initial_train_eval = evaluate(model, train_tasks, background, opt)
    initial_eval = evaluate(model, eval_tasks, background, opt)

    train_log = []
    model.train()
    for step in range(1, args.steps + 1):
        task = train_tasks[(step - 1) % len(train_tasks)]
        pred_rgb = render_prediction(model, task, background, opt).clamp(0.0, 1.0)
        loss = F.mse_loss(pred_rgb, task["target"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        mse = float(loss.detach().item())
        row = {
            "step": step,
            "sample": task["sample"],
            "frame_gap": task["frame_gap"],
            "target_frame_index": task["target_frame_index"],
            "normalized_time": task["normalized_time"],
            "rgb_mse": mse,
            "rgb_psnr": psnr_from_mse(mse),
        }
        train_log.append(row)
        print(json.dumps(row), flush=True)

    final_train_eval = evaluate(model, train_tasks, background, opt)
    final_eval = evaluate(model, eval_tasks, background, opt)
    train_csv = args.summary_root / "stage21_train_rgb_losses.csv"
    write_train_csv(train_log, train_csv)
    checkpoint_info = save_model(model, args.heavy_root / "stage21_anchor_adapter.safetensors")

    summary = {
        "stage": 21,
        "mode": "Gaussian-anchor-only decoder adapter RGB smoke",
        "manifest": str(args.manifest),
        "samples": args.samples,
        "eval_samples": args.eval_samples,
        "frame_gap": args.frame_gap,
        "max_train_rows": args.max_train_rows,
        "max_eval_rows": args.max_eval_rows,
        "targets_per_row": args.targets_per_row,
        "train_task_count": len(train_tasks),
        "eval_task_count": len(eval_tasks),
        "steps": args.steps,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "quant_bits": args.quant_bits,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_train_model_psnr_avg": average(initial_train_eval, "model_psnr"),
        "final_train_model_psnr_avg": average(final_train_eval, "model_psnr"),
        "initial_eval_model_psnr_avg": average(initial_eval, "model_psnr"),
        "final_eval_model_psnr_avg": average(final_eval, "model_psnr"),
        "initial_eval_linear_psnr_avg": average(initial_eval, "linear_psnr"),
        "final_eval_linear_psnr_avg": average(final_eval, "linear_psnr"),
        "initial_train_eval": initial_train_eval,
        "final_train_eval": final_train_eval,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "train_log": train_log,
        "train_losses_csv": str(train_csv),
        "external_adapter_checkpoint": checkpoint_info,
        "notes": "Smoke only. Input payload is q8 static keyframe anchors plus timestamp; no non-keyframe RGB/Gaussian/motion/residual is transmitted.",
    }
    summary_path = args.summary_root / "stage21_gaussian_anchor_only_adapter_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "train_losses_csv": str(train_csv),
        "external_adapter_checkpoint": checkpoint_info["path"],
        "initial_eval_model_psnr_avg": summary["initial_eval_model_psnr_avg"],
        "final_eval_model_psnr_avg": summary["final_eval_model_psnr_avg"],
        "initial_train_model_psnr_avg": summary["initial_train_model_psnr_avg"],
        "final_train_model_psnr_avg": summary["final_train_model_psnr_avg"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
