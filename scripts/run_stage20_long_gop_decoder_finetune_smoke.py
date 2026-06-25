import argparse
import csv
import json
import math
import random
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/streamsplat.safetensors")
DEFAULT_STAGE1_CACHE = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage1_streamsplat_fair_metrics/cache")
DEFAULT_HEAVY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage20_long_gop_decoder_finetune_smoke")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage20_long_gop_decoder_finetune_smoke"
TRAIN_PREFIXES = ("model.decoder", "model.gs_dynamic_predictor", "model.encoder_proj")
FREEZE_PREFIXES = ("model.gs_predictor", "model.condition_encoder", "model.gaussian_upsampler")


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from model.splat_model_inference import SplatModel  # noqa: E402


def load_model(checkpoint, device):
    opt = Options()
    opt.resume = str(checkpoint)
    opt.compile = False
    opt.input_frames = 1
    opt.epoch = 0
    model = SplatModel(opt).to(device)
    safetensors_device = "cuda:0" if device.type == "cuda" else "cpu"
    state = load_file(str(checkpoint), device=safetensors_device)
    new_state = OrderedDict()
    for key, value in state.items():
        if "_orig_mod." in key and not opt.compile:
            key = key.replace("_orig_mod.", "")
        new_state[key] = value
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    return model, opt, missing, unexpected


def set_freeze_policy(model):
    train_params = 0
    freeze_params = 0
    train_tensors = 0
    freeze_tensors = 0
    for name, param in model.named_parameters():
        trainable = any(name == prefix or name.startswith(prefix + ".") for prefix in TRAIN_PREFIXES)
        param.requires_grad = trainable
        if trainable:
            train_params += param.numel()
            train_tensors += 1
        else:
            freeze_params += param.numel()
            freeze_tensors += 1
    return {
        "train_prefixes": list(TRAIN_PREFIXES),
        "freeze_prefixes": list(FREEZE_PREFIXES),
        "train_parameter_tensors": train_tensors,
        "freeze_parameter_tensors": freeze_tensors,
        "train_parameters": train_params,
        "freeze_parameters": freeze_params,
        "total_parameters": train_params + freeze_params,
        "train_parameter_percent": train_params / max(train_params + freeze_params, 1) * 100.0,
    }


def get_image(path, h, w):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def get_depth(path, h, w):
    depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(path)
    return cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def load_cached_sample(sample, opt, cache_root):
    frame_dir = cache_root / sample / "frames"
    depth_dir = cache_root / sample / "depths_vitl_u16"
    frame_files = sorted(frame_dir.glob("*.png"))
    depth_files = sorted(depth_dir.glob("*.png"))
    if not frame_files or len(frame_files) != len(depth_files):
        raise RuntimeError(f"Missing cached frames/depths for {sample}: frames={len(frame_files)} depths={len(depth_files)}")
    frames = [get_image(path, opt.image_height, opt.image_width) for path in frame_files]
    depths = [get_depth(path, opt.image_height, opt.image_width) for path in depth_files]
    return {"frames": frames, "depths": depths, "total_frames": len(frames)}


def build_pairs(total_frames, gap):
    selected = list(range(0, total_frames, gap))
    if selected[-1] != total_frames - 1:
        selected.append(total_frames - 1)
    return [(selected[i], selected[i + 1]) for i in range(len(selected) - 1)]


def make_pair_index(samples, gaps):
    pairs = []
    for sample, data in samples.items():
        for gap in gaps:
            for a, b in build_pairs(data["total_frames"], gap):
                pairs.append({"sample": sample, "gap": gap, "start": a, "end": b})
    return pairs


def select_eval_pairs(samples, gaps, max_pairs):
    selected = []
    for sample, data in samples.items():
        for gap in gaps:
            pairs = build_pairs(data["total_frames"], gap)
            if not pairs:
                continue
            a, b = pairs[min(len(pairs) - 1, len(pairs) // 2)]
            selected.append({"sample": sample, "gap": gap, "start": a, "end": b})
    return selected[:max_pairs]


def normalize_depths(depths):
    max_depth = depths.flatten(1).max(dim=1)[0][:, None, None, None, None]
    min_depth = depths.flatten(1).min(dim=1)[0][:, None, None, None, None]
    return (depths - min_depth) / (max_depth - min_depth + 1e-8)


def prepare_pair(data, pair, device):
    a, b = pair["start"], pair["end"]
    indices = list(range(a, b + 1))
    endpoint_frames = np.stack([data["frames"][a], data["frames"][b]], axis=0)
    endpoint_depths = np.stack([data["depths"][a], data["depths"][b]], axis=0)
    target_frames = np.stack([data["frames"][idx] for idx in indices], axis=0)

    frames = torch.from_numpy(endpoint_frames).float().to(device) / 255.0
    frames = frames.permute(0, 3, 1, 2).unsqueeze(0)
    depths = torch.from_numpy(endpoint_depths).float().to(device).unsqueeze(0).unsqueeze(2)
    depths = normalize_depths(depths)
    target = torch.from_numpy(target_frames).float().to(device) / 255.0
    target = target.permute(0, 3, 1, 2).unsqueeze(0)
    timestamps = torch.linspace(0.0, 1.0, len(indices), device=device).unsqueeze(0)
    return {"frames": frames, "depths": depths, "timestamps": timestamps}, target


def set_output_frames(model, output_frames):
    model.opt.output_frames = output_frames
    model.model.opt.output_frames = output_frames


def mse_to_psnr(mse):
    if mse <= 1e-12:
        return 100.0
    return -10.0 * math.log10(mse)


def evaluate(model, samples, pairs, device, max_pairs):
    model.eval()
    rows = []
    with torch.no_grad():
        for pair in pairs[:max_pairs]:
            data, target = prepare_pair(samples[pair["sample"]], pair, device)
            set_output_frames(model, target.shape[1])
            output = model(data)
            pred = output["pred_frames"].clamp(0.0, 1.0)
            all_mse = float(F.mse_loss(pred, target).item())
            if pred.shape[1] > 2:
                middle_mse = float(F.mse_loss(pred[:, 1:-1], target[:, 1:-1]).item())
            else:
                middle_mse = None
            given_pred = torch.stack([pred[:, 0], pred[:, -1]], dim=1)
            given_target = torch.stack([target[:, 0], target[:, -1]], dim=1)
            given_mse = float(F.mse_loss(given_pred, given_target).item())
            rows.append({
                **pair,
                "segment_length": pair["end"] - pair["start"],
                "all_mse": all_mse,
                "all_psnr": mse_to_psnr(all_mse),
                "middle_mse": middle_mse,
                "middle_psnr": None if middle_mse is None else mse_to_psnr(middle_mse),
                "given_mse": given_mse,
                "given_psnr": mse_to_psnr(given_mse),
            })
    return rows


def average_metric(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return sum(values) / max(len(values), 1)


def write_train_csv(rows, path):
    fields = ["step", "sample", "gap", "start", "end", "segment_length", "loss", "psnr"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_trainable_state(model, path):
    state = {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}
    save_file(state, str(path))
    return {"path": str(path), "tensor_count": len(state), "parameter_count": sum(t.numel() for t in state.values())}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="*", default=["n3dv", "robot"])
    parser.add_argument("--train_gaps", nargs="*", type=int, default=[2, 4, 8, 12, 16])
    parser.add_argument("--eval_gaps", nargs="*", type=int, default=[4, 8, 16])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_eval_pairs", type=int, default=6)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stage1_cache", type=Path, default=DEFAULT_STAGE1_CACHE)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Stage20 needs CUDA because SplatModel constructs CUDA background tensors.")

    model, opt, missing, unexpected = load_model(args.checkpoint, device)
    freeze_summary = set_freeze_policy(model)
    samples = {sample: load_cached_sample(sample, opt, args.stage1_cache) for sample in args.samples}
    train_pairs = make_pair_index(samples, args.train_gaps)
    eval_pairs = select_eval_pairs(samples, args.eval_gaps, args.max_eval_pairs)
    random.shuffle(train_pairs)

    initial_eval = evaluate(model, samples, eval_pairs, device, len(eval_pairs))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)

    train_rows = []
    model.train()
    for step in range(args.steps):
        pair = train_pairs[step % len(train_pairs)]
        data, target = prepare_pair(samples[pair["sample"]], pair, device)
        set_output_frames(model, target.shape[1])
        output = model(data)
        pred = output["pred_frames"].clamp(0.0, 1.0)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        mse = float(loss.detach().item())
        row = {
            "step": step + 1,
            **pair,
            "segment_length": pair["end"] - pair["start"],
            "loss": mse,
            "psnr": mse_to_psnr(mse),
        }
        train_rows.append(row)
        print(json.dumps(row), flush=True)

    final_eval = evaluate(model, samples, eval_pairs, device, len(eval_pairs))
    train_csv = args.summary_root / "stage20_train_losses.csv"
    write_train_csv(train_rows, train_csv)
    trainable_state_path = args.heavy_root / "stage20_trainable_state.safetensors"
    checkpoint_info = save_trainable_state(model, trainable_state_path)

    summary = {
        "stage": 20,
        "mode": "long-GOP Dynamic Decoder fine-tune smoke",
        "checkpoint": str(args.checkpoint),
        "samples": args.samples,
        "train_gaps": args.train_gaps,
        "eval_gaps": args.eval_gaps,
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "freeze_summary": freeze_summary,
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "missing_keys_sample": list(missing)[:20],
        "unexpected_keys_sample": list(unexpected)[:20],
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "initial_eval_all_psnr_avg": average_metric(initial_eval, "all_psnr"),
        "final_eval_all_psnr_avg": average_metric(final_eval, "all_psnr"),
        "initial_eval_middle_psnr_avg": average_metric(initial_eval, "middle_psnr"),
        "final_eval_middle_psnr_avg": average_metric(final_eval, "middle_psnr"),
        "initial_eval_given_psnr_avg": average_metric(initial_eval, "given_psnr"),
        "final_eval_given_psnr_avg": average_metric(final_eval, "given_psnr"),
        "train_losses_csv": str(train_csv),
        "train_rows": train_rows,
        "external_trainable_checkpoint": checkpoint_info,
        "notes": "Smoke only. Trains original StreamSplat temporal/dynamic decoder path from pretrained checkpoint; static encoder/predictor and condition encoder are frozen.",
    }
    summary_path = args.summary_root / "stage20_long_gop_decoder_finetune_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "train_losses_csv": str(train_csv),
        "external_trainable_checkpoint": str(trainable_state_path),
        "initial_eval_all_psnr_avg": summary["initial_eval_all_psnr_avg"],
        "final_eval_all_psnr_avg": summary["final_eval_all_psnr_avg"],
        "initial_eval_middle_psnr_avg": summary["initial_eval_middle_psnr_avg"],
        "final_eval_middle_psnr_avg": summary["final_eval_middle_psnr_avg"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
