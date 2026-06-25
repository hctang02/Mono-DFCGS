import argparse
import csv
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/streamsplat.safetensors")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage18_decoder_freeze_policy"

sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from model.splat_model_inference import SplatModel  # noqa: E402


POLICY_RULES = [
    ("model.decoder", "train", "Temporal/dynamic token decoder for endpoint-conditioned long-GOP adaptation."),
    ("model.gs_dynamic_predictor", "train", "Predicts dynamic xyz / opacity terms; primary target for sparse-keyframe motion adaptation."),
    ("model.encoder_proj", "train", "Small projection before dynamic Gaussian predictor; keep trainable with dynamic path."),
    ("model.gs_predictor", "freeze", "Static encoder and static Gaussian predictor inherited from StreamSplat."),
    ("model.condition_encoder", "freeze", "DINO condition encoder; expensive and not codec-specific for first fine-tune smoke."),
    ("model.gaussian_upsampler", "freeze", "Shared upsampler; freeze in first smoke to avoid destabilizing static Gaussian features."),
]


MODULE_PREFIXES = [
    "model.gs_predictor.encoder",
    "model.gs_predictor.gaussian_upsampler",
    "model.gs_predictor.predictor",
    "model.condition_encoder",
    "model.decoder",
    "model.gaussian_upsampler",
    "model.gs_dynamic_predictor",
    "model.encoder_proj",
]


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
    model.eval()
    return model, missing, unexpected


def count_params(params):
    total = 0
    trainable = 0
    for param in params:
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def classify_param(name):
    for prefix, action, reason in POLICY_RULES:
        if name.startswith(prefix + ".") or name == prefix:
            return action, prefix, reason
    return "freeze", "default", "Freeze by default for first long-GOP decoder adaptation smoke."


def summarize_modules(model):
    rows = []
    named_params = list(model.named_parameters())
    total_params, default_trainable = count_params(param for _name, param in named_params)
    rows.append({
        "module": "TOTAL",
        "parameters": total_params,
        "default_trainable_parameters": default_trainable,
        "parameter_percent": 100.0,
    })
    for prefix in MODULE_PREFIXES:
        params = [param for name, param in named_params if name.startswith(prefix + ".")]
        total, trainable = count_params(params)
        rows.append({
            "module": prefix,
            "parameters": total,
            "default_trainable_parameters": trainable,
            "parameter_percent": (total / total_params * 100.0) if total_params else 0.0,
        })
    other = [param for name, param in named_params if not any(name.startswith(prefix + ".") for prefix in MODULE_PREFIXES)]
    total, trainable = count_params(other)
    rows.append({
        "module": "OTHER",
        "parameters": total,
        "default_trainable_parameters": trainable,
        "parameter_percent": (total / total_params * 100.0) if total_params else 0.0,
    })
    return rows


def summarize_policy(model):
    policy_rows = []
    action_counts = defaultdict(int)
    action_params = defaultdict(int)
    for name, param in model.named_parameters():
        action, matched_rule, reason = classify_param(name)
        n = param.numel()
        action_counts[action] += 1
        action_params[action] += n
        policy_rows.append({
            "parameter": name,
            "shape": list(param.shape),
            "numel": n,
            "policy": action,
            "matched_rule": matched_rule,
            "reason": reason,
        })
    summary = {
        "train_parameter_tensors": action_counts["train"],
        "freeze_parameter_tensors": action_counts["freeze"],
        "train_parameters": action_params["train"],
        "freeze_parameters": action_params["freeze"],
        "total_parameters": sum(action_params.values()),
    }
    summary["train_parameter_percent"] = (
        summary["train_parameters"] / summary["total_parameters"] * 100.0 if summary["total_parameters"] else 0.0
    )
    return policy_rows, summary


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("SplatModel inference path constructs CUDA background tensors; run Stage18 on an available CUDA GPU.")
    model, missing, unexpected = load_model(args.checkpoint, device)
    module_rows = summarize_modules(model)
    policy_rows, policy_summary = summarize_policy(model)

    module_csv = args.summary_root / "stage18_module_parameter_summary.csv"
    policy_csv = args.summary_root / "stage18_freeze_policy_parameters.csv"
    summary_json = args.summary_root / "stage18_decoder_freeze_policy_summary.json"
    write_csv(module_rows, module_csv, ["module", "parameters", "default_trainable_parameters", "parameter_percent"])
    write_csv(policy_rows, policy_csv, ["parameter", "shape", "numel", "policy", "matched_rule", "reason"])
    summary = {
        "stage": 18,
        "mode": "decoder freeze policy report",
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "module_summary_csv": str(module_csv),
        "freeze_policy_csv": str(policy_csv),
        "module_summary": module_rows,
        "freeze_policy_summary": policy_summary,
        "recommended_first_finetune_train_prefixes": [
            "model.decoder",
            "model.gs_dynamic_predictor",
            "model.encoder_proj",
        ],
        "recommended_first_finetune_freeze_prefixes": [
            "model.gs_predictor",
            "model.condition_encoder",
            "model.gaussian_upsampler",
        ],
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "missing_keys_sample": list(missing)[:20],
        "unexpected_keys_sample": list(unexpected)[:20],
        "notes": "First long-GOP adaptation should freeze static feature/Gaussian extraction and train only the temporal/dynamic decoder path.",
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_json),
        "module_csv": str(module_csv),
        "policy_csv": str(policy_csv),
        **policy_summary,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
