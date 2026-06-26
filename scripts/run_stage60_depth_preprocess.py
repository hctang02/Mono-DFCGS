import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage60_depth_preprocess"
DEFAULT_STREAMSPLAT_ROOT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat")
DEFAULT_DAVIS_ROOT = Path("/mnt/hdd2tC/tmp/opencode/datasets/DAVIS")
DEFAULT_VOS_ROOT = Path("/mnt/hdd2tC/tmp/opencode/datasets/YouTube-VOS")

FRAME_FIELDS = [
    "dataset", "split", "sequence", "frame", "image_path", "depth_path", "status", "message",
]


def read_split(path):
    path = Path(path)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sorted_images(path):
    path = Path(path)
    if not path.exists():
        return []
    return sorted(path.glob("*.jpg"), key=lambda p: p.name)


def davis_frames(root):
    root = Path(root)
    image_root = root / "JPEGImages/Full-Resolution"
    split_names = []
    for split in ("train", "val"):
        split_names.extend((split, name) for name in read_split(root / "ImageSets/2017" / f"{split}.txt"))
    if not split_names and image_root.exists():
        split_names = [("unspecified", path.name) for path in sorted(path for path in image_root.iterdir() if path.is_dir())]
    for split, sequence in split_names:
        for image_path in sorted_images(image_root / sequence):
            depth_path = Path(str(image_path).replace("JPEGImages", "depthImages")).with_name(f"{image_path.stem}_pred.png")
            yield {
                "dataset": "DAVIS",
                "split": split,
                "sequence": sequence,
                "frame": image_path.stem,
                "image_path": image_path,
                "depth_path": depth_path,
            }


def vos_frames(root, include_train=True, include_valid=True):
    root = Path(root)
    split_dirs = []
    if include_train:
        split_dirs.append(("train", root / "train/JPEGImages"))
    if include_valid:
        split_dirs.append(("valid", root / "valid/JPEGImages"))
    for split, image_root in split_dirs:
        if not image_root.exists():
            continue
        for sequence_dir in sorted((path for path in image_root.iterdir() if path.is_dir()), key=lambda p: p.name):
            for image_path in sorted_images(sequence_dir):
                depth_path = Path(str(image_path).replace("JPEGImages", "depthImages")).with_name(f"{image_path.stem}_pred.png")
                yield {
                    "dataset": "YouTube-VOS",
                    "split": split,
                    "sequence": sequence_dir.name,
                    "frame": image_path.stem,
                    "image_path": image_path,
                    "depth_path": depth_path,
                }


def resize_to_shorter_side(image, target_size):
    width, height = image.size
    if width <= height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    return TF.resize(image, size=(new_height, new_width), interpolation=InterpolationMode.BILINEAR)


def resize_to_multiple_of_14(image):
    width, height = image.size
    new_width = max(14, round(width / 14) * 14)
    new_height = max(14, round(height / 14) * 14)
    return TF.resize(image, size=(new_height, new_width), interpolation=InterpolationMode.BILINEAR)


def load_model(streamsplat_root, model_name, device):
    sys.path.insert(0, str(streamsplat_root))
    from model.depth_wrapper import DepthAnythingWrapper

    model = DepthAnythingWrapper(model_name)
    model.eval().to(device)
    return model


def predict_depth(model, image_path, target_short_side, device):
    frame = Image.open(image_path).convert("RGB")
    model_input = resize_to_multiple_of_14(resize_to_shorter_side(frame, target_short_side))
    tensor = TF.to_tensor(model_input).unsqueeze(0).to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                depth = model(tensor).detach().float()
        else:
            depth = model(tensor).detach().float()
        depth = F.interpolate(depth[:, None], size=(frame.height, frame.width), mode="bilinear", align_corners=True)[0, 0]
    depth_np = depth.cpu().numpy()
    min_value = float(np.min(depth_np))
    max_value = float(np.max(depth_np))
    if max_value - min_value < 1e-8:
        return np.zeros_like(depth_np, dtype=np.uint8)
    return ((depth_np - min_value) / (max_value - min_value) * 255.0).clip(0, 255).astype(np.uint8)


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FRAME_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--streamsplat_root", type=Path, default=DEFAULT_STREAMSPLAT_ROOT)
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--vos_root", type=Path, default=DEFAULT_VOS_ROOT)
    parser.add_argument("--datasets", nargs="*", default=["davis"])
    parser.add_argument("--include_vos_train", action="store_true")
    parser.add_argument("--include_vos_valid", action="store_true")
    parser.add_argument("--model_name", default="vitl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_short_side", type=int, default=518)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    frame_rows = []
    candidates = []
    if "davis" in args.datasets:
        candidates.extend(davis_frames(args.davis_root))
    if "vos" in args.datasets:
        candidates.extend(vos_frames(args.vos_root, include_train=args.include_vos_train, include_valid=args.include_vos_valid))
    if args.max_frames > 0:
        candidates = candidates[: args.max_frames]

    model = load_model(args.streamsplat_root, args.model_name, device)
    processed = 0
    skipped = 0
    failed = 0
    for index, item in enumerate(candidates, start=1):
        depth_path = item["depth_path"]
        if args.skip_existing and depth_path.exists():
            skipped += 1
            frame_rows.append({**{key: str(value) for key, value in item.items()}, "status": "skipped", "message": "depth exists"})
            continue
        try:
            depth = predict_depth(model, item["image_path"], args.target_short_side, device)
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(depth_path), depth)
            processed += 1
            status = "processed"
            message = ""
        except Exception as exc:
            failed += 1
            status = "failed"
            message = repr(exc)
        frame_rows.append({**{key: str(value) for key, value in item.items()}, "status": status, "message": message})
        if args.log_every > 0 and index % args.log_every == 0:
            print(f"Stage60 {index}/{len(candidates)} processed={processed} skipped={skipped} failed={failed}", flush=True)

    frame_csv = args.summary_root / "stage60_depth_preprocess_frames.csv"
    summary_json = args.summary_root / "stage60_depth_preprocess_summary.json"
    write_csv(frame_rows, frame_csv)
    summary = {
        "stage": 60,
        "mode": "DepthAnything V2 depth preprocessing for DAVIS/YouTube-VOS",
        "streamsplat_root": str(args.streamsplat_root),
        "davis_root": str(args.davis_root),
        "vos_root": str(args.vos_root),
        "datasets": args.datasets,
        "include_vos_train": args.include_vos_train,
        "include_vos_valid": args.include_vos_valid,
        "model_name": args.model_name,
        "device": str(device),
        "target_short_side": args.target_short_side,
        "skip_existing": args.skip_existing,
        "candidate_frames": len(candidates),
        "processed_frames": processed,
        "skipped_frames": skipped,
        "failed_frames": failed,
        "frame_csv": str(frame_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failed:
        raise RuntimeError(f"Stage60 failed for {failed} frames")


if __name__ == "__main__":
    raise SystemExit(main())
