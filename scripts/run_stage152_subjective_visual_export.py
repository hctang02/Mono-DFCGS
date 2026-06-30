import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage152_subjective_visual_export"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage152_subjective_visual_export")


sys.path.insert(0, str(REPO_ROOT))
from configs.options_inference import Options  # noqa: E402
from mono_dfcgs.gaussian_codec import flatten_static_anchor, unflatten_static_anchor  # noqa: E402
from mono_dfcgs.residual_sideinfo_codec import (  # noqa: E402
    decode_residual_sideinfo_entropy,
    encode_topk_residual_sideinfo_entropy,
)
from scripts.run_stage21_gaussian_anchor_only_adapter_smoke import load_rgb, psnr_from_mse, render_anchor  # noqa: E402
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import (  # noqa: E402
    DEFAULT_DENSE_MANIFEST,
    DEFAULT_TASK_MANIFEST,
    build_dense_index,
    linear_anchor,
    load_anchor,
    parse_task_rows,
    select_balanced,
)


FRAME_FIELDS = [
    "video",
    "gap",
    "frame_number",
    "task_id",
    "sequence",
    "left_index",
    "right_index",
    "target_index",
    "normalized_time",
    "base_psnr",
    "recovered_psnr",
    "delta_psnr",
    "payload_bytes",
]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tensor_to_rgb8(tensor):
    image = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if image.dim() == 5:
        image = image[0, 0]
    elif image.dim() == 4:
        image = image[0]
    if image.shape[0] != 3:
        raise ValueError(f"expected channel-first RGB image, got {tuple(image.shape)}")
    return (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def psnr_from_render(pred, target):
    target_for_loss = target
    if pred.dim() == 5 and target.dim() == 4:
        target_for_loss = target.unsqueeze(1)
    elif pred.dim() == 4 and target.dim() == 5 and target.shape[1] == 1:
        target_for_loss = target.squeeze(1)
    mse = float(F.mse_loss(pred, target_for_loss).item())
    return psnr_from_mse(mse)


def put_label(image, text, y, color=(255, 255, 255)):
    cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def make_canvas(target_rgb, base_rgb, recovered_rgb, title, base_psnr, recovered_psnr, payload_bytes):
    panels = [target_rgb, base_rgb, recovered_rgb]
    h, w, _ = target_rgb.shape
    header = 52
    canvas = np.zeros((h + header, w * 3, 3), dtype=np.uint8)
    for idx, panel in enumerate(panels):
        canvas[header:, idx * w:(idx + 1) * w, :] = panel
    put_label(canvas, title, 18)
    put_label(canvas, "target", 42)
    cv2.putText(canvas, f"linear base PSNR {base_psnr:.2f}", (w + 8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"linear base PSNR {base_psnr:.2f}", (w + 8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"recovered PSNR {recovered_psnr:.2f}, payload {payload_bytes:.0f} B", (2 * w + 8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"recovered PSNR {recovered_psnr:.2f}, payload {payload_bytes:.0f} B", (2 * w + 8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_task(task, dense_index, cache, background, opt, device, keep_fraction, side_bits, zlib_level):
    left = load_anchor(task["left_anchor_source_item"], task["left_anchor_source_side"], device, bits=task["bits"], cache=cache)
    right = load_anchor(task["right_anchor_source_item"], task["right_anchor_source_side"], device, bits=task["bits"], cache=cache)
    dense_key = (task["dataset"], task["split"], task["sequence"], task["target_index"])
    if dense_key not in dense_index:
        raise KeyError(f"Missing dense target anchor for {dense_key}")
    target_item, target_side = dense_index[dense_key]
    target_anchor = load_anchor(target_item, target_side, device, bits=None, cache=cache)
    target_attrs = flatten_static_anchor(target_anchor)
    target_rgb = load_rgb(task["target_rgb_path"], opt.image_height, opt.image_width, device)
    base_anchor = linear_anchor(left, right, task["normalized_time"])
    base_attrs = flatten_static_anchor(base_anchor)
    payload, info = encode_topk_residual_sideinfo_entropy(
        base_attrs,
        target_attrs,
        keep_fraction,
        side_bits,
        zlib_level=zlib_level,
    )
    recovered_attrs = decode_residual_sideinfo_entropy(base_attrs, payload)
    recovered_anchor = unflatten_static_anchor(recovered_attrs)
    base_render = render_anchor(base_anchor, background, opt).clamp(0.0, 1.0)
    recovered_render = render_anchor(recovered_anchor, background, opt).clamp(0.0, 1.0)
    return {
        "target_rgb8": tensor_to_rgb8(target_rgb),
        "base_rgb8": tensor_to_rgb8(base_render),
        "recovered_rgb8": tensor_to_rgb8(recovered_render),
        "base_psnr": psnr_from_render(base_render, target_rgb),
        "recovered_psnr": psnr_from_render(recovered_render, target_rgb),
        "payload_bytes": info["payload_bytes"],
    }


def export_gap_video(gap, tasks, args, dense_index, background, opt, device):
    video_path = args.heavy_root / f"stage152_gap{gap}_target_base_recovered.mp4"
    contact_path = args.heavy_root / f"stage152_gap{gap}_contact_sheet.jpg"
    frames = []
    frame_rows = []
    cache = None if args.disable_cache else {}
    selected = select_balanced(tasks, args.frames_per_gap, args.seed + int(gap))
    for frame_number, task in enumerate(selected, start=1):
        rendered = render_task(task, dense_index, cache, background, opt, device, args.keep_fraction, args.side_bits, args.zlib_level)
        title = f"gap{gap} {task['sequence']} target {task['target_index']} t={task['normalized_time']:.3f}"
        canvas = make_canvas(
            rendered["target_rgb8"],
            rendered["base_rgb8"],
            rendered["recovered_rgb8"],
            title,
            rendered["base_psnr"],
            rendered["recovered_psnr"],
            rendered["payload_bytes"],
        )
        frames.append(canvas)
        frame_rows.append({
            "video": str(video_path),
            "gap": gap,
            "frame_number": frame_number,
            "task_id": task["task_id"],
            "sequence": task["sequence"],
            "left_index": task["left_index"],
            "right_index": task["right_index"],
            "target_index": task["target_index"],
            "normalized_time": task["normalized_time"],
            "base_psnr": rendered["base_psnr"],
            "recovered_psnr": rendered["recovered_psnr"],
            "delta_psnr": rendered["recovered_psnr"] - rendered["base_psnr"],
            "payload_bytes": rendered["payload_bytes"],
        })
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not frames:
        raise RuntimeError(f"No frames selected for gap {gap}")
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    columns = min(args.contact_columns, len(frames))
    rows = int(np.ceil(len(frames) / columns))
    sheet = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        row = idx // columns
        col = idx % columns
        sheet[row * h:(row + 1) * h, col * w:(col + 1) * w, :] = frame
    cv2.imwrite(str(contact_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return {
        "gap": gap,
        "video_path": str(video_path),
        "contact_sheet_path": str(contact_path),
        "frame_count": len(frames),
        "mean_base_psnr": sum(row["base_psnr"] for row in frame_rows) / len(frame_rows),
        "mean_recovered_psnr": sum(row["recovered_psnr"] for row in frame_rows) / len(frame_rows),
        "mean_delta_psnr": sum(row["delta_psnr"] for row in frame_rows) / len(frame_rows),
        "mean_payload_bytes": sum(row["payload_bytes"] for row in frame_rows) / len(frame_rows),
    }, frame_rows


def write_report(summary, path):
    lines = [
        "# Stage152 Subjective Visual Export",
        "",
        "## Videos",
        "",
        "| gap | frames | video | contact sheet | mean base PSNR | mean recovered PSNR |",
        "|---:|---:|---|---|---:|---:|",
    ]
    for row in summary["videos"]:
        lines.append(
            f"| {row['gap']} | {row['frame_count']} | `{row['video_path']}` | `{row['contact_sheet_path']}` | {row['mean_base_psnr']:.6f} | {row['mean_recovered_psnr']:.6f} |"
        )
    lines.extend([
        "",
        "## Layout",
        "",
        "Each frame is: target RGB | linear base render | recovered side-info render.",
        "",
        "## Contract",
        "",
        "The recovered panel uses the Stage151 policy: linear base plus decoded q6/top10 entropy index+value residual side-info payload.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--task_split", default="eval")
    parser.add_argument("--codecs", nargs="+", default=["q12"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--frames_per_gap", type=int, default=24)
    parser.add_argument("--keep_fraction", type=float, default=0.1)
    parser.add_argument("--side_bits", type=int, default=6)
    parser.add_argument("--zlib_level", type=int, default=9)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--contact_columns", type=int, default=4)
    parser.add_argument("--disable_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    opt = Options()
    opt.compile = False
    opt.output_frames = 1
    opt.down_resolution = ()
    background = torch.tensor(opt.background_color, dtype=torch.float32, device=device)
    all_tasks = parse_task_rows(args.task_manifest, args.task_split, args.codecs, args.gaps)
    if not all_tasks:
        raise RuntimeError("No tasks selected")
    dense_index = build_dense_index(args.dense_manifest, sorted({row["split"] for row in all_tasks}))
    videos = []
    all_frame_rows = []
    for gap in args.gaps:
        gap_tasks = [task for task in all_tasks if int(task["reference_gap"]) == int(gap)]
        video, frame_rows = export_gap_video(gap, gap_tasks, args, dense_index, background, opt, device)
        videos.append(video)
        all_frame_rows.extend(frame_rows)
    frames_csv = args.summary_root / "stage152_subjective_visual_frames.csv"
    summary_json = args.summary_root / "stage152_subjective_visual_export_summary.json"
    package_json = args.summary_root / "stage152_subjective_visual_export_package.json"
    report_md = args.summary_root / "stage152_subjective_visual_export_report.md"
    write_csv(all_frame_rows, frames_csv, FRAME_FIELDS)
    summary = {
        "stage": 152,
        "mode": "subjective visual export",
        "policy": "middle_frame_recovery_linear_base_entropy_sideinfo_v1",
        "summary_root": str(args.summary_root),
        "heavy_root": str(args.heavy_root),
        "videos": videos,
        "frames_csv": str(frames_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "notes": "Video outputs are intentionally stored outside git under the heavy root.",
    }
    package = {
        "stage": 152,
        "mode": summary["mode"],
        "policy": summary["policy"],
        "videos": videos,
        "frames_csv": str(frames_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary, report_md)
    print(json.dumps({"package": str(package_json), "videos": videos}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
