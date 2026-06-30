import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage153_middle_multimetric_badcase_eval"
DEFAULT_HEAVY_ROOT = Path("/data/hctang/tmp/opencode/mono_dfcgs_runs/stage153_middle_multimetric_badcase_eval")


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


ROW_FIELDS = [
    "task_id",
    "sequence",
    "gap",
    "codec",
    "left_index",
    "right_index",
    "target_index",
    "normalized_time",
    "method",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "payload_bytes",
    "delta_psnr_vs_base",
    "delta_ssim_vs_base",
    "delta_ms_ssim_vs_base",
    "delta_lpips_vs_base",
]

SUMMARY_FIELDS = [
    "gap",
    "method",
    "task_count",
    "mean_psnr",
    "min_psnr",
    "p10_psnr",
    "mean_ssim",
    "min_ssim",
    "p10_ssim",
    "mean_ms_ssim",
    "min_ms_ssim",
    "p10_ms_ssim",
    "mean_lpips",
    "max_lpips",
    "p90_lpips",
    "mean_payload_bytes",
]

BADCASE_FIELDS = [
    "rank_type",
    "rank",
    "task_id",
    "sequence",
    "gap",
    "target_index",
    "normalized_time",
    "base_psnr",
    "recovered_psnr",
    "base_ssim",
    "recovered_ssim",
    "base_ms_ssim",
    "recovered_ms_ssim",
    "base_lpips",
    "recovered_lpips",
    "payload_bytes",
    "contact_sheet_path",
]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def to_nchw(tensor):
    image = tensor.float().clamp(0.0, 1.0)
    if image.dim() == 5:
        if image.shape[1] != 1:
            raise ValueError(f"cannot squeeze non-singleton time dimension from {tuple(image.shape)}")
        image = image[:, 0]
    if image.dim() != 4 or image.shape[1] != 3:
        raise ValueError(f"expected [B,3,H,W], got {tuple(image.shape)}")
    return image


def tensor_to_rgb8(tensor):
    image = to_nchw(tensor).detach().cpu()[0]
    return (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def psnr_metric(pred, target):
    pred = to_nchw(pred)
    target = to_nchw(target)
    mse = float(F.mse_loss(pred, target).item())
    return psnr_from_mse(mse)


def ssim_metric(pred, target):
    pred_np = to_nchw(pred).detach().cpu()[0].numpy()
    target_np = to_nchw(target).detach().cpu()[0].numpy()
    return float(structural_similarity(target_np, pred_np, win_size=11, gaussian_weights=True, channel_axis=0, data_range=1.0))


def ms_ssim_metric(pred, target, ms_ssim_module):
    if ms_ssim_module is None:
        return None
    pred = to_nchw(pred)
    target = to_nchw(target)
    with torch.no_grad():
        return float(ms_ssim_module(pred, target).item())


def lpips_metric(pred, target, lpips_model):
    if lpips_model is None:
        return None
    pred = to_nchw(pred)
    target = to_nchw(target)
    with torch.no_grad():
        return float(lpips_model(target, pred, normalize=True)[:, 0, 0, 0].item())


def load_metric_modules(device, disable_lpips, disable_ms_ssim):
    lpips_model = None
    lpips_error = None
    if not disable_lpips:
        try:
            from lpips import LPIPS

            lpips_model = LPIPS(net="vgg").to(device).eval()
            for param in lpips_model.parameters():
                param.requires_grad_(False)
        except Exception as exc:  # pragma: no cover - diagnostic path
            lpips_error = repr(exc)
    ms_ssim_module = None
    ms_ssim_error = None
    if not disable_ms_ssim:
        try:
            from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

            ms_ssim_module = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device).eval()
        except Exception as exc:  # pragma: no cover - diagnostic path
            ms_ssim_error = repr(exc)
    return lpips_model, lpips_error, ms_ssim_module, ms_ssim_error


def render_task(task, dense_index, cache, background, opt, device, keep_fraction, side_bits, zlib_level):
    left = load_anchor(task["left_anchor_source_item"], task["left_anchor_source_side"], device, bits=task["bits"], cache=cache)
    right = load_anchor(task["right_anchor_source_item"], task["right_anchor_source_side"], device, bits=task["bits"], cache=cache)
    dense_key = (task["dataset"], task["split"], task["sequence"], task["target_index"])
    if dense_key not in dense_index:
        raise KeyError(f"Missing dense target anchor for {dense_key}")
    target_item, target_side = dense_index[dense_key]
    target_anchor = load_anchor(target_item, target_side, device, bits=None, cache=cache)
    target_rgb = to_nchw(load_rgb(task["target_rgb_path"], opt.image_height, opt.image_width, device))
    base_anchor = linear_anchor(left, right, task["normalized_time"])
    base_attrs = flatten_static_anchor(base_anchor)
    target_attrs = flatten_static_anchor(target_anchor)
    payload, info = encode_topk_residual_sideinfo_entropy(
        base_attrs,
        target_attrs,
        keep_fraction,
        side_bits,
        zlib_level=zlib_level,
    )
    recovered_attrs = decode_residual_sideinfo_entropy(base_attrs, payload)
    recovered_anchor = unflatten_static_anchor(recovered_attrs)
    base_render = to_nchw(render_anchor(base_anchor, background, opt).clamp(0.0, 1.0))
    recovered_render = to_nchw(render_anchor(recovered_anchor, background, opt).clamp(0.0, 1.0))
    return {
        "target_rgb": target_rgb,
        "base_render": base_render,
        "recovered_render": recovered_render,
        "payload_bytes": info["payload_bytes"],
    }


def compute_method_metrics(pred, target, lpips_model, ms_ssim_module):
    return {
        "psnr": psnr_metric(pred, target),
        "ssim": ssim_metric(pred, target),
        "ms_ssim": ms_ssim_metric(pred, target, ms_ssim_module),
        "lpips": lpips_metric(pred, target, lpips_model),
    }


def percentile(values, q):
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def mean(values):
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not values:
        return None
    return float(np.mean(values))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["gap"]), row["method"])].append(row)
    out = []
    for (gap, method), group in sorted(groups.items()):
        out.append({
            "gap": gap,
            "method": method,
            "task_count": len(group),
            "mean_psnr": mean(row["psnr"] for row in group),
            "min_psnr": percentile((row["psnr"] for row in group), 0),
            "p10_psnr": percentile((row["psnr"] for row in group), 10),
            "mean_ssim": mean(row["ssim"] for row in group),
            "min_ssim": percentile((row["ssim"] for row in group), 0),
            "p10_ssim": percentile((row["ssim"] for row in group), 10),
            "mean_ms_ssim": mean(row["ms_ssim"] for row in group),
            "min_ms_ssim": percentile((row["ms_ssim"] for row in group), 0),
            "p10_ms_ssim": percentile((row["ms_ssim"] for row in group), 10),
            "mean_lpips": mean(row["lpips"] for row in group),
            "max_lpips": percentile((row["lpips"] for row in group), 100),
            "p90_lpips": percentile((row["lpips"] for row in group), 90),
            "mean_payload_bytes": mean(row["payload_bytes"] for row in group),
        })
    return out


def put_label(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def make_canvas(target, base, recovered, task, base_metrics, recovered_metrics, payload_bytes):
    target_rgb = tensor_to_rgb8(target)
    base_rgb = tensor_to_rgb8(base)
    recovered_rgb = tensor_to_rgb8(recovered)
    h, w, _ = target_rgb.shape
    header = 58
    canvas = np.zeros((h + header, w * 3, 3), dtype=np.uint8)
    canvas[header:, :w] = target_rgb
    canvas[header:, w:2 * w] = base_rgb
    canvas[header:, 2 * w:] = recovered_rgb
    title = f"gap{task['reference_gap']} {task['sequence']} target {task['target_index']} t={task['normalized_time']:.3f}"
    put_label(canvas, title, 8, 18)
    put_label(canvas, "target", 8, 45)
    put_label(canvas, f"base P {base_metrics['psnr']:.2f} S {base_metrics['ssim']:.3f}", w + 8, 45)
    lpips = recovered_metrics["lpips"]
    lpips_text = "NA" if lpips is None else f"{lpips:.3f}"
    put_label(canvas, f"recovered P {recovered_metrics['psnr']:.2f} S {recovered_metrics['ssim']:.3f} L {lpips_text} B {payload_bytes:.0f}", 2 * w + 8, 45)
    return canvas


def save_badcase_contact_sheets(cases, rendered_cache, args):
    paths = {}
    for rank_type, group in cases.items():
        frames = []
        for case in group:
            key = case["task_id"]
            if key not in rendered_cache:
                continue
            cached = rendered_cache[key]
            frames.append(make_canvas(
                cached["target_rgb"],
                cached["base_render"],
                cached["recovered_render"],
                cached["task"],
                cached["base_metrics"],
                cached["recovered_metrics"],
                cached["payload_bytes"],
            ))
        if not frames:
            continue
        h, w, _ = frames[0].shape
        columns = min(args.contact_columns, len(frames))
        rows = int(math.ceil(len(frames) / columns))
        sheet = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
        for idx, frame in enumerate(frames):
            row = idx // columns
            col = idx % columns
            sheet[row * h:(row + 1) * h, col * w:(col + 1) * w] = frame
        path = args.heavy_root / f"stage153_badcases_{rank_type}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        paths[rank_type] = str(path)
    return paths


def build_badcases(pair_rows, top_n):
    cases = {
        "highest_recovered_lpips": sorted(
            [row for row in pair_rows if row["recovered_lpips"] is not None],
            key=lambda row: float(row["recovered_lpips"]),
            reverse=True,
        )[:top_n],
        "lowest_recovered_ssim": sorted(pair_rows, key=lambda row: float(row["recovered_ssim"]))[:top_n],
        "lowest_recovered_psnr": sorted(pair_rows, key=lambda row: float(row["recovered_psnr"]))[:top_n],
        "largest_lpips_regression": sorted(
            [row for row in pair_rows if row["recovered_lpips"] is not None and row["base_lpips"] is not None],
            key=lambda row: float(row["recovered_lpips"]) - float(row["base_lpips"]),
            reverse=True,
        )[:top_n],
    }
    out = []
    for rank_type, rows in cases.items():
        for rank, row in enumerate(rows, start=1):
            item = dict(row)
            item["rank_type"] = rank_type
            item["rank"] = rank
            out.append(item)
    return cases, out


def write_report(summary_rows, badcase_rows, package, path):
    lines = [
        "# Stage153 Middle Multi-Metric Bad-Case Evaluation",
        "",
        "## Summary",
        "",
        "| gap | method | tasks | PSNR mean | PSNR p10 | SSIM mean | MS-SSIM mean | LPIPS mean | LPIPS p90 | payload bytes |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['gap']} | {row['method']} | {row['task_count']} | {float(row['mean_psnr']):.6f} | {float(row['p10_psnr']):.6f} | {float(row['mean_ssim']):.6f} | {format_optional(row['mean_ms_ssim'])} | {format_optional(row['mean_lpips'])} | {format_optional(row['p90_lpips'])} | {float(row['mean_payload_bytes']):.3f} |"
        )
    lines.extend([
        "",
        "## Bad-Case Contact Sheets",
        "",
    ])
    for key, value in package["badcase_contact_sheets"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Worst Recovered Cases",
        "",
        "| rank type | rank | sequence | gap | target | base PSNR | recovered PSNR | base SSIM | recovered SSIM | base LPIPS | recovered LPIPS |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in badcase_rows[: min(40, len(badcase_rows))]:
        lines.append(
            f"| {row['rank_type']} | {row['rank']} | {row['sequence']} | {row['gap']} | {row['target_index']} | {float(row['base_psnr']):.6f} | {float(row['recovered_psnr']):.6f} | {float(row['base_ssim']):.6f} | {float(row['recovered_ssim']):.6f} | {format_optional(row['base_lpips'])} | {format_optional(row['recovered_lpips'])} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Stage153 is diagnostic: it evaluates whether PSNR gains correspond to SSIM/LPIPS and visual sanity.",
        "- If LPIPS or bad-case sheets show visually broken frames, the next stage must move from linear-base recovery to original StreamSplat-guided recovery.",
        "- Side-info remains rate-counted; target dense anchors are used only encoder-side to build the payload and metrics.",
        "",
        "## Outputs",
        "",
        f"- rows CSV: `{package['rows_csv']}`",
        f"- pair rows CSV: `{package['pair_rows_csv']}`",
        f"- badcases CSV: `{package['badcases_csv']}`",
        f"- summary CSV: `{package['summary_csv']}`",
        f"- summary JSON: `{package['summary_json']}`",
        f"- package JSON: `{package['package_json']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_optional(value):
    if value is None:
        return "NA"
    value = float(value)
    if math.isnan(value):
        return "NA"
    return f"{value:.6f}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--task_split", default="eval")
    parser.add_argument("--codecs", nargs="+", default=["q12"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--max_tasks", type=int, default=120)
    parser.add_argument("--keep_fraction", type=float, default=0.1)
    parser.add_argument("--side_bits", type=int, default=6)
    parser.add_argument("--zlib_level", type=int, default=9)
    parser.add_argument("--top_badcases", type=int, default=12)
    parser.add_argument("--contact_columns", type=int, default=4)
    parser.add_argument("--disable_cache", action="store_true")
    parser.add_argument("--disable_lpips", action="store_true")
    parser.add_argument("--disable_ms_ssim", action="store_true")
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
    tasks = select_balanced(parse_task_rows(args.task_manifest, args.task_split, args.codecs, args.gaps), args.max_tasks, args.seed)
    dense_index = build_dense_index(args.dense_manifest, sorted({row["split"] for row in tasks}))
    lpips_model, lpips_error, ms_ssim_module, ms_ssim_error = load_metric_modules(device, args.disable_lpips, args.disable_ms_ssim)
    cache = None if args.disable_cache else {}
    rows = []
    pair_rows = []
    rendered_cache = {}
    for index, task in enumerate(tasks, start=1):
        rendered = render_task(task, dense_index, cache, background, opt, device, args.keep_fraction, args.side_bits, args.zlib_level)
        base_metrics = compute_method_metrics(rendered["base_render"], rendered["target_rgb"], lpips_model, ms_ssim_module)
        recovered_metrics = compute_method_metrics(rendered["recovered_render"], rendered["target_rgb"], lpips_model, ms_ssim_module)
        base_row = {
            "task_id": task["task_id"],
            "sequence": task["sequence"],
            "gap": task["reference_gap"],
            "codec": task["codec"],
            "left_index": task["left_index"],
            "right_index": task["right_index"],
            "target_index": task["target_index"],
            "normalized_time": task["normalized_time"],
            "method": "linear_base",
            "payload_bytes": 0,
            **base_metrics,
            "delta_psnr_vs_base": 0.0,
            "delta_ssim_vs_base": 0.0,
            "delta_ms_ssim_vs_base": 0.0,
            "delta_lpips_vs_base": 0.0,
        }
        recovered_row = {
            "task_id": task["task_id"],
            "sequence": task["sequence"],
            "gap": task["reference_gap"],
            "codec": task["codec"],
            "left_index": task["left_index"],
            "right_index": task["right_index"],
            "target_index": task["target_index"],
            "normalized_time": task["normalized_time"],
            "method": "stage151_recovered_linear_base_sideinfo",
            "payload_bytes": rendered["payload_bytes"],
            **recovered_metrics,
            "delta_psnr_vs_base": recovered_metrics["psnr"] - base_metrics["psnr"],
            "delta_ssim_vs_base": recovered_metrics["ssim"] - base_metrics["ssim"],
            "delta_ms_ssim_vs_base": optional_delta(recovered_metrics["ms_ssim"], base_metrics["ms_ssim"]),
            "delta_lpips_vs_base": optional_delta(recovered_metrics["lpips"], base_metrics["lpips"]),
        }
        rows.extend([base_row, recovered_row])
        pair_rows.append({
            "task_id": task["task_id"],
            "sequence": task["sequence"],
            "gap": task["reference_gap"],
            "target_index": task["target_index"],
            "normalized_time": task["normalized_time"],
            "base_psnr": base_metrics["psnr"],
            "recovered_psnr": recovered_metrics["psnr"],
            "base_ssim": base_metrics["ssim"],
            "recovered_ssim": recovered_metrics["ssim"],
            "base_ms_ssim": base_metrics["ms_ssim"],
            "recovered_ms_ssim": recovered_metrics["ms_ssim"],
            "base_lpips": base_metrics["lpips"],
            "recovered_lpips": recovered_metrics["lpips"],
            "payload_bytes": rendered["payload_bytes"],
            "contact_sheet_path": "",
        })
        rendered_cache[task["task_id"]] = {
            **rendered,
            "task": task,
            "base_metrics": base_metrics,
            "recovered_metrics": recovered_metrics,
        }
        if index % 20 == 0:
            print(json.dumps({"processed": index, "total": len(tasks)}), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary_rows = summarize(rows)
    cases, badcase_rows = build_badcases(pair_rows, args.top_badcases)
    contact_sheets = save_badcase_contact_sheets(cases, rendered_cache, args)
    for row in badcase_rows:
        row["contact_sheet_path"] = contact_sheets.get(row["rank_type"], "")
    rows_csv = args.summary_root / "stage153_middle_multimetric_rows.csv"
    pair_rows_csv = args.summary_root / "stage153_middle_multimetric_pair_rows.csv"
    badcases_csv = args.summary_root / "stage153_middle_multimetric_badcases.csv"
    summary_csv = args.summary_root / "stage153_middle_multimetric_summary.csv"
    summary_json = args.summary_root / "stage153_middle_multimetric_summary.json"
    package_json = args.summary_root / "stage153_middle_multimetric_badcase_eval_package.json"
    report_md = args.summary_root / "stage153_middle_multimetric_badcase_eval_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(pair_rows, pair_rows_csv, [field for field in BADCASE_FIELDS if field not in {"rank_type", "rank", "contact_sheet_path"}])
    write_csv(badcase_rows, badcases_csv, BADCASE_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    package = {
        "stage": 153,
        "mode": "middle multi-metric bad-case evaluation",
        "policy_under_test": "stage151_recovered_linear_base_sideinfo",
        "task_count": len(tasks),
        "summary_rows": summary_rows,
        "lpips_available": lpips_model is not None,
        "lpips_error": lpips_error,
        "ms_ssim_available": ms_ssim_module is not None,
        "ms_ssim_error": ms_ssim_error,
        "badcase_contact_sheets": contact_sheets,
        "rows_csv": str(rows_csv),
        "pair_rows_csv": str(pair_rows_csv),
        "badcases_csv": str(badcases_csv),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "heavy_root": str(args.heavy_root),
        "notes": "Contact sheets are stored outside git. Target dense anchors are used encoder-side only to produce the rate-counted residual payload and diagnostics.",
    }
    summary_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(summary_rows, badcase_rows, package, report_md)
    print(json.dumps({"package": str(package_json), "summary": summary_rows, "contact_sheets": contact_sheets}, indent=2))


def optional_delta(value, base):
    if value is None or base is None:
        return None
    return float(value) - float(base)


if __name__ == "__main__":
    raise SystemExit(main())
