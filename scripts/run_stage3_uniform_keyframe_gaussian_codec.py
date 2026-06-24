import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_CSV = REPO_ROOT / "experiments/stage1_streamsplat_fair_metrics/stage1_streamsplat_fair_metrics_summary.csv"
DEFAULT_STAGE2_CSV = REPO_ROOT / "experiments/stage2_keyframe_gaussian_anchor/stage2_keyframe_gaussian_anchor_summary.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage3_uniform_keyframe_gaussian_codec"


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def build_rows(stage1_rows, stage2_rows):
    quality = {(row["sample"], row["frame_gap"]): row for row in stage1_rows}
    rows = []
    for rate in stage2_rows:
        key = (rate["sample"], rate["frame_gap"])
        q = quality.get(key)
        if q is None:
            continue
        rows.append({
            "method": "Mono-DFCGS uniform keyframe Gaussian",
            "sample": rate["sample"],
            "frame_gap": int(rate["frame_gap"]),
            "keyframe_count": int(rate["keyframe_count"]),
            "keyframe_ratio": to_float(rate, "keyframe_ratio"),
            "profile": rate["profile"],
            "codec": rate["codec"],
            "opacity_threshold": to_float(rate, "opacity_threshold"),
            "gaussians_kept": int(rate["gaussians_kept"]),
            "gaussian_keep_ratio": to_float(rate, "keep_ratio"),
            "total_transmitted_mib": to_float(rate, "total_mib"),
            "avg_transmitted_mib_per_frame": to_float(rate, "avg_mib_per_video_frame"),
            "all_psnr_avg": to_float(q, "all_psnr_avg"),
            "all_ssim_avg": to_float(q, "all_ssim_avg"),
            "middle_psnr_avg": to_float(q, "middle_psnr_avg"),
            "middle_ssim_avg": to_float(q, "middle_ssim_avg"),
            "given_psnr_avg": to_float(q, "given_psnr_avg"),
            "given_ssim_avg": to_float(q, "given_ssim_avg"),
            "rate_note": "Only transmitted keyframe Gaussian anchor payload is counted; decoder-generated intermediate Gaussians are excluded.",
            "quality_note": "Quality comes from StreamSplat sparse-keyframe reconstruction at the same uniform gap.",
        })
    return rows


def write_csv(rows, path):
    fields = [
        "method",
        "sample",
        "frame_gap",
        "keyframe_count",
        "keyframe_ratio",
        "profile",
        "codec",
        "opacity_threshold",
        "gaussians_kept",
        "gaussian_keep_ratio",
        "total_transmitted_mib",
        "avg_transmitted_mib_per_frame",
        "all_psnr_avg",
        "all_ssim_avg",
        "middle_psnr_avg",
        "middle_ssim_avg",
        "given_psnr_avg",
        "given_ssim_avg",
        "rate_note",
        "quality_note",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_main_rows(rows):
    return [
        row for row in rows
        if row["profile"] == "static_anchor"
        and row["codec"] == "q8"
        and abs(row["opacity_threshold"] - 0.0) < 1e-9
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_csv", type=Path, default=DEFAULT_STAGE1_CSV)
    parser.add_argument("--stage2_csv", type=Path, default=DEFAULT_STAGE2_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.stage1_csv.exists():
        raise FileNotFoundError(args.stage1_csv)
    if not args.stage2_csv.exists():
        raise FileNotFoundError(args.stage2_csv)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    rows = build_rows(read_csv(args.stage1_csv), read_csv(args.stage2_csv))
    main_rows = select_main_rows(rows)

    summary_path = args.summary_root / "stage3_uniform_keyframe_gaussian_codec_summary.json"
    csv_path = args.summary_root / "stage3_uniform_keyframe_gaussian_codec_summary.csv"
    main_csv_path = args.summary_root / "stage3_uniform_keyframe_gaussian_codec_main_q8.csv"
    summary_path.write_text(json.dumps({"rows": rows, "main_rows": main_rows}, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    write_csv(main_rows, main_csv_path)
    print(json.dumps({
        "summary": str(summary_path),
        "csv": str(csv_path),
        "main_csv": str(main_csv_path),
        "rows": len(rows),
        "main_rows": len(main_rows),
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
