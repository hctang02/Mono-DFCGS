import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE51_CSV = REPO_ROOT / "experiments/stage51_high_rate_multibit_rd/stage51_high_rate_multibit_rd.csv"
DEFAULT_STAGE57_CSV = REPO_ROOT / "experiments/stage57_compact_anchor_codec/stage57_compact_anchor_codec.csv"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage58_compression_rd_ablation"


RD_FIELDS = [
    "sample", "method", "reference_gap", "bits", "codec_variant", "rate_scope",
    "rate_mib_per_frame", "all_psnr", "linear_all_psnr", "adapter_all_psnr",
    "total_frames", "keyframe_count", "source_rate_table", "notes",
]

MEAN_FIELDS = [
    "codec_variant", "method", "bits", "reference_gap", "sample_count",
    "mean_rate_mib_per_frame", "mean_all_psnr", "min_all_psnr", "max_all_psnr",
]

CODEC_SUMMARY_FIELDS = [
    "codec_variant", "rows", "mean_rate_mib_per_frame", "min_rate_mib_per_frame",
    "max_rate_mib_per_frame", "mean_all_psnr", "rate_scope",
]

SAVING_FIELDS = [
    "sample", "method", "reference_gap", "bits", "baseline_variant", "compact_variant",
    "baseline_rate_mib_per_frame", "compact_rate_mib_per_frame", "rate_saving_percent", "all_psnr",
]

VARIANT_COLORS = {
    "legacy_dtype_raw": "#7f7f7f",
    "legacy_dtype_zlib": "#1f77b4",
    "compact_bitpack_raw_payload_estimate": "#ff7f0e",
    "stage57_compact_raw_actual": "#2ca02c",
    "stage57_compact_zlib_actual": "#d62728",
}


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, fields, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_stage57_index(stage57_rows):
    out = {}
    for row in stage57_rows:
        key = (row["sample"], row["method"], int(row["reference_gap"]), int(row["bits"]))
        out[key] = row
    return out


def add_rd_row(rows, source, codec_variant, rate_scope, rate_value, source_rate_table, notes):
    rows.append({
        "sample": source["sample"],
        "method": source["method"],
        "reference_gap": int(source["reference_gap"]),
        "bits": int(source["bits"]),
        "codec_variant": codec_variant,
        "rate_scope": rate_scope,
        "rate_mib_per_frame": float(rate_value),
        "all_psnr": float(source["adapter_all_psnr"]),
        "linear_all_psnr": float(source["linear_all_psnr"]),
        "adapter_all_psnr": float(source["adapter_all_psnr"]),
        "total_frames": int(source["total_frames"]),
        "keyframe_count": int(source["keyframe_count"]),
        "source_rate_table": source_rate_table,
        "notes": notes,
    })


def build_rd_rows(stage51_rows, stage57_rows):
    stage57_index = build_stage57_index(stage57_rows)
    rows = []
    for source in stage51_rows:
        add_rd_row(
            rows,
            source,
            "legacy_dtype_raw",
            "actual Stage50 dtype raw container including metadata",
            source["raw_mib_per_frame"],
            "stage51/stage50",
            "Stage50 storage prototype: q<=8 uint8, q>8 uint16.",
        )
        add_rd_row(
            rows,
            source,
            "legacy_dtype_zlib",
            "actual Stage50 dtype zlib container including metadata",
            source["zlib_mib_per_frame"],
            "stage51/stage50",
            "Generic zlib over the Stage50 dtype payload.",
        )
        add_rd_row(
            rows,
            source,
            "compact_bitpack_raw_payload_estimate",
            "theoretical bitpacked payload only, no metadata/header",
            source["theoretical_bitpacked_mib_per_frame"],
            "stage51/stage50",
            "Payload-only estimate used before Stage57 actual compact containers.",
        )

        key = (source["sample"], source["method"], int(source["reference_gap"]), int(source["bits"]))
        stage57 = stage57_index.get(key)
        if stage57 is None:
            continue
        add_rd_row(
            rows,
            source,
            "stage57_compact_raw_actual",
            "actual Stage57 bitpacked raw container including metadata",
            stage57["compact_raw_mib_per_frame"],
            "stage57",
            "Available for the Stage57 formal subset only.",
        )
        add_rd_row(
            rows,
            source,
            "stage57_compact_zlib_actual",
            "actual Stage57 bitpacked zlib container including metadata",
            stage57["compact_zlib_mib_per_frame"],
            "stage57",
            "Available for the Stage57 formal subset only.",
        )
    return rows


def aggregate_mean(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["codec_variant"], row["method"], int(row["bits"]), int(row["reference_gap"]))].append(row)
    out = []
    for (codec_variant, method, bits, gap), group in sorted(grouped.items()):
        rates = [float(row["rate_mib_per_frame"]) for row in group]
        psnrs = [float(row["all_psnr"]) for row in group]
        out.append({
            "codec_variant": codec_variant,
            "method": method,
            "bits": bits,
            "reference_gap": gap,
            "sample_count": len(group),
            "mean_rate_mib_per_frame": float(np.mean(rates)),
            "mean_all_psnr": float(np.mean(psnrs)),
            "min_all_psnr": float(np.min(psnrs)),
            "max_all_psnr": float(np.max(psnrs)),
        })
    return out


def aggregate_codec_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["codec_variant"]].append(row)
    out = []
    for codec_variant, group in sorted(grouped.items()):
        rates = [float(row["rate_mib_per_frame"]) for row in group]
        psnrs = [float(row["all_psnr"]) for row in group]
        scopes = sorted({row["rate_scope"] for row in group})
        out.append({
            "codec_variant": codec_variant,
            "rows": len(group),
            "mean_rate_mib_per_frame": float(np.mean(rates)),
            "min_rate_mib_per_frame": float(np.min(rates)),
            "max_rate_mib_per_frame": float(np.max(rates)),
            "mean_all_psnr": float(np.mean(psnrs)),
            "rate_scope": "; ".join(scopes),
        })
    return out


def actual_compact_savings(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["sample"], row["method"], int(row["reference_gap"]), int(row["bits"]))
        grouped[key][row["codec_variant"]] = row
    out = []
    pairs = [
        ("legacy_dtype_raw", "stage57_compact_raw_actual"),
        ("legacy_dtype_zlib", "stage57_compact_zlib_actual"),
    ]
    for (sample, method, gap, bits), variants in sorted(grouped.items()):
        for baseline_variant, compact_variant in pairs:
            baseline = variants.get(baseline_variant)
            compact = variants.get(compact_variant)
            if baseline is None or compact is None:
                continue
            baseline_rate = float(baseline["rate_mib_per_frame"])
            compact_rate = float(compact["rate_mib_per_frame"])
            saving = 0.0 if baseline_rate == 0.0 else 100.0 * (baseline_rate - compact_rate) / baseline_rate
            out.append({
                "sample": sample,
                "method": method,
                "reference_gap": gap,
                "bits": bits,
                "baseline_variant": baseline_variant,
                "compact_variant": compact_variant,
                "baseline_rate_mib_per_frame": baseline_rate,
                "compact_rate_mib_per_frame": compact_rate,
                "rate_saving_percent": saving,
                "all_psnr": float(compact["all_psnr"]),
            })
    return out


def aggregate_savings_by_bits(saving_rows):
    grouped = defaultdict(list)
    for row in saving_rows:
        grouped[(row["baseline_variant"], row["compact_variant"], int(row["bits"]))].append(row)
    out = []
    for (baseline_variant, compact_variant, bits), group in sorted(grouped.items()):
        out.append({
            "baseline_variant": baseline_variant,
            "compact_variant": compact_variant,
            "bits": bits,
            "rows": len(group),
            "mean_rate_saving_percent": float(np.mean([row["rate_saving_percent"] for row in group])),
            "mean_baseline_rate_mib_per_frame": float(np.mean([row["baseline_rate_mib_per_frame"] for row in group])),
            "mean_compact_rate_mib_per_frame": float(np.mean([row["compact_rate_mib_per_frame"] for row in group])),
            "mean_all_psnr": float(np.mean([row["all_psnr"] for row in group])),
        })
    return out


def plot_full_mean(mean_rows, out_path):
    variants = ["legacy_dtype_raw", "legacy_dtype_zlib", "compact_bitpack_raw_payload_estimate"]
    methods = ["uniform", "rendered_prior_0p1"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9), sharey=True)
    for ax, method in zip(axes, methods):
        for variant in variants:
            points = sorted(
                [row for row in mean_rows if row["method"] == method and row["codec_variant"] == variant],
                key=lambda row: float(row["mean_rate_mib_per_frame"]),
            )
            if not points:
                continue
            xs = [float(row["mean_rate_mib_per_frame"]) for row in points]
            ys = [float(row["mean_all_psnr"]) for row in points]
            ax.scatter(xs, ys, s=24, color=VARIANT_COLORS[variant], label=variant, alpha=0.82)
            for row in points:
                ax.annotate(
                    f"q{row['bits']}g{row['reference_gap']}",
                    (float(row["mean_rate_mib_per_frame"]), float(row["mean_all_psnr"])),
                    textcoords="offset points",
                    xytext=(2, 2),
                    fontsize=5,
                    color=VARIANT_COLORS[variant],
                )
        ax.set_title(method)
        ax.set_xlabel("Mean rate (MiB/frame)")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Mean all-frame PSNR (dB)")
    fig.suptitle("Stage58 Compression RD Ablation: Full Stage51 Rows", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_actual_subset(mean_rows, out_path):
    variants = ["legacy_dtype_zlib", "stage57_compact_zlib_actual", "stage57_compact_raw_actual"]
    points = [
        row for row in mean_rows
        if row["method"] == "uniform" and int(row["reference_gap"]) == 16 and row["codec_variant"] in variants
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for variant in variants:
        variant_points = sorted([row for row in points if row["codec_variant"] == variant], key=lambda row: int(row["bits"]))
        if not variant_points:
            continue
        xs = [float(row["mean_rate_mib_per_frame"]) for row in variant_points]
        ys = [float(row["mean_all_psnr"]) for row in variant_points]
        ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=5.0, color=VARIANT_COLORS[variant], label=variant)
        for row in variant_points:
            ax.annotate(
                f"q{row['bits']}",
                (float(row["mean_rate_mib_per_frame"]), float(row["mean_all_psnr"])),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color=VARIANT_COLORS[variant],
            )
    ax.set_xlabel("Mean rate (MiB/frame)")
    ax.set_ylabel("Mean all-frame PSNR (dB)")
    ax.set_title("Stage58 Actual Stage57 Compact Subset: Uniform Gap16")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage51_csv", type=Path, default=DEFAULT_STAGE51_CSV)
    parser.add_argument("--stage57_csv", type=Path, default=DEFAULT_STAGE57_CSV)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)

    stage51_rows = read_csv_rows(args.stage51_csv)
    stage57_rows = read_csv_rows(args.stage57_csv)
    rd_rows = build_rd_rows(stage51_rows, stage57_rows)
    mean_rows = aggregate_mean(rd_rows)
    codec_summary_rows = aggregate_codec_summary(rd_rows)
    saving_rows = actual_compact_savings(rd_rows)
    saving_summary_rows = aggregate_savings_by_bits(saving_rows)

    rd_csv = args.summary_root / "stage58_compression_rd_ablation.csv"
    mean_csv = args.summary_root / "stage58_mean_compression_rd.csv"
    codec_summary_csv = args.summary_root / "stage58_codec_summary.csv"
    saving_csv = args.summary_root / "stage58_actual_compact_savings.csv"
    saving_summary_csv = args.summary_root / "stage58_actual_compact_savings_by_bits.csv"
    full_plot = args.summary_root / "stage58_full_mean_compression_rd.png"
    actual_plot = args.summary_root / "stage58_actual_compact_subset_rd.png"
    summary_json = args.summary_root / "stage58_compression_rd_ablation_summary.json"

    write_csv(rd_rows, RD_FIELDS, rd_csv)
    write_csv(mean_rows, MEAN_FIELDS, mean_csv)
    write_csv(codec_summary_rows, CODEC_SUMMARY_FIELDS, codec_summary_csv)
    write_csv(saving_rows, SAVING_FIELDS, saving_csv)
    write_csv(saving_summary_rows, [
        "baseline_variant", "compact_variant", "bits", "rows", "mean_rate_saving_percent",
        "mean_baseline_rate_mib_per_frame", "mean_compact_rate_mib_per_frame", "mean_all_psnr",
    ], saving_summary_csv)
    plot_full_mean(mean_rows, full_plot)
    plot_actual_subset(mean_rows, actual_plot)

    summary = {
        "stage": 58,
        "mode": "compression RD ablation from existing all-frame PSNR rows",
        "stage51_csv": str(args.stage51_csv),
        "stage57_csv": str(args.stage57_csv),
        "rd_csv": str(rd_csv),
        "mean_csv": str(mean_csv),
        "codec_summary_csv": str(codec_summary_csv),
        "saving_csv": str(saving_csv),
        "saving_summary_csv": str(saving_summary_csv),
        "plots": [str(full_plot), str(actual_plot)],
        "stage51_rows": len(stage51_rows),
        "stage57_rows": len(stage57_rows),
        "rd_rows": len(rd_rows),
        "mean_rows": len(mean_rows),
        "codec_summary_rows": codec_summary_rows,
        "actual_compact_savings_by_bits": saving_summary_rows,
        "notes": (
            "Full rows use Stage51 all-frame PSNR and Stage50 legacy/theoretical rates. "
            "Actual Stage57 compact raw/zlib rates are available only for the Stage57 formal subset."
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_json),
        "rd_csv": str(rd_csv),
        "mean_csv": str(mean_csv),
        "plots": summary["plots"],
        "rd_rows": len(rd_rows),
        "actual_compact_savings_by_bits": saving_summary_rows,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
