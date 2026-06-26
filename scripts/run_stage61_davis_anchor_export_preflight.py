import argparse
import csv
import json
import math
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAVIS_ROOT = Path("/mnt/hdd2tC/tmp/opencode/datasets/DAVIS")
DEFAULT_HEAVY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage61_davis_anchor_export")
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage61_davis_anchor_export_preflight"

PAIR_ANCHOR_MIB_STAGE6 = 1.828125

SEQUENCE_FIELDS = [
    "dataset",
    "split",
    "sequence",
    "frame_count",
    "depth_count",
    "mask_count",
    "ready_for_anchor_export",
    "image_dir",
    "depth_dir",
    "mask_dir",
]

PLAN_FIELDS = [
    "dataset",
    "split",
    "sequence",
    "gap",
    "frame_count",
    "pair_count",
    "unique_anchor_count",
    "estimated_pair_pt_mib",
    "estimated_dedup_static_anchor_mib",
    "ready_for_anchor_export",
]

TOTAL_FIELDS = [
    "scope",
    "gap",
    "sequence_count",
    "frame_count",
    "pair_count",
    "unique_anchor_count",
    "estimated_pair_pt_mib",
    "estimated_dedup_static_anchor_mib",
]


def bool_text(value):
    return "true" if value else "false"


def read_split(path):
    path = Path(path)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sorted_files(path, pattern):
    path = Path(path)
    if not path.exists():
        return []
    return sorted(path.glob(pattern), key=lambda p: p.name)


def inspect_davis_sequences(root):
    root = Path(root)
    image_root = root / "JPEGImages/Full-Resolution"
    depth_root = root / "depthImages/Full-Resolution"
    mask_root = root / "Annotations_unsupervised/Full-Resolution"
    split_names = []
    for split in ("train", "val"):
        split_names.extend((split, name) for name in read_split(root / "ImageSets/2017" / f"{split}.txt"))
    if not split_names and image_root.exists():
        split_names = [("unspecified", path.name) for path in sorted(image_root.iterdir()) if path.is_dir()]

    rows = []
    for split, sequence in split_names:
        image_dir = image_root / sequence
        depth_dir = depth_root / sequence
        mask_dir = mask_root / sequence
        frames = sorted_files(image_dir, "*.jpg")
        depths = sorted_files(depth_dir, "*_pred.png")
        masks = sorted_files(mask_dir, "*.png")
        ready = bool(frames) and len(frames) == len(depths) and len(frames) == len(masks)
        rows.append({
            "dataset": "DAVIS",
            "split": split,
            "sequence": sequence,
            "frame_count": len(frames),
            "depth_count": len(depths),
            "mask_count": len(masks),
            "ready_for_anchor_export": bool_text(ready),
            "image_dir": str(image_dir),
            "depth_dir": str(depth_dir),
            "mask_dir": str(mask_dir),
        })
    return rows


def selected_count_for_gap(frame_count, gap):
    if frame_count <= 0:
        return 0
    selected = list(range(0, frame_count, gap))
    if selected[-1] != frame_count - 1:
        selected.append(frame_count - 1)
    return len(selected)


def estimate_plan(sequence_rows, gaps, pair_anchor_mib):
    per_anchor_mib = pair_anchor_mib / 2.0
    rows = []
    for sequence_row in sequence_rows:
        frame_count = int(sequence_row["frame_count"])
        ready = sequence_row["ready_for_anchor_export"] == "true"
        for gap in gaps:
            unique_anchor_count = selected_count_for_gap(frame_count, gap) if ready else 0
            pair_count = max(unique_anchor_count - 1, 0)
            rows.append({
                "dataset": sequence_row["dataset"],
                "split": sequence_row["split"],
                "sequence": sequence_row["sequence"],
                "gap": gap,
                "frame_count": frame_count,
                "pair_count": pair_count,
                "unique_anchor_count": unique_anchor_count,
                "estimated_pair_pt_mib": pair_count * pair_anchor_mib,
                "estimated_dedup_static_anchor_mib": unique_anchor_count * per_anchor_mib,
                "ready_for_anchor_export": sequence_row["ready_for_anchor_export"],
            })
    return rows


def aggregate_totals(plan_rows):
    totals = []
    scopes = [("all", plan_rows)]
    for split in sorted({row["split"] for row in plan_rows}):
        scopes.append((split, [row for row in plan_rows if row["split"] == split]))
    for scope, rows in scopes:
        for gap in sorted({int(row["gap"]) for row in rows}):
            gap_rows = [row for row in rows if int(row["gap"]) == gap]
            totals.append({
                "scope": scope,
                "gap": gap,
                "sequence_count": len(gap_rows),
                "frame_count": sum(int(row["frame_count"]) for row in gap_rows),
                "pair_count": sum(int(row["pair_count"]) for row in gap_rows),
                "unique_anchor_count": sum(int(row["unique_anchor_count"]) for row in gap_rows),
                "estimated_pair_pt_mib": sum(float(row["estimated_pair_pt_mib"]) for row in gap_rows),
                "estimated_dedup_static_anchor_mib": sum(float(row["estimated_dedup_static_anchor_mib"]) for row in gap_rows),
            })
    totals.append({
        "scope": "all_gaps_total",
        "gap": "all",
        "sequence_count": len({(row["split"], row["sequence"]) for row in plan_rows}),
        "frame_count": sum(int(row["frame_count"]) for row in plan_rows if int(row["gap"]) == min(int(r["gap"]) for r in plan_rows)),
        "pair_count": sum(int(row["pair_count"]) for row in plan_rows),
        "unique_anchor_count": sum(int(row["unique_anchor_count"]) for row in plan_rows),
        "estimated_pair_pt_mib": sum(float(row["estimated_pair_pt_mib"]) for row in plan_rows),
        "estimated_dedup_static_anchor_mib": sum(float(row["estimated_dedup_static_anchor_mib"]) for row in plan_rows),
    })
    return totals


def existing_parent(path):
    path = Path(path)
    for candidate in [path, *path.parents]:
        if candidate.exists():
            return candidate
    return Path("/")


def disk_free_mib(path):
    usage = shutil.disk_usage(existing_parent(path))
    return usage.free / (1024.0 * 1024.0)


def write_csv(rows, fields, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summary, totals):
    all_gaps = next(row for row in totals if row["scope"] == "all_gaps_total")
    lines = [
        "# Stage61 DAVIS Anchor Export Preflight",
        "",
        "## Summary",
        "",
        f"- DAVIS root: `{summary['davis_root']}`.",
        f"- Ready sequences: `{summary['ready_sequence_count']}` / `{summary['sequence_count']}`.",
        f"- Frames: `{summary['frame_count']}`.",
        f"- Gaps: `{summary['gaps']}`.",
        f"- Free space at heavy root mount: `{summary['free_mib']:.2f}` MiB.",
        f"- Required reserve: `{summary['reserve_mib']:.2f}` MiB.",
        f"- Estimated pair-pt output for all gaps: `{all_gaps['estimated_pair_pt_mib']:.2f}` MiB.",
        f"- Full all-gap export safe: `{summary['safe_to_full_export']}`.",
        "",
        "## Totals",
        "",
        "| scope | gap | sequences | frames | pairs | pair-pt MiB | dedup anchor MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in totals:
        lines.append(
            f"| {row['scope']} | {row['gap']} | {row['sequence_count']} | {row['frame_count']} | "
            f"{row['pair_count']} | {float(row['estimated_pair_pt_mib']):.2f} | "
            f"{float(row['estimated_dedup_static_anchor_mib']):.2f} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        summary["decision"],
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis_root", type=Path, default=DEFAULT_DAVIS_ROOT)
    parser.add_argument("--heavy_root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--gaps", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--pair_anchor_mib", type=float, default=PAIR_ANCHOR_MIB_STAGE6)
    parser.add_argument("--reserve_mib", type=float, default=2048.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    gaps = sorted(set(args.gaps))
    if any(gap <= 0 for gap in gaps):
        raise ValueError(f"Gaps must be positive, got {gaps}")

    sequence_rows = inspect_davis_sequences(args.davis_root)
    plan_rows = estimate_plan(sequence_rows, gaps, args.pair_anchor_mib)
    totals = aggregate_totals(plan_rows)
    all_gaps = next(row for row in totals if row["scope"] == "all_gaps_total")
    free_mib = disk_free_mib(args.heavy_root)
    needed_mib = float(all_gaps["estimated_pair_pt_mib"]) + args.reserve_mib
    safe = free_mib >= needed_mib
    ready_sequences = [row for row in sequence_rows if row["ready_for_anchor_export"] == "true"]
    decision = (
        "Full DAVIS all-gap anchor export is safe to launch with the requested reserve."
        if safe else
        "Do not launch full DAVIS all-gap anchor export on this mount yet; free space is below the estimated output plus reserve."
    )

    sequence_csv = args.summary_root / "stage61_davis_sequences.csv"
    plan_csv = args.summary_root / "stage61_davis_anchor_export_plan.csv"
    totals_csv = args.summary_root / "stage61_davis_anchor_export_totals.csv"
    summary_json = args.summary_root / "stage61_davis_anchor_export_preflight_summary.json"
    report_md = args.summary_root / "stage61_davis_anchor_export_preflight_report.md"

    write_csv(sequence_rows, SEQUENCE_FIELDS, sequence_csv)
    write_csv(plan_rows, PLAN_FIELDS, plan_csv)
    write_csv(totals, TOTAL_FIELDS, totals_csv)
    summary = {
        "stage": 61,
        "mode": "DAVIS large-scale anchor export preflight",
        "davis_root": str(args.davis_root),
        "heavy_root": str(args.heavy_root),
        "summary_root": str(args.summary_root),
        "gaps": gaps,
        "pair_anchor_mib": args.pair_anchor_mib,
        "per_anchor_mib": args.pair_anchor_mib / 2.0,
        "reserve_mib": args.reserve_mib,
        "free_mib": free_mib,
        "needed_mib_all_gaps_plus_reserve": needed_mib,
        "safe_to_full_export": bool_text(safe),
        "sequence_count": len(sequence_rows),
        "ready_sequence_count": len(ready_sequences),
        "frame_count": sum(int(row["frame_count"]) for row in sequence_rows),
        "estimated_all_gaps_pair_pt_mib": float(all_gaps["estimated_pair_pt_mib"]),
        "estimated_all_gaps_dedup_static_anchor_mib": float(all_gaps["estimated_dedup_static_anchor_mib"]),
        "sequence_csv": str(sequence_csv),
        "plan_csv": str(plan_csv),
        "totals_csv": str(totals_csv),
        "report_md": str(report_md),
        "decision": decision,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(report_md, summary, totals)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "safe_to_full_export": summary["safe_to_full_export"],
        "free_mib": free_mib,
        "needed_mib_all_gaps_plus_reserve": needed_mib,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
