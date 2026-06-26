import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage55_large_scale_data_preflight"
DEFAULT_STAGE33_MANIFEST = REPO_ROOT / "experiments/stage33_dense_gap1_anchor_export/stage33_dense_gap1_anchor_manifest.csv"
DEFAULT_STREAMSPLAT_ROOT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat")

DEFAULT_DAVIS_ROOTS = [
    "/mnt/hdd2tC/tmp/opencode/datasets/DAVIS",
    "/mnt/hdd2tC/tmp/opencode/datasets/davis",
    "/mnt/hdd2tC/tmp/opencode/DAVIS",
    "/mnt/hdd2tC/datasets/DAVIS",
    "/mnt/hdd2tC/datasets/davis",
    "/mnt/hdd2tC/haocheng/datasets/DAVIS",
    "/mnt/hdd2tC/haocheng/datasets/davis",
]

DEFAULT_VOS_ROOTS = [
    "/mnt/hdd2tC/tmp/opencode/datasets/YouTube-VOS",
    "/mnt/hdd2tC/tmp/opencode/datasets/youtube_vos",
    "/mnt/hdd2tC/tmp/opencode/YouTube-VOS",
    "/mnt/hdd2tC/datasets/YouTube-VOS",
    "/mnt/hdd2tC/haocheng/datasets/YouTube-VOS",
]

DEFAULT_RE10K_ROOTS = [
    "/mnt/hdd2tC/tmp/opencode/datasets/RealEstate10K",
    "/mnt/hdd2tC/tmp/opencode/datasets/re10k",
    "/mnt/hdd2tC/datasets/RealEstate10K",
    "/mnt/hdd2tC/haocheng/datasets/RealEstate10K",
]

DEFAULT_CO3D_ROOTS = [
    "/mnt/hdd2tC/tmp/opencode/datasets/CO3D",
    "/mnt/hdd2tC/tmp/opencode/datasets/co3d",
    "/mnt/hdd2tC/datasets/CO3D",
    "/mnt/hdd2tC/haocheng/datasets/CO3D",
]

ROOT_FIELDS = [
    "dataset",
    "root",
    "exists",
    "provider",
    "provider_layout_ready",
    "ready_for_depth_preprocess",
    "ready_for_anchor_export",
    "sequence_count",
    "sampled_sequence_count",
    "checked_frame_count",
    "checked_depth_pred_count",
    "checked_mask_count",
    "scan_limited",
    "role_for_mono_dfcgs",
    "blocking_reason",
    "notes",
]

SEQUENCE_FIELDS = [
    "dataset",
    "root",
    "split",
    "sequence",
    "frame_count",
    "depth_pred_count",
    "mask_count",
    "provider_layout_ready",
    "ready_for_depth_preprocess",
    "ready_for_anchor_export",
    "image_dir",
    "depth_dir",
    "mask_dir",
]

ASSET_FIELDS = [
    "asset",
    "sample",
    "row_count",
    "unique_anchor_frames",
    "min_frame",
    "max_frame",
    "external_path_exists_count",
    "notes",
]

PROTOCOL_FIELDS = [
    "protocol_item",
    "dataset",
    "input_scope",
    "provider_or_source",
    "local_status",
    "mono_dfcgs_role",
    "rate_quality_caveat",
    "next_action",
]


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, fields, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_split(path):
    path = Path(path)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sorted_dirs(path):
    path = Path(path)
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir()], key=lambda p: p.name)


def count_files(path, pattern):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def bool_text(value):
    return "true" if value else "false"


def limited(items, limit):
    if limit <= 0:
        return list(items), False
    items = list(items)
    return items[:limit], len(items) > limit


def davis_sequence_row(root, split, sequence):
    image_dir = root / "JPEGImages/Full-Resolution" / sequence
    depth_dir = root / "depthImages/Full-Resolution" / sequence
    mask_dir = root / "Annotations_unsupervised/Full-Resolution" / sequence
    frame_count = count_files(image_dir, "*.jpg")
    depth_pred_count = count_files(depth_dir, "*_pred.png")
    mask_count = count_files(mask_dir, "*.png")
    provider_layout_ready = image_dir.exists() and mask_dir.exists()
    ready_for_depth_preprocess = frame_count > 0 and mask_count >= frame_count
    ready_for_anchor_export = ready_for_depth_preprocess and depth_pred_count >= frame_count
    return {
        "dataset": "DAVIS",
        "root": str(root),
        "split": split,
        "sequence": sequence,
        "frame_count": frame_count,
        "depth_pred_count": depth_pred_count,
        "mask_count": mask_count,
        "provider_layout_ready": bool_text(provider_layout_ready),
        "ready_for_depth_preprocess": bool_text(ready_for_depth_preprocess),
        "ready_for_anchor_export": bool_text(ready_for_anchor_export),
        "image_dir": str(image_dir),
        "depth_dir": str(depth_dir),
        "mask_dir": str(mask_dir),
    }


def inspect_davis(root, max_sequences):
    root = Path(root)
    train_names = read_split(root / "ImageSets/2017/train.txt")
    val_names = read_split(root / "ImageSets/2017/val.txt")
    strict_image_root = root / "JPEGImages/Full-Resolution"
    strict_mask_root = root / "Annotations_unsupervised/Full-Resolution"
    compatible_image_root = strict_image_root if strict_image_root.exists() else root / "JPEGImages/480p"
    names_with_split = [("train", name) for name in train_names] + [("val", name) for name in val_names]
    if not names_with_split and compatible_image_root.exists():
        names_with_split = [("unspecified", p.name) for p in sorted_dirs(compatible_image_root)]
    sampled, scan_limited = limited(names_with_split, max_sequences)
    sequence_rows = [davis_sequence_row(root, split, name) for split, name in sampled]
    provider_layout_ready = bool(train_names or val_names) and strict_image_root.exists() and strict_mask_root.exists()
    ready_for_depth_preprocess = any(row["ready_for_depth_preprocess"] == "true" for row in sequence_rows)
    ready_for_anchor_export = any(row["ready_for_anchor_export"] == "true" for row in sequence_rows)
    blocking = []
    if not root.exists():
        blocking.append("root missing")
    if root.exists() and not (train_names or val_names):
        blocking.append("ImageSets/2017 split txt missing or empty")
    if root.exists() and not strict_image_root.exists():
        blocking.append("JPEGImages/Full-Resolution missing for official provider")
    if root.exists() and not strict_mask_root.exists():
        blocking.append("Annotations_unsupervised/Full-Resolution missing for official provider")
    if provider_layout_ready and not ready_for_anchor_export:
        blocking.append("predicted depth *_pred.png missing or incomplete")
    return {
        "root": {
            "dataset": "DAVIS",
            "root": str(root),
            "exists": bool_text(root.exists()),
            "provider": "StreamSplat provider_davis",
            "provider_layout_ready": bool_text(provider_layout_ready),
            "ready_for_depth_preprocess": bool_text(ready_for_depth_preprocess),
            "ready_for_anchor_export": bool_text(ready_for_anchor_export),
            "sequence_count": len(names_with_split),
            "sampled_sequence_count": len(sequence_rows),
            "checked_frame_count": sum(row["frame_count"] for row in sequence_rows),
            "checked_depth_pred_count": sum(row["depth_pred_count"] for row in sequence_rows),
            "checked_mask_count": sum(row["mask_count"] for row in sequence_rows),
            "scan_limited": bool_text(scan_limited),
            "role_for_mono_dfcgs": "single-view video expansion candidate",
            "blocking_reason": "; ".join(blocking),
            "notes": "Official provider requires Full-Resolution JPEGImages, unsupervised masks, split txt, and depthImages/*_pred.png before training/anchor export.",
        },
        "sequences": sequence_rows,
    }


def vos_sequence_row(root, split, sequence):
    base = root / ("train" if split == "train" else "valid")
    image_dir = base / "JPEGImages" / sequence
    depth_dir = base / "depthImages" / sequence
    mask_dir = base / "Annotations" / sequence
    frame_count = count_files(image_dir, "*.jpg")
    depth_pred_count = count_files(depth_dir, "*_pred.png")
    mask_count = count_files(mask_dir, "*.png") if split == "train" else frame_count
    provider_layout_ready = image_dir.exists() and (split != "train" or mask_dir.exists())
    ready_for_depth_preprocess = frame_count > 0 and (split != "train" or mask_count >= frame_count)
    ready_for_anchor_export = ready_for_depth_preprocess and depth_pred_count >= frame_count
    return {
        "dataset": "YouTube-VOS",
        "root": str(root),
        "split": split,
        "sequence": sequence,
        "frame_count": frame_count,
        "depth_pred_count": depth_pred_count,
        "mask_count": mask_count,
        "provider_layout_ready": bool_text(provider_layout_ready),
        "ready_for_depth_preprocess": bool_text(ready_for_depth_preprocess),
        "ready_for_anchor_export": bool_text(ready_for_anchor_export),
        "image_dir": str(image_dir),
        "depth_dir": str(depth_dir),
        "mask_dir": str(mask_dir) if split == "train" else "",
    }


def inspect_vos(root, max_sequences):
    root = Path(root)
    names_with_split = []
    for split, dirname in (("train", "train"), ("valid", "valid")):
        image_root = root / dirname / "JPEGImages"
        names_with_split.extend((split, p.name) for p in sorted_dirs(image_root))
    sampled, scan_limited = limited(names_with_split, max_sequences)
    sequence_rows = [vos_sequence_row(root, split, name) for split, name in sampled]
    train_images = root / "train/JPEGImages"
    valid_images = root / "valid/JPEGImages"
    train_masks = root / "train/Annotations"
    provider_layout_ready = (train_images.exists() and train_masks.exists()) or valid_images.exists()
    ready_for_depth_preprocess = any(row["ready_for_depth_preprocess"] == "true" for row in sequence_rows)
    ready_for_anchor_export = any(row["ready_for_anchor_export"] == "true" for row in sequence_rows)
    blocking = []
    if not root.exists():
        blocking.append("root missing")
    if root.exists() and not train_images.exists() and not valid_images.exists():
        blocking.append("train/valid JPEGImages missing")
    if train_images.exists() and not train_masks.exists():
        blocking.append("train/Annotations missing for training split")
    if provider_layout_ready and not ready_for_anchor_export:
        blocking.append("predicted depth *_pred.png missing or incomplete")
    return {
        "root": {
            "dataset": "YouTube-VOS",
            "root": str(root),
            "exists": bool_text(root.exists()),
            "provider": "StreamSplat provider_vos",
            "provider_layout_ready": bool_text(provider_layout_ready),
            "ready_for_depth_preprocess": bool_text(ready_for_depth_preprocess),
            "ready_for_anchor_export": bool_text(ready_for_anchor_export),
            "sequence_count": len(names_with_split),
            "sampled_sequence_count": len(sequence_rows),
            "checked_frame_count": sum(row["frame_count"] for row in sequence_rows),
            "checked_depth_pred_count": sum(row["depth_pred_count"] for row in sequence_rows),
            "checked_mask_count": sum(row["mask_count"] for row in sequence_rows),
            "scan_limited": bool_text(scan_limited),
            "role_for_mono_dfcgs": "single-view video expansion candidate",
            "blocking_reason": "; ".join(blocking),
            "notes": "Official provider uses train/valid JPEGImages, train Annotations, and depthImages/*_pred.png.",
        },
        "sequences": sequence_rows,
    }


def inspect_re10k(root):
    root = Path(root)
    split_rows = []
    for split in ("train", "test"):
        split_root = root / split
        chunk_count = count_files(split_root, "*.torch")
        depth_count = count_files(split_root, "depth_*.pt")
        split_rows.append((split, split_root.exists(), (split_root / "index.json").exists(), chunk_count, depth_count))
    provider_layout_ready = any(exists and index and chunks > 0 for _split, exists, index, chunks, _depths in split_rows)
    ready_for_anchor_export = any(exists and index and chunks > 0 and depths >= chunks for _split, exists, index, chunks, depths in split_rows)
    blocking = []
    if not root.exists():
        blocking.append("root missing")
    if root.exists() and not provider_layout_ready:
        blocking.append("train/test index.json or .torch chunks missing")
    if provider_layout_ready and not ready_for_anchor_export:
        blocking.append("depth_*.pt chunks missing or incomplete")
    return {
        "dataset": "RE10K",
        "root": str(root),
        "exists": bool_text(root.exists()),
        "provider": "StreamSplat provider_re10k_map",
        "provider_layout_ready": bool_text(provider_layout_ready),
        "ready_for_depth_preprocess": bool_text(provider_layout_ready),
        "ready_for_anchor_export": bool_text(ready_for_anchor_export),
        "sequence_count": "",
        "sampled_sequence_count": "",
        "checked_frame_count": "",
        "checked_depth_pred_count": sum(row[4] for row in split_rows),
        "checked_mask_count": "",
        "scan_limited": "false",
        "role_for_mono_dfcgs": "training-only unless converted to single-view video protocol",
        "blocking_reason": "; ".join(blocking),
        "notes": "Provider expects train/test index.json, *.torch chunks, and matching depth_*.pt chunks; do not use multi-view/camera information for final monocular codec claims.",
    }


def inspect_co3d(root):
    root = Path(root)
    categories = sorted_dirs(root)
    set_list_count = 0
    manyview_dev_count = 0
    manyview_test_count = 0
    for category in categories:
        set_list_root = category / "set_lists"
        jsons = list(set_list_root.glob("*.json")) if set_list_root.exists() else []
        set_list_count += len(jsons)
        manyview_dev_count += sum(1 for path in jsons if "manyview_dev" in path.name)
        manyview_test_count += sum(1 for path in jsons if "manyview_test" in path.name)
    provider_layout_ready = bool(categories) and set_list_count > 0
    blocking = []
    if not root.exists():
        blocking.append("root missing")
    if root.exists() and not categories:
        blocking.append("category directories missing")
    if categories and set_list_count == 0:
        blocking.append("category set_lists/*.json missing")
    return {
        "dataset": "CO3D",
        "root": str(root),
        "exists": bool_text(root.exists()),
        "provider": "StreamSplat provider_co3d",
        "provider_layout_ready": bool_text(provider_layout_ready),
        "ready_for_depth_preprocess": bool_text(provider_layout_ready),
        "ready_for_anchor_export": "unknown",
        "sequence_count": len(categories),
        "sampled_sequence_count": "",
        "checked_frame_count": "",
        "checked_depth_pred_count": "",
        "checked_mask_count": "",
        "scan_limited": "false",
        "role_for_mono_dfcgs": "training-only unless converted to single-view video protocol",
        "blocking_reason": "; ".join(blocking),
        "notes": f"Found set_list_count={set_list_count}, manyview_dev={manyview_dev_count}, manyview_test={manyview_test_count}; predicted depths are path-derived as predict_depths/*_pred.png.",
    }


def summarize_current_assets(stage33_manifest):
    rows = read_csv(stage33_manifest)
    by_sample = defaultdict(list)
    for row in rows:
        by_sample[row.get("sample", "")].append(row)
    out = []
    for sample, sample_rows in sorted(by_sample.items()):
        frames = set()
        exists_count = 0
        for row in sample_rows:
            if row.get("left_index") not in {None, ""}:
                frames.add(int(row["left_index"]))
            if row.get("right_index") not in {None, ""}:
                frames.add(int(row["right_index"]))
            if row.get("dataset_item") and Path(row["dataset_item"]).exists():
                exists_count += 1
        out.append({
            "asset": "stage33_dense_gap1_anchor_manifest",
            "sample": sample,
            "row_count": len(sample_rows),
            "unique_anchor_frames": len(frames),
            "min_frame": min(frames) if frames else "",
            "max_frame": max(frames) if frames else "",
            "external_path_exists_count": exists_count,
            "notes": "Current local dense anchors; useful for development but not large-scale dataset expansion.",
        })
    if not rows:
        out.append({
            "asset": "stage33_dense_gap1_anchor_manifest",
            "sample": "",
            "row_count": 0,
            "unique_anchor_frames": 0,
            "min_frame": "",
            "max_frame": "",
            "external_path_exists_count": 0,
            "notes": f"Manifest missing or empty: {stage33_manifest}",
        })
    return out


def build_protocol_matrix(root_rows, asset_rows, streamsplat_root):
    ready_davis = sum(1 for row in root_rows if row["dataset"] == "DAVIS" and row["ready_for_anchor_export"] == "true")
    ready_vos = sum(1 for row in root_rows if row["dataset"] == "YouTube-VOS" and row["ready_for_anchor_export"] == "true")
    provider_files = [
        streamsplat_root / "datasets/provider_davis.py",
        streamsplat_root / "datasets/provider_vos.py",
        streamsplat_root / "datasets/provider_re10k_map.py",
        streamsplat_root / "datasets/provider_co3d.py",
    ]
    providers_present = all(path.exists() for path in provider_files)
    current_samples = len([row for row in asset_rows if row["row_count"]])
    return [
        {
            "protocol_item": "current_local_development_set",
            "dataset": "n3dv/meetroom/driving/robot",
            "input_scope": "single-view video",
            "provider_or_source": "Stage33 dense gap1 anchors",
            "local_status": f"ready for development anchors: {current_samples} samples",
            "mono_dfcgs_role": "debug/RD development only",
            "rate_quality_caveat": "small local sample set, not paper-scale dataset evidence",
            "next_action": "Use for selector/adapter iteration while waiting for larger mounted datasets.",
        },
        {
            "protocol_item": "streamsplat_original_davis",
            "dataset": "DAVIS",
            "input_scope": "single-view video sequences",
            "provider_or_source": "provider_davis.py",
            "local_status": f"ready roots: {ready_davis}",
            "mono_dfcgs_role": "preferred next large-scale single-view expansion",
            "rate_quality_caveat": "requires generated predicted depths and masks; no multiview inputs allowed",
            "next_action": "Mount DAVIS 2017 Full-Resolution layout, run depth preprocessing, then export anchors.",
        },
        {
            "protocol_item": "streamsplat_original_vos",
            "dataset": "YouTube-VOS",
            "input_scope": "single-view video object sequences",
            "provider_or_source": "provider_vos.py",
            "local_status": f"ready roots: {ready_vos}",
            "mono_dfcgs_role": "secondary single-view expansion after DAVIS",
            "rate_quality_caveat": "valid split uses dummy masks in provider; train masks required for train split",
            "next_action": "Mount YouTube-VOS train/valid, generate depthImages/*_pred.png, then export anchors.",
        },
        {
            "protocol_item": "streamsplat_combined_training",
            "dataset": "RE10K/CO3D/DAVIS/VOS",
            "input_scope": "mixed; RE10K/CO3D are not final monocular codec eval by default",
            "provider_or_source": "provider_combined.py",
            "local_status": "provider files present" if providers_present else "provider files missing",
            "mono_dfcgs_role": "possible pretraining source only",
            "rate_quality_caveat": "do not use multiview/camera information in final transmitted bitstream or codec claims",
            "next_action": "Use only after defining a single-view extraction/evaluation protocol.",
        },
    ]


def write_report(summary, protocol_rows, path):
    lines = [
        "# Stage55 Large-Scale Data Preflight",
        "",
        "## Status",
        "",
        f"- Root candidates checked: {summary['root_candidate_count']}",
        f"- Provider-layout-ready roots: {summary['provider_layout_ready_root_count']}",
        f"- Anchor-export-ready roots: {summary['anchor_export_ready_root_count']}",
        f"- DAVIS/YouTube-VOS sampled sequence rows: {summary['sequence_row_count']}",
        f"- Current local anchor samples: {summary['current_anchor_sample_count']}",
        "",
        "## Protocol Matrix",
        "",
        "| Protocol | Dataset | Local status | Mono-DFCGS role | Next action |",
        "|---|---|---|---|---|",
    ]
    for row in protocol_rows:
        lines.append(
            f"| {row['protocol_item']} | {row['dataset']} | {row['local_status']} | {row['mono_dfcgs_role']} | {row['next_action']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Stage55 is a read-only preflight. It does not download data, preprocess depth, export anchors, or train models.",
        "- DAVIS and YouTube-VOS are the cleanest next expansion targets because they are single-view video sequence datasets in the StreamSplat codebase.",
        "- RE10K and CO3D can be useful for pretraining only after a single-view extraction protocol is specified; they should not be used to support final monocular codec claims with multiview information.",
        "- A root is anchor-export-ready only when provider layout exists and predicted depth files are already present in the provider-derived `depthImages/*_pred.png` locations.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--streamsplat_root", type=Path, default=DEFAULT_STREAMSPLAT_ROOT)
    parser.add_argument("--stage33_manifest", type=Path, default=DEFAULT_STAGE33_MANIFEST)
    parser.add_argument("--davis_roots", nargs="*", default=DEFAULT_DAVIS_ROOTS)
    parser.add_argument("--vos_roots", nargs="*", default=DEFAULT_VOS_ROOTS)
    parser.add_argument("--re10k_roots", nargs="*", default=DEFAULT_RE10K_ROOTS)
    parser.add_argument("--co3d_roots", nargs="*", default=DEFAULT_CO3D_ROOTS)
    parser.add_argument("--max_sequences_per_root", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)

    root_rows = []
    sequence_rows = []
    for root in args.davis_roots:
        info = inspect_davis(root, args.max_sequences_per_root)
        root_rows.append(info["root"])
        sequence_rows.extend(info["sequences"])
    for root in args.vos_roots:
        info = inspect_vos(root, args.max_sequences_per_root)
        root_rows.append(info["root"])
        sequence_rows.extend(info["sequences"])
    root_rows.extend(inspect_re10k(root) for root in args.re10k_roots)
    root_rows.extend(inspect_co3d(root) for root in args.co3d_roots)

    asset_rows = summarize_current_assets(args.stage33_manifest)
    protocol_rows = build_protocol_matrix(root_rows, asset_rows, args.streamsplat_root)

    root_csv = args.summary_root / "stage55_root_preflight.csv"
    sequence_csv = args.summary_root / "stage55_sequence_preflight_sample.csv"
    asset_csv = args.summary_root / "stage55_current_anchor_assets.csv"
    protocol_csv = args.summary_root / "stage55_protocol_matrix.csv"
    summary_json = args.summary_root / "stage55_large_scale_data_preflight_summary.json"
    report_md = args.summary_root / "stage55_large_scale_data_preflight_report.md"

    write_csv(root_rows, ROOT_FIELDS, root_csv)
    write_csv(sequence_rows, SEQUENCE_FIELDS, sequence_csv)
    write_csv(asset_rows, ASSET_FIELDS, asset_csv)
    write_csv(protocol_rows, PROTOCOL_FIELDS, protocol_csv)

    provider_layout_ready = [row for row in root_rows if row["provider_layout_ready"] == "true"]
    anchor_export_ready = [row for row in root_rows if row["ready_for_anchor_export"] == "true"]
    current_anchor_samples = [row for row in asset_rows if row["row_count"]]
    summary = {
        "stage": 55,
        "mode": "large-scale dataset expansion preflight",
        "root_candidate_count": len(root_rows),
        "provider_layout_ready_root_count": len(provider_layout_ready),
        "anchor_export_ready_root_count": len(anchor_export_ready),
        "sequence_row_count": len(sequence_rows),
        "current_anchor_sample_count": len(current_anchor_samples),
        "root_csv": str(root_csv),
        "sequence_csv": str(sequence_csv),
        "asset_csv": str(asset_csv),
        "protocol_csv": str(protocol_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "streamsplat_root": str(args.streamsplat_root),
        "stage33_manifest": str(args.stage33_manifest),
        "notes": [
            "This preflight is read-only and does not download or preprocess datasets.",
            "DAVIS/YouTube-VOS are single-view video expansion targets for Mono-DFCGS.",
            "RE10K/CO3D should be treated as possible pretraining sources only unless converted to a single-view protocol.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, protocol_rows, report_md)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
