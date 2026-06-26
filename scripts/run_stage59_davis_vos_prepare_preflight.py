import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage59_davis_vos_prepare_preflight"
DEFAULT_STREAMSPLAT_ROOT = Path("/mnt/hdd2tC/tmp/opencode/StreamSplat")

DEFAULT_DAVIS_ROOTS = [
    Path("/mnt/hdd2tC/tmp/opencode/datasets/DAVIS"),
    Path("/mnt/hdd2tC/tmp/opencode/datasets/davis"),
    Path("/mnt/hdd2tC/tmp/opencode/DAVIS"),
    Path("/mnt/hdd2tC/datasets/DAVIS"),
    Path("/mnt/hdd2tC/haocheng/datasets/DAVIS"),
]

DEFAULT_VOS_ROOTS = [
    Path("/mnt/hdd2tC/tmp/opencode/datasets/YouTube-VOS"),
    Path("/mnt/hdd2tC/tmp/opencode/datasets/youtube_vos"),
    Path("/mnt/hdd2tC/tmp/opencode/YouTube-VOS"),
    Path("/mnt/hdd2tC/datasets/YouTube-VOS"),
    Path("/mnt/hdd2tC/haocheng/datasets/YouTube-VOS"),
]

ROOT_FIELDS = [
    "dataset", "root", "exists", "provider_layout_ready", "ready_for_depth_preprocess",
    "ready_for_anchor_export", "sequence_count", "sampled_sequence_count", "sampled_frame_count",
    "sampled_mask_count", "sampled_depth_pred_count", "missing_items", "next_action",
]

LAYOUT_FIELDS = [
    "dataset", "component", "expected_path", "required_for", "exists", "notes",
]

CHECKLIST_FIELDS = [
    "step", "dataset", "action", "url", "target_path", "blocking", "notes",
]

PROVIDER_FIELDS = [
    "dataset", "provider_file", "exists", "provider_class", "image_root_rule", "mask_root_rule", "depth_path_rule",
]


def bool_text(value):
    return "true" if bool(value) else "false"


def count_files(path, pattern):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def sorted_dirs(path):
    path = Path(path)
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir()], key=lambda p: p.name)


def read_split(path):
    path = Path(path)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(rows, fields, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sample_items(items, limit):
    items = list(items)
    if limit <= 0:
        return items
    return items[:limit]


def inspect_davis_root(root, max_sequences):
    root = Path(root)
    train_txt = root / "ImageSets/2017/train.txt"
    val_txt = root / "ImageSets/2017/val.txt"
    image_root = root / "JPEGImages/Full-Resolution"
    mask_root = root / "Annotations_unsupervised/Full-Resolution"
    depth_root = root / "depthImages/Full-Resolution"
    train_names = read_split(train_txt)
    val_names = read_split(val_txt)
    names = [("train", name) for name in train_names] + [("val", name) for name in val_names]
    if not names and image_root.exists():
        names = [("unspecified", p.name) for p in sorted_dirs(image_root)]
    sampled = sample_items(names, max_sequences)
    frame_count = 0
    mask_count = 0
    depth_count = 0
    for _split, sequence in sampled:
        frame_count += count_files(image_root / sequence, "*.jpg")
        mask_count += count_files(mask_root / sequence, "*.png")
        depth_count += count_files(depth_root / sequence, "*_pred.png")

    missing = []
    if not root.exists():
        missing.append("root")
    if root.exists() and not train_txt.exists():
        missing.append("ImageSets/2017/train.txt")
    if root.exists() and not val_txt.exists():
        missing.append("ImageSets/2017/val.txt")
    if root.exists() and not image_root.exists():
        missing.append("JPEGImages/Full-Resolution")
    if root.exists() and not mask_root.exists():
        missing.append("Annotations_unsupervised/Full-Resolution")
    if root.exists() and image_root.exists() and not names:
        missing.append("sequence dirs or split entries")
    provider_layout_ready = root.exists() and train_txt.exists() and val_txt.exists() and image_root.exists() and mask_root.exists() and bool(names)
    ready_for_depth_preprocess = provider_layout_ready and frame_count > 0 and mask_count >= frame_count
    ready_for_anchor_export = ready_for_depth_preprocess and depth_count >= frame_count
    if provider_layout_ready and not ready_for_anchor_export:
        missing.append("depthImages/Full-Resolution/<sequence>/*_pred.png")
    next_action = "ready for Stage60 depth preprocess" if ready_for_depth_preprocess and not ready_for_anchor_export else "ready for Stage61 anchor export" if ready_for_anchor_export else "download or mount missing DAVIS files"
    return {
        "dataset": "DAVIS",
        "root": str(root),
        "exists": bool_text(root.exists()),
        "provider_layout_ready": bool_text(provider_layout_ready),
        "ready_for_depth_preprocess": bool_text(ready_for_depth_preprocess),
        "ready_for_anchor_export": bool_text(ready_for_anchor_export),
        "sequence_count": len(names),
        "sampled_sequence_count": len(sampled),
        "sampled_frame_count": frame_count,
        "sampled_mask_count": mask_count,
        "sampled_depth_pred_count": depth_count,
        "missing_items": "; ".join(missing),
        "next_action": next_action,
    }


def inspect_vos_root(root, max_sequences):
    root = Path(root)
    train_images = root / "train/JPEGImages"
    train_masks = root / "train/Annotations"
    train_depth = root / "train/depthImages"
    valid_images = root / "valid/JPEGImages"
    valid_depth = root / "valid/depthImages"
    names = []
    names.extend(("train", p.name) for p in sorted_dirs(train_images))
    names.extend(("valid", p.name) for p in sorted_dirs(valid_images))
    sampled = sample_items(names, max_sequences)
    frame_count = 0
    mask_count = 0
    depth_count = 0
    for split, sequence in sampled:
        if split == "train":
            frame_count += count_files(train_images / sequence, "*.jpg")
            mask_count += count_files(train_masks / sequence, "*.png")
            depth_count += count_files(train_depth / sequence, "*_pred.png")
        else:
            frames = count_files(valid_images / sequence, "*.jpg")
            frame_count += frames
            mask_count += frames
            depth_count += count_files(valid_depth / sequence, "*_pred.png")

    missing = []
    if not root.exists():
        missing.append("root")
    if root.exists() and not train_images.exists():
        missing.append("train/JPEGImages")
    if root.exists() and not train_masks.exists():
        missing.append("train/Annotations")
    if root.exists() and not valid_images.exists():
        missing.append("valid/JPEGImages")
    if root.exists() and train_images.exists() and valid_images.exists() and not names:
        missing.append("sequence dirs")
    provider_layout_ready = root.exists() and train_images.exists() and train_masks.exists() and valid_images.exists() and bool(names)
    ready_for_depth_preprocess = provider_layout_ready and frame_count > 0 and mask_count >= frame_count
    ready_for_anchor_export = ready_for_depth_preprocess and depth_count >= frame_count
    if provider_layout_ready and not ready_for_anchor_export:
        missing.append("train/valid/depthImages/<sequence>/*_pred.png")
    next_action = "ready for Stage60 depth preprocess" if ready_for_depth_preprocess and not ready_for_anchor_export else "ready for Stage61 anchor export" if ready_for_anchor_export else "download or mount missing YouTube-VOS files"
    return {
        "dataset": "YouTube-VOS",
        "root": str(root),
        "exists": bool_text(root.exists()),
        "provider_layout_ready": bool_text(provider_layout_ready),
        "ready_for_depth_preprocess": bool_text(ready_for_depth_preprocess),
        "ready_for_anchor_export": bool_text(ready_for_anchor_export),
        "sequence_count": len(names),
        "sampled_sequence_count": len(sampled),
        "sampled_frame_count": frame_count,
        "sampled_mask_count": mask_count,
        "sampled_depth_pred_count": depth_count,
        "missing_items": "; ".join(missing),
        "next_action": next_action,
    }


def layout_rows(davis_root, vos_root):
    rows = []
    davis_specs = [
        ("train split", "ImageSets/2017/train.txt", "provider init"),
        ("val split", "ImageSets/2017/val.txt", "provider init"),
        ("frames", "JPEGImages/Full-Resolution/<sequence>/*.jpg", "depth preprocess and training"),
        ("masks", "Annotations_unsupervised/Full-Resolution/<sequence>/*.png", "training supervision"),
        ("predicted depth", "depthImages/Full-Resolution/<sequence>/*_pred.png", "training and anchor export"),
    ]
    for component, rel_path, required_for in davis_specs:
        check_path = davis_root / rel_path.split("/<sequence>")[0].replace("*.jpg", "").replace("*.png", "")
        rows.append({
            "dataset": "DAVIS",
            "component": component,
            "expected_path": str(davis_root / rel_path),
            "required_for": required_for,
            "exists": bool_text(check_path.exists()),
            "notes": "Expected by StreamSplat provider_davis.py",
        })
    vos_specs = [
        ("train frames", "train/JPEGImages/<sequence>/*.jpg", "depth preprocess and training"),
        ("train masks", "train/Annotations/<sequence>/*.png", "training supervision"),
        ("train predicted depth", "train/depthImages/<sequence>/*_pred.png", "training and anchor export"),
        ("valid frames", "valid/JPEGImages/<sequence>/*.jpg", "validation depth preprocess"),
        ("valid predicted depth", "valid/depthImages/<sequence>/*_pred.png", "validation and anchor export"),
    ]
    for component, rel_path, required_for in vos_specs:
        check_path = vos_root / rel_path.split("/<sequence>")[0].replace("*.jpg", "").replace("*.png", "")
        rows.append({
            "dataset": "YouTube-VOS",
            "component": component,
            "expected_path": str(vos_root / rel_path),
            "required_for": required_for,
            "exists": bool_text(check_path.exists()),
            "notes": "Expected by StreamSplat provider_vos.py",
        })
    return rows


def provider_rows(streamsplat_root):
    davis_provider = streamsplat_root / "datasets/provider_davis.py"
    vos_provider = streamsplat_root / "datasets/provider_vos.py"
    return [
        {
            "dataset": "DAVIS",
            "provider_file": str(davis_provider),
            "exists": bool_text(davis_provider.exists()),
            "provider_class": "DAVISDataset",
            "image_root_rule": "<root>/JPEGImages/Full-Resolution",
            "mask_root_rule": "<root>/Annotations_unsupervised/Full-Resolution",
            "depth_path_rule": "image_path.replace('JPEGImages', 'depthImages') + '_pred.png'",
        },
        {
            "dataset": "YouTube-VOS",
            "provider_file": str(vos_provider),
            "exists": bool_text(vos_provider.exists()),
            "provider_class": "VOSDataset",
            "image_root_rule": "<root>/train_or_valid/JPEGImages",
            "mask_root_rule": "<root>/train/Annotations; valid uses all-one masks",
            "depth_path_rule": "image_path.replace('JPEGImages', 'depthImages') + '_pred.png'",
        },
    ]


def checklist_rows(davis_root, vos_root):
    return [
        {
            "step": 1,
            "dataset": "DAVIS",
            "action": "Download or mount DAVIS 2017 train/val images and unsupervised annotations",
            "url": "https://davischallenge.org/",
            "target_path": str(davis_root),
            "blocking": "yes_if_root_missing",
            "notes": "Use official DAVIS terms. StreamSplat provider expects Full-Resolution layout.",
        },
        {
            "step": 2,
            "dataset": "DAVIS",
            "action": "Verify ImageSets/2017 train.txt and val.txt plus JPEGImages/Full-Resolution and Annotations_unsupervised/Full-Resolution",
            "url": "https://davischallenge.org/",
            "target_path": str(davis_root),
            "blocking": "yes_if_missing",
            "notes": "Stage60 depth preprocessing can start only after frames are present.",
        },
        {
            "step": 3,
            "dataset": "DAVIS",
            "action": "Generate depthImages/Full-Resolution/<sequence>/*_pred.png",
            "url": "local StreamSplat preprocess_depth_davis.py or Mono-DFCGS Stage60 script",
            "target_path": str(davis_root / "depthImages/Full-Resolution"),
            "blocking": "yes_for_anchor_export",
            "notes": "Depth files are required by provider __getitem__ and anchor export.",
        },
        {
            "step": 4,
            "dataset": "YouTube-VOS",
            "action": "Download or mount YouTube-VOS train and valid JPEGImages plus train Annotations",
            "url": "https://youtube-vos.org/",
            "target_path": str(vos_root),
            "blocking": "yes_if_root_missing",
            "notes": "Use official YouTube-VOS access terms. The provider expects train and valid folders.",
        },
        {
            "step": 5,
            "dataset": "YouTube-VOS",
            "action": "Verify train/JPEGImages, train/Annotations, and valid/JPEGImages sequence folders",
            "url": "https://youtube-vos.org/",
            "target_path": str(vos_root),
            "blocking": "yes_if_missing",
            "notes": "Valid split does not require annotations in StreamSplat provider_vos.py.",
        },
        {
            "step": 6,
            "dataset": "YouTube-VOS",
            "action": "Generate train/valid/depthImages/<sequence>/*_pred.png",
            "url": "Mono-DFCGS Stage60 script to be added",
            "target_path": str(vos_root),
            "blocking": "yes_for_anchor_export",
            "notes": "A VOS depth preprocessing script is needed because StreamSplat only includes preprocess_depth_davis.py.",
        },
    ]


def write_report(summary, root_rows, checklist, path):
    lines = [
        "# Stage59 DAVIS/YouTube-VOS Prepare Preflight",
        "",
        "## Summary",
        "",
        f"- DAVIS provider-ready roots: `{summary['davis_provider_ready_roots']}`.",
        f"- DAVIS anchor-export-ready roots: `{summary['davis_anchor_ready_roots']}`.",
        f"- YouTube-VOS provider-ready roots: `{summary['vos_provider_ready_roots']}`.",
        f"- YouTube-VOS anchor-export-ready roots: `{summary['vos_anchor_ready_roots']}`.",
        "",
        "## Root Status",
        "",
        "| Dataset | Root | Exists | Provider-ready | Depth-preprocess-ready | Anchor-export-ready | Missing | Next action |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in root_rows:
        lines.append(
            f"| {row['dataset']} | `{row['root']}` | {row['exists']} | {row['provider_layout_ready']} | "
            f"{row['ready_for_depth_preprocess']} | {row['ready_for_anchor_export']} | {row['missing_items']} | {row['next_action']} |"
        )
    lines.extend([
        "",
        "## Download/Prepare Checklist",
        "",
        "| Step | Dataset | Action | URL | Target | Blocking |",
        "|---:|---|---|---|---|---|",
    ])
    for row in checklist:
        lines.append(
            f"| {row['step']} | {row['dataset']} | {row['action']} | {row['url']} | `{row['target_path']}` | {row['blocking']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--streamsplat_root", type=Path, default=DEFAULT_STREAMSPLAT_ROOT)
    parser.add_argument("--davis_roots", nargs="*", type=Path, default=DEFAULT_DAVIS_ROOTS)
    parser.add_argument("--vos_roots", nargs="*", type=Path, default=DEFAULT_VOS_ROOTS)
    parser.add_argument("--max_sequences", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)

    davis_root = args.davis_roots[0]
    vos_root = args.vos_roots[0]
    root_rows = [inspect_davis_root(root, args.max_sequences) for root in args.davis_roots]
    root_rows.extend(inspect_vos_root(root, args.max_sequences) for root in args.vos_roots)
    layout = layout_rows(davis_root, vos_root)
    providers = provider_rows(args.streamsplat_root)
    checklist = checklist_rows(davis_root, vos_root)

    root_csv = args.summary_root / "stage59_dataset_root_status.csv"
    layout_csv = args.summary_root / "stage59_expected_provider_layout.csv"
    provider_csv = args.summary_root / "stage59_streamsplat_provider_check.csv"
    checklist_csv = args.summary_root / "stage59_download_prepare_checklist.csv"
    report_md = args.summary_root / "stage59_davis_vos_prepare_report.md"
    summary_json = args.summary_root / "stage59_davis_vos_prepare_preflight_summary.json"

    write_csv(root_rows, ROOT_FIELDS, root_csv)
    write_csv(layout, LAYOUT_FIELDS, layout_csv)
    write_csv(providers, PROVIDER_FIELDS, provider_csv)
    write_csv(checklist, CHECKLIST_FIELDS, checklist_csv)

    summary = {
        "stage": 59,
        "mode": "DAVIS/YouTube-VOS download and prepare preflight",
        "streamsplat_root": str(args.streamsplat_root),
        "davis_primary_root": str(davis_root),
        "vos_primary_root": str(vos_root),
        "root_csv": str(root_csv),
        "layout_csv": str(layout_csv),
        "provider_csv": str(provider_csv),
        "checklist_csv": str(checklist_csv),
        "report_md": str(report_md),
        "davis_provider_ready_roots": sum(1 for row in root_rows if row["dataset"] == "DAVIS" and row["provider_layout_ready"] == "true"),
        "davis_anchor_ready_roots": sum(1 for row in root_rows if row["dataset"] == "DAVIS" and row["ready_for_anchor_export"] == "true"),
        "vos_provider_ready_roots": sum(1 for row in root_rows if row["dataset"] == "YouTube-VOS" and row["provider_layout_ready"] == "true"),
        "vos_anchor_ready_roots": sum(1 for row in root_rows if row["dataset"] == "YouTube-VOS" and row["ready_for_anchor_export"] == "true"),
        "notes": "This stage does not download data. Official datasets may require manual download, login, or acceptance of terms.",
    }
    write_report(summary, root_rows, checklist, report_md)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "root_csv": str(root_csv),
        "checklist_csv": str(checklist_csv),
        "davis_provider_ready_roots": summary["davis_provider_ready_roots"],
        "davis_anchor_ready_roots": summary["davis_anchor_ready_roots"],
        "vos_provider_ready_roots": summary["vos_provider_ready_roots"],
        "vos_anchor_ready_roots": summary["vos_anchor_ready_roots"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
