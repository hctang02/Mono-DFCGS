import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage8_davis_vos_manifest"

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

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")
DEPTH_EXTS = ("*.png", "*.npy", "*.npz")


def list_files(path, exts):
    p = Path(path)
    if not p.exists():
        return []
    files = []
    for ext in exts:
        files.extend(p.glob(ext))
    return sorted(files)


def read_split(path):
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_existing(paths):
    for path in paths:
        p = Path(path)
        if p.exists():
            return p
    return None


def row_for_sequence(dataset, split, sequence, image_dir, depth_dir=None, mask_dir=None, expected_provider=""):
    frames = list_files(image_dir, IMAGE_EXTS)
    depths = list_files(depth_dir, DEPTH_EXTS) if depth_dir else []
    mask_exists = bool(mask_dir and Path(mask_dir).exists())
    return {
        "dataset": dataset,
        "split": split,
        "sequence": sequence,
        "frame_count": len(frames),
        "depth_count": len(depths),
        "missing_depth_count": max(len(frames) - len(depths), 0),
        "image_dir": str(image_dir),
        "depth_dir": str(depth_dir) if depth_dir else "",
        "mask_dir": str(mask_dir) if mask_dir else "",
        "first_frame": str(frames[0]) if frames else "",
        "last_frame": str(frames[-1]) if frames else "",
        "mask_available": mask_exists,
        "ready_for_depth": len(frames) > 0,
        "ready_for_anchor_export": len(frames) > 0 and len(frames) == len(depths),
        "expected_provider": expected_provider,
    }


def inspect_davis(root):
    root = Path(root)
    image_root = first_existing([
        root / "JPEGImages/Full-Resolution",
        root / "JPEGImages/480p",
    ])
    depth_root = first_existing([
        root / "depthImages/Full-Resolution",
        root / "depthImages/480p",
    ])
    mask_root = first_existing([
        root / "Annotations_unsupervised/Full-Resolution",
        root / "Annotations_unsupervised/480p",
        root / "Annotations/Full-Resolution",
        root / "Annotations/480p",
    ])
    split_map = {
        "train": read_split(root / "ImageSets/2017/train.txt"),
        "val": read_split(root / "ImageSets/2017/val.txt"),
    }
    rows = []
    if image_root:
        all_sequences = sorted([p.name for p in image_root.iterdir() if p.is_dir()])
        for split, names in split_map.items():
            for sequence in names:
                if sequence in all_sequences:
                    rows.append(row_for_sequence(
                        "DAVIS",
                        split,
                        sequence,
                        image_root / sequence,
                        depth_root / sequence if depth_root else None,
                        mask_root / sequence if mask_root else None,
                        "provider_davis",
                    ))
        if not rows:
            for sequence in all_sequences:
                rows.append(row_for_sequence(
                    "DAVIS",
                    "unspecified",
                    sequence,
                    image_root / sequence,
                    depth_root / sequence if depth_root else None,
                    mask_root / sequence if mask_root else None,
                    "provider_davis",
                ))
    return {
        "dataset": "DAVIS",
        "root": str(root),
        "exists": root.exists(),
        "image_root": str(image_root) if image_root else "",
        "depth_root": str(depth_root) if depth_root else "",
        "mask_root": str(mask_root) if mask_root else "",
        "sequences": rows,
        "required_layout": "ImageSets/2017/*.txt, JPEGImages/{Full-Resolution or 480p}, optional Annotations*, depthImages generated before anchor export",
    }


def inspect_vos(root):
    root = Path(root)
    rows = []
    for split in ("train", "valid"):
        image_root = root / split / "JPEGImages"
        depth_root = root / split / "depthImages"
        mask_root = root / split / "Annotations"
        if not image_root.exists():
            continue
        for seq_dir in sorted([p for p in image_root.iterdir() if p.is_dir()]):
            rows.append(row_for_sequence(
                "YouTube-VOS",
                split,
                seq_dir.name,
                seq_dir,
                depth_root / seq_dir.name if depth_root.exists() else None,
                mask_root / seq_dir.name if mask_root.exists() else None,
                "provider_vos",
            ))
    return {
        "dataset": "YouTube-VOS",
        "root": str(root),
        "exists": root.exists(),
        "image_root": "",
        "depth_root": "",
        "mask_root": "",
        "sequences": rows,
        "required_layout": "train/valid JPEGImages, train Annotations, depthImages generated before anchor export",
    }


def write_sequence_csv(rows, path):
    fields = [
        "dataset", "split", "sequence", "frame_count", "depth_count", "missing_depth_count",
        "image_dir", "depth_dir", "mask_dir", "first_frame", "last_frame", "mask_available",
        "ready_for_depth", "ready_for_anchor_export", "expected_provider",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_root_csv(rows, path):
    fields = [
        "dataset", "root", "exists", "sequence_count", "ready_for_depth_count",
        "ready_for_anchor_export_count", "frame_count", "depth_count", "required_layout",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_requirements(path, root_summaries):
    available = [row for row in root_summaries if row["sequence_count"]]
    lines = [
        "# Stage 8 Dataset Setup Requirements",
        "",
        "Stage 8 prepares sequence manifests for DAVIS and YouTube-VOS before Mono-DFCGS anchor export.",
        "",
        "## Current Status",
        "",
        f"- Dataset roots with sequences detected: {len(available)}",
        f"- Dataset roots checked: {len(root_summaries)}",
        "",
        "## Required DAVIS Layout",
        "",
        "```text",
        "DAVIS/",
        "  ImageSets/2017/train.txt",
        "  ImageSets/2017/val.txt",
        "  JPEGImages/Full-Resolution/<sequence>/*.jpg",
        "  Annotations_unsupervised/Full-Resolution/<sequence>/*.png",
        "  depthImages/Full-Resolution/<sequence>/*.png",
        "```",
        "",
        "`JPEGImages/480p`, `Annotations/480p`, and `depthImages/480p` are also accepted by this manifest script, but StreamSplat provider compatibility may require path adaptation.",
        "",
        "## Required YouTube-VOS Layout",
        "",
        "```text",
        "YouTube-VOS/",
        "  train/JPEGImages/<sequence>/*.jpg",
        "  valid/JPEGImages/<sequence>/*.jpg",
        "  train/Annotations/<sequence>/*.png",
        "  train/depthImages/<sequence>/*.png",
        "```",
        "",
        "## Next Action",
        "",
        "Mount or download a DAVIS 2017 root into one of the checked paths, or pass it explicitly with `--davis_roots /path/to/DAVIS`. Then run depth preprocessing before anchor export.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_root(info):
    seqs = info["sequences"]
    return {
        "dataset": info["dataset"],
        "root": info["root"],
        "exists": info["exists"],
        "sequence_count": len(seqs),
        "ready_for_depth_count": sum(1 for row in seqs if row["ready_for_depth"]),
        "ready_for_anchor_export_count": sum(1 for row in seqs if row["ready_for_anchor_export"]),
        "frame_count": sum(row["frame_count"] for row in seqs),
        "depth_count": sum(row["depth_count"] for row in seqs),
        "required_layout": info["required_layout"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--davis_roots", nargs="*", default=DEFAULT_DAVIS_ROOTS)
    parser.add_argument("--vos_roots", nargs="*", default=DEFAULT_VOS_ROOTS)
    parser.add_argument("--strict", action="store_true", help="Fail if no sequence manifest rows are produced.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)

    root_infos = []
    root_infos.extend(inspect_davis(root) for root in args.davis_roots)
    root_infos.extend(inspect_vos(root) for root in args.vos_roots)

    sequence_rows = []
    for info in root_infos:
        sequence_rows.extend(info["sequences"])
    root_summaries = [summarize_root(info) for info in root_infos]

    ready_for_depth = [row for row in sequence_rows if row["ready_for_depth"]]
    ready_for_anchor_export = [row for row in sequence_rows if row["ready_for_anchor_export"]]
    summary = {
        "root_summaries": root_summaries,
        "sequence_count": len(sequence_rows),
        "ready_for_depth_count": len(ready_for_depth),
        "ready_for_anchor_export_count": len(ready_for_anchor_export),
        "frame_count": sum(row["frame_count"] for row in sequence_rows),
        "depth_count": sum(row["depth_count"] for row in sequence_rows),
        "recommendation": (
            "No sequence rows were detected. Mount/download DAVIS 2017 or YouTube-VOS and rerun this script."
            if not sequence_rows else
            "Run depth preprocessing for rows with ready_for_depth=true and ready_for_anchor_export=false, then export anchors."
        ),
    }

    sequence_csv = args.summary_root / "stage8_sequence_manifest.csv"
    root_csv = args.summary_root / "stage8_root_preflight.csv"
    summary_json = args.summary_root / "stage8_preflight_summary.json"
    requirements_md = args.summary_root / "stage8_dataset_setup_requirements.md"

    write_sequence_csv(sequence_rows, sequence_csv)
    write_root_csv(root_summaries, root_csv)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_requirements(requirements_md, root_summaries)

    print(json.dumps({
        "summary": str(summary_json),
        "sequence_csv": str(sequence_csv),
        "root_csv": str(root_csv),
        "requirements": str(requirements_md),
        "sequence_count": len(sequence_rows),
        "ready_for_depth_count": len(ready_for_depth),
        "ready_for_anchor_export_count": len(ready_for_anchor_export),
    }, indent=2))

    if args.strict and not sequence_rows:
        raise RuntimeError("No DAVIS/YouTube-VOS sequences detected.")


if __name__ == "__main__":
    raise SystemExit(main())
