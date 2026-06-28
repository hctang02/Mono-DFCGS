import argparse
import csv
import json
import math
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage92_residual_sideinfo_entropy_preflight"


sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.gaussian_codec import flatten_static_anchor  # noqa: E402
from mono_dfcgs.residual_sideinfo_codec import (  # noqa: E402
    FLOAT16_BYTES,
    HEADER_STRUCT,
    _unpack_ints,
    encode_topk_residual_sideinfo,
)
from scripts.run_stage85_dynamic_residual_sideinfo_preflight import (  # noqa: E402
    DEFAULT_ADAPTER,
    DEFAULT_DENSE_MANIFEST,
    DEFAULT_TASK_MANIFEST,
    build_dense_index,
    linear_anchor,
    load_adapter,
    load_anchor,
    parse_task_rows,
    select_balanced,
)


ROW_FIELDS = [
    "task_id",
    "sequence",
    "codec",
    "reference_gap",
    "target_index",
    "base_method",
    "keep_fraction",
    "side_bits",
    "keep_gaussians",
    "payload_bytes",
    "payload_mib_per_intermediate_frame",
    "theoretical_bytes_without_header",
    "header_bytes",
    "metadata_bytes",
    "index_bytes",
    "residual_bytes",
    "payload_entropy_bits_per_byte",
    "residual_entropy_bits_per_byte",
    "zlib_whole_bytes",
    "component_zlib_bytes",
    "residual_zlib_bytes",
    "delta_index_zlib_bytes",
    "best_candidate",
    "best_bytes",
    "best_mib_per_intermediate_frame",
    "best_ratio_vs_raw",
    "best_savings_bytes_vs_raw",
]

SUMMARY_FIELDS = [
    "base_method",
    "codec",
    "reference_gap",
    "task_count",
    "mean_payload_bytes",
    "mean_payload_mib_per_intermediate_frame",
    "mean_theoretical_bytes_without_header",
    "mean_zlib_whole_bytes",
    "mean_component_zlib_bytes",
    "mean_residual_zlib_bytes",
    "mean_delta_index_zlib_bytes",
    "mean_best_bytes",
    "mean_best_mib_per_intermediate_frame",
    "mean_best_ratio_vs_raw",
    "mean_best_savings_bytes_vs_raw",
    "best_candidate_counts",
]

CANDIDATE_FIELDS = [
    "candidate",
    "row_count",
    "mean_bytes",
    "mean_mib_per_intermediate_frame",
    "mean_ratio_vs_raw",
    "mean_savings_bytes_vs_raw",
]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def average(rows, key):
    values = [float(row[key]) for row in rows]
    return sum(values) / max(len(values), 1)


def byte_entropy(data):
    if not data:
        return 0.0
    counts = Counter(data)
    total = float(len(data))
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def split_payload(payload):
    header_size = HEADER_STRUCT.size
    magic, version, side_bits, index_bits, _flags, attr_dim, gaussian_count, keep_count = HEADER_STRUCT.unpack(payload[:header_size])
    metadata_bytes = int(attr_dim) * 2 * FLOAT16_BYTES
    metadata = payload[header_size:header_size + metadata_bytes]
    offset = header_size + metadata_bytes
    index_bytes = (int(keep_count) * int(index_bits) + 7) // 8
    index_payload = payload[offset:offset + index_bytes]
    offset += index_bytes
    residual_values = int(keep_count) * int(attr_dim)
    residual_bytes = (residual_values * int(side_bits) + 7) // 8
    residual_payload = payload[offset:offset + residual_bytes]
    return {
        "header": payload[:header_size],
        "metadata": metadata,
        "index_payload": index_payload,
        "residual_payload": residual_payload,
        "side_bits": int(side_bits),
        "index_bits": int(index_bits),
        "attr_dim": int(attr_dim),
        "gaussian_count": int(gaussian_count),
        "keep_count": int(keep_count),
    }


def delta_index_bytes(index_payload, keep_count, index_bits):
    indices = sorted(_unpack_ints(index_payload, int(keep_count), int(index_bits)))
    if not indices:
        return b""
    deltas = np.diff(np.array([-1] + indices, dtype=np.int64)).astype("<u2")
    return deltas.tobytes()


def compressed_sizes(payload, level):
    parts = split_payload(payload)
    header = parts["header"]
    metadata = parts["metadata"]
    index_payload = parts["index_payload"]
    residual_payload = parts["residual_payload"]
    delta_payload = delta_index_bytes(index_payload, parts["keep_count"], parts["index_bits"])
    return {
        "raw_fixed": len(payload),
        "zlib_whole": len(zlib.compress(payload, level)),
        "component_zlib": len(header) + len(zlib.compress(metadata, level)) + len(zlib.compress(index_payload, level)) + len(zlib.compress(residual_payload, level)),
        "residual_zlib": len(header) + len(metadata) + len(index_payload) + len(zlib.compress(residual_payload, level)),
        "delta_index_zlib": len(header) + len(zlib.compress(metadata, level)) + len(zlib.compress(delta_payload, level)) + len(zlib.compress(residual_payload, level)),
    }


def summarize_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["base_method"], row["codec"], row["reference_gap"])].append(row)
    out = []
    for (method, codec, gap), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        candidate_counts = Counter(row["best_candidate"] for row in items)
        out.append({
            "base_method": method,
            "codec": codec,
            "reference_gap": gap,
            "task_count": len(items),
            "mean_payload_bytes": average(items, "payload_bytes"),
            "mean_payload_mib_per_intermediate_frame": average(items, "payload_mib_per_intermediate_frame"),
            "mean_theoretical_bytes_without_header": average(items, "theoretical_bytes_without_header"),
            "mean_zlib_whole_bytes": average(items, "zlib_whole_bytes"),
            "mean_component_zlib_bytes": average(items, "component_zlib_bytes"),
            "mean_residual_zlib_bytes": average(items, "residual_zlib_bytes"),
            "mean_delta_index_zlib_bytes": average(items, "delta_index_zlib_bytes"),
            "mean_best_bytes": average(items, "best_bytes"),
            "mean_best_mib_per_intermediate_frame": average(items, "best_mib_per_intermediate_frame"),
            "mean_best_ratio_vs_raw": average(items, "best_ratio_vs_raw"),
            "mean_best_savings_bytes_vs_raw": average(items, "best_savings_bytes_vs_raw"),
            "best_candidate_counts": json.dumps(dict(sorted(candidate_counts.items())), sort_keys=True),
        })
    return out


def summarize_candidates(rows):
    candidates = {
        "raw_fixed": "payload_bytes",
        "zlib_whole": "zlib_whole_bytes",
        "component_zlib": "component_zlib_bytes",
        "residual_zlib": "residual_zlib_bytes",
        "delta_index_zlib": "delta_index_zlib_bytes",
    }
    out = []
    for candidate, field in candidates.items():
        mean_bytes = average(rows, field)
        mean_raw = average(rows, "payload_bytes")
        out.append({
            "candidate": candidate,
            "row_count": len(rows),
            "mean_bytes": mean_bytes,
            "mean_mib_per_intermediate_frame": mean_bytes / (1024.0 * 1024.0),
            "mean_ratio_vs_raw": mean_bytes / mean_raw if mean_raw > 0.0 else 0.0,
            "mean_savings_bytes_vs_raw": mean_raw - mean_bytes,
        })
    return sorted(out, key=lambda row: row["mean_bytes"])


def write_report(summary, summary_rows, candidate_rows, path):
    lines = [
        "# Stage92 Residual Side-Info Entropy Preflight",
        "",
        "## Configuration",
        "",
        f"- task count: `{summary['task_count']}`",
        f"- codecs: `{summary['codecs']}`",
        f"- gaps: `{summary['gaps']}`",
        f"- keep fraction: `{summary['keep_fraction']}`",
        f"- side bits: `{summary['side_bits']}`",
        f"- zlib level: `{summary['zlib_level']}`",
        "- no rendering is run in this stage",
        "",
        "## Candidate Summary",
        "",
        "| candidate | mean bytes | mean MiB/intermediate | ratio vs raw | savings bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in candidate_rows:
        lines.append(
            f"| {row['candidate']} | {row['mean_bytes']} | {row['mean_mib_per_intermediate_frame']} | {row['mean_ratio_vs_raw']} | {row['mean_savings_bytes_vs_raw']} |"
        )
    lines.extend([
        "",
        "## Gap/Method Summary",
        "",
        "| base | gap | tasks | raw bytes | zlib whole | component zlib | delta-index zlib | best bytes | best ratio | best counts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in summary_rows:
        lines.append(
            f"| {row['base_method']} | {row['reference_gap']} | {row['task_count']} | {row['mean_payload_bytes']} | {row['mean_zlib_whole_bytes']} | {row['mean_component_zlib_bytes']} | {row['mean_delta_index_zlib_bytes']} | {row['mean_best_bytes']} | {row['mean_best_ratio_vs_raw']} | `{row['best_candidate_counts']}` |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `raw_fixed` is the Stage91 fixed-length byte payload.",
        "- `zlib_whole` compresses the whole fixed payload with zlib.",
        "- `component_zlib` keeps the header raw but zlib-compresses metadata, indices, and residual bytes separately.",
        "- `delta_index_zlib` sorts selected indices, stores uint16 deltas, and zlib-compresses those deltas plus metadata and residual bytes.",
        "- These are preflight estimates for lossless byte coding; no rendered metrics are recomputed here.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--dense_manifest", type=Path, default=DEFAULT_DENSE_MANIFEST)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--task_split", default="eval")
    parser.add_argument("--codecs", nargs="+", default=["q12"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--max_tasks", type=int, default=60)
    parser.add_argument("--keep_fraction", type=float, default=0.1)
    parser.add_argument("--side_bits", type=int, default=6)
    parser.add_argument("--zlib_level", type=int, default=9)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tasks = select_balanced(parse_task_rows(args.task_manifest, args.task_split, args.codecs, args.gaps), args.max_tasks, args.seed)
    if not tasks:
        raise RuntimeError("No tasks selected")
    dense_index = build_dense_index(args.dense_manifest, sorted({row["split"] for row in tasks}))
    model = load_adapter(args.adapter, args.hidden_dim, device)

    cache = {}
    rows = []
    with torch.no_grad():
        for task in tasks:
            left = load_anchor(task["left_anchor_source_item"], task["left_anchor_source_side"], device, bits=task["bits"], cache=cache)
            right = load_anchor(task["right_anchor_source_item"], task["right_anchor_source_side"], device, bits=task["bits"], cache=cache)
            dense_key = (task["dataset"], task["split"], task["sequence"], task["target_index"])
            if dense_key not in dense_index:
                raise KeyError(f"Missing dense target anchor for {dense_key}")
            target_item, target_side = dense_index[dense_key]
            target_anchor = load_anchor(target_item, target_side, device, bits=None, cache=cache)
            target_attrs = flatten_static_anchor(target_anchor)
            t = torch.tensor([task["normalized_time"]], dtype=torch.float32, device=device)
            base_attrs_by_method = {
                "linear": flatten_static_anchor(linear_anchor(left, right, task["normalized_time"])),
                "stage65_adapter": flatten_static_anchor(model(left, right, t, apply_output_constraints=False)),
            }
            for method, base_attrs in base_attrs_by_method.items():
                payload, info = encode_topk_residual_sideinfo(base_attrs, target_attrs, args.keep_fraction, args.side_bits)
                parts = split_payload(payload)
                sizes = compressed_sizes(payload, args.zlib_level)
                candidates = {
                    "raw_fixed": sizes["raw_fixed"],
                    "zlib_whole": sizes["zlib_whole"],
                    "component_zlib": sizes["component_zlib"],
                    "residual_zlib": sizes["residual_zlib"],
                    "delta_index_zlib": sizes["delta_index_zlib"],
                }
                best_candidate, best_bytes = min(candidates.items(), key=lambda item: item[1])
                rows.append({
                    "task_id": task["task_id"],
                    "sequence": task["sequence"],
                    "codec": task["codec"],
                    "reference_gap": task["reference_gap"],
                    "target_index": task["target_index"],
                    "base_method": method,
                    "keep_fraction": args.keep_fraction,
                    "side_bits": args.side_bits,
                    "keep_gaussians": info["keep_count"],
                    "payload_bytes": info["payload_bytes"],
                    "payload_mib_per_intermediate_frame": info["payload_bytes"] / (1024.0 * 1024.0),
                    "theoretical_bytes_without_header": info["theoretical_bytes_without_header"],
                    "header_bytes": len(parts["header"]),
                    "metadata_bytes": len(parts["metadata"]),
                    "index_bytes": len(parts["index_payload"]),
                    "residual_bytes": len(parts["residual_payload"]),
                    "payload_entropy_bits_per_byte": byte_entropy(payload),
                    "residual_entropy_bits_per_byte": byte_entropy(parts["residual_payload"]),
                    "zlib_whole_bytes": sizes["zlib_whole"],
                    "component_zlib_bytes": sizes["component_zlib"],
                    "residual_zlib_bytes": sizes["residual_zlib"],
                    "delta_index_zlib_bytes": sizes["delta_index_zlib"],
                    "best_candidate": best_candidate,
                    "best_bytes": best_bytes,
                    "best_mib_per_intermediate_frame": best_bytes / (1024.0 * 1024.0),
                    "best_ratio_vs_raw": best_bytes / info["payload_bytes"] if info["payload_bytes"] > 0 else 0.0,
                    "best_savings_bytes_vs_raw": info["payload_bytes"] - best_bytes,
                })
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_rows = summarize_rows(rows)
    candidate_rows = summarize_candidates(rows)
    rows_csv = args.summary_root / "stage92_residual_sideinfo_entropy_rows.csv"
    summary_csv = args.summary_root / "stage92_residual_sideinfo_entropy_summary.csv"
    candidates_csv = args.summary_root / "stage92_residual_sideinfo_entropy_candidates.csv"
    summary_json = args.summary_root / "stage92_residual_sideinfo_entropy_summary.json"
    report_md = args.summary_root / "stage92_residual_sideinfo_entropy_report.md"
    write_csv(rows, rows_csv, ROW_FIELDS)
    write_csv(summary_rows, summary_csv, SUMMARY_FIELDS)
    write_csv(candidate_rows, candidates_csv, CANDIDATE_FIELDS)
    summary = {
        "stage": 92,
        "mode": "residual side-info entropy preflight",
        "task_manifest": str(args.task_manifest),
        "dense_manifest": str(args.dense_manifest),
        "adapter": str(args.adapter),
        "task_split": args.task_split,
        "codecs": args.codecs,
        "gaps": args.gaps,
        "task_count": len(tasks),
        "row_count": len(rows),
        "keep_fraction": args.keep_fraction,
        "side_bits": args.side_bits,
        "zlib_level": args.zlib_level,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "candidates_csv": str(candidates_csv),
        "report_md": str(report_md),
        "summary_rows": summary_rows,
        "candidate_rows": candidate_rows,
        "notes": [
            "No rendering is run in Stage92; payloads are regenerated only for byte/compression analysis.",
            "Residuals remain teacher-derived from dense target anchors.",
            "Candidate compression sizes include codec container assumptions described in the report.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, summary_rows, candidate_rows, report_md)
    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "task_count": len(tasks),
        "row_count": len(rows),
        "best_candidate": candidate_rows[0]["candidate"] if candidate_rows else None,
        "best_mean_bytes": candidate_rows[0]["mean_bytes"] if candidate_rows else None,
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
