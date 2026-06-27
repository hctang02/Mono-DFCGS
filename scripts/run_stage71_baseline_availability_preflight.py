import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage71_baseline_availability_preflight"

FCGS_REPO = Path("/mnt/hdd2tC/hctang/third_party/FCGS")
DFCGS_REPO = Path("/mnt/hdd2tC/hctang/third_party/D-FCGS")
CWGS_SUMMARY_ROOT = Path("/mnt/hdd2tC/tmp/opencode/multisequence_rd/rd_sweep/ours")

STAGE52_SUMMARY_CSV = REPO_ROOT / "experiments/stage52_fcgs_dfcgs_baseline_preflight/stage52_baseline_summary_records.csv"
STAGE53_EXTERNAL_CSV = REPO_ROOT / "experiments/stage53_baseline_comparison_scaffold/stage53_external_baseline_rows.csv"
STAGE70_BASELINE_CSV = REPO_ROOT / "experiments/stage70_scoped_davis_rd_package/stage70_baseline_status.csv"
STAGE70_SUMMARY_JSON = REPO_ROOT / "experiments/stage70_scoped_davis_rd_package/stage70_scoped_davis_rd_package_summary.json"
STAGE61_MANIFEST = REPO_ROOT / "experiments/stage61_davis_anchor_export_data_full/stage61_davis_anchor_export_manifest.csv"
DAVIS_ROOT = Path("/data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS")


CODE_FIELDS = [
    "method",
    "repo_path",
    "repo_exists",
    "git_commit",
    "readme_present",
    "python_file_count",
    "davis_mentions",
    "entrypoints_found",
    "checkpoints_found",
    "required_tools",
    "missing_tools",
    "expected_input_protocol",
    "davis_adapter_status",
    "local_code_status",
    "notes",
]

ARTIFACT_FIELDS = [
    "artifact_group",
    "method",
    "root",
    "pattern",
    "root_exists",
    "file_count",
    "record_count",
    "davis_related_count",
    "rate_records",
    "quality_records",
    "stage53_candidate_records",
    "fair_local_records",
    "samples",
    "codec_modes",
    "comparison_status",
    "notes",
]

FAIRNESS_FIELDS = [
    "method",
    "local_code_status",
    "local_artifact_status",
    "davis_scoped_apples_to_apples_ready",
    "input_protocol_status",
    "rate_status",
    "all_psnr_status",
    "fair_comparison_status",
    "next_action",
]

MISSING_FIELDS = [
    "method",
    "category",
    "requirement",
    "current_status",
    "priority",
]


def read_csv(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_text(path, limit_chars=200_000):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:limit_chars]


def bool_text(value):
    return "true" if value else "false"


def count_files(root, pattern, cap=100_000):
    if not root.exists():
        return 0
    count = 0
    for _path in root.glob(pattern):
        count += 1
        if count >= cap:
            return count
    return count


def rel_existing(root, paths):
    if not root.exists():
        return []
    out = []
    for item in paths:
        path = root / item
        if path.exists():
            out.append(item)
    return out


def git_commit(repo):
    git_dir = repo / ".git"
    if not git_dir.exists():
        return ""
    if git_dir.is_file():
        text = read_text(git_dir, 4096).strip()
        if text.startswith("gitdir:"):
            target = text.split(":", 1)[1].strip()
            git_dir = (repo / target).resolve()
    head = git_dir / "HEAD"
    head_text = read_text(head, 4096).strip()
    if not head_text:
        return ""
    if head_text.startswith("ref:"):
        ref = head_text.split(" ", 1)[1].strip()
        return read_text(git_dir / ref, 4096).strip()
    return head_text


def davis_mentions(repo):
    if not repo.exists():
        return 0
    total = 0
    candidates = [repo / "README.md"] + list(repo.glob("*.py")) + list((repo / "scripts").glob("*.py") if (repo / "scripts").exists() else [])
    for path in candidates:
        text = read_text(path, 120_000).lower()
        total += text.count("davis")
    return total


def summarize_values(rows, key, limit=12):
    values = sorted({str(row.get(key, "")) for row in rows if row.get(key, "") not in (None, "")})
    if len(values) > limit:
        return " ".join(values[:limit]) + f" ... (+{len(values) - limit})"
    return " ".join(values)


def is_truthy(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def has_value(row, key):
    return row.get(key) not in (None, "")


def code_inventory_rows():
    fcgs_entrypoints = rel_existing(FCGS_REPO, [
        "encode_single_scene.py",
        "decode_single_scene.py",
        "decode_single_scene_validate.py",
    ])
    dfcgs_entrypoints = rel_existing(DFCGS_REPO, [
        "scripts/run_dfcgs_infer.py",
        "scripts/run_dfcgs_train.py",
        "train_fcgsd.py",
        "validate.py",
        "summerize.py",
    ])
    return [
        {
            "method": "FCGS",
            "repo_path": str(FCGS_REPO),
            "repo_exists": bool_text(FCGS_REPO.exists()),
            "git_commit": git_commit(FCGS_REPO),
            "readme_present": bool_text((FCGS_REPO / "README.md").exists()),
            "python_file_count": count_files(FCGS_REPO, "**/*.py"),
            "davis_mentions": davis_mentions(FCGS_REPO),
            "entrypoints_found": " ".join(fcgs_entrypoints),
            "checkpoints_found": count_files(FCGS_REPO / "checkpoints", "checkpoint_*.pkl"),
            "required_tools": "tmc3 cuda_ext torch lpips",
            "missing_tools": " ".join(tool for tool in ["tmc3"] if shutil.which(tool) is None),
            "expected_input_protocol": "static 3DGS point_cloud.ply plus Gaussian-Splatting Scene source_path for validation",
            "davis_adapter_status": "no DAVIS or Mono-DFCGS anchor adapter detected",
            "local_code_status": "present" if FCGS_REPO.exists() else "missing",
            "notes": "Can compress existing .ply 3DGS; Stage61 DAVIS anchors are tensor manifests, not FCGS-ready .ply scenes.",
        },
        {
            "method": "D-FCGS",
            "repo_path": str(DFCGS_REPO),
            "repo_exists": bool_text(DFCGS_REPO.exists()),
            "git_commit": git_commit(DFCGS_REPO),
            "readme_present": bool_text((DFCGS_REPO / "README.md").exists()),
            "python_file_count": count_files(DFCGS_REPO, "**/*.py"),
            "davis_mentions": davis_mentions(DFCGS_REPO),
            "entrypoints_found": " ".join(dfcgs_entrypoints),
            "checkpoints_found": count_files(DFCGS_REPO / "ckpt", "*.pth"),
            "required_tools": "tmc3 zstd cuda_ext torch pytorch3d colmap tiny-cuda-nn",
            "missing_tools": " ".join(tool for tool in ["tmc3", "zstd"] if shutil.which(tool) is None),
            "expected_input_protocol": "multi-view per-frame Gaussian sequence with 3DGStream/Colmap-style frame folders",
            "davis_adapter_status": "no monocular DAVIS adapter detected",
            "local_code_status": "present" if DFCGS_REPO.exists() else "missing",
            "notes": "README pipeline assumes multiview video and per-frame GS generation; direct use on monocular DAVIS would violate the Mono-DFCGS input protocol unless adapted carefully.",
        },
        {
            "method": "CWGS",
            "repo_path": "",
            "repo_exists": "false",
            "git_commit": "",
            "readme_present": "false",
            "python_file_count": "0",
            "davis_mentions": "0",
            "entrypoints_found": "",
            "checkpoints_found": "0",
            "required_tools": "unknown",
            "missing_tools": "code_checkout",
            "expected_input_protocol": "not locally inventoried",
            "davis_adapter_status": "no code checkout detected; old cwgs_rd_summary artifacts only",
            "local_code_status": "missing_optional",
            "notes": "Optional supplemental baseline. Local old CWGS-like summaries exist under multisequence_rd but are not DAVIS scoped.",
        },
    ]


def artifact_rows():
    rows = []
    stage52_rows = read_csv(STAGE52_SUMMARY_CSV)
    rows.append({
        "artifact_group": "stage52_fcgs_dfcgs_summary_records",
        "method": "FCGS/D-FCGS",
        "root": str(STAGE52_SUMMARY_CSV.parent),
        "pattern": STAGE52_SUMMARY_CSV.name,
        "root_exists": bool_text(STAGE52_SUMMARY_CSV.parent.exists()),
        "file_count": 1 if STAGE52_SUMMARY_CSV.exists() else 0,
        "record_count": len(stage52_rows),
        "davis_related_count": sum(1 for row in stage52_rows if "davis" in json.dumps(row).lower()),
        "rate_records": sum(1 for row in stage52_rows if has_value(row, "avg_size_mib_per_frame")),
        "quality_records": sum(1 for row in stage52_rows if has_value(row, "psnr_avg") and row.get("dummy_reference_images") == "False"),
        "stage53_candidate_records": sum(1 for row in stage52_rows if is_truthy(row.get("stage53_candidate"))),
        "fair_local_records": 0,
        "samples": summarize_values(stage52_rows, "sample"),
        "codec_modes": summarize_values(stage52_rows, "codec_mode"),
        "comparison_status": "old local diagnostic/reference only",
        "notes": "Rows are non-DAVIS and use full FCGS/D-FCGS codec MiB/frame, not Mono-DFCGS keyframe-anchor rate.",
    })

    stage53_rows = read_csv(STAGE53_EXTERNAL_CSV)
    rows.append({
        "artifact_group": "stage53_external_baseline_rows",
        "method": "FCGS/D-FCGS",
        "root": str(STAGE53_EXTERNAL_CSV.parent),
        "pattern": STAGE53_EXTERNAL_CSV.name,
        "root_exists": bool_text(STAGE53_EXTERNAL_CSV.parent.exists()),
        "file_count": 1 if STAGE53_EXTERNAL_CSV.exists() else 0,
        "record_count": len(stage53_rows),
        "davis_related_count": sum(1 for row in stage53_rows if "davis" in json.dumps(row).lower()),
        "rate_records": sum(1 for row in stage53_rows if has_value(row, "rate_mib_per_frame")),
        "quality_records": sum(1 for row in stage53_rows if has_value(row, "quality_psnr") and is_truthy(row.get("quality_reliable_for_input_video"))),
        "stage53_candidate_records": sum(1 for row in stage53_rows if is_truthy(row.get("stage53_candidate"))),
        "fair_local_records": sum(1 for row in stage53_rows if is_truthy(row.get("fair_local_run"))),
        "samples": summarize_values(stage53_rows, "sample"),
        "codec_modes": summarize_values(stage53_rows, "variant", limit=6),
        "comparison_status": "explicitly not apples-to-apples",
        "notes": "Stage53 fair_local_run is false for all external rows.",
    })

    stage70_rows = read_csv(STAGE70_BASELINE_CSV)
    rows.append({
        "artifact_group": "stage70_baseline_status",
        "method": "FCGS/D-FCGS/CWGS",
        "root": str(STAGE70_BASELINE_CSV.parent),
        "pattern": STAGE70_BASELINE_CSV.name,
        "root_exists": bool_text(STAGE70_BASELINE_CSV.parent.exists()),
        "file_count": 1 if STAGE70_BASELINE_CSV.exists() else 0,
        "record_count": len(stage70_rows),
        "davis_related_count": len(stage70_rows),
        "rate_records": 0,
        "quality_records": 0,
        "stage53_candidate_records": 0,
        "fair_local_records": 0,
        "samples": "DAVIS scoped eval subset status rows",
        "codec_modes": summarize_values(stage70_rows, "method"),
        "comparison_status": "baseline status only",
        "notes": "Stage70 correctly marks FCGS/D-FCGS/CWGS as not locally evaluated apples-to-apples.",
    })

    cwgs_paths = sorted(CWGS_SUMMARY_ROOT.glob("**/cwgs_rd_summary.json")) if CWGS_SUMMARY_ROOT.exists() else []
    cwgs_records = []
    for path in cwgs_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            payload = {"path": str(path)}
        payload["path"] = str(path)
        cwgs_records.append(payload)
    rows.append({
        "artifact_group": "legacy_cwgs_rd_summaries",
        "method": "CWGS",
        "root": str(CWGS_SUMMARY_ROOT),
        "pattern": "**/cwgs_rd_summary.json",
        "root_exists": bool_text(CWGS_SUMMARY_ROOT.exists()),
        "file_count": len(cwgs_paths),
        "record_count": len(cwgs_records),
        "davis_related_count": sum(1 for row in cwgs_records if "davis" in json.dumps(row).lower()),
        "rate_records": sum(1 for row in cwgs_records if row.get("bitstream_mib") not in (None, "")),
        "quality_records": sum(1 for row in cwgs_records if row.get("psnr_avg") not in (None, "")),
        "stage53_candidate_records": 0,
        "fair_local_records": 0,
        "samples": summarize_values(cwgs_records, "sample"),
        "codec_modes": "cwgs_rd_summary",
        "comparison_status": "old non-DAVIS reference only",
        "notes": "Old CWGS-like summaries use multisequence_rd references, not DAVIS scoped Stage70 frames.",
    })
    return rows


def fairness_rows(code_rows, artifact_inventory):
    code_status = {row["method"]: row["local_code_status"] for row in code_rows}
    stage52 = next(row for row in artifact_inventory if row["artifact_group"] == "stage52_fcgs_dfcgs_summary_records")
    stage53 = next(row for row in artifact_inventory if row["artifact_group"] == "stage53_external_baseline_rows")
    cwgs = next(row for row in artifact_inventory if row["artifact_group"] == "legacy_cwgs_rd_summaries")
    return [
        {
            "method": "FCGS",
            "local_code_status": code_status.get("FCGS", "unknown"),
            "local_artifact_status": f"{stage52['record_count']} old FCGS/D-FCGS summary rows; {stage53['fair_local_records']} fair rows",
            "davis_scoped_apples_to_apples_ready": "false",
            "input_protocol_status": "expects 3DGS .ply and Gaussian-Splatting Scene cameras; Stage61 DAVIS anchors are .pt tensors",
            "rate_status": "FCGS bitstream bytes can be counted after a DAVIS wrapper exists; no DAVIS scoped bitstream MiB/frame yet",
            "all_psnr_status": "no DAVIS eval-subset all-frame PSNR table from local FCGS run",
            "fair_comparison_status": "not ready; old rows are non-DAVIS and Stage53 marks them non-apples-to-apples",
            "next_action": "implement a DAVIS FCGS wrapper or anchor-to-ply conversion, run lmd sweep, decode/render same frames, and count actual bitstreams plus metadata",
        },
        {
            "method": "D-FCGS",
            "local_code_status": code_status.get("D-FCGS", "unknown"),
            "local_artifact_status": f"{stage52['record_count']} old FCGS/D-FCGS summary rows; {stage53['fair_local_records']} fair rows",
            "davis_scoped_apples_to_apples_ready": "false",
            "input_protocol_status": "expects multiview per-frame GS sequences; direct monocular DAVIS use would require a protocol-safe adapter",
            "rate_status": "old GOP summaries include full I-frame plus P-frame codec rate; DAVIS scoped rate missing",
            "all_psnr_status": "no DAVIS eval-subset all-frame PSNR table from local D-FCGS run",
            "fair_comparison_status": "not ready; protocol and data source differ from Mono-DFCGS DAVIS scoped eval",
            "next_action": "decide whether to adapt D-FCGS to StreamSplat/DAVIS Gaussian sequences without multiview leakage or keep it as external-reference-only",
        },
        {
            "method": "CWGS",
            "local_code_status": code_status.get("CWGS", "unknown"),
            "local_artifact_status": f"{cwgs['record_count']} old cwgs_rd_summary rows; no code checkout",
            "davis_scoped_apples_to_apples_ready": "false",
            "input_protocol_status": "local code/protocol not inventoried; old artifacts are multisequence_rd, not DAVIS",
            "rate_status": "old bitstream_mib exists but not DAVIS scoped or protocol-aligned",
            "all_psnr_status": "old psnr_avg exists for old references; no DAVIS scoped all-frame PSNR",
            "fair_comparison_status": "optional supplemental baseline only after FCGS/D-FCGS",
            "next_action": "only pursue after primary FCGS/D-FCGS baselines or if a local CWGS code checkout/protocol is provided",
        },
    ]


def missing_field_rows():
    rows = []
    shared = [
        ("frame_protocol", "same DAVIS val eval sequences/frames/gaps as Stage70", "not yet run for any external baseline", "high"),
        ("quality", "all-frame PSNR against the same resized DAVIS RGB targets", "missing", "high"),
        ("rate", "actual transmitted baseline bitstreams MiB/frame plus necessary metadata", "missing for DAVIS scoped runs", "high"),
        ("reporting", "per-sequence and mean rate/PSNR tables plus RD curve", "missing for external baselines", "medium"),
        ("storage", "heavy decoded frames/bitstreams kept outside git under /data or tmp", "policy defined but not exercised for new baseline", "medium"),
    ]
    for method in ["FCGS", "D-FCGS", "CWGS"]:
        for category, requirement, status, priority in shared:
            rows.append({
                "method": method,
                "category": category,
                "requirement": requirement,
                "current_status": status,
                "priority": priority,
            })
    rows.extend([
        {
            "method": "FCGS",
            "category": "input_adapter",
            "requirement": "convert each DAVIS/StreamSplat keyframe Gaussian anchor or full frame GS to FCGS-compatible .ply without adding non-monocular side input",
            "current_status": "not implemented",
            "priority": "high",
        },
        {
            "method": "FCGS",
            "category": "runner",
            "requirement": "batch encode/decode/validate lmd sweep over Stage70 scoped DAVIS sequences",
            "current_status": "not implemented",
            "priority": "high",
        },
        {
            "method": "D-FCGS",
            "category": "input_adapter",
            "requirement": "construct a protocol-safe monocular DAVIS Gaussian sequence accepted by D-FCGS, or document why this baseline is incompatible",
            "current_status": "not implemented; upstream expects multiview/3DGStream-style sequences",
            "priority": "high",
        },
        {
            "method": "D-FCGS",
            "category": "gop_policy",
            "requirement": "align I/P frame or GoF policy with Stage70 gaps 4/8/16 and count I-frame side rate",
            "current_status": "not implemented",
            "priority": "high",
        },
        {
            "method": "CWGS",
            "category": "code",
            "requirement": "local code checkout and reproducible DAVIS runner",
            "current_status": "missing; only old cwgs_rd_summary artifacts found",
            "priority": "low",
        },
    ])
    return rows


def write_report(summary, code_rows, artifact_inventory, fairness, missing_rows, path):
    lines = [
        "# Stage71 Baseline Availability Preflight",
        "",
        "## Bottom Line",
        "",
        "No local FCGS/D-FCGS/CWGS artifact is ready to enter the Stage70 DAVIS scoped RD table as an apples-to-apples baseline.",
        "FCGS and D-FCGS code are present, but their input protocols do not directly match the current Mono-DFCGS DAVIS anchor/eval setup.",
        "",
        "## Code Inventory",
        "",
        "| Method | Code status | DAVIS mentions | Entrypoints | Missing tools |",
        "|---|---|---:|---|---|",
    ]
    for row in code_rows:
        lines.append(
            f"| {row['method']} | {row['local_code_status']} | {row['davis_mentions']} | `{row['entrypoints_found']}` | `{row['missing_tools']}` |"
        )
    lines.extend([
        "",
        "## Artifact Inventory",
        "",
        "| Group | Method | Records | DAVIS-related | Rate rows | Quality rows | Fair rows | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in artifact_inventory:
        lines.append(
            f"| {row['artifact_group']} | {row['method']} | {row['record_count']} | {row['davis_related_count']} | {row['rate_records']} | {row['quality_records']} | {row['fair_local_records']} | {row['comparison_status']} |"
        )
    lines.extend([
        "",
        "## Fairness Status",
        "",
        "| Method | Ready | Input protocol | Rate | All-frame PSNR | Next action |",
        "|---|---|---|---|---|---|",
    ])
    for row in fairness:
        lines.append(
            f"| {row['method']} | {row['davis_scoped_apples_to_apples_ready']} | {row['input_protocol_status']} | {row['rate_status']} | {row['all_psnr_status']} | {row['next_action']} |"
        )
    high_missing = [row for row in missing_rows if row["priority"] == "high"]
    lines.extend([
        "",
        "## High-Priority Missing Items",
        "",
        "| Method | Category | Requirement | Status |",
        "|---|---|---|---|",
    ])
    for row in high_missing:
        lines.append(f"| {row['method']} | {row['category']} | {row['requirement']} | {row['current_status']} |")
    lines.extend([
        "",
        "## Summary JSON",
        "",
        f"- `{summary['summary_json']}`",
        "",
    ])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_summary(code_rows, artifact_inventory, fairness, missing_rows, output_paths):
    stage70_summary = {}
    if STAGE70_SUMMARY_JSON.exists():
        stage70_summary = json.loads(STAGE70_SUMMARY_JSON.read_text(encoding="utf-8"))
    return {
        "stage": 71,
        "mode": "baseline availability preflight",
        "davis_root_exists": DAVIS_ROOT.exists(),
        "stage61_manifest_exists": STAGE61_MANIFEST.exists(),
        "stage70_samples": stage70_summary.get("samples", []),
        "stage70_gaps": stage70_summary.get("gaps", []),
        "code_inventory_count": len(code_rows),
        "artifact_inventory_count": len(artifact_inventory),
        "fairness_ready_methods": [row["method"] for row in fairness if row["davis_scoped_apples_to_apples_ready"] == "true"],
        "not_ready_methods": [row["method"] for row in fairness if row["davis_scoped_apples_to_apples_ready"] != "true"],
        "high_priority_missing_count_by_method": dict(Counter(row["method"] for row in missing_rows if row["priority"] == "high")),
        "outputs": output_paths,
        "summary_json": output_paths["summary_json"],
        "notes": [
            "FCGS and D-FCGS code are present locally, but no DAVIS scoped apples-to-apples run exists.",
            "Old FCGS/D-FCGS/CWGS artifacts are useful diagnostics only because data/protocol/rate scopes differ from Stage70.",
            "Next fair baseline step is a DAVIS-specific wrapper/runner with explicit bitstream-rate and all-frame PSNR accounting.",
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)

    code_rows = code_inventory_rows()
    artifacts = artifact_rows()
    fairness = fairness_rows(code_rows, artifacts)
    missing_rows = missing_field_rows()

    code_csv = args.summary_root / "stage71_baseline_code_inventory.csv"
    artifact_csv = args.summary_root / "stage71_baseline_artifact_inventory.csv"
    fairness_csv = args.summary_root / "stage71_baseline_fairness_status.csv"
    missing_csv = args.summary_root / "stage71_baseline_missing_fields.csv"
    summary_json = args.summary_root / "stage71_baseline_availability_preflight_summary.json"
    report_md = args.summary_root / "stage71_baseline_availability_preflight_report.md"

    write_csv(code_rows, code_csv, CODE_FIELDS)
    write_csv(artifacts, artifact_csv, ARTIFACT_FIELDS)
    write_csv(fairness, fairness_csv, FAIRNESS_FIELDS)
    write_csv(missing_rows, missing_csv, MISSING_FIELDS)

    output_paths = {
        "code_csv": str(code_csv),
        "artifact_csv": str(artifact_csv),
        "fairness_csv": str(fairness_csv),
        "missing_csv": str(missing_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
    }
    summary = build_summary(code_rows, artifacts, fairness, missing_rows, output_paths)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, code_rows, artifacts, fairness, missing_rows, report_md)

    print(json.dumps({
        "summary": str(summary_json),
        "report": str(report_md),
        "fairness_ready_methods": summary["fairness_ready_methods"],
        "not_ready_methods": summary["not_ready_methods"],
        "high_priority_missing_count_by_method": summary["high_priority_missing_count_by_method"],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
