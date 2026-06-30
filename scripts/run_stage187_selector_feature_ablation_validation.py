import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE163_ROWS = REPO_ROOT / "experiments/stage163_davis_rgb_motion_selector_data/stage163_davis_rgb_motion_selector_feature_rows.csv"
DEFAULT_STAGE165_PACKAGE = REPO_ROOT / "experiments/stage165_multifeature_keyframe_schedule_preflight/stage165_multifeature_keyframe_schedule_preflight_package.json"
DEFAULT_TASK_MANIFEST = REPO_ROOT / "experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_tasks.csv"
DEFAULT_STAGE186_RD_QUALITY = REPO_ROOT / "experiments/stage186_full_sequence_quality_validation/stage186_measured_rd_quality_points.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage187_selector_feature_ablation_validation"

FEATURES = [
    "rgb_motion_proxy_score",
    "rgb_mad_linear_interp_target",
    "rgb_mad_left_right",
    "edge_mad_left_right",
    "hist_chi_left_right",
]

VARIANTS = [
    ("full_stage165_features", FEATURES, "current Stage165 five-feature gate"),
    ("drop_rgb_motion_proxy", [f for f in FEATURES if f != "rgb_motion_proxy_score"], "remove combined RGB/motion proxy score"),
    ("drop_interp_rgb", [f for f in FEATURES if f != "rgb_mad_linear_interp_target"], "remove target-vs-linear RGB cue"),
    ("drop_endpoint_rgb", [f for f in FEATURES if f != "rgb_mad_left_right"], "remove endpoint RGB difference cue"),
    ("drop_edge_motion", [f for f in FEATURES if f != "edge_mad_left_right"], "remove edge motion cue"),
    ("drop_hist_motion", [f for f in FEATURES if f != "hist_chi_left_right"], "remove histogram motion cue"),
    ("rgb_only", ["rgb_mad_linear_interp_target", "rgb_mad_left_right"], "use only direct RGB cues"),
    ("motion_proxy_edge_hist", ["rgb_motion_proxy_score", "edge_mad_left_right", "hist_chi_left_right"], "use proxy plus edge/hist motion cues"),
    ("proxy_only", ["rgb_motion_proxy_score"], "use only combined RGB/motion proxy"),
    ("edge_hist_only", ["edge_mad_left_right", "hist_chi_left_right"], "use only edge and histogram cues"),
]

ABLATION_FIELDS = [
    "variant",
    "description",
    "feature_count",
    "features",
    "rank_threshold",
    "min_votes",
    "selected_count",
    "keyframe_count",
    "extra_keyframes_vs_gap8",
    "metadata_bytes",
    "hard_quality_count",
    "selected_hard_count",
    "precision_hard",
    "recall_hard",
    "f1_hard",
    "high_payload_count",
    "selected_high_payload_count",
    "precision_payload",
    "recall_payload",
    "f1_payload",
    "mean_selected_psnr",
    "mean_unselected_psnr",
    "mean_selected_lpips",
    "mean_unselected_lpips",
    "mean_selected_payload_bytes",
    "mean_unselected_payload_bytes",
    "selected_count_delta_vs_full",
    "recall_hard_delta_vs_full",
    "recall_payload_delta_vs_full",
]

SELECTED_FIELDS = [
    "variant",
    "task_id",
    "sequence",
    "target_index",
    "vote_count",
    "selector_score",
    "selected_for_extra_keyframe",
    "hard_quality_label",
    "high_payload_label",
    "stage158_psnr",
    "stage158_lpips",
    "payload_bytes",
]

CONTEXT_FIELDS = ["item", "value", "note"]


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def f1(precision, recall):
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def parse_task_frames(path):
    out = {}
    for row in read_csv(path):
        if row["task_split"] == "eval" and row["codec"] == "q12":
            out[row["sequence"]] = int(row["total_frames"])
    return out


def percentile_ranks(rows, col):
    indexed = sorted((float(row[col]), idx) for idx, row in enumerate(rows))
    n = len(indexed)
    ranks = [0.0] * n
    if n <= 1:
        return ranks
    for rank, (_, idx) in enumerate(indexed):
        ranks[idx] = rank / (n - 1)
    return ranks


def prepare_rows(raw_rows):
    rows = []
    for row in raw_rows:
        item = dict(row)
        for feature in FEATURES:
            item[feature] = float(item[feature])
        item["target_index"] = int(item["target_index"])
        item["stage158_psnr"] = float(item["stage158_psnr"])
        item["stage158_lpips"] = float(item["stage158_lpips"])
        item["payload_bytes"] = float(item["payload_bytes"])
        item["hard_quality_label"] = int(float(item["stage158_psnr"]) < 26.0 or float(item["stage158_lpips"]) > 0.22)
        item["high_payload_label"] = int(float(item["payload_bytes"]) > 220000.0)
        rows.append(item)
    ranks = {feature: percentile_ranks(rows, feature) for feature in FEATURES}
    for idx, row in enumerate(rows):
        for feature in FEATURES:
            row[f"{feature}_rank"] = ranks[feature][idx]
    return rows


def uniform_indices(total_frames, gap):
    indices = list(range(0, int(total_frames), int(gap)))
    last = int(total_frames) - 1
    if indices[-1] != last:
        indices.append(last)
    return sorted(set(indices))


def metadata_bytes_for_schedule(total_frames_by_sequence, selected_rows, mode_bytes):
    by_sequence = defaultdict(set)
    for row in selected_rows:
        if int(row["selected_for_extra_keyframe"]):
            by_sequence[row["sequence"]].add(int(row["target_index"]))
    total_keyframes = 0
    total_gap8_keyframes = 0
    total_bits = 0
    for sequence, total_frames in sorted(total_frames_by_sequence.items()):
        gap8 = uniform_indices(total_frames, 8)
        adaptive = sorted(set(gap8) | by_sequence.get(sequence, set()))
        bits_per_index = math.ceil(math.log2(max(int(total_frames), 2)))
        total_bits += int(mode_bytes) * 8 + len(adaptive) * bits_per_index
        total_keyframes += len(adaptive)
        total_gap8_keyframes += len(gap8)
    return total_keyframes, total_keyframes - total_gap8_keyframes, math.ceil(total_bits / 8.0)


def evaluate_variant(rows, features, threshold, min_votes, total_frames_by_sequence, mode_bytes):
    selected_rows = []
    detailed_rows = []
    for row in rows:
        votes = sum(int(float(row[f"{feature}_rank"]) >= float(threshold)) for feature in features)
        score = votes + mean(row[f"{feature}_rank"] for feature in features)
        selected = int(votes >= int(min_votes))
        item = dict(row)
        item["vote_count"] = votes
        item["selector_score"] = score
        item["selected_for_extra_keyframe"] = selected
        detailed_rows.append(item)
        if selected:
            selected_rows.append(item)
    unselected_rows = [row for row in detailed_rows if not int(row["selected_for_extra_keyframe"])]
    hard_total = sum(int(row["hard_quality_label"]) for row in detailed_rows)
    payload_total = sum(int(row["high_payload_label"]) for row in detailed_rows)
    hard_selected = sum(int(row["hard_quality_label"]) for row in selected_rows)
    payload_selected = sum(int(row["high_payload_label"]) for row in selected_rows)
    precision_hard = hard_selected / len(selected_rows) if selected_rows else 0.0
    recall_hard = hard_selected / hard_total if hard_total else 0.0
    precision_payload = payload_selected / len(selected_rows) if selected_rows else 0.0
    recall_payload = payload_selected / payload_total if payload_total else 0.0
    keyframe_count, extra_keyframes, metadata_bytes = metadata_bytes_for_schedule(total_frames_by_sequence, detailed_rows, mode_bytes)
    return {
        "selected_rows": detailed_rows,
        "summary": {
            "feature_count": len(features),
            "features": " ".join(features),
            "rank_threshold": float(threshold),
            "min_votes": int(min_votes),
            "selected_count": len(selected_rows),
            "keyframe_count": keyframe_count,
            "extra_keyframes_vs_gap8": extra_keyframes,
            "metadata_bytes": metadata_bytes,
            "hard_quality_count": hard_total,
            "selected_hard_count": hard_selected,
            "precision_hard": precision_hard,
            "recall_hard": recall_hard,
            "f1_hard": f1(precision_hard, recall_hard),
            "high_payload_count": payload_total,
            "selected_high_payload_count": payload_selected,
            "precision_payload": precision_payload,
            "recall_payload": recall_payload,
            "f1_payload": f1(precision_payload, recall_payload),
            "mean_selected_psnr": mean(row["stage158_psnr"] for row in selected_rows),
            "mean_unselected_psnr": mean(row["stage158_psnr"] for row in unselected_rows),
            "mean_selected_lpips": mean(row["stage158_lpips"] for row in selected_rows),
            "mean_unselected_lpips": mean(row["stage158_lpips"] for row in unselected_rows),
            "mean_selected_payload_bytes": mean(row["payload_bytes"] for row in selected_rows),
            "mean_unselected_payload_bytes": mean(row["payload_bytes"] for row in unselected_rows),
        },
    }


def build_context(stage186_rows, stage165_package):
    adaptive = next(row for row in stage186_rows if row["schedule"] == "stage165_adaptive")
    gap8 = next(row for row in stage186_rows if row["schedule"] == "uniform_gap8")
    gap4 = next(row for row in stage186_rows if row["schedule"] == "uniform_gap4")
    return [
        {"item": "stage165_policy", "value": stage165_package["selected_policy"]["candidate"], "note": "Fixed gate used for ablations."},
        {"item": "stage165_selected_count", "value": stage165_package["selected_policy"]["selected_count"], "note": "Original selected row count on Stage163 rows."},
        {"item": "stage186_adaptive_rate", "value": adaptive["total_mib_per_frame"], "note": "Measured full-sequence rate for current adaptive schedule."},
        {"item": "stage186_adaptive_psnr_delta_vs_gap8", "value": adaptive["delta_psnr_vs_gap8"], "note": "Full-sequence quality context; not recomputed for ablation schedules."},
        {"item": "stage186_gap8_rate", "value": gap8["total_mib_per_frame"], "note": "Measured fixed gap8 rate."},
        {"item": "stage186_gap4_rate", "value": gap4["total_mib_per_frame"], "note": "Measured fixed gap4 rate."},
    ]


def write_report(ablation_rows, context_rows, package, path):
    best_low_budget = package["recommended_low_budget_variant"]
    shortlist = package["stage188_low_budget_shortlist"]
    lines = [
        "# Stage187 Selector Feature Ablation Validation",
        "",
        "## Scope",
        "",
        "This is a selector-label/protocol ablation over Stage163 rows. It does not claim measured full-sequence RD for ablation schedules.",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Recommended conservative Stage188 low-budget candidate: `{best_low_budget}`.",
        "",
        "## Context",
        "",
    ]
    for row in context_rows:
        lines.append(f"- `{row['item']}`: `{row['value']}`. {row['note']}")
    lines.extend([
        "",
        "## Feature Ablation Table",
        "",
        "| variant | features | selected | keyframes | hard recall | payload recall | hard precision | payload precision | selected payload | delta selected vs full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in ablation_rows:
        lines.append(
            f"| {row['variant']} | {row['feature_count']} | {row['selected_count']} | {row['keyframe_count']} | "
            f"{float(row['recall_hard']):.6f} | {float(row['recall_payload']):.6f} | {float(row['precision_hard']):.6f} | {float(row['precision_payload']):.6f} | "
            f"{float(row['mean_selected_payload_bytes'] or 0.0):.3f} | {row['selected_count_delta_vs_full']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The five-feature gate remains the highest-recall selector among the evaluated variants.",
        "- Lower-feature variants reduce selected rows and keyframes, which may help Stage188 reduce measured rate, but recall drops.",
        "- Stage188 should evaluate budget/threshold variants as explicit RD points, prioritizing variants with fewer selected rows but acceptable hard/high-payload recall.",
        "",
        "## Stage188 Low-Budget Shortlist",
        "",
        "| variant | selected | delta selected | hard recall | payload recall | note |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in shortlist:
        lines.append(
            f"| {row['variant']} | {row['selected_count']} | {row['selected_count_delta_vs_full']} | "
            f"{float(row['recall_hard']):.6f} | {float(row['recall_payload']):.6f} | {row['stage188_note']} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Ablation summary CSV: `{package['ablation_csv']}`",
        f"- Ablation row dump CSV: `{package['selected_rows_csv']}`",
        f"- Context CSV: `{package['context_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage163_rows", type=Path, default=DEFAULT_STAGE163_ROWS)
    parser.add_argument("--stage165_package", type=Path, default=DEFAULT_STAGE165_PACKAGE)
    parser.add_argument("--task_manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--stage186_rd_quality", type=Path, default=DEFAULT_STAGE186_RD_QUALITY)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mode_bytes", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage165_package = read_json(args.stage165_package)
    threshold = float(stage165_package["selected_policy"]["rank_threshold"])
    min_votes = int(stage165_package["selected_policy"]["min_votes"])
    rows = prepare_rows(read_csv(args.stage163_rows))
    total_frames_by_sequence = parse_task_frames(args.task_manifest)
    ablation_rows = []
    selected_out = []
    full_summary = None
    for variant, features, description in VARIANTS:
        result = evaluate_variant(rows, features, threshold, min_votes, total_frames_by_sequence, args.mode_bytes)
        summary = result["summary"]
        summary["variant"] = variant
        summary["description"] = description
        if variant == "full_stage165_features":
            full_summary = summary
        ablation_rows.append(summary)
        for selected in result["selected_rows"]:
            selected_out.append({
                "variant": variant,
                "task_id": selected["task_id"],
                "sequence": selected["sequence"],
                "target_index": selected["target_index"],
                "vote_count": selected["vote_count"],
                "selector_score": selected["selector_score"],
                "selected_for_extra_keyframe": selected["selected_for_extra_keyframe"],
                "hard_quality_label": selected["hard_quality_label"],
                "high_payload_label": selected["high_payload_label"],
                "stage158_psnr": selected["stage158_psnr"],
                "stage158_lpips": selected["stage158_lpips"],
                "payload_bytes": selected["payload_bytes"],
            })
    for row in ablation_rows:
        row["selected_count_delta_vs_full"] = int(row["selected_count"]) - int(full_summary["selected_count"])
        row["recall_hard_delta_vs_full"] = float(row["recall_hard"]) - float(full_summary["recall_hard"])
        row["recall_payload_delta_vs_full"] = float(row["recall_payload"]) - float(full_summary["recall_payload"])
    ablation_rows.sort(key=lambda row: (int(row["selected_count"]), -float(row["recall_payload"]), -float(row["recall_hard"])))
    candidate_pool = [row for row in ablation_rows if row["variant"] != "full_stage165_features" and int(row["selected_count"]) < int(full_summary["selected_count"])]
    recommended = sorted(candidate_pool, key=lambda row: (float(row["recall_payload"]), float(row["recall_hard"]), -int(row["selected_count"])), reverse=True)[0]
    shortlist = []
    for variant, note in [
        ("drop_interp_rgb", "conservative recall-preserving feature ablation"),
        ("motion_proxy_edge_hist", "small budget reduction with no hard-recall loss"),
        ("edge_hist_only", "two-feature motion-only stress point"),
        ("drop_hist_motion", "larger budget reduction with no hard-recall loss but lower payload recall"),
        ("drop_edge_motion", "larger selected-count reduction; expected quality-risk stress point"),
    ]:
        row = next(item for item in ablation_rows if item["variant"] == variant)
        shortlist_row = dict(row)
        shortlist_row["stage188_note"] = note
        shortlist.append(shortlist_row)
    context_rows = build_context(read_csv(args.stage186_rd_quality), stage165_package)
    ablation_csv = args.output_root / "stage187_selector_feature_ablation_summary.csv"
    selected_rows_csv = args.output_root / "stage187_selector_feature_ablation_rows.csv"
    context_csv = args.output_root / "stage187_selector_feature_ablation_context.csv"
    package_json = args.output_root / "stage187_selector_feature_ablation_validation_package.json"
    report_md = args.output_root / "stage187_selector_feature_ablation_validation_report.md"
    write_csv(ablation_rows, ablation_csv, ABLATION_FIELDS)
    write_csv(selected_out, selected_rows_csv, SELECTED_FIELDS)
    write_csv(context_rows, context_csv, CONTEXT_FIELDS)
    package = {
        "stage": 187,
        "status": "selector_feature_ablation_validation_packaged",
        "decision": "feature_ablation_ready_for_budget_sensitivity",
        "scope": "stage163_label_protocol_ablation_not_full_rd",
        "threshold": threshold,
        "min_votes": min_votes,
        "full_variant": full_summary,
        "recommended_low_budget_variant": recommended["variant"],
        "recommended_low_budget_summary": recommended,
        "stage188_low_budget_shortlist": shortlist,
        "ablation_rows": ablation_rows,
        "context_rows": context_rows,
        "ablation_csv": str(ablation_csv),
        "selected_rows_csv": str(selected_rows_csv),
        "context_csv": str(context_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(ablation_rows, context_rows, package, report_md)
    print(json.dumps({
        "package": str(package_json),
        "decision": package["decision"],
        "recommended_low_budget_variant": package["recommended_low_budget_variant"],
        "recommended_selected_count": recommended["selected_count"],
        "recommended_recall_payload": recommended["recall_payload"],
        "recommended_recall_hard": recommended["recall_hard"],
        "stage188_low_budget_shortlist": [row["variant"] for row in shortlist],
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
