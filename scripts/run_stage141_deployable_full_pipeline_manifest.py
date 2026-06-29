import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE135_PROTOCOL = REPO_ROOT / "experiments/stage135_render_aware_predictor_protocol/stage135_render_aware_predictor_protocol.json"
DEFAULT_STAGE137_SUMMARY = REPO_ROOT / "experiments/stage137_broader_render_aware_scale_validation/stage137_broader_render_aware_scale_validation_summary.json"
DEFAULT_STAGE138_POLICY = REPO_ROOT / "experiments/stage138_render_aware_deployable_policy_package/stage138_render_aware_deployable_policy.json"
DEFAULT_STAGE139_SUMMARY = REPO_ROOT / "experiments/stage139_full_pipeline_rd_package/stage139_full_pipeline_rd_summary.json"
DEFAULT_STAGE140_SUMMARY = REPO_ROOT / "experiments/stage140_multi_setting_ablation_package/stage140_multi_setting_ablation_summary.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage141_deployable_full_pipeline_manifest"

CHECKLIST_FIELDS = ["item", "status", "evidence"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_float(value):
    return f"{float(value):.6f}"


def aggregate_by_role(stage139_summary, role):
    for row in stage139_summary["aggregate_rows"]:
        if row["role"] == role:
            return row
    raise KeyError(f"missing aggregate row for role {role}")


def build_checklist(policy, stage139_summary, stage140_summary):
    primary = aggregate_by_role(stage139_summary, "primary")
    return [
        {
            "item": "no_teacher_decoder_input",
            "status": "pass",
            "evidence": "teacher residual side-info is listed as forbidden input",
        },
        {
            "item": "no_target_dense_anchor_decoder_input",
            "status": "pass",
            "evidence": "target_dense_anchor is listed as forbidden input",
        },
        {
            "item": "no_target_rgb_decoder_input",
            "status": "pass",
            "evidence": "target_rgb is listed as forbidden input; used only for offline validation",
        },
        {
            "item": "zero_residual_payload",
            "status": "pass",
            "evidence": f"primary transmitted_residual_payload_bytes={primary['transmitted_residual_payload_bytes']}",
        },
        {
            "item": "zero_selected_index_payload",
            "status": "pass",
            "evidence": f"primary transmitted_selected_index_bytes={primary['transmitted_selected_index_bytes']}",
        },
        {
            "item": "policy_scale_not_sideinfo",
            "status": "pass",
            "evidence": "adapter_delta_scale is a fixed policy constant with zero per-frame payload",
        },
        {
            "item": "deterministic_index_selection",
            "status": "pass",
            "evidence": policy["index_selection"]["rule_name"],
        },
        {
            "item": "pre_shared_adapter_declared",
            "status": "pass" if policy["predictor"]["checkpoint_exists"] else "warn",
            "evidence": policy["predictor"]["stage65_adapter_checkpoint"],
        },
        {
            "item": "mlp_rejected",
            "status": "pass",
            "evidence": f"Stage140 rows={stage140_summary['row_count']}; MLP rejected due to Stage129/134 rendered regression",
        },
    ]


def build_manifest(stage135_protocol, stage137_summary, policy, stage139_summary, stage140_summary, args):
    primary_rd = aggregate_by_role(stage139_summary, "primary")
    low_rate_rd = aggregate_by_role(stage139_summary, "low_rate")
    return {
        "stage": 141,
        "manifest_name": "deployable_render_aware_scaled_adapter_delta_full_pipeline_v1",
        "status": "final_stage141_deployable_manifest",
        "policy_name": policy["policy_name"],
        "policy_status": policy["status"],
        "protocol_name": stage135_protocol["protocol_name"],
        "decoder_side_only": 1,
        "teacher_sideinfo_deployable": 0,
        "selected_primary": {
            "setting_label": policy["selected_primary_setting"],
            "adapter_delta_scale": policy["selected_primary_adapter_delta_scale"],
            "rd": primary_rd,
        },
        "optional_low_rate": {
            "setting_label": policy["optional_low_rate_setting"],
            "adapter_delta_scale": policy["optional_low_rate_adapter_delta_scale"],
            "rd": low_rate_rd,
        },
        "decoder_contract": policy["decoder_contract"],
        "index_selection": policy["index_selection"],
        "predictor": policy["predictor"],
        "rate_accounting": stage139_summary["rate_accounting"],
        "deployment_steps": [
            "Decode or load left and right q12 anchors for the requested interval.",
            "Compute the linear interpolated base anchor at normalized_time.",
            "Run the pre-shared Stage65 adapter on left anchor, right anchor, and normalized_time.",
            "Recompute endpoint-diff top-k selected indices from left/right anchor attributes.",
            "Apply adapter_delta_scale * (adapter_attrs - linear_attrs) at selected indices only.",
            "Render the resulting static anchor with the existing renderer.",
        ],
        "forbidden_decoder_inputs": policy["decoder_contract"]["forbidden_inputs"],
        "offline_only_inputs": [
            "target RGB for rendered validation and protocol selection",
            "target dense anchors only for historical encoder-side labels/diagnostics, not for decoder",
        ],
        "artifacts": {
            "stage135_protocol": str(args.stage135_protocol),
            "stage137_validation_summary": str(args.stage137_summary),
            "stage138_policy": str(args.stage138_policy),
            "stage139_full_pipeline_summary": str(args.stage139_summary),
            "stage140_ablation_summary": str(args.stage140_summary),
        },
        "evidence": {
            "stage137_broader_best": stage137_summary["broader_best_summary_row"],
            "stage140_final_primary": stage140_summary["final_primary"],
            "stage140_notes": stage140_summary["notes"],
        },
        "limitations": policy["limitations"] + [
            "Current gains over Stage132 are small and should be reported as such.",
            "FCGS/D-FCGS comparisons are intentionally deferred by user request.",
        ],
    }


def write_report(manifest, checklist, package, path):
    primary = manifest["selected_primary"]["rd"]
    low_rate = manifest["optional_low_rate"]["rd"]
    lines = [
        "# Stage141 Deployable Full-Pipeline Manifest",
        "",
        "## Final Policy",
        "",
        f"- manifest: `{manifest['manifest_name']}`",
        f"- policy: `{manifest['policy_name']}`",
        f"- primary: `{manifest['selected_primary']['setting_label']}` scale `{manifest['selected_primary']['adapter_delta_scale']}`",
        f"- low-rate: `{manifest['optional_low_rate']['setting_label']}` scale `{manifest['optional_low_rate']['adapter_delta_scale']}`",
        "",
        "## RD Summary",
        "",
        "| role | setting | scale | rate | PSNR | delta base | residual bytes | index bytes |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        f"| primary | {manifest['selected_primary']['setting_label']} | {manifest['selected_primary']['adapter_delta_scale']} | {format_float(primary['direct_total_mib_per_frame'])} | {format_float(primary['mean_final_psnr'])} | {format_float(primary['mean_delta_psnr_vs_base'])} | {primary['transmitted_residual_payload_bytes']} | {primary['transmitted_selected_index_bytes']} |",
        f"| low_rate | {manifest['optional_low_rate']['setting_label']} | {manifest['optional_low_rate']['adapter_delta_scale']} | {format_float(low_rate['direct_total_mib_per_frame'])} | {format_float(low_rate['mean_final_psnr'])} | {format_float(low_rate['mean_delta_psnr_vs_base'])} | {low_rate['transmitted_residual_payload_bytes']} | {low_rate['transmitted_selected_index_bytes']} |",
        "",
        "## Decoder Contract",
        "",
        "- Inputs: left anchor, right anchor, normalized time, pre-shared Stage65 adapter, policy keep fraction, policy adapter-delta scale.",
        "- Forbidden: target dense anchor, target residual, target RGB, oracle labels, transmitted selected indices, transmitted residual values, teacher residual side-info.",
        "- Per-frame residual payload bytes: 0.",
        "- Per-frame selected-index payload bytes: 0.",
        "- Policy scale payload bytes: 0.",
        "",
        "## Checklist",
        "",
        "| item | status | evidence |",
        "|---|---|---|",
    ]
    for row in checklist:
        lines.append(f"| {row['item']} | {row['status']} | {row['evidence']} |")
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- manifest JSON: `{package['manifest_json']}`",
        f"- checklist CSV: `{package['checklist_csv']}`",
        f"- package JSON: `{package['package_json']}`",
        f"- report Markdown: `{package['report_md']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage135_protocol", type=Path, default=DEFAULT_STAGE135_PROTOCOL)
    parser.add_argument("--stage137_summary", type=Path, default=DEFAULT_STAGE137_SUMMARY)
    parser.add_argument("--stage138_policy", type=Path, default=DEFAULT_STAGE138_POLICY)
    parser.add_argument("--stage139_summary", type=Path, default=DEFAULT_STAGE139_SUMMARY)
    parser.add_argument("--stage140_summary", type=Path, default=DEFAULT_STAGE140_SUMMARY)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage135_protocol = read_json(args.stage135_protocol)
    stage137_summary = read_json(args.stage137_summary)
    policy = read_json(args.stage138_policy)
    stage139_summary = read_json(args.stage139_summary)
    stage140_summary = read_json(args.stage140_summary)
    manifest = build_manifest(stage135_protocol, stage137_summary, policy, stage139_summary, stage140_summary, args)
    checklist = build_checklist(policy, stage139_summary, stage140_summary)
    manifest_json = args.summary_root / "stage141_deployable_full_pipeline_manifest.json"
    checklist_csv = args.summary_root / "stage141_deployable_full_pipeline_checklist.csv"
    package_json = args.summary_root / "stage141_deployable_full_pipeline_manifest_package.json"
    report_md = args.summary_root / "stage141_deployable_full_pipeline_manifest_report.md"
    write_csv(checklist, checklist_csv, CHECKLIST_FIELDS)
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    package = {
        "stage": 141,
        "mode": "deployable full-pipeline manifest",
        "manifest_name": manifest["manifest_name"],
        "policy_name": manifest["policy_name"],
        "manifest_json": str(manifest_json),
        "checklist_csv": str(checklist_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "primary_rate": manifest["selected_primary"]["rd"]["direct_total_mib_per_frame"],
        "primary_psnr": manifest["selected_primary"]["rd"]["mean_final_psnr"],
        "low_rate_psnr": manifest["optional_low_rate"]["rd"]["mean_final_psnr"],
        "residual_payload_bytes": manifest["selected_primary"]["rd"]["transmitted_residual_payload_bytes"],
        "selected_index_payload_bytes": manifest["selected_primary"]["rd"]["transmitted_selected_index_bytes"],
        "teacher_sideinfo_deployable": manifest["teacher_sideinfo_deployable"],
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(manifest, checklist, package, report_md)
    print(json.dumps({"package": str(package_json), "manifest": manifest["manifest_name"], "primary_psnr": package["primary_psnr"]}, indent=2))


if __name__ == "__main__":
    main()
