import csv
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "experiments/stage203_gs_latent_residual_codec_design"
STAGE202_PACKAGE = REPO_ROOT / "experiments/stage202_predictor_only_broader_validation/stage202_predictor_only_broader_validation_package.json"

sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.residual_sideinfo_codec import (  # noqa: E402
    decode_residual_sideinfo_entropy,
    decode_selected_residual_values_sideinfo_entropy,
    encode_selected_residual_values_sideinfo_entropy,
    encode_topk_residual_sideinfo_entropy,
)


CANDIDATE_FIELDS = [
    "codec",
    "status",
    "encoder_inputs",
    "decoder_inputs",
    "payload_contents",
    "rate_accounting",
    "reason",
    "next_stage_action",
]
ROUNDTRIP_FIELDS = [
    "codec",
    "status",
    "payload_bytes",
    "raw_reference_bytes",
    "residual_mse_before",
    "residual_mse_after",
    "mse_reduction",
    "detail",
]
RULE_FIELDS = ["rule", "value", "rationale"]
AUDIT_FIELDS = ["audit", "status", "value", "requirement", "detail"]
PROTOCOL_FIELDS = ["item", "value"]


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def toy_attrs(seed=203, count=128, attr_dim=13):
    torch.manual_seed(seed)
    base = torch.randn(1, count, attr_dim) * 0.25
    noise = torch.randn_like(base) * 0.03
    hot = torch.arange(0, count, 11)
    noise[:, hot, :] += torch.randn(1, int(hot.numel()), attr_dim) * 0.25
    target = base + noise
    return base, target


def mse(a, b):
    return float(torch.mean((a.float() - b.float()) ** 2).item())


def roundtrip_topk():
    base, target = toy_attrs()
    payload, info = encode_topk_residual_sideinfo_entropy(base, target, keep_fraction=0.1, side_bits=6, zlib_level=9)
    decoded = decode_residual_sideinfo_entropy(base, payload)
    before = mse(base, target)
    after = mse(decoded, target)
    return {
        "codec": "gs_attr_topk_residual_entropy_v1",
        "status": "pass" if len(payload) == int(info["payload_bytes"]) and after < before else "fail",
        "payload_bytes": len(payload),
        "raw_reference_bytes": info["fixed_payload_bytes"],
        "residual_mse_before": before,
        "residual_mse_after": after,
        "mse_reduction": 1.0 - after / before if before > 0.0 else 0.0,
        "detail": f"keep_count={info['keep_count']}; side_bits={info['side_bits']}; delta_zlib_bytes={info['delta_zlib_bytes']}; residual_zlib_bytes={info['residual_zlib_bytes']}",
    }


def roundtrip_deterministic():
    base, target = toy_attrs(seed=204)
    diff_energy = torch.sum((target - base)[0] ** 2, dim=-1)
    selected = torch.sort(torch.topk(diff_energy, k=12, largest=True).indices).values
    payload, info = encode_selected_residual_values_sideinfo_entropy(base, target, selected, side_bits=6, zlib_level=9)
    decoded = decode_selected_residual_values_sideinfo_entropy(base, payload, selected)
    before = mse(base, target)
    after = mse(decoded, target)
    return {
        "codec": "gs_attr_deterministic_index_residual_entropy_v1",
        "status": "pass" if len(payload) == int(info["payload_bytes"]) and after < before else "fail",
        "payload_bytes": len(payload),
        "raw_reference_bytes": info["raw_deterministic_payload_bytes"],
        "residual_mse_before": before,
        "residual_mse_after": after,
        "mse_reduction": 1.0 - after / before if before > 0.0 else 0.0,
        "detail": f"keep_count={info['keep_count']}; side_bits={info['side_bits']}; index_bytes={info['index_bytes']}; residual_zlib_bytes={info['residual_zlib_bytes']}",
    }


def candidate_rows():
    return [
        {
            "codec": "gs_attr_topk_residual_entropy_v1",
            "status": "selected_primary_for_stage204",
            "encoder_inputs": "predictor_base_gs;target_dense_anchor;optional_target_rgb_for_quality_diagnostics",
            "decoder_inputs": "predictor_base_gs;transmitted_residual_payload",
            "payload_contents": "header;per-attribute fp16 min/max metadata;sorted index deltas;quantized residual values;zlib component lengths",
            "rate_accounting": "payload_bytes=len(payload); include every residual payload byte in total RD",
            "reason": "Highest practical headroom because encoder can select target-residual top-k GS attributes while payload remains GS-native and decode-capable.",
            "next_stage_action": "Stage204 smoke on real Stage199 tasks with q6 keep sweeps and rendered metrics.",
        },
        {
            "codec": "gs_attr_deterministic_index_residual_entropy_v1",
            "status": "low_rate_ablation",
            "encoder_inputs": "predictor_base_gs;target_dense_anchor_for_residual_values;decoder-reproducible selected indices",
            "decoder_inputs": "predictor_base_gs;transmitted_residual_value_payload;decoder-recomputed selected indices",
            "payload_contents": "header;per-attribute fp16 min/max metadata;quantized residual values;zlib component lengths;no indices",
            "rate_accounting": "payload_bytes=len(payload); omitted index bytes must be reported as saved bytes not hidden side-info",
            "reason": "Lower rate because selected indices are deterministic, but likely lower quality if index rule misses target residual energy.",
            "next_stage_action": "Use as Stage204/205 low-rate ablation after primary top-k smoke.",
        },
        {
            "codec": "learned_gs_latent_residual_v1",
            "status": "deferred",
            "encoder_inputs": "predictor_base_gs;target_dense_anchor;target_rgb_render_loss",
            "decoder_inputs": "predictor_base_gs;transmitted_latent_bitstream;shared_entropy_model_weights",
            "payload_contents": "entropy-coded latent tokens and metadata",
            "rate_accounting": "all latent bytes counted; entropy model weights declared shared method parameters",
            "reason": "Potentially better RD but requires training after Stage204 verifies residual headroom.",
            "next_stage_action": "Revisit after fixed residual codec smoke shows enough quality headroom.",
        },
        {
            "codec": "rgb_image_residual_postprocess",
            "status": "rejected_final_method",
            "encoder_inputs": "target RGB/image residual",
            "decoder_inputs": "image residual or target RGB-like correction",
            "payload_contents": "image-domain correction",
            "rate_accounting": "would need counting but violates user-approved final route",
            "reason": "User and Stage197 rejected RGB/image residual post-processing as final method.",
            "next_stage_action": "Do not use as final method; at most historical upper-bound context.",
        },
    ]


def rule_rows():
    return [
        {
            "rule": "payload_bytes",
            "value": "len(payload)",
            "rationale": "Header, metadata, indices/deltas, quantized residual values, and compressed component lengths are all transmitted bytes.",
        },
        {
            "rule": "total_rate",
            "value": "keyframe_bytes + schedule_metadata_bytes + residual_payload_bytes + optional_latent_payload_bytes",
            "rationale": "Stage197 requires every transmitted side-info byte to be included in RD.",
        },
        {
            "rule": "decoder_base",
            "value": "decoded q-keyframe GS plus predictor/refiner output",
            "rationale": "Target dense anchors are encoder-side labels/residual sources only.",
        },
        {
            "rule": "forbidden_payload",
            "value": "RGB/image residual post-processing",
            "rationale": "Final method must remain GS-native compression.",
        },
        {
            "rule": "stage204_primary_setting",
            "value": "q6 keep_fraction sweep 0.05 0.10 0.20 with gs_attr_topk_residual_entropy_v1",
            "rationale": "Stage158 used q6/keep1.0 successfully; Stage204 should test lower-rate GS-native variants first.",
        },
    ]


def protocol_rows():
    return [
        {"item": "stage204_base", "value": "linear_or_zero_init_TemporalBasisGSRefiner predictor base from Stage201/202"},
        {"item": "stage204_task_manifest", "value": "experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_tasks.csv"},
        {"item": "stage204_gaps", "value": "4 8"},
        {"item": "stage204_keyframe_codec", "value": "q12"},
        {"item": "stage204_primary_codec", "value": "gs_attr_topk_residual_entropy_v1"},
        {"item": "stage204_settings", "value": "side_bits=6; keep_fraction=0.05,0.10,0.20; zlib_level=9"},
        {"item": "stage204_metrics", "value": "rendered PSNR plus payload_bytes and MSE reduction; discard any shape-mismatched metrics"},
        {"item": "stage204_decoder_inputs", "value": "predictor_base_gs plus transmitted counted GS residual payload only"},
    ]


def audit_rows(stage202, roundtrips):
    statuses = {row["codec"]: row["status"] for row in roundtrips}
    return [
        {
            "audit": "stage202_predictor_headroom_context",
            "status": "pass" if stage202["decision"] == "predictor_only_broader_training_headroom_not_observed" else "review",
            "value": stage202["best_delta_psnr_vs_linear"],
            "requirement": "residual codec prioritized after predictor-only headroom check",
            "detail": stage202["decision"],
        },
        {
            "audit": "primary_codec_roundtrip",
            "status": statuses.get("gs_attr_topk_residual_entropy_v1", "missing"),
            "value": next(row["mse_reduction"] for row in roundtrips if row["codec"] == "gs_attr_topk_residual_entropy_v1"),
            "requirement": "primary payload decodes and reduces toy residual MSE",
            "detail": "encode_topk_residual_sideinfo_entropy/decode_residual_sideinfo_entropy",
        },
        {
            "audit": "deterministic_codec_roundtrip",
            "status": statuses.get("gs_attr_deterministic_index_residual_entropy_v1", "missing"),
            "value": next(row["mse_reduction"] for row in roundtrips if row["codec"] == "gs_attr_deterministic_index_residual_entropy_v1"),
            "requirement": "low-rate deterministic payload decodes and reduces toy residual MSE",
            "detail": "indices are decoder-known and not hidden side-info",
        },
        {
            "audit": "stage197_decoder_contract",
            "status": "pass",
            "value": 0,
            "requirement": "decoder excludes target dense anchors and target RGB/image residuals",
            "detail": "decoder uses predictor base GS plus transmitted counted GS-native residual payload",
        },
        {
            "audit": "image_residual_rejected",
            "status": "pass",
            "value": 0,
            "requirement": "final method remains GS-native",
            "detail": "RGB/image residual appears only as rejected candidate",
        },
    ]


def write_report(package, candidates, roundtrips, audits, protocol, path):
    lines = [
        "# Stage203 GS Latent/Residual Codec Design",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Primary codec: `{package['primary_codec']}`.",
        "- Payload bytes are counted with `len(payload)`.",
        "",
        "## Candidates",
        "",
        "| codec | status | reason | next |",
        "|---|---|---|---|",
    ]
    for row in candidates:
        lines.append(f"| {row['codec']} | {row['status']} | {row['reason']} | {row['next_stage_action']} |")
    lines.extend([
        "",
        "## Toy Roundtrips",
        "",
        "| codec | status | payload bytes | MSE before | MSE after | reduction |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in roundtrips:
        lines.append(
            f"| {row['codec']} | {row['status']} | {row['payload_bytes']} | {float(row['residual_mse_before']):.8f} | {float(row['residual_mse_after']):.8f} | {float(row['mse_reduction']):.6f} |"
        )
    lines.extend([
        "",
        "## Audit",
        "",
        "| audit | status | value | detail |",
        "|---|---|---:|---|",
    ])
    for row in audits:
        lines.append(f"| {row['audit']} | {row['status']} | {row['value']} | {row['detail']} |")
    lines.extend([
        "",
        "## Stage204 Protocol",
        "",
        "| item | value |",
        "|---|---|",
    ])
    for row in protocol:
        lines.append(f"| {row['item']} | {row['value']} |")
    lines.extend([
        "",
        "## Decoder Contract",
        "",
        "- Encoder may use target dense anchors to form GS residual payloads.",
        "- Decoder uses predictor/base GS plus transmitted counted GS-native residual payloads.",
        "- Target dense anchors, target RGB/image residuals, and oracle quality labels are not decoder inputs.",
        "",
        "## Outputs",
        "",
        f"- candidates: `{package['candidates_csv']}`",
        f"- roundtrips: `{package['roundtrip_csv']}`",
        f"- rate rules: `{package['rate_rules_csv']}`",
        f"- audit: `{package['audit_csv']}`",
        f"- Stage204 protocol: `{package['protocol_csv']}`",
        f"- primary contract: `{package['contract_json']}`",
        f"- package: `{package['package_json']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stage202 = read_json(STAGE202_PACKAGE)
    candidates = candidate_rows()
    rules = rule_rows()
    protocol = protocol_rows()
    roundtrips = [roundtrip_topk(), roundtrip_deterministic()]
    audits = audit_rows(stage202, roundtrips)
    decision = "gs_attr_topk_residual_entropy_v1_selected_for_stage204_smoke"
    if any(row["status"] not in {"pass"} for row in audits):
        decision = "gs_residual_codec_design_needs_review"

    candidates_csv = OUTPUT_ROOT / "stage203_codec_candidates.csv"
    roundtrip_csv = OUTPUT_ROOT / "stage203_codec_toy_roundtrips.csv"
    rate_rules_csv = OUTPUT_ROOT / "stage203_rate_accounting_rules.csv"
    audit_csv = OUTPUT_ROOT / "stage203_decoder_contract_audit.csv"
    protocol_csv = OUTPUT_ROOT / "stage203_stage204_smoke_protocol.csv"
    contract_json = OUTPUT_ROOT / "stage203_primary_codec_contract.json"
    package_json = OUTPUT_ROOT / "stage203_gs_latent_residual_codec_design_package.json"
    report_md = OUTPUT_ROOT / "stage203_gs_latent_residual_codec_design_report.md"

    contract = {
        "primary_codec": "gs_attr_topk_residual_entropy_v1",
        "implementation": "mono_dfcgs.residual_sideinfo_codec.encode_topk_residual_sideinfo_entropy",
        "decoder": "mono_dfcgs.residual_sideinfo_codec.decode_residual_sideinfo_entropy",
        "encoder_side_sources": ["predictor_base_gs", "target_dense_anchor", "optional target RGB diagnostics"],
        "decoder_inputs": ["predictor_base_gs", "transmitted_residual_payload"],
        "forbidden_decoder_inputs": ["target_dense_anchor", "target_rgb_or_image_residual", "oracle_quality_labels"],
        "payload_counting": "payload_bytes = len(payload); include in total RD",
        "stage204_default_settings": {"side_bits": 6, "keep_fraction": [0.05, 0.10, 0.20], "zlib_level": 9},
    }
    package = {
        "stage": 203,
        "name": "gs_latent_residual_codec_design",
        "decision": decision,
        "primary_codec": contract["primary_codec"],
        "stage202_package": str(STAGE202_PACKAGE),
        "candidates_csv": str(candidates_csv),
        "roundtrip_csv": str(roundtrip_csv),
        "rate_rules_csv": str(rate_rules_csv),
        "audit_csv": str(audit_csv),
        "protocol_csv": str(protocol_csv),
        "contract_json": str(contract_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "roundtrip_rows": roundtrips,
        "audit_rows": audits,
    }

    write_csv(candidates, candidates_csv, CANDIDATE_FIELDS)
    write_csv(roundtrips, roundtrip_csv, ROUNDTRIP_FIELDS)
    write_csv(rules, rate_rules_csv, RULE_FIELDS)
    write_csv(audits, audit_csv, AUDIT_FIELDS)
    write_csv(protocol, protocol_csv, PROTOCOL_FIELDS)
    contract_json.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, candidates, roundtrips, audits, protocol, report_md)
    print(json.dumps({"package": str(package_json), "decision": decision, "primary_codec": contract["primary_codec"]}, indent=2))


if __name__ == "__main__":
    main()
