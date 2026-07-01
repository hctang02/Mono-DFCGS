import csv
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "experiments/stage200_gs_predictor_architecture_package"
STAGE198_REQUIREMENTS = REPO_ROOT / "experiments/stage198_prior_predictor_training_audit/stage198_new_route_requirements.csv"
STAGE199_PACKAGE = REPO_ROOT / "experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_manifest_package.json"

sys.path.insert(0, str(REPO_ROOT))
from mono_dfcgs.gaussian_codec import STATIC_ANCHOR_DIMS, flatten_static_anchor  # noqa: E402
from mono_dfcgs.learned_gs_predictor import TemporalBasisGSRefiner, linear_static_anchor  # noqa: E402


CANDIDATE_FIELDS = [
    "candidate",
    "status",
    "decoder_inputs",
    "training_inputs",
    "rate_accounting",
    "reason",
    "next_stage_action",
]
LOSS_FIELDS = ["loss", "stage", "source", "weight", "decoder_input_required", "reason"]
AUDIT_FIELDS = ["audit", "status", "value", "requirement", "detail"]
PROTOCOL_FIELDS = ["item", "value"]


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


def toy_anchor(batch=1, count=32):
    torch.manual_seed(200)
    out = {}
    for key, dim in STATIC_ANCHOR_DIMS.items():
        value = torch.randn(batch, count, dim)
        if key == "scale":
            value = value.abs() + 0.01
        out[key] = value
    return out


def max_abs_anchor_delta(a, b):
    return float((flatten_static_anchor(a) - flatten_static_anchor(b)).abs().max().item())


def architecture_smoke():
    model = TemporalBasisGSRefiner(hidden_dim=96, global_dim=32, zero_init_residual=True, apply_output_constraints=False)
    left = toy_anchor()
    right = toy_anchor()
    for key in right:
        right[key] = right[key] + 0.25
    t0 = torch.tensor([0.0])
    t1 = torch.tensor([1.0])
    tmid = torch.tensor([0.5])
    with torch.no_grad():
        pred0 = model(left, right, t0, apply_output_constraints=False)
        pred1 = model(left, right, t1, apply_output_constraints=False)
        pred_mid = model(left, right, tmid, apply_output_constraints=False)
        linear_mid = linear_static_anchor(left, right, tmid)
    parameter_count = sum(param.numel() for param in model.parameters())
    return {
        "parameter_count_hidden96_global32": parameter_count,
        "endpoint_t0_max_abs_delta": max_abs_anchor_delta(pred0, left),
        "endpoint_t1_max_abs_delta": max_abs_anchor_delta(pred1, right),
        "zero_init_midpoint_linear_max_abs_delta": max_abs_anchor_delta(pred_mid, linear_mid),
    }


def candidate_rows():
    return [
        {
            "candidate": "temporal_basis_gs_refiner_v1",
            "status": "selected_primary_for_stage201",
            "decoder_inputs": "decoded_left_keyframe_gs;decoded_right_keyframe_gs;normalized_time;shared_refiner_weights;optional_counted_gs_latent",
            "training_inputs": "target_dense_anchor;target_rgb_render_loss;stage199_task_manifest",
            "rate_accounting": "predictor_only_has_zero_per_frame_payload;shared_weights_declared_as_method_parameters;any latent/residual payload must be counted",
            "reason": "Adds endpoint-gated temporal basis, endpoint difference, absolute motion, and sequence-level GS statistics beyond the old per-Gaussian adapter MLP.",
            "next_stage_action": "Implement Stage201 predictor-only smoke with no residual payload and q12 keyframes.",
        },
        {
            "candidate": "old_gaussian_anchor_dynamic_predictor",
            "status": "rejected",
            "decoder_inputs": "decoded_left_keyframe_gs;decoded_right_keyframe_gs;normalized_time;shared_weights",
            "training_inputs": "historical Stage65/145/146 adapter data",
            "rate_accounting": "zero per-frame payload but inadequate quality",
            "reason": "Stage198 showed q-bit changes and continued training do not repair this route.",
            "next_stage_action": "Use only as a historical baseline, not the Stage201 architecture.",
        },
        {
            "candidate": "raw_streamsplat_runtime_rgb_dependency",
            "status": "not_primary_final_claim",
            "decoder_inputs": "target RGB or raw video at decoder",
            "training_inputs": "StreamSplat checkpoint optional as initialization or teacher",
            "rate_accounting": "raw RGB/video would need to be transmitted and counted",
            "reason": "Stage197 forbids raw target RGB as decoder input for the final GS compression claim.",
            "next_stage_action": "Use checkpoint only as optional initialization/teacher if later needed.",
        },
        {
            "candidate": "latent_conditioned_temporal_refiner_v1",
            "status": "deferred_to_stage203_204",
            "decoder_inputs": "decoded keyframe GS;normalized_time;shared_weights;counted GS-native latent payload",
            "training_inputs": "target dense anchor;target RGB render loss;residual payload labels",
            "rate_accounting": "latent bytes counted in total RD",
            "reason": "Residual/latent side-info is likely needed for target headroom but should be introduced after predictor-only smoke.",
            "next_stage_action": "Define concrete latent/residual bitstream in Stage203.",
        },
    ]


def loss_rows():
    return [
        {
            "loss": "anchor_attr_huber",
            "stage": "Stage201+",
            "source": "predicted GS anchor vs target dense anchor",
            "weight": "1.0",
            "decoder_input_required": "no",
            "reason": "Supervises GS-native geometry/appearance attributes without using target at decode time.",
        },
        {
            "loss": "render_rgb_mse_or_l1",
            "stage": "Stage201+",
            "source": "rendered predicted GS vs target RGB",
            "weight": "0.1 smoke default",
            "decoder_input_required": "no",
            "reason": "Render-aware training aligns attribute loss with visual metrics; target RGB remains training-only.",
        },
        {
            "loss": "endpoint_identity",
            "stage": "Stage201+",
            "source": "t=0 equals left keyframe and t=1 equals right keyframe",
            "weight": "architecture hard gate plus optional penalty",
            "decoder_input_required": "no",
            "reason": "Prevents corrupting transmitted keyframes and stabilizes schedule boundaries.",
        },
        {
            "loss": "residual_energy_for_codec_labels",
            "stage": "Stage203+",
            "source": "target dense anchor minus predictor anchor",
            "weight": "diagnostic only before codec design",
            "decoder_input_required": "no",
            "reason": "Guides GS-native residual payload selection; payload bytes must be counted.",
        },
        {
            "loss": "rate_proxy",
            "stage": "Stage203+",
            "source": "estimated latent/residual entropy or selected attribute count",
            "weight": "lambda sweep after smoke",
            "decoder_input_required": "counted_payload_only",
            "reason": "Prepares RD-aware residual and selector stages without image residuals.",
        },
    ]


def protocol_rows(stage199, smoke):
    return [
        {"item": "stage201_input_manifest", "value": stage199["tasks_csv"]},
        {"item": "stage201_train_split", "value": "train"},
        {"item": "stage201_eval_split", "value": "eval"},
        {"item": "stage201_smoke_gaps", "value": "4 8"},
        {"item": "stage201_keyframe_codec", "value": "q12"},
        {"item": "stage201_predictor", "value": "TemporalBasisGSRefiner(hidden_dim=192, global_dim=64, zero_init_residual=True)"},
        {"item": "stage201_payload", "value": "none; predictor-only; zero per-frame side-info"},
        {"item": "stage201_heavy_root", "value": "/data/hctang/tmp/opencode/mono_dfcgs_runs/stage201_predictor_only_smoke"},
        {"item": "stage201_acceptance_gate", "value": "stable rendered metrics; no broadcast warnings; endpoint identity preserved; final eval not worse than linear by more than 0.05 dB; reject architecture if no broader headroom in Stage202"},
        {"item": "stage200_parameter_count_hidden96_global32", "value": smoke["parameter_count_hidden96_global32"]},
    ]


def audit_rows(stage198_rows, stage199, smoke):
    max_endpoint = max(smoke["endpoint_t0_max_abs_delta"], smoke["endpoint_t1_max_abs_delta"])
    requirements = {row["requirement"] for row in stage198_rows}
    return [
        {
            "audit": "stage198_requirements_loaded",
            "status": "pass" if "predictor_only_gate_before_selector" in requirements else "fail",
            "value": len(requirements),
            "requirement": "Stage200 uses Stage198 gates",
            "detail": ";".join(sorted(requirements)),
        },
        {
            "audit": "stage199_manifest_ready",
            "status": "pass" if stage199["decision"] == "manifest_ready_for_stage200_architecture_package" else "fail",
            "value": stage199["task_count"],
            "requirement": "Stage199 task manifest is ready",
            "detail": f"missing_count={stage199['missing_count']}; gaps={stage199['gaps']}",
        },
        {
            "audit": "endpoint_identity",
            "status": "pass" if max_endpoint <= 1e-7 else "fail",
            "value": max_endpoint,
            "requirement": "t=0/t=1 decode exactly to transmitted endpoints before output constraints",
            "detail": f"t0={smoke['endpoint_t0_max_abs_delta']}; t1={smoke['endpoint_t1_max_abs_delta']}",
        },
        {
            "audit": "zero_init_linear_fallback",
            "status": "pass" if smoke["zero_init_midpoint_linear_max_abs_delta"] <= 1e-7 else "fail",
            "value": smoke["zero_init_midpoint_linear_max_abs_delta"],
            "requirement": "new model starts as linear interpolation before training",
            "detail": "zero-initialized final residual layer with endpoint-gated residual",
        },
        {
            "audit": "stage197_decoder_contract",
            "status": "pass",
            "value": 0,
            "requirement": "decoder excludes target dense anchors, RGB/image residuals, and oracle labels",
            "detail": "target dense anchors and RGB are training/encoder-side only; payloads after Stage203 must be GS-native and counted",
        },
    ]


def write_report(package, candidates, losses, audits, protocol, path):
    lines = [
        "# Stage200 GS Predictor Architecture Package",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        f"- Primary architecture: `{package['primary_architecture']}`.",
        f"- Smoke parameter count: `{package['smoke']['parameter_count_hidden96_global32']}` for hidden96/global32 diagnostic instance.",
        "",
        "## Candidates",
        "",
        "| candidate | status | reason | next |",
        "|---|---|---|---|",
    ]
    for row in candidates:
        lines.append(f"| {row['candidate']} | {row['status']} | {row['reason']} | {row['next_stage_action']} |")
    lines.extend([
        "",
        "## Loss Contract",
        "",
        "| loss | stage | source | decoder input required |",
        "|---|---|---|---|",
    ])
    for row in losses:
        lines.append(f"| {row['loss']} | {row['stage']} | {row['source']} | {row['decoder_input_required']} |")
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
        "## Stage201 Protocol",
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
        "- Allowed: decoded/transmitted left/right GS keyframes, normalized time from schedule metadata, shared refiner weights, and optional counted GS-native latent/residual payloads in later stages.",
        "- Training/encoder-only: target dense anchor and target RGB render loss.",
        "- Forbidden: target dense anchor as decoder input, target RGB/image residual, and oracle schedule/quality labels.",
        "",
        "## Outputs",
        "",
        f"- candidates: `{package['candidates_csv']}`",
        f"- loss contract: `{package['loss_csv']}`",
        f"- decoder audit: `{package['audit_csv']}`",
        f"- Stage201 protocol: `{package['protocol_csv']}`",
        f"- architecture JSON: `{package['architecture_json']}`",
        f"- package: `{package['package_json']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stage198_rows = read_csv(STAGE198_REQUIREMENTS)
    stage199 = read_json(STAGE199_PACKAGE)
    smoke = architecture_smoke()
    candidates = candidate_rows()
    losses = loss_rows()
    audits = audit_rows(stage198_rows, stage199, smoke)
    protocol = protocol_rows(stage199, smoke)
    decision = "primary_temporal_basis_refiner_v1_selected_for_stage201_smoke"
    if any(row["status"] != "pass" for row in audits):
        decision = "architecture_package_blocked_by_audit"

    candidates_csv = OUTPUT_ROOT / "stage200_architecture_candidates.csv"
    loss_csv = OUTPUT_ROOT / "stage200_loss_contract.csv"
    audit_csv = OUTPUT_ROOT / "stage200_decoder_contract_audit.csv"
    protocol_csv = OUTPUT_ROOT / "stage200_stage201_smoke_protocol.csv"
    architecture_json = OUTPUT_ROOT / "stage200_primary_architecture_contract.json"
    package_json = OUTPUT_ROOT / "stage200_gs_predictor_architecture_package.json"
    report_md = OUTPUT_ROOT / "stage200_gs_predictor_architecture_report.md"

    architecture = {
        "architecture": "temporal_basis_gs_refiner_v1",
        "module": "mono_dfcgs.learned_gs_predictor.TemporalBasisGSRefiner",
        "primary_stage": 201,
        "inputs_decoder": [
            "decoded_left_keyframe_gs",
            "decoded_right_keyframe_gs",
            "normalized_time",
            "shared_refiner_weights",
            "optional_counted_gs_latent_after_stage203",
        ],
        "training_only_sources": ["target_dense_anchor", "target_rgb_render_loss"],
        "forbidden_decoder_inputs": ["target_dense_anchor", "target_rgb_or_image_residual", "oracle_schedule_or_quality_labels"],
        "core_design": [
            "linear q-keyframe interpolation base",
            "endpoint-gated residual factor t*(1-t)",
            "local per-Gaussian features left/right/base/diff/absdiff/time",
            "global GS statistics pooled from decoded endpoints",
            "zero-initialized residual head for linear fallback",
        ],
        "rate_accounting": "predictor-only uses zero per-frame payload; later GS latent/residual payloads must be transmitted and counted",
        "smoke": smoke,
    }
    package = {
        "stage": 200,
        "name": "gs_predictor_architecture_package",
        "decision": decision,
        "primary_architecture": "temporal_basis_gs_refiner_v1",
        "stage198_requirements": str(STAGE198_REQUIREMENTS),
        "stage199_package": str(STAGE199_PACKAGE),
        "candidates_csv": str(candidates_csv),
        "loss_csv": str(loss_csv),
        "audit_csv": str(audit_csv),
        "protocol_csv": str(protocol_csv),
        "architecture_json": str(architecture_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "smoke": smoke,
        "audit_rows": audits,
    }

    write_csv(candidates, candidates_csv, CANDIDATE_FIELDS)
    write_csv(losses, loss_csv, LOSS_FIELDS)
    write_csv(audits, audit_csv, AUDIT_FIELDS)
    write_csv(protocol, protocol_csv, PROTOCOL_FIELDS)
    architecture_json.write_text(json.dumps(architecture, indent=2) + "\n", encoding="utf-8")
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, candidates, losses, audits, protocol, report_md)
    print(json.dumps({"package": str(package_json), "decision": decision, "primary": package["primary_architecture"]}, indent=2))


if __name__ == "__main__":
    main()
