import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "experiments/stage197_learned_gs_compression_protocol"

CONTRACT_FIELDS = ["scope", "item", "status", "rationale"]
MODULE_FIELDS = ["module", "runtime_side", "allowed_inputs", "outputs", "rate_accounting", "training_notes"]
STAGE_FIELDS = ["stage", "name", "goal", "gate", "commit_required"]


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def contract_rows():
    return [
        {
            "scope": "primary_runtime_decoder",
            "item": "transmitted_q_keyframe_gs_bitstreams",
            "status": "allowed",
            "rationale": "Keyframes are the compressed GS stream anchors and every keyframe byte is counted.",
        },
        {
            "scope": "primary_runtime_decoder",
            "item": "transmitted_schedule_metadata",
            "status": "allowed",
            "rationale": "Decoder receives keyframe indices/segment metadata; selector features are not needed at decode time.",
        },
        {
            "scope": "primary_runtime_decoder",
            "item": "normalized_time",
            "status": "allowed",
            "rationale": "Time within a keyframe segment is deterministic from transmitted schedule.",
        },
        {
            "scope": "primary_runtime_decoder",
            "item": "shared_gs_predictor_refiner_weights",
            "status": "allowed",
            "rationale": "Codec model weights are shared once as method parameters, not per-video hidden side information.",
        },
        {
            "scope": "primary_runtime_decoder",
            "item": "transmitted_gs_latent_or_residual_bitstreams",
            "status": "allowed_counted",
            "rationale": "Residual/latent payloads are GS-native and must be included in total rate.",
        },
        {
            "scope": "primary_runtime_decoder",
            "item": "target_dense_anchor",
            "status": "forbidden",
            "rationale": "Target anchors may supervise training or produce encoder-side residual payloads, but cannot be a decoder input.",
        },
        {
            "scope": "primary_runtime_decoder",
            "item": "target_rgb_or_image_residual",
            "status": "forbidden",
            "rationale": "User rejected image-domain post-processing; final method must remain GS compression.",
        },
        {
            "scope": "primary_runtime_decoder",
            "item": "oracle_schedule_or_quality_labels",
            "status": "forbidden",
            "rationale": "Oracle labels can train/evaluate but must be represented by transmitted schedule or learned encoder decisions at inference.",
        },
        {
            "scope": "encoder_training",
            "item": "target_dense_anchor_and_rgb",
            "status": "allowed_training_only",
            "rationale": "Encoder/training can use targets to learn predictors, residual payloads, and selector labels.",
        },
        {
            "scope": "streamsplat_checkpoint",
            "item": "frozen_streamsplat_as_initialization_or_teacher",
            "status": "allowed_optional",
            "rationale": "The checkpoint can initialize or supervise GS-native modules, but primary runtime decoder should not require raw target RGB.",
        },
        {
            "scope": "streamsplat_checkpoint",
            "item": "full_runtime_streamsplat_raw_rgb_dependency",
            "status": "not_primary_final_claim",
            "rationale": "Using raw video/image inputs at decoder would undermine a GS-compression claim unless those inputs are explicitly transmitted and counted.",
        },
    ]


def module_rows():
    return [
        {
            "module": "keyframe_codec",
            "runtime_side": "encoder_and_decoder",
            "allowed_inputs": "GS keyframe anchors selected by encoder",
            "outputs": "keyframe bitstreams and decoded keyframe GS",
            "rate_accounting": "all keyframe bitstream bytes counted; schedule metadata counted",
            "training_notes": "Reuse existing anchor bitstream infrastructure first; later q/rate variants can be ablated.",
        },
        {
            "module": "gs_predictor_refiner",
            "runtime_side": "decoder",
            "allowed_inputs": "decoded left/right keyframe GS, normalized time, shared weights, optional decoded GS latent",
            "outputs": "predicted or refined intermediate GS",
            "rate_accounting": "shared weights declared as codec parameters; per-video latent bytes counted if used",
            "training_notes": "Train with render-aware losses and endpoint identity constraints; do not continue old weak adapter unchanged.",
        },
        {
            "module": "gs_latent_residual_codec",
            "runtime_side": "encoder_and_decoder",
            "allowed_inputs": "encoder uses target GS/RGB to form payload; decoder uses only decoded payload plus predictor state",
            "outputs": "GS-native correction payload and corrected GS",
            "rate_accounting": "all residual/latent bytes counted in total RD",
            "training_notes": "Prioritize sparse GS attribute residual or learned latent residual; no image residual final method.",
        },
        {
            "module": "encoder_selector_and_budget_allocator",
            "runtime_side": "encoder_only",
            "allowed_inputs": "input RGB/motion proxies, decoded/encoded GS stats, predictor uncertainty, estimated residual cost",
            "outputs": "keyframe schedule and residual budget decisions",
            "rate_accounting": "only transmitted schedule and payloads counted; encoder-only features are not decoder inputs",
            "training_notes": "Train after predictor/residual edge RD tables show oracle headroom.",
        },
    ]


def stage_rows():
    return [
        {"stage": 198, "name": "prior_predictor_training_audit", "goal": "document why old adapter training failed", "gate": "old adapter route rejected", "commit_required": 1},
        {"stage": 199, "name": "learned_gs_training_manifest", "goal": "build multi-gap train/eval task manifest", "gate": "complete lightweight references and split audit", "commit_required": 1},
        {"stage": 200, "name": "gs_predictor_architecture_package", "goal": "define new predictor/refiner candidates", "gate": "architecture and loss contract ready", "commit_required": 1},
        {"stage": 201, "name": "predictor_only_smoke", "goal": "test new predictor without residual/selector", "gate": "> old adapter and stable render metrics", "commit_required": 1},
        {"stage": 202, "name": "predictor_only_broader_validation", "goal": "validate predictor over multiple gaps", "gate": "broad improvement or architecture rejection", "commit_required": 1},
        {"stage": 203, "name": "gs_latent_residual_codec_design", "goal": "define GS-native residual/latent codec", "gate": "codec payload is decodable and counted", "commit_required": 1},
        {"stage": 204, "name": "residual_codec_smoke", "goal": "test predictor plus GS residual on small set", "gate": "quality headroom near target", "commit_required": 1},
        {"stage": 205, "name": "fixed_gap_predictive_codec_validation", "goal": "validate predictor+residual without selector", "gate": "fixed-gap quality/rate headroom", "commit_required": 1},
        {"stage": 206, "name": "edge_rd_table", "goal": "measure segment-level costs for schedule optimization", "gate": "edge coverage complete for selected gaps", "commit_required": 1},
        {"stage": 207, "name": "dp_oracle_schedule", "goal": "compute oracle keyframe schedules", "gate": "oracle beats fixed baselines", "commit_required": 1},
        {"stage": 208, "name": "selector_training_data", "goal": "convert oracle schedules to encoder labels/features", "gate": "feature-source audit passes", "commit_required": 1},
        {"stage": 209, "name": "encoder_selector_training", "goal": "train deployable schedule predictor", "gate": "learned selector approaches oracle", "commit_required": 1},
        {"stage": 210, "name": "selector_residual_budget_joint_training", "goal": "jointly tune schedule and residual allocation", "gate": "joint RD improves selector-only", "commit_required": 1},
        {"stage": 211, "name": "full_sequence_measured_rd", "goal": "compare final adaptive against fixed gaps", "gate": "full-sequence RD-quality complete", "commit_required": 1},
        {"stage": 212, "name": "ablation_package", "goal": "prove predictor/residual/selector contributions", "gate": "paper-facing ablations complete", "commit_required": 1},
        {"stage": 213, "name": "subjective_visual_export", "goal": "export videos/contact sheets outside git", "gate": "visual evidence paths recorded", "commit_required": 1},
    ]


def write_report(package, contract, modules, stages, path):
    lines = [
        "# Stage197 Learned GS Compression Protocol",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        "- Primary runtime decoder uses transmitted GS keyframes, schedule, time, shared GS codec weights, and counted GS-native latent/residual payloads only.",
        "- RGB/image residual post-processing is rejected as a final method.",
        "- StreamSplat checkpoint may initialize or supervise modules, but raw RGB-dependent StreamSplat runtime is not the primary final codec claim.",
        "",
        "## Contract",
        "",
        "| scope | item | status | rationale |",
        "|---|---|---|---|",
    ]
    for row in contract:
        lines.append(f"| {row['scope']} | {row['item']} | {row['status']} | {row['rationale']} |")
    lines.extend(["", "## Modules", "", "| module | runtime side | outputs | rate accounting |", "|---|---|---|---|"])
    for row in modules:
        lines.append(f"| {row['module']} | {row['runtime_side']} | {row['outputs']} | {row['rate_accounting']} |")
    lines.extend(["", "## Stage Gates", "", "| stage | name | goal | gate |", "|---:|---|---|---|"])
    for row in stages:
        lines.append(f"| {row['stage']} | {row['name']} | {row['goal']} | {row['gate']} |")
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Contract CSV: `{package['contract_csv']}`",
        f"- Module CSV: `{package['module_csv']}`",
        f"- Stage plan CSV: `{package['stage_plan_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    contract = contract_rows()
    modules = module_rows()
    stages = stage_rows()
    contract_csv = OUTPUT_ROOT / "stage197_decoder_contract.csv"
    module_csv = OUTPUT_ROOT / "stage197_module_contract.csv"
    stage_plan_csv = OUTPUT_ROOT / "stage197_stage_plan.csv"
    package_json = OUTPUT_ROOT / "stage197_learned_gs_compression_protocol_package.json"
    report_md = OUTPUT_ROOT / "stage197_learned_gs_compression_protocol_report.md"
    write_csv(contract, contract_csv, CONTRACT_FIELDS)
    write_csv(modules, module_csv, MODULE_FIELDS)
    write_csv(stages, stage_plan_csv, STAGE_FIELDS)
    package = {
        "stage": 197,
        "status": "learned_gs_compression_protocol_complete",
        "decision": "primary_gs_native_predictive_codec_protocol_defined",
        "final_method_rejects_rgb_image_residual": True,
        "primary_decoder_runtime_contract": "decoded_gs_keyframes_plus_schedule_time_shared_weights_and_counted_gs_latent_residual_payloads",
        "streamsplat_usage": "allowed_as_initialization_teacher_or_optional_diagnostic_base_not_primary_raw_rgb_runtime_claim",
        "contract_csv": str(contract_csv.relative_to(REPO_ROOT)),
        "module_csv": str(module_csv.relative_to(REPO_ROOT)),
        "stage_plan_csv": str(stage_plan_csv.relative_to(REPO_ROOT)),
        "package_json": str(package_json.relative_to(REPO_ROOT)),
        "report_md": str(report_md.relative_to(REPO_ROOT)),
        "next": "Stage198 audit prior decoder predictor training before building new manifest.",
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, contract, modules, stages, report_md)
    print(json.dumps({"package": str(package_json), "decision": package["decision"]}, indent=2))


if __name__ == "__main__":
    main()
