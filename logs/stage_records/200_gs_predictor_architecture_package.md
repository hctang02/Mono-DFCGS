# Stage200 GS Predictor Architecture Package

Date: 2026-07-01

## Goal

Define the new GS-native predictor/refiner architecture and loss contract for Stage201 predictor-only smoke, under the Stage197 decoder contract and Stage198 rejection of the old adapter route.

## Plan

- Add a lightweight executable predictor module that does not continue `GaussianAnchorDynamicPredictor` unchanged.
- Select a primary decoder-side GS refiner candidate for Stage201.
- Package candidate comparison, loss contract, decoder-contract audit, and Stage201 smoke protocol.
- Validate import, parameter count, endpoint identity, and zero-initialized linear fallback on CPU.

## Success Criteria

- Primary architecture is selected and has a clear Stage201 protocol.
- Runtime decoder inputs stay limited to decoded/transmitted GS keyframes, schedule/time, shared weights, and optional counted GS-native payloads.
- Target dense anchors and RGB remain training/encoder-only labels.
- New module preserves endpoint identity at `t=0` and `t=1` and starts as linear interpolation before training.

## Execution

- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile mono_dfcgs/learned_gs_predictor.py scripts/run_stage200_gs_predictor_architecture_package.py`
- Pre-run GPU check: `nvidia-smi` at 2026-07-01 23:34:59; Stage200 itself is CPU-only.
- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage200_gs_predictor_architecture_package.py`

## Outputs

- New module: `mono_dfcgs/learned_gs_predictor.py`
- Output root: `experiments/stage200_gs_predictor_architecture_package/`
- Candidates: `experiments/stage200_gs_predictor_architecture_package/stage200_architecture_candidates.csv`
- Loss contract: `experiments/stage200_gs_predictor_architecture_package/stage200_loss_contract.csv`
- Decoder audit: `experiments/stage200_gs_predictor_architecture_package/stage200_decoder_contract_audit.csv`
- Stage201 protocol: `experiments/stage200_gs_predictor_architecture_package/stage200_stage201_smoke_protocol.csv`
- Architecture contract: `experiments/stage200_gs_predictor_architecture_package/stage200_primary_architecture_contract.json`
- Package: `experiments/stage200_gs_predictor_architecture_package/stage200_gs_predictor_architecture_package.json`
- Report: `experiments/stage200_gs_predictor_architecture_package/stage200_gs_predictor_architecture_report.md`

## Results

- Decision: `primary_temporal_basis_refiner_v1_selected_for_stage201_smoke`.
- Primary architecture: `temporal_basis_gs_refiner_v1` implemented as `mono_dfcgs.learned_gs_predictor.TemporalBasisGSRefiner`.
- Diagnostic parameter count for hidden96/global32 instance: `43501`.
- Endpoint identity smoke: t0 max abs delta `0.0`, t1 max abs delta `0.0`.
- Zero-init midpoint linear fallback max abs delta: `0.0`.
- Stage198 requirements loaded: `predictor_only_gate_before_selector`, `gs_native_residual_payload`, `edge_rd_headroom_before_selector_training`, `full_sequence_metrics_only_for_strong_claim`.
- Stage199 manifest ready: `29204` tasks, missing count `0`, gaps `[2, 4, 6, 8, 12, 16]`.
- Stage201 smoke protocol: q12 predictor-only, gaps `4 8`, no per-frame payload, heavy outputs under `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage201_predictor_only_smoke`.
- Runtime decoder still forbids target dense anchors, target RGB/image residuals, and oracle schedule/quality labels.

## Decision

- Proceed to Stage201 predictor-only smoke with `TemporalBasisGSRefiner`.
