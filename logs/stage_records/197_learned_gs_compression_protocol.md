# Stage197 Learned GS Compression Protocol

Date: 2026-07-01

## Goal

Define the new GS-native learned predictive compression route after Stage196 rejected selector/keyframe-quantization tuning for the requested `uniform_gap2 + 1 dB` full-sequence target.

## Plan

- Freeze the new method scope before training.
- Explicitly reject RGB/image residual post-processing as a final method.
- Define decoder allowed/forbidden inputs for the new learned GS compression route.
- Define how the StreamSplat checkpoint may be used without turning the codec into a raw-image decoder.
- Define stage gates for Stage198-213.

## Success Criteria

- Produce lightweight protocol CSV/JSON/Markdown outputs.
- State one primary runtime decoder contract.
- State which future branches are optional diagnostics rather than final claims.

## Execution

- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage197_learned_gs_compression_protocol.py`
- Pre-run GPU check: `nvidia-smi` at 2026-07-01 23:03:12; Stage197 itself is CPU-only.
- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage197_learned_gs_compression_protocol.py`

## Outputs

- Output root: `experiments/stage197_learned_gs_compression_protocol/`
- Package: `experiments/stage197_learned_gs_compression_protocol/stage197_learned_gs_compression_protocol_package.json`
- Report: `experiments/stage197_learned_gs_compression_protocol/stage197_learned_gs_compression_protocol_report.md`
- Decoder contract: `experiments/stage197_learned_gs_compression_protocol/stage197_decoder_contract.csv`
- Module contract: `experiments/stage197_learned_gs_compression_protocol/stage197_module_contract.csv`
- Stage plan: `experiments/stage197_learned_gs_compression_protocol/stage197_stage_plan.csv`

## Results

- Primary runtime decoder uses transmitted GS keyframes, schedule, normalized time, shared GS codec weights, and counted GS-native latent/residual payloads only.
- RGB/image residual post-processing is rejected as a final method.
- StreamSplat checkpoint may initialize or supervise modules, but raw RGB-dependent StreamSplat runtime is not the primary final codec claim.
- Stage gates for Stage198-213 are packaged in `stage197_stage_plan.csv`.

## Decision

- Decision: `primary_gs_native_predictive_codec_protocol_defined`.
- Proceed to Stage198 prior predictor training audit, then Stage199 learned GS training manifest.
