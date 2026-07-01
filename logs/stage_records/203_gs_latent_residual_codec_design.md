# Stage203 GS Latent/Residual Codec Design

Date: 2026-07-02

## Goal

Define a decode-capable, rate-counted, GS-native latent/residual codec for Stage204 after Stage202 showed predictor-only headroom is not observed.

## Plan

- Review existing GS residual side-info codec functions.
- Select a primary Stage204 codec that uses GS attribute residual payloads rather than RGB/image residuals.
- Package codec candidates, payload accounting rules, decoder contract, and Stage204 smoke protocol.
- Run CPU toy roundtrip checks for primary and low-rate deterministic codecs.

## Success Criteria

- Primary codec payload is byte-decodable from predictor/base GS plus transmitted payload.
- Payload bytes are explicitly counted from `len(payload)` and included in total RD.
- Target dense anchors are encoder-side residual sources only, not decoder inputs.
- RGB/image residuals remain rejected for the final method.

## Execution

- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage203_gs_latent_residual_codec_design.py`
- Pre-run GPU check: `nvidia-smi` at 2026-07-02 00:03:05; Stage203 itself is CPU-only.
- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage203_gs_latent_residual_codec_design.py`

## Outputs

- Output root: `experiments/stage203_gs_latent_residual_codec_design/`
- Candidates: `experiments/stage203_gs_latent_residual_codec_design/stage203_codec_candidates.csv`
- Toy roundtrips: `experiments/stage203_gs_latent_residual_codec_design/stage203_codec_toy_roundtrips.csv`
- Rate rules: `experiments/stage203_gs_latent_residual_codec_design/stage203_rate_accounting_rules.csv`
- Decoder audit: `experiments/stage203_gs_latent_residual_codec_design/stage203_decoder_contract_audit.csv`
- Stage204 protocol: `experiments/stage203_gs_latent_residual_codec_design/stage203_stage204_smoke_protocol.csv`
- Primary contract: `experiments/stage203_gs_latent_residual_codec_design/stage203_primary_codec_contract.json`
- Package: `experiments/stage203_gs_latent_residual_codec_design/stage203_gs_latent_residual_codec_design_package.json`
- Report: `experiments/stage203_gs_latent_residual_codec_design/stage203_gs_latent_residual_codec_design_report.md`

## Results

- Decision: `gs_attr_topk_residual_entropy_v1_selected_for_stage204_smoke`.
- Primary codec: `gs_attr_topk_residual_entropy_v1` implemented by `encode_topk_residual_sideinfo_entropy` and decoded by `decode_residual_sideinfo_entropy`.
- Payload accounting: `payload_bytes = len(payload)`; include header, fp16 metadata, sorted index deltas, q residual values, zlib component lengths, and all compressed component bytes.
- Top-k entropy toy roundtrip: payload `246` bytes, residual MSE before/after `0.006919591687619686/0.0008317023166455328`, MSE reduction `0.8798047118685358`, status pass.
- Deterministic-index entropy toy roundtrip: payload `217` bytes, residual MSE before/after `0.007950554601848125/0.0008250265964306891`, MSE reduction `0.8962303087335681`, status pass.
- Stage204 protocol: q12 gaps `4 8`, primary codec `gs_attr_topk_residual_entropy_v1`, `side_bits=6`, keep fractions `0.05,0.10,0.20`, `zlib_level=9`, rendered PSNR plus counted payload bytes.
- Decoder contract: decoder uses predictor/base GS plus transmitted counted GS residual payload only; target dense anchors and RGB/image residuals remain forbidden decoder inputs.

## Decision

- Proceed to Stage204 residual codec smoke on real Stage199 tasks.
