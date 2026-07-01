# Stage204 Residual Codec Smoke

Date: 2026-07-02

## Goal

Test `gs_attr_topk_residual_entropy_v1` on real Stage199 q12 tasks with rendered metrics and counted GS-native residual payload bytes.

## Plan

- Use a small eval subset from Stage199 q12 gaps `4` and `8`.
- Use linear/zero-init predictor base from Stage201/202.
- Encode top-k GS attribute residual payloads with side bits `6`, zlib level `9`, and keep fractions `0.05,0.10,0.20`.
- Decode payloads using only predictor/base GS plus transmitted residual payload.
- Render base and corrected anchors, compute PSNR, anchor MSE reduction, and exact payload bytes.

## Success Criteria

- Script compiles and runs with `CUDA_VISIBLE_DEVICES=<idle_gpu> ... --device cuda`.
- All rendered metrics use explicit target-shape alignment.
- Payload bytes are exact `len(payload)` values and are nonzero for residual rows.
- At least one setting improves eval PSNR over linear/base by more than `0.5` dB to demonstrate residual headroom.

## Execution

- Syntax check: `CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage204_residual_codec_smoke.py`
- Pre-run GPU check: `nvidia-smi` at 2026-07-02 00:11:52.
- Command: `CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage204_residual_codec_smoke.py --device cuda --max_tasks 12`

## Outputs

- Output root: `experiments/stage204_residual_codec_smoke/`
- Selected tasks: `experiments/stage204_residual_codec_smoke/stage204_selected_tasks.csv`
- Rows: `experiments/stage204_residual_codec_smoke/stage204_residual_codec_smoke_rows.csv`
- Summary: `experiments/stage204_residual_codec_smoke/stage204_residual_codec_smoke_summary.csv`
- Gates: `experiments/stage204_residual_codec_smoke/stage204_residual_codec_smoke_gates.csv`
- Package: `experiments/stage204_residual_codec_smoke/stage204_residual_codec_smoke_package.json`
- Report: `experiments/stage204_residual_codec_smoke/stage204_residual_codec_smoke_report.md`

## Results

- Decision: `residual_codec_smoke_positive_headroom`.
- Scope: `12` eval tasks, q12 gaps `4,8`, linear/zero-init predictor base, side bits `6`, zlib level `9`, keep fractions `0.05,0.10,0.20`.
- Base mean PSNR: `19.999033822428466`.
- `topk_keep0p05_q6`: mean payload `15679.583333333334` bytes, corrected PSNR `22.39462808214667`, dPSNR `+2.3955942597182047`, anchor MSE reduction `0.5102364961382247`.
- `topk_keep0p1_q6`: mean payload `29836.75` bytes, corrected PSNR `23.804629721924517`, dPSNR `+3.8055958994960553`, anchor MSE reduction `0.688443254187152`.
- `topk_keep0p2_q6`: mean payload `55761.833333333336` bytes, corrected PSNR `25.552135029430517`, dPSNR `+5.553101207002054`, anchor MSE reduction `0.8435545876871436`.
- Gates passed: metric rows ok, counted nonzero payload, positive anchor MSE reduction, positive rendered headroom, Stage197 decoder contract.
- Interpretation: GS-native residual payload has real quality headroom on true rendered tasks; proceed to fixed-gap predictive codec validation with exact payload bytes included in total RD.

## Decision

- Proceed to Stage205 fixed-gap predictive codec validation.
