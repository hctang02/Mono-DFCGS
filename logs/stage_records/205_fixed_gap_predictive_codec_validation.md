# Stage205 Fixed-Gap Predictive Codec Validation

Date: 2026-07-02

## Goal

Validate fixed-gap predictor-plus-GS-residual codec behavior on real Stage199 tasks before building edge RD tables.

## Plan

- Sample eval tasks separately for fixed gaps `4,8,12`.
- Reuse the Stage204 `gs_attr_topk_residual_entropy_v1` implementation.
- Evaluate q6 keep fractions `0.05,0.10,0.20` with exact residual payload bytes.
- Summarize rendered PSNR, dPSNR vs base, residual payload bytes, and residual MiB per intermediate by fixed gap.

## Success Criteria

- Metrics render with explicit target-shape alignment and no errors.
- Every residual payload byte is counted from `len(payload)`.
- Each tested fixed gap has at least one residual setting with mean dPSNR > `0.5` dB vs base.
- Results are scoped as sampled fixed-gap validation, not full-sequence RD.

## Execution

- Syntax check: `CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage205_fixed_gap_predictive_codec_validation.py`
- Pre-run GPU check: `nvidia-smi` at 2026-07-02 00:18:54.
- Command: `CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage205_fixed_gap_predictive_codec_validation.py --device cuda --max_tasks_per_gap 8`

## Outputs

- Output root: `experiments/stage205_fixed_gap_predictive_codec_validation/`
- Selected tasks: `experiments/stage205_fixed_gap_predictive_codec_validation/stage205_selected_tasks.csv`
- Rows: `experiments/stage205_fixed_gap_predictive_codec_validation/stage205_fixed_gap_predictive_rows.csv`
- Summary: `experiments/stage205_fixed_gap_predictive_codec_validation/stage205_fixed_gap_predictive_summary.csv`
- Best by gap: `experiments/stage205_fixed_gap_predictive_codec_validation/stage205_fixed_gap_predictive_best_by_gap.csv`
- Gates: `experiments/stage205_fixed_gap_predictive_codec_validation/stage205_fixed_gap_predictive_gates.csv`
- Package: `experiments/stage205_fixed_gap_predictive_codec_validation/stage205_fixed_gap_predictive_codec_validation_package.json`
- Report: `experiments/stage205_fixed_gap_predictive_codec_validation/stage205_fixed_gap_predictive_codec_validation_report.md`

## Results

- Decision: `fixed_gap_predictive_codec_positive_headroom`.
- Scope: sampled fixed-gap validation, not full-sequence RD.
- Tested `24` eval tasks: `8` each for gaps `4,8,12`; q12 keyframes; q6 top-k residual settings `0.05,0.10,0.20`; exact payload bytes counted from `len(payload)`.
- Best gap4: `topk_keep0p2_q6`, payload `56283.0` bytes (`0.05367565155029297` MiB/intermediate), base/corrected PSNR `20.599420543944056/25.301132561904456`, dPSNR `+4.701712017960398`.
- Best gap8: `topk_keep0p2_q6`, payload `58359.875` bytes (`0.0556563138961792` MiB/intermediate), base/corrected PSNR `17.667283985227805/23.694548271579496`, dPSNR `+6.027264286351697`.
- Best gap12: `topk_keep0p2_q6`, payload `57028.5` bytes (`0.05438661575317383` MiB/intermediate), base/corrected PSNR `19.19559473773822/24.991767735164355`, dPSNR `+5.796172997426137`.
- Gates passed: metric rows ok, counted nonzero payload, gap coverage, each-gap positive headroom, Stage197 decoder contract.
- Interpretation: fixed-gap sampled validation supports moving to Stage206 edge RD table with exact keyframe/residual costs.

## Decision

- Proceed to Stage206 edge RD table.
