# Stage198 Prior Predictor Training Audit

Date: 2026-07-01

## Goal

Audit the prior decoder-side GS predictor/adapter experiments to decide whether the new learned GS compression route should reuse, continue training, or reject the old adapter path.

## Plan

- Read Stage78, Stage143, Stage145, Stage146, Stage154, Stage157/158, and Stage196 evidence.
- Quantify old adapter middle-frame quality, q-bit sensitivity, training gains, and remaining gap to the requested full-sequence target.
- Separate old adapter predictor failure from Stage158 GS residual side-info success.
- Package non-goals and requirements for the new predictor/residual modules.

## Success Criteria

- A lightweight audit report exists.
- Decision clearly says whether to continue old adapter training.
- New route requirements are explicit for Stage199-201.

## Execution

- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage198_prior_predictor_training_audit.py`
- Pre-run GPU check: `nvidia-smi` at 2026-07-01 23:09:57; Stage198 itself is CPU-only.
- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage198_prior_predictor_training_audit.py`
- Rerun after fixing Stage154/157 CSV column names: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage198_prior_predictor_training_audit.py`

## Outputs

- Output root: `experiments/stage198_prior_predictor_training_audit/`
- Package: `experiments/stage198_prior_predictor_training_audit/stage198_prior_predictor_training_audit_package.json`
- Report: `experiments/stage198_prior_predictor_training_audit/stage198_prior_predictor_training_audit_report.md`
- Evidence: `experiments/stage198_prior_predictor_training_audit/stage198_prior_predictor_evidence.csv`
- Decisions: `experiments/stage198_prior_predictor_training_audit/stage198_prior_predictor_decisions.csv`
- Requirements: `experiments/stage198_prior_predictor_training_audit/stage198_new_route_requirements.csv`

## Results

Key evidence:

- Stage78 old q12 adapter middle PSNR: gap4 `18.256196169477683`, gap8 `17.06969395261803`.
- Stage143 q12-to-float32 old adapter change: gap4 `+0.00004890061461537698` dB, gap8 `-0.000016654591867393265` dB.
- Stage143 float32 dense-direct minus adapter: gap4 `+11.49432172291868` dB, gap8 `+12.677631798990458` dB.
- Stage145 old-adapter training best gain: `+0.011427207695934527` dB after `80` steps over `6691` train rows.
- Stage146 continuation best gain: `0.0` dB; final change `-0.019897326878577533` dB.
- Stage154 original StreamSplat base: gap4 `22.06421822428011`, gap8 `20.33727549162514`.
- Stage157/158 GS residual side-info success: gap4 `29.780485398070507`, gap8 `29.578682359235195`; this is residual-sideinfo success, not old adapter success.

New route requirements:

- Predictor-only gate before selector training.
- GS-native residual payload, not image residual.
- Edge-RD oracle headroom before learned selector training.
- Full-sequence measured metrics for any strong claim.

## Decision

- Decision: `old_adapter_route_rejected_new_predictor_required`.
- Do not continue `GaussianAnchorDynamicPredictor` / Stage65 adapter unchanged.
- Stage200 must propose a stronger temporal refiner/motion-field architecture with render-aware and RD-aware gates.
