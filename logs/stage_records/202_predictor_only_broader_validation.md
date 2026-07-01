# Stage202 Predictor-Only Broader Validation

Date: 2026-07-01

## Goal

Test whether `TemporalBasisGSRefiner` has predictor-only training headroom beyond the Stage201 plumbing/no-regression smoke.

## Plan

- Reuse Stage199 q12 tasks over a broader gap set than Stage201.
- Run multiple short training configurations from scratch on the same selected train/eval tasks.
- Compare each config against linear interpolation and the zero-init predictor.
- Save checkpoints outside git under `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage202_predictor_only_broader_validation`.
- Decide whether predictor-only training has positive rendered PSNR headroom or should yield priority to GS-native residual/latent codec design.

## Success Criteria

- Script compiles and runs with `CUDA_VISIBLE_DEVICES=<idle_gpu> ... --device cuda`.
- All metric rows render with explicit target-shape alignment.
- At least one config improves best eval PSNR over linear by `>0.05` dB to claim predictor-only headroom.
- If no config improves, record that predictor-only training headroom was not observed and Stage203 residual codec becomes mandatory.

## Execution

- Syntax check: `CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage202_predictor_only_broader_validation.py`
- Pre-run GPU check: `nvidia-smi` at 2026-07-01 23:54:54.
- Command: `CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage202_predictor_only_broader_validation.py --device cuda --max_train_tasks 16 --max_eval_tasks 16`

## Outputs

- Output root: `experiments/stage202_predictor_only_broader_validation/`
- Selected tasks: `experiments/stage202_predictor_only_broader_validation/stage202_selected_tasks.csv`
- Config results: `experiments/stage202_predictor_only_broader_validation/stage202_predictor_only_config_results.csv`
- Summary: `experiments/stage202_predictor_only_broader_validation/stage202_predictor_only_summary.csv`
- Per-task metrics: `experiments/stage202_predictor_only_broader_validation/stage202_predictor_only_metrics.csv`
- Train log: `experiments/stage202_predictor_only_broader_validation/stage202_predictor_only_train_log.csv`
- Gates: `experiments/stage202_predictor_only_broader_validation/stage202_predictor_only_gates.csv`
- Package: `experiments/stage202_predictor_only_broader_validation/stage202_predictor_only_broader_validation_package.json`
- Report: `experiments/stage202_predictor_only_broader_validation/stage202_predictor_only_broader_validation_report.md`
- Checkpoints outside git: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage202_predictor_only_broader_validation/`

## Results

- Decision: `predictor_only_broader_training_headroom_not_observed`.
- Gaps: `2,4,8,12`; q12; train/eval tasks: `16` / `16`; configs: `3`; per-frame payload bytes: `0`.
- Shared linear eval PSNR: `19.239936914617395`.
- `anchor_render_lr2e4`: best eval PSNR `19.239936914617395`, best delta `0.0` dB, final delta `-0.0014714499478820642` dB, best step `0`.
- `anchor_only_lr1e3`: best eval PSNR `19.240604951854333`, best delta `+0.0006680372369380905` dB, final delta `+0.0006680372369380905` dB, best step `32`.
- `render_heavy_lr1e4`: best eval PSNR `19.239936914617395`, best delta `0.0` dB, final delta `-0.0008068938220269217` dB, best step `0`.
- Gates passed: metric rows ok, endpoint identity all configs, predictor-only payload, any-config no-regression.
- Gate failed: `predictor_headroom_positive`; best delta `+0.0006680372369380905` dB is far below `>0.05` dB.
- Interpretation: predictor-only learned headroom is not observed under these smoke/broader settings. Stage203 should prioritize GS-native residual/latent side-info; selector training should remain deferred until edge RD/oracle headroom exists.

## Decision

- Proceed to Stage203 GS latent/residual codec design with predictor-only headroom marked insufficient.
