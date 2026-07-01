# Stage201 Predictor-Only Smoke

Date: 2026-07-01

## Goal

Run a small predictor-only training and rendered evaluation smoke for `TemporalBasisGSRefiner` using Stage199 q12 tasks, with zero per-frame residual/latent payload.

## Plan

- Use Stage199 train/eval tasks for q12 gaps `4` and `8`.
- Train a small `TemporalBasisGSRefiner` instance with target dense anchors and target RGB as training/eval labels only.
- Evaluate linear, initial predictor, final predictor, and best predictor rendered PSNR plus anchor MSE.
- Save checkpoints outside git under `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage201_predictor_only_smoke`.
- Audit endpoint identity, no residual payload, metric shape alignment, no missing references, and no-regression gate.

## Success Criteria

- Script compiles and runs with `CUDA_VISIBLE_DEVICES=<idle_gpu> ... --device cuda`.
- Render target shapes are aligned explicitly; no broadcast-based metrics are trusted.
- Best predictor eval PSNR is not worse than linear by more than `0.05` dB.
- Decoder contract remains predictor-only: no target dense anchor, RGB/image residual, oracle labels, or uncounted payload at decode time.

## Execution

- Syntax check: `CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage201_predictor_only_smoke.py`
- Pre-run GPU check: `nvidia-smi` at 2026-07-01 23:46:03.
- Command: `CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage201_predictor_only_smoke.py --device cuda --max_train_tasks 8 --max_eval_tasks 8 --train_steps 16 --eval_every 8 --hidden_dim 192 --global_dim 64 --render_loss_weight 0.02`

## Outputs

- Output root: `experiments/stage201_predictor_only_smoke/`
- Selected tasks: `experiments/stage201_predictor_only_smoke/stage201_selected_tasks.csv`
- Per-task metrics: `experiments/stage201_predictor_only_smoke/stage201_predictor_only_smoke_metrics.csv`
- Summary: `experiments/stage201_predictor_only_smoke/stage201_predictor_only_smoke_summary.csv`
- Train log: `experiments/stage201_predictor_only_smoke/stage201_predictor_only_train_log.csv`
- Gates: `experiments/stage201_predictor_only_smoke/stage201_predictor_only_gates.csv`
- Package: `experiments/stage201_predictor_only_smoke/stage201_predictor_only_smoke_package.json`
- Report: `experiments/stage201_predictor_only_smoke/stage201_predictor_only_smoke_report.md`
- Best checkpoint outside git: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage201_predictor_only_smoke/temporal_basis_gs_refiner_best.safetensors`
- Final checkpoint outside git: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage201_predictor_only_smoke/temporal_basis_gs_refiner_final.safetensors`

## Results

- Decision: `predictor_only_smoke_passed_no_regression_gate`.
- Train/eval tasks: `8` / `8`; gaps `4,8`; q12; train steps `16`; hidden/global dims `192/64`; render loss weight `0.02`.
- Per-frame payload bytes: `0`.
- Best step: `0`; best eval PSNR `20.52573876541323`.
- Linear eval PSNR: `20.52573876541323`; predictor-best eval PSNR: `20.52573876541323`; predictor-final eval PSNR: `20.509261273864784`.
- Predictor-best eval delta vs linear: `0.0` dB; predictor-final eval delta vs linear: `-0.016477491548446466` dB.
- Predictor-best train delta vs linear: `0.0` dB; predictor-final train delta vs linear: `-0.0003789591630756206` dB.
- Best eval sanity delta vs Stage78 q12 old adapter reference over gap4/8: `+2.862793704365373` dB. This reference is historical and protocol-different, so it is only a sanity floor.
- Gates passed: metric rows ok, endpoint identity, predictor-only payload, best eval no-regression vs linear, best eval vs old-adapter sanity floor, final checkpoint written outside git.
- Important interpretation: Stage201 proves executable/stable predictor-only plumbing, not learned quality improvement. The best checkpoint is the zero-init/linear fallback at step `0`; Stage202 must test whether longer/broader training can create actual predictor-only headroom before promoting residual/selector stages.

## Decision

- Proceed to Stage202 predictor-only broader validation/training-headroom check.
