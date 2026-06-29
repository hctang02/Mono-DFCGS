# Stage145 Large-Scale Adapter Training Launch

Date: 2026-06-30

## Goal

Start the model-side quality-rescue path after Stage144: train the dynamic Gaussian anchor adapter on substantially more Stage79 tasks, prioritizing gap4/gap8 q12 middle frames, because higher q-bit does not recover the missing middle-frame PSNR.

## Plan

- Reuse the Stage79 adapter task manifest as the data source.
- Prioritize codec `q12` and gaps `4 8`, matching the corrected middle-frame target gaps from Stage75/142.
- Avoid loading the full train set into GPU memory. Stage80 eagerly materializes all selected tasks, so Stage145 should use lazy per-step task loading with a small eval cache.
- Train from the current Stage65 `rgb_h256` checkpoint, not from scratch.
- Keep test-time/deployable inputs decoder-safe: endpoint Gaussian anchors plus normalized time only. Target RGB is offline training supervision only.
- Store checkpoints and optimizer state outside git under `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage145_large_scale_adapter_training_launch`.
- Store only lightweight summaries/logs under `experiments/stage145_large_scale_adapter_training_launch`.
- Run a syntax check and a short GPU smoke run before launching a longer run.
- Check `nvidia-smi` before every Python execution.

## Expected Configuration

- Train rows available from Stage79 for q12 gap4/8: `3087 + 3604 = 6691`.
- Eval rows available from Stage79 for q12 gap4/8: `1463 + 1707 = 3170`.
- Initial launch should use many train rows but a bounded eval sample, so validation remains feasible.
- Best checkpoint selection should protect gap4/gap8 rather than optimize only overall mean.

## Success Criteria

- Stage145 script compiles and runs a smoke training pass on GPU.
- The stage package records available/selected task counts, eval PSNR by gap, best checkpoint path, and whether the best run improves over the Stage65 initialization on the sampled validation set.
- No heavy checkpoint, `.pt`, or tensor payload is committed.

## Risks

- RGB render-loss training is slow because each step renders a Gaussian anchor.
- Full validation over thousands of rows may be too slow for an interactive stage, so this stage should separate training coverage from eval sample size.
- A short smoke run may not improve PSNR; that would validate infrastructure, not solve the quality gap.

## Status

Completed as a bounded launch/infrastructure stage.

## Implementation

Added:

```text
scripts/run_stage145_large_scale_adapter_training_launch.py
```

Key implementation details:

- Uses Stage79 `stage79_adapter_training_tasks.csv`.
- Supports q12 gap4/gap8 large train selection.
- Keeps eval sample materialized on GPU for stable validation.
- Lazy-loads train tasks per step with bounded LRU caches for anchors and RGB targets.
- Initializes from Stage65 `rgb_h256` best checkpoint.
- Saves `best_adapter.safetensors`, `final_adapter.safetensors`, and `latest_training_state.pt` under the heavy root outside git.

## Runs

GPU checks were performed before both Python executions. GPU3 was idle and used via `CUDA_VISIBLE_DEVICES=3`.

Syntax check and smoke:

```text
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage145_large_scale_adapter_training_launch.py
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage145_large_scale_adapter_training_launch.py --device cuda --summary_root experiments/stage145_large_scale_adapter_training_launch_smoke --heavy_root /data/hctang/tmp/opencode/mono_dfcgs_runs/stage145_large_scale_adapter_training_launch_smoke --max_train_tasks 32 --max_eval_tasks 4 --steps 2 --eval_interval 1 --log_interval 1
```

Main bounded launch:

```text
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage145_large_scale_adapter_training_launch.py --device cuda --max_train_tasks 0 --max_eval_tasks 32 --steps 80 --eval_interval 40 --log_interval 20
```

The smoke summary files were removed and are not part of the stage package.

## Outputs

```text
experiments/stage145_large_scale_adapter_training_launch/stage145_large_scale_adapter_training_launch_summary.json
experiments/stage145_large_scale_adapter_training_launch/stage145_large_scale_adapter_training_launch_report.md
experiments/stage145_large_scale_adapter_training_launch/stage145_large_scale_adapter_training_launch_package.json
experiments/stage145_large_scale_adapter_training_launch/stage145_train_log.csv
experiments/stage145_large_scale_adapter_training_launch/stage145_validation_log.csv
experiments/stage145_large_scale_adapter_training_launch/stage145_initial_eval_rows.csv
experiments/stage145_large_scale_adapter_training_launch/stage145_best_eval_rows.csv
experiments/stage145_large_scale_adapter_training_launch/stage145_final_eval_rows.csv
experiments/stage145_large_scale_adapter_training_launch/stage145_selected_train_rows.csv
experiments/stage145_large_scale_adapter_training_launch/stage145_selected_eval_rows.csv
```

Heavy outputs outside git:

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage145_large_scale_adapter_training_launch/best_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage145_large_scale_adapter_training_launch/final_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage145_large_scale_adapter_training_launch/latest_training_state.pt
```

Output sizes:

- git summary root: `3.0M`.
- heavy root: `7.8M`.

## Results

- Available train rows: `6691` q12 gap4/gap8 rows.
- Selected train rows: `6691`.
- Available eval rows: `3170`.
- Selected eval rows: `32`.
- Selected train sequences: `60`.
- Selected eval sequences: `24`.
- Steps: `80`.
- Eval interval: `40`.
- Initial mean PSNR: `19.30193946735575`.
- Best/final mean PSNR: `19.313366675051686`.
- Mean improvement over initialization on this eval sample: `+0.011427207695936569 dB`.
- Initial min gap margin over linear: `0.03760697804374987`.
- Best/final min gap margin over linear: `0.05164998002879758`.
- Gap4 model PSNR improved from `20.05304674887969` to `20.067089750864742`.
- Gap8 model PSNR improved from `18.45068454829528` to `18.459147189130217`.
- Best step: `80`.

## Conclusion

Stage145 validates the large-scale lazy-load training path and selects the full q12 gap4/gap8 Stage79 train set without blowing GPU memory. The bounded 80-step launch gives only a small sampled improvement, so it does not solve the paper-level middle-frame gap. The next stage should run a longer/resumable training schedule and/or modify the objective/model, with rate-counted side-info fallback still active if feed-forward training saturates.
