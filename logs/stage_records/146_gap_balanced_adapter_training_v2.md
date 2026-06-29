# Stage146 Gap-Balanced Adapter Training V2

Date: 2026-06-30

## Goal

Continue the model-side middle-frame quality rescue after Stage145 by running a longer q12 gap4/gap8 adapter training schedule with broader validation, initialized from the Stage145 best checkpoint.

## Plan

- Reuse the Stage145 lazy-load trainer, but parameterize output filenames so Stage146 does not overwrite Stage145 outputs.
- Use Stage79 q12 gap4/gap8 train rows as the train source.
- Initialize from `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage145_large_scale_adapter_training_launch/best_adapter.safetensors`.
- Use a bounded but larger eval sample than Stage145 to reduce sample noise.
- Keep gap-balanced weighting and `min_gap_margin` selection to protect both gap4 and gap8.
- Store heavy checkpoints under `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage146_gap_balanced_adapter_training_v2`.
- Store only lightweight summaries and CSV logs under `experiments/stage146_gap_balanced_adapter_training_v2`.
- Check `nvidia-smi` before Python execution and use the idle GPU.

## Candidate Command

```text
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage145_large_scale_adapter_training_launch.py --stage 146 --summary_name stage146_gap_balanced_adapter_training_v2 --output_prefix stage146 --device cuda --summary_root experiments/stage146_gap_balanced_adapter_training_v2 --heavy_root /data/hctang/tmp/opencode/mono_dfcgs_runs/stage146_gap_balanced_adapter_training_v2 --init_checkpoint /data/hctang/tmp/opencode/mono_dfcgs_runs/stage145_large_scale_adapter_training_launch/best_adapter.safetensors --max_train_tasks 0 --max_eval_tasks 64 --steps 240 --eval_interval 80 --log_interval 40
```

## Success Criteria

- Script compiles after output-name parameterization.
- Stage146 runs to completion on GPU.
- Package records whether longer training improves over the Stage145 best initialization on a larger eval sample.
- Heavy checkpoint/state files are not committed.

## Risk

If the longer run still yields only tiny gains, the current adapter architecture/objective is likely saturated and the next step should shift to architecture/objective changes or rate-counted side-info fallback.

## Status

Completed.

## Implementation

Updated:

```text
scripts/run_stage145_large_scale_adapter_training_launch.py
```

The trainer now accepts:

- `--stage_label`
- `--mode`
- `--output_prefix`
- `--summary_name`

This lets Stage146 reuse the Stage145 lazy-load trainer without overwriting Stage145 filenames.

## Run

GPU checks were performed before py_compile and before the training run. GPU3 was selected for the final training run.

Syntax check:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage145_large_scale_adapter_training_launch.py
```

Training:

```text
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage145_large_scale_adapter_training_launch.py --stage 146 --stage_label "Stage146 Gap-Balanced Adapter Training V2" --mode "gap-balanced q12 gap4/gap8 adapter training v2" --summary_name stage146_gap_balanced_adapter_training_v2 --output_prefix stage146 --device cuda --summary_root experiments/stage146_gap_balanced_adapter_training_v2 --heavy_root /data/hctang/tmp/opencode/mono_dfcgs_runs/stage146_gap_balanced_adapter_training_v2 --init_checkpoint /data/hctang/tmp/opencode/mono_dfcgs_runs/stage145_large_scale_adapter_training_launch/best_adapter.safetensors --max_train_tasks 0 --max_eval_tasks 64 --steps 240 --eval_interval 80 --log_interval 40
```

## Outputs

```text
experiments/stage146_gap_balanced_adapter_training_v2/stage146_gap_balanced_adapter_training_v2_summary.json
experiments/stage146_gap_balanced_adapter_training_v2/stage146_gap_balanced_adapter_training_v2_report.md
experiments/stage146_gap_balanced_adapter_training_v2/stage146_gap_balanced_adapter_training_v2_package.json
experiments/stage146_gap_balanced_adapter_training_v2/stage146_train_log.csv
experiments/stage146_gap_balanced_adapter_training_v2/stage146_validation_log.csv
experiments/stage146_gap_balanced_adapter_training_v2/stage146_initial_eval_rows.csv
experiments/stage146_gap_balanced_adapter_training_v2/stage146_best_eval_rows.csv
experiments/stage146_gap_balanced_adapter_training_v2/stage146_final_eval_rows.csv
experiments/stage146_gap_balanced_adapter_training_v2/stage146_selected_train_rows.csv
experiments/stage146_gap_balanced_adapter_training_v2/stage146_selected_eval_rows.csv
```

Heavy outputs outside git:

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage146_gap_balanced_adapter_training_v2/best_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage146_gap_balanced_adapter_training_v2/final_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage146_gap_balanced_adapter_training_v2/latest_training_state.pt
```

Output sizes:

- git summary root: `3.0M`.
- heavy root: `7.8M`.

## Results

- Available train rows: `6691` q12 gap4/gap8 rows.
- Selected train rows: `6691`.
- Available eval rows: `3170`.
- Selected eval rows: `64`.
- Selected eval sequences: `30`.
- Steps: `240`.
- Eval interval: `80`.
- Init checkpoint: Stage145 best adapter.
- Initial mean PSNR: `19.697228869272262`.
- Best mean PSNR: `19.697228869272262`.
- Final mean PSNR: `19.677331542393684`.
- Best step: `0`.
- Initial/best min gap margin: `0.12251526389602437`.
- Final min gap margin: `0.10497032571835363`.
- Gap4 model PSNR regressed from `20.05174662536213` to `20.03420168718447`.
- Gap8 model PSNR regressed from `19.31983899988628` to `19.297437517293815`.

## Conclusion

Stage146 shows that continuing the same RGB render-loss objective/schedule from the Stage145 best checkpoint does not improve a broader 64-row validation sample. The best checkpoint remains the initialization at step `0`, and the final checkpoint regresses. The next step should not be blind longer training with the same settings; it should change objective/model selection or move to a rate-counted side-info fallback.
