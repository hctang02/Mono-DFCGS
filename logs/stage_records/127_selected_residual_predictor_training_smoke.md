# Stage127 Selected Residual Predictor Training Smoke

Date: 2026-06-29

## Goal

Train a small dedicated selected residual value predictor smoke using the Stage126 dataset package.

## Plan

- Add a lightweight `SelectedResidualValueMLP` model.
- Add a Stage127 training smoke script that samples per-Gaussian selected residual features/labels from Stage126 rows.
- Train separate small models for `q4_top20` and `q4_top10`.
- Save model checkpoints only to an external non-git path.
- Commit only train logs, metrics, package JSON, report, and checkpoint manifest paths.
- Check `nvidia-smi` before running Python and use an idle GPU.

## Status

Completed.

## Implementation

Updated:

```text
mono_dfcgs/residual_value_predictor.py
```

Added:

```text
scripts/run_stage127_selected_residual_predictor_training_smoke.py
```

The script trains separate small MLPs for q4/top20 and q4/top10 using sampled selected-Gaussian features from Stage126. Checkpoints are saved outside git.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile mono_dfcgs/residual_value_predictor.py scripts/run_stage127_selected_residual_predictor_training_smoke.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage127_selected_residual_predictor_training_smoke.py
```

## Outputs

Repo outputs:

```text
experiments/stage127_selected_residual_predictor_training_smoke/stage127_selected_residual_predictor_training_smoke_metrics.csv
experiments/stage127_selected_residual_predictor_training_smoke/stage127_selected_residual_predictor_training_smoke_train_log.csv
experiments/stage127_selected_residual_predictor_training_smoke/stage127_selected_residual_predictor_training_smoke_package.json
experiments/stage127_selected_residual_predictor_training_smoke/stage127_selected_residual_predictor_training_smoke_report.md
```

External checkpoint root:

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage127_selected_residual_predictor_training_smoke
```

## Results

| setting | role | train samples | eval samples | train reduction | eval reduction | eval zero MSE | eval pred MSE | checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|---|
| q4_top20 | primary | 61440 | 30720 | 0.1315208011384117 | 0.08808295024199653 | 0.016353415325284004 | 0.01491295825690031 | `/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage127_selected_residual_predictor_training_smoke/q4_top20/selected_residual_value_mlp.safetensors` |
| q4_top10 | low_rate | 61440 | 30720 | 0.13946377906555096 | 0.10293283119315721 | 0.018400665372610092 | 0.016506632789969444 | `/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage127_selected_residual_predictor_training_smoke/q4_top10/selected_residual_value_mlp.safetensors` |

## Conclusion

- Dedicated selected residual MLP training is functional.
- Both settings reduce eval residual MSE versus zero residual labels.
- Stage128 should package these checkpoints into an integrated predictor codec manifest without committing checkpoint files.
