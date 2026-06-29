# Stage127 Selected Residual Predictor Training Smoke

## Scope

- Trains small per-Gaussian MLP predictors for selected residual values.
- Checkpoints are saved outside git; repo stores only metrics and manifests.
- Target dense anchors are used only for training/eval labels.

## Metrics

| setting | role | train samples | eval samples | train reduction | eval reduction | eval zero MSE | eval pred MSE | checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|---|
| q4_top20 | primary | 61440 | 30720 | 0.131521 | 0.088083 | 0.016353 | 0.014913 | `/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage127_selected_residual_predictor_training_smoke/q4_top20/selected_residual_value_mlp.safetensors` |
| q4_top10 | low_rate | 61440 | 30720 | 0.139464 | 0.102933 | 0.018401 | 0.016507 | `/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage127_selected_residual_predictor_training_smoke/q4_top10/selected_residual_value_mlp.safetensors` |

## Outputs

- metrics CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage127_selected_residual_predictor_training_smoke/stage127_selected_residual_predictor_training_smoke_metrics.csv`
- train log CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage127_selected_residual_predictor_training_smoke/stage127_selected_residual_predictor_training_smoke_train_log.csv`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage127_selected_residual_predictor_training_smoke/stage127_selected_residual_predictor_training_smoke_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage127_selected_residual_predictor_training_smoke/stage127_selected_residual_predictor_training_smoke_report.md`
