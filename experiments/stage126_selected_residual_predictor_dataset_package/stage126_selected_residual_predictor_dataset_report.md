# Stage126 Selected Residual Predictor Dataset Package

## Scope

- Builds task-level manifests and normalization stats for a dedicated selected residual value predictor.
- Target dense anchors are used only for train/eval labels and aggregate stats.
- No per-Gaussian tensors, anchors, or checkpoints are saved.

## Dataset Summary

| split | setting | role | keep | tasks | samples | feature dim | residual dim | feature rms | label rms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| eval | q4_top10 | low_rate | 0.1 | 60 | 221160 | 56 | 13 | 0.468612 | 0.134277 |
| eval | q4_top20 | primary | 0.2 | 60 | 442380 | 56 | 13 | 0.461016 | 0.125494 |
| train | q4_top10 | low_rate | 0.1 | 120 | 442320 | 56 | 13 | 0.469948 | 0.141504 |
| train | q4_top20 | primary | 0.2 | 120 | 884760 | 56 | 13 | 0.460378 | 0.129748 |

## Train Normalization Stats

| setting | role | train tasks | train samples | feature rms | label rms |
|---|---|---:|---:|---:|---:|
| q4_top20 | primary | 120 | 884760 | 0.460789 | 0.132446 |
| q4_top10 | low_rate | 120 | 442320 | 0.470433 | 0.143832 |

## Outputs

- dataset rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage126_selected_residual_predictor_dataset_package/stage126_selected_residual_predictor_dataset_rows.csv`
- dataset summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage126_selected_residual_predictor_dataset_package/stage126_selected_residual_predictor_dataset_summary.csv`
- stats CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage126_selected_residual_predictor_dataset_package/stage126_selected_residual_predictor_train_stats.csv`
- stats JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage126_selected_residual_predictor_dataset_package/stage126_selected_residual_predictor_train_stats.json`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage126_selected_residual_predictor_dataset_package/stage126_selected_residual_predictor_dataset_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage126_selected_residual_predictor_dataset_package/stage126_selected_residual_predictor_dataset_report.md`
