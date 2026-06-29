# Stage126 Selected Residual Predictor Dataset Package

Date: 2026-06-29

## Goal

Prepare a small, reusable dataset package for training a dedicated selected residual value predictor.

## Plan

- Add a feature builder for selected residual value prediction.
- Add a Stage126 package script that selects balanced train/eval tasks from the Stage79 manifest.
- Use deterministic endpoint-diff selected indices for Stage123 primary `q4_top20` and low-rate `q4_top10`.
- Use target dense anchors only to compute training labels and normalization stats.
- Save CSV/JSON/Markdown metadata and normalization stats only; do not save per-Gaussian tensors, anchors, or checkpoints.
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
scripts/run_stage126_selected_residual_predictor_dataset_package.py
```

The feature builder uses only decoder-available left/right/base attributes and normalized time. Dense target anchors are used only to compute selected residual labels and normalization stats.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile mono_dfcgs/residual_value_predictor.py scripts/run_stage126_selected_residual_predictor_dataset_package.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage126_selected_residual_predictor_dataset_package.py
```

## Outputs

```text
experiments/stage126_selected_residual_predictor_dataset_package/stage126_selected_residual_predictor_dataset_rows.csv
experiments/stage126_selected_residual_predictor_dataset_package/stage126_selected_residual_predictor_dataset_summary.csv
experiments/stage126_selected_residual_predictor_dataset_package/stage126_selected_residual_predictor_train_stats.csv
experiments/stage126_selected_residual_predictor_dataset_package/stage126_selected_residual_predictor_train_stats.json
experiments/stage126_selected_residual_predictor_dataset_package/stage126_selected_residual_predictor_dataset_package.json
experiments/stage126_selected_residual_predictor_dataset_package/stage126_selected_residual_predictor_dataset_report.md
```

## Results

| split | setting | role | keep | tasks | samples | feature dim | residual dim | feature rms | label rms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| eval | q4_top10 | low_rate | 0.1 | 60 | 221160 | 56 | 13 | 0.4686119039853414 | 0.13427672560016315 |
| eval | q4_top20 | primary | 0.2 | 60 | 442380 | 56 | 13 | 0.461015955110391 | 0.12549363765865565 |
| train | q4_top10 | low_rate | 0.1 | 120 | 442320 | 56 | 13 | 0.4699477088948091 | 0.14150386080145835 |
| train | q4_top20 | primary | 0.2 | 120 | 884760 | 56 | 13 | 0.46037831778327626 | 0.1297483650036156 |

Train normalization stats:

| setting | train tasks | train samples | feature rms | label rms |
|---|---:|---:|---:|---:|
| q4_top20 | 120 | 884760 | 0.46078898367726934 | 0.13244603771996774 |
| q4_top10 | 120 | 442320 | 0.4704328888850035 | 0.1438315651016054 |

## Conclusion

- Stage126 provides the manifest and normalization stats needed for Stage127 training.
- It saves only metadata and aggregate stats, not per-Gaussian tensors.
- Target dense anchors remain train-label/offline-stat inputs only.
