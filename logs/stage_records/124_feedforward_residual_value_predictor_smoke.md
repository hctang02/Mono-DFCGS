# Stage124 Feed-Forward Residual Value Predictor Smoke

Date: 2026-06-29

## Goal

Create the first no-teacher, decoder-side feed-forward residual value predictor smoke.

## Implementation

Added:

```text
mono_dfcgs/residual_value_predictor.py
scripts/run_stage124_feedforward_residual_value_predictor_smoke.py
```

The predictor smoke uses the existing Stage65 adapter as a feed-forward residual value predictor over a linear base. Predicted selected residual values are `adapter_attrs - linear_attrs` at deterministic endpoint-diff selected indices.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`. The first run exposed Stage78 rate table field-name mismatch; the script was fixed and rerun after another GPU check.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile mono_dfcgs/residual_value_predictor.py scripts/run_stage124_feedforward_residual_value_predictor_smoke.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage124_feedforward_residual_value_predictor_smoke.py
```

## Outputs

```text
experiments/stage124_feedforward_residual_value_predictor_smoke/stage124_feedforward_residual_value_predictor_smoke_rows.csv
experiments/stage124_feedforward_residual_value_predictor_smoke/stage124_feedforward_residual_value_predictor_smoke_summary.csv
experiments/stage124_feedforward_residual_value_predictor_smoke/stage124_feedforward_residual_value_predictor_smoke_summary.json
experiments/stage124_feedforward_residual_value_predictor_smoke/stage124_feedforward_residual_value_predictor_smoke_report.md
```

## Configuration

| item | value |
|---|---:|
| predictor | adapter_delta_selected_v1 |
| task count | 12 |
| row count | 24 |
| base method | linear |
| full predictor method | stage65_adapter |
| residual payload bytes | 0 |
| selected-index payload bytes | 0 |
| target dense anchors | not loaded or used |

## Results

| setting | role | keep | tasks | rate | base PSNR | full adapter PSNR | selected PSNR | delta base | delta full | positives | near full |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q4_top10 | low_rate | 0.1 | 12 | 0.1292426995089139 | 20.079639580609125 | 20.18242498795657 | 20.107502942874657 | 0.027863362265533247 | -0.0749220450819122 | 9/12 | 3/12 |
| q4_top20 | primary | 0.2 | 12 | 0.1292426995089139 | 20.079639580609125 | 20.18242498795657 | 20.099010301231182 | 0.019370720622054066 | -0.08341468672539139 | 9/12 | 3/12 |

## Conclusion

- The first no-teacher feed-forward residual value predictor smoke is functional.
- It uses no residual side-info payload and no transmitted selected indices.
- It gives a small positive rendered gain over linear base on this 12-task smoke.
- It remains below full Stage65 adapter quality, so broader validation or a dedicated selected residual value predictor is needed next.
