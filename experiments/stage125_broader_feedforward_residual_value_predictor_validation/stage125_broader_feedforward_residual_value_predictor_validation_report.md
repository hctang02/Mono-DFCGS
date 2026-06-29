# Stage125 Broader Feed-Forward Residual Value Predictor Validation

## Scope

- Broadens Stage124 from 12 eval tasks to a 60-task rendered validation slice.
- Uses `adapter_delta_selected_v1`: `adapter_attrs - linear_attrs` at deterministic endpoint-diff selected indices.
- Transmits no residual payload bytes and no selected-index bytes.
- Does not load target dense anchors; target RGB is used only for offline rendered metrics.

## Summary

| predictor | setting | role | keep | tasks | rate | base PSNR | full predictor PSNR | selected PSNR | delta base | delta full | positives | near full |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adapter_delta_selected_v1 | q4_top10 | low_rate | 0.1 | 60 | 0.117298 | 18.950803 | 19.187643 | 18.994813 | 0.044010 | -0.192830 | 51/60 | 13/60 |
| adapter_delta_selected_v1 | q4_top20 | primary | 0.2 | 60 | 0.117298 | 18.950803 | 19.187643 | 19.010259 | 0.059456 | -0.177384 | 48/60 | 13/60 |

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage125_broader_feedforward_residual_value_predictor_validation/stage125_broader_feedforward_residual_value_predictor_validation_rows.csv`
- summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage125_broader_feedforward_residual_value_predictor_validation/stage125_broader_feedforward_residual_value_predictor_validation_summary.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage125_broader_feedforward_residual_value_predictor_validation/stage125_broader_feedforward_residual_value_predictor_validation_summary.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage125_broader_feedforward_residual_value_predictor_validation/stage125_broader_feedforward_residual_value_predictor_validation_report.md`
