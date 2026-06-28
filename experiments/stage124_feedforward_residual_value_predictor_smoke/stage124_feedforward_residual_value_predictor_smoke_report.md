# Stage124 Feed-Forward Residual Value Predictor Smoke

## Scope

- Uses the Stage65 adapter as a feed-forward residual value predictor over a linear base.
- Predicted residual values are `adapter_attrs - linear_attrs` at deterministic endpoint-diff selected indices.
- No residual payload bytes and no selected-index bytes are transmitted in this smoke.
- Target dense anchors are not used; target RGB is used only for offline rendered metrics.

## Summary

| predictor | setting | role | keep | tasks | rate | base PSNR | full predictor PSNR | selected PSNR | delta base | delta full | near full |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adapter_delta_selected_v1 | q4_top10 | low_rate | 0.1 | 12 | 0.129243 | 20.079640 | 20.182425 | 20.107503 | 0.027863 | -0.074922 | 3/12 |
| adapter_delta_selected_v1 | q4_top20 | primary | 0.2 | 12 | 0.129243 | 20.079640 | 20.182425 | 20.099010 | 0.019371 | -0.083415 | 3/12 |

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage124_feedforward_residual_value_predictor_smoke/stage124_feedforward_residual_value_predictor_smoke_rows.csv`
- summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage124_feedforward_residual_value_predictor_smoke/stage124_feedforward_residual_value_predictor_smoke_summary.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage124_feedforward_residual_value_predictor_smoke/stage124_feedforward_residual_value_predictor_smoke_summary.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage124_feedforward_residual_value_predictor_smoke/stage124_feedforward_residual_value_predictor_smoke_report.md`
