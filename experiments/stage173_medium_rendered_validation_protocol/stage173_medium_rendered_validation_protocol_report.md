# Stage173 Medium Rendered Validation Protocol

## Scope

This is a protocol-only stage. It selects medium-scale targets and schedule rows for Stage174 without rendering.
Stage174 should reuse Stage167/168/170 rows and render only rows with `requires_stage174_render=1`.

## Target Counts

- `false_negative_residual`: `8` targets
- `high_payload_residual_control`: `4` targets
- `high_payload_residual_control_extension`: `8` targets
- `normal_residual_control`: `4` targets
- `positive_promoted`: `14` targets
- `positive_promoted_extension`: `8` targets
- `selector_false_positive_keyframe_control`: `4` targets

## Summary

| category | targets | schedule rows | existing | new renders | keyframe markers | mean payload | mean PSNR | mean LPIPS | hard | high payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| false_negative_residual | 8 | 24 | 24 | 0 | 0 | 156341.250 | 26.161450 | 0.211616 | 8 | 0 |
| high_payload_residual_control | 4 | 12 | 12 | 0 | 0 | 241017.500 | 31.089584 | 0.165360 | 0 | 4 |
| high_payload_residual_control_extension | 8 | 24 | 0 | 23 | 1 | 224830.375 | 29.631000 | 0.168569 | 0 | 8 |
| normal_residual_control | 4 | 12 | 0 | 12 | 0 | 194930.500 | 33.427849 | 0.077536 | 0 | 0 |
| positive_promoted | 14 | 42 | 42 | 0 | 0 | 234956.071 | 28.317968 | 0.221776 | 12 | 13 |
| positive_promoted_extension | 8 | 24 | 6 | 11 | 7 | 220645.750 | 26.728115 | 0.224265 | 8 | 3 |
| selector_false_positive_keyframe_control | 4 | 12 | 0 | 8 | 4 | 150710.000 | 30.090534 | 0.173586 | 0 | 0 |

## Stage174 Contract

- Render only rows with `requires_stage174_render=1`.
- Reuse rows marked by `existing_source` from Stage167/168/170.
- For target keyframes, record `target_keyframe_no_middle_render`; do not claim middle-render metrics.
- Keep Stage158 `streamsplat_guided_half_anchor_entropy_residual_v1` fixed.
- Keep heavy contact sheets outside repo.

## Outputs

- Targets CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage173_medium_rendered_validation_protocol/stage173_medium_validation_targets.csv`
- Schedule rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage173_medium_rendered_validation_protocol/stage173_medium_validation_schedule_rows.csv`
- Summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage173_medium_rendered_validation_protocol/stage173_medium_validation_summary.csv`
