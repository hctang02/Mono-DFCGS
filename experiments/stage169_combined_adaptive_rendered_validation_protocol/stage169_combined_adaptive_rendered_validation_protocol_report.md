# Stage169 Combined Adaptive Rendered Validation Protocol

## Scope

This is a protocol-only stage. It selects targets and schedule rows for Stage170 without running any rendering.
Stage170 should reuse existing Stage167/168 rows and render only missing schedule rows.

## Target Counts

- `false_negative_residual`: `8` targets
- `high_payload_residual_control`: `4` targets
- `positive_promoted`: `14` targets

## Summary

| category | targets | schedule rows | existing | new renders | keyframe markers | mean payload | hard | high payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| false_negative_residual | 8 | 24 | 24 | 0 | 0 | 156341.250 | 8 | 0 |
| high_payload_residual_control | 4 | 12 | 0 | 12 | 0 | 241017.500 | 0 | 4 |
| positive_promoted | 14 | 42 | 18 | 14 | 10 | 234956.071 | 12 | 13 |

## Stage170 Contract

- Render only rows with `requires_stage170_render=1`.
- For adaptive promoted targets, record `target_keyframe_no_middle_render`; do not claim middle-render metrics.
- Reuse Stage167/168 smoke rows when `existing_source` is present.
- Count adaptive schedule metadata and residual payloads consistently in the Stage170 report.
- Decoder receives transmitted schedule/keyframes only; RGB/motion selector features remain encoder-side only.

## Outputs

- Targets CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage169_combined_adaptive_rendered_validation_protocol/stage169_combined_validation_targets.csv`
- Schedule row CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage169_combined_adaptive_rendered_validation_protocol/stage169_combined_validation_schedule_rows.csv`
- Summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage169_combined_adaptive_rendered_validation_protocol/stage169_combined_validation_summary.csv`
