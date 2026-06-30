# Stage179 Broader Sampled Adaptive Validation Protocol

## Scope

This is a protocol-only stage. It expands the Stage174 medium validation target set and prepares Stage180 execution inputs.
Stage179 performs no rendering and writes no heavy media.

## Target Counts

- `broader_false_negative_residual`: `1` targets
- `broader_normal_residual_control`: `5` targets
- `broader_positive_promoted`: `18` targets
- `broader_sequence_coverage_probe`: `9` targets
- `broader_weak_sequence_probe`: `7` targets
- `false_negative_residual`: `8` targets
- `high_payload_residual_control`: `4` targets
- `high_payload_residual_control_extension`: `8` targets
- `normal_residual_control`: `4` targets
- `positive_promoted`: `14` targets
- `positive_promoted_extension`: `8` targets
- `selector_false_positive_keyframe_control`: `4` targets

## Work Summary

- Targets: `90`.
- Schedule rows: `270`.
- Stage174 core targets retained: `50`.
- New targets beyond Stage174: `40`.
- Existing middle metric rows: `118`.
- Existing keyframe metric rows: `32`.
- Stage180 middle renders required: `88`.
- Stage180 q12 keyframe metrics required: `32`.

## Category Summary

| category | targets | core | schedule rows | existing middle | existing keyframe | new renders | new keyframes | keyframe rows | mean payload | mean PSNR | mean LPIPS | hard | high payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| broader_false_negative_residual | 1 | 0 | 3 | 0 | 0 | 3 | 0 | 0 | 220263.000 | 32.332240 | 0.108919 | 0 | 1 |
| broader_normal_residual_control | 5 | 0 | 15 | 0 | 0 | 15 | 0 | 0 | 206396.000 | 32.236286 | 0.142569 | 0 | 0 |
| broader_positive_promoted | 18 | 0 | 54 | 0 | 0 | 34 | 20 | 20 | 242724.167 | 28.708515 | 0.161411 | 1 | 17 |
| broader_sequence_coverage_probe | 9 | 0 | 27 | 0 | 0 | 20 | 7 | 7 | 226055.556 | 29.755813 | 0.156365 | 0 | 7 |
| broader_weak_sequence_probe | 7 | 0 | 21 | 0 | 0 | 16 | 5 | 5 | 217299.714 | 31.211105 | 0.149776 | 0 | 5 |
| false_negative_residual | 8 | 8 | 24 | 23 | 1 | 0 | 0 | 1 | 156341.250 | 26.161450 | 0.211616 | 8 | 0 |
| high_payload_residual_control | 4 | 4 | 12 | 12 | 0 | 0 | 0 | 0 | 241017.500 | 31.089584 | 0.165360 | 0 | 4 |
| high_payload_residual_control_extension | 8 | 8 | 24 | 23 | 1 | 0 | 0 | 1 | 224830.375 | 29.631000 | 0.168569 | 0 | 8 |
| normal_residual_control | 4 | 4 | 12 | 12 | 0 | 0 | 0 | 0 | 194930.500 | 33.427849 | 0.077536 | 0 | 0 |
| positive_promoted | 14 | 14 | 42 | 25 | 17 | 0 | 0 | 17 | 234956.071 | 28.317968 | 0.221776 | 12 | 13 |
| positive_promoted_extension | 8 | 8 | 24 | 15 | 9 | 0 | 0 | 9 | 220645.750 | 26.728115 | 0.224265 | 8 | 3 |
| selector_false_positive_keyframe_control | 4 | 4 | 12 | 8 | 4 | 0 | 0 | 4 | 150710.000 | 30.090534 | 0.173586 | 0 | 0 |

## Reuse And New Work

| source type | source | rows |
|---|---|---:|
| existing_keyframe_metric | stage177:stage168 | 2 |
| existing_keyframe_metric | stage177:stage170 | 18 |
| existing_keyframe_metric | stage177:stage174_keyframe_marker | 12 |
| existing_middle_metric | stage174:stage168 | 4 |
| existing_middle_metric | stage174:stage170 | 60 |
| existing_middle_metric | stage174:stage174_rendered | 54 |
| new_work | stage180_middle_render | 88 |
| new_work | stage180_q12_keyframe_metric | 32 |

## Stage180 Contract

- Render only middle rows with `requires_stage180_render=1`.
- Render q12 target keyframe metrics only for keyframe rows with `requires_stage180_keyframe_metric=1`.
- Reuse Stage174 middle metrics and Stage177 final-quality keyframe metrics where available.
- Keep Stage158 `streamsplat_guided_half_anchor_entropy_residual_v1` fixed.
- Keep heavy contact sheets outside git if Stage180 exports visuals.

## Outputs

- Targets CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage179_broader_sampled_adaptive_validation_protocol/stage179_broader_validation_targets.csv`
- Schedule rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage179_broader_sampled_adaptive_validation_protocol/stage179_broader_validation_schedule_rows.csv`
- Summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage179_broader_sampled_adaptive_validation_protocol/stage179_broader_validation_summary.csv`
- Source summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage179_broader_sampled_adaptive_validation_protocol/stage179_broader_validation_source_summary.csv`
