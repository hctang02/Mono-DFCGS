# Stage191 Fixed-Gap Expansion Protocol

## Scope

This stage builds a protocol for expanded fixed-gap full-sequence measurement. It does not measure payloads or quality.

## Schedules

| schedule | frames | keyframes | residual rows | keyframe ratio | metadata bytes |
|---|---:|---:|---:|---:|---:|
| uniform_gap2 | 1999 | 1025 | 974 | 0.512756 | 1 |
| uniform_gap4 | 1999 | 536 | 1463 | 0.268134 | 1 |
| uniform_gap6 | 1999 | 372 | 1627 | 0.186093 | 1 |
| uniform_gap8 | 1999 | 292 | 1707 | 0.146073 | 1 |
| uniform_gap16 | 1999 | 169 | 1830 | 0.084542 | 1 |
| stage165_adaptive | 1999 | 358 | 1641 | 0.179090 | 327 |

## Reuse Coverage For Stage192

| scope | expected | existing ok | missing | reuse fraction |
|---|---:|---:|---:|---:|
| payload_single_keyframe | 1065 | 596 | 469 | 0.559624 |
| payload_residual | 7791 | 3472 | 4319 | 0.445642 |
| payload_schedule_packed_keyframe_group | 180 | 90 | 90 | 0.500000 |
| quality_single_keyframe | 1065 | 596 | 469 | 0.559624 |
| quality_residual | 7791 | 3472 | 4319 | 0.445642 |

## Decision

- Decision: `measure_expanded_fixed_gap_baselines_next`.
- Stage192 should measure the missing gap2/gap6/gap16 payload and quality rows, reusing existing gap4/gap8/adaptive rows.
- This expanded baseline set is required before claiming the selector beats fixed-gap schedules.

## Outputs

- Frame/schedule rows: `experiments/stage191_fixed_gap_expansion_protocol/stage191_expanded_fixed_gap_frame_schedule_rows.csv`
- Unique keyframe rows: `experiments/stage191_fixed_gap_expansion_protocol/stage191_unique_keyframe_measurement_rows.csv`
- Unique residual rows: `experiments/stage191_fixed_gap_expansion_protocol/stage191_unique_stage158_residual_measurement_rows.csv`
- Summary CSV: `experiments/stage191_fixed_gap_expansion_protocol/stage191_expanded_fixed_gap_schedule_summary.csv`
- Reuse coverage CSV: `experiments/stage191_fixed_gap_expansion_protocol/stage191_existing_measurement_reuse_coverage.csv`
- Missing measurements CSV: `experiments/stage191_fixed_gap_expansion_protocol/stage191_missing_measurements_for_stage192.csv`
