# Stage183 Full-Sequence Payload Measurement Protocol

## Scope

This is a protocol-only stage for full-sequence payload measurement. It does not run bitstream measurement or rendering.

## Summary

- Frame/schedule rows: `5997`.
- Unique q12 keyframe payload measurements: `596`.
- Unique Stage158 residual payload measurements: `3472`.
- Frozen selector policy: `rgb_motion_rank_gate_gap8_plus_extra_targets_v1_sampled_candidate`.

## Per-Schedule Counts

| schedule | frames | keyframes | residual rows | keyframe ratio | unique keyframes | unique residuals |
|---|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 1999 | 292 | 1707 | 0.146073 | 292 | 1707 |
| stage165_adaptive | 1999 | 358 | 1641 | 0.179090 | 358 | 1641 |
| uniform_gap4 | 1999 | 536 | 1463 | 0.268134 | 536 | 1463 |

## Execution Contract For Next Stage

- Measure actual q12 keyframe anchor payload for rows in the unique keyframe table.
- Measure Stage158 q6/keep1.0 entropy residual payload plus counted selector byte for rows in the unique residual table.
- Reuse identical measurement keys across schedules where listed by `used_by_schedules`.
- Do not use target dense anchors, target RGB, rendered metrics, or oracle labels as decoder-side inputs.
- Keep any heavy intermediate bitstreams outside git unless they are small CSV/JSON summaries.

## Outputs

- Frame/schedule protocol CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage183_full_sequence_payload_measurement_protocol/stage183_full_sequence_frame_schedule_payload_rows.csv`
- Unique keyframe measurement CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage183_full_sequence_payload_measurement_protocol/stage183_unique_keyframe_payload_measurement_rows.csv`
- Unique residual measurement CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage183_full_sequence_payload_measurement_protocol/stage183_unique_stage158_residual_payload_measurement_rows.csv`
- Summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage183_full_sequence_payload_measurement_protocol/stage183_full_sequence_payload_measurement_summary.csv`
