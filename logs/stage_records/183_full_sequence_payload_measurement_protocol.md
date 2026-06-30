# Stage183 Full-Sequence Payload Measurement Protocol

Date: 2026-06-30

## Goal

Define the exact full-sequence payload measurement protocol needed after Stage182 freezes the current adaptive selector candidate.

## Plan

- Enumerate all frame/schedule rows for uniform gap8, Stage165 adaptive, and uniform gap4 across the Stage165 `30` DAVIS sequences / `1999` frames.
- Mark keyframe rows that require q12 keyframe bitstream measurement.
- Mark non-keyframe rows that require Stage158 residual payload encode/measurement.
- Generate unique keyframe measurement rows and unique residual measurement rows to avoid duplicate work across schedules where possible.
- Summarize per-schedule row counts and unique measurement counts.
- Do not run heavy rendering or payload encoding in this stage.

## Success Criteria

- Protocol rows cover `1999 * 3` frame/schedule rows.
- Per-schedule keyframe/non-keyframe counts match Stage165 counts.
- Unique keyframe and residual measurement tables are produced for the next execution stage.

## Execution

- Pre-run `nvidia-smi`: GPU0 was busy; GPUs1/2/3/5/6/7 were idle. This stage did not run GPU rendering or payload encoding.
- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage183_full_sequence_payload_measurement_protocol.py`

## Results

- Status: `full_sequence_payload_measurement_protocol_packaged`.
- Frame/schedule rows: `5997` (`1999` frames x `3` schedules).
- Unique q12 keyframe payload measurements: `596`.
- Unique Stage158 residual payload measurements: `3472`.
- Frozen selector policy: `rgb_motion_rank_gate_gap8_plus_extra_targets_v1_sampled_candidate`.

Per-schedule counts:

| schedule | frames | keyframes | residual rows | keyframe ratio | unique keyframes | unique residuals |
|---|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 1999 | 292 | 1707 | 0.146073 | 292 | 1707 |
| stage165_adaptive | 1999 | 358 | 1641 | 0.179090 | 358 | 1641 |
| uniform_gap4 | 1999 | 536 | 1463 | 0.268134 | 536 | 1463 |

## Outputs

- Package: `experiments/stage183_full_sequence_payload_measurement_protocol/stage183_full_sequence_payload_measurement_protocol_package.json`
- Report: `experiments/stage183_full_sequence_payload_measurement_protocol/stage183_full_sequence_payload_measurement_protocol_report.md`
- Frame/schedule CSV: `experiments/stage183_full_sequence_payload_measurement_protocol/stage183_full_sequence_frame_schedule_payload_rows.csv`
- Unique keyframe CSV: `experiments/stage183_full_sequence_payload_measurement_protocol/stage183_unique_keyframe_payload_measurement_rows.csv`
- Unique residual CSV: `experiments/stage183_full_sequence_payload_measurement_protocol/stage183_unique_stage158_residual_payload_measurement_rows.csv`
- Summary CSV: `experiments/stage183_full_sequence_payload_measurement_protocol/stage183_full_sequence_payload_measurement_summary.csv`

## Next Step

- Stage184 should run actual full-sequence payload measurement from the Stage183 unique measurement CSVs.
- Measure q12 keyframe bitstreams for unique keyframe rows.
- Measure Stage158 q6/keep1.0 entropy residual payload plus counted selector byte for unique residual rows.
- Reuse identical measurement keys across schedules.
- Keep heavy bitstreams or temporary tensors outside git; commit only lightweight measurement summaries and logs.
