# Stage184 Full-Sequence Payload Measurement Execution

Date: 2026-07-01

## Goal

Measure actual full-sequence payloads after Stage183 enumerated the unique workload for uniform gap8, Stage165 adaptive, and uniform gap4 schedules.

## Plan

- Use Stage183 unique keyframe rows to measure q12 Gaussian anchor bitstream bytes.
- Use Stage183 unique residual rows to measure Stage158 q6/keep1.0 entropy residual payload bytes with the counted one-byte half selector.
- Use the Stage158 PSNR-based best-half selector to keep the same policy contract as Stage157/158/167/174/180.
- Also measure schedule/sequence-packed q12 keyframe bitstreams so Stage185 can aggregate a realistic main-anchor rate without per-keyframe container overcount.
- Support smoke limits and resume so the full run can survive interruption.
- Keep heavy temporary bitstreams out of git; commit only CSV/JSON/Markdown summaries.

## Success Criteria

- Unique keyframe rows measured: `596 / 596`.
- Unique residual rows measured: `3472 / 3472`.
- Schedule-packed keyframe rows measured for all `3 * 30` schedule/sequence groups.
- Output package marks the run complete and ready for Stage185 aggregation.

## Execution

- Pre-run `nvidia-smi`: GPU0 and GPU4 were busy; GPU1 was idle.
- Initial smoke with physical `--device cuda:1` hit a CUDA illegal memory access in the renderer cleanup path.
- Reran residual smoke with `CUDA_VISIBLE_DEVICES=1 --device cuda`; this mapped GPU1 to logical `cuda` and passed.
- Batch-size smoke tests passed for `batch_size=2`, `4`, and `8`.
- Full command: `CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage184_full_sequence_payload_measurement_execution.py --device cuda --batch_size 8 --flush_every 16`.

## Results

- Status: `full_sequence_payload_measurement_complete`.
- Unique q12 single-anchor keyframe bitstreams: `596 / 596`, total `409.0400505065918` MiB, mean `719646.9463087248` bytes.
- Unique Stage158 residual payloads: `3472 / 3472`, total `711.6095418930054` MiB, mean `214912.64026497694` bytes.
- Schedule/sequence-packed q12 keyframe bitstreams: `90 / 90`, total `813.8109636306763` MiB, mean `9481584.944444444` bytes.
- Error scan found no error rows beyond CSV headers.

## Outputs

- Package: `experiments/stage184_full_sequence_payload_measurement_execution/stage184_full_sequence_payload_measurement_execution_package.json`
- Report: `experiments/stage184_full_sequence_payload_measurement_execution/stage184_full_sequence_payload_measurement_execution_report.md`
- Summary CSV: `experiments/stage184_full_sequence_payload_measurement_execution/stage184_payload_measurement_summary.csv`
- Unique keyframe payload CSV: `experiments/stage184_full_sequence_payload_measurement_execution/stage184_unique_keyframe_payload_measurements.csv`
- Unique residual payload CSV: `experiments/stage184_full_sequence_payload_measurement_execution/stage184_unique_stage158_residual_payload_measurements.csv`
- Schedule-packed keyframe payload CSV: `experiments/stage184_full_sequence_payload_measurement_execution/stage184_schedule_packed_keyframe_payload_measurements.csv`

## Next Step

- Stage185 should aggregate measured payloads by schedule using schedule-packed keyframe bytes, residual measurement keys from Stage183 frame/schedule rows, and exact schedule metadata bytes.
