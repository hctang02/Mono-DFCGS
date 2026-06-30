# Stage184 Full-Sequence Payload Measurement Execution

## Scope

This stage measures actual q12 keyframe bitstream bytes and Stage158 q6/keep1.0 residual payload bytes from the Stage183 protocol.

## Summary

| item | expected | measured | complete | total MiB | mean bytes |
|---|---:|---:|---:|---:|---:|
| unique_keyframe_single_anchor_bitstreams | 596 | 596 | 1 | 409.040051 | 719646.946 |
| unique_stage158_residual_payloads | 3472 | 3472 | 1 | 711.609542 | 214912.640 |
| schedule_sequence_packed_keyframe_bitstreams | 90 | 90 | 1 | 813.810964 | 9481584.944 |

## Policy Contract

- Keyframes are encoded as q12 Gaussian anchor bitstreams using the Mono-DFCGS anchor container.
- Residual rows use Stage158 `best_half_selector` with PSNR-based half selection, q6 residual entropy payload, keep fraction 1.0, and one counted selector byte.
- Target dense anchors and target RGB are encoder-side measurement inputs only; decoder-side inputs remain the transmitted schedule, keyframe bitstreams, normalized time, residual payload, and half selector byte.

## Outputs

- Unique keyframe payload CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage184_full_sequence_payload_measurement_execution/stage184_unique_keyframe_payload_measurements.csv`
- Unique residual payload CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage184_full_sequence_payload_measurement_execution/stage184_unique_stage158_residual_payload_measurements.csv`
- Schedule-packed keyframe payload CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage184_full_sequence_payload_measurement_execution/stage184_schedule_packed_keyframe_payload_measurements.csv`
- Summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage184_full_sequence_payload_measurement_execution/stage184_payload_measurement_summary.csv`
