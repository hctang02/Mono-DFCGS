# Stage194 All-Keyframe Q12 Upper-Bound

## Decision

- Decision: `all_keyframe_q12_improves_gap2_but_below_target_margin`.
- Complete: `True`.
- Reference best fixed baseline: `uniform_gap2`.

## All-Keyframe RD-Quality

| schedule | MiB/frame | keyframes | PSNR | SSIM | MS-SSIM | LPIPS | dPSNR vs gap2 | dLPIPS vs gap2 | +1dB pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap1 | 0.686173294472 | 1999 | 29.856468 | 0.885744 | 0.988490 | 0.137564 | 0.201653 | -0.014117 | 0 |

## Validation

| item | expected | actual | status |
|---|---:|---:|---|
| protocol_frame_count | 1999 | 1999 | ok |
| unique_keyframe_payload_rows | 1999 | 1999 | ok |
| unique_keyframe_quality_rows | 1999 | 1999 | ok |
| schedule_keyframe_groups | 30 | 30 | ok |
| missing_payload_rows | 0 | 0 | ok |
| missing_quality_rows | 0 | 0 | ok |
| missing_schedule_groups | 0 | 0 | ok |
| final_quality_rows | 1999 | 1999 | ok |

## Interpretation

- `uniform_gap1` is an upper-bound diagnostic, not an adaptive selector or a practical low-rate point.
- If even all-frame q12 keyframes do not reach the +1 dB target over Stage192 `uniform_gap2`, the bottleneck is representation/codec quality rather than selector thresholding.
- Rate uses measured schedule-packed q12 keyframe bitstreams plus fixed-gap metadata for consistency with Stage192.

## Outputs

- Summary CSV: `experiments/stage194_all_keyframe_q12_upper_bound/stage194_all_keyframe_q12_summary.csv`
- Final quality rows: `experiments/stage194_all_keyframe_q12_upper_bound/stage194_all_keyframe_quality_rows.csv`
- Keyframe quality CSV: `experiments/stage194_all_keyframe_q12_upper_bound/stage194_unique_keyframe_quality_metrics.csv`
- Schedule-packed payload CSV: `experiments/stage194_all_keyframe_q12_upper_bound/stage194_schedule_packed_keyframe_payload_measurements.csv`
