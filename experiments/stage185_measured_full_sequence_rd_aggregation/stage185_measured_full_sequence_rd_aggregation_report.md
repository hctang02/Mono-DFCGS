# Stage185 Measured Full-Sequence RD Aggregation

## Decision

- Decision: `adaptive_measured_rate_not_lower_than_gap8`.
- Adaptive measured total delta vs gap8: `0.014876756863691831` MiB/frame.
- Adaptive measured total delta vs gap4: `-0.04002601161725883` MiB/frame.

## Measured Full-Sequence Rate

| schedule | frames | keyframes | residuals | keyframe MiB/frame | residual MiB/frame | metadata MiB/frame | total MiB/frame | delta vs gap8 | delta vs gap4 | Stage180 PSNR | Stage180 LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 1999 | 292 | 1707 | 0.100233103288 | 0.175633072197 | 0.000000000477 | 0.275866175962 | 0.000000000000 | -0.054902768481 | 29.206326 | 0.176652 |
| stage165_adaptive | 1999 | 358 | 1641 | 0.122888083694 | 0.167854693128 | 0.000000156004 | 0.290742932826 | 0.014876756864 | -0.040026011617 | 29.770753 | 0.142780 |
| uniform_gap4 | 1999 | 536 | 1463 | 0.183987849351 | 0.146781094615 | 0.000000000477 | 0.330768944443 | 0.054902768481 | 0.000000000000 | 29.464217 | 0.162457 |

## Measured Versus Stage181 Proxy

| schedule | measured total | Stage181 proxy | measured - proxy |
|---|---:|---:|---:|
| uniform_gap8 | 0.275866175962 | 0.307317939585 | -0.031451763623 |
| stage165_adaptive | 0.290742932826 | 0.191645608757 | 0.099097324069 |
| uniform_gap4 | 0.330768944443 | 0.368866429914 | -0.038097485471 |

## Component Fractions

| schedule | component | MiB/frame | fraction |
|---|---|---:|---:|
| uniform_gap8 | q12_schedule_packed_keyframes | 0.100233103288 | 0.363340 |
| uniform_gap8 | stage158_residual_payloads | 0.175633072197 | 0.636660 |
| uniform_gap8 | schedule_metadata | 0.000000000477 | 0.000000 |
| stage165_adaptive | q12_schedule_packed_keyframes | 0.122888083694 | 0.422669 |
| stage165_adaptive | stage158_residual_payloads | 0.167854693128 | 0.577330 |
| stage165_adaptive | schedule_metadata | 0.000000156004 | 0.000001 |
| uniform_gap4 | q12_schedule_packed_keyframes | 0.183987849351 | 0.556243 |
| uniform_gap4 | stage158_residual_payloads | 0.146781094615 | 0.443757 |
| uniform_gap4 | schedule_metadata | 0.000000000477 | 0.000000 |

## Validation

| item | expected | actual | status |
|---|---:|---:|---|
| uniform_gap8_missing_residual_measurements | 0 | 0 | ok |
| uniform_gap8_missing_schedule_keyframe_measurements | 0 | 0 | ok |
| stage165_adaptive_missing_residual_measurements | 0 | 0 | ok |
| stage165_adaptive_missing_schedule_keyframe_measurements | 0 | 0 | ok |
| uniform_gap4_missing_residual_measurements | 0 | 0 | ok |
| uniform_gap4_missing_schedule_keyframe_measurements | 0 | 0 | ok |
| frame_schedule_rows_covered | 5997 | 5997 | ok |

## Interpretation

- This is the first measured full-sequence payload aggregation for the frozen adaptive schedule candidate.
- Stage180 quality values remain sampled broader quality evidence; Stage186 should expand quality reporting before final paper claims.
- Keyframe rate uses schedule/sequence-packed q12 bitstreams, avoiding per-keyframe container overcount.

## Outputs

- Total RD CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage185_measured_full_sequence_rd_aggregation/stage185_measured_full_sequence_total_rd.csv`
- Sequence RD CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage185_measured_full_sequence_rd_aggregation/stage185_measured_sequence_rd_breakdown.csv`
- Component CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage185_measured_full_sequence_rd_aggregation/stage185_measured_component_breakdown.csv`
- Validation CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage185_measured_full_sequence_rd_aggregation/stage185_aggregation_validation.csv`
