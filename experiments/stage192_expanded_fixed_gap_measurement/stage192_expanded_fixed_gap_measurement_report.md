# Stage192 Expanded Fixed-Gap Measurement

## Decision

- Decision: `current_adaptive_not_strong_against_expanded_fixed_gaps`.
- Complete: `True`.
- Best fixed gap by PSNR: `uniform_gap2`.

## Expanded RD-Quality

| schedule | MiB/frame | keyframes | PSNR | SSIM | MS-SSIM | LPIPS | dPSNR vs best fixed | dLPIPS vs best fixed | pass +1dB/no-regression |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap2 | 0.449546882187 | 1025 | 29.654815 | 0.878376 | 0.986617 | 0.151681 | 0.000000 | 0.000000 | 0 |
| uniform_gap4 | 0.330768944443 | 536 | 29.535716 | 0.873944 | 0.985529 | 0.159472 | -0.119099 | 0.007790 | 0 |
| uniform_gap6 | 0.293445062375 | 372 | 29.448738 | 0.870619 | 0.984894 | 0.164575 | -0.206078 | 0.012894 | 0 |
| uniform_gap8 | 0.275866175962 | 292 | 29.373965 | 0.867626 | 0.984343 | 0.168692 | -0.280850 | 0.017010 | 0 |
| uniform_gap16 | 0.251478830178 | 169 | 29.199329 | 0.860357 | 0.983030 | 0.177326 | -0.455487 | 0.025645 | 0 |
| stage165_adaptive | 0.290742932826 | 358 | 29.425583 | 0.869294 | 0.984647 | 0.165937 | -0.229233 | 0.014256 | 0 |

## Validation

| item | expected | actual | status |
|---|---:|---:|---|
| uniform_gap2_missing_residual_payload | 0 | 0 | ok |
| uniform_gap2_missing_schedule_keyframe_payload | 0 | 0 | ok |
| uniform_gap4_missing_residual_payload | 0 | 0 | ok |
| uniform_gap4_missing_schedule_keyframe_payload | 0 | 0 | ok |
| uniform_gap6_missing_residual_payload | 0 | 0 | ok |
| uniform_gap6_missing_schedule_keyframe_payload | 0 | 0 | ok |
| uniform_gap8_missing_residual_payload | 0 | 0 | ok |
| uniform_gap8_missing_schedule_keyframe_payload | 0 | 0 | ok |
| uniform_gap16_missing_residual_payload | 0 | 0 | ok |
| uniform_gap16_missing_schedule_keyframe_payload | 0 | 0 | ok |
| stage165_adaptive_missing_residual_payload | 0 | 0 | ok |
| stage165_adaptive_missing_schedule_keyframe_payload | 0 | 0 | ok |
| uniform_gap2_final_quality_rows | 1999 | 1999 | ok |
| uniform_gap4_final_quality_rows | 1999 | 1999 | ok |
| uniform_gap6_final_quality_rows | 1999 | 1999 | ok |
| uniform_gap8_final_quality_rows | 1999 | 1999 | ok |
| uniform_gap16_final_quality_rows | 1999 | 1999 | ok |
| stage165_adaptive_final_quality_rows | 1999 | 1999 | ok |
| unique_keyframe_payload_rows | 1065 | 1065 | ok |
| unique_residual_payload_rows | 7791 | 7791 | ok |
| unique_keyframe_quality_rows | 1065 | 1065 | ok |
| unique_residual_quality_rows | 7791 | 7791 | ok |
| missing_final_measurements | 0 | 0 | ok |

## Interpretation

- This stage is the expanded fixed-gap baseline check requested before stronger selector claims.
- If current adaptive does not beat the best fixed-gap baseline, Stage193 must compute oracle headroom before additional selector tuning.
- The target for a strong selector claim is about +1 dB PSNR over the best tested fixed gap with no SSIM/MS-SSIM/LPIPS regression.

## Outputs

- RD-quality CSV: `experiments/stage192_expanded_fixed_gap_measurement/stage192_expanded_fixed_gap_rd_quality_points.csv`
- Total RD CSV: `experiments/stage192_expanded_fixed_gap_measurement/stage192_expanded_fixed_gap_total_rd.csv`
- Quality summary CSV: `experiments/stage192_expanded_fixed_gap_measurement/stage192_full_sequence_quality_summary.csv`
- Final quality CSV: `experiments/stage192_expanded_fixed_gap_measurement/stage192_full_sequence_quality_by_schedule.csv`
