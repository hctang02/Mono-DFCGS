# Stage180 Broader Sampled Adaptive Validation Execution

## Scope

This executes the Stage179 90-target broader sampled protocol by reusing Stage174/177 rows and rendering only missing middle/keyframe metrics.

## Execution Summary

- Protocol rows covered: `270 / 270`.
- New middle renders: `88` / expected `88`.
- New q12 keyframe metrics: `32` / expected `32`.
- Reused Stage174 rows: `150`.
- Keyframe marker rows: `64`.

## Source Coverage

| source | rows | rendered | keyframe markers |
|---|---:|---:|---:|
| stage174:stage168 | 6 | 4 | 2 |
| stage174:stage170 | 78 | 60 | 18 |
| stage174:stage174_keyframe_marker | 12 | 0 | 12 |
| stage174:stage174_rendered | 54 | 54 | 0 |
| stage180_keyframe_marker | 32 | 0 | 32 |
| stage180_rendered | 88 | 88 | 0 |

## Overall Final Quality

| schedule | targets | keyframes | middle recovery | mean PSNR | p10 PSNR | mean LPIPS | mean payload bytes/target |
|---|---:|---:|---:|---:|---:|---:|---:|
| stage165_adaptive | 90 | 56 | 34 | 29.770753 | 25.424549 | 0.142780 | 74673.433 |
| uniform_gap4 | 90 | 8 | 82 | 29.464217 | 25.431154 | 0.162457 | 196008.433 |
| uniform_gap8 | 90 | 0 | 90 | 29.206326 | 25.418355 | 0.176652 | 219878.578 |

## Paired Adaptive Delta

- Adaptive minus uniform gap8 PSNR: `0.5644261202320328` dB.
- Adaptive minus uniform gap4 PSNR: `0.306535729521994` dB.
- Adaptive minus uniform gap8 LPIPS: `-0.0338725696835253`.
- Adaptive minus uniform gap4 LPIPS: `-0.019677375422583687`.

## Category Delta

| category | targets | adaptive keyframes | gap8 PSNR | adaptive PSNR | gap4 PSNR | delta vs gap8 | delta vs gap4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| broader_false_negative_residual | 1 | 0 | 32.080420 | 32.080420 | 32.332240 | 0.000000 | -0.251820 |
| broader_normal_residual_control | 5 | 0 | 32.214940 | 32.214940 | 32.236173 | 0.000000 | -0.021233 |
| broader_positive_promoted | 18 | 18 | 28.563591 | 29.393398 | 28.949399 | 0.829807 | 0.443999 |
| broader_sequence_coverage_probe | 9 | 7 | 29.784158 | 30.412147 | 30.140690 | 0.627988 | 0.271456 |
| broader_weak_sequence_probe | 7 | 5 | 30.766547 | 31.795333 | 31.231473 | 1.028786 | 0.563861 |
| false_negative_residual | 8 | 0 | 26.203641 | 26.192677 | 26.518971 | -0.010964 | -0.326294 |
| high_payload_residual_control | 4 | 0 | 31.039273 | 31.354837 | 31.078070 | 0.315565 | 0.276767 |
| high_payload_residual_control_extension | 8 | 0 | 29.539710 | 29.539710 | 29.831574 | 0.000000 | -0.291864 |
| normal_residual_control | 4 | 0 | 33.406462 | 33.406462 | 33.417668 | 0.000000 | -0.011207 |
| positive_promoted | 14 | 14 | 28.467816 | 29.716121 | 28.836222 | 1.248305 | 0.879899 |
| positive_promoted_extension | 8 | 8 | 26.910021 | 27.256666 | 26.755422 | 0.346645 | 0.501244 |
| selector_false_positive_keyframe_control | 4 | 4 | 30.071755 | 30.467869 | 30.188225 | 0.396114 | 0.279645 |

## Interpretation

- This is still sampled broader validation, not final full-sequence RD.
- Adaptive keyframe rows use q12 keyframe reconstruction quality, not Stage158 middle-recovery quality.
- Stage158 middle recovery remains fixed as `streamsplat_guided_half_anchor_entropy_residual_v1`.

## Outputs

- Render rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_broader_validation_rows.csv`
- Final quality CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_final_quality_by_schedule.csv`
- Target delta CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_adaptive_vs_fixed_gap_target_deltas.csv`
- Final summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_final_quality_summary.csv`
- Category delta CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage180_broader_sampled_adaptive_validation_execution/stage180_category_delta_summary.csv`
