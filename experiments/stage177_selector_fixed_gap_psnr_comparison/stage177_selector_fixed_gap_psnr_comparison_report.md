# Stage177 Selector Fixed-Gap PSNR Comparison

## Scope

This compares final target quality for fixed uniform gap8, Stage165 adaptive, and fixed uniform gap4 on the Stage174 medium target set.
Rendered middle-recovery rows reuse Stage174 metrics. Target-keyframe rows are evaluated by rendering the target q12 keyframe anchor.

## Overall PSNR

| schedule | targets | keyframes | middle recovery | mean PSNR | p10 PSNR | mean LPIPS | mean payload bytes/target |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 50 | 0 | 50 | 28.756927 | 25.130052 | 0.189479 | 211348.720 |
| stage165_adaptive | 50 | 26 | 24 | 29.217096 | 25.223247 | 0.157234 | 95704.400 |
| uniform_gap4 | 50 | 6 | 44 | 28.945814 | 25.108181 | 0.177610 | 181242.240 |

## Paired Adaptive Delta

- Adaptive minus uniform gap8 PSNR: `0.460169` dB.
- Adaptive minus uniform gap4 PSNR: `0.271282` dB.
- Adaptive keyframe targets: `26` / `50`.

## Category Delta

| category | targets | adaptive keyframes | gap8 PSNR | adaptive PSNR | gap4 PSNR | delta vs gap8 | delta vs gap4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| false_negative_residual | 8 | 0 | 26.203641 | 26.192677 | 26.518971 | -0.010964 | -0.326294 |
| high_payload_residual_control | 4 | 0 | 31.039273 | 31.354837 | 31.078070 | 0.315565 | 0.276767 |
| high_payload_residual_control_extension | 8 | 0 | 29.539710 | 29.539710 | 29.831574 | 0.000000 | -0.291864 |
| normal_residual_control | 4 | 0 | 33.406462 | 33.406462 | 33.417668 | 0.000000 | -0.011207 |
| positive_promoted | 14 | 14 | 28.467816 | 29.716121 | 28.836222 | 1.248305 | 0.879899 |
| positive_promoted_extension | 8 | 8 | 26.910021 | 27.256666 | 26.755422 | 0.346645 | 0.501244 |
| selector_false_positive_keyframe_control | 4 | 4 | 30.071755 | 30.467869 | 30.188225 | 0.396114 | 0.279645 |

## Interpretation

- This table answers the fixed-gap versus selector question in PSNR terms on the Stage174 medium set.
- Adaptive keyframe improvements are q12 keyframe reconstruction quality, not Stage158 middle recovery quality.
- Residual rows where adaptive keeps the same segment as gap8 should match gap8 by construction.
- This remains a sampled medium-set comparison, not final full-sequence RD.

## Outputs

- Per-schedule quality CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage177_selector_fixed_gap_psnr_comparison/stage177_final_quality_by_schedule.csv`
- Per-target delta CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage177_selector_fixed_gap_psnr_comparison/stage177_adaptive_vs_fixed_gap_target_deltas.csv`
- Schedule summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage177_selector_fixed_gap_psnr_comparison/stage177_final_quality_summary.csv`
- Category delta CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage177_selector_fixed_gap_psnr_comparison/stage177_category_delta_summary.csv`
