# Stage186 Full-Sequence Quality Validation

## Decision

- Decision: `adaptive_quality_rate_between_gap8_and_gap4`.
- Adaptive full-sequence PSNR delta vs gap8: `0.05161782022076622` dB.
- Adaptive full-sequence PSNR delta vs gap4: `-0.11013314698813303` dB.
- Adaptive full-sequence LPIPS delta vs gap8: `-0.0027543204711221736`.
- Adaptive measured rate delta vs gap8: `0.014876756863691831` MiB/frame.

## Full-Sequence Quality

| schedule | frames | keyframes | residuals | PSNR | p10 PSNR | keyframe PSNR | residual PSNR | SSIM | MS-SSIM | LPIPS | p90 LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 1999 | 292 | 1707 | 29.373965 | 25.427151 | 29.841138 | 29.294050 | 0.867626 | 0.984343 | 0.168692 | 0.218845 |
| stage165_adaptive | 1999 | 358 | 1641 | 29.425583 | 25.445570 | 29.891894 | 29.323852 | 0.869294 | 0.984647 | 0.165937 | 0.216011 |
| uniform_gap4 | 1999 | 536 | 1463 | 29.535716 | 25.491284 | 29.869428 | 29.413454 | 0.873944 | 0.985529 | 0.159472 | 0.210157 |

## Measured RD-Quality Points

| schedule | MiB/frame | dRate vs gap8 | PSNR | dPSNR vs gap8 | dPSNR vs gap4 | SSIM | MS-SSIM | LPIPS | dLPIPS vs gap8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 0.275866175962 | 0.000000000000 | 29.373965 | 0.000000 | -0.161751 | 0.867626 | 0.984343 | 0.168692 | 0.000000 |
| stage165_adaptive | 0.290742932826 | 0.014876756864 | 29.425583 | 0.051618 | -0.110133 | 0.869294 | 0.984647 | 0.165937 | -0.002754 |
| uniform_gap4 | 0.330768944443 | 0.054902768481 | 29.535716 | 0.161751 | 0.000000 | 0.873944 | 0.985529 | 0.159472 | -0.009220 |

## Validation

| item | expected | actual | status |
|---|---:|---:|---|
| unique_keyframe_quality_rows | 596 | 596 | ok |
| unique_residual_quality_rows | 3472 | 3472 | ok |
| final_frame_schedule_quality_rows | 5997 | 5997 | ok |
| missing_final_measurements | 0 | 0 | ok |

## Interpretation

- This stage uses full unique keyframe and residual render measurements, then maps them back to all frame/schedule rows.
- Stage165 adaptive improves all reported full-sequence quality metrics over uniform gap8, but it remains below uniform gap4 quality.
- Stage185 measured rate also places adaptive between gap8 and gap4: higher than gap8, lower than gap4.
- The next selector work should seek a lower-budget adaptive point that keeps most of the gap8 quality gain while reducing or eliminating the gap8 rate overhead.

## Outputs

- Unique keyframe quality CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage186_full_sequence_quality_validation/stage186_unique_keyframe_quality_metrics.csv`
- Unique residual quality CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage186_full_sequence_quality_validation/stage186_unique_stage158_residual_quality_metrics.csv`
- Full frame/schedule quality CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage186_full_sequence_quality_validation/stage186_full_sequence_quality_by_schedule.csv`
- Summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage186_full_sequence_quality_validation/stage186_full_sequence_quality_summary.csv`
- Measured RD-quality CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage186_full_sequence_quality_validation/stage186_measured_rd_quality_points.csv`
