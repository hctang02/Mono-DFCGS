# Stage168 Positive-Coverage Rendered Smoke

## Scope

This smoke targets Stage165 adaptive rows promoted to keyframes, so adaptive rows are intentionally marked as no-middle-render keyframes rather than rendered middle predictions.
Uniform gap8 recovery is rendered to measure the middle-frame payload/quality that adaptive promotion avoids.

## Summary

| schedule | targets | rendered | keyframes | PSNR | SSIM | MS-SSIM | LPIPS | payload bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 8 | 8 | 0 | 27.889178 | 0.816386 | 0.978293 | 0.219333 | 242546.250000 |
| stage165_adaptive | 8 | 0 | 8 | NA | NA | NA | NA | NA |
| uniform_gap4 | 8 | 7 | 1 | 28.175229 | 0.818067 | 0.977952 | 0.231044 | 232906.142857 |

## Promotion Takeaway

- Adaptive promoted targets: `8` / `8`.
- Uniform gap8 rendered PSNR/LPIPS on the same targets: `27.889178` / `0.219333`.
- Uniform gap8 mean middle residual payload avoided by promotion: `242546.250000` bytes.

## Targets

| rank | sequence | target | hard | high payload | reason | score |
|---:|---|---:|---:|---:|---|---:|
| 1 | motocross-jump | 13 | 1 | 1 | positive_sequence;hard_quality_promoted;high_payload_promoted;stage166_smoke_sequence | 249.336 |
| 2 | motocross-jump | 22 | 1 | 1 | positive_sequence;hard_quality_promoted;high_payload_promoted;stage166_smoke_sequence | 248.318 |
| 3 | motocross-jump | 23 | 1 | 1 | positive_sequence;hard_quality_promoted;high_payload_promoted;stage166_smoke_sequence | 247.169 |
| 4 | camel | 75 | 1 | 1 | positive_sequence;hard_quality_promoted;high_payload_promoted;stage166_smoke_sequence | 245.427 |
| 5 | cows | 60 | 1 | 1 | positive_sequence;hard_quality_promoted;high_payload_promoted;stage166_smoke_sequence | 245.158 |
| 6 | cows | 19 | 1 | 1 | positive_sequence;hard_quality_promoted;high_payload_promoted;stage166_smoke_sequence | 244.503 |
| 7 | camel | 47 | 1 | 1 | positive_sequence;hard_quality_promoted;high_payload_promoted;stage166_smoke_sequence | 244.050 |
| 8 | camel | 51 | 1 | 1 | positive_sequence;hard_quality_promoted;high_payload_promoted;stage166_smoke_sequence | 243.550 |

## Decision

- Decision: `positive_promotions_confirmed_for_broader_validation`.
- Heavy contact sheet: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage168_positive_coverage_rendered_smoke/stage168_positive_coverage_rendered_smoke_contact_sheet.jpg`.
- This complements Stage167; a broader validation should combine positive promotions and remaining false negatives.
