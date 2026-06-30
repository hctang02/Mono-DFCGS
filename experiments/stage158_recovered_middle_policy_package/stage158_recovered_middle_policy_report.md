# Stage158 Recovered Middle-Frame Policy Package

## Policy

- Name: `streamsplat_guided_half_anchor_entropy_residual_v1`
- Base: `original_streamsplat_target_time_half_anchor`
- Correction: `entropy_coded_residual_to_encoder_side_target_dense_anchor`
- Keep fraction: `1.0`
- Side bits: `6`
- Selector metadata: `1 byte/intermediate`

## Evidence

| gap | tasks | PSNR | p10 PSNR | SSIM | MS-SSIM | LPIPS | original PSNR | original LPIPS | payload bytes | direct rate ref | delta PSNR | delta LPIPS | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 60 | 29.780485 | 25.726691 | 0.877938 | 0.985088 | 0.166020 | 22.064218 | 0.301495 | 209392.833 | 0.381631 | 7.716267 | -0.135475 | True |
| 8 | 60 | 29.578682 | 25.326198 | 0.869660 | 0.983847 | 0.178535 | 20.337275 | 0.359337 | 215967.883 | 0.303588 | 9.241407 | -0.180802 | True |

## Decisions

| item | decision | evidence |
|---|---|---|
| quality_target | passes_sampled_middle_target | minimum gap mean PSNR = 29.578682359235195, target = 26.0 |
| structural_metrics | ssim_and_msssim_improve | min SSIM delta = 0.2771290277441343, min MS-SSIM delta = 0.1837212438384692 |
| perceptual_metric | lpips_improves | max LPIPS delta vs original = -0.13547528696556885 |
| rate_accounting | residual_and_selector_bytes_counted | max reference direct total rate = 0.3816307879574477 MiB/frame |
| method_status | current_quality_safe_gs_domain_candidate | Stage157 validates StreamSplat-guided half-anchor Gaussian residual side-info on 120 sampled q12 gap4/gap8 tasks. |

## Decoder Contract

Allowed decoder inputs:

- Original StreamSplat endpoint inputs/base used to produce target-time half anchors.
- Normalized time.
- Encoded entropy residual payload for the selected half-anchor.
- Counted half-selector metadata.

Forbidden decoder inputs:

- Unencoded target dense anchor.
- Target RGB.
- Unencoded target residual tensors.
- Oracle labels not represented in the transmitted payload.

## Notes

- Stage155 image residual is retained only as an upper-bound diagnostic.
- Stage157 is the current GS-domain quality-safe candidate.
- The target dense anchor is used encoder-side to build residual payloads and for offline diagnostics only.
