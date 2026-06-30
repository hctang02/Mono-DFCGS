# Stage167 Adaptive Schedule Rendered Smoke

## Scope

This is a small rendered smoke on sampled targets that remain middle-recovery targets under the Stage165 adaptive schedule.
The target set is intentionally biased toward Stage166 hard false negatives, so the mean metrics are stress-case indicators rather than representative sequence averages.
It is not a full sequence-level RD validation.

## Summary

| schedule | targets | rendered | keyframes | PSNR | SSIM | MS-SSIM | LPIPS | payload bytes | delta PSNR vs gap8 | delta LPIPS vs gap8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 8 | 8 | 0 | 26.203641 | 0.834367 | 0.981569 | 0.207921 | 156389.625000 | 0.000000 | 0.000000 |
| stage165_adaptive | 8 | 8 | 0 | 26.192677 | 0.833774 | 0.981409 | 0.208831 | 155993.625000 | -0.010964 | 0.000910 |
| uniform_gap4 | 8 | 7 | 1 | 25.683246 | 0.843334 | 0.983516 | 0.193556 | 149221.571429 | 0.103399 | -0.004473 |

## Targets

| rank | sequence | target | reason | score |
|---:|---|---:|---|---:|
| 1 | dance-twirl | 66 | adaptive_hard_false_negative;hard_quality_label;stage166_smoke_sequence | 178.716 |
| 2 | breakdance | 61 | adaptive_hard_false_negative;hard_quality_label;stage166_smoke_sequence | 175.586 |
| 3 | breakdance | 6 | adaptive_hard_false_negative;hard_quality_label;stage166_smoke_sequence | 174.245 |
| 4 | breakdance | 55 | adaptive_hard_false_negative;hard_quality_label;stage166_smoke_sequence | 172.610 |
| 5 | breakdance | 25 | adaptive_hard_false_negative;hard_quality_label;stage166_smoke_sequence | 172.136 |
| 6 | bike-packing | 11 | adaptive_hard_false_negative;hard_quality_label;stage166_smoke_sequence | 172.077 |
| 7 | bike-packing | 26 | adaptive_hard_false_negative;hard_quality_label;stage166_smoke_sequence | 169.918 |
| 8 | dogs-jump | 20 | adaptive_hard_false_negative;hard_quality_label | 156.786 |

## Decision

- Decision: `inspect_smoke_before_scaling`.
- Heavy contact sheet: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage167_adaptive_schedule_rendered_smoke/stage167_adaptive_schedule_rendered_smoke_contact_sheet.jpg`.
- Keep this as smoke evidence only; broader adaptive rendered validation is still required before final claims.
