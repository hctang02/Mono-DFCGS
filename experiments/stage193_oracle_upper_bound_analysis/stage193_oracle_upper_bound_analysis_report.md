# Stage193 Oracle Upper-Bound Analysis

## Decision

- Decision: `framewise_oracle_upper_bound_below_target_margin`.
- Best fixed gap by PSNR: `uniform_gap2` with PSNR `29.654815328772308`.

## Oracle Summary

| oracle | schedule consistent | PSNR | SSIM | MS-SSIM | LPIPS | dPSNR vs best fixed | dLPIPS vs best fixed | +1dB pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| framewise_psnr_oracle | 0 | 29.749038 | 0.881661 | 0.987087 | 0.146418 | 0.094223 | -0.005264 | 0 |
| schedule_path_psnr_oracle | 1 | 29.670134 | 0.878858 | 0.986708 | 0.150848 | 0.015319 | -0.000833 | 0 |

## Interpretation

- `framewise_psnr_oracle` is an optimistic non-schedule-consistent upper bound that picks the best measured schedule output independently per frame.
- `schedule_path_psnr_oracle` is schedule-consistent, but limited to measured keyframe nodes and measured Stage158 residual edges from Stage192.
- If the framewise oracle is below the +1 dB target, selector tuning over the current measured representation cannot plausibly satisfy the requested strong claim.

## Outputs

- Summary CSV: `experiments/stage193_oracle_upper_bound_analysis/stage193_oracle_summary.csv`
- Framewise oracle rows: `experiments/stage193_oracle_upper_bound_analysis/stage193_framewise_psnr_oracle_rows.csv`
- Schedule path oracle rows: `experiments/stage193_oracle_upper_bound_analysis/stage193_schedule_path_psnr_oracle_rows.csv`
- Sequence summary: `experiments/stage193_oracle_upper_bound_analysis/stage193_oracle_sequence_summary.csv`
