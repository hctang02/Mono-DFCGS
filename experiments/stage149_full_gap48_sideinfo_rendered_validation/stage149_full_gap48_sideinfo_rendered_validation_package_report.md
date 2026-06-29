# Stage149 Full Gap4/Gap8 Side-Info Rendered Validation Package

## Summary

| gap | target | base PSNR | entropy PSNR | entropy-target | delta base | payload bytes | side MiB | direct rate | amortized rate | decode diff | positives | decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 23.004337 | 20.210771 | 22.768595 | -0.235742 | 2.557825 | 34943.741 | 0.033325 | 0.215263 | 0.206932 | 0.000000 | 1463/1463 | passes_full_eval_validation |
| 8 | 21.560049 | 19.067098 | 21.857518 | 0.297469 | 2.790420 | 35172.316 | 0.033543 | 0.131168 | 0.126975 | 0.000000 | 1707/1707 | passes_full_eval_validation |

## Decisions

| item | decision | evidence |
|---|---|---|
| entropy_decode | matches_fixed_decode | max decoded abs diff vs fixed = 0.0 |
| target_alignment | passes_full_eval_validation | worst entropy gap to corrected target = -0.23574200497678177 dB with tolerance 0.25 dB |
| task_positivity | positive_on_all_full_eval_tasks | min positive delta fraction = 1.0 |
| rate_accounting | all_sideinfo_bytes_counted | max direct total rate = 0.21526316902466064 MiB/frame; payload bytes from experiments/stage149_full_gap48_sideinfo_rendered_validation/stage149_full_gap48_sideinfo_rendered_validation_rows.csv |
| next_step | run_full_video_rd | Full q12 gap4/gap8 eval rows pass; next is full-video RD packaging. |

## Contract

- This package is based on actual entropy payload encode/decode/render results.
- Side-info payload bytes are counted in direct and amortized total rate.
- Decoder inputs remain endpoint anchors, normalized time, and encoded payload only.
- Decoder does not receive target dense anchors, target RGB, or unencoded residual tensors.

## Outputs

- rows CSV: `experiments/stage149_full_gap48_sideinfo_rendered_validation/stage149_full_gap48_sideinfo_rendered_validation_package_rows.csv`
- decisions CSV: `experiments/stage149_full_gap48_sideinfo_rendered_validation/stage149_full_gap48_sideinfo_rendered_validation_package_decisions.csv`
- summary JSON: `experiments/stage149_full_gap48_sideinfo_rendered_validation/stage149_full_gap48_sideinfo_rendered_validation_package_summary.json`
- package JSON: `experiments/stage149_full_gap48_sideinfo_rendered_validation/stage149_full_gap48_sideinfo_rendered_validation_package.json`
- report Markdown: `experiments/stage149_full_gap48_sideinfo_rendered_validation/stage149_full_gap48_sideinfo_rendered_validation_package_report.md`
