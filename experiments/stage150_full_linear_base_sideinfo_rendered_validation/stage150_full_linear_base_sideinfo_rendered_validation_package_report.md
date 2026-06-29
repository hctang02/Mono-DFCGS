# Stage150 Full Linear-Base Side-Info Rendered Validation Package

## Summary

| gap | target | base PSNR | entropy PSNR | entropy-target | delta base | payload bytes | side MiB | direct rate | amortized rate | decode diff | positives | decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 23.004337 | 19.983964 | 23.104893 | 0.100556 | 3.120929 | 30062.867 | 0.028670 | 0.210608 | 0.203441 | 0.000000 | 1463/1463 | passes_full_eval_validation |
| 8 | 21.560049 | 18.766429 | 22.020189 | 0.460140 | 3.253760 | 30203.919 | 0.028805 | 0.126430 | 0.122830 | 0.000000 | 1707/1707 | passes_full_eval_validation |

## Decisions

| item | decision | evidence |
|---|---|---|
| entropy_decode | matches_fixed_decode | max decoded abs diff vs fixed = 0.0 |
| target_alignment | passes_full_eval_validation | worst entropy gap to corrected target = 0.10055620282386002 dB with tolerance 0.25 dB |
| task_positivity | positive_on_all_full_eval_tasks | min positive delta fraction = 1.0 |
| rate_accounting | all_sideinfo_bytes_counted | max direct total rate = 0.21060840528077832 MiB/frame; payload bytes from experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_rows.csv |
| next_step | run_full_video_rd | Full q12 gap4/gap8 eval rows pass; next is full-video RD packaging. |

## Contract

- This package is based on actual entropy payload encode/decode/render results.
- Side-info payload bytes are counted in direct and amortized total rate.
- Decoder inputs remain endpoint anchors, normalized time, and encoded payload only.
- Decoder does not receive target dense anchors, target RGB, or unencoded residual tensors.

## Outputs

- rows CSV: `experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_package_rows.csv`
- decisions CSV: `experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_package_decisions.csv`
- summary JSON: `experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_package_summary.json`
- package JSON: `experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_package.json`
- report Markdown: `experiments/stage150_full_linear_base_sideinfo_rendered_validation/stage150_full_linear_base_sideinfo_rendered_validation_package_report.md`
