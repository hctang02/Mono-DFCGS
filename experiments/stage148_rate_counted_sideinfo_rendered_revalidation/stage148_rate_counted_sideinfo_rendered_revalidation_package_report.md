# Stage148 Rate-Counted Side-Info Rendered Revalidation Package

## Summary

| gap | target | base PSNR | entropy PSNR | entropy-target | delta base | payload bytes | side MiB | direct rate | amortized rate | decode diff | positives | decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 23.004337 | 20.264133 | 22.850144 | -0.154194 | 2.586010 | 34785.517 | 0.033174 | 0.215112 | 0.206819 | 0.000000 | 60/60 | passes_sample_revalidation |
| 8 | 21.560049 | 19.062353 | 21.965724 | 0.405675 | 2.903371 | 35153.033 | 0.033525 | 0.131150 | 0.126959 | 0.000000 | 60/60 | passes_sample_revalidation |

## Decisions

| item | decision | evidence |
|---|---|---|
| entropy_decode | matches_fixed_decode | max decoded abs diff vs fixed = 0.0 |
| target_alignment | passes_sample_revalidation | worst entropy gap to corrected target = -0.15419354559560006 dB with tolerance 0.25 dB |
| task_positivity | positive_on_all_sampled_tasks | min positive delta fraction = 1.0 |
| rate_accounting | all_sideinfo_bytes_counted | max direct total rate = 0.21511227459583468 MiB/frame; payload bytes from experiments/stage148_rate_counted_sideinfo_rendered_revalidation/stage148_rate_counted_sideinfo_rendered_revalidation_rows.csv |
| next_step | run_full_all_row_or_full_video_rd | Stage148 is sampled rendered revalidation, not full all-row eval. |

## Contract

- This package is based on actual entropy payload encode/decode/render results.
- Side-info payload bytes are counted in direct and amortized total rate.
- Decoder inputs remain endpoint anchors, normalized time, and encoded payload only.
- Decoder does not receive target dense anchors, target RGB, or unencoded residual tensors.

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage148_rate_counted_sideinfo_rendered_revalidation/stage148_rate_counted_sideinfo_rendered_revalidation_package_rows.csv`
- decisions CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage148_rate_counted_sideinfo_rendered_revalidation/stage148_rate_counted_sideinfo_rendered_revalidation_package_decisions.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage148_rate_counted_sideinfo_rendered_revalidation/stage148_rate_counted_sideinfo_rendered_revalidation_package_summary.json`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage148_rate_counted_sideinfo_rendered_revalidation/stage148_rate_counted_sideinfo_rendered_revalidation_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage148_rate_counted_sideinfo_rendered_revalidation/stage148_rate_counted_sideinfo_rendered_revalidation_package_report.md`
