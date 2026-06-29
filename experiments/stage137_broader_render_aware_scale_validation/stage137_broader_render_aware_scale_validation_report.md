# Stage137 Broader Render-Aware Scale Validation

## Scope

- Broadens Stage136 from 12 tasks to a 60-task rendered validation slice.
- Sweeps the Stage135 scale candidates for q4/top20 and q4/top10.
- Uses only decoder-side endpoint anchors, normalized time, and the pre-shared Stage65 adapter.
- Transmits no residual payload bytes and no selected-index bytes.
- Does not load target dense anchors or teacher residual side-info.
- Target RGB is used only for offline rendered PSNR metrics.

## Stage136 Smoke-Selected Candidate

- setting: `q4_top20`
- adapter delta scale: `0.5`
- mean selected PSNR: `19.014217`
- mean delta vs base: `0.063414`
- positives: `49/60`

## Broader Best Candidate

- setting: `q4_top20`
- adapter delta scale: `0.75`
- mean selected PSNR: `19.022110`
- mean delta vs base: `0.071307`
- positives: `49/60`

## Summary

| setting | role | keep | scale | tasks | rate | base PSNR | full PSNR | selected PSNR | delta base | delta full | positives | near full |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q4_top10 | low_rate | 0.1 | 0.0 | 60 | 0.117298 | 18.950803 | 19.187643 | 18.950803 | 0.000000 | -0.236840 | 0/60 | 13/60 |
| q4_top10 | low_rate | 0.1 | 0.25 | 60 | 0.117298 | 18.950803 | 19.187643 | 18.975705 | 0.024902 | -0.211938 | 48/60 | 12/60 |
| q4_top10 | low_rate | 0.1 | 0.5 | 60 | 0.117298 | 18.950803 | 19.187643 | 18.991120 | 0.040317 | -0.196523 | 51/60 | 12/60 |
| q4_top10 | low_rate | 0.1 | 0.75 | 60 | 0.117298 | 18.950803 | 19.187643 | 18.997891 | 0.047088 | -0.189753 | 49/60 | 12/60 |
| q4_top10 | low_rate | 0.1 | 1.0 | 60 | 0.117298 | 18.950803 | 19.187643 | 18.994813 | 0.044010 | -0.192830 | 51/60 | 13/60 |
| q4_top10 | low_rate | 0.1 | 1.25 | 60 | 0.117298 | 18.950803 | 19.187643 | 18.965717 | 0.014914 | -0.221926 | 45/60 | 14/60 |
| q4_top20 | primary | 0.2 | 0.0 | 60 | 0.117298 | 18.950803 | 19.187643 | 18.950803 | 0.000000 | -0.236840 | 0/60 | 13/60 |
| q4_top20 | primary | 0.2 | 0.25 | 60 | 0.117298 | 18.950803 | 19.187643 | 18.991578 | 0.040775 | -0.196065 | 48/60 | 12/60 |
| q4_top20 | primary | 0.2 | 0.5 | 60 | 0.117298 | 18.950803 | 19.187643 | 19.014217 | 0.063414 | -0.173427 | 49/60 | 13/60 |
| q4_top20 | primary | 0.2 | 0.75 | 60 | 0.117298 | 18.950803 | 19.187643 | 19.022110 | 0.071307 | -0.165534 | 49/60 | 12/60 |
| q4_top20 | primary | 0.2 | 1.0 | 60 | 0.117298 | 18.950803 | 19.187643 | 19.010259 | 0.059456 | -0.177384 | 48/60 | 13/60 |
| q4_top20 | primary | 0.2 | 1.25 | 60 | 0.117298 | 18.950803 | 19.187643 | 18.937363 | -0.013440 | -0.250280 | 44/60 | 13/60 |

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage137_broader_render_aware_scale_validation/stage137_broader_render_aware_scale_validation_rows.csv`
- summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage137_broader_render_aware_scale_validation/stage137_broader_render_aware_scale_validation_summary.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage137_broader_render_aware_scale_validation/stage137_broader_render_aware_scale_validation_summary.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage137_broader_render_aware_scale_validation/stage137_broader_render_aware_scale_validation_report.md`
