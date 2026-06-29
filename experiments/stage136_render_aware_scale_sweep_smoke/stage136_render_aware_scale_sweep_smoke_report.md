# Stage136 Render-Aware Scale Sweep Smoke

## Scope

- Runs the Stage135 render-aware adapter-delta scale protocol on a smoke slice.
- Uses only decoder-side endpoint anchors, normalized time, and the pre-shared Stage65 adapter.
- Transmits no residual payload bytes and no selected-index bytes.
- Does not load target dense anchors or teacher residual side-info.
- Target RGB is used only for offline rendered PSNR metrics.

## Best Smoke Candidate

- setting: `q4_top20`
- adapter delta scale: `0.5`
- mean selected PSNR: `20.135994`
- mean delta vs base: `0.056355`

## Summary

| setting | role | keep | scale | tasks | rate | base PSNR | full PSNR | selected PSNR | delta base | delta full | positives | near full |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q4_top10 | low_rate | 0.1 | 0.0 | 12 | 0.129243 | 20.079640 | 20.182425 | 20.079640 | 0.000000 | -0.102785 | 0/12 | 3/12 |
| q4_top10 | low_rate | 0.1 | 0.25 | 12 | 0.129243 | 20.079640 | 20.182425 | 20.106775 | 0.027135 | -0.075650 | 10/12 | 3/12 |
| q4_top10 | low_rate | 0.1 | 0.5 | 12 | 0.129243 | 20.079640 | 20.182425 | 20.118435 | 0.038795 | -0.063990 | 11/12 | 3/12 |
| q4_top10 | low_rate | 0.1 | 0.75 | 12 | 0.129243 | 20.079640 | 20.182425 | 20.119030 | 0.039390 | -0.063395 | 9/12 | 3/12 |
| q4_top10 | low_rate | 0.1 | 1.0 | 12 | 0.129243 | 20.079640 | 20.182425 | 20.107503 | 0.027863 | -0.074922 | 9/12 | 3/12 |
| q4_top10 | low_rate | 0.1 | 1.25 | 12 | 0.129243 | 20.079640 | 20.182425 | 20.060252 | -0.019388 | -0.122173 | 8/12 | 3/12 |
| q4_top20 | primary | 0.2 | 0.0 | 12 | 0.129243 | 20.079640 | 20.182425 | 20.079640 | 0.000000 | -0.102785 | 0/12 | 3/12 |
| q4_top20 | primary | 0.2 | 0.25 | 12 | 0.129243 | 20.079640 | 20.182425 | 20.122724 | 0.043084 | -0.059701 | 10/12 | 3/12 |
| q4_top20 | primary | 0.2 | 0.5 | 12 | 0.129243 | 20.079640 | 20.182425 | 20.135994 | 0.056355 | -0.046430 | 8/12 | 3/12 |
| q4_top20 | primary | 0.2 | 0.75 | 12 | 0.129243 | 20.079640 | 20.182425 | 20.129236 | 0.049596 | -0.053189 | 9/12 | 3/12 |
| q4_top20 | primary | 0.2 | 1.0 | 12 | 0.129243 | 20.079640 | 20.182425 | 20.099010 | 0.019371 | -0.083415 | 9/12 | 3/12 |
| q4_top20 | primary | 0.2 | 1.25 | 12 | 0.129243 | 20.079640 | 20.182425 | 20.003406 | -0.076234 | -0.179019 | 7/12 | 3/12 |

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage136_render_aware_scale_sweep_smoke/stage136_render_aware_scale_sweep_smoke_rows.csv`
- summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage136_render_aware_scale_sweep_smoke/stage136_render_aware_scale_sweep_smoke_summary.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage136_render_aware_scale_sweep_smoke/stage136_render_aware_scale_sweep_smoke_summary.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage136_render_aware_scale_sweep_smoke/stage136_render_aware_scale_sweep_smoke_report.md`
