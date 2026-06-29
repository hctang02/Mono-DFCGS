# Stage139 Full-Pipeline RD Package

## Scope

- Packages full-pipeline rate accounting for the Stage138 deployable no-teacher predictor policy.
- Main stream rate is q12 linear-anchor rate from Stage78/Stage137.
- Residual payload, selected-index payload, and policy-scale payload are zero per frame.
- Teacher residual side-info is not part of the deployable pipeline.

## Policy

- policy: `deployable_render_aware_scaled_adapter_delta_selected_residual_codec_v1`
- primary: `q4_top20` scale `0.75`
- low-rate: `q4_top10` scale `0.75`

## Aggregate RD

| role | setting | keep | scale | tasks | rate | final PSNR | delta base | delta Stage132 scale1 | residual bytes | index bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | q4_top20 | 0.2 | 0.75 | 60 | 0.117298 | 19.022110 | 0.071307 | 0.011850 | 0 | 0 |
| low_rate | q4_top10 | 0.1 | 0.75 | 60 | 0.117298 | 18.997891 | 0.047088 | 0.003077 | 0 | 0 |

## Gap Breakdown

| role | setting | gap | tasks | q12 anchor rate | final PSNR | delta base | delta Stage132 scale1 |
|---|---|---:|---:|---:|---:|---:|---:|
| primary | q4_top20 | 4 | 23 | 0.181938 | 19.732134 | 0.082986 | 0.026550 |
| primary | q4_top20 | 8 | 19 | 0.097625 | 19.494181 | 0.083656 | 0.002416 |
| primary | q4_top20 | 16 | 18 | 0.055469 | 17.616558 | 0.043347 | 0.003024 |
| low_rate | q4_top10 | 4 | 23 | 0.181938 | 19.707571 | 0.058423 | 0.010187 |
| low_rate | q4_top10 | 8 | 19 | 0.097625 | 19.458984 | 0.048459 | 0.000075 |
| low_rate | q4_top10 | 16 | 18 | 0.055469 | 17.604367 | 0.031156 | -0.002838 |

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage139_full_pipeline_rd_package/stage139_full_pipeline_rd_rows.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage139_full_pipeline_rd_package/stage139_full_pipeline_rd_summary.json`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage139_full_pipeline_rd_package/stage139_full_pipeline_rd_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage139_full_pipeline_rd_package/stage139_full_pipeline_rd_report.md`
