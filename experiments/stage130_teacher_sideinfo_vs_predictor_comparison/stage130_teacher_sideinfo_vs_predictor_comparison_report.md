# Stage130 Teacher Side-Info Vs Predictor Comparison

## Summary

| stage | method | setting | deployable | teacher | rate | PSNR | delta base | delta full | residual bytes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 122 | teacher_compressed_sideinfo | q4_top10 | 0 | 1 | 0.121246 | 19.738488 | 1.254468 | nan | 15190.475 |
| 125 | adapter_delta_selected_predictor | q4_top10 | 1 | 0 | 0.117298 | 18.994813 | 0.044010 | -0.19282981274515912 | 0 |
| 129 | dedicated_mlp_selected_predictor | q4_top10 | 1 | 0 | 0.117298 | 18.865778 | -0.085025 | -0.321865539568306 | 0 |
| 122 | teacher_compressed_sideinfo | q4_top20 | 0 | 1 | 0.133768 | 20.689271 | 2.205251 | nan | 28320.791666666668 |
| 125 | adapter_delta_selected_predictor | q4_top20 | 1 | 0 | 0.117298 | 19.010259 | 0.059456 | -0.17738394265066096 | 0 |
| 129 | dedicated_mlp_selected_predictor | q4_top20 | 1 | 0 | 0.117298 | 18.765203 | -0.185600 | -0.4224402424824056 | 0 |

## Interpretation

- Teacher side-info has the highest quality but requires encoder-side target residual values.
- Adapter-delta selected predictor is the current best no-teacher predictor point.
- Dedicated MLP predictor is deployable in input contract but not render-safe yet.

## Outputs

- comparison CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage130_teacher_sideinfo_vs_predictor_comparison/stage130_teacher_sideinfo_vs_predictor_comparison_rows.csv`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage130_teacher_sideinfo_vs_predictor_comparison/stage130_teacher_sideinfo_vs_predictor_comparison_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage130_teacher_sideinfo_vs_predictor_comparison/stage130_teacher_sideinfo_vs_predictor_comparison_report.md`
