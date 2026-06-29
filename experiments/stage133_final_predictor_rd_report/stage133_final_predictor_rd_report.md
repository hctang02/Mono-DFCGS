# Stage133 Final Predictor RD Report

## Final Recommendation

- Deployable policy: `deployable_adapter_delta_selected_residual_codec_v1`.
- Primary deployable setting: `q4_top20`.
- Optional low-rate setting: `q4_top10`.
- Teacher residual side-info is retained only as a quality reference.
- Dedicated MLP predictor is rejected until render-aware training fixes PSNR regression.

## RD Table

| role | method | setting | deployable | teacher | rate | PSNR | delta base | residual bytes | index bytes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| final_deployable_low_rate | adapter_delta_selected_predictor | q4_top10 | 1 | 0 | 0.117298 | 18.994813 | 0.044010 | 0 | 0 |
| final_deployable_primary | adapter_delta_selected_predictor | q4_top20 | 1 | 0 | 0.117298 | 19.010259 | 0.059456 | 0 | 0 |
| rejected_render_regression | dedicated_mlp_selected_predictor | q4_top10 | 1 | 0 | 0.117298 | 18.865778 | -0.085025 | 0 | 0 |
| rejected_render_regression | dedicated_mlp_selected_predictor | q4_top20 | 1 | 0 | 0.117298 | 18.765203 | -0.185600 | 0 | 0 |
| teacher_reference_only | teacher_compressed_sideinfo | q4_top10 | 0 | 1 | 0.121246 | 19.738488 | 1.254468 | 15190.475 | 0 |
| teacher_reference_only | teacher_compressed_sideinfo | q4_top20 | 0 | 1 | 0.133768 | 20.689271 | 2.205251 | 28320.791666666668 | 0 |

## Plot

- PNG: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage133_final_predictor_rd_report/stage133_final_predictor_rd_plot.png`
- PDF: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage133_final_predictor_rd_report/stage133_final_predictor_rd_plot.pdf`

## Outputs

- final rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage133_final_predictor_rd_report/stage133_final_predictor_rd_rows.csv`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage133_final_predictor_rd_report/stage133_final_predictor_rd_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage133_final_predictor_rd_report/stage133_final_predictor_rd_report.md`
