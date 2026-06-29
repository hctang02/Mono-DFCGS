# Stage131 Predictor Ablation Package

## Key Takeaways

- Adapter-delta selected predictor has small but stable positive rendered gain over linear base.
- Dedicated MLP reduces residual MSE but regresses rendered PSNR, so MSE labels are not enough.
- q4/top20 is the best no-teacher adapter-delta point; q4/top10 is lower keep but slightly lower quality.
- Teacher residual side-info remains a non-deployable upper-quality reference.

## Rows

| group | stage | method | setting | tasks | rate | PSNR | delta base | eval MSE reduction | deployable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| validation_scale_adapter_delta | 124 | adapter_delta_selected_predictor | q4_top10 | 12 | 0.129243 | 20.107503 | 0.027863 | nan | 1 |
| validation_scale_adapter_delta | 125 | adapter_delta_selected_predictor | q4_top10 | 60 | 0.117298 | 18.994813 | 0.044010 | nan | 1 |
| mse_vs_render_dedicated_mlp | 127 | dedicated_mlp_selected_predictor_mse_eval | q4_top10 | 60 | nan | nan | nan | 0.102933 | 1 |
| mse_vs_render_dedicated_mlp | 129 | dedicated_mlp_selected_predictor_rendered | q4_top10 | 60 | 0.117298 | 18.865778 | -0.085025 | nan | 1 |
| validation_scale_adapter_delta | 124 | adapter_delta_selected_predictor | q4_top20 | 12 | 0.129243 | 20.099010 | 0.019371 | nan | 1 |
| validation_scale_adapter_delta | 125 | adapter_delta_selected_predictor | q4_top20 | 60 | 0.117298 | 19.010259 | 0.059456 | nan | 1 |
| mse_vs_render_dedicated_mlp | 127 | dedicated_mlp_selected_predictor_mse_eval | q4_top20 | 60 | nan | nan | nan | 0.088083 | 1 |
| mse_vs_render_dedicated_mlp | 129 | dedicated_mlp_selected_predictor_rendered | q4_top20 | 60 | 0.117298 | 18.765203 | -0.185600 | nan | 1 |
| teacher_vs_predictor_keep_fraction | 122 | teacher_compressed_sideinfo | q4_top10 | 60 | 0.121246 | 19.738488 | 1.254468 | nan | 0 |
| teacher_vs_predictor_keep_fraction | 125 | adapter_delta_selected_predictor | q4_top10 | 60 | 0.117298 | 18.994813 | 0.044010 | nan | 1 |
| teacher_vs_predictor_keep_fraction | 129 | dedicated_mlp_selected_predictor | q4_top10 | 60 | 0.117298 | 18.865778 | -0.085025 | nan | 1 |
| teacher_vs_predictor_keep_fraction | 122 | teacher_compressed_sideinfo | q4_top20 | 60 | 0.133768 | 20.689271 | 2.205251 | nan | 0 |
| teacher_vs_predictor_keep_fraction | 125 | adapter_delta_selected_predictor | q4_top20 | 60 | 0.117298 | 19.010259 | 0.059456 | nan | 1 |
| teacher_vs_predictor_keep_fraction | 129 | dedicated_mlp_selected_predictor | q4_top20 | 60 | 0.117298 | 18.765203 | -0.185600 | nan | 1 |

## Outputs

- ablation CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage131_predictor_ablation_package/stage131_predictor_ablation_rows.csv`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage131_predictor_ablation_package/stage131_predictor_ablation_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage131_predictor_ablation_package/stage131_predictor_ablation_report.md`
