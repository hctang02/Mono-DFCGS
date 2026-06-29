# Stage134 MLP Render Regression Diagnostic

## Summary

| group | setting | gap | tasks | adapter delta | MLP delta base | MLP delta adapter | MLP base regressions | MLP adapter regressions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gap | q4_top10 | 16 | 18 | 0.033994 | -0.069494 | -0.103489 | 7 | 11 |
| gap | q4_top10 | 4 | 23 | 0.048236 | -0.065378 | -0.113614 | 11 | 21 |
| gap | q4_top10 | 8 | 19 | 0.048384 | -0.123523 | -0.171907 | 10 | 15 |
| gap | q4_top20 | 16 | 18 | 0.040323 | -0.169600 | -0.209922 | 9 | 11 |
| gap | q4_top20 | 4 | 23 | 0.056435 | -0.147153 | -0.203588 | 13 | 21 |
| gap | q4_top20 | 8 | 19 | 0.081240 | -0.247299 | -0.328539 | 12 | 15 |
| setting | q4_top10 | all | 60 | 0.044010 | -0.085025 | -0.129036 | 28 | 47 |
| setting | q4_top20 | all | 60 | 0.059456 | -0.185600 | -0.245056 | 34 | 47 |

## Worst MLP Regressions Vs Adapter-Delta

| task | sequence | gap | setting | base | adapter | MLP | MLP-adapter |
|---|---|---:|---|---:|---:|---:|---:|
| stage79_00027031 | india | 8 | q4_top20 | 21.848984 | 22.064413 | 20.852532 | -1.211881 |
| stage79_00027268 | judo | 8 | q4_top20 | 30.111798 | 29.039396 | 27.946791 | -1.092605 |
| stage79_00027310 | judo | 16 | q4_top20 | 27.307187 | 27.120205 | 26.065455 | -1.054750 |
| stage79_00026493 | gold-fish | 16 | q4_top20 | 25.894456 | 25.745591 | 24.831162 | -0.914429 |
| stage79_00028856 | motocross-jump | 8 | q4_top20 | 21.332582 | 21.315955 | 20.449461 | -0.866494 |
| stage79_00028536 | mbike-trick | 4 | q4_top20 | 23.424839 | 23.867333 | 23.050577 | -0.816756 |
| stage79_00030952 | soapbox | 8 | q4_top20 | 20.263274 | 20.571570 | 19.846249 | -0.725322 |
| stage79_00027310 | judo | 16 | q4_top10 | 27.307187 | 27.293157 | 26.582273 | -0.710884 |
| stage79_00027031 | india | 8 | q4_top10 | 21.848984 | 21.979548 | 21.285597 | -0.693951 |
| stage79_00028856 | motocross-jump | 8 | q4_top10 | 21.332582 | 21.243495 | 20.556629 | -0.686867 |
| stage79_00026699 | horsejump-high | 8 | q4_top20 | 19.303614 | 19.736352 | 19.094445 | -0.641907 |
| stage79_00030141 | pigs | 16 | q4_top20 | 26.505739 | 26.342086 | 25.754613 | -0.587472 |

## Diagnosis

- MLP residual MSE reduction does not align with rendered PSNR.
- MLP underperforms adapter-delta on most tasks, so the next protocol must use render-aware selection/training criteria.
- Stage135 should formalize render-aware predictor training around rendered PSNR/RGB loss, not attribute MSE alone.

## Outputs

- diagnostic rows: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage134_mlp_render_regression_diagnostic/stage134_mlp_render_regression_diagnostic_rows.csv`
- summary rows: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage134_mlp_render_regression_diagnostic/stage134_mlp_render_regression_diagnostic_summary.csv`
- worst rows: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage134_mlp_render_regression_diagnostic/stage134_mlp_render_regression_worst_rows.csv`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage134_mlp_render_regression_diagnostic/stage134_mlp_render_regression_diagnostic_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage134_mlp_render_regression_diagnostic/stage134_mlp_render_regression_diagnostic_report.md`
