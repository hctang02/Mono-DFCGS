# Stage140 Multi-Setting Ablation Package

## Conclusion

- Final primary: `q4_top20` scale `0.75`.
- Final primary PSNR: `19.022110` at rate `0.117298`.
- Improvement over Stage132 q4/top20 scale1: `0.011850` dB.
- Dedicated MLP remains rejected because rendered PSNR regresses despite residual-MSE improvements.
- Teacher residual side-info is not used as a deployable or optimization target in this package.

## Selected Rows

| setting | role | keep | scale | rate | PSNR | delta base | delta Stage132 |
|---|---|---:|---:|---:|---:|---:|---:|
| q4_top10 | low_rate | 0.1 | 0.75 | 0.117298 | 18.997891 | 0.047088 | 0.003077 |
| q4_top20 | primary | 0.2 | 0.75 | 0.117298 | 19.022110 | 0.071307 | 0.011850 |

## Rejected MLP Rows

| setting | role | rate | PSNR | delta base | reason |
|---|---|---:|---:|---:|---|
| q4_top10 | low_rate | 0.117298 | 18.865778 | -0.085025 | rendered PSNR regression in Stage129 and Stage134 |
| q4_top20 | primary | 0.117298 | 18.765203 | -0.185600 | rendered PSNR regression in Stage129 and Stage134 |

## Outputs

- ablation CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage140_multi_setting_ablation_package/stage140_multi_setting_ablation_rows.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage140_multi_setting_ablation_package/stage140_multi_setting_ablation_summary.json`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage140_multi_setting_ablation_package/stage140_multi_setting_ablation_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage140_multi_setting_ablation_package/stage140_multi_setting_ablation_report.md`
