# Stage153 Middle Multi-Metric Bad-Case Evaluation

## Summary

| gap | method | tasks | PSNR mean | PSNR p10 | SSIM mean | MS-SSIM mean | LPIPS mean | LPIPS p90 | payload bytes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | linear_base | 60 | 19.779005 | 14.861659 | 0.472654 | 0.617425 | 0.362247 | 0.511952 | 0.000 |
| 4 | stage151_recovered_linear_base_sideinfo | 60 | 22.895456 | 17.653617 | 0.602004 | 0.772679 | 0.347528 | 0.508378 | 29908.517 |
| 8 | linear_base | 60 | 18.526650 | 14.041838 | 0.435803 | 0.548916 | 0.400331 | 0.566730 | 0.000 |
| 8 | stage151_recovered_linear_base_sideinfo | 60 | 21.809852 | 17.072955 | 0.563607 | 0.722634 | 0.384234 | 0.538803 | 30091.617 |

## Bad-Case Contact Sheets

- highest_recovered_lpips: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage153_middle_multimetric_badcase_eval/stage153_badcases_highest_recovered_lpips.jpg`
- lowest_recovered_ssim: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage153_middle_multimetric_badcase_eval/stage153_badcases_lowest_recovered_ssim.jpg`
- lowest_recovered_psnr: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage153_middle_multimetric_badcase_eval/stage153_badcases_lowest_recovered_psnr.jpg`
- largest_lpips_regression: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage153_middle_multimetric_badcase_eval/stage153_badcases_largest_lpips_regression.jpg`

## Worst Recovered Cases

| rank type | rank | sequence | gap | target | base PSNR | recovered PSNR | base SSIM | recovered SSIM | base LPIPS | recovered LPIPS |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| highest_recovered_lpips | 1 | motocross-jump | 8 | 13 | 14.041967 | 17.096495 | 0.391883 | 0.452920 | 0.647512 | 0.612672 |
| highest_recovered_lpips | 2 | bmx-trees | 8 | 35 | 14.471705 | 16.530294 | 0.162866 | 0.306102 | 0.626612 | 0.593241 |
| highest_recovered_lpips | 3 | shooting | 8 | 12 | 13.631672 | 16.652654 | 0.276429 | 0.399144 | 0.621630 | 0.587621 |
| highest_recovered_lpips | 4 | libby | 8 | 29 | 17.274640 | 19.498656 | 0.258107 | 0.383856 | 0.613038 | 0.579080 |
| highest_recovered_lpips | 5 | motocross-jump | 8 | 22 | 11.155776 | 14.017364 | 0.411326 | 0.474759 | 0.607636 | 0.572699 |
| highest_recovered_lpips | 6 | motocross-jump | 4 | 22 | 11.807717 | 14.469214 | 0.374274 | 0.475780 | 0.609220 | 0.570376 |
| highest_recovered_lpips | 7 | loading | 8 | 13 | 13.673749 | 16.847326 | 0.262823 | 0.428980 | 0.569773 | 0.546533 |
| highest_recovered_lpips | 8 | libby | 4 | 38 | 17.571666 | 20.652278 | 0.289964 | 0.433793 | 0.584921 | 0.539702 |
| highest_recovered_lpips | 9 | loading | 8 | 35 | 14.040679 | 16.861093 | 0.271752 | 0.437245 | 0.564312 | 0.537944 |
| highest_recovered_lpips | 10 | soapbox | 8 | 61 | 14.546673 | 17.597889 | 0.277697 | 0.421875 | 0.566392 | 0.529891 |
| highest_recovered_lpips | 11 | shooting | 4 | 7 | 14.975649 | 18.106242 | 0.341965 | 0.470037 | 0.524882 | 0.521547 |
| highest_recovered_lpips | 12 | motocross-jump | 4 | 23 | 12.424607 | 16.019436 | 0.444369 | 0.538955 | 0.562754 | 0.521398 |
| lowest_recovered_ssim | 1 | goat | 8 | 75 | 14.884560 | 17.237333 | 0.095890 | 0.303808 | 0.534325 | 0.509174 |
| lowest_recovered_ssim | 2 | bmx-trees | 8 | 35 | 14.471705 | 16.530294 | 0.162866 | 0.306102 | 0.626612 | 0.593241 |
| lowest_recovered_ssim | 3 | drift-straight | 8 | 19 | 12.735263 | 15.422413 | 0.122155 | 0.320316 | 0.522561 | 0.503450 |
| lowest_recovered_ssim | 4 | goat | 4 | 3 | 15.789835 | 18.203420 | 0.117682 | 0.327234 | 0.468858 | 0.479362 |
| lowest_recovered_ssim | 5 | goat | 8 | 78 | 16.223862 | 18.658392 | 0.115869 | 0.337386 | 0.427783 | 0.423776 |
| lowest_recovered_ssim | 6 | drift-straight | 4 | 37 | 13.835748 | 17.054628 | 0.171179 | 0.337867 | 0.478046 | 0.468409 |
| lowest_recovered_ssim | 7 | bmx-trees | 4 | 35 | 15.461306 | 17.720171 | 0.175865 | 0.338088 | 0.527350 | 0.516287 |
| lowest_recovered_ssim | 8 | goat | 4 | 46 | 16.487572 | 18.708111 | 0.120731 | 0.347738 | 0.508562 | 0.481246 |
| lowest_recovered_ssim | 9 | cows | 8 | 60 | 17.081896 | 19.453432 | 0.163638 | 0.351997 | 0.470482 | 0.440217 |
| lowest_recovered_ssim | 10 | parkour | 8 | 90 | 16.846177 | 19.137117 | 0.235224 | 0.381798 | 0.504180 | 0.496758 |
| lowest_recovered_ssim | 11 | drift-straight | 4 | 2 | 13.356301 | 16.467353 | 0.188750 | 0.383606 | 0.494541 | 0.466129 |
| lowest_recovered_ssim | 12 | libby | 8 | 29 | 17.274640 | 19.498656 | 0.258107 | 0.383856 | 0.613038 | 0.579080 |
| lowest_recovered_psnr | 1 | motocross-jump | 8 | 22 | 11.155776 | 14.017364 | 0.411326 | 0.474759 | 0.607636 | 0.572699 |
| lowest_recovered_psnr | 2 | motocross-jump | 4 | 22 | 11.807717 | 14.469214 | 0.374274 | 0.475780 | 0.609220 | 0.570376 |
| lowest_recovered_psnr | 3 | scooter-black | 4 | 37 | 11.607173 | 14.891555 | 0.236286 | 0.398263 | 0.490940 | 0.467697 |
| lowest_recovered_psnr | 4 | drift-straight | 8 | 19 | 12.735263 | 15.422413 | 0.122155 | 0.320316 | 0.522561 | 0.503450 |
| lowest_recovered_psnr | 5 | scooter-black | 4 | 27 | 12.429354 | 15.699783 | 0.241082 | 0.416138 | 0.469116 | 0.456334 |
| lowest_recovered_psnr | 6 | motocross-jump | 4 | 23 | 12.424607 | 16.019436 | 0.444369 | 0.538955 | 0.562754 | 0.521398 |
| lowest_recovered_psnr | 7 | drift-straight | 4 | 2 | 13.356301 | 16.467353 | 0.188750 | 0.383606 | 0.494541 | 0.466129 |
| lowest_recovered_psnr | 8 | bmx-trees | 8 | 35 | 14.471705 | 16.530294 | 0.162866 | 0.306102 | 0.626612 | 0.593241 |
| lowest_recovered_psnr | 9 | shooting | 8 | 12 | 13.631672 | 16.652654 | 0.276429 | 0.399144 | 0.621630 | 0.587621 |
| lowest_recovered_psnr | 10 | loading | 8 | 13 | 13.673749 | 16.847326 | 0.262823 | 0.428980 | 0.569773 | 0.546533 |
| lowest_recovered_psnr | 11 | loading | 8 | 35 | 14.040679 | 16.861093 | 0.271752 | 0.437245 | 0.564312 | 0.537944 |
| lowest_recovered_psnr | 12 | drift-straight | 4 | 37 | 13.835748 | 17.054628 | 0.171179 | 0.337867 | 0.478046 | 0.468409 |
| largest_lpips_regression | 1 | soapbox | 8 | 15 | 22.015407 | 25.106168 | 0.493351 | 0.612222 | 0.295821 | 0.323822 |
| largest_lpips_regression | 2 | india | 4 | 73 | 19.949218 | 23.671931 | 0.466756 | 0.633329 | 0.290429 | 0.314045 |
| largest_lpips_regression | 3 | drift-straight | 8 | 9 | 16.414579 | 20.580593 | 0.414357 | 0.585116 | 0.286172 | 0.307862 |
| largest_lpips_regression | 4 | camel | 8 | 23 | 18.822514 | 20.875628 | 0.359144 | 0.482136 | 0.298733 | 0.318438 |

## Interpretation

- Stage153 is diagnostic: it evaluates whether PSNR gains correspond to SSIM/LPIPS and visual sanity.
- If LPIPS or bad-case sheets show visually broken frames, the next stage must move from linear-base recovery to original StreamSplat-guided recovery.
- Side-info remains rate-counted; target dense anchors are used only encoder-side to build the payload and metrics.

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage153_middle_multimetric_badcase_eval/stage153_middle_multimetric_rows.csv`
- pair rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage153_middle_multimetric_badcase_eval/stage153_middle_multimetric_pair_rows.csv`
- badcases CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage153_middle_multimetric_badcase_eval/stage153_middle_multimetric_badcases.csv`
- summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage153_middle_multimetric_badcase_eval/stage153_middle_multimetric_summary.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage153_middle_multimetric_badcase_eval/stage153_middle_multimetric_summary.json`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage153_middle_multimetric_badcase_eval/stage153_middle_multimetric_badcase_eval_package.json`
