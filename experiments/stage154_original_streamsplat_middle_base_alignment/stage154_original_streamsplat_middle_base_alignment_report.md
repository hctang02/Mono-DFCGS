# Stage154 Original StreamSplat Middle Base Alignment

## Summary

| gap | method | tasks | PSNR mean | PSNR p10 | SSIM mean | MS-SSIM mean | LPIPS mean | LPIPS p90 | delta PSNR vs Stage151 | delta LPIPS vs Stage151 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | original_streamsplat_middle_base | 60 | 22.064218 | 17.192962 | 0.600809 | 0.801367 | 0.301495 | 0.424175 | -0.831238 | -0.046033 |
| 8 | original_streamsplat_middle_base | 60 | 20.337275 | 14.745159 | 0.520331 | 0.700537 | 0.359337 | 0.551970 | -1.472576 | -0.024897 |

## Bad-Case Contact Sheets

- highest_original_lpips: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage154_original_streamsplat_middle_base_alignment/stage154_badcases_highest_original_lpips.jpg`
- lowest_original_ssim: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage154_original_streamsplat_middle_base_alignment/stage154_badcases_lowest_original_ssim.jpg`
- lowest_original_psnr: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage154_original_streamsplat_middle_base_alignment/stage154_badcases_lowest_original_psnr.jpg`
- largest_psnr_drop_vs_stage151: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage154_original_streamsplat_middle_base_alignment/stage154_badcases_largest_psnr_drop_vs_stage151.jpg`

## Worst Original StreamSplat Cases

| rank type | rank | sequence | gap | target | original PSNR | original SSIM | original LPIPS | Stage151 PSNR | Stage151 LPIPS |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| highest_original_lpips | 1 | motocross-jump | 8 | 13 | 14.721935 | 0.426884 | 0.677003 | 17.096495 | 0.612672 |
| highest_original_lpips | 2 | motocross-jump | 8 | 22 | 12.472898 | 0.421006 | 0.668851 | 14.017364 | 0.572699 |
| highest_original_lpips | 3 | shooting | 8 | 12 | 13.832040 | 0.292904 | 0.656685 | 16.652654 | 0.587621 |
| highest_original_lpips | 4 | motocross-jump | 4 | 22 | 13.382143 | 0.424940 | 0.650311 | 14.469214 | 0.570376 |
| highest_original_lpips | 5 | motocross-jump | 4 | 23 | 14.094693 | 0.447696 | 0.620822 | 16.019436 | 0.521398 |
| highest_original_lpips | 6 | bmx-trees | 8 | 35 | 15.552224 | 0.199599 | 0.576309 | 16.530294 | 0.593241 |
| highest_original_lpips | 7 | lab-coat | 8 | 36 | 14.747739 | 0.376497 | 0.567558 | 18.525502 | 0.472656 |
| highest_original_lpips | 8 | lab-coat | 8 | 35 | 14.324228 | 0.352954 | 0.563428 | 18.029071 | 0.488257 |
| highest_original_lpips | 9 | shooting | 8 | 9 | 17.212635 | 0.374890 | 0.550697 | 19.117397 | 0.512528 |
| highest_original_lpips | 10 | india | 8 | 27 | 18.494547 | 0.468735 | 0.512913 | 19.802258 | 0.481327 |
| highest_original_lpips | 11 | india | 8 | 30 | 17.664604 | 0.517101 | 0.510182 | 19.114715 | 0.462863 |
| highest_original_lpips | 12 | drift-straight | 8 | 19 | 14.424543 | 0.162181 | 0.498366 | 15.422413 | 0.503450 |
| lowest_original_ssim | 1 | goat | 8 | 75 | 14.551624 | 0.089162 | 0.487634 | 17.237333 | 0.509174 |
| lowest_original_ssim | 2 | goat | 4 | 3 | 16.282981 | 0.138955 | 0.394840 | 18.203420 | 0.479362 |
| lowest_original_ssim | 3 | goat | 8 | 78 | 16.448159 | 0.158483 | 0.375463 | 18.658392 | 0.423776 |
| lowest_original_ssim | 4 | drift-straight | 8 | 19 | 14.424543 | 0.162181 | 0.498366 | 15.422413 | 0.503450 |
| lowest_original_ssim | 5 | bmx-trees | 4 | 35 | 16.676978 | 0.198176 | 0.423349 | 17.720171 | 0.516287 |
| lowest_original_ssim | 6 | bmx-trees | 8 | 35 | 15.552224 | 0.199599 | 0.576309 | 16.530294 | 0.593241 |
| lowest_original_ssim | 7 | goat | 4 | 46 | 18.115857 | 0.231812 | 0.418133 | 18.708111 | 0.481246 |
| lowest_original_ssim | 8 | dog | 4 | 6 | 20.655548 | 0.261673 | 0.386242 | 22.344150 | 0.416744 |
| lowest_original_ssim | 9 | shooting | 8 | 12 | 13.832040 | 0.292904 | 0.656685 | 16.652654 | 0.587621 |
| lowest_original_ssim | 10 | dance-twirl | 8 | 68 | 16.853718 | 0.294074 | 0.456838 | 20.904268 | 0.468947 |
| lowest_original_ssim | 11 | blackswan | 8 | 10 | 19.851389 | 0.325820 | 0.394063 | 21.865117 | 0.399855 |
| lowest_original_ssim | 12 | camel | 8 | 75 | 19.213472 | 0.331522 | 0.308488 | 20.408097 | 0.360247 |
| lowest_original_psnr | 1 | motocross-jump | 8 | 22 | 12.472898 | 0.421006 | 0.668851 | 14.017364 | 0.572699 |
| lowest_original_psnr | 2 | motocross-jump | 4 | 22 | 13.382143 | 0.424940 | 0.650311 | 14.469214 | 0.570376 |
| lowest_original_psnr | 3 | shooting | 8 | 12 | 13.832040 | 0.292904 | 0.656685 | 16.652654 | 0.587621 |
| lowest_original_psnr | 4 | motocross-jump | 4 | 23 | 14.094693 | 0.447696 | 0.620822 | 16.019436 | 0.521398 |
| lowest_original_psnr | 5 | lab-coat | 8 | 35 | 14.324228 | 0.352954 | 0.563428 | 18.029071 | 0.488257 |
| lowest_original_psnr | 6 | drift-straight | 8 | 19 | 14.424543 | 0.162181 | 0.498366 | 15.422413 | 0.503450 |
| lowest_original_psnr | 7 | goat | 8 | 75 | 14.551624 | 0.089162 | 0.487634 | 17.237333 | 0.509174 |
| lowest_original_psnr | 8 | scooter-black | 4 | 37 | 14.566760 | 0.393495 | 0.371429 | 14.891555 | 0.467697 |
| lowest_original_psnr | 9 | motocross-jump | 8 | 13 | 14.721935 | 0.426884 | 0.677003 | 17.096495 | 0.612672 |
| lowest_original_psnr | 10 | lab-coat | 8 | 36 | 14.747739 | 0.376497 | 0.567558 | 18.525502 | 0.472656 |
| lowest_original_psnr | 11 | scooter-black | 8 | 4 | 15.212313 | 0.484601 | 0.373527 | 17.570195 | 0.433052 |
| lowest_original_psnr | 12 | bmx-trees | 8 | 35 | 15.552224 | 0.199599 | 0.576309 | 16.530294 | 0.593241 |
| largest_psnr_drop_vs_stage151 | 1 | judo | 8 | 18 | 25.751269 | 0.864597 | 0.227466 | 30.916451 | 0.163401 |
| largest_psnr_drop_vs_stage151 | 2 | mbike-trick | 8 | 3 | 21.182562 | 0.470698 | 0.300695 | 25.984507 | 0.294047 |
| largest_psnr_drop_vs_stage151 | 3 | judo | 4 | 18 | 26.808279 | 0.879633 | 0.200938 | 31.550147 | 0.147370 |
| largest_psnr_drop_vs_stage151 | 4 | gold-fish | 4 | 70 | 25.574814 | 0.797020 | 0.165782 | 30.182137 | 0.147051 |
| largest_psnr_drop_vs_stage151 | 5 | dogs-jump | 4 | 17 | 25.234390 | 0.777798 | 0.237108 | 29.831807 | 0.173394 |
| largest_psnr_drop_vs_stage151 | 6 | breakdance | 4 | 25 | 20.003485 | 0.728360 | 0.246260 | 24.400077 | 0.200676 |
| largest_psnr_drop_vs_stage151 | 7 | breakdance | 4 | 55 | 20.542737 | 0.755303 | 0.215566 | 24.913170 | 0.179625 |
| largest_psnr_drop_vs_stage151 | 8 | breakdance | 8 | 6 | 19.164570 | 0.702839 | 0.265842 | 23.532120 | 0.215556 |
| largest_psnr_drop_vs_stage151 | 9 | breakdance | 8 | 61 | 18.670945 | 0.641762 | 0.293297 | 23.010645 | 0.230242 |
| largest_psnr_drop_vs_stage151 | 10 | bike-packing | 8 | 11 | 18.997019 | 0.622581 | 0.351908 | 23.105580 | 0.293334 |
| largest_psnr_drop_vs_stage151 | 11 | dance-twirl | 8 | 68 | 16.853718 | 0.294074 | 0.456838 | 20.904268 | 0.468947 |
| largest_psnr_drop_vs_stage151 | 12 | mbike-trick | 8 | 6 | 22.111921 | 0.516444 | 0.318609 | 26.061284 | 0.287242 |

## Decision Use

- Stage154 establishes the original StreamSplat-guided base profile on the same task-sampled diagnostic protocol as Stage153.
- If original StreamSplat is visually more stable but lower PSNR, Stage155 should apply rate-counted side-info on top of this base rather than linear interpolation.
- If original StreamSplat is not better on this task protocol, the next model stage still needs a StreamSplat-guided adapter rather than sparse linear residual correction.

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage154_original_streamsplat_middle_base_alignment/stage154_original_streamsplat_middle_rows.csv`
- badcases CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage154_original_streamsplat_middle_base_alignment/stage154_original_streamsplat_middle_badcases.csv`
- summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage154_original_streamsplat_middle_base_alignment/stage154_original_streamsplat_middle_summary.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage154_original_streamsplat_middle_base_alignment/stage154_original_streamsplat_middle_summary.json`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage154_original_streamsplat_middle_base_alignment/stage154_original_streamsplat_middle_base_alignment_package.json`
