# Stage157 Selected Half-Anchor Broader Validation

## Summary

| gap | tasks | PSNR | p10 PSNR | SSIM | MS-SSIM | LPIPS | p90 LPIPS | original PSNR | original LPIPS | payload bytes | direct rate ref | delta PSNR | delta LPIPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 60 | 29.780485 | 25.726691 | 0.877938 | 0.985088 | 0.166020 | 0.209794 | 22.064218 | 0.301495 | 209392.833333 | 0.381631 | 7.716267 | -0.135475 |
| 8 | 60 | 29.578682 | 25.326198 | 0.869660 | 0.983847 | 0.178535 | 0.239232 | 20.337275 | 0.359337 | 215967.883333 | 0.303588 | 9.241407 | -0.180802 |

## Bad Cases

- Worst-LPIPS contact sheet: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_worst_lpips_contact_sheet.jpg`

| rank type | rank | sequence | gap | target | half | PSNR | SSIM | LPIPS | payload bytes |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| highest_lpips | 1 | motocross-jump | 4 | 22 | right | 30.119210 | 0.799854 | 0.335737 | 253183.000 |
| highest_lpips | 2 | motocross-jump | 8 | 13 | right | 31.063830 | 0.781243 | 0.319136 | 263358.000 |
| highest_lpips | 3 | shooting | 8 | 12 | left | 32.642755 | 0.872109 | 0.287687 | 249775.000 |
| highest_lpips | 4 | dogs-jump | 8 | 20 | right | 30.570198 | 0.818132 | 0.277165 | 167862.000 |
| highest_lpips | 5 | dance-twirl | 4 | 66 | right | 26.123000 | 0.789943 | 0.273550 | 197156.000 |
| highest_lpips | 6 | dance-twirl | 8 | 68 | right | 26.483362 | 0.796150 | 0.270898 | 212763.000 |
| highest_lpips | 7 | motocross-jump | 4 | 23 | right | 32.663031 | 0.891189 | 0.265142 | 241694.000 |
| highest_lpips | 8 | motocross-jump | 8 | 22 | right | 32.515662 | 0.891152 | 0.246195 | 247999.000 |
| highest_lpips | 9 | india | 8 | 30 | right | 32.043730 | 0.864327 | 0.244911 | 230255.000 |
| highest_lpips | 10 | india | 8 | 27 | left | 31.727686 | 0.859652 | 0.238601 | 233735.000 |
| highest_lpips | 11 | bike-packing | 8 | 11 | left | 25.622483 | 0.798668 | 0.231927 | 180774.000 |
| highest_lpips | 12 | dance-twirl | 4 | 82 | left | 27.389017 | 0.834274 | 0.224784 | 215387.000 |
| lowest_psnr | 1 | scooter-black | 8 | 4 | left | 24.488569 | 0.860408 | 0.203754 | 218516.000 |
| lowest_psnr | 2 | cows | 8 | 42 | left | 24.676417 | 0.765328 | 0.216960 | 215843.000 |
| lowest_psnr | 3 | cows | 8 | 60 | left | 24.684719 | 0.765132 | 0.208816 | 231584.000 |
| lowest_psnr | 4 | cows | 4 | 19 | right | 24.703186 | 0.773359 | 0.208441 | 225030.000 |
| lowest_psnr | 5 | cows | 4 | 15 | right | 24.708307 | 0.768624 | 0.204665 | 208856.000 |
| lowest_psnr | 6 | bike-packing | 8 | 26 | left | 25.048494 | 0.831531 | 0.193434 | 159176.000 |
| lowest_psnr | 7 | scooter-black | 8 | 10 | left | 25.139114 | 0.879352 | 0.205808 | 223623.000 |
| lowest_psnr | 8 | breakdance | 8 | 6 | right | 25.242664 | 0.844199 | 0.197324 | 142447.000 |
| lowest_psnr | 9 | camel | 8 | 75 | left | 25.335479 | 0.789262 | 0.199380 | 244271.000 |
| lowest_psnr | 10 | breakdance | 8 | 61 | right | 25.427564 | 0.851563 | 0.179372 | 155857.000 |
| lowest_psnr | 11 | camel | 4 | 51 | right | 25.533399 | 0.800710 | 0.189615 | 225503.000 |
| lowest_psnr | 12 | breakdance | 4 | 55 | right | 25.543567 | 0.859148 | 0.171395 | 126100.000 |
| lowest_ssim | 1 | cows | 8 | 60 | left | 24.684719 | 0.765132 | 0.208816 | 231584.000 |
| lowest_ssim | 2 | cows | 8 | 42 | left | 24.676417 | 0.765328 | 0.216960 | 215843.000 |
| lowest_ssim | 3 | cows | 4 | 15 | right | 24.708307 | 0.768624 | 0.204665 | 208856.000 |
| lowest_ssim | 4 | cows | 4 | 19 | right | 24.703186 | 0.773359 | 0.208441 | 225030.000 |
| lowest_ssim | 5 | motocross-jump | 8 | 13 | right | 31.063830 | 0.781243 | 0.319136 | 263358.000 |
| lowest_ssim | 6 | camel | 8 | 75 | left | 25.335479 | 0.789262 | 0.199380 | 244271.000 |
| lowest_ssim | 7 | dance-twirl | 4 | 66 | right | 26.123000 | 0.789943 | 0.273550 | 197156.000 |
| lowest_ssim | 8 | dance-twirl | 8 | 68 | right | 26.483362 | 0.796150 | 0.270898 | 212763.000 |
| lowest_ssim | 9 | bike-packing | 8 | 11 | left | 25.622483 | 0.798668 | 0.231927 | 180774.000 |
| lowest_ssim | 10 | motocross-jump | 4 | 22 | right | 30.119210 | 0.799854 | 0.335737 | 253183.000 |
| lowest_ssim | 11 | parkour | 8 | 90 | left | 25.567700 | 0.800016 | 0.223020 | 223869.000 |
| lowest_ssim | 12 | paragliding-launch | 8 | 18 | left | 29.263436 | 0.800267 | 0.200591 | 222111.000 |

## Contract

- This validates only `best_half_selector/keep1.0/q6`.
- The target dense anchor is encoder-side only and is not a decoder input.
- Decoder receives original StreamSplat base, entropy residual payload, normalized time, and one counted half-selector byte.

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_rows.csv`
- summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_summary.csv`
- badcases CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_badcases.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_summary.json`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage157_selected_half_anchor_broader_validation/stage157_selected_half_anchor_broader_validation_package.json`
