# Stage156 StreamSplat Half-Anchor Gaussian Residual Side-Info

## Summary

| gap | policy | keep | bits | tasks | PSNR mean | PSNR p10 | SSIM mean | MS-SSIM mean | LPIPS mean | LPIPS p90 | payload bytes | direct rate ref | delta PSNR | delta LPIPS |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | best_half_selector | 0.2 | 4 | 30 | 23.126016 | 18.975417 | 0.647239 | 0.845219 | 0.348193 | 0.481044 | 30771.200000 | 0.211284 | 0.804190 | 0.050708 |
| 4 | best_half_selector | 0.2 | 6 | 30 | 23.495141 | 19.084285 | 0.667487 | 0.854017 | 0.326047 | 0.456803 | 55507.433333 | 0.234874 | 1.173315 | 0.028562 |
| 4 | best_half_selector | 0.4 | 4 | 30 | 23.684324 | 19.804221 | 0.661662 | 0.857898 | 0.358246 | 0.456019 | 52498.533333 | 0.232005 | 1.362498 | 0.060761 |
| 4 | best_half_selector | 0.4 | 6 | 30 | 24.442977 | 20.090843 | 0.704555 | 0.876343 | 0.315581 | 0.412512 | 100859.300000 | 0.278125 | 2.121151 | 0.018096 |
| 4 | best_half_selector | 1.0 | 4 | 30 | 26.349792 | 22.891766 | 0.741420 | 0.927690 | 0.319903 | 0.412269 | 91178.666667 | 0.268893 | 4.027967 | 0.022418 |
| 4 | best_half_selector | 1.0 | 6 | 30 | 29.880609 | 26.065687 | 0.879575 | 0.985351 | 0.164580 | 0.210143 | 207591.666667 | 0.379913 | 7.558783 | -0.132905 |
| 4 | fixed_left_half | 0.2 | 4 | 30 | 20.192949 | 13.948729 | 0.567703 | 0.743926 | 0.434660 | 0.575857 | 30355.033333 | 0.210887 | -2.128877 | 0.137175 |
| 4 | fixed_left_half | 0.2 | 6 | 30 | 20.560405 | 14.009207 | 0.599910 | 0.757373 | 0.409758 | 0.566485 | 55289.233333 | 0.234666 | -1.761421 | 0.112273 |
| 4 | fixed_left_half | 0.4 | 4 | 30 | 21.404446 | 16.295185 | 0.589078 | 0.789729 | 0.421079 | 0.515417 | 51835.200000 | 0.231372 | -0.917380 | 0.123594 |
| 4 | fixed_left_half | 0.4 | 6 | 30 | 22.196282 | 16.671408 | 0.650749 | 0.815294 | 0.372998 | 0.482638 | 100828.433333 | 0.278096 | -0.125543 | 0.075513 |
| 4 | fixed_left_half | 1.0 | 4 | 30 | 25.113472 | 21.569993 | 0.687017 | 0.902872 | 0.351026 | 0.430880 | 94393.433333 | 0.271959 | 2.791646 | 0.053541 |
| 4 | fixed_left_half | 1.0 | 6 | 30 | 29.402615 | 25.682445 | 0.861855 | 0.981509 | 0.185494 | 0.236311 | 211415.566667 | 0.383560 | 7.080790 | -0.111991 |
| 4 | fixed_right_half | 0.2 | 4 | 30 | 20.392880 | 15.849679 | 0.567546 | 0.762472 | 0.427133 | 0.551844 | 30800.766667 | 0.211312 | -1.928946 | 0.129648 |
| 4 | fixed_right_half | 0.2 | 6 | 30 | 20.681673 | 16.132231 | 0.595318 | 0.774630 | 0.402760 | 0.531061 | 55663.100000 | 0.235023 | -1.640153 | 0.105275 |
| 4 | fixed_right_half | 0.4 | 4 | 30 | 21.580958 | 17.725395 | 0.580481 | 0.798555 | 0.424415 | 0.532330 | 52664.300000 | 0.232163 | -0.740868 | 0.126930 |
| 4 | fixed_right_half | 0.4 | 6 | 30 | 22.276666 | 18.490166 | 0.640659 | 0.824979 | 0.377411 | 0.482415 | 101814.666667 | 0.279036 | -0.045160 | 0.079926 |
| 4 | fixed_right_half | 1.0 | 4 | 30 | 25.641553 | 22.599941 | 0.703751 | 0.911764 | 0.342458 | 0.448862 | 95193.833333 | 0.272722 | 3.319727 | 0.044973 |
| 4 | fixed_right_half | 1.0 | 6 | 30 | 29.660264 | 25.723518 | 0.870334 | 0.983956 | 0.176548 | 0.251807 | 214094.633333 | 0.386115 | 7.338439 | -0.120937 |
| 8 | best_half_selector | 0.2 | 4 | 30 | 21.967290 | 17.640831 | 0.589582 | 0.792351 | 0.385087 | 0.533596 | 31984.466667 | 0.128128 | 1.442109 | 0.026872 |
| 8 | best_half_selector | 0.2 | 6 | 30 | 22.314967 | 17.618372 | 0.615469 | 0.802799 | 0.368166 | 0.523826 | 57261.600000 | 0.152234 | 1.789786 | 0.009951 |
| 8 | best_half_selector | 0.4 | 4 | 30 | 22.668976 | 18.836361 | 0.615586 | 0.824434 | 0.387267 | 0.490636 | 55182.900000 | 0.150252 | 2.143795 | 0.029052 |
| 8 | best_half_selector | 0.4 | 6 | 30 | 23.346907 | 18.917331 | 0.662422 | 0.843057 | 0.349881 | 0.452910 | 105098.333333 | 0.197855 | 2.821726 | -0.008334 |
| 8 | best_half_selector | 1.0 | 4 | 30 | 25.882061 | 22.643228 | 0.718232 | 0.918756 | 0.333796 | 0.430054 | 96088.433333 | 0.189262 | 5.356880 | -0.024419 |
| 8 | best_half_selector | 1.0 | 6 | 30 | 29.547440 | 25.535196 | 0.870055 | 0.984077 | 0.177570 | 0.233225 | 214067.133333 | 0.301776 | 9.022259 | -0.180644 |
| 8 | fixed_left_half | 0.2 | 4 | 30 | 18.822242 | 14.605860 | 0.493601 | 0.676013 | 0.484592 | 0.651695 | 31373.833333 | 0.127546 | -1.702939 | 0.126377 |
| 8 | fixed_left_half | 0.2 | 6 | 30 | 19.161634 | 14.797905 | 0.530968 | 0.693157 | 0.461350 | 0.627200 | 56796.233333 | 0.151790 | -1.363547 | 0.103136 |
| 8 | fixed_left_half | 0.4 | 4 | 30 | 20.239720 | 16.195310 | 0.523775 | 0.746903 | 0.460512 | 0.571691 | 54610.700000 | 0.149706 | -0.285461 | 0.102298 |
| 8 | fixed_left_half | 0.4 | 6 | 30 | 20.975260 | 16.699663 | 0.595289 | 0.780908 | 0.413843 | 0.521292 | 105204.533333 | 0.197956 | 0.450079 | 0.055628 |
| 8 | fixed_left_half | 1.0 | 4 | 30 | 24.661423 | 21.366084 | 0.657203 | 0.887839 | 0.363485 | 0.443959 | 99862.433333 | 0.192862 | 4.136243 | 0.005270 |
| 8 | fixed_left_half | 1.0 | 6 | 30 | 28.908205 | 25.524841 | 0.844577 | 0.978536 | 0.202989 | 0.296580 | 221160.700000 | 0.308541 | 8.383024 | -0.155226 |
| 8 | fixed_right_half | 0.2 | 4 | 30 | 19.328298 | 14.754059 | 0.520356 | 0.706495 | 0.466984 | 0.607931 | 31736.433333 | 0.127892 | -1.196882 | 0.108769 |
| 8 | fixed_right_half | 0.2 | 6 | 30 | 19.660077 | 14.968070 | 0.563109 | 0.728967 | 0.441431 | 0.574563 | 57183.566667 | 0.152160 | -0.865104 | 0.083217 |
| 8 | fixed_right_half | 0.4 | 4 | 30 | 20.464903 | 17.309291 | 0.535056 | 0.752399 | 0.454946 | 0.559490 | 55426.166667 | 0.150484 | -0.060278 | 0.096731 |
| 8 | fixed_right_half | 0.4 | 6 | 30 | 21.253202 | 17.645982 | 0.615372 | 0.795777 | 0.403104 | 0.469849 | 105784.933333 | 0.198510 | 0.728022 | 0.044890 |
| 8 | fixed_right_half | 1.0 | 4 | 30 | 24.122354 | 19.009416 | 0.637805 | 0.871075 | 0.377717 | 0.505206 | 100265.600000 | 0.193246 | 3.597173 | 0.019502 |
| 8 | fixed_right_half | 1.0 | 6 | 30 | 28.796993 | 25.170511 | 0.846921 | 0.978840 | 0.205176 | 0.266660 | 220415.033333 | 0.307830 | 8.271813 | -0.153039 |

## Original Baseline

| gap | PSNR mean | SSIM mean | MS-SSIM mean | LPIPS mean |
|---:|---:|---:|---:|---:|
| 4 | 22.321826 | 0.612966 | 0.812469 | 0.297485 |
| 8 | 20.525181 | 0.536849 | 0.713839 | 0.358215 |

## Best Setting

- Best sampled setting: `{'gap': 4, 'method': 'streamsplat_half_anchor_entropy_residual', 'half_policy': 'best_half_selector', 'keep_fraction': 1.0, 'side_bits': 6, 'task_count': 30, 'mean_psnr': 29.88060850586717, 'min_psnr': 24.708306541723843, 'p10_psnr': 26.06568680730128, 'mean_ssim': 0.8795753101507823, 'p10_ssim': 0.8259783327579499, 'mean_ms_ssim': 0.9853506326675415, 'mean_lpips': 0.16458002875248592, 'p90_lpips': 0.2101434350013733, 'mean_payload_bytes': 207591.66666666666, 'mean_side_mib_per_intermediate': 0.19797484079996744, 'mean_direct_total_mib_per_frame_ref': 0.3799130615678806, 'mean_delta_psnr_vs_original': 7.55878292393898, 'mean_delta_ssim_vs_original': 0.2666095033288002, 'mean_delta_ms_ssim_vs_original': 0.17288174827893574, 'mean_delta_lpips_vs_original': -0.1329050615429878}`
- Worst-LPIPS contact sheet: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage156_streamsplat_half_anchor_gaussian_residual/stage156_best_half_selector_worst_lpips_contact_sheet.jpg`

## Bad Cases

| rank type | rank | sequence | gap | target | half | keep | bits | PSNR | SSIM | LPIPS | payload bytes |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| best_half_highest_lpips | 1 | dance-twirl | 4 | 66 | right | 1.0 | 6 | 26.123000 | 0.789943 | 0.273550 | 197156.000 |
| best_half_highest_lpips | 2 | dance-twirl | 8 | 68 | right | 1.0 | 6 | 26.483362 | 0.796150 | 0.270898 | 212763.000 |
| best_half_highest_lpips | 3 | motocross-jump | 4 | 23 | right | 1.0 | 6 | 32.663031 | 0.891189 | 0.265142 | 241694.000 |
| best_half_highest_lpips | 4 | motocross-jump | 8 | 22 | right | 1.0 | 6 | 32.515662 | 0.891152 | 0.246195 | 247999.000 |
| best_half_highest_lpips | 5 | india | 8 | 30 | right | 1.0 | 6 | 32.043730 | 0.864327 | 0.244911 | 230255.000 |
| best_half_highest_lpips | 6 | bike-packing | 8 | 11 | left | 1.0 | 6 | 25.622483 | 0.798668 | 0.231927 | 180774.000 |
| best_half_highest_lpips | 7 | parkour | 8 | 90 | left | 1.0 | 6 | 25.567700 | 0.800016 | 0.223020 | 223869.000 |
| best_half_highest_lpips | 8 | judo | 4 | 18 | left | 1.0 | 6 | 30.011250 | 0.890257 | 0.219624 | 107700.000 |
| best_half_highest_lpips | 9 | mbike-trick | 8 | 6 | right | 1.0 | 6 | 29.778762 | 0.875795 | 0.210188 | 182121.000 |
| best_half_highest_lpips | 10 | parkour | 4 | 34 | left | 1.0 | 6 | 29.050035 | 0.828786 | 0.209090 | 225453.000 |
| best_half_highest_lpips | 11 | libby | 8 | 29 | right | 1.0 | 6 | 32.139548 | 0.856943 | 0.208868 | 262762.000 |
| best_half_highest_lpips | 12 | cows | 8 | 60 | left | 1.0 | 6 | 24.684719 | 0.765132 | 0.208816 | 231584.000 |
| best_half_lowest_psnr | 1 | cows | 8 | 60 | left | 1.0 | 6 | 24.684719 | 0.765132 | 0.208816 | 231584.000 |
| best_half_lowest_psnr | 2 | cows | 4 | 15 | right | 1.0 | 6 | 24.708307 | 0.768624 | 0.204665 | 208856.000 |
| best_half_lowest_psnr | 3 | scooter-black | 8 | 10 | left | 1.0 | 6 | 25.139114 | 0.879352 | 0.205808 | 223623.000 |
| best_half_lowest_psnr | 4 | breakdance | 8 | 6 | right | 1.0 | 6 | 25.242664 | 0.844199 | 0.197324 | 142447.000 |
| best_half_lowest_psnr | 5 | camel | 4 | 51 | right | 1.0 | 6 | 25.533399 | 0.800710 | 0.189615 | 225503.000 |
| best_half_lowest_psnr | 6 | parkour | 8 | 90 | left | 1.0 | 6 | 25.567700 | 0.800016 | 0.223020 | 223869.000 |
| best_half_lowest_psnr | 7 | bike-packing | 8 | 11 | left | 1.0 | 6 | 25.622483 | 0.798668 | 0.231927 | 180774.000 |
| best_half_lowest_psnr | 8 | breakdance | 4 | 25 | left | 1.0 | 6 | 25.713629 | 0.865872 | 0.168760 | 121358.000 |
| best_half_lowest_psnr | 9 | bike-packing | 4 | 1 | left | 1.0 | 6 | 26.104804 | 0.833530 | 0.202624 | 134579.000 |
| best_half_lowest_psnr | 10 | dance-twirl | 4 | 66 | right | 1.0 | 6 | 26.123000 | 0.789943 | 0.273550 | 197156.000 |
| best_half_lowest_psnr | 11 | camel | 8 | 23 | right | 1.0 | 6 | 26.203967 | 0.824133 | 0.180395 | 231360.000 |
| best_half_lowest_psnr | 12 | dance-twirl | 8 | 68 | right | 1.0 | 6 | 26.483362 | 0.796150 | 0.270898 | 212763.000 |
| best_half_lowest_ssim | 1 | cows | 8 | 60 | left | 1.0 | 6 | 24.684719 | 0.765132 | 0.208816 | 231584.000 |
| best_half_lowest_ssim | 2 | cows | 4 | 15 | right | 1.0 | 6 | 24.708307 | 0.768624 | 0.204665 | 208856.000 |
| best_half_lowest_ssim | 3 | dance-twirl | 4 | 66 | right | 1.0 | 6 | 26.123000 | 0.789943 | 0.273550 | 197156.000 |
| best_half_lowest_ssim | 4 | dance-twirl | 8 | 68 | right | 1.0 | 6 | 26.483362 | 0.796150 | 0.270898 | 212763.000 |
| best_half_lowest_ssim | 5 | bike-packing | 8 | 11 | left | 1.0 | 6 | 25.622483 | 0.798668 | 0.231927 | 180774.000 |
| best_half_lowest_ssim | 6 | parkour | 8 | 90 | left | 1.0 | 6 | 25.567700 | 0.800016 | 0.223020 | 223869.000 |
| best_half_lowest_ssim | 7 | camel | 4 | 51 | right | 1.0 | 6 | 25.533399 | 0.800710 | 0.189615 | 225503.000 |
| best_half_lowest_ssim | 8 | camel | 8 | 23 | right | 1.0 | 6 | 26.203967 | 0.824133 | 0.180395 | 231360.000 |
| best_half_lowest_ssim | 9 | horsejump-high | 8 | 20 | right | 1.0 | 6 | 27.513156 | 0.824295 | 0.207461 | 226888.000 |
| best_half_lowest_ssim | 10 | parkour | 4 | 34 | left | 1.0 | 6 | 29.050035 | 0.828786 | 0.209090 | 225453.000 |
| best_half_lowest_ssim | 11 | bike-packing | 4 | 1 | left | 1.0 | 6 | 26.104804 | 0.833530 | 0.202624 | 134579.000 |
| best_half_lowest_ssim | 12 | soapbox | 8 | 15 | right | 1.0 | 6 | 29.513428 | 0.838307 | 0.195829 | 213191.000 |

## Contract

- The target dense anchor is used encoder-side only to produce the residual payload and offline metrics.
- Decoder inputs are original StreamSplat endpoint inputs/base, normalized time, entropy residual payload, and a counted one-byte half selector for `best_half_selector`.
- Decoder does not receive unencoded target dense anchors, target RGB, or unencoded residual tensors.

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage156_streamsplat_half_anchor_gaussian_residual/stage156_streamsplat_half_anchor_rows.csv`
- summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage156_streamsplat_half_anchor_gaussian_residual/stage156_streamsplat_half_anchor_summary.csv`
- badcases CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage156_streamsplat_half_anchor_gaussian_residual/stage156_streamsplat_half_anchor_badcases.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage156_streamsplat_half_anchor_gaussian_residual/stage156_streamsplat_half_anchor_summary.json`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage156_streamsplat_half_anchor_gaussian_residual/stage156_streamsplat_half_anchor_gaussian_residual_package.json`
