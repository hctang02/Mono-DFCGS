# Stage155 StreamSplat-Base Side-Info Upper-Bound Sweep

## Summary

| gap | method | bits | tasks | PSNR mean | PSNR p10 | SSIM mean | MS-SSIM mean | LPIPS mean | LPIPS p90 | payload bytes | direct rate ref | delta PSNR | delta LPIPS |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | image_residual_sideinfo_full_frame | 3 | 30 | 25.134759 | 22.747345 | 0.726357 | 0.923685 | 0.313501 | 0.415658 | 55879.366667 | 0.235229 | 2.812933 | 0.016015 |
| 4 | image_residual_sideinfo_full_frame | 4 | 30 | 32.280546 | 30.361704 | 0.878672 | 0.977598 | 0.152828 | 0.205055 | 78548.733333 | 0.256848 | 9.958721 | -0.144657 |
| 4 | image_residual_sideinfo_full_frame | 5 | 30 | 38.582637 | 37.034630 | 0.955648 | 0.993758 | 0.063664 | 0.093891 | 152725.966667 | 0.327589 | 16.260812 | -0.233821 |
| 4 | image_residual_sideinfo_full_frame | 6 | 30 | 44.611106 | 42.852296 | 0.986849 | 0.998398 | 0.021616 | 0.051908 | 208014.666667 | 0.380316 | 22.289280 | -0.275869 |
| 4 | original_streamsplat_base | 0 | 30 | 22.321826 | 18.184874 | 0.612966 | 0.812469 | 0.297485 | 0.420194 | 0.000000 | 0.181938 | 0.000000 | 0.000000 |
| 8 | image_residual_sideinfo_full_frame | 3 | 30 | 24.811277 | 22.062703 | 0.708892 | 0.914862 | 0.325586 | 0.423949 | 56604.500000 | 0.151608 | 4.286096 | -0.032629 |
| 8 | image_residual_sideinfo_full_frame | 4 | 30 | 31.718739 | 29.540456 | 0.863465 | 0.974260 | 0.170844 | 0.249120 | 82349.400000 | 0.176160 | 11.193559 | -0.187371 |
| 8 | image_residual_sideinfo_full_frame | 5 | 30 | 37.935986 | 36.321179 | 0.950047 | 0.992997 | 0.072469 | 0.130468 | 159691.566667 | 0.249919 | 17.410806 | -0.285746 |
| 8 | image_residual_sideinfo_full_frame | 6 | 30 | 44.071608 | 42.390908 | 0.985547 | 0.998248 | 0.023286 | 0.051691 | 215642.000000 | 0.303278 | 23.546428 | -0.334929 |
| 8 | original_streamsplat_base | 0 | 30 | 20.525181 | 15.471776 | 0.536849 | 0.713839 | 0.358215 | 0.552383 | 0.000000 | 0.097625 | 0.000000 | 0.000000 |

## Gaussian-Domain Diagnostic

- Shape-matched tasks: `0/60`.
- Max static-eval render diff vs dynamic render: `0.0`.
- If shape-matched tasks are zero, direct dense-anchor residual side-info is not a valid final Gaussian-domain codec for original StreamSplat because the base has a different Gaussian correspondence/count from Stage61 dense target anchors.

## Best Setting

- Best sampled setting: `{'gap': 4, 'method': 'image_residual_sideinfo_full_frame', 'side_bits': 4, 'task_count': 30, 'mean_psnr': 32.280546434337076, 'min_psnr': 29.30915536016212, 'p10_psnr': 30.36170381171098, 'mean_ssim': 0.8786723375320434, 'min_ssim': 0.7870612144470215, 'p10_ssim': 0.8279148757457733, 'mean_ms_ssim': 0.9775982022285461, 'mean_lpips': 0.15282797639568646, 'p90_lpips': 0.20505481958389282, 'mean_payload_bytes': 78548.73333333334, 'mean_side_mib_per_intermediate': 0.0749099095662435, 'mean_direct_total_mib_per_frame_ref': 0.2568481303341566, 'mean_delta_psnr_vs_original': 9.958720852408895, 'mean_delta_ssim_vs_original': 0.2657065307100614, 'mean_delta_ms_ssim_vs_original': 0.1651293178399404, 'mean_delta_lpips_vs_original': -0.14465711389978728}`
- Worst-LPIPS contact sheet: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage155_streamsplat_base_sideinfo_upper_bound/stage155_best_setting_worst_lpips_contact_sheet.jpg`

## Bad Cases

| rank type | rank | sequence | gap | target | method | bits | PSNR | SSIM | LPIPS | payload bytes |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| best_setting_highest_lpips | 1 | shooting | 4 | 7 | image_residual_sideinfo_full_frame | 4 | 31.042856 | 0.787061 | 0.362374 | 63795.000 |
| best_setting_highest_lpips | 2 | shooting | 8 | 9 | image_residual_sideinfo_full_frame | 4 | 31.498265 | 0.786920 | 0.322325 | 75615.000 |
| best_setting_highest_lpips | 3 | motocross-jump | 4 | 23 | image_residual_sideinfo_full_frame | 4 | 32.141099 | 0.803875 | 0.299190 | 81415.000 |
| best_setting_highest_lpips | 4 | india | 8 | 30 | image_residual_sideinfo_full_frame | 4 | 31.521074 | 0.802642 | 0.275832 | 69400.000 |
| best_setting_highest_lpips | 5 | motocross-jump | 8 | 22 | image_residual_sideinfo_full_frame | 4 | 31.759368 | 0.807489 | 0.272686 | 79279.000 |
| best_setting_highest_lpips | 6 | scooter-black | 8 | 10 | image_residual_sideinfo_full_frame | 4 | 28.365447 | 0.818228 | 0.246502 | 86793.000 |
| best_setting_highest_lpips | 7 | horsejump-high | 8 | 20 | image_residual_sideinfo_full_frame | 4 | 29.174371 | 0.753123 | 0.238629 | 83995.000 |
| best_setting_highest_lpips | 8 | mbike-trick | 8 | 6 | image_residual_sideinfo_full_frame | 4 | 30.517429 | 0.864108 | 0.236433 | 68490.000 |
| best_setting_highest_lpips | 9 | lab-coat | 8 | 36 | image_residual_sideinfo_full_frame | 4 | 29.875084 | 0.790654 | 0.232631 | 79895.000 |
| best_setting_highest_lpips | 10 | car-shadow | 8 | 35 | image_residual_sideinfo_full_frame | 4 | 31.612389 | 0.877647 | 0.215370 | 67713.000 |
| best_setting_highest_lpips | 11 | dogs-jump | 4 | 1 | image_residual_sideinfo_full_frame | 4 | 31.199727 | 0.847785 | 0.213204 | 48692.000 |
| best_setting_highest_lpips | 12 | paragliding-launch | 8 | 71 | image_residual_sideinfo_full_frame | 4 | 31.146980 | 0.765052 | 0.205574 | 73987.000 |
| best_setting_lowest_psnr | 1 | scooter-black | 8 | 10 | image_residual_sideinfo_full_frame | 4 | 28.365447 | 0.818228 | 0.246502 | 86793.000 |
| best_setting_lowest_psnr | 2 | horsejump-high | 8 | 20 | image_residual_sideinfo_full_frame | 4 | 29.174371 | 0.753123 | 0.238629 | 83995.000 |
| best_setting_lowest_psnr | 3 | car-roundabout | 4 | 29 | image_residual_sideinfo_full_frame | 4 | 29.309155 | 0.841775 | 0.187474 | 82701.000 |
| best_setting_lowest_psnr | 4 | dance-twirl | 8 | 68 | image_residual_sideinfo_full_frame | 4 | 29.479409 | 0.855716 | 0.192004 | 100643.000 |
| best_setting_lowest_psnr | 5 | car-roundabout | 8 | 15 | image_residual_sideinfo_full_frame | 4 | 29.547239 | 0.844081 | 0.193884 | 75899.000 |
| best_setting_lowest_psnr | 6 | drift-straight | 8 | 9 | image_residual_sideinfo_full_frame | 4 | 29.701196 | 0.855842 | 0.188776 | 94605.000 |
| best_setting_lowest_psnr | 7 | scooter-black | 4 | 27 | image_residual_sideinfo_full_frame | 4 | 29.819108 | 0.876200 | 0.183178 | 76174.000 |
| best_setting_lowest_psnr | 8 | lab-coat | 8 | 36 | image_residual_sideinfo_full_frame | 4 | 29.875084 | 0.790654 | 0.232631 | 79895.000 |
| best_setting_lowest_psnr | 9 | bmx-trees | 8 | 35 | image_residual_sideinfo_full_frame | 4 | 30.137169 | 0.861841 | 0.131860 | 95739.000 |
| best_setting_lowest_psnr | 10 | breakdance | 8 | 6 | image_residual_sideinfo_full_frame | 4 | 30.186875 | 0.903967 | 0.097237 | 77953.000 |
| best_setting_lowest_psnr | 11 | loading | 8 | 13 | image_residual_sideinfo_full_frame | 4 | 30.305513 | 0.866105 | 0.204309 | 79251.000 |
| best_setting_lowest_psnr | 12 | breakdance | 4 | 25 | image_residual_sideinfo_full_frame | 4 | 30.309072 | 0.899717 | 0.109979 | 82411.000 |
| best_setting_lowest_ssim | 1 | horsejump-high | 8 | 20 | image_residual_sideinfo_full_frame | 4 | 29.174371 | 0.753123 | 0.238629 | 83995.000 |
| best_setting_lowest_ssim | 2 | paragliding-launch | 8 | 71 | image_residual_sideinfo_full_frame | 4 | 31.146980 | 0.765052 | 0.205574 | 73987.000 |
| best_setting_lowest_ssim | 3 | shooting | 8 | 9 | image_residual_sideinfo_full_frame | 4 | 31.498265 | 0.786920 | 0.322325 | 75615.000 |
| best_setting_lowest_ssim | 4 | shooting | 4 | 7 | image_residual_sideinfo_full_frame | 4 | 31.042856 | 0.787061 | 0.362374 | 63795.000 |
| best_setting_lowest_ssim | 5 | lab-coat | 8 | 36 | image_residual_sideinfo_full_frame | 4 | 29.875084 | 0.790654 | 0.232631 | 79895.000 |
| best_setting_lowest_ssim | 6 | india | 8 | 30 | image_residual_sideinfo_full_frame | 4 | 31.521074 | 0.802642 | 0.275832 | 69400.000 |
| best_setting_lowest_ssim | 7 | motocross-jump | 4 | 23 | image_residual_sideinfo_full_frame | 4 | 32.141099 | 0.803875 | 0.299190 | 81415.000 |
| best_setting_lowest_ssim | 8 | motocross-jump | 8 | 22 | image_residual_sideinfo_full_frame | 4 | 31.759368 | 0.807489 | 0.272686 | 79279.000 |
| best_setting_lowest_ssim | 9 | kite-surf | 8 | 2 | image_residual_sideinfo_full_frame | 4 | 32.284269 | 0.811874 | 0.183562 | 61938.000 |
| best_setting_lowest_ssim | 10 | scooter-black | 8 | 10 | image_residual_sideinfo_full_frame | 4 | 28.365447 | 0.818228 | 0.246502 | 86793.000 |
| best_setting_lowest_ssim | 11 | india | 4 | 73 | image_residual_sideinfo_full_frame | 4 | 31.225884 | 0.827652 | 0.204149 | 81716.000 |
| best_setting_lowest_ssim | 12 | car-shadow | 4 | 1 | image_residual_sideinfo_full_frame | 4 | 32.104922 | 0.827944 | 0.204145 | 75556.000 |

## Decision

- `image_residual_sideinfo_full_frame` is an upper-bound diagnostic, not the final GS-feature method.
- A successful high-rate setting proves that the original StreamSplat base can be made visually and numerically strong if enough rate-counted auxiliary information is available.
- If dense-anchor shape matching fails, the next final-method path should be a StreamSplat-guided adapter or a residual codec defined on the original StreamSplat target-time Gaussian set, not direct residuals to Stage61 dense anchors.

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage155_streamsplat_base_sideinfo_upper_bound/stage155_streamsplat_base_sideinfo_rows.csv`
- summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage155_streamsplat_base_sideinfo_upper_bound/stage155_streamsplat_base_sideinfo_summary.csv`
- diagnostics CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage155_streamsplat_base_sideinfo_upper_bound/stage155_streamsplat_base_gaussian_diagnostics.csv`
- badcases CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage155_streamsplat_base_sideinfo_upper_bound/stage155_streamsplat_base_sideinfo_badcases.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage155_streamsplat_base_sideinfo_upper_bound/stage155_streamsplat_base_sideinfo_summary.json`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage155_streamsplat_base_sideinfo_upper_bound/stage155_streamsplat_base_sideinfo_upper_bound_package.json`
