# Stage88 Residual Side-Info RD Package

Date: 2026-06-28

## Goal

Convert Stage87 quantized residual side-info smoke into a small RD accounting package with transmitted side-info included in total rate.

## Inputs

```text
experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_rate_table.csv
experiments/stage87_quantized_residual_sideinfo_smoke/stage87_quantized_residual_sideinfo_summary.csv
```

## Implementation

Added:

```text
scripts/run_stage88_residual_sideinfo_rd_package.py
```

Rate definitions:

```text
direct_total_mib_per_frame = q12_main_anchor_mib_per_frame + side_info_mib_per_intermediate_frame
amortized_total_mib_per_frame = q12_main_anchor_mib_per_frame + side_info_mib_per_intermediate_frame * ((gap - 1) / gap)
```

The direct total follows the immediate Stage88 accounting requirement. The amortized total is a uniform-gap full-video approximation because Stage87 side-info is measured per intermediate frame.

## Run

GPU check was performed before execution. GPU0 was busy, but the script is a CPU summary job and did not use GPU.

## Outputs

```text
experiments/stage88_residual_sideinfo_rd_package/stage88_residual_sideinfo_rd_rows.csv
experiments/stage88_residual_sideinfo_rd_package/stage88_residual_sideinfo_rd_points.csv
experiments/stage88_residual_sideinfo_rd_package/stage88_residual_sideinfo_rd_summary.json
experiments/stage88_residual_sideinfo_rd_package/stage88_residual_sideinfo_rd_report.md
experiments/stage88_residual_sideinfo_rd_package/stage88_residual_sideinfo_rd_direct.png
experiments/stage88_residual_sideinfo_rd_package/stage88_residual_sideinfo_rd_amortized.png
```

Rows: `24` RD rows and `60` point rows.

## Low-Rate Operating Point

q6 top10% residual side-info:

| base | gap | q12 main MiB/frame | side MiB/intermediate | direct total | amortized total | side PSNR | delta PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear | 4 | 0.18193822076791313 | 0.041353702545166016 | 0.22329192331307915 | 0.21295349767678765 | 23.326992309396783 | 3.3371957812025705 |
| linear | 8 | 0.09762538675351436 | 0.041353702545166016 | 0.13897908929868036 | 0.1338098764805346 | 20.94547122800524 | 2.8243534745057453 |
| linear | 16 | 0.055468969746314975 | 0.041353702545166016 | 0.09682267229148099 | 0.09423806588240811 | 23.589870525188754 | 4.476855554636258 |
| stage65_adapter | 4 | 0.18193822076791313 | 0.041353702545166016 | 0.22329192331307915 | 0.21295349767678765 | 23.35258386164915 | 2.869845655272099 |
| stage65_adapter | 8 | 0.09762538675351436 | 0.041353702545166016 | 0.13897908929868036 | 0.1338098764805346 | 20.943546028225864 | 2.411187534241218 |
| stage65_adapter | 16 | 0.055468969746314975 | 0.041353702545166016 | 0.09682267229148099 | 0.09423806588240811 | 22.481910804608162 | 3.406900716022574 |

## Conclusion

- q6 top10% is the preferred low-rate residual side-info operating point from this smoke.
- q8 top10% gives almost identical PSNR with higher side-info rate.
- Stage88 is an accounting package over the Stage87 12-task smoke, not final full-video RD.
