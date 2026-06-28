# Stage88 Residual Side-Info RD Package

## Scope

- Main rate comes from Stage78 q12 static-anchor rate table.
- Rendered quality comes from Stage87 quantized residual side-info smoke.
- Side-info is treated as transmitted information and included in total rate.
- This is a 12-task smoke package, not a full-video RD benchmark.

## Rate Definitions

- `direct_total_mib_per_frame = q12_main_anchor_mib_per_frame + side_info_mib_per_intermediate_frame`.
- `amortized_total_mib_per_frame = q12_main_anchor_mib_per_frame + side_info_mib_per_intermediate_frame * ((gap - 1) / gap)`.
- The direct total is conservative for the Stage87 per-intermediate-frame side-info smoke; the amortized total is a uniform-gap full-video approximation.

## Low-Rate Operating Point: keep 0.1, q6 residual

| base | gap | main MiB/frame | side MiB/intermediate | direct total | amortized total | base PSNR | side PSNR | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 4 | 0.181938 | 0.041354 | 0.223292 | 0.212953 | 19.989797 | 23.326992 | 3.337196 |
| linear | 8 | 0.097625 | 0.041354 | 0.138979 | 0.133810 | 18.121118 | 20.945471 | 2.824353 |
| linear | 16 | 0.055469 | 0.041354 | 0.096823 | 0.094238 | 19.113015 | 23.589871 | 4.476856 |
| stage65_adapter | 4 | 0.181938 | 0.041354 | 0.223292 | 0.212953 | 20.482738 | 23.352584 | 2.869846 |
| stage65_adapter | 8 | 0.097625 | 0.041354 | 0.138979 | 0.133810 | 18.532358 | 20.943546 | 2.411188 |
| stage65_adapter | 16 | 0.055469 | 0.041354 | 0.096823 | 0.094238 | 19.075010 | 22.481911 | 3.406901 |

## Full RD Rows

| base | gap | keep | bits | direct total | amortized total | side PSNR | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear | 4 | 0.1 | 6 | 0.223292 | 0.212953 | 23.326992 | 3.337196 |
| linear | 4 | 0.1 | 8 | 0.234716 | 0.221522 | 23.331522 | 3.341725 |
| linear | 4 | 0.25 | 6 | 0.285259 | 0.259429 | 26.035583 | 6.045786 |
| linear | 4 | 0.25 | 8 | 0.313824 | 0.280852 | 26.067254 | 6.077458 |
| linear | 8 | 0.1 | 6 | 0.138979 | 0.133810 | 20.945471 | 2.824353 |
| linear | 8 | 0.1 | 8 | 0.150404 | 0.143806 | 20.951419 | 2.830302 |
| linear | 8 | 0.25 | 6 | 0.200946 | 0.188031 | 23.471187 | 5.350069 |
| linear | 8 | 0.25 | 8 | 0.229511 | 0.213025 | 23.483477 | 5.362359 |
| linear | 16 | 0.1 | 6 | 0.096823 | 0.094238 | 23.589871 | 4.476856 |
| linear | 16 | 0.1 | 8 | 0.108247 | 0.104949 | 23.595824 | 4.482809 |
| linear | 16 | 0.25 | 6 | 0.158790 | 0.152332 | 25.948424 | 6.835409 |
| linear | 16 | 0.25 | 8 | 0.187354 | 0.179112 | 25.959284 | 6.846269 |
| stage65_adapter | 4 | 0.1 | 6 | 0.223292 | 0.212953 | 23.352584 | 2.869846 |
| stage65_adapter | 4 | 0.1 | 8 | 0.234716 | 0.221522 | 23.359259 | 2.876520 |
| stage65_adapter | 4 | 0.25 | 6 | 0.285259 | 0.259429 | 25.295132 | 4.812394 |
| stage65_adapter | 4 | 0.25 | 8 | 0.313824 | 0.280852 | 25.309307 | 4.826569 |
| stage65_adapter | 8 | 0.1 | 6 | 0.138979 | 0.133810 | 20.943546 | 2.411188 |
| stage65_adapter | 8 | 0.1 | 8 | 0.150404 | 0.143806 | 20.944235 | 2.411876 |
| stage65_adapter | 8 | 0.25 | 6 | 0.200946 | 0.188031 | 22.856351 | 4.323993 |
| stage65_adapter | 8 | 0.25 | 8 | 0.229511 | 0.213025 | 22.863307 | 4.330948 |
| stage65_adapter | 16 | 0.1 | 6 | 0.096823 | 0.094238 | 22.481911 | 3.406901 |
| stage65_adapter | 16 | 0.1 | 8 | 0.108247 | 0.104949 | 22.487897 | 3.412887 |
| stage65_adapter | 16 | 0.25 | 6 | 0.158790 | 0.152332 | 24.477412 | 5.402402 |
| stage65_adapter | 16 | 0.25 | 8 | 0.187354 | 0.179112 | 24.495531 | 5.420521 |

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage88_residual_sideinfo_rd_package/stage88_residual_sideinfo_rd_rows.csv`
- point CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage88_residual_sideinfo_rd_package/stage88_residual_sideinfo_rd_points.csv`
- direct RD plot: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage88_residual_sideinfo_rd_package/stage88_residual_sideinfo_rd_direct.png`
- amortized RD plot: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage88_residual_sideinfo_rd_package/stage88_residual_sideinfo_rd_amortized.png`

## Conclusion

- q6 top10 residual side-info is the most attractive low-rate point in this smoke: it adds roughly `0.041354 MiB/intermediate-frame` and preserves multi-dB rendered PSNR gains.
- q8 side-info gives nearly identical PSNR at a higher rate; q6 should be the default follow-up operating point unless larger eval contradicts it.
- The next step is a larger eval or a real bitstream/entropy-coded side-info implementation; this package does not yet claim final RD.
