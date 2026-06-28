# Stage90 Broader q6 Residual Side-Info RD Package

## Scope

- Main rate comes from Stage78 q12 static-anchor rate table.
- Rendered quality comes from Stage89 60-task broader q6 residual side-info eval.
- Side-info is treated as transmitted information and included in total rate.
- This package uses a 60-task broader q6 top10 eval slice, not a full-video final RD benchmark.

## Rate Definitions

- `direct_total_mib_per_frame = q12_main_anchor_mib_per_frame + side_info_mib_per_intermediate_frame`.
- `amortized_total_mib_per_frame = q12_main_anchor_mib_per_frame + side_info_mib_per_intermediate_frame * ((gap - 1) / gap)`.
- The direct total is conservative for the per-intermediate-frame side-info eval; the amortized total is a uniform-gap full-video approximation.

## Low-Rate Operating Point: keep 0.1, q6 residual

| base | gap | main MiB/frame | side MiB/intermediate | direct total | amortized total | base PSNR | side PSNR | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 4 | 0.181938 | 0.041354 | 0.223292 | 0.212953 | 19.984796 | 23.403666 | 3.418870 |
| linear | 8 | 0.097625 | 0.041354 | 0.138979 | 0.133810 | 18.457590 | 21.513745 | 3.056155 |
| linear | 16 | 0.055469 | 0.041354 | 0.096823 | 0.094238 | 17.084215 | 20.344221 | 3.260006 |
| stage65_adapter | 4 | 0.181938 | 0.041354 | 0.223292 | 0.212953 | 20.041732 | 22.840999 | 2.799266 |
| stage65_adapter | 8 | 0.097625 | 0.041354 | 0.138979 | 0.133810 | 18.706547 | 21.398942 | 2.692395 |
| stage65_adapter | 16 | 0.055469 | 0.041354 | 0.096823 | 0.094238 | 17.328230 | 20.292005 | 2.963774 |

## Full RD Rows

| base | gap | keep | bits | direct total | amortized total | side PSNR | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear | 4 | 0.1 | 6 | 0.223292 | 0.212953 | 23.403666 | 3.418870 |
| linear | 8 | 0.1 | 6 | 0.138979 | 0.133810 | 21.513745 | 3.056155 |
| linear | 16 | 0.1 | 6 | 0.096823 | 0.094238 | 20.344221 | 3.260006 |
| stage65_adapter | 4 | 0.1 | 6 | 0.223292 | 0.212953 | 22.840999 | 2.799266 |
| stage65_adapter | 8 | 0.1 | 6 | 0.138979 | 0.133810 | 21.398942 | 2.692395 |
| stage65_adapter | 16 | 0.1 | 6 | 0.096823 | 0.094238 | 20.292005 | 2.963774 |

## Outputs

- rows CSV: `experiments/stage90_broader_q6_residual_sideinfo_rd_package/stage90_broader_q6_residual_sideinfo_rd_rows.csv`
- point CSV: `experiments/stage90_broader_q6_residual_sideinfo_rd_package/stage90_broader_q6_residual_sideinfo_rd_points.csv`
- direct RD plot: `experiments/stage90_broader_q6_residual_sideinfo_rd_package/stage90_broader_q6_residual_sideinfo_rd_direct.png`
- amortized RD plot: `experiments/stage90_broader_q6_residual_sideinfo_rd_package/stage90_broader_q6_residual_sideinfo_rd_amortized.png`

## Conclusion

- q6 top10 residual side-info adds `0.041354 MiB/intermediate-frame` and preserves multi-dB rendered PSNR gains in this package.
- This package only includes the requested side-info operating points; it is not a cross-bit sweep.
- The next step is a real bitstream/entropy-coded side-info implementation; this package does not yet claim final RD.
