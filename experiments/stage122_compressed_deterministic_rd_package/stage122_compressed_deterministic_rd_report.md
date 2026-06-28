# Stage122 Compressed Deterministic RD Package

## Scope

- Quality and deterministic side-info rates come from Stage121 broader rendered validation.
- Stage96 entropy-coded q6/top10 index+value side-info is included as reference.
- Every transmitted side-info byte is counted in direct and amortized total rates.
- Residual values are teacher-derived; this is not residual value prediction.

## Setting Summary

| role | setting | keep | bits | side bytes | direct | amortized | PSNR | delta base | delta q6 | dRate entropy | dPSNR entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | q4_top20 | 0.2 | 4 | 28320.791667 | 0.133768 | 0.130054 | 20.689271 | 2.205251 | 0.922302 | -0.004194 | -0.830532 |
| low_rate | q4_top10 | 0.1 | 4 | 15190.475000 | 0.121246 | 0.119247 | 19.738488 | 1.254468 | -0.028480 | -0.016717 | -1.781315 |
| near_anchor | q5_top10 | 0.1 | 5 | 24809.950000 | 0.130420 | 0.127150 | 19.761048 | 1.277028 | -0.005921 | -0.007543 | -1.758755 |
| anchor | q6_top10 | 0.1 | 6 | 29442.208333 | 0.134838 | 0.130956 | 19.766969 | 1.282949 | 0.000000 | -0.003125 | -1.752834 |

## Package Recommendation

- Primary candidate: `q4_top20`.
- Low-rate candidate: `q4_top10`.
- Near-anchor candidate: `q5_top10`.
- Anchor candidate: `q6_top10`.

## Outputs

- RD rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage122_compressed_deterministic_rd_package/stage122_compressed_deterministic_rd_rows.csv`
- RD points CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage122_compressed_deterministic_rd_package/stage122_compressed_deterministic_rd_points.csv`
- setting summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage122_compressed_deterministic_rd_package/stage122_compressed_deterministic_rd_setting_summary.csv`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage122_compressed_deterministic_rd_package/stage122_compressed_deterministic_rd_package.json`
