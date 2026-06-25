# Stage 43 Selector Evidence Synthesis

| stage | evidence | deployable selector | actual bitstream | points | +all | mean Δall PSNR | mean Δmiddle PSNR | interpretation |
|---|---|---|---|---:|---:|---:|---:|---|
| 26 | leave-one-out anchor adapter vs linear interpolation | uniform only | estimated q8 rate | 16 | 16 | 0.079796 | 0.112505 | best current adapter generalization evidence; not a selector gain |
| 36 | dense anchor-attribute oracle selector vs uniform | no | yes, zlib q8 anchors | 12 | 12 | 0.197152 | 0.225436 | strong selector upper bound; uses non-deployable dense intermediate anchors |
| 39 | raw ridge predicted selector: length_only_ridge | yes | estimated q8 rate | 12 | 0 | -0.035285 | -0.038178 | negative selector result against uniform |
| 39 | raw ridge predicted selector: full_feature_ridge | yes | estimated q8 rate | 12 | 1 | -0.292395 | -0.364151 | negative selector result against uniform |
| 41 | normalized/rank predicted selector: length_sample_z_rank | yes | estimated q8 rate | 12 | 0 | -0.505437 | -0.591863 | negative selector result against uniform |
| 41 | normalized/rank predicted selector: full_sample_z_rank | yes | estimated q8 rate | 12 | 0 | -0.847949 | -0.994500 | negative selector result against uniform |
| 41 | normalized/rank predicted selector: full_sample_z_zlog | yes | estimated q8 rate | 12 | 0 | -0.913330 | -1.082226 | negative selector result against uniform |
| 42 | uniform-prior calibrated selector: full_sample_z_rank_prior_0p0 | yes | estimated q8 rate | 12 | 0 | -0.847949 | -0.994500 | calibration fallback; exact uniform points=0 |
| 42 | uniform-prior calibrated selector: full_sample_z_rank_prior_0p05 | yes | estimated q8 rate | 12 | 0 | -0.604156 | -0.700220 | calibration fallback; exact uniform points=0 |
| 42 | uniform-prior calibrated selector: full_sample_z_rank_prior_0p1 | yes | estimated q8 rate | 12 | 0 | -0.447431 | -0.512512 | calibration fallback; exact uniform points=0 |
| 42 | uniform-prior calibrated selector: full_sample_z_rank_prior_0p3 | yes | estimated q8 rate | 12 | 1 | -0.217569 | -0.248952 | calibration fallback; exact uniform points=0 |
| 42 | uniform-prior calibrated selector: full_sample_z_rank_prior_1p0 | yes | estimated q8 rate | 12 | 3 | -0.099327 | -0.109025 | calibration fallback; exact uniform points=6 |
| 42 | uniform-prior calibrated selector: full_sample_z_rank_prior_3p0 | yes | estimated q8 rate | 12 | 1 | -0.014865 | -0.014910 | calibration fallback; exact uniform points=9 |
| 42 | uniform-prior calibrated selector: full_sample_z_rank_prior_10p0 | yes | estimated q8 rate | 12 | 1 | -0.011239 | -0.010832 | calibration fallback; exact uniform points=10 |

## Takeaway

Current deployable predicted selectors do not beat uniform. Dense oracle/proxy selector remains a useful upper bound, while Stage26 leave-one-out adapter gains remain the strongest deployable codec evidence.
