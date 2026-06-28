# Stage119 Actual Compressed Deterministic Sweep

## Configuration

- task count: `12`
- keep fractions: `[0.025, 0.05, 0.1, 0.15, 0.2]`
- side bits: `[2, 3, 4, 5, 6, 8]`
- zlib level: `9`
- no rendering, no training, no checkpoint, no heavy tensor output
- Stage96 q6/top10 entropy side-info is used as rate reference only

## Setting Summary

| keep | bits | mean comp bytes | mean comp MiB | comp/raw | groups below Stage96 entropy | max comp/entropy | max decode diff | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.025 | 2 | 1966.1027777777774 | 0.001875 | 0.641051 | 6/6 | 0.063053 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.025 | 3 | 3548.2527777777777 | 0.003384 | 0.777273 | 6/6 | 0.110890 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.025 | 4 | 4350.077777777778 | 0.004149 | 0.717479 | 6/6 | 0.135883 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.025 | 5 | 6580.222222222223 | 0.006275 | 0.870170 | 6/6 | 0.203634 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.025 | 6 | 7768.891666666666 | 0.007409 | 0.857494 | 6/6 | 0.239151 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.025 | 8 | 9557.019444444444 | 0.009114 | 0.792719 | 6/6 | 0.295144 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.05 | 2 | 3507.836111111111 | 0.003345 | 0.578851 | 6/6 | 0.112750 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.05 | 3 | 6536.563888888889 | 0.006234 | 0.721873 | 6/6 | 0.205146 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.05 | 4 | 8065.533333333333 | 0.007692 | 0.669339 | 6/6 | 0.253692 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.05 | 5 | 12683.53888888889 | 0.012096 | 0.843040 | 6/6 | 0.394823 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.05 | 6 | 15040.72222222222 | 0.014344 | 0.833743 | 6/6 | 0.466709 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.05 | 8 | 18539.280555555557 | 0.017680 | 0.771538 | 6/6 | 0.572385 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.1 | 2 | 6340.708333333333 | 0.006047 | 0.526200 | 6/6 | 0.204744 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.1 | 3 | 12120.455555555556 | 0.011559 | 0.671866 | 6/6 | 0.387479 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.1 | 4 | 14982.574999999999 | 0.014288 | 0.623521 | 6/6 | 0.479591 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.1 | 5 | 24537.56111111111 | 0.023401 | 0.817401 | 6/6 | 0.773032 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.1 | 6 | 29235.55 | 0.027881 | 0.811896 | 6/6 | 0.913960 | 0.0 | stage115_q6_top10_setting |
| 0.1 | 8 | 35831.40277777778 | 0.034171 | 0.746674 | 0/6 | 1.107651 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.15 | 2 | 9129.680555555555 | 0.008707 | 0.505996 | 6/6 | 0.296916 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.15 | 3 | 17546.752777777776 | 0.016734 | 0.649182 | 6/6 | 0.565036 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.15 | 4 | 21636.180555555555 | 0.020634 | 0.600755 | 6/6 | 0.695573 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.15 | 5 | 36014.88333333333 | 0.034346 | 0.800295 | 0/6 | 1.141002 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.15 | 6 | 43113.72222222222 | 0.041116 | 0.798580 | 0/6 | 1.353729 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.15 | 8 | 52942.79444444445 | 0.050490 | 0.735725 | 0/6 | 1.637692 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.2 | 2 | 11817.727777777778 | 0.011270 | 0.491729 | 6/6 | 0.382328 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.2 | 3 | 22663.67222222222 | 0.021614 | 0.629302 | 6/6 | 0.730766 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.2 | 4 | 28043.888888888887 | 0.026745 | 0.584309 | 6/6 | 0.896310 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.2 | 5 | 47111.169444444444 | 0.044929 | 0.785500 | 0/6 | 1.495874 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.2 | 6 | 56646.330555555556 | 0.054022 | 0.787225 | 0/6 | 1.780546 | 0.0 | cross_setting_rate_only_quality_unknown |
| 0.2 | 8 | 69727.56666666667 | 0.066497 | 0.726942 | 0/6 | 2.161015 | 0.0 | cross_setting_rate_only_quality_unknown |

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage119_actual_compressed_deterministic_sweep/stage119_actual_compressed_deterministic_sweep_rows.csv`
- group summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage119_actual_compressed_deterministic_sweep/stage119_actual_compressed_deterministic_sweep_group_summary.csv`
- setting summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage119_actual_compressed_deterministic_sweep/stage119_actual_compressed_deterministic_sweep_setting_summary.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage119_actual_compressed_deterministic_sweep/stage119_actual_compressed_deterministic_sweep_summary.json`

## Notes

- q6/top10 is the Stage115/118 setting and is decode-equivalent to previous deterministic q6/top10 payloads.
- Non-q6/top10 settings are rate-only until rendered validation is run.
- Residual values remain teacher-derived from dense target anchors.
