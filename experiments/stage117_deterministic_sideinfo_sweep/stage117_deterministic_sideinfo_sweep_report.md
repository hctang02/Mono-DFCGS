# Stage117 Deterministic Side-Info Sweep

## Scope

- Sweeps deterministic-index value-only payload size over keep fraction and side bits.
- The entropy reference is Stage116 / Stage96 measured q6 top10 entropy-coded index+value side-info.
- Non-q6/top10 rows are cross-setting rate-only comparisons; rendered quality is unknown.
- Every deterministic transmitted byte is counted: header, metadata, and q residual values. Indices are not transmitted.

## Geometry

- gaussian count: `36860`
- attr dim: `13`
- Stage115 reference keep fraction: `0.1`
- Stage115 reference side bits: `6`

## Setting Summary

| keep | bits | keep count | det bytes | det MiB | saved index bytes | groups below Stage96 entropy | max det/entropy | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.025 | 2 | 922 | 3067 | 0.002925 | 1844 | 6/6 | 0.102679 | quality unknown |
| 0.025 | 3 | 922 | 4565 | 0.004354 | 1844 | 6/6 | 0.152830 | quality unknown |
| 0.025 | 4 | 922 | 6063 | 0.005782 | 1844 | 6/6 | 0.202981 | quality unknown |
| 0.025 | 5 | 922 | 7562 | 0.007212 | 1844 | 6/6 | 0.253165 | quality unknown |
| 0.025 | 6 | 922 | 9060 | 0.008640 | 1844 | 6/6 | 0.303316 | quality unknown |
| 0.025 | 8 | 922 | 12056 | 0.011497 | 1844 | 6/6 | 0.403618 | quality unknown |
| 0.05 | 2 | 1843 | 6060 | 0.005779 | 3686 | 6/6 | 0.202880 | quality unknown |
| 0.05 | 3 | 1843 | 9055 | 0.008636 | 3686 | 6/6 | 0.303149 | quality unknown |
| 0.05 | 4 | 1843 | 12050 | 0.011492 | 3686 | 6/6 | 0.403417 | quality unknown |
| 0.05 | 5 | 1843 | 15045 | 0.014348 | 3686 | 6/6 | 0.503685 | quality unknown |
| 0.05 | 6 | 1843 | 18040 | 0.017204 | 3686 | 6/6 | 0.603954 | quality unknown |
| 0.05 | 8 | 1843 | 24029 | 0.022916 | 3686 | 6/6 | 0.804457 | quality unknown |
| 0.1 | 2 | 3686 | 12050 | 0.011492 | 7372 | 6/6 | 0.403417 | quality unknown |
| 0.1 | 3 | 3686 | 18040 | 0.017204 | 7372 | 6/6 | 0.603954 | quality unknown |
| 0.1 | 4 | 3686 | 24029 | 0.022916 | 7372 | 6/6 | 0.804457 | quality unknown |
| 0.1 | 5 | 3686 | 30019 | 0.028628 | 7372 | 5/6 | 1.004994 | quality unknown |
| 0.1 | 6 | 3686 | 36009 | 0.034341 | 7372 | 0/6 | 1.205531 | Stage115 setting |
| 0.1 | 8 | 3686 | 47988 | 0.045765 | 7372 | 0/6 | 1.606571 | quality unknown |
| 0.15 | 2 | 5529 | 18040 | 0.017204 | 11058 | 6/6 | 0.603954 | quality unknown |
| 0.15 | 3 | 5529 | 27024 | 0.025772 | 11058 | 6/6 | 0.904726 | quality unknown |
| 0.15 | 4 | 5529 | 36009 | 0.034341 | 11058 | 0/6 | 1.205531 | quality unknown |
| 0.15 | 5 | 5529 | 44994 | 0.042910 | 11058 | 0/6 | 1.506336 | quality unknown |
| 0.15 | 6 | 5529 | 53978 | 0.051477 | 11058 | 0/6 | 1.807108 | quality unknown |
| 0.15 | 8 | 5529 | 71947 | 0.068614 | 11058 | 0/6 | 2.408684 | quality unknown |
| 0.2 | 2 | 7372 | 24029 | 0.022916 | 14744 | 6/6 | 0.804457 | quality unknown |
| 0.2 | 3 | 7372 | 36009 | 0.034341 | 14744 | 0/6 | 1.205531 | quality unknown |
| 0.2 | 4 | 7372 | 47988 | 0.045765 | 14744 | 0/6 | 1.606571 | quality unknown |
| 0.2 | 5 | 7372 | 59968 | 0.057190 | 14744 | 0/6 | 2.007644 | quality unknown |
| 0.2 | 6 | 7372 | 71947 | 0.068614 | 14744 | 0/6 | 2.408684 | quality unknown |
| 0.2 | 8 | 7372 | 95906 | 0.091463 | 14744 | 0/6 | 3.210798 | quality unknown |

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage117_deterministic_sideinfo_sweep/stage117_deterministic_sideinfo_sweep_rows.csv`
- summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage117_deterministic_sideinfo_sweep/stage117_deterministic_sideinfo_sweep_setting_summary.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage117_deterministic_sideinfo_sweep/stage117_deterministic_sideinfo_sweep_summary.json`

## Conclusion

- The Stage115 q6 top10 deterministic payload is `36009 bytes`, matching the derived geometry.
- Lower keep fractions or lower side bits can beat the Stage96 q6 top10 entropy reference in rate, but their rendered quality is not validated here.
- Stage118 should only package RD points after rendered quality exists for the selected deterministic settings.
