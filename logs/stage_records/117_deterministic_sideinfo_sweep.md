# Stage117 Deterministic Side-Info Sweep

Date: 2026-06-29

## Goal

Sweep deterministic-index value-only residual side-info payload size over keep fraction and side bits, using measured Stage96 q6/top10 entropy-coded index+value side-info as a rate reference.

## Implementation

Stage92 was inspected first and found to cover only q6/top10, not a multi q-bit / keep-fraction sweep. Stage117 therefore derives deterministic payload geometry from Stage115 and computes payload bytes by formula.

Added:

```text
scripts/run_stage117_deterministic_sideinfo_sweep.py
```

## Inputs

```text
experiments/stage115_deterministic_index_residual_codec_smoke/stage115_deterministic_index_residual_codec_summary.csv
experiments/stage116_deterministic_vs_entropy_sideinfo_accounting/stage116_deterministic_vs_entropy_sideinfo_accounting_rows.csv
```

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`, although the script is CPU-only accounting.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage117_deterministic_sideinfo_sweep.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage117_deterministic_sideinfo_sweep.py
```

## Outputs

```text
experiments/stage117_deterministic_sideinfo_sweep/stage117_deterministic_sideinfo_sweep_rows.csv
experiments/stage117_deterministic_sideinfo_sweep/stage117_deterministic_sideinfo_sweep_setting_summary.csv
experiments/stage117_deterministic_sideinfo_sweep/stage117_deterministic_sideinfo_sweep_summary.json
experiments/stage117_deterministic_sideinfo_sweep/stage117_deterministic_sideinfo_sweep_report.md
```

## Configuration

| item | value |
|---|---:|
| gaussian count | 36860 |
| attr dim | 13 |
| keep fractions | `[0.025, 0.05, 0.1, 0.15, 0.2]` |
| side bits | `[2, 3, 4, 5, 6, 8]` |
| row count | 180 |
| setting count | 30 |
| entropy reference | Stage96 q6/top10 measured entropy-coded index+value side-info |

## Key Results

| keep | bits | deterministic bytes | deterministic MiB | groups below Stage96 entropy | max det/entropy | note |
|---:|---:|---:|---:|---:|---:|---|
| 0.1 | 6 | 36009 | 0.034340858459472656 | 0/6 | 1.2055306636015155 | Stage115 setting |
| 0.1 | 5 | 30019 | 0.02862834930419922 | 5/6 | 1.0049938901567357 | quality unknown |
| 0.1 | 4 | 24029 | 0.02291584014892578 | 6/6 | 0.8044571167119557 | quality unknown |
| 0.05 | 6 | 18040 | 0.01720428466796875 | 6/6 | 0.6039538218604055 | quality unknown |
| 0.025 | 6 | 9060 | 0.008640289306640625 | 6/6 | 0.3033160546593832 | quality unknown |

## Conclusion

- The q6/top10 deterministic payload from Stage115 is reproduced exactly: `36009 bytes`.
- Lower side bits or keep fractions can beat the measured Stage96 q6/top10 entropy-coded reference in rate.
- These are cross-setting rate-only comparisons, so rendered quality is unknown.
- Stage118 should render/validate a shortlist such as q5/top10, q4/top10, and q6/top5 before any broader RD package.
