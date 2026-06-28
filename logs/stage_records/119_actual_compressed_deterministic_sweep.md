# Stage119 Actual Compressed Deterministic Sweep

Date: 2026-06-29

## Goal

Measure actual compressed deterministic value-only payload sizes over q-bit and keep-fraction settings using real task residual values.

## Implementation

Added:

```text
scripts/run_stage119_actual_compressed_deterministic_sweep.py
```

The script uses the Stage118 compressed deterministic codec helpers and verifies compressed decode against raw deterministic decode for every row.

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage119_actual_compressed_deterministic_sweep.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage119_actual_compressed_deterministic_sweep.py
```

## Outputs

```text
experiments/stage119_actual_compressed_deterministic_sweep/stage119_actual_compressed_deterministic_sweep_rows.csv
experiments/stage119_actual_compressed_deterministic_sweep/stage119_actual_compressed_deterministic_sweep_group_summary.csv
experiments/stage119_actual_compressed_deterministic_sweep/stage119_actual_compressed_deterministic_sweep_setting_summary.csv
experiments/stage119_actual_compressed_deterministic_sweep/stage119_actual_compressed_deterministic_sweep_summary.json
experiments/stage119_actual_compressed_deterministic_sweep/stage119_actual_compressed_deterministic_sweep_report.md
```

## Configuration

| item | value |
|---|---:|
| selector policy | `strict_safe_endpoint_selector_v1` |
| selected candidate | `endpoint_diff_baseline` |
| task count | 12 |
| keep fractions | `[0.025, 0.05, 0.1, 0.15, 0.2]` |
| side bits | `[2, 3, 4, 5, 6, 8]` |
| zlib level | 9 |
| row count | 720 |
| group count | 180 |
| setting count | 30 |
| max decode diff | 0.0 |

## Shortlist Results

| keep | bits | mean compressed bytes | mean MiB/intermediate | groups below Stage96 entropy | max comp/entropy | note |
|---:|---:|---:|---:|---:|---:|---|
| 0.1 | 6 | 29235.55 | 0.02788119316101074 | 6/6 | 0.9139598369766582 | q6/top10 quality anchor |
| 0.1 | 5 | 24537.56111111111 | 0.023400841818915472 | 6/6 | 0.7730317895516859 | q5/top10 candidate |
| 0.1 | 4 | 14982.574999999999 | 0.01428849697113037 | 6/6 | 0.4795912560207484 | q4/top10 candidate |
| 0.05 | 6 | 15040.72222222222 | 0.01434395048353407 | 6/6 | 0.4667089499820716 | q6/top5 candidate |
| 0.2 | 4 | 28043.888888888887 | 0.02674473656548394 | 6/6 | 0.8963100407558355 | q4/top20 candidate |

## Conclusion

- Actual compressed deterministic payloads are lower than the raw formula-only estimates from Stage117.
- The shortlist settings are all below the Stage96 q6/top10 entropy-coded index+value reference in rate.
- Non-q6/top10 settings are still rate-only; rendered quality must be measured in Stage120.
