# Stage92 Residual Side-Info Entropy Preflight

Date: 2026-06-28

## Goal

Estimate lossless compression potential for the Stage91 fixed q6 top10 residual side-info payload.

## Scope

- Tasks: `60` q12 eval tasks.
- Rows: `120` method payload rows.
- Gaps: `4`, `8`, `16`.
- Base methods: linear and Stage65 adapter.
- Keep fraction: `0.1`.
- Side bits: `6`.
- No rendering is run in this stage.

## Implementation

Added:

```text
scripts/run_stage92_residual_sideinfo_entropy_preflight.py
```

Compression candidates:

- `raw_fixed`
- `zlib_whole`
- `component_zlib`
- `residual_zlib`
- `delta_index_zlib`

## Run

GPU check was performed before execution. GPU1 was idle, so the run used:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage92_residual_sideinfo_entropy_preflight.py
```

## Outputs

```text
experiments/stage92_residual_sideinfo_entropy_preflight/stage92_residual_sideinfo_entropy_rows.csv
experiments/stage92_residual_sideinfo_entropy_preflight/stage92_residual_sideinfo_entropy_summary.csv
experiments/stage92_residual_sideinfo_entropy_preflight/stage92_residual_sideinfo_entropy_candidates.csv
experiments/stage92_residual_sideinfo_entropy_preflight/stage92_residual_sideinfo_entropy_summary.json
experiments/stage92_residual_sideinfo_entropy_preflight/stage92_residual_sideinfo_entropy_report.md
```

## Candidate Summary

| candidate | mean bytes | mean MiB/intermediate | ratio vs raw | savings bytes |
|---|---:|---:|---:|---:|
| delta_index_zlib | 32623.583333333332 | 0.031112273534138996 | 0.7520246959114205 | 10757.416666666668 |
| component_zlib | 37162.98333333333 | 0.035441382726033525 | 0.8566649762184673 | 6218.01666666667 |
| residual_zlib | 37422.4 | 0.03568878173828125 | 0.8626449367234504 | 5958.5999999999985 |
| zlib_whole | 37426.28333333333 | 0.035692485173543294 | 0.8627344536394581 | 5954.716666666667 |
| raw_fixed | 43381.0 | 0.04137134552001953 | 1.0 | 0.0 |

## Conclusion

- `delta_index_zlib` is the best candidate for all 120 payload rows.
- Mean side-info rate can drop from `0.04137134552001953` to `0.031112273534138996 MiB/intermediate-frame` in this preflight.
- Linear residual payloads are more compressible than Stage65 adapter residual payloads.
- Stage93 should implement a real decode-capable codec v2 based on sorted index deltas and zlib-compressed components.
