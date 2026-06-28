# Stage116 Deterministic vs Entropy Side-Info Accounting

Date: 2026-06-29

## Goal

Compare Stage115 deterministic-index value-only residual side-info against existing entropy-coded index+value residual side-info while counting every transmitted side-info byte.

## Implementation

Added:

```text
scripts/run_stage116_deterministic_vs_entropy_sideinfo_accounting.py
```

The script reads existing summaries only and produces accounting rows/points. It does not train, render, or write anchors/checkpoints/heavy tensors.

## Inputs

```text
experiments/stage93_residual_sideinfo_entropy_codec_smoke/stage93_residual_sideinfo_entropy_codec_summary.csv
experiments/stage96_broader_entropy_residual_sideinfo_rd_package/stage96_broader_entropy_residual_sideinfo_rd_rows.csv
experiments/stage115_deterministic_index_residual_codec_smoke/stage115_deterministic_index_residual_codec_summary.csv
experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_rate_table.csv
```

## Run

GPU check was performed before execution. GPU2 was idle and used with `CUDA_VISIBLE_DEVICES=2`, although the script is CPU-only accounting.

Syntax check and run:

```text
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage116_deterministic_vs_entropy_sideinfo_accounting.py
CUDA_VISIBLE_DEVICES=2 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage116_deterministic_vs_entropy_sideinfo_accounting.py
```

## Outputs

```text
experiments/stage116_deterministic_vs_entropy_sideinfo_accounting/stage116_deterministic_vs_entropy_sideinfo_accounting_rows.csv
experiments/stage116_deterministic_vs_entropy_sideinfo_accounting/stage116_deterministic_vs_entropy_sideinfo_accounting_points.csv
experiments/stage116_deterministic_vs_entropy_sideinfo_accounting/stage116_deterministic_vs_entropy_sideinfo_accounting_summary.json
experiments/stage116_deterministic_vs_entropy_sideinfo_accounting/stage116_deterministic_vs_entropy_sideinfo_accounting_report.md
```

## Configuration

| item | value |
|---|---:|
| row count | 12 |
| point count | 72 |
| deterministic payload | `36009 bytes` |
| deterministic MiB/intermediate | `0.034340858459472656` |
| deterministic quality status | `not_rendered_rate_only` |

## Stage96 Broader Comparison

| base | gap | entropy side MiB | deterministic side MiB | deterministic / entropy |
|---|---:|---:|---:|---:|
| linear | 4 | 0.028486092885335285 | 0.034340858459472656 | 1.2055306636015155 |
| linear | 8 | 0.02903069947895251 | 0.034340858459472656 | 1.182915295732714 |
| linear | 16 | 0.028838696687117867 | 0.034340858459472656 | 1.1907909303963997 |
| stage65_adapter | 4 | 0.033147705925835505 | 0.034340858459472656 | 1.0359950259093857 |
| stage65_adapter | 8 | 0.033867986578690376 | 0.034340858459472656 | 1.0139622082252686 |
| stage65_adapter | 16 | 0.03376620748768682 | 0.034340858459472656 | 1.017018522793695 |

## Deterministic Total Rates

All side-info bytes are counted.

| gap | direct MiB/frame | amortized MiB/frame |
|---:|---:|---:|
| 4 | 0.2162790792273858 | 0.20769386461251763 |
| 8 | 0.131966245212987 | 0.12767363790555292 |
| 16 | 0.08980982820578763 | 0.08766352455207059 |

## Conclusion

- Deterministic value-only payload removes selected-index bytes, saving `7372 bytes` vs fixed index+value payload.
- Stage96 broader entropy-coded index+value payload remains smaller than deterministic value-only payload for all listed groups.
- Deterministic endpoint-diff residual quality is not rendered in Stage116, so the package is rate accounting only.
- Stage117 should sweep q-bit and keep fraction before broader RD packaging.
