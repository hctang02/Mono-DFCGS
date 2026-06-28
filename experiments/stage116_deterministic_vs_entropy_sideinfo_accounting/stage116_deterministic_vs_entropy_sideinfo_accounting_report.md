# Stage116 Deterministic vs Entropy Side-Info Accounting

## Scope

- Stage115 deterministic-index value-only payload is compared against entropy-coded index+value residual side-info.
- Stage93 is the matched 12-task entropy smoke source; Stage96 is the broader 60-task entropy RD reference.
- Deterministic rows are rate-only here because endpoint-diff deterministic residual indices were not rendered in Stage115.
- Every transmitted side-info byte is counted: headers, metadata, residual values, and transmitted indices when present.

## Rows

| source | base | gap | entropy side | deterministic side | det - entropy | det/entropy | det direct | det amortized | quality |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stage93_entropy_smoke | linear | 4 | 0.028385 | 0.034341 | 0.005956 | 1.209831 | 0.216279 | 0.207694 | not_rendered_rate_only |
| stage93_entropy_smoke | linear | 8 | 0.029388 | 0.034341 | 0.004953 | 1.168539 | 0.131966 | 0.127674 | not_rendered_rate_only |
| stage93_entropy_smoke | linear | 16 | 0.028190 | 0.034341 | 0.006151 | 1.218187 | 0.089810 | 0.087664 | not_rendered_rate_only |
| stage93_entropy_smoke | stage65_adapter | 4 | 0.033403 | 0.034341 | 0.000937 | 1.028065 | 0.216279 | 0.207694 | not_rendered_rate_only |
| stage93_entropy_smoke | stage65_adapter | 8 | 0.034437 | 0.034341 | -0.000096 | 0.997209 | 0.131966 | 0.127674 | not_rendered_rate_only |
| stage93_entropy_smoke | stage65_adapter | 16 | 0.033293 | 0.034341 | 0.001048 | 1.031474 | 0.089810 | 0.087664 | not_rendered_rate_only |
| stage96_entropy_broader | linear | 4 | 0.028486 | 0.034341 | 0.005855 | 1.205531 | 0.216279 | 0.207694 | not_rendered_rate_only |
| stage96_entropy_broader | linear | 8 | 0.029031 | 0.034341 | 0.005310 | 1.182915 | 0.131966 | 0.127674 | not_rendered_rate_only |
| stage96_entropy_broader | linear | 16 | 0.028839 | 0.034341 | 0.005502 | 1.190791 | 0.089810 | 0.087664 | not_rendered_rate_only |
| stage96_entropy_broader | stage65_adapter | 4 | 0.033148 | 0.034341 | 0.001193 | 1.035995 | 0.216279 | 0.207694 | not_rendered_rate_only |
| stage96_entropy_broader | stage65_adapter | 8 | 0.033868 | 0.034341 | 0.000473 | 1.013962 | 0.131966 | 0.127674 | not_rendered_rate_only |
| stage96_entropy_broader | stage65_adapter | 16 | 0.033766 | 0.034341 | 0.000575 | 1.017019 | 0.089810 | 0.087664 | not_rendered_rate_only |

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage116_deterministic_vs_entropy_sideinfo_accounting/stage116_deterministic_vs_entropy_sideinfo_accounting_rows.csv`
- points CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage116_deterministic_vs_entropy_sideinfo_accounting/stage116_deterministic_vs_entropy_sideinfo_accounting_points.csv`
- summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage116_deterministic_vs_entropy_sideinfo_accounting/stage116_deterministic_vs_entropy_sideinfo_accounting_summary.json`

## Conclusion

- Deterministic value-only side-info removes transmitted selected-index bytes and stays at `0.034340858459472656` MiB/intermediate for this q6 top10 setup.
- Existing zlib entropy-coded index+value side-info is still smaller for linear groups and remains close for Stage65 adapter groups.
- Stage116 is an accounting package, not a rendered quality validation for deterministic endpoint-diff residuals.
