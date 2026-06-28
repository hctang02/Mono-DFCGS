# Stage94 Entropy-Coded Residual Side-Info RD Package

Date: 2026-06-28

## Goal

Attach Stage93 actual entropy-coded residual side-info payload bytes to q12 main-anchor RD accounting.

## Inputs

```text
experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_rate_table.csv
experiments/stage93_residual_sideinfo_entropy_codec_smoke/stage93_residual_sideinfo_entropy_codec_summary.csv
```

## Implementation

Added:

```text
scripts/run_stage94_entropy_residual_sideinfo_rd_package.py
```

## Outputs

```text
experiments/stage94_entropy_residual_sideinfo_rd_package/stage94_entropy_residual_sideinfo_rd_rows.csv
experiments/stage94_entropy_residual_sideinfo_rd_package/stage94_entropy_residual_sideinfo_rd_points.csv
experiments/stage94_entropy_residual_sideinfo_rd_package/stage94_entropy_residual_sideinfo_rd_summary.json
experiments/stage94_entropy_residual_sideinfo_rd_package/stage94_entropy_residual_sideinfo_rd_report.md
```

## Results

| base | gap | entropy side MiB/intermediate | entropy direct total | entropy amortized total | PSNR | delta PSNR |
|---|---:|---:|---:|---:|---:|---:|
| linear | 4 | 0.028384844462076824 | 0.21032306522998995 | 0.20322685411447075 | 23.327021755289916 | 3.3372252270957077 |
| linear | 8 | 0.029387855529785158 | 0.12701324228329952 | 0.12333976034207637 | 20.94541228168298 | 2.8242945281834873 |
| linear | 16 | 0.028190135955810547 | 0.08365910570212552 | 0.08189722220488736 | 23.589545045225233 | 4.476530074672733 |
| stage65_adapter | 4 | 0.03340339660644531 | 0.21534161737435845 | 0.20699076822274712 | 23.352831018389125 | 2.870092812012075 |
| stage65_adapter | 8 | 0.03443698883056641 | 0.13206237558408077 | 0.12775775198025996 | 20.943699759081163 | 2.41134126509652 |
| stage65_adapter | 16 | 0.03329300880432129 | 0.08876197855063626 | 0.08668116550036618 | 22.48232041707169 | 3.407310328486104 |

## Conclusion

- Entropy codec v2 reduces side-info rate while preserving decoded rendered quality.
- Side-info remains transmitted information and is included in total rate.
- This is still a 12-task codec smoke RD package, not final full-video RD.
