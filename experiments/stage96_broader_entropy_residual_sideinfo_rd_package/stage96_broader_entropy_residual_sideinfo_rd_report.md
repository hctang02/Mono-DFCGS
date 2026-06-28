# Stage96 Broader Entropy Residual Side-Info RD Package

## Scope

- Main rate comes from Stage78 q12 static-anchor rate table.
- Side-info rate comes from Stage93 actual entropy-coded payload bytes.
- Rendered quality comes from Stage95 60-task broader entropy codec eval.
- This package uses the 60-task broader q6 top10 entropy eval slice, not final full-video RD.

## RD Rows

| base | gap | main | fixed side | entropy side | fixed direct | entropy direct | entropy amortized | entropy PSNR | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 4 | 0.181938 | 0.041371 | 0.028486 | 0.223310 | 0.210424 | 0.203303 | 23.403532 | 3.418736 |
| linear | 8 | 0.097625 | 0.041371 | 0.029031 | 0.138997 | 0.126656 | 0.123027 | 21.513627 | 3.056038 |
| linear | 16 | 0.055469 | 0.041371 | 0.028839 | 0.096840 | 0.084308 | 0.082505 | 20.344148 | 3.259932 |
| stage65_adapter | 4 | 0.181938 | 0.041371 | 0.033148 | 0.223310 | 0.215086 | 0.206799 | 22.841151 | 2.799419 |
| stage65_adapter | 8 | 0.097625 | 0.041371 | 0.033868 | 0.138997 | 0.131493 | 0.127260 | 21.399011 | 2.692464 |
| stage65_adapter | 16 | 0.055469 | 0.041371 | 0.033766 | 0.096840 | 0.089235 | 0.087125 | 20.292022 | 2.963792 |

## Outputs

- rows CSV: `experiments/stage96_broader_entropy_residual_sideinfo_rd_package/stage96_broader_entropy_residual_sideinfo_rd_rows.csv`
- points CSV: `experiments/stage96_broader_entropy_residual_sideinfo_rd_package/stage96_broader_entropy_residual_sideinfo_rd_points.csv`

## Conclusion

- Entropy coding reduces side-info rate without changing decoded rendered quality relative to fixed q6 residual side-info.
- Side-info remains transmitted information and is included in both direct and amortized total rates.
- If this package uses a broader eval summary, it is the preferred RD reference over the 12-task smoke package.
