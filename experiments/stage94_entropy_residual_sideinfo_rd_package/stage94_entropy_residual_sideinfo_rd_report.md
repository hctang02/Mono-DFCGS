# Stage94 Entropy-Coded Residual Side-Info RD Package

## Scope

- Main rate comes from Stage78 q12 static-anchor rate table.
- Side-info rate comes from Stage93 actual entropy-coded payload bytes.
- Rendered quality comes from Stage93 entropy codec smoke.
- This is a 12-task codec smoke package, not final full-video RD.

## RD Rows

| base | gap | main | fixed side | entropy side | fixed direct | entropy direct | entropy amortized | entropy PSNR | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 4 | 0.181938 | 0.041371 | 0.028385 | 0.223310 | 0.210323 | 0.203227 | 23.327022 | 3.337225 |
| linear | 8 | 0.097625 | 0.041371 | 0.029388 | 0.138997 | 0.127013 | 0.123340 | 20.945412 | 2.824295 |
| linear | 16 | 0.055469 | 0.041371 | 0.028190 | 0.096840 | 0.083659 | 0.081897 | 23.589545 | 4.476530 |
| stage65_adapter | 4 | 0.181938 | 0.041371 | 0.033403 | 0.223310 | 0.215342 | 0.206991 | 23.352831 | 2.870093 |
| stage65_adapter | 8 | 0.097625 | 0.041371 | 0.034437 | 0.138997 | 0.132062 | 0.127758 | 20.943700 | 2.411341 |
| stage65_adapter | 16 | 0.055469 | 0.041371 | 0.033293 | 0.096840 | 0.088762 | 0.086681 | 22.482320 | 3.407310 |

## Outputs

- rows CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage94_entropy_residual_sideinfo_rd_package/stage94_entropy_residual_sideinfo_rd_rows.csv`
- points CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage94_entropy_residual_sideinfo_rd_package/stage94_entropy_residual_sideinfo_rd_points.csv`

## Conclusion

- Entropy coding reduces side-info rate without changing decoded rendered quality relative to fixed q6 residual side-info.
- Side-info remains transmitted information and is included in both direct and amortized total rates.
- The next step is broader eval with the entropy codec v2, using the Stage89 60-task slice.
