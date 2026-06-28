# Stage96 Broader Entropy Residual Side-Info RD Package

Date: 2026-06-28

## Goal

Attach Stage95 broader entropy-coded residual side-info rates to q12 main-anchor RD accounting.

## Inputs

```text
experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_rate_table.csv
experiments/stage95_broader_entropy_residual_sideinfo_eval/stage95_broader_entropy_residual_sideinfo_eval_summary.csv
```

## Implementation

Updated:

```text
scripts/run_stage94_entropy_residual_sideinfo_rd_package.py
```

The script now supports stage number, mode, output prefix, report title, quality source, and scope note parameters. Defaults preserve Stage94 output naming.

## Outputs

```text
experiments/stage96_broader_entropy_residual_sideinfo_rd_package/stage96_broader_entropy_residual_sideinfo_rd_rows.csv
experiments/stage96_broader_entropy_residual_sideinfo_rd_package/stage96_broader_entropy_residual_sideinfo_rd_points.csv
experiments/stage96_broader_entropy_residual_sideinfo_rd_package/stage96_broader_entropy_residual_sideinfo_rd_summary.json
experiments/stage96_broader_entropy_residual_sideinfo_rd_package/stage96_broader_entropy_residual_sideinfo_rd_report.md
```

## Results

| base | gap | entropy side MiB/intermediate | entropy direct total | entropy amortized total | PSNR | delta PSNR |
|---|---:|---:|---:|---:|---:|---:|
| linear | 4 | 0.028486092885335285 | 0.21042431365324843 | 0.2033027904319146 | 23.403532118481326 | 3.4187356956292225 |
| linear | 8 | 0.02903069947895251 | 0.12665608623246688 | 0.1230272487975978 | 21.51362735577916 | 3.05603754877287 |
| linear | 16 | 0.028838696687117867 | 0.08430766643343285 | 0.08250524789048798 | 20.344147709407626 | 3.2599322732336695 |
| stage65_adapter | 4 | 0.033147705925835505 | 0.21508592669374865 | 0.20679900021228975 | 22.841151135422116 | 2.7994188265782376 |
| stage65_adapter | 8 | 0.033867986578690376 | 0.13149337333220473 | 0.12725987500986843 | 21.39901144086742 | 2.692464222747122 |
| stage65_adapter | 16 | 0.03376620748768682 | 0.08923517723400179 | 0.08712478926602138 | 20.292022340267458 | 2.9637922008291264 |

## Conclusion

- Stage96 is the preferred broader entropy-coded RD reference for the current residual side-info line.
- Side-info remains explicitly counted as transmitted rate.
- The current limitation is that residual side-info is still teacher-derived; future stages should focus on deployable side-info generation or learned residual prediction.
