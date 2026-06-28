# Stage90 Broader q6 Residual Side-Info RD Package

Date: 2026-06-28

## Goal

Attach Stage89 60-task q6 top10 residual side-info eval to total-rate RD accounting.

## Inputs

```text
experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_rate_table.csv
experiments/stage89_broader_q6_residual_sideinfo_eval/stage89_broader_q6_residual_sideinfo_eval_summary.csv
```

## Implementation

Updated:

```text
scripts/run_stage88_residual_sideinfo_rd_package.py
```

The script now supports stage number, mode, output prefix, report title, quality source, scope note, and plot y-label parameters. Defaults preserve Stage88 output naming.

## Rate Definitions

```text
direct_total_mib_per_frame = q12_main_anchor_mib_per_frame + side_info_mib_per_intermediate_frame
amortized_total_mib_per_frame = q12_main_anchor_mib_per_frame + side_info_mib_per_intermediate_frame * ((gap - 1) / gap)
```

## Outputs

```text
experiments/stage90_broader_q6_residual_sideinfo_rd_package/stage90_broader_q6_residual_sideinfo_rd_rows.csv
experiments/stage90_broader_q6_residual_sideinfo_rd_package/stage90_broader_q6_residual_sideinfo_rd_points.csv
experiments/stage90_broader_q6_residual_sideinfo_rd_package/stage90_broader_q6_residual_sideinfo_rd_summary.json
experiments/stage90_broader_q6_residual_sideinfo_rd_package/stage90_broader_q6_residual_sideinfo_rd_report.md
```

RD plots were generated locally but not committed because image files are ignored by `.gitignore`.

## Results

| base | gap | tasks | direct total | amortized total | side PSNR | delta PSNR |
|---|---:|---:|---:|---:|---:|---:|
| linear | 4 | 18 | 0.22329192331307915 | 0.21295349767678765 | 23.403666365261454 | 3.4188699424093465 |
| linear | 8 | 19 | 0.13897908929868036 | 0.1338098764805346 | 21.513744759131683 | 3.056154952125395 |
| linear | 16 | 23 | 0.09682267229148099 | 0.09423806588240811 | 20.344221350298344 | 3.2600059141243887 |
| stage65_adapter | 4 | 18 | 0.22329192331307915 | 0.21295349767678765 | 22.84099865483423 | 2.7992663459903557 |
| stage65_adapter | 8 | 19 | 0.13897908929868036 | 0.1338098764805346 | 21.398942062762643 | 2.6923948446423474 |
| stage65_adapter | 16 | 23 | 0.09682267229148099 | 0.09423806588240811 | 20.29200458015274 | 2.9637744407144093 |

## Conclusion

- Broader q6 top10 residual side-info remains strong after explicit total-rate accounting.
- Side-info rate is transmitted information and must remain included in RD claims.
- This is still not a final codec because residual side-info is teacher-derived and not entropy-coded/bitstream-packed.
