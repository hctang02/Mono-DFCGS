# Stage188 Lower-Budget Selector Sensitivity

## Scope

This stage reuses Stage184/186 measured rows. New candidate rates use an additive single-anchor keyframe scope, not the Stage185 schedule-packed keyframe scope.

## Decision

- Decision: `lower_budget_positive_quality_candidates_found_but_gap8_rate_not_reached`.
- Lowest-rate positive candidate: `interval_top10pct_cells` at `0.2773746177516859` MiB/frame additive, PSNR `29.38112562842953`.
- Its additive rate delta vs gap8 is `0.0014903267483045712` MiB/frame; it reduces full adaptive overhead by `0.01339097998630051` MiB/frame but does not reach gap8 rate.
- Highest-quality below-full candidate: `interval_top90pct_cells` at `0.289479501370253` MiB/frame additive, PSNR `29.424507356466457`.
- Balanced half-overhead candidate: `interval_score_ge4p0` at `0.2829920490602662` MiB/frame additive, PSNR `29.41013285788653`.

## Context

- `stage185_rate_scope`: `measured_schedule_packed_q12_keyframes_plus_measured_stage158_residual_payloads_plus_exact_metadata`. Packed keyframe scope for Stage185/186 context, not reused as new candidate scope.
- `stage185_gap8_mib_per_frame`: `0.2758661759621266`. Packed-keyframe measured baseline.
- `stage185_adaptive_mib_per_frame`: `0.2907429328258184`. Packed-keyframe measured full Stage165 rate.
- `stage185_gap4_mib_per_frame`: `0.33076894444307725`. Packed-keyframe measured gap4 rate.
- `stage186_adaptive_psnr`: `29.4255826920606`. Full Stage165 quality context.
- `stage188_rate_scope`: `measured_single_anchor_additive_keyframes_plus_measured_stage158_residuals_plus_exact_metadata`. Apples-to-apples sensitivity scope for all Stage188 candidates and baselines.

## Fully Covered Additive RD-Quality

| candidate | family | keyframes | cells kept | MiB/frame | dRate vs gap8 | dRate vs full | PSNR | dPSNR vs gap8 | SSIM | MS-SSIM | LPIPS | dLPIPS vs gap8 | all-metric positive |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | baseline | 292 | 0 | 0.275884291003 | 0.000000000000 | -0.014881306735 | 29.373965 | 0.000000 | 0.867626 | 0.984343 | 0.168692 | 0.000000 | 0 |
| interval_top10pct_cells | interval_budget_reuse | 299 | 6 | 0.277374617752 | 0.001490326748 | -0.013390979986 | 29.381126 | 0.007161 | 0.867833 | 0.984378 | 0.168325 | -0.000367 | 1 |
| interval_top20pct_cells | interval_budget_reuse | 306 | 12 | 0.278945738223 | 0.003061447220 | -0.011819859515 | 29.387187 | 0.013222 | 0.868010 | 0.984410 | 0.168048 | -0.000644 | 1 |
| interval_top30pct_cells | interval_budget_reuse | 314 | 18 | 0.280760905813 | 0.004876614810 | -0.010004691925 | 29.400502 | 0.026537 | 0.868323 | 0.984462 | 0.167361 | -0.001331 | 1 |
| interval_top40pct_cells | interval_budget_reuse | 321 | 24 | 0.282313781956 | 0.006429490952 | -0.008451815782 | 29.408900 | 0.034935 | 0.868597 | 0.984510 | 0.166879 | -0.001813 | 1 |
| interval_score_ge4p0 | interval_budget_reuse | 324 | 27 | 0.282992049060 | 0.007107758057 | -0.007773548678 | 29.410133 | 0.036168 | 0.868644 | 0.984519 | 0.166827 | -0.001865 | 1 |
| interval_top50pct_cells | interval_budget_reuse | 328 | 30 | 0.283883254131 | 0.007998963127 | -0.006882343607 | 29.411049 | 0.037084 | 0.868674 | 0.984525 | 0.166787 | -0.001905 | 1 |
| interval_score_ge3p0 | interval_budget_reuse | 330 | 32 | 0.284342459526 | 0.008458168522 | -0.006423138212 | 29.411598 | 0.037633 | 0.868713 | 0.984533 | 0.166735 | -0.001956 | 1 |
| row_proxy_only | row_level_feature_ablation | 331 |  | 0.284533634253 | 0.008649343249 | -0.006231963485 | 29.411642 | 0.037678 | 0.868719 | 0.984535 | 0.166729 | -0.001962 | 1 |
| interval_top60pct_cells | interval_budget_reuse | 334 | 36 | 0.285203492898 | 0.009319201894 | -0.005562104840 | 29.413452 | 0.039487 | 0.868783 | 0.984547 | 0.166620 | -0.002072 | 1 |
| row_rgb_only | row_level_feature_ablation | 336 |  | 0.285687663187 | 0.009803372183 | -0.005077934551 | 29.418358 | 0.044393 | 0.868951 | 0.984573 | 0.166447 | -0.002244 | 1 |
| interval_score_ge2p0 | interval_budget_reuse | 338 | 40 | 0.286103565852 | 0.010219274848 | -0.004662031886 | 29.417498 | 0.043533 | 0.868944 | 0.984575 | 0.166434 | -0.002258 | 1 |
| interval_top75pct_cells | interval_budget_reuse | 343 | 45 | 0.287222508253 | 0.011338217250 | -0.003543089485 | 29.421030 | 0.047065 | 0.869115 | 0.984605 | 0.166206 | -0.002486 | 1 |
| row_drop_edge_motion | row_level_feature_ablation | 348 |  | 0.288508673320 | 0.012624382317 | -0.002256924418 | 29.422905 | 0.048940 | 0.869131 | 0.984613 | 0.166122 | -0.002570 | 1 |
| row_drop_hist_motion | row_level_feature_ablation | 349 |  | 0.288580314346 | 0.012696023343 | -0.002185283392 | 29.421868 | 0.047904 | 0.869160 | 0.984617 | 0.166191 | -0.002501 | 1 |
| interval_top90pct_cells | interval_budget_reuse | 353 | 54 | 0.289479501370 | 0.013595210367 | -0.001286096368 | 29.424507 | 0.050542 | 0.869250 | 0.984635 | 0.166018 | -0.002674 | 1 |
| row_edge_hist_only | row_level_feature_ablation | 356 |  | 0.290326213407 | 0.014441922404 | -0.000439384331 | 29.423469 | 0.049504 | 0.869231 | 0.984637 | 0.166013 | -0.002679 | 1 |
| row_drop_interp_rgb | row_level_feature_ablation | 357 |  | 0.290545773661 | 0.014661482658 | -0.000219824077 | 29.423890 | 0.049925 | 0.869242 | 0.984639 | 0.165996 | -0.002696 | 1 |
| row_motion_proxy_edge_hist | row_level_feature_ablation | 357 |  | 0.290545773661 | 0.014661482658 | -0.000219824077 | 29.423890 | 0.049925 | 0.869242 | 0.984639 | 0.165996 | -0.002696 | 1 |
| stage165_adaptive_full | baseline | 358 | 59 | 0.290765597738 | 0.014881306735 | 0.000000000000 | 29.425583 | 0.051618 | 0.869294 | 0.984647 | 0.165937 | -0.002754 | 1 |
| row_drop_endpoint_rgb | row_level_feature_ablation | 358 |  | 0.290765597738 | 0.014881306735 | 0.000000000000 | 29.425583 | 0.051618 | 0.869294 | 0.984647 | 0.165937 | -0.002754 | 1 |
| row_drop_rgb_motion_proxy | row_level_feature_ablation | 358 |  | 0.290765597738 | 0.014881306735 | 0.000000000000 | 29.425583 | 0.051618 | 0.869294 | 0.984647 | 0.165937 | -0.002754 | 1 |
| uniform_gap4 | baseline | 536 |  | 0.330803858751 | 0.054919567747 | 0.040038261013 | 29.535716 | 0.161751 | 0.873944 | 0.985529 | 0.159472 | -0.009220 | 1 |

## Row-Level Ablation Coverage Audit

| candidate | unique targets | keyframes | residuals | missing residual payloads | missing residual quality | status |
|---|---:|---:|---:|---:|---:|---|
| drop_edge_motion | 56 | 348 | 1651 | 0 | 0 | complete_reused_stage184_stage186_rows |
| drop_endpoint_rgb | 66 | 358 | 1641 | 0 | 0 | complete_reused_stage184_stage186_rows |
| drop_hist_motion | 57 | 349 | 1650 | 0 | 0 | complete_reused_stage184_stage186_rows |
| drop_interp_rgb | 65 | 357 | 1642 | 0 | 0 | complete_reused_stage184_stage186_rows |
| drop_rgb_motion_proxy | 66 | 358 | 1641 | 0 | 0 | complete_reused_stage184_stage186_rows |
| edge_hist_only | 64 | 356 | 1643 | 0 | 0 | complete_reused_stage184_stage186_rows |
| full_stage165_features | 66 | 358 | 1641 | 0 | 0 | complete_reused_stage184_stage186_rows |
| motion_proxy_edge_hist | 65 | 357 | 1642 | 0 | 0 | complete_reused_stage184_stage186_rows |
| proxy_only | 39 | 331 | 1668 | 0 | 0 | complete_reused_stage184_stage186_rows |
| rgb_only | 44 | 336 | 1663 | 0 | 0 | complete_reused_stage184_stage186_rows |

## Interpretation

- Interval-level candidates are fully measured because each gap8 interval either reuses full Stage165 adaptive rows or uniform gap8 rows.
- Row-level ablations are audited after collapsing duplicate sampled rows to unique target frames. In this run all audited row-level variants were fully covered by existing measured rows.
- The lowest-rate positive candidate reduces most of the full adaptive overhead but still remains above uniform gap8 rate under the additive sensitivity scope.
- Additive single-anchor keyframe rates are for sensitivity only and should not be mixed numerically with Stage185 schedule-packed keyframe rates.

## Outputs

- Sensitivity CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage188_lower_budget_selector_sensitivity/stage188_lower_budget_selector_sensitivity_rd_quality.csv`
- Interval cells CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage188_lower_budget_selector_sensitivity/stage188_interval_cell_ranking.csv`
- Row-level coverage CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage188_lower_budget_selector_sensitivity/stage188_row_level_ablation_coverage_audit.csv`
- Context CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage188_lower_budget_selector_sensitivity/stage188_lower_budget_selector_sensitivity_context.csv`
