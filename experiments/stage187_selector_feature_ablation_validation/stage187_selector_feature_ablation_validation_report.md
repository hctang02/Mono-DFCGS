# Stage187 Selector Feature Ablation Validation

## Scope

This is a selector-label/protocol ablation over Stage163 rows. It does not claim measured full-sequence RD for ablation schedules.

## Decision

- Decision: `feature_ablation_ready_for_budget_sensitivity`.
- Recommended conservative Stage188 low-budget candidate: `drop_interp_rgb`.

## Context

- `stage165_policy`: `rank_gate_t0.65_votes1`. Fixed gate used for ablations.
- `stage165_selected_count`: `70`. Original selected row count on Stage163 rows.
- `stage186_adaptive_rate`: `0.2907429328258184`. Measured full-sequence rate for current adaptive schedule.
- `stage186_adaptive_psnr_delta_vs_gap8`: `0.05161782022076622`. Full-sequence quality context; not recomputed for ablation schedules.
- `stage186_gap8_rate`: `0.2758661759621266`. Measured fixed gap8 rate.
- `stage186_gap4_rate`: `0.33076894444307725`. Measured fixed gap4 rate.

## Feature Ablation Table

| variant | features | selected | keyframes | hard recall | payload recall | hard precision | payload precision | selected payload | delta selected vs full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| proxy_only | 1 | 42 | 331 | 0.400000 | 0.541667 | 0.285714 | 0.928571 | 240519.071 | -28 |
| rgb_only | 2 | 48 | 336 | 0.400000 | 0.638889 | 0.250000 | 0.958333 | 240711.229 | -22 |
| drop_edge_motion | 4 | 60 | 348 | 0.433333 | 0.750000 | 0.216667 | 0.900000 | 233634.467 | -10 |
| drop_hist_motion | 4 | 61 | 349 | 0.733333 | 0.736111 | 0.360656 | 0.868852 | 236943.016 | -9 |
| edge_hist_only | 2 | 67 | 356 | 0.733333 | 0.777778 | 0.328358 | 0.835821 | 231203.642 | -3 |
| motion_proxy_edge_hist | 3 | 68 | 357 | 0.733333 | 0.791667 | 0.323529 | 0.838235 | 231489.338 | -2 |
| drop_interp_rgb | 4 | 69 | 357 | 0.733333 | 0.805556 | 0.318841 | 0.840580 | 231662.899 | -1 |
| full_stage165_features | 5 | 70 | 358 | 0.733333 | 0.819444 | 0.314286 | 0.842857 | 232024.857 | 0 |
| drop_rgb_motion_proxy | 4 | 70 | 358 | 0.733333 | 0.819444 | 0.314286 | 0.842857 | 232024.857 | 0 |
| drop_endpoint_rgb | 4 | 70 | 358 | 0.733333 | 0.819444 | 0.314286 | 0.842857 | 232024.857 | 0 |

## Interpretation

- The five-feature gate remains the highest-recall selector among the evaluated variants.
- Lower-feature variants reduce selected rows and keyframes, which may help Stage188 reduce measured rate, but recall drops.
- Stage188 should evaluate budget/threshold variants as explicit RD points, prioritizing variants with fewer selected rows but acceptable hard/high-payload recall.

## Stage188 Low-Budget Shortlist

| variant | selected | delta selected | hard recall | payload recall | note |
|---|---:|---:|---:|---:|---|
| drop_interp_rgb | 69 | -1 | 0.733333 | 0.805556 | conservative recall-preserving feature ablation |
| motion_proxy_edge_hist | 68 | -2 | 0.733333 | 0.791667 | small budget reduction with no hard-recall loss |
| edge_hist_only | 67 | -3 | 0.733333 | 0.777778 | two-feature motion-only stress point |
| drop_hist_motion | 61 | -9 | 0.733333 | 0.736111 | larger budget reduction with no hard-recall loss but lower payload recall |
| drop_edge_motion | 60 | -10 | 0.433333 | 0.750000 | larger selected-count reduction; expected quality-risk stress point |

## Outputs

- Ablation summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage187_selector_feature_ablation_validation/stage187_selector_feature_ablation_summary.csv`
- Ablation row dump CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage187_selector_feature_ablation_validation/stage187_selector_feature_ablation_rows.csv`
- Context CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage187_selector_feature_ablation_validation/stage187_selector_feature_ablation_context.csv`
