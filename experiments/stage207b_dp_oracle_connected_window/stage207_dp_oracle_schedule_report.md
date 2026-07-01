# Stage207 DP Oracle Schedule

## Decision

- Decision: `dp_oracle_schedule_ready_for_selector_labels`.
- Scope: Stage206 sampled-edge DP preflight; not full-sequence schedule RD.

## Fixed Setting Baselines

| setting | cost bytes | mean PSNR | mean dPSNR |
|---|---:|---:|---:|
| topk_keep0p05_q6 | 8893010 | 23.104663 | 2.878658 |
| topk_keep0p1_q6 | 9764534 | 24.438135 | 4.212130 |
| topk_keep0p2_q6 | 11344328 | 25.732034 | 5.506029 |

## Budget Oracle

| budget | fixed PSNR | oracle PSNR | delta | chosen cost |
|---|---:|---:|---:|---:|
| topk_keep0p05_q6 | 23.104663 | 23.104663 | 0.000000 | 8893010 |
| topk_keep0p1_q6 | 24.438135 | 24.454310 | 0.016174 | 9761325 |
| topk_keep0p2_q6 | 25.732034 | 25.732034 | 0.000000 | 11344328 |

## Connectivity Audit

| sequence | edges | components | connected transitions | status |
|---|---:|---:|---:|---|
| bike-packing | 11 | 1 | 14 | pass |

## Gates

| gate | status | value | threshold | detail |
|---|---|---|---|---|
| stage206_prereq | pass | edge_rd_table_ready_for_stage207_dp | edge_rd_table_ready_for_stage207_dp | experiments/stage206b_connected_edge_rd_expansion/stage206b_connected_edge_rd_expansion_package.json |
| edge_option_coverage | pass | 33 | 33 | one option per edge/setting from Stage206 edge rows |
| fixed_baselines_present | pass | 3 | >=2 settings | topk_keep0p05_q6;topk_keep0p1_q6;topk_keep0p2_q6 |
| budget_oracle_nonnegative_gain | pass | 0.016174288512960544 | >=0 dB vs same-budget fixed settings | residual-budget oracle only; not a schedule oracle |
| schedule_graph_connected | pass | 14 | >0 connected edge transitions | Stage206 sampled edges are not enough for nontrivial schedule DP if this fails |
| stage197_decoder_contract | pass | 0 | no target dense/RGB decoder input | Stage207 reads measured edge costs only; decoder contract inherited from Stage206 |

## Outputs

- options: `experiments/stage207b_dp_oracle_connected_window/stage207_edge_option_rows.csv`
- fixed baselines: `experiments/stage207b_dp_oracle_connected_window/stage207_fixed_setting_baselines.csv`
- frontier: `experiments/stage207b_dp_oracle_connected_window/stage207_budget_frontier.csv`
- budget oracle: `experiments/stage207b_dp_oracle_connected_window/stage207_budget_oracle_rows.csv`
- graph connectivity: `experiments/stage207b_dp_oracle_connected_window/stage207_graph_connectivity.csv`
- gates: `experiments/stage207b_dp_oracle_connected_window/stage207_dp_oracle_gates.csv`
- package: `experiments/stage207b_dp_oracle_connected_window/stage207_dp_oracle_schedule_package.json`
