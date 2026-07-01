# Stage207 DP Oracle Schedule

## Decision

- Decision: `dp_oracle_schedule_ready_for_selector_labels`.
- Scope: Stage206 sampled-edge DP preflight; not full-sequence schedule RD.

## Fixed Setting Baselines

| setting | cost bytes | mean PSNR | mean dPSNR |
|---|---:|---:|---:|
| topk_keep0p05_q6 | 17777952 | 21.632219 | 2.380684 |
| topk_keep0p1_q6 | 19538702 | 22.999680 | 3.748145 |
| topk_keep0p2_q6 | 22782640 | 24.831585 | 5.580050 |

## Budget Oracle

| budget | fixed PSNR | oracle PSNR | delta | chosen cost |
|---|---:|---:|---:|---:|
| topk_keep0p05_q6 | 21.632219 | 21.632219 | 0.000000 | 17777952 |
| topk_keep0p1_q6 | 22.999680 | 23.068663 | 0.068983 | 19527034 |
| topk_keep0p2_q6 | 24.831585 | 24.831585 | -0.000000 | 22782640 |

## Connectivity Audit

| sequence | edges | components | connected transitions | status |
|---|---:|---:|---:|---|
| bike-packing | 11 | 1 | 14 | pass |
| parkour | 11 | 1 | 14 | pass |

## Gates

| gate | status | value | threshold | detail |
|---|---|---|---|---|
| stage206_prereq | pass | edge_rd_table_ready_for_stage207_dp | edge_rd_table_ready_for_stage207_dp | experiments/stage206c_multisequence_connected_edge_rd/stage206c_multisequence_connected_edge_rd_package.json |
| edge_option_coverage | pass | 66 | 66 | one option per edge/setting from Stage206 edge rows |
| fixed_baselines_present | pass | 3 | >=2 settings | topk_keep0p05_q6;topk_keep0p1_q6;topk_keep0p2_q6 |
| budget_oracle_nonnegative_gain | pass | 0.06898291414485058 | >=0 dB vs same-budget fixed settings | residual-budget oracle only; not a schedule oracle |
| schedule_graph_connected | pass | 28 | >0 connected edge transitions | Stage206 sampled edges are not enough for nontrivial schedule DP if this fails |
| stage197_decoder_contract | pass | 0 | no target dense/RGB decoder input | Stage207 reads measured edge costs only; decoder contract inherited from Stage206 |

## Outputs

- options: `experiments/stage207c_dp_oracle_multisequence_connected/stage207_edge_option_rows.csv`
- fixed baselines: `experiments/stage207c_dp_oracle_multisequence_connected/stage207_fixed_setting_baselines.csv`
- frontier: `experiments/stage207c_dp_oracle_multisequence_connected/stage207_budget_frontier.csv`
- budget oracle: `experiments/stage207c_dp_oracle_multisequence_connected/stage207_budget_oracle_rows.csv`
- graph connectivity: `experiments/stage207c_dp_oracle_multisequence_connected/stage207_graph_connectivity.csv`
- gates: `experiments/stage207c_dp_oracle_multisequence_connected/stage207_dp_oracle_gates.csv`
- package: `experiments/stage207c_dp_oracle_multisequence_connected/stage207_dp_oracle_schedule_package.json`
