# Stage207 DP Oracle Schedule

## Decision

- Decision: `dp_oracle_schedule_graph_insufficient`.
- Scope: Stage206 sampled-edge DP preflight; not full-sequence schedule RD.

## Fixed Setting Baselines

| setting | cost bytes | mean PSNR | mean dPSNR |
|---|---:|---:|---:|
| topk_keep0p05_q6 | 4905660 | 22.519470 | 1.627648 |
| topk_keep0p1_q6 | 5430475 | 23.633188 | 2.741366 |
| topk_keep0p2_q6 | 6399092 | 25.382112 | 4.490290 |

## Budget Oracle

| budget | fixed PSNR | oracle PSNR | delta | chosen cost |
|---|---:|---:|---:|---:|
| topk_keep0p05_q6 | 22.519470 | 22.519470 | 0.000000 | 4905660 |
| topk_keep0p1_q6 | 23.633188 | 23.651042 | 0.017854 | 5417451 |
| topk_keep0p2_q6 | 25.382112 | 25.382112 | 0.000000 | 6399092 |

## Connectivity Audit

| sequence | edges | components | connected transitions | status |
|---|---:|---:|---:|---|
| bike-packing | 1 | 1 | 0 | fail |
| dog | 1 | 1 | 0 | fail |
| dogs-jump | 1 | 1 | 0 | fail |
| paragliding-launch | 1 | 1 | 0 | fail |
| parkour | 2 | 2 | 0 | fail |

## Gates

| gate | status | value | threshold | detail |
|---|---|---|---|---|
| stage206_prereq | pass | edge_rd_table_ready_for_stage207_dp | edge_rd_table_ready_for_stage207_dp | /mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage206_edge_rd_table/stage206_edge_rd_table_package.json |
| edge_option_coverage | pass | 18 | 18 | one option per edge/setting from Stage206 edge rows |
| fixed_baselines_present | pass | 3 | >=2 settings | topk_keep0p05_q6;topk_keep0p1_q6;topk_keep0p2_q6 |
| budget_oracle_nonnegative_gain | pass | 0.017853956775965685 | >=0 dB vs same-budget fixed settings | residual-budget oracle only; not a schedule oracle |
| schedule_graph_connected | fail | 0 | >0 connected edge transitions | Stage206 sampled edges are not enough for nontrivial schedule DP if this fails |
| stage197_decoder_contract | pass | 0 | no target dense/RGB decoder input | Stage207 reads measured edge costs only; decoder contract inherited from Stage206 |

## Outputs

- options: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage207_dp_oracle_schedule/stage207_edge_option_rows.csv`
- fixed baselines: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage207_dp_oracle_schedule/stage207_fixed_setting_baselines.csv`
- frontier: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage207_dp_oracle_schedule/stage207_budget_frontier.csv`
- budget oracle: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage207_dp_oracle_schedule/stage207_budget_oracle_rows.csv`
- graph connectivity: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage207_dp_oracle_schedule/stage207_graph_connectivity.csv`
- gates: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage207_dp_oracle_schedule/stage207_dp_oracle_gates.csv`
- package: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage207_dp_oracle_schedule/stage207_dp_oracle_schedule_package.json`
