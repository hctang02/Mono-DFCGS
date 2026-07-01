# Stage207 DP Oracle Schedule

Date: 2026-07-02

## Goal

Compute a DP oracle over Stage206 edge RD rows and determine whether the current edge table is sufficient to create schedule-level oracle labels before selector training.

## Plan

- Read Stage206 edge RD rows and verify Stage206 decision.
- Build edge options from q6 keep-fraction settings using `dp_incremental_bytes` and target-count-weighted PSNR score.
- Compute fixed-setting baselines and a residual-budget DP frontier over the selected Stage206 edges.
- Audit graph connectivity to check whether selected Stage206 edges form nontrivial schedule paths.
- Promote selector-label generation only if the edge graph supports schedule-level DP; otherwise record the blocker explicitly.

## Success Criteria

- Stage206 prerequisite decision is `edge_rd_table_ready_for_stage207_dp`.
- Every selected edge has one option per residual setting.
- Budget DP runs without hidden side-info and reports same-budget oracle deltas.
- Schedule graph has at least one connected edge transition before claiming schedule-level oracle readiness.

## Execution

- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage207_dp_oracle_schedule.py`
- Pre-run GPU check: `nvidia-smi` at 2026-07-02 00:50:48. Stage207 is CPU CSV/DP aggregation and did not use GPU.
- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage207_dp_oracle_schedule.py`

## Outputs

- Output root: `experiments/stage207_dp_oracle_schedule/`
- Edge options: `experiments/stage207_dp_oracle_schedule/stage207_edge_option_rows.csv`
- Fixed setting baselines: `experiments/stage207_dp_oracle_schedule/stage207_fixed_setting_baselines.csv`
- Budget frontier: `experiments/stage207_dp_oracle_schedule/stage207_budget_frontier.csv`
- Budget oracle rows: `experiments/stage207_dp_oracle_schedule/stage207_budget_oracle_rows.csv`
- Graph connectivity: `experiments/stage207_dp_oracle_schedule/stage207_graph_connectivity.csv`
- Gates: `experiments/stage207_dp_oracle_schedule/stage207_dp_oracle_gates.csv`
- Package: `experiments/stage207_dp_oracle_schedule/stage207_dp_oracle_schedule_package.json`
- Report: `experiments/stage207_dp_oracle_schedule/stage207_dp_oracle_schedule_report.md`

## Results

- Decision: `dp_oracle_schedule_graph_insufficient`.
- Stage206 prerequisite passed: `edge_rd_table_ready_for_stage207_dp`.
- Edge option coverage passed: `18/18` options from `6` edges and `3` residual settings.
- Fixed setting baselines over `37` sampled internal targets:
  - `topk_keep0p05_q6`: cost `4905660` bytes, mean PSNR `22.519469828354364`, mean dPSNR `+1.627647700365514`.
  - `topk_keep0p1_q6`: cost `5430475` bytes, mean PSNR `23.633188273126244`, mean dPSNR `+2.7413661451373974`.
  - `topk_keep0p2_q6`: cost `6399092` bytes, mean PSNR `25.382111703299287`, mean dPSNR `+4.490289575310435`.
- Residual-budget oracle ran and found a same-budget gain of `+0.017853956775965685` dB at the `topk_keep0p1_q6` budget, but this is only residual-budget allocation over sampled independent edges, not a schedule oracle.
- Schedule graph connectivity failed: `0` connected edge transitions. Per-sequence connectivity rows all failed (`bike-packing`, `dog`, `dogs-jump`, `paragliding-launch`, `parkour`). Stage206 sampled edges are isolated and cannot support nontrivial schedule-level DP.
- Decoder contract inherited from Stage206 passed; Stage207 reads measured edge costs only and uses no target dense/RGB decoder input.

## Decision

- Do not proceed to Stage208 selector labels or Stage209 selector training from the current Stage206 sampled edge table.
- Required next work: build an expanded connected edge RD table over at least one sequence/window with contiguous candidate edges, then rerun Stage207 DP oracle schedule.
