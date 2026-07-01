# Stage207b DP Oracle Connected Window

Date: 2026-07-02

## Goal

Rerun Stage207 DP oracle on the Stage206b connected edge table and verify that the schedule-level connectivity blocker is resolved for a small window.

## Plan

- Use `experiments/stage206b_connected_edge_rd_expansion/stage206b_connected_edge_rd_expansion_package.json` as the edge package.
- Run `scripts/run_stage207_dp_oracle_schedule.py` with a separate Stage207b output root.
- Check option coverage, fixed baselines, budget oracle gain, graph connectivity, and decoder contract.
- Treat this as a small connected-window proof, not final full-sequence schedule evidence.

## Success Criteria

- Stage206b prerequisite decision is `edge_rd_table_ready_for_stage207_dp`.
- Edge options cover every connected-window edge and residual setting.
- Schedule graph has connected transitions.
- DP oracle reports a nonnegative same-budget gain.
- Stage197 decoder contract remains valid.

## Execution

- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage207_dp_oracle_schedule.py --stage206_package experiments/stage206b_connected_edge_rd_expansion/stage206b_connected_edge_rd_expansion_package.json --output_root experiments/stage207b_dp_oracle_connected_window`

## Outputs

- Output root: `experiments/stage207b_dp_oracle_connected_window/`
- Edge options: `experiments/stage207b_dp_oracle_connected_window/stage207_edge_option_rows.csv`
- Fixed baselines: `experiments/stage207b_dp_oracle_connected_window/stage207_fixed_setting_baselines.csv`
- Budget frontier: `experiments/stage207b_dp_oracle_connected_window/stage207_budget_frontier.csv`
- Budget oracle rows: `experiments/stage207b_dp_oracle_connected_window/stage207_budget_oracle_rows.csv`
- Graph connectivity: `experiments/stage207b_dp_oracle_connected_window/stage207_graph_connectivity.csv`
- Gates: `experiments/stage207b_dp_oracle_connected_window/stage207_dp_oracle_gates.csv`
- Package: `experiments/stage207b_dp_oracle_connected_window/stage207_dp_oracle_schedule_package.json`
- Report: `experiments/stage207b_dp_oracle_connected_window/stage207_dp_oracle_schedule_report.md`

## Results

- Decision: `dp_oracle_schedule_ready_for_selector_labels`.
- Edge option coverage passed: `33/33` options from `11` connected edges and `3` residual settings.
- Schedule graph connectivity passed: `14` connected transitions in `bike-packing`, `1` component, `11` edges, `7` nodes.
- Fixed baselines over `61` sampled internal targets:
  - `topk_keep0p05_q6`: cost `8893010` bytes, mean PSNR `23.104662593777928`, mean dPSNR `+2.8786575187184025`.
  - `topk_keep0p1_q6`: cost `9764534` bytes, mean PSNR `24.43813548907444`, mean dPSNR `+4.212130414014913`.
  - `topk_keep0p2_q6`: cost `11344328` bytes, mean PSNR `25.73203365395984`, mean dPSNR `+5.5060285789003105`.
- Same-budget residual allocation gain: `+0.016174288512960544` dB at the `topk_keep0p1_q6` budget.
- Decoder contract passed.

## Decision

- The small connected-window DP oracle gate passes.
- This is enough to validate DP plumbing and local selector-label feasibility, but it is still too small for robust selector training; proceed to a multi-sequence connected expansion before full Stage208/209 promotion.
