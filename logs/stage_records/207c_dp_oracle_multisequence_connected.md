# Stage207c DP Oracle Multi-Sequence Connected

Date: 2026-07-02

## Goal

Rerun Stage207 DP oracle on the Stage206c two-sequence connected edge table before promoting Stage208 selector-label data generation.

## Plan

- Use `experiments/stage206c_multisequence_connected_edge_rd/stage206c_multisequence_connected_edge_rd_package.json`.
- Run `scripts/run_stage207_dp_oracle_schedule.py` with `experiments/stage207c_dp_oracle_multisequence_connected` output root.
- Check edge option coverage, fixed baselines, budget oracle gain, graph connectivity, and decoder contract.

## Success Criteria

- Stage206c prerequisite decision is `edge_rd_table_ready_for_stage207_dp`.
- Edge option coverage is complete.
- Schedule graph connectivity passes for both sequences.
- Same-budget DP oracle gain is nonnegative.
- Stage197 decoder contract passes.

## Execution

- Command: `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage207_dp_oracle_schedule.py --stage206_package experiments/stage206c_multisequence_connected_edge_rd/stage206c_multisequence_connected_edge_rd_package.json --output_root experiments/stage207c_dp_oracle_multisequence_connected`

## Outputs

- Output root: `experiments/stage207c_dp_oracle_multisequence_connected/`
- Edge options: `experiments/stage207c_dp_oracle_multisequence_connected/stage207_edge_option_rows.csv`
- Fixed baselines: `experiments/stage207c_dp_oracle_multisequence_connected/stage207_fixed_setting_baselines.csv`
- Budget frontier: `experiments/stage207c_dp_oracle_multisequence_connected/stage207_budget_frontier.csv`
- Budget oracle rows: `experiments/stage207c_dp_oracle_multisequence_connected/stage207_budget_oracle_rows.csv`
- Graph connectivity: `experiments/stage207c_dp_oracle_multisequence_connected/stage207_graph_connectivity.csv`
- Gates: `experiments/stage207c_dp_oracle_multisequence_connected/stage207_dp_oracle_gates.csv`
- Package: `experiments/stage207c_dp_oracle_multisequence_connected/stage207_dp_oracle_schedule_package.json`
- Report: `experiments/stage207c_dp_oracle_multisequence_connected/stage207_dp_oracle_schedule_report.md`

## Results

- Decision: `dp_oracle_schedule_ready_for_selector_labels`.
- Edge option coverage passed: `66/66` options from `22` connected edges and `3` settings.
- Schedule graph connectivity passed: `28` connected transitions total, with `bike-packing` and `parkour` each having `11` edges, `7` nodes, `1` component, and `14` transitions.
- Fixed baselines over `122` internal targets:
  - `topk_keep0p05_q6`: cost `17777952` bytes, mean PSNR `21.63221901330533`, mean dPSNR `+2.3806840851579874`.
  - `topk_keep0p1_q6`: cost `19538702` bytes, mean PSNR `22.999679901928367`, mean dPSNR `+3.748144973781018`.
  - `topk_keep0p2_q6`: cost `22782640` bytes, mean PSNR `24.831585273128514`, mean dPSNR `+5.580050344981165`.
- Same-budget residual allocation gain: `+0.06898291414485058` dB at the `topk_keep0p1_q6` budget.
- Decoder contract passed.

## Decision

- Proceed to Stage208 selector-label data packaging for the Stage206c/207c connected multi-sequence scope.
