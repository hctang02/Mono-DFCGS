# Stage206b Connected Edge RD Expansion

Date: 2026-07-02

## Goal

Build a small connected-window edge RD table so Stage207 can run a nontrivial schedule-level DP oracle instead of only isolated-edge budget allocation.

## Plan

- Select one complete connected eval window by default, preferring `bike-packing` then `parkour`.
- Use a 24-frame window and include every complete manifest edge inside it for gaps `4,8,12`.
- Reuse Stage206 accounting: exact q12 endpoint keyframe bitstream bytes, exact residual `len(payload)` bytes, and explicit `2` schedule metadata bytes per edge.
- Reuse q6 top-k residual keep fractions `0.05,0.10,0.20`.
- After this table is built, rerun Stage207 with `--stage206_package` pointing to the Stage206b package.

## Success Criteria

- At least one selected window has connected edge transitions.
- Target metric rows and edge rows have no errors.
- Every selected edge has complete intermediate target coverage.
- Keyframe/residual/metadata bytes are counted explicitly.
- Each tested gap has positive edge-level dPSNR headroom.

## Execution

- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage206b_connected_edge_rd_expansion.py scripts/run_stage207_dp_oracle_schedule.py`
- Pre-run GPU check: `nvidia-smi` at 2026-07-02 01:22:02; GPU1 was idle.
- Command: `CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage206b_connected_edge_rd_expansion.py --device cuda --max_windows 1 --window_frames 24`

## Outputs

- Output root: `experiments/stage206b_connected_edge_rd_expansion/`
- Windows: `experiments/stage206b_connected_edge_rd_expansion/stage206b_connected_windows.csv`
- Selected edges: `experiments/stage206b_connected_edge_rd_expansion/stage206b_selected_edges.csv`
- Target metric rows: `experiments/stage206b_connected_edge_rd_expansion/stage206b_target_metric_rows.csv`
- Edge RD rows: `experiments/stage206b_connected_edge_rd_expansion/stage206b_edge_rd_rows.csv`
- Summary: `experiments/stage206b_connected_edge_rd_expansion/stage206b_edge_rd_summary.csv`
- Best by gap: `experiments/stage206b_connected_edge_rd_expansion/stage206b_edge_rd_best_by_gap.csv`
- Gates: `experiments/stage206b_connected_edge_rd_expansion/stage206b_edge_rd_gates.csv`
- Package: `experiments/stage206b_connected_edge_rd_expansion/stage206b_connected_edge_rd_expansion_package.json`
- Report: `experiments/stage206b_connected_edge_rd_expansion/stage206b_connected_edge_rd_expansion_report.md`

## Results

- Decision: `edge_rd_table_ready_for_stage207_dp`.
- Selected one connected eval window: `bike-packing:00000:00024`.
- Window coverage: `11` edges, `61` internal target rows, `6` gap4 chain edges, `14` connected transitions.
- Tested q6 top-k residual keep fractions `0.05,0.10,0.20` with q12 endpoint keyframes and `2` metadata bytes per edge.
- Best gap4: `topk_keep0p2_q6`, edge total `1605417.1666666667` bytes, DP incremental `885768.6666666666` bytes, residual `166117.66666666666` bytes, corrected PSNR `26.031263851493254`, dPSNR `+4.986391989862658`.
- Best gap8: `topk_keep0p2_q6`, edge total `1833385.6666666667` bytes, DP incremental `1113739.0` bytes, residual `394089.3333333333` bytes, corrected PSNR `25.821074970091768`, dPSNR `+5.676789260065095`.
- Best gap12: `topk_keep0p2_q6`, edge total `2063898.0` bytes, DP incremental `1344249.5` bytes, residual `624597.5` bytes, corrected PSNR `25.40221496330657`, dPSNR `+5.768186956092006`.
- Gates passed: Stage205 prereq, target metric rows ok, edge rows ok, gap coverage, payload counted nonzero, schedule metadata counted, positive edge headroom, Stage197 decoder contract.

## Decision

- Use this connected-window package for Stage207b DP oracle rerun.
