# Stage206c Multi-Sequence Connected Edge RD

Date: 2026-07-02

## Goal

Expand the connected edge RD table beyond one local window so Stage207 can validate a stronger multi-sequence DP oracle before Stage208 selector-label packaging.

## Plan

- Reuse the Stage206b connected expansion script with reusable `stage_label` support.
- Select one complete 24-frame connected window from each of `bike-packing` and `parkour`.
- Include all complete manifest edges inside each window for gaps `4,8,12`.
- Reuse q12 endpoint keyframes, q6 top-k residual keep fractions `0.05,0.10,0.20`, exact residual `len(payload)` bytes, and explicit `2` metadata bytes per edge.

## Success Criteria

- At least two sequence windows are selected.
- Each sequence has connected edge transitions.
- Target metric rows and edge rows have no errors.
- Each gap has positive edge-level dPSNR headroom.
- Stage197 decoder contract remains valid.

## Execution

- Syntax check was already run before Stage206b/206c: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage206b_connected_edge_rd_expansion.py`.
- Pre-run GPU check: `nvidia-smi` at 2026-07-02 01:29:19; GPU1 was idle.
- Command: `CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage206b_connected_edge_rd_expansion.py --device cuda --output_root experiments/stage206c_multisequence_connected_edge_rd --stage_label 206c --stage_name multisequence_connected_edge_rd --max_windows 2 --max_windows_per_sequence 1 --window_frames 24 --sequences bike-packing parkour`

## Outputs

- Output root: `experiments/stage206c_multisequence_connected_edge_rd/`
- Windows: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_connected_windows.csv`
- Selected edges: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_selected_edges.csv`
- Target metric rows: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_target_metric_rows.csv`
- Edge RD rows: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_edge_rd_rows.csv`
- Summary: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_edge_rd_summary.csv`
- Best by gap: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_edge_rd_best_by_gap.csv`
- Gates: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_edge_rd_gates.csv`
- Package: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_multisequence_connected_edge_rd_package.json`
- Report: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_multisequence_connected_edge_rd_report.md`

## Results

- Decision: `edge_rd_table_ready_for_stage207_dp`.
- Windows: `bike-packing:00000:00024` and `parkour:00000:00024`.
- Coverage: `2` windows, `22` edges, `122` internal target rows, `12` gap4 chain edges, `28` connected transitions.
- Best gap4: `topk_keep0p2_q6`, edge total `1608215.0` bytes, DP incremental `888568.5833333334` bytes, residual `168918.75` bytes, corrected PSNR `25.43046561068816`, dPSNR `+5.249391228248168`.
- Best gap8: `topk_keep0p2_q6`, edge total `1838711.0` bytes, DP incremental `1119065.5` bytes, residual `399415.1666666667` bytes, corrected PSNR `24.86196786390912`, dPSNR `+5.709504375364108`.
- Best gap12: `topk_keep0p2_q6`, edge total `2070998.5` bytes, DP incremental `1351356.0` bytes, residual `631707.25` bytes, corrected PSNR `24.312590705743677`, dPSNR `+5.727019866033535`.
- Gates passed: Stage205 prereq, target metric rows ok, edge rows ok, gap coverage, payload counted nonzero, schedule metadata counted, positive edge headroom, Stage197 decoder contract.

## Decision

- Use this multi-sequence connected package for Stage207c DP oracle.
