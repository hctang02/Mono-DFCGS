# Stage206 Edge RD Table

Date: 2026-07-02

## Goal

Build an edge-level RD table for Stage207 DP schedule optimization using GS-native predictor plus counted residual payloads.

## Plan

- Sample eval edges for gaps `4,8,12` and include every manifest target inside each selected edge.
- Reuse Stage204/205 predictor-plus-top-k GS residual evaluation for q6 keep fractions `0.05,0.10,0.20`.
- Measure exact endpoint q12 keyframe bytes with `encode_anchor_bitstream(..., bits=12, payload_encoding=bitpack)`.
- Count residual payload bytes from `len(payload)` and record explicit per-edge schedule metadata bytes.
- Emit both `edge_total_bytes_once` and `dp_incremental_bytes` so Stage207 can count shared keyframes correctly.

## Success Criteria

- Stage205 prerequisite decision is `fixed_gap_predictive_codec_positive_headroom`.
- Target metric rows and edge rows have no errors.
- Every selected edge has complete intermediate target coverage from the manifest.
- Keyframe, residual, and schedule metadata bytes are explicit and non-hidden.
- Each tested gap has a positive edge-level dPSNR headroom setting.

## Execution

- Syntax check: `PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage206_edge_rd_table.py`
- Pre-run GPU check: `nvidia-smi` at 2026-07-02 00:39:50; GPU1 was idle.
- Command: `CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage206_edge_rd_table.py --device cuda --max_edges_per_gap 2`

## Outputs

- Output root: `experiments/stage206_edge_rd_table/`
- Selected edges: `experiments/stage206_edge_rd_table/stage206_selected_edges.csv`
- Target metric rows: `experiments/stage206_edge_rd_table/stage206_target_metric_rows.csv`
- Edge RD rows: `experiments/stage206_edge_rd_table/stage206_edge_rd_rows.csv`
- Summary: `experiments/stage206_edge_rd_table/stage206_edge_rd_summary.csv`
- Best by gap: `experiments/stage206_edge_rd_table/stage206_edge_rd_best_by_gap.csv`
- Gates: `experiments/stage206_edge_rd_table/stage206_edge_rd_gates.csv`
- Package: `experiments/stage206_edge_rd_table/stage206_edge_rd_table_package.json`
- Report: `experiments/stage206_edge_rd_table/stage206_edge_rd_table_report.md`

## Results

- Decision: `edge_rd_table_ready_for_stage207_dp`.
- Scope: sampled edge-level RD preflight for Stage207 DP, not final full-sequence RD.
- Selected `6` edges and `37` internal target rows over gaps `4,8,12`; tested q6 top-k residual keep fractions `0.05,0.10,0.20`.
- Accounting fields are explicit: `edge_total_bytes_once = left_keyframe_bytes + right_keyframe_bytes + residual_payload_bytes + schedule_metadata_bytes`; `dp_incremental_bytes = right_keyframe_bytes + residual_payload_bytes + schedule_metadata_bytes` for Stage207 path accounting; schedule metadata is provisionally counted as `2` bytes per edge.
- Best gap4: `topk_keep0p2_q6`, mean edge total `1604771.0` bytes, mean DP incremental `885123.0` bytes, mean residual `165471.0` bytes, corrected PSNR `26.59671835498372`, dPSNR `+5.301434075900853`.
- Best gap8: `topk_keep0p2_q6`, mean edge total `1701646.5` bytes, mean DP incremental `981999.5` bytes, mean residual `262346.5` bytes, corrected PSNR `22.501153480651734`, dPSNR `+4.267633006441421`.
- Best gap12: `topk_keep0p2_q6`, mean edge total `2052070.0` bytes, mean DP incremental `1332423.5` bytes, mean residual `612781.0` bytes, corrected PSNR `26.07283417481182`, dPSNR `+4.164336484774056`.
- Gates passed: Stage205 prereq, target metric rows ok, edge rows ok, gap coverage, counted nonzero keyframe/residual payloads, schedule metadata counted, each-gap positive edge headroom, Stage197 decoder contract.

## Decision

- Proceed to Stage207 DP oracle schedule using Stage206 edge RD rows. Stage207 must still count initial left keyframes once per path and must not treat sampled Stage206 as final full-sequence RD.
