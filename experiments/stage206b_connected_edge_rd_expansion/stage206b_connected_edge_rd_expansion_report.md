# Stage206b Connected Edge RD Expansion

## Decision

- Decision: `edge_rd_table_ready_for_stage207_dp`.
- Windows: `1`; edges: `11`; target rows: `61`; settings: `3`.
- Scope: small connected-window expansion for Stage207 rerun, not final full-sequence RD.

## Windows

| window | sequence | start | end | edges | targets | connected transitions |
|---|---|---:|---:|---:|---:|---:|
| bike-packing:00000:00024 | bike-packing | 0 | 24 | 11 | 61 | 14 |

## Best By Gap

| gap | best setting | keep | edge total bytes | DP incremental bytes | residual bytes | corrected PSNR | dPSNR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | topk_keep0p2_q6 | 0.2 | 1605417.167 | 885768.667 | 166117.667 | 26.031264 | 4.986392 |
| 8 | topk_keep0p2_q6 | 0.2 | 1833385.667 | 1113739.000 | 394089.333 | 25.821075 | 5.676789 |
| 12 | topk_keep0p2_q6 | 0.2 | 2063898.000 | 1344249.500 | 624597.500 | 25.402215 | 5.768187 |

## Gates

| gate | status | value | threshold | detail |
|---|---|---|---|---|
| stage205_prereq | pass | fixed_gap_predictive_codec_positive_headroom | fixed_gap_predictive_codec_positive_headroom | /mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage205_fixed_gap_predictive_codec_validation/stage205_fixed_gap_predictive_codec_validation_package.json |
| target_metric_rows_ok | pass | 0 | 0 | shape-mismatched metrics are errors |
| edge_rows_ok | pass | 0 | 0 |  |
| gap_coverage | pass | 0 | 0 missing gaps |  |
| payload_counted_nonzero | pass | residual=46862.0;right_keyframe=719646.0 | >0 residual and keyframe bytes | keyframe bytes use encode_anchor_bitstream; residual bytes use len(payload) |
| schedule_metadata_counted | pass | 2 | >=0 bytes explicitly recorded per edge | Stage207 may replace provisional syntax but cannot hide metadata bytes |
| each_gap_positive_edge_headroom | pass | 4.986391989862658 | > 0.5 dB for every gap |  |
| stage197_decoder_contract | pass | 0 | no target dense/RGB decoder input | decoder uses transmitted q-keyframes, normalized time, schedule metadata, and counted GS residual payload |

## Outputs

- windows: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage206b_connected_edge_rd_expansion/stage206b_connected_windows.csv`
- selected edges: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage206b_connected_edge_rd_expansion/stage206b_selected_edges.csv`
- target rows: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage206b_connected_edge_rd_expansion/stage206b_target_metric_rows.csv`
- edge RD rows: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage206b_connected_edge_rd_expansion/stage206b_edge_rd_rows.csv`
- summary: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage206b_connected_edge_rd_expansion/stage206b_edge_rd_summary.csv`
- best by gap: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage206b_connected_edge_rd_expansion/stage206b_edge_rd_best_by_gap.csv`
- gates: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage206b_connected_edge_rd_expansion/stage206b_edge_rd_gates.csv`
- package: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage206b_connected_edge_rd_expansion/stage206b_connected_edge_rd_expansion_package.json`
