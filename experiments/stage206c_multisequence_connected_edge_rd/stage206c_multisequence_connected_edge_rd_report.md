# Stage206c Connected Edge RD Expansion

## Decision

- Decision: `edge_rd_table_ready_for_stage207_dp`.
- Windows: `2`; edges: `22`; target rows: `122`; settings: `3`.
- Scope: small connected-window expansion for Stage207 rerun, not final full-sequence RD.

## Windows

| window | sequence | start | end | edges | targets | connected transitions |
|---|---|---:|---:|---:|---:|---:|
| bike-packing:00000:00024 | bike-packing | 0 | 24 | 11 | 61 | 14 |
| parkour:00000:00024 | parkour | 0 | 24 | 11 | 61 | 14 |

## Best By Gap

| gap | best setting | keep | edge total bytes | DP incremental bytes | residual bytes | corrected PSNR | dPSNR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | topk_keep0p2_q6 | 0.2 | 1608215.000 | 888568.583 | 168918.750 | 25.430466 | 5.249391 |
| 8 | topk_keep0p2_q6 | 0.2 | 1838711.000 | 1119065.500 | 399415.167 | 24.861968 | 5.709504 |
| 12 | topk_keep0p2_q6 | 0.2 | 2070998.500 | 1351356.000 | 631707.250 | 24.312591 | 5.727020 |

## Gates

| gate | status | value | threshold | detail |
|---|---|---|---|---|
| stage205_prereq | pass | fixed_gap_predictive_codec_positive_headroom | fixed_gap_predictive_codec_positive_headroom | /mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage205_fixed_gap_predictive_codec_validation/stage205_fixed_gap_predictive_codec_validation_package.json |
| target_metric_rows_ok | pass | 0 | 0 | shape-mismatched metrics are errors |
| edge_rows_ok | pass | 0 | 0 |  |
| gap_coverage | pass | 0 | 0 missing gaps |  |
| payload_counted_nonzero | pass | residual=46862.0;right_keyframe=719637.0 | >0 residual and keyframe bytes | keyframe bytes use encode_anchor_bitstream; residual bytes use len(payload) |
| schedule_metadata_counted | pass | 2 | >=0 bytes explicitly recorded per edge | Stage207 may replace provisional syntax but cannot hide metadata bytes |
| each_gap_positive_edge_headroom | pass | 5.249391228248168 | > 0.5 dB for every gap |  |
| stage197_decoder_contract | pass | 0 | no target dense/RGB decoder input | decoder uses transmitted q-keyframes, normalized time, schedule metadata, and counted GS residual payload |

## Outputs

- windows: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_connected_windows.csv`
- selected edges: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_selected_edges.csv`
- target rows: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_target_metric_rows.csv`
- edge RD rows: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_edge_rd_rows.csv`
- summary: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_edge_rd_summary.csv`
- best by gap: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_edge_rd_best_by_gap.csv`
- gates: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_edge_rd_gates.csv`
- package: `experiments/stage206c_multisequence_connected_edge_rd/stage206c_multisequence_connected_edge_rd_package.json`
