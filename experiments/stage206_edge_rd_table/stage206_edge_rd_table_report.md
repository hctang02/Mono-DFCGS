# Stage206 Edge RD Table

## Decision

- Decision: `edge_rd_table_ready_for_stage207_dp`.
- Selected edges: `6`; target rows: `37`; settings: `3`.
- Scope: sampled edge-level RD preflight for Stage207 DP, not final full-sequence RD.

## Best By Gap

| gap | best setting | keep | edge total bytes | DP incremental bytes | residual bytes | corrected PSNR | dPSNR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | topk_keep0p2_q6 | 0.2 | 1604771.000 | 885123.000 | 165471.000 | 26.596718 | 5.301434 |
| 8 | topk_keep0p2_q6 | 0.2 | 1701646.500 | 981999.500 | 262346.500 | 22.501153 | 4.267633 |
| 12 | topk_keep0p2_q6 | 0.2 | 2052070.000 | 1332423.500 | 612781.000 | 26.072834 | 4.164336 |

## Accounting

- `edge_total_bytes_once = left_keyframe_bytes + right_keyframe_bytes + residual_payload_bytes + schedule_metadata_bytes`.
- `dp_incremental_bytes = right_keyframe_bytes + residual_payload_bytes + schedule_metadata_bytes`; Stage207 must add the initial left keyframe once per path.
- Residual payload bytes are exact `len(payload)` from the GS-native residual codec.
- Keyframe bytes are exact `encode_anchor_bitstream(..., q12, bitpack)` lengths for endpoint anchors.

## Gates

| gate | status | value | threshold | detail |
|---|---|---|---|---|
| stage205_prereq | pass | fixed_gap_predictive_codec_positive_headroom | fixed_gap_predictive_codec_positive_headroom | /mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage205_fixed_gap_predictive_codec_validation/stage205_fixed_gap_predictive_codec_validation_package.json |
| target_metric_rows_ok | pass | 0 | 0 | shape-mismatched metrics are errors |
| edge_rows_ok | pass | 0 | 0 |  |
| gap_coverage | pass | 0 | 0 missing gaps |  |
| payload_counted_nonzero | pass | residual=32320.0;right_keyframe=719639.0 | >0 residual and keyframe bytes | keyframe bytes use encode_anchor_bitstream; residual bytes use len(payload) |
| schedule_metadata_counted | pass | 2 | >=0 bytes explicitly recorded per edge | Stage207 may replace provisional syntax but cannot hide metadata bytes |
| each_gap_positive_edge_headroom | pass | 4.164336484774056 | > 0.5 dB for every gap |  |
| stage197_decoder_contract | pass | 0 | no target dense/RGB decoder input | decoder uses transmitted q-keyframes, normalized time, schedule metadata, and counted GS residual payload |

## Outputs

- selected edges: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage206_edge_rd_table/stage206_selected_edges.csv`
- target rows: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage206_edge_rd_table/stage206_target_metric_rows.csv`
- edge RD rows: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage206_edge_rd_table/stage206_edge_rd_rows.csv`
- summary: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage206_edge_rd_table/stage206_edge_rd_summary.csv`
- best by gap: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage206_edge_rd_table/stage206_edge_rd_best_by_gap.csv`
- gates: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage206_edge_rd_table/stage206_edge_rd_gates.csv`
- package: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage206_edge_rd_table/stage206_edge_rd_table_package.json`
