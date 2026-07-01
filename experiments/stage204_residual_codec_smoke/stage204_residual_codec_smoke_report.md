# Stage204 Residual Codec Smoke

## Decision

- Decision: `residual_codec_smoke_positive_headroom`.
- Tasks: `12`; settings: `3`.
- Best setting: `topk_keep0p2_q6`.
- Best mean dPSNR vs base: `5.553101207002054` dB.

## Summary

| setting | keep | q | tasks | payload bytes | base PSNR | corrected PSNR | dPSNR | anchor MSE reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| topk_keep0p05_q6 | 0.05 | 6 | 12 | 15679.583 | 19.999034 | 22.394628 | 2.395594 | 0.510236 |
| topk_keep0p1_q6 | 0.1 | 6 | 12 | 29836.750 | 19.999034 | 23.804630 | 3.805596 | 0.688443 |
| topk_keep0p2_q6 | 0.2 | 6 | 12 | 55761.833 | 19.999034 | 25.552135 | 5.553101 | 0.843555 |

## Gates

| gate | status | value | threshold | detail |
|---|---|---:|---|---|
| metric_rows_ok | pass | 0 | 0 | all rendered metrics must use explicit target-shape alignment |
| payload_counted_nonzero | pass | 13790.0 | >0 bytes and counted as len(payload) | residual payload bytes include headers, metadata, indices/deltas, q values, and zlib bytes |
| residual_anchor_mse_reduction_positive | pass | 0.8435545876871436 | >0 | best_setting=topk_keep0p2_q6 |
| residual_render_headroom_positive | pass | 5.553101207002054 | > 0.5 dB vs base | best_setting=topk_keep0p2_q6 |
| stage197_decoder_contract | pass | 0 | no target dense/RGB decoder input | decoder uses base GS plus transmitted counted GS residual payload |

## Decoder Contract

- Encoder uses target dense anchors only to produce GS residual payloads.
- Decoder uses predictor/base GS plus transmitted counted GS-native residual payloads.
- No RGB/image residual or target dense anchor is used as a decoder input.

## Outputs

- selected tasks: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage204_residual_codec_smoke/stage204_selected_tasks.csv`
- rows: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage204_residual_codec_smoke/stage204_residual_codec_smoke_rows.csv`
- summary: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage204_residual_codec_smoke/stage204_residual_codec_smoke_summary.csv`
- gates: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage204_residual_codec_smoke/stage204_residual_codec_smoke_gates.csv`
- package: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage204_residual_codec_smoke/stage204_residual_codec_smoke_package.json`
