# Stage205 Fixed-Gap Predictive Codec Validation

## Decision

- Decision: `fixed_gap_predictive_codec_positive_headroom`.
- Tasks: `24`; settings: `3`.
- Scope: sampled fixed-gap validation, not full-sequence RD.

## Best By Gap

| gap | best setting | keep | payload bytes | MiB/intermediate | base PSNR | corrected PSNR | dPSNR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | topk_keep0p2_q6 | 0.2 | 56283.000 | 0.053676 | 20.599421 | 25.301133 | 4.701712 |
| 8 | topk_keep0p2_q6 | 0.2 | 58359.875 | 0.055656 | 17.667284 | 23.694548 | 6.027264 |
| 12 | topk_keep0p2_q6 | 0.2 | 57028.500 | 0.054387 | 19.195595 | 24.991768 | 5.796173 |

## Gates

| gate | status | value | threshold | detail |
|---|---|---:|---|---|
| metric_rows_ok | pass | 0 | 0 | shape-mismatched metrics are errors |
| payload_counted_nonzero | pass | 15353.0 | >0 bytes | payload_bytes are exact len(payload) |
| gap_coverage | pass | 0 | 0 missing gaps |  |
| each_gap_positive_headroom | pass | 4.701712017960398 | > 0.5 dB for every gap |  |
| stage197_decoder_contract | pass | 0 | no target dense/RGB decoder input | decoder uses base GS plus transmitted counted GS residual payload |

## Outputs

- selected tasks: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage205_fixed_gap_predictive_codec_validation/stage205_selected_tasks.csv`
- rows: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage205_fixed_gap_predictive_codec_validation/stage205_fixed_gap_predictive_rows.csv`
- summary: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage205_fixed_gap_predictive_codec_validation/stage205_fixed_gap_predictive_summary.csv`
- best by gap: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage205_fixed_gap_predictive_codec_validation/stage205_fixed_gap_predictive_best_by_gap.csv`
- gates: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage205_fixed_gap_predictive_codec_validation/stage205_fixed_gap_predictive_gates.csv`
- package: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage205_fixed_gap_predictive_codec_validation/stage205_fixed_gap_predictive_codec_validation_package.json`
