# Stage128 Predictor Codec Integration Package

## Scope

- Packages Stage127 selected residual MLP checkpoints into a predictor-only codec manifest.
- Residual values and selected indices are not transmitted per frame.
- Checkpoints remain outside git and are referenced by path.

## Settings

| setting | role | keep | hidden | eval reduction | ckpt exists | load ok | residual bytes | index bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| q4_top20 | primary | 0.2 | 128 | 0.088083 | 1 | 1 | 0 | 0 |
| q4_top10 | low_rate | 0.1 | 128 | 0.102933 | 1 | 1 | 0 | 0 |

## Decoder Contract

- Recompute endpoint-diff selected indices from left/right anchors.
- Build MLP features from left/right/base attrs and normalized time.
- Decode residual values with pre-shared MLP weights and Stage126 normalization stats.
- Do not use target dense anchors, target residuals, target RGB, oracle labels, transmitted indices, or transmitted residual values.

## Outputs

- policy JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage128_predictor_codec_integration_package/stage128_predictor_codec_integration_policy.json`
- settings CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage128_predictor_codec_integration_package/stage128_predictor_codec_integration_settings.csv`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage128_predictor_codec_integration_package/stage128_predictor_codec_integration_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage128_predictor_codec_integration_package/stage128_predictor_codec_integration_report.md`
