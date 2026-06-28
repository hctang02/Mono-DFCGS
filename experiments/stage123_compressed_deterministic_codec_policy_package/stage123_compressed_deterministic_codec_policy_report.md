# Stage123 Compressed Deterministic Codec Policy Package

## Scope

- Packages selector, deterministic selected-index rule, side-info codec, and RD settings into one manifest.
- Selected indices are decoder-reproducible and are not transmitted.
- Side-info payload uses compressed deterministic value-only residuals.
- Residual values remain teacher-derived; this is not a residual value predictor.

## Policy

- policy: `compressed_deterministic_value_only_residual_codec_v1`
- selector: `strict_safe_endpoint_selector_v1`
- selected candidate: `endpoint_diff_baseline`
- index rule: `endpoint_diff_topk_v1`
- side-info codec: `compressed_deterministic_value_only_residual_sideinfo_v1`
- payload magic: `RSDZ`
- header bytes: `26`

## Settings

| role | setting | keep | bits | payload bytes | direct | amortized | PSNR | delta q6 | dRate entropy | dPSNR entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | q4_top20 | 0.2 | 4 | 28320.791667 | 0.133768 | 0.130054 | 20.689271 | 0.922302 | -0.004194 | -0.830532 |
| low_rate | q4_top10 | 0.1 | 4 | 15190.475000 | 0.121246 | 0.119247 | 19.738488 | -0.028480 | -0.016717 | -1.781315 |
| near_anchor | q5_top10 | 0.1 | 5 | 24809.950000 | 0.130420 | 0.127150 | 19.761048 | -0.005921 | -0.007543 | -1.758755 |
| anchor | q6_top10 | 0.1 | 6 | 29442.208333 | 0.134838 | 0.130956 | 19.766969 | 0.000000 | -0.003125 | -1.752834 |

## Decoder Contract

- keep count: `min(max(round(N * keep_fraction), 0), N)`
- score: `sum((right_attrs[0].float() - left_attrs[0].float()) ** 2, dim=-1)`
- selection: `select top keep_count largest scores, then sort selected indices ascending`
- forbidden decoder inputs: target dense anchor, target residual, target RGB, oracle task label, transmitted selected indices.

## Outputs

- policy JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage123_compressed_deterministic_codec_policy_package/stage123_compressed_deterministic_codec_policy.json`
- settings CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage123_compressed_deterministic_codec_policy_package/stage123_compressed_deterministic_codec_policy_settings.csv`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage123_compressed_deterministic_codec_policy_package/stage123_compressed_deterministic_codec_policy_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage123_compressed_deterministic_codec_policy_package/stage123_compressed_deterministic_codec_policy_report.md`
