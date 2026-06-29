# Stage132 Deployable Predictor Policy Package

## Policy

- policy: `deployable_adapter_delta_selected_residual_codec_v1`
- status: `current_best_no_teacher_deployable`
- primary setting: `q4_top20`
- optional low-rate setting: `q4_top10`
- adapter checkpoint exists: `1`

## Settings

| role | setting | keep | rate | PSNR | delta base | delta full | residual bytes | index bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| primary | q4_top20 | 0.2 | 0.117298 | 19.010259 | 0.059456 | -0.177384 | 0 | 0 |
| low_rate | q4_top10 | 0.1 | 0.117298 | 18.994813 | 0.044010 | -0.192830 | 0 | 0 |

## Decoder Contract

- Inputs: left anchor, right anchor, normalized time, pre-shared Stage65 adapter, policy keep fraction.
- Forbidden: target dense anchor, target residual, target RGB, oracle labels, transmitted selected indices, transmitted residual values.
- Per-frame residual payload bytes: 0.
- Per-frame selected-index payload bytes: 0.

## Outputs

- policy JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage132_deployable_predictor_policy_package/stage132_deployable_predictor_policy.json`
- settings CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage132_deployable_predictor_policy_package/stage132_deployable_predictor_policy_settings.csv`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage132_deployable_predictor_policy_package/stage132_deployable_predictor_policy_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage132_deployable_predictor_policy_package/stage132_deployable_predictor_policy_report.md`
