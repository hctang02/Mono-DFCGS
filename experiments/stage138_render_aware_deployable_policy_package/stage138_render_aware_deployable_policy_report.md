# Stage138 Render-Aware Deployable Policy Package

## Policy

- policy: `deployable_render_aware_scaled_adapter_delta_selected_residual_codec_v1`
- status: `current_best_no_teacher_deployable_render_aware`
- replaces: `deployable_adapter_delta_selected_residual_codec_v1`
- primary setting: `q4_top20` scale `0.75`
- optional low-rate setting: `q4_top10` scale `0.75`
- adapter checkpoint exists: `1`

## Settings

| role | setting | keep | scale | rate | PSNR | delta base | delta full | delta Stage132 | residual bytes | index bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | q4_top20 | 0.2 | 0.75 | 0.117298 | 19.022110 | 0.071307 | -0.165534 | 0.011850 | 0 | 0 |
| low_rate | q4_top10 | 0.1 | 0.75 | 0.117298 | 18.997891 | 0.047088 | -0.189753 | 0.003077 | 0 | 0 |

## Decoder Contract

- Inputs: left anchor, right anchor, normalized time, pre-shared Stage65 adapter, policy keep fraction, policy adapter-delta scale.
- Forbidden: target dense anchor, target residual, target RGB, oracle labels, transmitted selected indices, transmitted residual values, teacher residual side-info.
- Per-frame residual payload bytes: 0.
- Per-frame selected-index payload bytes: 0.

## Outputs

- policy JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage138_render_aware_deployable_policy_package/stage138_render_aware_deployable_policy.json`
- settings CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage138_render_aware_deployable_policy_package/stage138_render_aware_deployable_policy_settings.csv`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage138_render_aware_deployable_policy_package/stage138_render_aware_deployable_policy_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage138_render_aware_deployable_policy_package/stage138_render_aware_deployable_policy_report.md`
