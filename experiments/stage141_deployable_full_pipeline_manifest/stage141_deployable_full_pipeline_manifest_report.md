# Stage141 Deployable Full-Pipeline Manifest

## Final Policy

- manifest: `deployable_render_aware_scaled_adapter_delta_full_pipeline_v1`
- policy: `deployable_render_aware_scaled_adapter_delta_selected_residual_codec_v1`
- primary: `q4_top20` scale `0.75`
- low-rate: `q4_top10` scale `0.75`

## RD Summary

| role | setting | scale | rate | PSNR | delta base | residual bytes | index bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| primary | q4_top20 | 0.75 | 0.117298 | 19.022110 | 0.071307 | 0 | 0 |
| low_rate | q4_top10 | 0.75 | 0.117298 | 18.997891 | 0.047088 | 0 | 0 |

## Decoder Contract

- Inputs: left anchor, right anchor, normalized time, pre-shared Stage65 adapter, policy keep fraction, policy adapter-delta scale.
- Forbidden: target dense anchor, target residual, target RGB, oracle labels, transmitted selected indices, transmitted residual values, teacher residual side-info.
- Per-frame residual payload bytes: 0.
- Per-frame selected-index payload bytes: 0.
- Policy scale payload bytes: 0.

## Checklist

| item | status | evidence |
|---|---|---|
| no_teacher_decoder_input | pass | teacher residual side-info is listed as forbidden input |
| no_target_dense_anchor_decoder_input | pass | target_dense_anchor is listed as forbidden input |
| no_target_rgb_decoder_input | pass | target_rgb is listed as forbidden input; used only for offline validation |
| zero_residual_payload | pass | primary transmitted_residual_payload_bytes=0 |
| zero_selected_index_payload | pass | primary transmitted_selected_index_bytes=0 |
| policy_scale_not_sideinfo | pass | adapter_delta_scale is a fixed policy constant with zero per-frame payload |
| deterministic_index_selection | pass | endpoint_diff_topk_v1 |
| pre_shared_adapter_declared | pass | /data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors |
| mlp_rejected | pass | Stage140 rows=18; MLP rejected due to Stage129/134 rendered regression |

## Outputs

- manifest JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage141_deployable_full_pipeline_manifest/stage141_deployable_full_pipeline_manifest.json`
- checklist CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage141_deployable_full_pipeline_manifest/stage141_deployable_full_pipeline_checklist.csv`
- package JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage141_deployable_full_pipeline_manifest/stage141_deployable_full_pipeline_manifest_package.json`
- report Markdown: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage141_deployable_full_pipeline_manifest/stage141_deployable_full_pipeline_manifest_report.md`
