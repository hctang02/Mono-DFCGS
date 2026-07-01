# Stage203 GS Latent/Residual Codec Design

## Decision

- Decision: `gs_attr_topk_residual_entropy_v1_selected_for_stage204_smoke`.
- Primary codec: `gs_attr_topk_residual_entropy_v1`.
- Payload bytes are counted with `len(payload)`.

## Candidates

| codec | status | reason | next |
|---|---|---|---|
| gs_attr_topk_residual_entropy_v1 | selected_primary_for_stage204 | Highest practical headroom because encoder can select target-residual top-k GS attributes while payload remains GS-native and decode-capable. | Stage204 smoke on real Stage199 tasks with q6 keep sweeps and rendered metrics. |
| gs_attr_deterministic_index_residual_entropy_v1 | low_rate_ablation | Lower rate because selected indices are deterministic, but likely lower quality if index rule misses target residual energy. | Use as Stage204/205 low-rate ablation after primary top-k smoke. |
| learned_gs_latent_residual_v1 | deferred | Potentially better RD but requires training after Stage204 verifies residual headroom. | Revisit after fixed residual codec smoke shows enough quality headroom. |
| rgb_image_residual_postprocess | rejected_final_method | User and Stage197 rejected RGB/image residual post-processing as final method. | Do not use as final method; at most historical upper-bound context. |

## Toy Roundtrips

| codec | status | payload bytes | MSE before | MSE after | reduction |
|---|---|---:|---:|---:|---:|
| gs_attr_topk_residual_entropy_v1 | pass | 246 | 0.00691959 | 0.00083170 | 0.879805 |
| gs_attr_deterministic_index_residual_entropy_v1 | pass | 217 | 0.00795055 | 0.00082503 | 0.896230 |

## Audit

| audit | status | value | detail |
|---|---|---:|---|
| stage202_predictor_headroom_context | pass | 0.0006680372369380905 | predictor_only_broader_training_headroom_not_observed |
| primary_codec_roundtrip | pass | 0.8798047118685358 | encode_topk_residual_sideinfo_entropy/decode_residual_sideinfo_entropy |
| deterministic_codec_roundtrip | pass | 0.8962303087335681 | indices are decoder-known and not hidden side-info |
| stage197_decoder_contract | pass | 0 | decoder uses predictor base GS plus transmitted counted GS-native residual payload |
| image_residual_rejected | pass | 0 | RGB/image residual appears only as rejected candidate |

## Stage204 Protocol

| item | value |
|---|---|
| stage204_base | linear_or_zero_init_TemporalBasisGSRefiner predictor base from Stage201/202 |
| stage204_task_manifest | experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_tasks.csv |
| stage204_gaps | 4 8 |
| stage204_keyframe_codec | q12 |
| stage204_primary_codec | gs_attr_topk_residual_entropy_v1 |
| stage204_settings | side_bits=6; keep_fraction=0.05,0.10,0.20; zlib_level=9 |
| stage204_metrics | rendered PSNR plus payload_bytes and MSE reduction; discard any shape-mismatched metrics |
| stage204_decoder_inputs | predictor_base_gs plus transmitted counted GS residual payload only |

## Decoder Contract

- Encoder may use target dense anchors to form GS residual payloads.
- Decoder uses predictor/base GS plus transmitted counted GS-native residual payloads.
- Target dense anchors, target RGB/image residuals, and oracle quality labels are not decoder inputs.

## Outputs

- candidates: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage203_gs_latent_residual_codec_design/stage203_codec_candidates.csv`
- roundtrips: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage203_gs_latent_residual_codec_design/stage203_codec_toy_roundtrips.csv`
- rate rules: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage203_gs_latent_residual_codec_design/stage203_rate_accounting_rules.csv`
- audit: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage203_gs_latent_residual_codec_design/stage203_decoder_contract_audit.csv`
- Stage204 protocol: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage203_gs_latent_residual_codec_design/stage203_stage204_smoke_protocol.csv`
- primary contract: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage203_gs_latent_residual_codec_design/stage203_primary_codec_contract.json`
- package: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage203_gs_latent_residual_codec_design/stage203_gs_latent_residual_codec_design_package.json`
