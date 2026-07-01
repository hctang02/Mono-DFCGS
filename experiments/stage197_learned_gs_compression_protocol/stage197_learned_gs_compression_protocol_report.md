# Stage197 Learned GS Compression Protocol

## Decision

- Decision: `primary_gs_native_predictive_codec_protocol_defined`.
- Primary runtime decoder uses transmitted GS keyframes, schedule, time, shared GS codec weights, and counted GS-native latent/residual payloads only.
- RGB/image residual post-processing is rejected as a final method.
- StreamSplat checkpoint may initialize or supervise modules, but raw RGB-dependent StreamSplat runtime is not the primary final codec claim.

## Contract

| scope | item | status | rationale |
|---|---|---|---|
| primary_runtime_decoder | transmitted_q_keyframe_gs_bitstreams | allowed | Keyframes are the compressed GS stream anchors and every keyframe byte is counted. |
| primary_runtime_decoder | transmitted_schedule_metadata | allowed | Decoder receives keyframe indices/segment metadata; selector features are not needed at decode time. |
| primary_runtime_decoder | normalized_time | allowed | Time within a keyframe segment is deterministic from transmitted schedule. |
| primary_runtime_decoder | shared_gs_predictor_refiner_weights | allowed | Codec model weights are shared once as method parameters, not per-video hidden side information. |
| primary_runtime_decoder | transmitted_gs_latent_or_residual_bitstreams | allowed_counted | Residual/latent payloads are GS-native and must be included in total rate. |
| primary_runtime_decoder | target_dense_anchor | forbidden | Target anchors may supervise training or produce encoder-side residual payloads, but cannot be a decoder input. |
| primary_runtime_decoder | target_rgb_or_image_residual | forbidden | User rejected image-domain post-processing; final method must remain GS compression. |
| primary_runtime_decoder | oracle_schedule_or_quality_labels | forbidden | Oracle labels can train/evaluate but must be represented by transmitted schedule or learned encoder decisions at inference. |
| encoder_training | target_dense_anchor_and_rgb | allowed_training_only | Encoder/training can use targets to learn predictors, residual payloads, and selector labels. |
| streamsplat_checkpoint | frozen_streamsplat_as_initialization_or_teacher | allowed_optional | The checkpoint can initialize or supervise GS-native modules, but primary runtime decoder should not require raw target RGB. |
| streamsplat_checkpoint | full_runtime_streamsplat_raw_rgb_dependency | not_primary_final_claim | Using raw video/image inputs at decoder would undermine a GS-compression claim unless those inputs are explicitly transmitted and counted. |

## Modules

| module | runtime side | outputs | rate accounting |
|---|---|---|---|
| keyframe_codec | encoder_and_decoder | keyframe bitstreams and decoded keyframe GS | all keyframe bitstream bytes counted; schedule metadata counted |
| gs_predictor_refiner | decoder | predicted or refined intermediate GS | shared weights declared as codec parameters; per-video latent bytes counted if used |
| gs_latent_residual_codec | encoder_and_decoder | GS-native correction payload and corrected GS | all residual/latent bytes counted in total RD |
| encoder_selector_and_budget_allocator | encoder_only | keyframe schedule and residual budget decisions | only transmitted schedule and payloads counted; encoder-only features are not decoder inputs |

## Stage Gates

| stage | name | goal | gate |
|---:|---|---|---|
| 198 | prior_predictor_training_audit | document why old adapter training failed | old adapter route rejected |
| 199 | learned_gs_training_manifest | build multi-gap train/eval task manifest | complete lightweight references and split audit |
| 200 | gs_predictor_architecture_package | define new predictor/refiner candidates | architecture and loss contract ready |
| 201 | predictor_only_smoke | test new predictor without residual/selector | > old adapter and stable render metrics |
| 202 | predictor_only_broader_validation | validate predictor over multiple gaps | broad improvement or architecture rejection |
| 203 | gs_latent_residual_codec_design | define GS-native residual/latent codec | codec payload is decodable and counted |
| 204 | residual_codec_smoke | test predictor plus GS residual on small set | quality headroom near target |
| 205 | fixed_gap_predictive_codec_validation | validate predictor+residual without selector | fixed-gap quality/rate headroom |
| 206 | edge_rd_table | measure segment-level costs for schedule optimization | edge coverage complete for selected gaps |
| 207 | dp_oracle_schedule | compute oracle keyframe schedules | oracle beats fixed baselines |
| 208 | selector_training_data | convert oracle schedules to encoder labels/features | feature-source audit passes |
| 209 | encoder_selector_training | train deployable schedule predictor | learned selector approaches oracle |
| 210 | selector_residual_budget_joint_training | jointly tune schedule and residual allocation | joint RD improves selector-only |
| 211 | full_sequence_measured_rd | compare final adaptive against fixed gaps | full-sequence RD-quality complete |
| 212 | ablation_package | prove predictor/residual/selector contributions | paper-facing ablations complete |
| 213 | subjective_visual_export | export videos/contact sheets outside git | visual evidence paths recorded |

## Outputs

- Contract CSV: `experiments/stage197_learned_gs_compression_protocol/stage197_decoder_contract.csv`
- Module CSV: `experiments/stage197_learned_gs_compression_protocol/stage197_module_contract.csv`
- Stage plan CSV: `experiments/stage197_learned_gs_compression_protocol/stage197_stage_plan.csv`
