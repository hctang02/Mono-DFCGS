# Stage200 GS Predictor Architecture Package

## Decision

- Decision: `primary_temporal_basis_refiner_v1_selected_for_stage201_smoke`.
- Primary architecture: `temporal_basis_gs_refiner_v1`.
- Smoke parameter count: `43501` for hidden96/global32 diagnostic instance.

## Candidates

| candidate | status | reason | next |
|---|---|---|---|
| temporal_basis_gs_refiner_v1 | selected_primary_for_stage201 | Adds endpoint-gated temporal basis, endpoint difference, absolute motion, and sequence-level GS statistics beyond the old per-Gaussian adapter MLP. | Implement Stage201 predictor-only smoke with no residual payload and q12 keyframes. |
| old_gaussian_anchor_dynamic_predictor | rejected | Stage198 showed q-bit changes and continued training do not repair this route. | Use only as a historical baseline, not the Stage201 architecture. |
| raw_streamsplat_runtime_rgb_dependency | not_primary_final_claim | Stage197 forbids raw target RGB as decoder input for the final GS compression claim. | Use checkpoint only as optional initialization/teacher if later needed. |
| latent_conditioned_temporal_refiner_v1 | deferred_to_stage203_204 | Residual/latent side-info is likely needed for target headroom but should be introduced after predictor-only smoke. | Define concrete latent/residual bitstream in Stage203. |

## Loss Contract

| loss | stage | source | decoder input required |
|---|---|---|---|
| anchor_attr_huber | Stage201+ | predicted GS anchor vs target dense anchor | no |
| render_rgb_mse_or_l1 | Stage201+ | rendered predicted GS vs target RGB | no |
| endpoint_identity | Stage201+ | t=0 equals left keyframe and t=1 equals right keyframe | no |
| residual_energy_for_codec_labels | Stage203+ | target dense anchor minus predictor anchor | no |
| rate_proxy | Stage203+ | estimated latent/residual entropy or selected attribute count | counted_payload_only |

## Audit

| audit | status | value | detail |
|---|---|---:|---|
| stage198_requirements_loaded | pass | 4 | edge_rd_headroom_before_selector_training;full_sequence_metrics_only_for_strong_claim;gs_native_residual_payload;predictor_only_gate_before_selector |
| stage199_manifest_ready | pass | 29204 | missing_count=0; gaps=[2, 4, 6, 8, 12, 16] |
| endpoint_identity | pass | 0.0 | t0=0.0; t1=0.0 |
| zero_init_linear_fallback | pass | 0.0 | zero-initialized final residual layer with endpoint-gated residual |
| stage197_decoder_contract | pass | 0 | target dense anchors and RGB are training/encoder-side only; payloads after Stage203 must be GS-native and counted |

## Stage201 Protocol

| item | value |
|---|---|
| stage201_input_manifest | /mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_tasks.csv |
| stage201_train_split | train |
| stage201_eval_split | eval |
| stage201_smoke_gaps | 4 8 |
| stage201_keyframe_codec | q12 |
| stage201_predictor | TemporalBasisGSRefiner(hidden_dim=192, global_dim=64, zero_init_residual=True) |
| stage201_payload | none; predictor-only; zero per-frame side-info |
| stage201_heavy_root | /data/hctang/tmp/opencode/mono_dfcgs_runs/stage201_predictor_only_smoke |
| stage201_acceptance_gate | stable rendered metrics; no broadcast warnings; endpoint identity preserved; final eval not worse than linear by more than 0.05 dB; reject architecture if no broader headroom in Stage202 |
| stage200_parameter_count_hidden96_global32 | 43501 |

## Decoder Contract

- Allowed: decoded/transmitted left/right GS keyframes, normalized time from schedule metadata, shared refiner weights, and optional counted GS-native latent/residual payloads in later stages.
- Training/encoder-only: target dense anchor and target RGB render loss.
- Forbidden: target dense anchor as decoder input, target RGB/image residual, and oracle schedule/quality labels.

## Outputs

- candidates: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage200_gs_predictor_architecture_package/stage200_architecture_candidates.csv`
- loss contract: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage200_gs_predictor_architecture_package/stage200_loss_contract.csv`
- decoder audit: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage200_gs_predictor_architecture_package/stage200_decoder_contract_audit.csv`
- Stage201 protocol: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage200_gs_predictor_architecture_package/stage200_stage201_smoke_protocol.csv`
- architecture JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage200_gs_predictor_architecture_package/stage200_primary_architecture_contract.json`
- package: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage200_gs_predictor_architecture_package/stage200_gs_predictor_architecture_package.json`
