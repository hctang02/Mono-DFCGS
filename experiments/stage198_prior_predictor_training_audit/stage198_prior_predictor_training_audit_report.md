# Stage198 Prior Predictor Training Audit

## Decision

- Decision: `old_adapter_route_rejected_new_predictor_required`.
- The old `GaussianAnchorDynamicPredictor` / Stage65 adapter route should not be continued unchanged.
- Stage201+ must test a new predictor/refiner architecture and a GS-native residual codec before selector training.

## Evidence

| stage | evidence | gap | metric | value | interpretation |
|---:|---|---:|---|---:|---|
| 78 | old_adapter_middle_psnr | 4 | mean_middle_psnr | 18.256196 | Old adapter gives only small gain over linear and remains far below recovery targets. |
| 78 | old_adapter_middle_psnr | 8 | mean_middle_psnr | 17.069694 | Old adapter gives only small gain over linear and remains far below recovery targets. |
| 143 | qbit_sensitivity_adapter | 4 | float32_minus_q12_middle_psnr | 0.000049 | Raising anchor q-bit does not repair the old adapter. |
| 143 | dense_direct_vs_adapter_ceiling | 4 | dense_direct_minus_adapter_middle_psnr | 11.494322 | The old adapter is the bottleneck, not renderer/data loading. |
| 143 | qbit_sensitivity_adapter | 8 | float32_minus_q12_middle_psnr | -0.000017 | Raising anchor q-bit does not repair the old adapter. |
| 143 | dense_direct_vs_adapter_ceiling | 8 | dense_direct_minus_adapter_middle_psnr | 12.677632 | The old adapter is the bottleneck, not renderer/data loading. |
| 145 | training_best_gain | 4,8 | best_mean_psnr_minus_initial | 0.011427 | Training gain is tiny or absent under the old architecture/objective. |
| 145 | training_final_change | 4,8 | final_mean_psnr_minus_initial | 0.011427 | Longer continuation did not provide reliable improvement. |
| 146 | training_best_gain | 4,8 | best_mean_psnr_minus_initial | 0.000000 | Training gain is tiny or absent under the old architecture/objective. |
| 146 | training_final_change | 4,8 | final_mean_psnr_minus_initial | -0.019897 | Longer continuation did not provide reliable improvement. |
| 154 | original_streamsplat_base | 4 | mean_psnr | 22.064218 | StreamSplat base is more plausible than old adapter but still not enough without GS residual side-info. |
| 154 | original_streamsplat_base | 8 | mean_psnr | 20.337275 | StreamSplat base is more plausible than old adapter but still not enough without GS residual side-info. |
| 157 | stage158_residual_sideinfo_success | 4 | mean_psnr | 29.780485 | Quality gain came from counted GS-domain residual side-info, not from the old adapter predictor alone. |
| 157 | stage158_residual_sideinfo_success | 8 | mean_psnr | 29.578682 | Quality gain came from counted GS-domain residual side-info, not from the old adapter predictor alone. |
| 196 | target_gap_remaining | full_sequence | best_ceiling_delta_psnr_vs_target | -0.769884 | Even best current keyframe/selector ceiling remains below the requested target. |

## Decisions

| item | decision | action |
|---|---|---|
| continue_old_adapter_training | reject | Do not spend Stage201+ on continuing GaussianAnchorDynamicPredictor unchanged. |
| raise_keyframe_qbits_for_old_adapter | reject | Use qbit as rate/ablation variable only, not the primary quality fix. |
| use_stage158_as_evidence_predictor_is_solved | reject | Design predictor and residual codec jointly; measure predictor-only before selector training. |
| new_predictor_requirement | require_new_architecture_and_loss | Stage200 must propose a stronger temporal refiner/motion-field architecture with render-aware and RD-aware gates. |

## Requirements

| requirement | stage gate |
|---|---|
| predictor_only_gate_before_selector | Stage201 must beat old adapter and show stable render metrics before Stage206+. |
| gs_native_residual_payload | Stage203/204 must define counted GS latent/residual bitstreams, not image residuals. |
| edge_rd_headroom_before_selector_training | Stage207 DP oracle must beat fixed baselines before Stage209 selector training is promoted. |
| full_sequence_metrics_only_for_strong_claim | Stage211 must include full-sequence PSNR/SSIM/MS-SSIM/LPIPS and measured bytes. |

## Outputs

- Evidence CSV: `experiments/stage198_prior_predictor_training_audit/stage198_prior_predictor_evidence.csv`
- Decision CSV: `experiments/stage198_prior_predictor_training_audit/stage198_prior_predictor_decisions.csv`
- Requirement CSV: `experiments/stage198_prior_predictor_training_audit/stage198_new_route_requirements.csv`
