# Stage196 Target Feasibility Branch

## Decision

- Decision: `selector_keyframe_representation_cannot_meet_target`.
- Target PSNR: `30.654815328772308` (`uniform_gap2 + 1 dB`).

## Ceilings

| source | ceiling | PSNR | dPSNR vs gap2 | dPSNR vs target | pass | interpretation |
|---:|---|---:|---:|---:|---:|---|
| 193 | framewise_psnr_oracle | 29.749038 | 0.094223 | -0.905777 | 0 | current measured selector/fixed-gap candidate-space ceiling |
| 193 | schedule_path_psnr_oracle | 29.670134 | 0.015319 | -0.984681 | 0 | current measured selector/fixed-gap candidate-space ceiling |
| 194 | all_q12_keyframes | 29.856468 | 0.201653 | -0.798347 | 0 | q12 keyframe representation ceiling |
| 195 | q16_keyframe | 29.884665 | 0.229850 | -0.770150 | 0 | higher-fidelity dense-anchor/rendering representation ceiling |
| 195 | float_dense_anchor | 29.884931 | 0.230116 | -0.769884 | 0 | higher-fidelity dense-anchor/rendering representation ceiling |

## Branch Options

| branch | status | next stage | claim risk |
|---|---|---|---|
| selector_or_keyframe_quantization_tuning | stop | none | Cannot honestly claim a large full-sequence selector gain over best fixed gap. |
| counted_rgb_or_image_residual_correction | viable_next_diagnostic | full_sequence_counted_rgb_residual_upper_bound | Less Gaussian-native; must be framed as a counted correction payload, not a pure GS keyframe selector. |
| new_dense_anchor_reconstruction_objective_or_model | viable_but_heavy | training_or_representation_redesign | Requires new training and may not finish quickly. |
| paper_claim_scope_adjustment | viable_writing_fallback | revise_claims_and_tables | Does not satisfy user's requested stronger full-sequence result. |

## Interpretation

- The current selector/keyframe branch is exhausted for the requested `+1 dB` full-sequence target.
- The next technically plausible route is a counted correction payload or a new reconstruction model/objective, not selector threshold tuning.
