# Stage182 Selector Refinement Or Freeze Decision

## Decision

- Decision: `freeze_current_candidate_and_run_full_sequence_payload_measurement_next`.
- Frozen policy: `rgb_motion_rank_gate_gap8_plus_extra_targets_v1_sampled_candidate`.
- Rationale: Stage180 broader quality and Stage181 rate proxy are both positive, while remaining risks are final-RD measurement risks rather than reasons to tune the selector immediately.

## Evidence

| area | status | metric | value | source | weight |
|---|---|---|---|---|---|
| candidate_policy | current_candidate | policy name | rgb_motion_rank_gate_gap8_plus_extra_targets_v1_sampled_candidate | Stage176 | freeze_support |
| broader_quality | positive | adaptive PSNR delta vs gap8 / gap4 | 0.5644261202320328 / 0.306535729521994 | Stage180 overall 90 targets | strong_freeze_support |
| broader_perceptual | positive | adaptive LPIPS delta vs gap8 / gap4 | -0.0338725696835253 / -0.019677375422583687 | Stage180 overall 90 targets | strong_freeze_support |
| rate_proxy | positive | adaptive/gap8/gap4 total proxy MiB/frame | 0.1916456087572328 / 0.3073179395851907 / 0.3688664299140155 | Stage181 | strong_freeze_support |
| positive_promotions | positive | broader_positive_promoted PSNR delta vs gap8 / gap4 | 0.829806900266485 / 0.4439986578568402 | Stage180 category deltas | freeze_support |
| weak_sequences | positive | broader_weak_sequence_probe PSNR delta vs gap8 / gap4 | 1.028785745355083 / 0.5638605351918257 | Stage180 category deltas | freeze_support |
| false_negatives | remaining_risk_but_bounded | false_negative_residual PSNR delta vs gap8 / gap4 | -0.010963752281610173 / -0.32629360438858646 | Stage180 category deltas | risk_track_not_blocking_freeze |
| false_positive_keyframes | precision_risk_but_quality_positive | false-positive control PSNR delta vs gap8 / gap4 | 0.39611419615648114 / 0.2796445234109868 | Stage180 category deltas | risk_track_not_blocking_freeze |
| rd_claim_scope | not_final_full_sequence_rd | residual payload scope | full_sequence_keyframe_metadata_plus_stage180_broader_sampled_residual_proxy | Stage181 | requires_next_measurement |
| decoder_contract | unchanged_valid | decoder receives schedule not RGB/motion features | schedule_metadata_transmitted_no_selector_feature_at_decoder | Stage162/176/181 | freeze_support |

## Risks

| risk | status | evidence | mitigation / next step |
|---|---|---|---|
| false_negatives_remain | known_bounded_risk | Stage180 false-negative residual delta vs gap8 is near neutral but remains below gap4. | Keep false-negative stress categories in final payload/full-sequence measurement; do not claim recall solved. |
| false_positive_keyframes | not_blocking_current_freeze | Stage180 false-positive keyframe controls improve final quality but may still waste keyframes in final RD. | Track exact keyframe payload in full RD; only tune threshold if actual bitstreams show rate regression. |
| sampled_residual_proxy | blocks_final_rd_claim | Stage181 residual payload is Stage180 broader sampled estimate, not all-frame measurement. | Run all-frame/full-sequence residual payload encode for non-keyframe recovered frames. |
| main_anchor_proxy | blocks_paper_level_rd_claim | Stage181 main-anchor payload is inherited from Stage172 proxy/interpolation. | Measure actual q12 keyframe bitstreams for every transmitted keyframe. |
| online_streaming_scope | not_claimed | Selector uses offline encoder-side RGB/motion features unless lookahead is specified. | State offline encoding scope or define lookahead for streaming experiments. |

## Next Steps

| step | goal | required before final claim |
|---|---|---|
| freeze_current_candidate | Treat Stage165 adaptive selector as the current frozen candidate for the next RD measurement. | No; this is the decision for the next stage. |
| full_sequence_payload_measurement | Measure actual q12 keyframe bitstreams and all-frame Stage158 residual payloads under gap8/adaptive/gap4 schedules. | Yes. |
| all_frame_quality_report | Report all-frame, keyframe-only, middle-only, per-sequence and per-category quality. | Yes. |
| selector_threshold_tuning | Only revisit threshold/min-votes if actual full payload measurement shows rate regression or unacceptable false-positive cost. | Conditional. |

## Non-Claims

- This does not claim final full-sequence RD.
- This does not claim false negatives are solved.
- This does not claim selector precision is optimal.
- This does not allow target dense anchors, target RGB, rendered quality/oracle labels, or unencoded residuals as decoder-side inputs.

## Outputs

- Evidence CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage182_selector_refinement_or_freeze_decision/stage182_freeze_decision_evidence.csv`
- Risks CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage182_selector_refinement_or_freeze_decision/stage182_freeze_decision_risks.csv`
- Next steps CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage182_selector_refinement_or_freeze_decision/stage182_freeze_decision_next_steps.csv`
