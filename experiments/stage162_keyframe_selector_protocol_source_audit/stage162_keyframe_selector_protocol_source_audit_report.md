# Stage162 Keyframe Selector Protocol Source Audit

## Scope

Adaptive keyframe selection is now the next component after Stage158/161 middle recovery.
The selector chooses a keyframe schedule; decoder receives the transmitted schedule and does not need RGB/motion features.

## Feature Source Audit

| feature group | source | feed-forward status | inference | caveat |
|---|---|---|---|---|
| raw_rgb_encoder_input | input video frames available to encoder; DAVIS RGB files in current experiments | feed-forward for offline video encoding; for online streaming requires declared lookahead window | yes | dataset RGB is the source signal being compressed, not extra side information |
| deterministic_rgb_motion_proxy | deterministic functions of encoder input RGB frames | feed-forward if computed only from input frames inside declared lookahead | yes_primary_cheap_tier | preferred first heuristic selector feature family |
| pretrained_motion_or_feature_network | fixed pretrained network fed only encoder input RGB frames | feed-forward with compute-cost caveat; no target dense anchors or rendered metrics may enter | yes_optional_higher_compute_tier | network compute and dependency must be documented; no bitrate charge unless features are transmitted |
| encoder_rd_probe | encoder-side candidate encodes/renders using available input and target anchors during compression | offline encoder-side RD probe, not a cheap single-pass feed-forward selector | allowed_as_expensive_encoder_optimization_tier_only_if_schedule_is_transmitted | must not be described as low-compute feed-forward; useful upper/teacher policy |
| rendered_quality_oracle | requires target RGB and rendered candidate output | not feed-forward selector input for deployable inference | no | allowed only to create labels/evaluate oracle schedules |
| target_dense_anchor_or_residual | Stage61 dense anchors or residual against target anchor | not allowed as selector inference feature except when it is encoded into transmitted Stage158 residual payload | no_for_selector_features | prevents oracle leakage into keyframe selector |

## Rate Accounting

| component | counting rule | notes |
|---|---|---|
| keyframe_anchor_payload | count q-bit Gaussian anchor payload/metadata for every transmitted keyframe | Use existing q12/qbit anchor accounting tables; if schedule changes keyframe count, anchor rate changes accordingly. |
| adaptive_keyframe_indices | count schedule metadata for non-uniform selectors: keyframe count plus sorted indices or segment lengths packed with ceil(log2(total_frames)) bits/index | Uniform fixed-gap schedules can be signaled by mode id only; adaptive schedules must transmit their indices or equivalent segment lengths. |
| selector_mode_id | count at least one small mode id when multiple schedule policies are available | Examples: uniform_gap4, uniform_gap8, rgb_motion_heuristic, learned_selector. |
| stage158_middle_residual_payload | count q6/keep1.0 entropy residual payload plus one-byte half selector for every recovered middle/intermediate frame | Payload may vary by segment difficulty; Stage158 current policy is quality-first and not rate-minimized. |
| feature_computation | not counted as bitrate if features are not transmitted, but compute/dependency tier must be reported | Pretrained optical flow or embeddings must be documented as higher-compute encoder tools. |

## Baselines

| baseline | selector inputs | metadata | role | status |
|---|---|---|---|---|
| uniform_gap4_plus_stage158 | none beyond total frame count | mode id or fixed policy id only | high-quality reference | primary baseline |
| uniform_gap8_plus_stage158 | none beyond total frame count | mode id or fixed policy id only | lower-keyframe-rate reference | primary baseline |
| stage16_segment_motion_or_rd_schedule | historical motion/segment-error features | adaptive keyframe indices or segment lengths | historical adaptive schedule reference | reuse for comparison, not final without DAVIS+Stage158 validation |
| rgb_motion_heuristic_v1 | raw RGB frame differences, block motion proxy, edge-change statistics | adaptive keyframe indices or segment lengths | first deployable selector candidate | next implementation target |
| learned_rgb_motion_selector_v1 | RGB/motion features only; no target anchors or rendered metrics at inference | adaptive keyframe indices or segment lengths | later learned selector candidate | after heuristic/oracle data package |
| oracle_rd_schedule | rendered RD/quality labels from candidate schedules | oracle schedule indices for accounting only | upper bound and training label source | not deployable inference selector |

## Decisions

| item | decision | details |
|---|---|---|
| codec_setting | quality_first_stage158_middle_recovery | Use Stage158 policy as middle/intermediate recovery component; do not over-optimize its residual rate in this phase. |
| selector_output | transmitted_keyframe_schedule | Selector outputs keyframe indices or segment lengths; decoder follows transmitted schedule and does not reproduce selector features. |
| feature_scope | encoder_side_rgb_motion_allowed | RGB/motion features are allowed if derived from input video frames available to encoder; source and compute tier must be logged. |
| feedforward_protocol | offline_video_feedforward_with_optional_lookahead | Current DAVIS protocol is offline encoding, so full input RGB can be used encoder-side. Online/streaming variants must declare a lookahead window. |
| forbidden_inference_inputs | no_target_dense_or_rendered_metric_leakage | Target dense anchors, unencoded residuals, target RGB quality labels, rendered PSNR/LPIPS, and oracle labels are not selector inference inputs. |
| evaluation_metrics | all_frame_keyframe_middle_and_sequence_level | Report all-frame, keyframe-only, middle-only, per-gap, and per-sequence PSNR/SSIM/MS-SSIM/LPIPS plus rate. |
| splitting | sequence_aware_validation | Learned selector validation should split by sequence; heuristic selector should report all selected DAVIS sequences and weak cases. |

## Next Stage

Stage163 should build the first DAVIS selector data package: compute cheap RGB/motion segment features, define candidate schedules, and attach Stage158-compatible rate/quality labels or oracle references for training/evaluation.
