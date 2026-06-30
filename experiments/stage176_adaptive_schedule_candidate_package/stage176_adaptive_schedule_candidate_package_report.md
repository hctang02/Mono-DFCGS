# Stage176 Adaptive Schedule Candidate Package

## Status

- Policy: `rgb_motion_rank_gate_gap8_plus_extra_targets_v1_sampled_candidate`.
- Status: sampled-validated candidate, not final full-sequence RD.
- Stage175 decision: `package_sampled_validated_candidate_and_scale_broader_validation`.

## Decoder Contract

- Decoder receives the transmitted keyframe schedule/keyframe indices and normal keyframe payloads.
- Decoder does not compute or receive RGB/motion selector features.
- Non-keyframe middle rows use fixed Stage158 recovery with counted residual payload and half selector.
- Target RGB, target dense anchors, rendered metrics, oracle labels, and unencoded residuals are forbidden inference inputs.

## Evidence

| area | status | metric | value | source |
|---|---|---|---:|---|
| selector_policy | selected_candidate | rank threshold / min votes / selected rows | 0.65 / 1 / 70 | Stage165 |
| selector_recall | positive_signal | hard recall / payload recall | 0.733333333333 / 0.819444444444 | Stage165 |
| schedule_size | bounded_between_uniforms | adaptive keyframes / frames / metadata bytes | 358 / 1999 / 327 | Stage165 |
| rate_proxy | rate_promising_sampled_proxy | adaptive / gap8 / gap4 MiB/frame | 0.194181515827 / 0.300453182577 / 0.370523510564 | Stage172 |
| medium_validation | complete | protocol rows / new renders | 150 / 54 | Stage174 |
| false_negatives | risk_neutral_quality | adaptive delta vs gap8 PSNR / LPIPS | -0.0109637522816 / 0.000909611582756 | Stage174 |
| positive_promotions | supports_schedule | positive extension gap8 PSNR / LPIPS / payload | 26.9100212409 / 0.211998779327 / 227130.875 | Stage174 |
| false_positive_keyframes | risk_precision | false-positive controls gap8 PSNR / LPIPS / payload | 30.0717550621 / 0.183046225458 / 157481.25 | Stage174/175 |
| decision | candidate_package | Stage175 branch | package_sampled_validated_candidate_and_scale_broader_validation | Stage175 |

## Limitations

- `not_final_full_sequence_rd`: Current rate is sampled/proxy and rendered evidence is medium sampled validation. Follow-up: Run broader or full-sequence RD with all keyframe, metadata, and residual payload counted.
- `selector_false_positive_keyframes`: Some promoted targets are not hard/high-payload labels and may waste extra keyframes. Follow-up: Broader validation should quantify false-positive keyframe overhead; selector refinement may add per-sequence budgets or stricter gates.
- `false_negatives_remain`: Hard rows missed by the selector remain essentially uniform-gap8 residual cases. Follow-up: Keep false-negative stress sets in broader validation and consider selector features that improve recall without large false-positive cost.
- `adaptive_keyframe_rows_have_no_middle_metrics`: Promoted rows cannot be compared by middle-render PSNR/LPIPS; they are transmitted keyframes. Follow-up: Judge promoted rows through keyframe rate, all-frame sequence metrics, and subjective keyframe continuity.
- `offline_feedforward_scope`: DAVIS experiments use offline input RGB; online streaming variants need declared lookahead. Follow-up: Document lookahead or restrict selector features for online settings.

## Next Validation

- `broader_sampled_validation`: Scale beyond the 50-target medium protocol before final claims. Requirements: Reuse existing rows where possible; include false negatives, false-positive keyframes, high-payload controls, normal controls, and weak subjective sequences.
- `full_sequence_rd_accounting`: Convert sampled proxy rate into full sequence/frame accounting. Requirements: Count all keyframe anchors, adaptive schedule metadata, Stage158 residual payloads for every non-keyframe recovery, and report MiB/frame.
- `all_frame_quality_report`: Move beyond target-row summaries. Requirements: Report all-frame, keyframe-only, middle-only, per-sequence, and per-category PSNR/SSIM/MS-SSIM/LPIPS.
- `selector_refinement_if_needed`: Reduce false-positive keyframes or false negatives if broader validation shows rate/quality issues. Requirements: Stay within Stage162-allowed encoder-side RGB/motion features; no rendered quality or target dense anchors as inference inputs.

## Outputs

- Candidate policy JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage176_adaptive_schedule_candidate_package/stage176_adaptive_keyframe_schedule_candidate_policy.json`
- Evidence CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage176_adaptive_schedule_candidate_package/stage176_candidate_evidence.csv`
- Limitations CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage176_adaptive_schedule_candidate_package/stage176_candidate_limitations.csv`
- Next validation CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage176_adaptive_schedule_candidate_package/stage176_next_validation_requirements.csv`
