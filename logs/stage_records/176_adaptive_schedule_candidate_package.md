# Stage176 Adaptive Schedule Candidate Package

Date: 2026-06-30

## Goal

Package the Stage165 adaptive keyframe schedule as the current sampled-validated candidate, with decoder contract, evidence chain, rate accounting, limitations, and broader-validation path.

## Plan

- Load Stage162 feature-source/decoder contract, Stage165 schedule package, Stage172 rate audit, Stage174 medium rendered validation, and Stage175 decision branch.
- Create a candidate policy JSON for `rgb_motion_rank_gate_gap8_plus_extra_targets_v1`.
- Record allowed encoder-side selector inputs, decoder-side transmitted schedule metadata, and forbidden inference inputs.
- Summarize evidence and limitations.
- Output lightweight JSON/CSV/Markdown package only.

## Success Criteria

- Package clearly marks the adaptive schedule as sampled-validated but not final full-sequence RD.
- Decoder contract and rate accounting rules are explicit.
- Next broader/full-sequence validation requirements are listed.

## Execution

- Checked `nvidia-smi` before running; Stage176 is CPU-only.
- Compiled and ran `scripts/run_stage176_adaptive_schedule_candidate_package.py` with `/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python`.

## Result

- Package: `experiments/stage176_adaptive_schedule_candidate_package/stage176_adaptive_schedule_candidate_package.json`.
- Report: `experiments/stage176_adaptive_schedule_candidate_package/stage176_adaptive_schedule_candidate_package_report.md`.
- Candidate policy JSON: `experiments/stage176_adaptive_schedule_candidate_package/stage176_adaptive_keyframe_schedule_candidate_policy.json`.
- Evidence CSV: `experiments/stage176_adaptive_schedule_candidate_package/stage176_candidate_evidence.csv`.
- Limitations CSV: `experiments/stage176_adaptive_schedule_candidate_package/stage176_candidate_limitations.csv`.
- Next validation CSV: `experiments/stage176_adaptive_schedule_candidate_package/stage176_next_validation_requirements.csv`.
- Policy: `rgb_motion_rank_gate_gap8_plus_extra_targets_v1_sampled_candidate`.
- Status: `sampled_validated_candidate_not_final_full_sequence_rd`.

## Candidate Contract

- Encoder-side selector inputs: input RGB frames, deterministic RGB/motion proxy features from input frames, optional fixed RGB-only pretrained motion/feature networks in a higher-compute tier.
- Decoder receives transmitted keyframe schedule/keyframe indices, normal keyframe payloads, Stage158 residual payloads for non-keyframe middle recovery rows, and normal StreamSplat endpoint inputs/normalized time for recovered middle rows.
- Decoder does not receive or compute RGB/motion selector features.
- Forbidden inference inputs: target RGB, target dense anchors except through encoded Stage158 payload, rendered metrics/oracle labels, unencoded target residual tensors.

## Evidence Summary

- Selector policy: rank threshold `0.65`, min votes `1`, selected rows `70`.
- Selector recall: hard `0.733333333333`, payload `0.819444444444`.
- Schedule size: `358` adaptive keyframes over `1999` frames, metadata `327` bytes.
- Rate proxy adaptive/gap8/gap4: `0.194181515827 / 0.300453182577 / 0.370523510564` MiB/frame.
- Medium validation: `150` protocol rows, `54` new renders.
- False-negative risk: adaptive delta vs gap8 PSNR `-0.0109637522816`, LPIPS `+0.000909611582756`.
- False-positive keyframe risk: false-positive controls under uniform gap8 are already PSNR/LPIPS/payload `30.0717550621 / 0.183046225458 / 157481.25`.

## Limitations

- Not final full-sequence RD.
- Selector false-positive keyframes and false negatives remain explicit risks.
- Adaptive keyframe rows have no middle-render metrics; they are judged through keyframe rate/all-frame sequence metrics.
- DAVIS experiments are offline feed-forward; online streaming variants require declared lookahead.

## Next Validation Requirements

- Broader sampled validation beyond the 50-target medium protocol.
- Full-sequence RD accounting with all keyframes, adaptive metadata, and Stage158 residual payloads counted.
- All-frame/keyframe-only/middle-only/per-sequence PSNR, SSIM, MS-SSIM, and LPIPS.
- Selector refinement only if broader validation shows false-positive overhead or false-negative quality risk.
