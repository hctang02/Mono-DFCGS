# Stage162 Keyframe Selector Protocol Source Audit

Date: 2026-06-30

## Goal

Start the adaptive keyframe selector line by locking the protocol and auditing which RGB/motion feature sources are allowed and feed-forward valid.

## User Direction

- After Stage158/161 middle-frame recovery, continue optimizing the next component: selector/keyframe selection.
- Encoder-side RGB/motion information may be allowed, but the source must be evaluated explicitly.
- The audit must say whether features come from dataset/input RGB, deterministic frame differences, network-estimated motion, or offline/oracle diagnostics.
- Feed-forward validity must be documented.

## Plan

- Define keyframe selector output: transmitted keyframe indices / GOP schedule.
- Define total rate accounting: keyframe anchor cost + keyframe-index metadata + Stage158 middle residual side-info.
- Define allowed selector feature tiers:
  - raw input RGB frames at encoder;
  - deterministic RGB/motion proxies from input frames;
  - optional pretrained motion/feature networks using only input RGB;
  - offline labels/diagnostics not allowed at inference.
- Define baselines and oracle references:
  - uniform gap4;
  - uniform gap8;
  - previous segment-error selector references;
  - heuristic RGB/motion selector;
  - oracle schedule for upper-bound only.
- Store protocol JSON/CSV/Markdown only.

## Success Criteria

- Package explicitly states which features are allowed at encoder inference, which are forbidden, and what metadata is transmitted.
- Package defines how adaptive keyframe selection will be evaluated with Stage158 middle-frame recovery.

## Command

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage162_keyframe_selector_protocol_source_audit.py && CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage162_keyframe_selector_protocol_source_audit.py
```

## Outputs

- `experiments/stage162_keyframe_selector_protocol_source_audit/stage162_keyframe_selector_protocol_source_audit_package.json`
- `experiments/stage162_keyframe_selector_protocol_source_audit/stage162_keyframe_selector_protocol_source_audit_report.md`
- `experiments/stage162_keyframe_selector_protocol_source_audit/stage162_feature_source_audit.csv`
- `experiments/stage162_keyframe_selector_protocol_source_audit/stage162_rate_accounting_rules.csv`
- `experiments/stage162_keyframe_selector_protocol_source_audit/stage162_selector_baselines.csv`
- `experiments/stage162_keyframe_selector_protocol_source_audit/stage162_protocol_decisions.csv`
- `experiments/stage162_keyframe_selector_protocol_source_audit/stage162_historical_selector_artifacts.csv`

## Result

- Packaged protocol status: `keyframe_selector_protocol_and_source_audit_packaged`.
- Fixed middle-recovery component: Stage158/161 `streamsplat_guided_half_anchor_entropy_residual_v1`.
- Selector output: transmitted keyframe indices or segment lengths.
- Decoder does not need selector RGB/motion features.

## Feature Source Decisions

| feature group | source | inference status | feed-forward assessment |
|---|---|---|---|
| raw RGB | input video frames / DAVIS RGB files | allowed | feed-forward for offline video encoding; online use must declare lookahead |
| deterministic RGB/motion proxy | frame differences, block SAD/MSE, edge/histogram changes | primary cheap tier | feed-forward if derived only from input frames |
| pretrained motion/feature network | fixed RAFT/GMFlow/DINO/ResNet from RGB only | optional high-compute tier | feed-forward with compute/dependency caveat |
| encoder RD probe | candidate Stage158 payload/quality probes | expensive encoder tier only | offline encoder-side RD, not cheap single-pass feed-forward |
| rendered quality oracle | PSNR/SSIM/MS-SSIM/LPIPS labels | not inference | label/evaluation only |
| target dense/residual | target anchor or unencoded residual tensors | forbidden as selector inference feature | only train/encoder-side label or actual encoded Stage158 payload |

## Rate Accounting Decisions

- Count keyframe anchor payload for every selected keyframe.
- Count adaptive keyframe indices or segment lengths when schedule is non-uniform.
- Count selector mode id if multiple policies can be selected.
- Count Stage158 residual payload plus one-byte half selector for each recovered middle/intermediate frame.
- Do not count local feature computation as bitrate if features are not transmitted, but report compute/dependency tier.

## Next

- Stage163 should build a DAVIS RGB/motion selector data package and attach Stage158-compatible rate/quality labels or oracle references.
