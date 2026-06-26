# Previous Rounds Summary

Date: 2026-06-26

This file summarizes key results before the dedicated `logs/stage_records/` folder was introduced.

## Current Project State

- Repository: `/mnt/hdd2tC/haocheng/Mono-DFCGS`
- Remote: `git@github.com:hctang02/Mono-DFCGS.git`
- Latest completed commit before this record structure: `efddf23 Record recent training and innovation notes`
- Main method direction: monocular dynamic 3DGS video codec that transmits compressed keyframe Gaussian anchors and predicts non-keyframes with a learned adapter.

## Key Protocol Decisions

- Default quality metric for user-facing summaries: all-frame PSNR.
- Main rate: transmitted Gaussian anchor bitstream MiB/frame.
- Decoder/model weights are not counted in the per-video main rate, but should be reported separately where relevant.
- `rendered_prior_0p1` is an oracle/calibrated selector upper bound, not final deployable selection.
- Final selector must be fully feed-forward and deterministic at test time.
- Optional side information is allowed for exploration, but must be counted in rate if transmitted.

## Main Results So Far

### Anchor-only Adapter Training

Stage25/26 trained the `GaussianAnchorDynamicPredictor` from scratch, initialized as linear interpolation plus zero residual. It did not fine-tune the original StreamSplat checkpoint.

| Method | Mean all-frame PSNR |
|---|---:|
| linear anchor interpolation | 27.4194 |
| trained adapter | 27.4992 |
| gain | +0.0798 dB |

Result: positive on 16/16 local held-out full-video q8 points, but gain is still small and training was short/small-scale.

### Stage49 q8 Adaptive RD

Stage49 extended q8 RD to `gap=1/2/3/4/8/16` with actual raw/zlib bitstreams.

| Selector | Mean all-frame PSNR delta vs uniform |
|---|---:|
| adaptive `rendered_prior_0p1` q8 zlib | +0.0542 dB |

Result: positive mean improvement, but selector is oracle/calibrated.

### Stage51 High-rate Multi-bit RD

Stage51 rendered q8/q10/q12/q16 anchors with actual zlib bitstream rates.

| Method | bits | Mean zlib MiB/frame | Mean all-frame PSNR |
|---|---:|---:|---:|
| uniform | q8 | 0.115670 | 28.428047 |
| adaptive `rendered_prior_0p1` | q8 | 0.115666 | 28.482271 |
| uniform | q10 | 0.178331 | 30.344503 |
| adaptive `rendered_prior_0p1` | q10 | 0.178327 | 30.402345 |
| uniform | q12 | 0.201844 | 30.689366 |
| adaptive `rendered_prior_0p1` | q12 | 0.201844 | 30.748005 |
| uniform | q16 | 0.237300 | 30.717664 |
| adaptive `rendered_prior_0p1` | q16 | 0.237296 | 30.776381 |

Current best mean point: adaptive q16, `0.237296 MiB/frame`, `30.776381 dB` all-frame PSNR.

Clean RD figures:

```text
experiments/stage51_high_rate_multibit_rd/stage51_clean_adaptive_all_psnr_by_bits.png
experiments/stage51_high_rate_multibit_rd/stage51_clean_uniform_all_psnr_by_bits.png
experiments/stage51_high_rate_multibit_rd/stage51_clean_mean_all_psnr_by_bits.png
experiments/stage51_high_rate_multibit_rd/stage51_clean_adaptive_delta_all_psnr_heatmap.png
```

### Selector Status

Stage48 fully feed-forward predicted selector was negative. Stage54 showed that the Stage48 candidate layout pool has weak oracle upper bound over uniform.

| Policy | Mean all-frame PSNR delta |
|---|---:|
| oracle best over Stage48 candidate pool + uniform | +0.0063 dB |
| leave-one-sample-out layout threshold | -0.0040 dB |

Conclusion: feed-forward selector needs decision-aware / DP-aware training, not just segment-cost correlation.

### FCGS / D-FCGS Baseline Work

Stage52 parsed local FCGS/D-FCGS logs and summaries. Stage53 built a comparison scaffold.

Important caveat: external baseline rows use full FCGS/D-FCGS codec MiB/frame, while Mono-DFCGS rows use Gaussian-anchor bitstream MiB/frame. Current external rows are not fair apples-to-apples main results.

### Dataset Readiness

Stage55 checked default DAVIS / YouTube-VOS / RE10K / CO3D paths.

| Item | Value |
|---|---:|
| root candidates checked | 20 |
| provider-layout-ready roots | 0 |
| anchor-export-ready roots | 0 |
| current local anchor samples | 4 |

Conclusion: large-scale training needs data mount/download and preprocessing first.

## Immediate Next Stages

- Stage56: protocol lock.
- Stage57: compact Gaussian anchor codec.
- Stage58: compression RD ablation.
- Stage59-63: stronger/longer adapter training, including teacher/baseline-style exploration and large-scale training.
- Stage64-66: fully feed-forward keyframe selector dataset/training/evaluation.
- Stage67: optional rate-counted side information.
- Stage68: final all-frame PSNR RD package.
