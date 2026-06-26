# Recent Training, Innovation, And Q&A Notes

Date: 2026-06-26

This note records the recent user-facing explanations and the current design direction for Mono-DFCGS. It is meant to preserve the discussion around training, innovation points, Stage3-vs-current PSNR, RD plotting, selector constraints, and possible side-information exploration.

## Reporting Decisions

- Future summaries should default to all-frame PSNR only unless middle-only PSNR is explicitly requested.
- `rendered_prior_0p1` must be described as an oracle/calibrated adaptive selector, not as the final deployable selector.
- The final selector must be fully feed-forward: no test-time rendered oracle, no PSNR/error lookahead, and no optimization using reconstruction results.
- If non-keyframe side information is transmitted later, it must be included in the rate accounting.

## Why Current PSNR Is Lower Than Stage3

Stage3 and the current anchor-only pipeline do not use the same decoding condition.

| Version | Decoding condition | Interpretation |
|---|---|---|
| Stage3 | StreamSplat sparse-keyframe reconstruction | Quality still used RGB/depth-conditioned StreamSplat inference. It was an early upper-reference / scaffold, not a strict Gaussian-anchor-only codec result. |
| Current anchor-only pipeline | Transmitted keyframe Gaussian anchors plus adapter prediction | Stricter codec setting. Non-keyframes are reconstructed from keyframe anchors and time, without non-keyframe RGB/depth inputs. |

Stage3 counted rate roughly as keyframe Gaussian anchors, but the quality side was still borrowed from StreamSplat selected-pair reconstruction. That reconstruction used selected input RGB frames, predicted/generated depth maps, and the original StreamSplat model features. Therefore Stage3 PSNR was higher because it had more information available at reconstruction time.

The current main line is more faithful to the proposed codec:

- Do not transmit non-keyframe RGB.
- Do not transmit non-keyframe depth.
- Do not transmit non-keyframe Gaussians.
- Do not transmit motion, deformation, or residual payloads in the current main rate.
- Keyframes are rendered directly from transmitted quantized anchors.
- Non-keyframes are predicted by the Gaussian anchor adapter.

## Current Best Stage51 Result

The latest high-rate RD result is Stage51. The current best-quality setting is `adaptive rendered_prior_0p1` with q16 anchors, using actual zlib-compressed Gaussian-anchor bitstream rate.

| Method | bits | mean zlib q-anchor MiB/frame | mean all-frame PSNR |
|---|---:|---:|---:|
| uniform | q8 | 0.115670 | 28.428047 |
| adaptive `rendered_prior_0p1` | q8 | 0.115666 | 28.482271 |
| uniform | q10 | 0.178331 | 30.344503 |
| adaptive `rendered_prior_0p1` | q10 | 0.178327 | 30.402345 |
| uniform | q12 | 0.201844 | 30.689366 |
| adaptive `rendered_prior_0p1` | q12 | 0.201844 | 30.748005 |
| uniform | q16 | 0.237300 | 30.717664 |
| adaptive `rendered_prior_0p1` | q16 | 0.237296 | 30.776381 |

Current best mean point:

| Configuration | Rate | All-frame PSNR |
|---|---:|---:|
| adaptive q16 | 0.237296 MiB/frame | 30.776381 dB |

Clean RD plot outputs:

```text
experiments/stage51_high_rate_multibit_rd/stage51_clean_adaptive_all_psnr_by_bits.png
experiments/stage51_high_rate_multibit_rd/stage51_clean_uniform_all_psnr_by_bits.png
experiments/stage51_high_rate_multibit_rd/stage51_clean_mean_all_psnr_by_bits.png
experiments/stage51_high_rate_multibit_rd/stage51_clean_adaptive_delta_all_psnr_heatmap.png
```

## Meaning Of bits, gap, And zlib

`bits` is the Gaussian anchor quantization precision.

| bits | Meaning | Effect |
|---:|---|---|
| q8 | 8-bit quantized anchors | Lower rate, lower quality |
| q10 | 10-bit quantized anchors | Higher rate, better quality |
| q12 | 12-bit quantized anchors | Higher quality |
| q16 | 16-bit quantized anchors | Highest tested rate/quality, but q12 to q16 gains are small |

`gap` is the keyframe interval.

| gap | Meaning |
|---:|---|
| 1 | Every frame is a keyframe |
| 2 | Keyframes every 2 frames |
| 4 | Keyframes every 4 frames |
| 8 | Keyframes every 8 frames |
| 16 | Keyframes every 16 frames |

Smaller gaps transmit more keyframes, increasing rate and quality. Larger gaps transmit fewer keyframes, lowering rate but making reconstruction harder.

`zlib` is a generic lossless compression method, similar to zip/gzip compression. In this project it means:

```text
quantized Gaussian anchor container -> zlib compression -> transmitted bitstream size
```

It is not a learned entropy model. It is a practical generic compression baseline for actual transmitted anchor bitstreams.

## Uniform vs Adaptive Keyframe Selection

Uniform selection uses fixed keyframe spacing, for example:

```text
0, 4, 8, 12, 16, ...
```

Adaptive selection tries to put the same keyframe budget in more useful places, for example:

```text
0, 3, 7, 12, 16, ...
```

In Stage51 q8 mean results:

| Method | Rate | All-frame PSNR |
|---|---:|---:|
| uniform | 0.115670 | 28.428047 |
| adaptive `rendered_prior_0p1` | 0.115666 | 28.482271 |

The rate is nearly identical because both use nearly the same keyframe budget. The improvement comes from moving the keyframes to better temporal positions. However, the current best adaptive selector uses rendered-error-related information and is therefore an oracle/calibrated upper-reference rather than a deployable selector.

## Current Transmitted Content

The current main bitstream mostly transmits compressed keyframe Gaussian anchors plus required metadata.

| Content | Transmitted now | Counted in main rate |
|---|---:|---:|
| quantized keyframe Gaussian anchors | yes | yes |
| zlib-compressed Gaussian anchor container | yes | yes |
| keyframe indices / timestamps / segment metadata | yes | yes, as small header/container overhead |
| quantization bits / shape / range / field metadata | yes | yes, as metadata |
| non-keyframe Gaussian | no | no |
| non-keyframe RGB | no | no |
| depth map | no | no |
| motion field / deformation payload | no | no |
| residual payload | no | no |
| decoder / adapter model weights | not per-video payload | not counted in main rate |

The current anchor fields are:

```text
rgb
opacity[:1]
scale
xyz[:, 0, :]
rot[:, 0, :]
```

That is 13 values per Gaussian.

## Current Training Process

There are two main training lines so far.

| Training line | Goal | Current status |
|---|---|---|
| Gaussian adapter training | Predict non-keyframe Gaussian anchors from left/right keyframe anchors and time | Completed on local small set; stable but small improvement |
| Keyframe selector training | Choose keyframes automatically without seeing reconstruction results | Explored several versions; final feed-forward selector not solved yet |

### Gaussian Adapter Training

The adapter is conceptually:

```text
left keyframe GS + right keyframe GS + time
        -> predicted middle-frame GS
        -> rendered image
```

Stages already completed:

| Stage | Work | Result |
|---|---|---|
| Stage9 | Gaussian attribute proxy training | Real anchor loading and training loop verified |
| Stage10/10b | Renderer/RGB-loss smoke | Rendering and differentiable path verified |
| Stage21 series | Anchor-only adapter training | Better than simple linear interpolation |
| Stage25 | Leave-one-out adapter training | One held-out fold per local sample |
| Stage26 | Full-video anchor-only evaluation | 16/16 points improved over linear interpolation |

Main Stage26 result:

| Method | Mean all-frame PSNR |
|---|---:|
| linear anchor interpolation | 27.4194 |
| trained adapter | 27.4992 |
| gain | +0.0798 dB |

This improvement is small but useful: it proves the anchor-only adapter learns something beyond linear interpolation.

### Keyframe Selector Training

The selector should eventually do:

```text
input video / encoder-side features
        -> predict which frames are important
        -> choose keyframes without looking at reconstruction output
```

What has been tried:

| Stage | Work | Result |
|---|---|---|
| Stage29/35 | Oracle/proxy keyframe selection | Positive but not deployable |
| Stage44/45/46/49/51 | Rendered-error/calibrated adaptive selection | Positive RD evidence, but oracle/calibrated |
| Stage47 | Feed-forward segment cost predictor | Cost correlation exists |
| Stage48 | Feed-forward predictor used for actual DP keyframe selection | Negative result |
| Stage54 | Decision-aware selector analysis | Current candidate layout pool is too weak |

Current conclusion: keyframe selection remains the main unresolved part. The next selector should be decision-aware, meaning it should learn whether an adaptive layout should replace uniform under a fixed budget, not just predict segment-level cost correlation.

## Required Future Exploration

Two directions must be explicit in future plans.

### 1. Fully Feed-Forward Keyframe Selection

Final selector requirements:

- Input only original video and encoder-side features available before reconstruction.
- No rendered oracle error.
- No PSNR/reconstruction lookahead.
- No test-time optimization over reconstructed outputs.
- Deterministic DP or deterministic selection after predictor output is acceptable.

Potential designs:

- Adaptive-or-uniform fallback classifier.
- Pairwise layout ranking.
- DP-aware training objective.
- Calibration that predicts actual RD gain rather than only segment distortion.

### 2. Optional Non-Keyframe Side Information

If keyframe-only anchors are not enough for high quality, side information can be explored. The rule is simple: if it is transmitted, it must be counted in the rate.

Possible side information:

| Side information | Possible role | Rate caveat |
|---|---|---|
| low-resolution depth | Help geometry prediction | Must be compressed and counted |
| sparse depth / depth keypoints | Cheaper geometry hint | Must be counted |
| low-dimensional motion hint | Help dynamic interpolation | Must be counted |
| residual Gaussian correction | Improve difficult non-keyframes | Must be counted; may become a P-frame residual codec |
| importance map | Guide predictor or selective refinement | Must be counted if transmitted |

Full non-keyframe GS transmission is not preferred for the main method because it weakens the keyframe-Gaussian-only contribution.

## Current Innovation Points

| Innovation point | Plain-language meaning | Current maturity |
|---|---|---|
| Keyframe-Gaussian-only dynamic 3DGS codec | Transmit keyframe Gaussians and reconstruct the whole video from them | Core idea established |
| Monocular video setting | No multiview/camera information for final codec claims | Constraint established |
| Actual Gaussian anchor bitstream | q8/q10/q12/q16 quantized and compressed containers exist | Implemented |
| Anchor-only dynamic predictor | Predict non-keyframe Gaussians from keyframe anchors | Trained with small positive gain |
| RD-aware adaptive keyframe selection | Place keyframes where they are more useful | Oracle/calibrated positive; feed-forward unresolved |
| Strict rate accounting | Main rate counts transmitted anchors and metadata only | Established |
| FCGS/D-FCGS comparison scaffold | External baseline table framework exists | Scaffold complete, fair comparison pending |

One-sentence positioning:

```text
Mono-DFCGS is a monocular dynamic 3DGS video codec that transmits compressed keyframe Gaussian anchors and uses a learned dynamic adapter to reconstruct non-keyframes, with future work focusing on deployable feed-forward keyframe selection and optional rate-counted side information.
```

## Main Current Limitations

- Fully feed-forward keyframe selection is not solved yet.
- Current adapter training is small-scale, only on four local samples.
- Current best adaptive RD result is oracle/calibrated, not deployable.
- DAVIS/YouTube-VOS data are not mounted, blocking large-scale training.
- Optional side information has not yet been designed or evaluated.
