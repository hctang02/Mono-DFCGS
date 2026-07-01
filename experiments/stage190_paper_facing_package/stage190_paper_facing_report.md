# Stage190 Paper-Facing Package

## Recommended Title

Mono-DFCGS: Recovery-Aware Adaptive Keyframe Scheduling for Monocular Dynamic Gaussian Splatting Compression

## Abstract Draft

We present Mono-DFCGS, a monocular dynamic Gaussian splatting compression pipeline that combines StreamSplat-guided Gaussian-domain middle-frame recovery with a recovery-aware adaptive keyframe schedule. The recovery module preserves the original dynamic prediction as the motion/geometry base and transmits a counted entropy-coded residual for one selected half-anchor, avoiding decoder-side target anchors, target RGB, or unencoded oracle residuals. The adaptive scheduler uses encoder-side RGB/motion cues to promote difficult or high-payload frames while transmitting only the resulting schedule. On full DAVIS sequence accounting, the current adaptive schedule is a measured middle rate-distortion point: it improves PSNR, SSIM, MS-SSIM, and LPIPS over uniform gap8, and remains below uniform gap4 rate, but it is not yet lower-rate than gap8. We further analyze feature ablations, lower-budget sensitivity, and failure cases to identify the next selector refinement targets.

## Core Claim

The current measured result should be framed as a recovery-aware adaptive middle RD point: better quality than uniform gap8 at higher rate, and lower rate than uniform gap4 at lower quality.

## Table 1. Full-Sequence RD-Quality

| schedule | keyframes | MiB/frame | dRate vs gap8 | PSNR | SSIM | MS-SSIM | LPIPS | note |
|---|---|---|---|---|---|---|---|---|
| Uniform gap8 | 292 | 0.275866 | 0.000000 | 29.373965 | 0.867626 | 0.984343 | 0.168692 | lower-rate fixed-gap baseline |
| Mono-DFCGS adaptive | 358 | 0.290743 | 0.014877 | 29.425583 | 0.869294 | 0.984647 | 0.165937 | middle RD point; better quality than gap8 but higher rate |
| Uniform gap4 | 536 | 0.330769 | 0.054903 | 29.535716 | 0.873944 | 0.985529 | 0.159472 | higher-quality fixed-gap reference |

Scope: Stage185/186 measured schedule-packed q12 keyframes plus measured Stage158 residual payloads plus exact metadata.

## Table 2. Selector Feature Ablation

| variant | features | selected rows | keyframes | hard recall | payload recall | note |
|---|---|---|---|---|---|---|
| full_stage165_features | 5 | 70 | 358 | 0.733333 | 0.819444 | full selector reference; highest payload recall |
| drop_interp_rgb | 4 | 69 | 357 | 0.733333 | 0.805556 | conservative lower-budget candidate |
| motion_proxy_edge_hist | 3 | 68 | 357 | 0.733333 | 0.791667 | small reduction with same hard recall |
| edge_hist_only | 2 | 67 | 356 | 0.733333 | 0.777778 | motion/edge stress point |
| drop_hist_motion | 4 | 61 | 349 | 0.733333 | 0.736111 | aggressive lower-budget candidate |
| drop_edge_motion | 4 | 60 | 348 | 0.433333 | 0.750000 | aggressive but lower hard recall |
| proxy_only | 1 | 42 | 331 | 0.400000 | 0.541667 | minimal one-feature proxy |
| rgb_only | 2 | 48 | 336 | 0.400000 | 0.638889 | direct RGB-only stress point |

Scope: Stage187 is a selector-label/protocol ablation, not measured full-sequence RD for each ablation schedule.

## Table 3. Lower-Budget Sensitivity

| candidate | keyframes | additive MiB/frame | dRate vs gap8 | PSNR | dPSNR vs gap8 | LPIPS | note |
|---|---|---|---|---|---|---|---|
| uniform_gap8 | 292 | 0.275884 | 0.000000 | 29.373965 | 0.000000 | 0.168692 | additive sensitivity baseline |
| interval_top10pct_cells | 299 | 0.277375 | 0.001490 | 29.381126 | 0.007161 | 0.168325 | lowest-rate positive-quality candidate |
| interval_score_ge4p0 | 324 | 0.282992 | 0.007108 | 29.410133 | 0.036168 | 0.166827 | balanced half-overhead candidate |
| interval_top90pct_cells | 353 | 0.289480 | 0.013595 | 29.424507 | 0.050542 | 0.166018 | near-full quality below full adaptive rate |
| stage165_adaptive_full | 358 | 0.290766 | 0.014881 | 29.425583 | 0.051618 | 0.165937 | full frozen selector under additive scope |
| uniform_gap4 | 536 | 0.330804 | 0.054920 | 29.535716 | 0.161751 | 0.159472 | additive fixed-gap high-quality reference |

Scope caveat: Stage188 uses measured single-anchor additive keyframes plus measured residuals and exact metadata. It is internally comparable across Stage188 candidates but not numerically interchangeable with Stage185 schedule-packed rates.

## Table 4. Candidate Failure Cases

| candidate | keyframes | MiB/frame | dRate vs full | PSNR | changed frames | worst changed dPSNR | note |
|---|---|---|---|---|---|---|---|
| interval_top10pct_cells | 299 | 0.277375 | -0.013391 | 29.381126 | 370 | -5.309561 | most rate-efficient but many changed frames vs full adaptive |
| interval_score_ge4p0 | 324 | 0.282992 | -0.007774 | 29.410133 | 223 | -2.389079 | balanced point with fewer changed frames |
| interval_top90pct_cells | 353 | 0.289480 | -0.001286 | 29.424507 | 35 | -1.277667 | near-full point with small quality loss |

## Promoted Rate-Risk Examples

| sequence | frame | dPSNR | dLPIPS | payload delta bytes | reason |
|---|---|---|---|---|---|
| horsejump-high | 15 | 0.177171 | -0.020872 | 506463 | small_unlabeled_gain large_local_payload_delta |
| drift-chicane | 6 | 0.166051 | -0.024552 | 594793 | small_unlabeled_gain large_local_payload_delta |

## Residual-Risk Hotspots

| sequence | residual risks | low PSNR | high LPIPS | high payload | max risk | worst frame |
|---|---|---|---|---|---|---|
| cows | 86 | 86 | 1 | 62 | 1.904395 | 92 |
| parkour | 75 | 27 | 26 | 74 | 1.560360 | 68 |
| camel | 73 | 61 | 0 | 67 | 1.393465 | 60 |
| goat | 73 | 0 | 0 | 73 | 0.789140 | 5 |
| breakdance | 72 | 72 | 2 | 0 | 1.748598 | 5 |
| soapbox | 72 | 0 | 1 | 71 | 0.673080 | 67 |
| bmx-trees | 67 | 3 | 6 | 67 | 0.707380 | 44 |
| car-roundabout | 55 | 13 | 0 | 55 | 0.856995 | 61 |
| dance-twirl | 54 | 2 | 35 | 28 | 1.665491 | 60 |
| india | 52 | 0 | 7 | 52 | 3.421599 | 36 |

## Decoder Contract

Allowed decoder inputs: original StreamSplat endpoint/base inputs, normalized time, encoded q6/keep1.0 entropy residual payload, counted one-byte half selector, and transmitted schedule/keyframe metadata.

Forbidden decoder inputs: target dense anchor, target RGB, unencoded target residual, rendered quality/oracle labels, or selector features that are not represented by transmitted schedule metadata.

## Claim Boundaries

| category | item | status | paper wording |
|---|---|---|---|
| claim | measured adaptive RD position | supported | The current adaptive schedule is a middle rate-distortion point between uniform gap8 and uniform gap4. |
| claim | quality over gap8 | supported | Adaptive improves PSNR, SSIM, MS-SSIM, and LPIPS over uniform gap8 on full-sequence evaluation. |
| non-claim | lower rate than gap8 | not supported | Do not claim the frozen adaptive schedule is lower-rate than uniform gap8. |
| scope caveat | Stage188 additive rates | separate scope | Stage188 additive rates compare lower-budget candidates internally and must not be numerically mixed with Stage185 schedule-packed rates. |
| decoder contract | allowed inputs | fixed | The decoder receives original StreamSplat endpoint/base inputs, normalized time, encoded q6 keep1.0 entropy residual payload, counted half selector, and transmitted schedule/keyframe metadata. |
| decoder contract | forbidden inputs | fixed | The decoder does not receive target dense anchors, target RGB, unencoded target residuals, or oracle labels. |
| selector contract | encoder-side features | fixed | RGB/motion selector features are encoder-side only; the transmitted schedule is counted and sufficient for decoding. |
| limitation | unoptimized selector | open | Stage189 shows broad residual-risk hotspots and rare high-cost promotions; further selector refinement is needed for an optimized RD frontier. |
| next measurement | schedule-packed lower-budget candidates | optional | If final claims require same-scope candidate RD, selected Stage188 candidates should be measured with schedule-packed keyframe streams. |

## Next Paper/Experiment Step

Use this package for the paper-facing method/results section. If same-scope lower-budget RD is needed, measure schedule-packed keyframe streams for one or two selected Stage188 candidates before making final rate-frontier claims.
