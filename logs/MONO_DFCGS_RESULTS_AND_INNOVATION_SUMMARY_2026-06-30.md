# Mono-DFCGS Results And Innovation Summary

Date: 2026-06-30

Purpose: consolidate the current Mono-DFCGS evidence chain, module-level innovation points, decoder contracts, non-claims, and next validation plan into one handoff log.

## Current Main Claim

The current strongest line is a StreamSplat-guided, Gaussian-domain, rate-counted middle-frame recovery method plus an encoder-side RGB/motion adaptive keyframe schedule.

The method is not a final full-sequence RD claim yet. It is a sampled-validated candidate with explicit decoder contracts and side-info accounting.

## Current Best Components

| component | current stage | policy | status |
|---|---:|---|---|
| middle-frame recovery | Stage158/161 | `streamsplat_guided_half_anchor_entropy_residual_v1` | current quality-first recovered middle-frame policy |
| keyframe selector protocol | Stage162 | transmitted schedule with encoder-side RGB/motion features | feature-source and decoder contract audited |
| adaptive keyframe schedule | Stage165/176 | `rgb_motion_rank_gate_gap8_plus_extra_targets_v1_sampled_candidate` | sampled-validated candidate, not final full-sequence RD |
| fixed-gap comparison | Stage177 | adaptive vs uniform gap8/gap4 | sampled-medium final target quality comparison completed |
| broader sampled validation | Stage180 | adaptive vs uniform gap8/gap4 | 90-target sampled-broader final quality comparison completed |

## Middle-Frame Recovery Evidence

Stage158/161 keeps original StreamSplat as the target-time motion/geometry base and corrects one selected half-anchor with counted q6/keep1.0 Gaussian-domain entropy residual side-info plus a counted one-byte half selector.

| gap | method | tasks | PSNR | SSIM | MS-SSIM | LPIPS | payload bytes | direct rate ref |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | original StreamSplat middle base | 60 | 22.064218 | 0.600809 | 0.801367 | 0.301495 | 0.000000 |  |
| 4 | Stage158 recovered middle | 60 | 29.780485 | 0.877938 | 0.985088 | 0.166020 | 209392.833333 | 0.381631 |
| 8 | original StreamSplat middle base | 60 | 20.337275 | 0.520331 | 0.700537 | 0.359337 | 0.000000 |  |
| 8 | Stage158 recovered middle | 60 | 29.578682 | 0.869660 | 0.983847 | 0.178535 | 215967.883333 | 0.303588 |

Key interpretation:

- Stage151 linear-base side-info recovered PSNR but had visual/perceptual risk.
- Stage154 established original StreamSplat as the preferred motion/geometry base because it had better LPIPS than the linear-base recovery.
- Stage155 proved achievability with image residual side-info, but image residual was an upper bound, not the final GS-feature method.
- Stage156 converted the idea into Gaussian-domain half-anchor residual correction.
- Stage157/158 validated and froze the quality-first policy.
- Stage160 added subjective evidence over 24 gap4 examples from 12 DAVIS sequences.

## Adaptive Keyframe Schedule Evidence

Stage165 builds an adaptive schedule from Stage162-allowed RGB/motion features. The decoder receives transmitted schedule/keyframe indices and does not compute or receive RGB/motion selector features.

| metric | value | source |
|---|---:|---|
| selected gate rank threshold | 0.65 | Stage165 |
| minimum votes | 1 | Stage165 |
| selected sampled rows | 70 / 120 | Stage165 |
| hard precision / recall / F1 | 0.314286 / 0.733333 / 0.440000 | Stage165 |
| payload precision / recall | 0.842857 / 0.819444 | Stage165 |
| adaptive keyframes | 358 / 1999 | Stage165 |
| schedule metadata | 327 bytes | Stage165/172 |

Stage172 sampled/proxy rate accounting:

| schedule | keyframes | extra vs gap8 | total proxy MiB/frame | delta vs gap8 |
|---|---:|---:|---:|---:|
| uniform_gap8 | 292 | 0 | 0.300453182577 | 0.000000000000 |
| Stage165 adaptive | 358 | 66 | 0.194181515827 | -0.106271666750 |
| uniform_gap4 | 536 | 244 | 0.370523510564 | 0.070070327987 |

Key interpretation:

- Adaptive adds 66 keyframes over uniform gap8, but avoids much more sampled Stage158 residual payload on selected high-cost targets.
- Schedule metadata is tiny relative to anchors/residual payload.
- This rate result is sampled/proxy, not final full-sequence RD.

## Stage177 Fixed-Gap Final Quality Comparison

Stage177 compares final target quality on the Stage174 medium 50-target set. Rendered middle-recovery rows reuse Stage174 metrics; target-keyframe rows render the q12 target keyframe anchor.

| schedule | targets | keyframes | middle recovery | mean PSNR | p10 PSNR | mean LPIPS | mean payload bytes/target |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 50 | 0 | 50 | 28.756927 | 25.130052 | 0.189479 | 211348.720 |
| Stage165 adaptive | 50 | 26 | 24 | 29.217096 | 25.223247 | 0.157234 | 95704.400 |
| uniform_gap4 | 50 | 6 | 44 | 28.945814 | 25.108181 | 0.177610 | 181242.240 |

Adaptive deltas:

- Adaptive minus uniform gap8 PSNR: `+0.4601686250283185` dB.
- Adaptive minus uniform gap4 PSNR: `+0.27128186202823` dB.
- Adaptive minus uniform gap8 LPIPS: `-0.03224517673254013`.
- Adaptive minus uniform gap4 LPIPS: `-0.02037559390068054`.
- Adaptive keyframe targets: `26 / 50`.

Category PSNR deltas:

| category | targets | adaptive keyframes | delta vs gap8 | delta vs gap4 |
|---|---:|---:|---:|---:|
| false_negative_residual | 8 | 0 | -0.010964 | -0.326294 |
| high_payload_residual_control | 4 | 0 | +0.315565 | +0.276767 |
| high_payload_residual_control_extension | 8 | 0 | 0.000000 | -0.291864 |
| normal_residual_control | 4 | 0 | 0.000000 | -0.011207 |
| positive_promoted | 14 | 14 | +1.248305 | +0.879899 |
| positive_promoted_extension | 8 | 8 | +0.346645 | +0.501244 |
| selector_false_positive_keyframe_control | 4 | 4 | +0.396114 | +0.279645 |

Interpretation caveat:

- Adaptive keyframe improvements are q12 keyframe reconstruction quality, not Stage158 middle-recovery quality.
- Residual rows where adaptive keeps the same segment as uniform gap8 should match gap8 by construction.
- The comparison is sampled-medium evidence, not final full-sequence RD.

## Stage180 Broader Sampled Validation

Stage180 executes the Stage179 broader sampled protocol. It covers 90 targets and 270 schedule rows, reusing Stage174/177 evidence where possible and rendering only missing middle/keyframe metrics.

Execution coverage:

| item | value |
|---|---:|
| targets | 90 |
| schedule rows covered | 270 / 270 |
| new Stage158 middle renders | 88 / 88 |
| new q12 keyframe metrics | 32 / 32 |
| reused Stage174 rows | 150 |
| keyframe marker rows | 64 |

Overall final target quality:

| schedule | targets | keyframes | middle recovery | mean PSNR | p10 PSNR | mean LPIPS | mean payload bytes/target |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage165 adaptive | 90 | 56 | 34 | 29.770753 | 25.424549 | 0.142780 | 74673.433 |
| uniform_gap4 | 90 | 8 | 82 | 29.464217 | 25.431154 | 0.162457 | 196008.433 |
| uniform_gap8 | 90 | 0 | 90 | 29.206326 | 25.418355 | 0.176652 | 219878.578 |

Adaptive deltas:

- Adaptive minus uniform gap8 PSNR: `+0.5644261202320328` dB.
- Adaptive minus uniform gap4 PSNR: `+0.306535729521994` dB.
- Adaptive minus uniform gap8 LPIPS: `-0.0338725696835253`.
- Adaptive minus uniform gap4 LPIPS: `-0.019677375422583687`.

Interpretation caveat:

- Stage180 is stronger than Stage177 because it expands from 50 to 90 sampled targets.
- It is still sampled broader validation, not final full-sequence RD.
- Adaptive keyframe rows use q12 keyframe reconstruction quality, not Stage158 middle-recovery quality.

## Module-Level Innovation Points

### StreamSplat-Guided Gaussian Recovery

The recovery method uses original StreamSplat target-time prediction as the motion/geometry base rather than linear interpolation or a weak RGB adapter. This addresses the observed LPIPS/visual risk of purely PSNR-oriented linear-base recovery.

### Half-Anchor Gaussian Residual Correction

The method corrects one 36864-Gaussian half-anchor instead of attempting an invalid residual against the full 73728-Gaussian original dynamic base. The selected half is transmitted by a counted one-byte selector.

### Counted Entropy Residual Payload

The Gaussian residual is quantized and entropy-coded. The residual payload and selector byte are counted in rate accounting; no hidden teacher side-info is treated as free.

### Explicit Decoder Contract

The decoder receives only original StreamSplat endpoint/base inputs, normalized time, encoded residual payload, half selector, and transmitted schedule/keyframe metadata. Target dense anchors, target RGB, unencoded residuals, rendered metrics, and oracle labels are forbidden decoder inputs.

### Encoder-Side RGB/Motion Adaptive Schedule

The adaptive schedule uses features derived from encoder input RGB/motion proxies. These features are not transmitted and are not needed by the decoder; only the resulting keyframe schedule is transmitted and counted.

### Sampled Rate/Quality Coupling

The adaptive schedule is evaluated jointly with Stage158 residual cost: selected hard/high-payload targets are promoted to keyframes, reducing expensive residual recovery on those targets while preserving middle recovery on unselected rows.

## Non-Claims And Risks

| item | status |
|---|---|
| final full-sequence RD | not claimed yet |
| all-frame quality report | not completed yet |
| selector precision solved | not claimed; false-positive keyframes remain |
| false negatives solved | not claimed; residual false negatives remain close to gap8 behavior |
| online streaming selector | not claimed; current setting is offline video encoding unless lookahead is declared |
| target dense/RGB as decoder input | forbidden |
| rendered quality/oracle as inference selector input | forbidden |

## Current Evidence Chain

| stage | role | result |
|---:|---|---|
| 151 | linear-base PSNR recovery reference | showed counted residual side-info can recover PSNR but not final visual base |
| 153 | multi-metric diagnostic | PSNR alone insufficient; LPIPS/bad cases matter |
| 154 | StreamSplat base alignment | original StreamSplat preferred as perceptual motion/geometry base |
| 155 | side-info achievability | image residual upper bound reaches 31-32 dB |
| 156 | Gaussian-domain discovery | half-anchor residual keep1/q6 reaches about 29.5-29.9 dB sampled |
| 157 | broader recovery validation | Stage156 selected policy stable on 120 tasks |
| 158 | recovery policy package | freezes `streamsplat_guided_half_anchor_entropy_residual_v1` |
| 160 | subjective evidence | 24 gap4 examples from 12 sequences exported outside git |
| 161 | method narrative | packages current middle-frame recovery claim |
| 162 | selector source audit | permits encoder-side RGB/motion features and transmitted schedule |
| 165 | adaptive schedule preflight | creates `rgb_motion_rank_gate_gap8_plus_extra_targets_v1` |
| 172 | rate accounting | adaptive sampled/proxy rate below gap8/gap4 after charges |
| 174 | medium rendered validation | covers 150 rows with 54 new renders |
| 176 | candidate package | freezes sampled-validated adaptive schedule candidate |
| 177 | fixed-gap PSNR comparison | adaptive beats gap8/gap4 in sampled-medium final target PSNR |
| 180 | broader sampled validation | adaptive beats gap8/gap4 on 90-target final quality with +0.5644 dB vs gap8 |

## Next Validation Plan

| next stage | goal | output |
|---:|---|---|
| 179 | broader sampled adaptive protocol | a larger target/schedule CSV with reuse/new-render/keyframe marker decisions |
| 180 | execute broader sampled validation | completed; 90-target final-quality comparison packaged |
| 181 | full-sequence RD accounting preflight | all-frame/keyframe/residual/metadata rate table |
| 182 | selector refinement or freeze decision | stricter gate/per-sequence budget if false positives are costly, otherwise freeze |
| 183 | paper-facing package | tables, figures, decoder contract, limitations, and subjective evidence paths |

## Canonical Paths

| item | path |
|---|---|
| Stage161 method package | `experiments/stage161_stage158_method_narrative_package/` |
| Stage162 selector audit | `experiments/stage162_keyframe_selector_protocol_source_audit/` |
| Stage172 rate audit | `experiments/stage172_keyframe_rate_accounting_audit/` |
| Stage176 adaptive candidate | `experiments/stage176_adaptive_schedule_candidate_package/` |
| Stage177 fixed-gap comparison | `experiments/stage177_selector_fixed_gap_psnr_comparison/` |
| Stage180 broader validation | `experiments/stage180_broader_sampled_adaptive_validation_execution/` |
| Stage160 subjective video | `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage160_stage158_extended_subjective_evidence/stage160_gap4_stage158_extended_subjective_evidence.mp4` |
| Stage160 contact sheet | `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage160_stage158_extended_subjective_evidence/stage160_gap4_stage158_extended_subjective_evidence_contact_sheet.jpg` |
