# Mono-DFCGS Results And Innovation Summary

Date: 2026-07-02

Purpose: consolidate the current Mono-DFCGS evidence chain, module-level innovation points, decoder contracts, non-claims, and next validation plan into one handoff log.

## Current Main Claim

The current strongest line is a StreamSplat-guided, Gaussian-domain, rate-counted middle-frame recovery method plus an encoder-side RGB/motion adaptive keyframe schedule.

The current measured full-sequence result is a middle RD point: adaptive improves full-sequence quality over uniform gap8 but uses higher measured rate, and it uses lower measured rate than uniform gap4 but has lower quality than gap4. Stage188 found lower-budget positive-quality sensitivity points, but the lowest positive point still remains above uniform gap8 rate under the additive sensitivity scope. Stage189 identifies the main failure modes for paper limitations and the next selector refinement. Stage190 packages the current paper-facing tables, decoder contract, claim boundaries, title, and abstract draft.

## Current Best Components

| component | current stage | policy | status |
|---|---:|---|---|
| middle-frame recovery | Stage158/161 | `streamsplat_guided_half_anchor_entropy_residual_v1` | current quality-first recovered middle-frame policy |
| keyframe selector protocol | Stage162 | transmitted schedule with encoder-side RGB/motion features | feature-source and decoder contract audited |
| adaptive keyframe schedule | Stage165/176 | `rgb_motion_rank_gate_gap8_plus_extra_targets_v1_sampled_candidate` | measured full-sequence middle RD point |
| fixed-gap comparison | Stage177 | adaptive vs uniform gap8/gap4 | sampled-medium final target quality comparison completed |
| broader sampled validation | Stage180 | adaptive vs uniform gap8/gap4 | 90-target sampled-broader final quality comparison completed |
| measured payload/RD | Stage184/185 | full-sequence payload measurement and RD aggregation | adaptive rate is between gap8 and gap4, not below gap8 |
| full-sequence quality | Stage186 | measured multi-metric quality for gap8/adaptive/gap4 | adaptive quality is between gap8 and gap4 |
| selector ablation | Stage187 | Stage163 label/protocol feature ablation | shortlist lower-budget variants for Stage188 |
| lower-budget sensitivity | Stage188 | interval/row-level additive measured reuse | positive lower-budget points found but gap8 rate not reached |
| failure-case analysis | Stage189 | promoted keyframe and residual-risk diagnostics | paper-facing failure cases and refinement targets identified |
| paper-facing package | Stage190 | tables, claims, decoder contract, title/abstract draft | current writing/slides handoff package |
| expanded fixed-gap protocol | Stage191 | gap2/gap4/gap6/gap8/gap16 plus adaptive protocol | missing measurement manifest for stronger baseline comparison |
| expanded fixed-gap measurement | Stage192 | measured RD-quality for gap2/gap4/gap6/gap8/gap16/adaptive | current adaptive does not beat best fixed gap |
| oracle upper bound | Stage193 | framewise and schedule-consistent oracles over Stage192 rows | current measured candidate space lacks `+1 dB` headroom |
| all-keyframe q12 upper bound | Stage194 | `uniform_gap1` q12 keyframes on all frames | q12 representation itself lacks `+1 dB` headroom |
| higher-fidelity keyframe upper bound | Stage195 | q16 and float dense-anchor keyframes on all frames | current dense-anchor/rendering representation lacks `+1 dB` headroom |
| target feasibility branch | Stage196 | Stage193-195 ceiling synthesis | selector/keyframe branch cannot meet requested target |
| learned GS compression protocol | Stage197 | GS-native predictive codec contract and Stage198-213 gates | new route rejects RGB/image residual final method |
| prior predictor audit | Stage198 | old adapter training and quality evidence | old adapter route rejected; new predictor required |
| learned GS training manifest | Stage199 | multi-gap train/eval task references | manifest ready for Stage200 architecture package |
| GS predictor architecture | Stage200 | `TemporalBasisGSRefiner` contract and module | selected for Stage201 predictor-only smoke |
| predictor-only smoke | Stage201 | q12 gap4/8 short training/rendered smoke | plumbing/no-regression passed; no learned gain yet |
| predictor-only broader validation | Stage202 | q12 multi-gap multi-config training-headroom check | predictor-only headroom not observed |
| GS residual codec design | Stage203 | counted GS attribute top-k residual entropy codec | selected for Stage204 smoke |
| residual codec smoke | Stage204 | real-task q6 top-k GS residual payload | positive rendered headroom with counted bytes |
| fixed-gap predictive validation | Stage205 | sampled fixed-gap q12 + GS residual RD-quality | positive headroom for gaps 4/8/12 |
| edge RD table | Stage206 | sampled edge-level q12 keyframe + counted GS residual costs | ready for Stage207 DP oracle |
| DP oracle schedule | Stage207 | residual-budget DP plus schedule connectivity audit | schedule graph insufficient; selector labels deferred |
| connected edge RD expansion | Stage206b | one connected bike-packing window | local DP graph ready |
| connected-window DP oracle | Stage207b | schedule DP over Stage206b graph | local selector-label feasibility passed |

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

## Stage181 Rate Accounting Preflight

Stage181 separates exact schedule/keyframe counts from sampled residual payload estimates.

Full-sequence keyframe and metadata counts over 30 DAVIS sequences / 1999 frames:

| schedule | keyframes | keyframe ratio | main anchor MiB/frame proxy | metadata bytes |
|---|---:|---:|---:|---:|
| uniform_gap8 | 292 | 0.146073 | 0.097625386754 | 1 |
| Stage165 adaptive | 358 | 0.179090 | 0.120431317266 | 327 |
| uniform_gap4 | 536 | 0.268134 | 0.181938220768 | 1 |

Combined proxy using Stage180 broader residual payload means:

| schedule | main anchor | metadata | residual proxy | total proxy | delta vs gap8 | delta vs gap4 |
|---|---:|---:|---:|---:|---:|---:|
| uniform_gap8 | 0.097625386754 | 0.000000000477 | 0.209692552355 | 0.307317939585 | 0.000000000000 | -0.061548490329 |
| Stage165 adaptive | 0.120431317266 | 0.000000156004 | 0.071214135488 | 0.191645608757 | -0.115672330828 | -0.177220821157 |
| uniform_gap4 | 0.181938220768 | 0.000000000477 | 0.186928208669 | 0.368866429914 | +0.061548490329 | 0.000000000000 |

Interpretation caveat:

- Keyframe indices and schedule metadata are exact for the Stage165 full schedule.
- Main-anchor payload remains Stage172 proxy/interpolated accounting.
- Residual payload remains Stage180 broader sampled estimate, not all-frame/full-sequence measurement.
- Final full-sequence RD still requires actual q12 keyframe bitstreams and all-frame Stage158 residual payload encodes.

## Stage182 Freeze Decision

Stage182 decides not to tune selector threshold/min-votes immediately. The current Stage165 adaptive policy is frozen for the next measurement.

Decision:

- `freeze_current_candidate_and_run_full_sequence_payload_measurement_next`

Frozen policy:

- `rgb_motion_rank_gate_gap8_plus_extra_targets_v1_sampled_candidate`

Evidence:

| evidence | value |
|---|---|
| Stage180 PSNR delta vs gap8 / gap4 | `+0.5644261202320328 / +0.306535729521994` dB |
| Stage180 LPIPS delta vs gap8 / gap4 | `-0.0338725696835253 / -0.019677375422583687` |
| Stage181 adaptive/gap8/gap4 proxy rate | `0.1916456087572328 / 0.3073179395851907 / 0.3688664299140155` MiB/frame |
| broader positive-promoted PSNR delta vs gap8 / gap4 | `+0.829806900266485 / +0.4439986578568402` dB |
| false-negative residual PSNR delta vs gap8 / gap4 | `-0.010963752281610173 / -0.32629360438858646` dB |
| false-positive keyframe control PSNR delta vs gap8 / gap4 | `+0.39611419615648114 / +0.2796445234109868` dB |

Interpretation:

- Current broader quality and rate proxy both support freezing the candidate.
- Remaining risks are final-RD measurement risks, not immediate selector-tuning blockers.
- Selector threshold tuning becomes conditional on exact full-sequence payload measurement showing rate regression or unacceptable false-positive cost.

## Stage183-188 Measured RD And Selector Sensitivity Update

Stage183-186 replace the Stage181 sampled/proxy rate optimism with measured full-sequence payload and quality.

Measured full-sequence RD-quality:

| schedule | MiB/frame | PSNR | SSIM | MS-SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| uniform_gap8 | `0.2758661759621266` | `29.373964871839835` | `0.867625699572828` | `0.9843430183660156` | `0.16869177970254404` |
| Stage165 adaptive | `0.2907429328258184` | `29.4255826920606` | `0.8692941793565335` | `0.9846469353830415` | `0.16593745923142186` |
| uniform_gap4 | `0.33076894444307725` | `29.535715839048734` | `0.8739438994697716` | `0.9855294218654929` | `0.15947172297849663` |

Interpretation:

- Adaptive improves all measured quality metrics over uniform gap8, but costs `+0.014876756863691831` MiB/frame.
- Adaptive is lower-rate than uniform gap4 by `-0.04002601161725883` MiB/frame, but is lower quality than gap4.
- The frozen Stage165 schedule should be described as an adaptive middle RD point, not as lower-rate than gap8.

## Selector Gain Interpretation: Sampled vs Full-Sequence

This note records the direct answer to the selector-module gain question. The earlier large selector gains came from Stage177/180 sampled validation on selector-relevant target frames, while the later Stage185/186 result is the full-sequence measured RD-quality table. These two views are complementary rather than contradictory.

Stage177 medium sampled validation:

| schedule | targets | adaptive/keyframe targets | PSNR | LPIPS | payload bytes/target |
|---|---:|---:|---:|---:|---:|
| uniform gap8 | `50` | `0` | `28.756927` | `0.189479` | `211348.720` |
| Stage165 adaptive | `50` | `26` | `29.217096` | `0.157234` | `95704.400` |
| uniform gap4 | `50` | `6` | `28.945814` | `0.177610` | `181242.240` |

Stage177 adaptive gains:

| comparison | PSNR gain | LPIPS gain |
|---|---:|---:|
| adaptive vs gap8 | `+0.4601686250283185` dB | `-0.03224517673254013` |
| adaptive vs gap4 | `+0.27128186202823` dB | `-0.02037559390068054` |

Stage180 broader sampled validation:

| schedule | targets | adaptive/keyframe targets | PSNR | LPIPS | payload bytes/target |
|---|---:|---:|---:|---:|---:|
| uniform gap8 | `90` | `0` | `29.206326` | `0.176652` | `219878.578` |
| Stage165 adaptive | `90` | `56` | `29.770753` | `0.142780` | `74673.433` |
| uniform gap4 | `90` | `8` | `29.464217` | `0.162457` | `196008.433` |

Stage180 adaptive gains:

| comparison | PSNR gain | LPIPS gain | payload reduction |
|---|---:|---:|---:|
| adaptive vs gap8 | `+0.5644261202320328` dB | `-0.0338725696835253` | about `-145205` bytes/target |
| adaptive vs gap4 | `+0.306535729521994` dB | `-0.019677375422583687` | about `-121335` bytes/target |

Stage186 full-sequence measured validation:

| schedule | MiB/frame | PSNR | SSIM | MS-SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| uniform gap8 | `0.2758661759621266` | `29.373964871839835` | `0.867625699572828` | `0.9843430183660156` | `0.16869177970254404` |
| Stage165 adaptive | `0.2907429328258184` | `29.4255826920606` | `0.8692941793565335` | `0.9846469353830415` | `0.16593745923142186` |
| uniform gap4 | `0.33076894444307725` | `29.535715839048734` | `0.8739438994697716` | `0.9855294218654929` | `0.15947172297849663` |

Stage186 adaptive gains/losses:

| comparison | rate delta | PSNR delta | SSIM delta | MS-SSIM delta | LPIPS delta |
|---|---:|---:|---:|---:|---:|
| adaptive vs gap8 | `+0.014876756863691831` MiB/frame | `+0.05161782022076622` dB | `+0.0016684797837055454` | `+0.0003039170170259231` | `-0.0027543204711221736` |
| adaptive vs gap4 | `-0.04002601161725883` MiB/frame | `-0.11013314698813303` dB | `-0.004649720113238054` | `-0.0008824864824513723` | `+0.006465736252925236` |

Writing interpretation:

- Use Stage177/180 when discussing how much the selector helps on targeted difficult/high-value frames.
- Use Stage185/186 as the final full-sequence RD-quality table.
- Do not claim Stage180 sampled gains as the full-sequence average.
- Do not claim the current selector beats gap8 in both rate and quality; full-sequence measured rate is higher than gap8.
- The honest claim is that the selector gives strong sampled-target gains and a measured full-sequence middle RD point between gap8 and gap4.

Stage187 feature ablation is label/protocol-only and does not claim measured RD for ablation schedules.

| variant | selected rows | keyframes | hard recall | payload recall | note |
|---|---:|---:|---:|---:|---|
| full Stage165 features | `70` | `358` | `0.7333333333333333` | `0.8194444444444444` | highest recall among evaluated variants |
| `drop_interp_rgb` | `69` | `357` | `0.7333333333333333` | `0.8055555555555556` | conservative low-budget Stage188 candidate |
| `motion_proxy_edge_hist` | `68` | `357` | `0.7333333333333333` | `0.7916666666666666` | small budget reduction |
| `edge_hist_only` | `67` | `356` | `0.7333333333333333` | `0.7777777777777778` | motion-only stress point |
| `drop_hist_motion` | `61` | `349` | `0.7333333333333333` | `0.7361111111111112` | more aggressive low-budget candidate |

Stage187 motivated Stage188 lower-budget schedule/RD sensitivity using Stage184/186 measured rows wherever schedule coverage allowed.

Stage188 evaluates lower-budget candidates using a separate additive sensitivity scope: `measured_single_anchor_additive_keyframes_plus_measured_stage158_residuals_plus_exact_metadata`. This scope is apples-to-apples across Stage188 candidates, but should not be mixed numerically with Stage185 schedule-packed keyframe rates.

Stage188 additive sensitivity points:

| candidate | keyframes | MiB/frame | delta rate vs gap8 | PSNR | delta PSNR vs gap8 | LPIPS | delta LPIPS vs gap8 | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| uniform_gap8 | `292` | `0.27588429100338135` | `0.0` | `29.373964871839874` | `0.0` | `0.16869177970254404` | `0.0` | additive baseline |
| `interval_top10pct_cells` | `299` | `0.2773746177516859` | `+0.0014903267483045712` | `29.38112562842953` | `+0.007160756589655648` | `0.16832458856830065` | `-0.0003671911342433831` | lowest-rate positive candidate |
| `interval_score_ge4p0` | `324` | `0.2829920490602662` | `+0.00710775805688485` | `29.41013285788653` | `+0.03616798604665661` | `0.16682702663534876` | `-0.001864753067195274` | balanced half-overhead candidate |
| Stage165 adaptive full | `358` | `0.29076559773798644` | `+0.014881306734605082` | `29.425582692060658` | `+0.05161782022078398` | `0.16593745923142186` | `-0.0027543204711221736` | full frozen selector under additive scope |

Stage188 decision: `lower_budget_positive_quality_candidates_found_but_gap8_rate_not_reached`.

## Stage189 Failure-Case Analysis

Stage189 analyzes why lower-budget adaptive variants remain above gap8 rate when they preserve positive quality and which frames/sequences define the remaining failure modes.

Candidate failure summary:

| candidate | keyframes | MiB/frame | delta rate vs gap8 | PSNR | delta PSNR vs gap8 | LPIPS | changed frames vs full | worst changed dPSNR vs full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `interval_top10pct_cells` | `299` | `0.2773746177516859` | `+0.0014903267483045712` | `29.38112562842953` | `+0.007160756589655648` | `0.16832458856830065` | `370` | `-5.309560997374348` |
| `interval_score_ge4p0` | `324` | `0.2829920490602662` | `+0.00710775805688485` | `29.41013285788653` | `+0.03616798604665661` | `0.16682702663534876` | `223` | `-2.3890793771766035` |
| `interval_top90pct_cells` | `353` | `0.289479501370253` | `+0.013595210366871668` | `29.424507356466457` | `+0.05054248462658251` | `0.16601754864188598` | `35` | `-1.2776672922430699` |

Promoted-keyframe risks:

- Promoted rows analyzed: `66`.
- Strong promoted rate-risk rows: `2`, specifically `drift-chicane` frame `6` and `horsejump-high` frame `15`.
- Interpretation: full adaptive overhead is not mostly explained by obvious bad promotions; only a small subset has small quality gain and large local payload delta.

Residual-risk hotspots:

| sequence | residual risks | low PSNR | high LPIPS | high payload | max residual risk | worst frame |
|---|---:|---:|---:|---:|---:|---:|
| `cows` | `86` | `86` | `1` | `62` | `1.904394667636713` | `92` |
| `parkour` | `75` | `27` | `26` | `74` | `1.5603596044904195` | `68` |
| `camel` | `73` | `61` | `0` | `67` | `1.3934650859013562` | `60` |
| `goat` | `73` | `0` | `0` | `73` | `0.78914` | `5` |
| `breakdance` | `72` | `72` | `2` | `0` | `1.7485984958204726` | `5` |
| `soapbox` | `72` | `0` | `1` | `71` | `0.67308` | `67` |
| `bmx-trees` | `67` | `3` | `6` | `67` | `0.70738` | `44` |

Stage189 decision: `failure_cases_identified_for_paper_and_next_selector_refinement`.

Interpretation:

- Lowest-rate Stage188 variants drop many adaptive cells and therefore change many frames relative to full adaptive.
- Remaining unpromoted residual risks are broader than promoted rate risks, especially low-PSNR `cows`/`breakdance`/`camel` frames and high-LPIPS/high-payload `motocross-jump`/`india` frames.
- A next selector refinement should not only suppress rare bad promotions; it should also target residual-risk hotspots without losing too much rate.

## Stage190 Paper-Facing Package

Stage190 packages the current evidence into a paper/slides handoff under `experiments/stage190_paper_facing_package/`.

Recommended title:

- `Mono-DFCGS: Recovery-Aware Adaptive Keyframe Scheduling for Monocular Dynamic Gaussian Splatting Compression`

Primary output:

- Report: `experiments/stage190_paper_facing_package/stage190_paper_facing_report.md`
- Package JSON: `experiments/stage190_paper_facing_package/stage190_paper_facing_package.json`

Table counts:

| table | rows |
|---|---:|
| measured RD-quality | `3` |
| selector ablation | `8` |
| lower-budget sensitivity | `6` |
| candidate failures | `3` |
| promoted rate risks | `2` |
| residual hotspots | `10` |
| claims and limitations | `9` |

Stage190 decision: `paper_facing_tables_and_claim_boundaries_packaged`.

Writing stance:

- Use Stage190 as the current paper-facing source of truth.
- Frame the adaptive result as a measured middle RD point, not a gap8-rate improvement.
- Keep Stage188 additive sensitivity separate from Stage185 schedule-packed RD.
- Keep decoder allowed/forbidden inputs explicit in the method section.

## Stage191 Expanded Fixed-Gap Protocol

The user requested a stronger selector validation: full-sequence results should compare against more than gap4/gap8, and the adaptive selector should show clear gains over the best tested fixed-gap schedule before claiming the module is effective.

Stage191 creates the protocol for `uniform_gap2`, `uniform_gap4`, `uniform_gap6`, `uniform_gap8`, `uniform_gap16`, and current `stage165_adaptive` over the same 30 DAVIS validation sequences / 1999 frames.

Schedule counts:

| schedule | keyframes | residual rows | keyframe ratio | metadata bytes |
|---|---:|---:|---:|---:|
| `uniform_gap2` | `1025` | `974` | `0.5127563781890946` | `1` |
| `uniform_gap4` | `536` | `1463` | `0.2681340670335168` | `1` |
| `uniform_gap6` | `372` | `1627` | `0.18609304652326164` | `1` |
| `uniform_gap8` | `292` | `1707` | `0.14607303651825912` | `1` |
| `uniform_gap16` | `169` | `1830` | `0.08454227113556778` | `1` |
| `stage165_adaptive` | `358` | `1641` | `0.1790895447723862` | `327` |

Reuse/missing measurement coverage for Stage192:

| scope | expected | existing ok | missing | reuse fraction |
|---|---:|---:|---:|---:|
| payload single keyframe | `1065` | `596` | `469` | `0.5596244131455399` |
| payload residual | `7791` | `3472` | `4319` | `0.44564240790655885` |
| payload schedule-packed keyframe group | `180` | `90` | `90` | `0.5` |
| quality single keyframe | `1065` | `596` | `469` | `0.5596244131455399` |
| quality residual | `7791` | `3472` | `4319` | `0.44564240790655885` |

Stage191 decision: `measure_expanded_fixed_gap_baselines_next`.

Next stance:

- Stage192 must measure the missing gap2/gap6/gap16 rows and aggregate expanded fixed-gap RD-quality.
- Stage193 should compute oracle upper bounds before training/tuning a new selector.
- The target is no longer just a middle RD point; the next selector should aim to beat the best tested fixed-gap schedule on full-sequence PSNR by a large margin, ideally around `+1 dB`, without SSIM/MS-SSIM/LPIPS regressions.

## Stage192 Expanded Fixed-Gap Measurement

Stage192 completes the expanded full-sequence measurement requested by the user. It measures missing gap2/gap6/gap16 payload and quality rows, reuses Stage184/186 where possible, and aggregates all schedules under the same measured scope.

Validation:

- All schedules have `1999/1999` final quality rows.
- Unique keyframe payload/quality rows: `1065/1065`.
- Unique residual payload/quality rows: `7791/7791`.
- Schedule-packed keyframe payload groups are complete for all expanded schedules.

Expanded measured RD-quality:

| schedule | MiB/frame | keyframes | PSNR | SSIM | MS-SSIM | LPIPS | dPSNR vs best fixed | dLPIPS vs best fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `uniform_gap2` | `0.4495468821866683` | `1025` | `29.654815328772308` | `0.878375951948018` | `0.9866168332910943` | `0.15168131759139583` | `0.0` | `0.0` |
| `uniform_gap4` | `0.33076894444307725` | `536` | `29.535715839048734` | `0.8739438994697716` | `0.9855294218654929` | `0.15947172297849663` | `-0.11909948972357398` | `0.007790405387100796` |
| `uniform_gap6` | `0.29344506237493745` | `372` | `29.448737801531657` | `0.8706193330169857` | `0.9848943316024086` | `0.16457491443157493` | `-0.20607752724065165` | `0.0128935968401791` |
| `uniform_gap8` | `0.2758661759621266` | `292` | `29.373964871839835` | `0.867625699572828` | `0.9843430183660156` | `0.16869177970254404` | `-0.28085045693247324` | `0.017010462111148206` |
| `uniform_gap16` | `0.2514788301781811` | `169` | `29.199328663742687` | `0.8603573565843285` | `0.9830296424521751` | `0.1773263292425331` | `-0.4554866650296212` | `0.02564501165113728` |
| `stage165_adaptive` | `0.2907429328258184` | `358` | `29.4255826920606` | `0.8692941793565335` | `0.9846469353830415` | `0.16593745923142186` | `-0.22923263671170702` | `0.014256141640026032` |

Stage192 decision: `current_adaptive_not_strong_against_expanded_fixed_gaps`.

Interpretation:

- The best fixed gap by PSNR is `uniform_gap2`, and current adaptive is below it by `-0.22923263671170702` dB PSNR with worse LPIPS by `+0.014256141640026032`.
- `uniform_gap6` is a near-rate baseline: rate `0.29344506237493745` MiB/frame vs adaptive `0.2907429328258184`, but gap6 has higher PSNR (`29.448737801531657` vs `29.4255826920606`) and lower LPIPS (`0.16457491443157493` vs `0.16593745923142186`).
- Therefore the old Stage165 selector cannot support a strong selector-module claim. Stage193 must compute oracle headroom before new selector design.

## Stage193 Oracle Upper-Bound Analysis

Stage193 tests whether a perfect selector over the current measured Stage158/fixed-gap candidate space could reach the user's requested full-sequence target: roughly `+1 dB` PSNR over the best tested fixed gap without SSIM/MS-SSIM/LPIPS regression.

Oracle summary:

| oracle | schedule-consistent | PSNR | SSIM | MS-SSIM | LPIPS | dPSNR vs `uniform_gap2` | dLPIPS vs `uniform_gap2` | +1dB/no-regression pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `framewise_psnr_oracle` | `0` | `29.749038180432017` | `0.8816605037662493` | `0.9870865034603846` | `0.14641779118318626` | `+0.09422285165970834` | `-0.005263526408209568` | `0` |
| `schedule_path_psnr_oracle` | `1` | `29.670134277041633` | `0.8788577944949724` | `0.9867078401912386` | `0.1508482470754357` | `+0.015318948269325006` | `-0.000833070515960127` | `0` |

Stage193 decision: `framewise_oracle_upper_bound_below_target_margin`.

Interpretation:

- Even the optimistic non-schedule-consistent framewise oracle is far below the requested `+1 dB` margin over best fixed `uniform_gap2`.
- Tuning the current selector over the current measured candidate space cannot plausibly produce a strong full-sequence selector claim.
- The next diagnostic should test a stronger representation upper bound, such as all-frame q12 keyframes (`gap1`), before continuing selector-only optimization.

## Stage194 All-Keyframe Q12 Upper-Bound

Stage194 measures `uniform_gap1`, where every DAVIS validation frame is coded as a q12 keyframe. This is an upper-bound diagnostic for q12 keyframe representation quality, not a practical adaptive selector.

Validation:

- Protocol frames: `1999/1999`.
- Unique keyframe payload rows: `1999/1999`.
- Unique keyframe quality rows: `1999/1999`.
- Schedule-packed keyframe groups: `30/30`.

All-keyframe q12 result:

| schedule | MiB/frame | keyframes | PSNR | SSIM | MS-SSIM | LPIPS | dPSNR vs `uniform_gap2` | dLPIPS vs `uniform_gap2` | +1dB/no-regression pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uniform_gap1` | `0.686173294471943` | `1999` | `29.85646819580043` | `0.8857439302277005` | `0.9884901848240099` | `0.13756442577459324` | `+0.20165286702812324` | `-0.01411689181680259` | `0` |

Stage194 decision: `all_keyframe_q12_improves_gap2_but_below_target_margin`.

Interpretation:

- Stage194 confirms Stage193's implication: selector tuning alone is insufficient.
- Even replacing every frame with a q12 keyframe improves best fixed gap2 by only about `+0.202 dB`, far below the requested `+1 dB`.
- The next useful diagnostic is higher-fidelity representation headroom, e.g. all-keyframe q16 and/or float dense-anchor quality, before designing another adaptive schedule.

## Stage195 Higher-Fidelity Keyframe Upper-Bound

Stage195 tests whether the missing headroom is only q12 quantization. It renders all `1999` frames as q16 dense-anchor keyframes and as float dense-anchor keyframes. q16 includes measured schedule-packed keyframe rate; float is quality-only with no deployable payload claim.

Validation:

- q16 schedule-packed keyframe groups: `30/30`.
- q16 quality rows: `1999/1999`.
- float dense-anchor quality rows: `1999/1999`.

Higher-fidelity upper bounds:

| representation | MiB/frame | PSNR | SSIM | MS-SSIM | LPIPS | dPSNR vs `uniform_gap2` | dLPIPS vs `uniform_gap2` | +1dB/no-regression pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `q16_keyframe` | `0.9146932160156617` | `29.884665362865746` | `0.8868697442192623` | `0.9886300584386145` | `0.13601533181894535` | `+0.22985003409343818` | `-0.015665985772450486` | `0` |
| `float_dense_anchor` | `NA` | `29.88493146578025` | `0.8868824350291219` | `0.9886313845897806` | `0.13598978392716465` | `+0.23011613700794342` | `-0.015691533664231178` | `0` |

Stage195 decision: `higher_fidelity_keyframes_improve_gap2_but_below_target_margin`.

Interpretation:

- The requested target is best fixed gap2 plus `+1 dB`, i.e. PSNR `30.654815328772308` without metric regression.
- q16 and float dense-anchor all-keyframe quality plateau at about `29.885` dB, so quantization is not the blocker.
- The current dense-anchor/rendering representation itself lacks enough full-sequence quality ceiling for the requested claim.
- Next work must change the reconstruction objective/model or introduce a different counted correction payload; selector-only or keyframe-quantization-only work should stop.

## Stage196 Target Feasibility Branch

Stage196 converts the Stage193-195 upper bounds into a branch decision.

Requested target:

- Best fixed reference: `uniform_gap2`.
- Target PSNR: `30.654815328772308` (`uniform_gap2 + 1 dB`).

Ceilings:

| source | ceiling | PSNR | dPSNR vs gap2 | dPSNR vs target | pass |
|---:|---|---:|---:|---:|---:|
| 193 | `framewise_psnr_oracle` | `29.749038180432017` | `+0.09422285165970834` | `-0.9057771483402917` | `0` |
| 193 | `schedule_path_psnr_oracle` | `29.670134277041633` | `+0.015318948269325006` | `-0.984681051730675` | `0` |
| 194 | `all_q12_keyframes` | `29.85646819580043` | `+0.20165286702812324` | `-0.7983471329718768` | `0` |
| 195 | `q16_keyframe` | `29.884665362865746` | `+0.22985003409343818` | `-0.7701499659065618` | `0` |
| 195 | `float_dense_anchor` | `29.88493146578025` | `+0.23011613700794342` | `-0.7698838629920566` | `0` |

Stage196 decision: `selector_keyframe_representation_cannot_meet_target`.

Branch options:

- `selector_or_keyframe_quantization_tuning`: stop; cannot honestly claim large full-sequence selector gain over best fixed gap.
- `counted_rgb_or_image_residual_correction`: viable next diagnostic; likely source of headroom based on Stage155 sampled image-residual upper bound, but less Gaussian-native.
- `new_dense_anchor_reconstruction_objective_or_model`: viable but heavy; needed if the final method must remain GS-native without RGB residual payload.
- `paper_claim_scope_adjustment`: viable writing fallback; supports sampled-target selector gains and measured middle RD point, but not the requested stronger result.

## Stage197 Learned GS Compression Protocol

The user rejected RGB/image residual post-processing as a final method and approved the GS-native learned predictive compression route through Stage213.

Stage197 decision: `primary_gs_native_predictive_codec_protocol_defined`.

Primary runtime decoder contract:

- Allowed: transmitted q-keyframe GS bitstreams, transmitted schedule metadata, normalized time, shared GS predictor/refiner weights, and transmitted counted GS latent/residual bitstreams.
- Forbidden: target dense anchor, target RGB/image residual, oracle schedule/quality labels.
- Training/encoder only: target dense anchor and RGB may supervise predictors, residual payload generation, and selector labels.
- StreamSplat checkpoint: allowed as initialization/teacher or optional diagnostic base, but raw RGB-dependent StreamSplat runtime is not the primary final codec claim.

Stage gates:

- Stage198: audit old predictor training failures.
- Stage199: build learned GS training manifest.
- Stage200: package new predictor/refiner architecture.
- Stage201-205: validate predictor and GS-native residual before selector training.
- Stage206: build edge RD table.
- Stage207: compute DP oracle schedules; current sampled graph is insufficient.
- Stage206b/207b: connected-window expansion and local DP oracle pass.
- Stage208-210: train selector and residual budget allocator.
- Stage211-213: full measured RD, ablations, and subjective visuals.

## Stage198 Prior Predictor Training Audit

Stage198 audits whether the old decoder-side adapter should be continued.

Stage198 decision: `old_adapter_route_rejected_new_predictor_required`.

Key evidence:

- Old q12 adapter middle PSNR was gap4 `18.256196169477683` and gap8 `17.06969395261803`.
- q12-to-float32 old-adapter middle PSNR changed by only gap4 `+0.00004890061461537698` dB and gap8 `-0.000016654591867393265` dB.
- Float32 dense-direct minus adapter remained huge: gap4 `+11.49432172291868` dB and gap8 `+12.677631798990458` dB.
- Stage145 continued training over `6691` train rows improved only `+0.011427207695934527` dB.
- Stage146 continuation had best step `0` and final change `-0.019897326878577533` dB.
- Stage157/158 success came from counted GS-domain residual side-info over StreamSplat base, not from the old adapter predictor alone.

New route requirements:

- Predictor-only gate before selector training.
- GS-native residual payload, not image residual.
- Edge-RD oracle headroom before learned selector training.
- Full-sequence measured PSNR/SSIM/MS-SSIM/LPIPS and bytes for any strong claim.

## Stage199 Learned GS Training Manifest

Stage199 packages the lightweight data references for the GS-native predictive codec route without copying anchors or tensors.

Stage199 decision: `manifest_ready_for_stage200_architecture_package`.

Coverage:

- Total q12 task rows: `29204`.
- Missing references: `0`.
- Train split: `60` sequences, `4209` frames.
- Eval split: `30` sequences, `1999` frames.
- Gaps covered in both splits: `2,4,6,8,12,16`.
- Eval tasks by gap: gap2 `974`, gap4 `1463`, gap6 `1627`, gap8 `1707`, gap12 `1788`, gap16 `1830`.
- Train tasks by gap: gap2 `2055`, gap4 `3087`, gap6 `3430`, gap8 `3604`, gap12 `3776`, gap16 `3863`.

Contract audit:

- Dense-anchor coverage: pass.
- RGB-label coverage: pass.
- Split separation: pass.
- Gap coverage: pass.
- Stage197 decoder contract: pass.
- Lightweight-reference-only: pass.

Decoder notes:

- Runtime decoder inputs remain transmitted GS keyframes, schedule metadata, normalized time, shared weights, and counted GS-native latent/residual payloads.
- Target dense anchors and target RGB are training/encoder-side labels only.
- Target dense anchors, target RGB/image residuals, and oracle schedule/quality labels remain forbidden decoder inputs.

## Stage200 GS Predictor Architecture Package

Stage200 defines the new predictor/refiner candidate for the learned GS route.

Stage200 decision: `primary_temporal_basis_refiner_v1_selected_for_stage201_smoke`.

Primary architecture: `temporal_basis_gs_refiner_v1`, implemented as `mono_dfcgs.learned_gs_predictor.TemporalBasisGSRefiner`.

Core design:

- Linear q-keyframe interpolation base.
- Endpoint-gated residual factor `t*(1-t)`.
- Local per-Gaussian features: left, right, base, diff, absdiff, and time basis.
- Global decoded-endpoint GS statistics.
- Zero-initialized residual head for linear fallback before training.

CPU architecture smoke:

- Diagnostic parameter count for hidden96/global32 instance: `43501`.
- Endpoint identity t0 max abs delta: `0.0`.
- Endpoint identity t1 max abs delta: `0.0`.
- Zero-init midpoint linear fallback max abs delta: `0.0`.

Stage201 protocol:

- Input manifest: `experiments/stage199_learned_gs_training_manifest/stage199_learned_gs_training_tasks.csv`.
- Gaps: `4 8`.
- Keyframe codec: `q12`.
- Payload: none; predictor-only; zero per-frame side-info.
- Heavy outputs: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage201_predictor_only_smoke`.

## Stage201 Predictor-Only Smoke

Stage201 runs a small q12 gap4/gap8 predictor-only smoke for `TemporalBasisGSRefiner`.

Stage201 decision: `predictor_only_smoke_passed_no_regression_gate`.

Run scope:

- Train tasks: `8`.
- Eval tasks: `8`.
- Train steps: `16`.
- Hidden/global dims: `192/64`.
- Render loss weight: `0.02`.
- Per-frame payload bytes: `0`.
- Best/final checkpoints are outside git under `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage201_predictor_only_smoke/`.

Rendered PSNR summary:

- Linear eval PSNR: `20.52573876541323`.
- Predictor-best eval PSNR: `20.52573876541323` at step `0`.
- Predictor-final eval PSNR: `20.509261273864784`.
- Predictor-best eval delta vs linear: `0.0` dB.
- Predictor-final eval delta vs linear: `-0.016477491548446466` dB.
- Predictor-best eval sanity delta vs historical Stage78 q12 old adapter reference over gap4/8: `+2.862793704365373` dB, but this is protocol-different and only a sanity floor.

Gate interpretation:

- Metric rows ok: pass.
- Endpoint identity: pass.
- Predictor-only payload: pass, `0` bytes per frame.
- Best eval no-regression vs linear: pass.
- Final checkpoint outside git: pass.
- Important: Stage201 proves executable/stable plumbing and no-regression only. It does not prove learned predictor quality improvement because best step is `0`; Stage202 must test longer/broader training headroom.

## Stage202 Predictor-Only Broader Validation

Stage202 tests whether `TemporalBasisGSRefiner` has predictor-only rendered headroom beyond the Stage201 smoke.

Stage202 decision: `predictor_only_broader_training_headroom_not_observed`.

Run scope:

- Gaps: `2,4,8,12`.
- Keyframe codec: `q12`.
- Train/eval tasks: `16` / `16`.
- Configs: `3`.
- Per-frame payload bytes: `0`.

Config results:

- Shared linear eval PSNR: `19.239936914617395`.
- `anchor_render_lr2e4`: best delta `0.0` dB, final delta `-0.0014714499478820642` dB, best step `0`.
- `anchor_only_lr1e3`: best delta `+0.0006680372369380905` dB, final delta `+0.0006680372369380905` dB, best step `32`.
- `render_heavy_lr1e4`: best delta `0.0` dB, final delta `-0.0008068938220269217` dB, best step `0`.

Gates:

- Metric rows ok: pass.
- Endpoint identity all configs: pass.
- Predictor-only payload: pass.
- Any-config no-regression: pass.
- Predictor headroom positive: fail; best delta `+0.0006680372369380905` dB is far below the `>0.05` dB gate.

Interpretation:

- Predictor-only learned quality headroom is not observed.
- Stage203 must prioritize GS-native residual/latent side-info.
- Selector training remains deferred until residual/edge RD oracle headroom exists.

## Stage203 GS Latent/Residual Codec Design

Stage203 defines the counted GS-native residual codec after predictor-only headroom was not observed.

Stage203 decision: `gs_attr_topk_residual_entropy_v1_selected_for_stage204_smoke`.

Primary codec:

- Implementation: `mono_dfcgs.residual_sideinfo_codec.encode_topk_residual_sideinfo_entropy`.
- Decoder: `mono_dfcgs.residual_sideinfo_codec.decode_residual_sideinfo_entropy`.
- Encoder-side sources: predictor/base GS, target dense anchor, optional target RGB diagnostics.
- Decoder inputs: predictor/base GS plus transmitted residual payload.
- Forbidden decoder inputs: target dense anchor, target RGB/image residual, oracle quality labels.
- Payload accounting: `payload_bytes = len(payload)` and include every transmitted header/metadata/index/value/compressed byte.

Toy roundtrips:

- `gs_attr_topk_residual_entropy_v1`: payload `246` bytes, MSE before/after `0.006919591687619686/0.0008317023166455328`, reduction `0.8798047118685358`.
- `gs_attr_deterministic_index_residual_entropy_v1`: payload `217` bytes, MSE before/after `0.007950554601848125/0.0008250265964306891`, reduction `0.8962303087335681`.

Stage204 protocol:

- Base: linear or zero-init `TemporalBasisGSRefiner` predictor base from Stage201/202.
- Gaps: `4 8`.
- Codec: `q12` keyframes plus `gs_attr_topk_residual_entropy_v1` residual payload.
- Settings: `side_bits=6`, keep fractions `0.05,0.10,0.20`, `zlib_level=9`.
- Metrics: rendered PSNR plus counted payload bytes and residual MSE reduction.

## Stage204 Residual Codec Smoke

Stage204 tests the selected GS-native residual codec on real Stage199 q12 tasks.

Stage204 decision: `residual_codec_smoke_positive_headroom`.

Scope:

- Eval tasks: `12`.
- Gaps: `4,8`.
- Keyframe codec: `q12`.
- Base: linear/zero-init predictor base.
- Residual codec: `gs_attr_topk_residual_entropy_v1`.
- Side bits: `6`.
- zlib level: `9`.

Summary:

- Base mean PSNR: `19.999033822428466`.
- `topk_keep0p05_q6`: mean payload `15679.583333333334` bytes, corrected PSNR `22.39462808214667`, dPSNR `+2.3955942597182047`, anchor MSE reduction `0.5102364961382247`.
- `topk_keep0p1_q6`: mean payload `29836.75` bytes, corrected PSNR `23.804629721924517`, dPSNR `+3.8055958994960553`, anchor MSE reduction `0.688443254187152`.
- `topk_keep0p2_q6`: mean payload `55761.833333333336` bytes, corrected PSNR `25.552135029430517`, dPSNR `+5.553101207002054`, anchor MSE reduction `0.8435545876871436`.

Gates:

- Metric rows ok: pass.
- Payload counted and nonzero: pass.
- Residual anchor MSE reduction positive: pass.
- Residual render headroom positive: pass.
- Stage197 decoder contract: pass.

Interpretation:

- GS-native residual payload has real rendered quality headroom.
- Stage205 should validate fixed-gap predictive codec RD with exact payload bytes included in total rate.

## Stage205 Fixed-Gap Predictive Codec Validation

Stage205 validates fixed-gap predictor-plus-GS-residual behavior on sampled real tasks before Stage206 edge RD tables.

Stage205 decision: `fixed_gap_predictive_codec_positive_headroom`.

Scope:

- Eval tasks: `24`, with `8` each for gaps `4,8,12`.
- Keyframe codec: `q12`.
- Residual settings: q6 top-k keep fractions `0.05,0.10,0.20`.
- Scope caveat: sampled fixed-gap validation, not full-sequence RD.

Best by gap:

- Gap4 best: `topk_keep0p2_q6`, payload `56283.0` bytes (`0.05367565155029297` MiB/intermediate), base/corrected PSNR `20.599420543944056/25.301132561904456`, dPSNR `+4.701712017960398`.
- Gap8 best: `topk_keep0p2_q6`, payload `58359.875` bytes (`0.0556563138961792` MiB/intermediate), base/corrected PSNR `17.667283985227805/23.694548271579496`, dPSNR `+6.027264286351697`.
- Gap12 best: `topk_keep0p2_q6`, payload `57028.5` bytes (`0.05438661575317383` MiB/intermediate), base/corrected PSNR `19.19559473773822/24.991767735164355`, dPSNR `+5.796172997426137`.

Gates:

- Metric rows ok: pass.
- Payload counted and nonzero: pass.
- Gap coverage: pass.
- Each-gap positive headroom: pass.
- Stage197 decoder contract: pass.

Interpretation:

- Fixed-gap sampled evidence supports building Stage206 edge RD tables with exact keyframe/residual costs.

## Stage206 Edge RD Table

Stage206 converts sampled fixed-gap target evidence into edge-level RD rows for Stage207 DP schedule optimization.

Stage206 decision: `edge_rd_table_ready_for_stage207_dp`.

Scope:

- Eval edges: `6`, with `2` sampled edges each for gaps `4,8,12`.
- Internal target rows: `37`, covering every manifest target inside selected edges.
- Keyframe codec: q12 endpoint anchors measured with `encode_anchor_bitstream(..., bits=12, payload_encoding=bitpack)`.
- Residual settings: q6 top-k keep fractions `0.05,0.10,0.20`.
- Schedule metadata: provisionally counted as `2` bytes per edge.
- Scope caveat: sampled edge RD preflight, not final full-sequence RD.

Accounting:

- `edge_total_bytes_once = left_keyframe_bytes + right_keyframe_bytes + residual_payload_bytes + schedule_metadata_bytes`.
- `dp_incremental_bytes = right_keyframe_bytes + residual_payload_bytes + schedule_metadata_bytes`; Stage207 must add the initial left keyframe once per path.
- Residual bytes are exact `len(payload)` from the transmitted GS-native residual payload.

Best by gap:

- Gap4 best: `topk_keep0p2_q6`, edge total `1604771.0` bytes, DP incremental `885123.0` bytes, residual `165471.0` bytes, corrected PSNR `26.59671835498372`, dPSNR `+5.301434075900853`.
- Gap8 best: `topk_keep0p2_q6`, edge total `1701646.5` bytes, DP incremental `981999.5` bytes, residual `262346.5` bytes, corrected PSNR `22.501153480651734`, dPSNR `+4.267633006441421`.
- Gap12 best: `topk_keep0p2_q6`, edge total `2052070.0` bytes, DP incremental `1332423.5` bytes, residual `612781.0` bytes, corrected PSNR `26.07283417481182`, dPSNR `+4.164336484774056`.

Gates:

- Stage205 prereq: pass.
- Target metric rows ok: pass.
- Edge rows ok: pass.
- Gap coverage: pass.
- Keyframe/residual payload counted and nonzero: pass.
- Schedule metadata counted: pass.
- Each-gap positive edge headroom: pass.
- Stage197 decoder contract: pass.

Interpretation:

- Stage206 is sufficient to start Stage207 DP oracle schedule construction, but it is not full-sequence RD.

## Stage207 DP Oracle Schedule

Stage207 tests whether the Stage206 sampled edge table can support oracle schedule construction before selector-label generation.

Stage207 decision: `dp_oracle_schedule_graph_insufficient`.

What passed:

- Stage206 prerequisite: pass (`edge_rd_table_ready_for_stage207_dp`).
- Edge option coverage: pass (`18/18` options from `6` edges and `3` residual settings).
- Fixed baselines present: pass (`topk_keep0p05_q6`, `topk_keep0p1_q6`, `topk_keep0p2_q6`).
- Residual-budget oracle: pass as a budget allocator preflight; best same-budget gain was `+0.017853956775965685` dB at the `topk_keep0p1_q6` budget.
- Decoder contract: pass.

Fixed setting baselines over `37` sampled internal targets:

- `topk_keep0p05_q6`: cost `4905660` bytes, mean PSNR `22.519469828354364`, mean dPSNR `+1.627647700365514`.
- `topk_keep0p1_q6`: cost `5430475` bytes, mean PSNR `23.633188273126244`, mean dPSNR `+2.7413661451373974`.
- `topk_keep0p2_q6`: cost `6399092` bytes, mean PSNR `25.382111703299287`, mean dPSNR `+4.490289575310435`.

Blocker:

- Schedule graph connectivity failed: `0` connected edge transitions.
- Current Stage206 sampled edges are isolated (`bike-packing`, `dog`, `dogs-jump`, `paragliding-launch`, `parkour`) and cannot form nontrivial schedule paths.
- Therefore Stage207 cannot honestly claim an oracle schedule that beats fixed baselines, and Stage208/209 selector training must remain deferred.

Required next work:

- Build an expanded connected edge RD table over at least one sequence/window with contiguous candidate edges.
- Rerun Stage207 DP oracle after connected edge coverage exists.

## Stage206b/207b Connected-Window DP Rerun

Stage206b builds the first small connected edge table after Stage207 found isolated-edge coverage insufficient.

Stage206b decision: `edge_rd_table_ready_for_stage207_dp`.

Stage206b scope:

- Window: `bike-packing:00000:00024`.
- Edges: `11`.
- Internal target rows: `61`.
- Gap4 chain edges: `6`.
- Connected edge transitions: `14`.
- Settings: q6 top-k residual keep fractions `0.05,0.10,0.20` with q12 endpoint keyframes and counted `2` metadata bytes per edge.

Stage206b best by gap:

- Gap4 best: `topk_keep0p2_q6`, corrected PSNR `26.031263851493254`, dPSNR `+4.986391989862658`.
- Gap8 best: `topk_keep0p2_q6`, corrected PSNR `25.821074970091768`, dPSNR `+5.676789260065095`.
- Gap12 best: `topk_keep0p2_q6`, corrected PSNR `25.40221496330657`, dPSNR `+5.768186956092006`.

Stage207b decision: `dp_oracle_schedule_ready_for_selector_labels`.

Stage207b results:

- Edge option coverage: `33/33` pass.
- Schedule graph connectivity: pass with `14` connected transitions, `1` component, `11` edges, `7` nodes.
- Fixed baselines over `61` internal targets:
  - `topk_keep0p05_q6`: cost `8893010` bytes, mean PSNR `23.104662593777928`, mean dPSNR `+2.8786575187184025`.
  - `topk_keep0p1_q6`: cost `9764534` bytes, mean PSNR `24.43813548907444`, mean dPSNR `+4.212130414014913`.
  - `topk_keep0p2_q6`: cost `11344328` bytes, mean PSNR `25.73203365395984`, mean dPSNR `+5.5060285789003105`.
- Same-budget residual allocation gain: `+0.016174288512960544` dB at the `topk_keep0p1_q6` budget.

Interpretation:

- Local connected-window DP plumbing is validated.
- This is not yet robust selector-training evidence; next step should expand to multi-sequence connected windows before full Stage208/209 promotion.

## Non-Claims And Risks

| item | status |
|---|---|
| final optimized adaptive RD | not claimed yet; Stage165 measured point is between gap8 and gap4 |
| all-frame quality report | completed for Stage165/gap8/gap4 in Stage186 |
| `+1 dB` full-sequence selector headroom in current candidate space | rejected by Stage193 oracle analysis |
| `+1 dB` full-sequence headroom from all q12 keyframes | rejected by Stage194 all-keyframe q12 upper bound |
| `+1 dB` full-sequence headroom from q16/float keyframes | rejected by Stage195 higher-fidelity upper bound |
| continuing selector/keyframe quantization branch for requested strong claim | rejected by Stage196 target feasibility branch |
| RGB/image residual post-processing final method | rejected by user and Stage197 protocol |
| continuing old `GaussianAnchorDynamicPredictor` unchanged | rejected by Stage198 audit |
| Stage199 manifest as model improvement | not claimed; it is a data/contract package for Stage200+ |
| Stage200 architecture as quality result | not claimed; quality gate begins at Stage201 predictor-only smoke |
| Stage201 predictor learned improvement | not claimed; best checkpoint is step `0` linear fallback |
| Stage202 predictor-only headroom | not observed; best delta is only `+0.0006680372369380905` dB |
| Stage203 residual codec quality on real frames | not claimed yet; Stage203 only designs and toy-tests the codec |
| Stage204 full-sequence RD | not claimed yet; Stage204 is a 12-task smoke and must be followed by Stage205+ |
| Stage205 full-sequence RD | not claimed; Stage205 is sampled fixed-gap validation only |
| Stage206 full-sequence RD | not claimed; Stage206 is sampled edge-level RD preflight for DP |
| Stage207 selector-label readiness | not claimed; schedule graph connectivity failed, so selector labels are deferred |
| Stage207b full selector-training readiness | not claimed; local connected-window DP passed but scope is one window only |
| selector precision solved | not claimed; Stage189 finds only `2/66` strong promoted rate-risk rows, but precision remains a tuning target |
| false negatives solved | not claimed; Stage189 finds `1179` residual-risk rows and sequence hotspots |
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
| 181 | rate accounting preflight | adaptive combined proxy 0.1916 MiB/frame, below gap8/gap4 under Stage180 residual proxy |
| 182 | freeze decision | freeze current adaptive selector and run full-sequence payload measurement next |
| 183 | full-sequence payload protocol | enumerates 5997 schedule/frame rows and unique keyframe/residual payload work |
| 184 | measured payload execution | measures all unique q12 keyframe and Stage158 residual payloads plus schedule-packed keyframe streams |
| 185 | measured RD aggregation | adaptive measured rate is between gap8 and gap4, not below gap8 |
| 186 | full-sequence quality validation | adaptive quality is between gap8 and gap4 across PSNR/SSIM/MS-SSIM/LPIPS |
| 187 | selector feature ablation | identifies lower-budget candidate features for Stage188 sensitivity |
| 188 | lower-budget selector sensitivity | positive lower-budget points found under additive scope, but gap8 rate not reached |
| 189 | failure-case analysis | identifies promoted rate-risk rows, residual-risk hotspots, and candidate-specific dropped-frame losses |
| 190 | paper-facing package | packages tables, claim boundaries, decoder contract, recommended title, and abstract draft |
| 191 | expanded fixed-gap protocol | prepares gap2/gap4/gap6/gap8/gap16/adaptive full-sequence measurement and missing-row manifest |
| 192 | expanded fixed-gap measurement | measures expanded fixed-gap RD-quality and shows current adaptive is not strong enough |
| 193 | oracle upper-bound analysis | shows current measured candidate space cannot reach the requested `+1 dB` full-sequence target |
| 194 | all-keyframe q12 upper bound | shows q12 keyframes on every frame still do not reach the requested `+1 dB` target |
| 195 | higher-fidelity keyframe upper bound | shows q16/float dense-anchor keyframes still do not reach the requested `+1 dB` target |
| 196 | target feasibility branch | concludes selector/keyframe representation branch cannot meet the requested target |
| 197 | learned GS compression protocol | defines GS-native predictive codec route and decoder contract for Stage198-213 |
| 198 | prior predictor training audit | rejects continuing old adapter unchanged and sets gates for the new route |
| 199 | learned GS training manifest | builds multi-gap lightweight train/eval references and passes contract audit |
| 200 | GS predictor architecture package | selects `TemporalBasisGSRefiner` and Stage201 predictor-only protocol |
| 201 | predictor-only smoke | validates executable no-payload predictor plumbing but shows no short-run learned gain |
| 202 | predictor-only broader validation | shows predictor-only training headroom is not observed and residual codec is mandatory |
| 203 | GS residual codec design | selects counted top-k GS attribute residual entropy codec for real-frame smoke |
| 204 | residual codec smoke | demonstrates positive rendered headroom from counted GS-native residual payload |
| 205 | fixed-gap predictive validation | shows sampled positive headroom for q12 + counted GS residual over gaps 4/8/12 |
| 206 | edge RD table | creates sampled edge-level keyframe/residual/metadata cost rows for Stage207 DP |
| 207 | DP oracle schedule | finds sampled Stage206 graph insufficient for nontrivial schedule oracle |
| 206b | connected edge RD expansion | creates one connected bike-packing window with 11 edges and 61 targets |
| 207b | connected-window DP oracle | validates local schedule DP connectivity and selector-label feasibility |

## Next Validation Plan

| next stage | goal | output |
|---:|---|---|
| 206c | multi-sequence connected edge RD | expand connected windows beyond one sequence before selector training |
| 207c | multi-sequence DP oracle | rerun oracle at larger connected scope |
| 208+ | new GS-native predictive codec execution | proceed after multi-sequence Stage207 gate passes |

## Canonical Paths

| item | path |
|---|---|
| Stage161 method package | `experiments/stage161_stage158_method_narrative_package/` |
| Stage162 selector audit | `experiments/stage162_keyframe_selector_protocol_source_audit/` |
| Stage172 rate audit | `experiments/stage172_keyframe_rate_accounting_audit/` |
| Stage176 adaptive candidate | `experiments/stage176_adaptive_schedule_candidate_package/` |
| Stage177 fixed-gap comparison | `experiments/stage177_selector_fixed_gap_psnr_comparison/` |
| Stage180 broader validation | `experiments/stage180_broader_sampled_adaptive_validation_execution/` |
| Stage181 rate preflight | `experiments/stage181_full_sequence_rd_accounting_preflight/` |
| Stage182 freeze decision | `experiments/stage182_selector_refinement_or_freeze_decision/` |
| Stage183 payload protocol | `experiments/stage183_full_sequence_payload_measurement_protocol/` |
| Stage184 payload measurement | `experiments/stage184_full_sequence_payload_measurement_execution/` |
| Stage185 measured RD | `experiments/stage185_measured_full_sequence_rd_aggregation/` |
| Stage186 full-sequence quality | `experiments/stage186_full_sequence_quality_validation/` |
| Stage187 feature ablation | `experiments/stage187_selector_feature_ablation_validation/` |
| Stage188 lower-budget sensitivity | `experiments/stage188_lower_budget_selector_sensitivity/` |
| Stage189 failure-case analysis | `experiments/stage189_failure_case_analysis/` |
| Stage190 paper-facing package | `experiments/stage190_paper_facing_package/` |
| Stage191 fixed-gap expansion protocol | `experiments/stage191_fixed_gap_expansion_protocol/` |
| Stage192 expanded fixed-gap measurement | `experiments/stage192_expanded_fixed_gap_measurement/` |
| Stage193 oracle upper-bound analysis | `experiments/stage193_oracle_upper_bound_analysis/` |
| Stage194 all-keyframe q12 upper bound | `experiments/stage194_all_keyframe_q12_upper_bound/` |
| Stage195 higher-fidelity keyframe upper bound | `experiments/stage195_higher_fidelity_keyframe_upper_bound/` |
| Stage196 target feasibility branch | `experiments/stage196_target_feasibility_branch/` |
| Stage197 learned GS compression protocol | `experiments/stage197_learned_gs_compression_protocol/` |
| Stage198 prior predictor training audit | `experiments/stage198_prior_predictor_training_audit/` |
| Stage199 learned GS training manifest | `experiments/stage199_learned_gs_training_manifest/` |
| Stage200 GS predictor architecture package | `experiments/stage200_gs_predictor_architecture_package/` |
| Stage201 predictor-only smoke | `experiments/stage201_predictor_only_smoke/` |
| Stage202 predictor-only broader validation | `experiments/stage202_predictor_only_broader_validation/` |
| Stage203 GS latent/residual codec design | `experiments/stage203_gs_latent_residual_codec_design/` |
| Stage204 residual codec smoke | `experiments/stage204_residual_codec_smoke/` |
| Stage205 fixed-gap predictive codec validation | `experiments/stage205_fixed_gap_predictive_codec_validation/` |
| Stage206 edge RD table | `experiments/stage206_edge_rd_table/` |
| Stage207 DP oracle schedule | `experiments/stage207_dp_oracle_schedule/` |
| Stage206b connected edge RD expansion | `experiments/stage206b_connected_edge_rd_expansion/` |
| Stage207b DP oracle connected window | `experiments/stage207b_dp_oracle_connected_window/` |
| Stage160 subjective video | `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage160_stage158_extended_subjective_evidence/stage160_gap4_stage158_extended_subjective_evidence.mp4` |
| Stage160 contact sheet | `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage160_stage158_extended_subjective_evidence/stage160_gap4_stage158_extended_subjective_evidence_contact_sheet.jpg` |
