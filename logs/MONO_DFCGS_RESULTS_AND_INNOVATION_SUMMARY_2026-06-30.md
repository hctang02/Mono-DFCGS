# Mono-DFCGS Results And Innovation Summary

Date: 2026-07-01

Purpose: consolidate the current Mono-DFCGS evidence chain, module-level innovation points, decoder contracts, non-claims, and next validation plan into one handoff log.

## Current Main Claim

The current strongest line is a StreamSplat-guided, Gaussian-domain, rate-counted middle-frame recovery method plus an encoder-side RGB/motion adaptive keyframe schedule.

The current measured full-sequence result is a middle RD point: adaptive improves full-sequence quality over uniform gap8 but uses higher measured rate, and it uses lower measured rate than uniform gap4 but has lower quality than gap4. Stage188 found lower-budget positive-quality sensitivity points, but the lowest positive point still remains above uniform gap8 rate under the additive sensitivity scope. Stage189 identifies the main failure modes for paper limitations and the next selector refinement.

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

## Non-Claims And Risks

| item | status |
|---|---|
| final optimized adaptive RD | not claimed yet; Stage165 measured point is between gap8 and gap4 |
| all-frame quality report | completed for Stage165/gap8/gap4 in Stage186 |
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

## Next Validation Plan

| next stage | goal | output |
|---:|---|---|
| 190 | paper-facing package | measured RD-quality tables, Stage187 ablation, Stage188 sensitivity, Stage189 failure cases, decoder contract, limitations, and method framing |
| optional refinement | schedule-packed Stage188 candidate measurement | same-scope RD for selected lower-budget candidates if needed for final claims |

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
| Stage160 subjective video | `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage160_stage158_extended_subjective_evidence/stage160_gap4_stage158_extended_subjective_evidence.mp4` |
| Stage160 contact sheet | `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage160_stage158_extended_subjective_evidence/stage160_gap4_stage158_extended_subjective_evidence_contact_sheet.jpg` |
