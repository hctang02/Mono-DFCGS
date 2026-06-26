# Future Work Plan: Stage56-68

Superseded note: the latest plan is now `logs/FUTURE_WORK_PLAN_STAGE56_70.md`, updated to include DAVIS and YouTube-VOS download/preparation stages before large-scale training.

Date: 2026-06-26

This plan records the next execution path requested by the user. Future planning discussions should also be written into logs before implementation.

## Core Requirements

- Future reports default to all-frame PSNR only unless the user explicitly asks for middle-only metrics.
- Every important stage should produce tables including size/rate and all-frame PSNR, plus RD curves when applicable.
- Keyframe selection must become fully feed-forward for final claims. Oracle or rendered-error-assisted selectors can only be used as upper bounds or training labels.
- Compression must become an explicit contribution, not only q8/q16 plus generic zlib.
- Gaussian adapter training and feed-forward keyframe selector training must be treated as major contributions, not smoke tests.
- Training must scale beyond short runs: use short runs for debugging, medium runs for method selection, and long large-scale runs for final results.
- Baseline-inspired adapter designs should be explored, especially StreamSplat teacher distillation or baseline-style architecture, while preserving the final codec input constraints.
- Optional non-keyframe side information can be explored if needed, but every transmitted side-information bit must be counted in rate.
- Large files, checkpoints, datasets, rendered caches, and bitstreams should stay outside git under `/mnt/hdd2tC/tmp/opencode/...`.

## Three Main Contribution Lines

| Contribution | Goal | Evidence Needed |
|---|---|---|
| Gaussian anchor compression | Design compact keyframe Gaussian anchor codec beyond raw/zlib prototype | Size table, all-PSNR table, RD curve, compression ablation |
| Gaussian adapter | Improve non-keyframe reconstruction beyond linear interpolation | Adapter-vs-linear all-PSNR gains, training curves, RD curves, long-training results |
| Feed-forward keyframe selector | Select keyframes without rendered oracle or test-time reconstruction feedback | Uniform-vs-feed-forward-adaptive table, RD curve, failure table, deployability proof |

Optional side-information is a fourth enhancement line, not the default keyframe-Gaussian-only main claim.

## Stage56: Protocol Lock

Goal: freeze the experimental protocol before adding more components.

Deliverables:

- Protocol report and JSON summary.
- Rate accounting rules for anchor-only and side-information variants.
- Selector deployment rules.
- Standard output table schemas.
- Stage record under `logs/stage_records/`.

Key decisions:

- Main rate: transmitted Gaussian anchor bitstream MiB/frame.
- Side-info rate: counted separately and in total rate if used.
- Main metric: all-frame PSNR.
- `rendered_prior_0p1`: oracle/calibrated upper bound only.
- Final selector: frozen predictor plus deterministic selection/DP, no rendered oracle.

## Stage57: Compact Anchor Codec

Goal: make compression a concrete contribution.

Tasks:

- Implement true bit-packing for q1-q16 anchor payloads.
- Keep a simple, deterministic metadata/header format.
- Validate encode/decode roundtrip against direct quantized-dequantized anchors.
- Report raw, zlib, prototype uint storage, and compact bitpacked rates.

Deliverables:

- Codec implementation.
- Roundtrip tests or script checks.
- Size table and summary JSON.
- Stage record.

## Stage58: Compression RD Ablation

Goal: connect compression changes to all-frame PSNR/rate.

Tasks:

- Compare raw, zlib, bitpacked, bitpacked+zlib.
- Add per-field quantization candidates for `rgb`, `opacity`, `scale`, `xyz`, `rot`.
- Add simple temporal delta coding between keyframe anchors if feasible.
- Add an entropy baseline if compact symbol streams are stable.

Deliverables:

- All-frame PSNR vs rate RD curves.
- Compression ablation table.
- Rate saving table.
- Stage record.

## Stage59: Adapter Training Infrastructure v2

Goal: prepare adapter training for medium/long runs.

Tasks:

- Add resume support.
- Add train/val/test splits.
- Add best-checkpoint selection by validation all-frame PSNR.
- Add structured training curves and checkpoint metadata.
- Keep checkpoints outside git.

Deliverables:

- Training script updates or new script.
- Short verification run.
- Stage record.

## Stage60: Medium Adapter Training

Goal: determine whether current adapter gains are limited by short training.

Training levels:

| Level | Steps | Use |
|---|---:|---|
| short | hundreds | debug only |
| medium | 5k-20k | method comparison |
| long | 50k-200k | final result |

Adapter variants:

- Adapter-A: current linear-residual adapter with longer training.
- Adapter-B: larger hidden dimension / deeper MLP / better temporal encoding.
- Adapter-C: stronger losses, including RGB render loss, Gaussian attribute loss, and temporal consistency.
- Adapter-D: validation-selected checkpoint.

Target: improve all-frame PSNR gain beyond the current small `+0.0798 dB` over linear interpolation.

Deliverables:

- all-frame PSNR table.
- Adapter-vs-linear RD curve.
- Training curves.
- Best checkpoint summary.
- Stage record.

## Stage61: Baseline / Teacher Adapter Study

Goal: answer whether baseline-inspired training/architecture improves adapter quality.

Candidate routes:

| Route | Description | Codec Constraint |
|---|---|---|
| StreamSplat teacher distillation | Use StreamSplat outputs as training teacher, not test-time input | Allowed if test-time uses only transmitted data |
| Baseline-style architecture | Borrow temporal/dynamic decoder structure but condition on keyframe anchors | Allowed |
| Direct StreamSplat decoder fine-tune | Fine-tune original modules | Only as controlled ablation; easy to violate codec constraints |
| Hybrid adapter | keyframe GS -> latent -> dynamic GS | Recommended |

Deliverables:

- Adapter comparison table.
- Complexity table: parameters and runtime.
- all-frame PSNR RD curves.
- Stage record.

## Stage62: Large-scale Data Pipeline

Goal: move beyond four local samples.

Tasks:

- Mount/download DAVIS / YouTube-VOS when available.
- Run depth preprocessing to create provider-required `depthImages/*_pred.png`.
- Export dense or multi-gap Gaussian anchors.
- Build train/val/test manifests.
- Support longer videos beyond 77/81-frame local samples.
- Keep all heavy outputs in `/mnt/hdd2tC/tmp/opencode/...`.

Data levels:

| Level | Scale | Use |
|---|---|---|
| Local-4 | current four samples | fast debug |
| Small-DAVIS | 10-20 videos | medium training trend |
| Full-DAVIS/VOS subset | 50-200 videos | long training |
| Long-video set | longer sequences | temporal stability |

Deliverables:

- Dataset readiness table.
- Anchor manifest summary.
- Stage record.

## Stage63: Long Adapter Training

Goal: produce final-quality adapter results.

Tasks:

- Train selected adapter variants on larger datasets.
- Track train/val/test all-frame PSNR.
- Evaluate across q8/q10/q12/q16 and multiple gaps.
- Produce final adapter RD curves.

Deliverables:

- Long-training curve.
- Held-out all-frame PSNR table.
- RD curve.
- Checkpoint metadata.
- Stage record.

## Stage64: Feed-forward Selector Dataset

Goal: prepare labels for deployable keyframe selector training.

Rules:

- Oracle/rendered RD labels can be generated offline for supervision.
- Test-time selector input cannot include rendered error, PSNR, reconstructed frames, or optimization over results.

Potential labels:

- Segment difficulty.
- Adaptive-vs-uniform gain.
- Pairwise layout ranking.
- DP-aware layout score.

Deliverables:

- Selector dataset CSV.
- Feature schema.
- Label schema.
- Stage record.

## Stage65: Medium Feed-forward Selector Training

Goal: train and evaluate deployable selector variants.

Variants:

| Variant | Goal |
|---|---|
| Selector-A | segment cost predictor baseline |
| Selector-B | adaptive-or-uniform classifier |
| Selector-C | pairwise layout ranker |
| Selector-D | DP-aware predictor |

Final inference form:

```text
input video / encoder-side features
-> frozen predictor
-> deterministic DP / deterministic selector
-> keyframe indices
```

Deliverables:

- Uniform-vs-feed-forward-adaptive all-frame PSNR table.
- Actual bitstream rate table.
- Failure table.
- RD curve.
- Stage record.

## Stage66: Long Feed-forward Selector Training

Goal: make selector robust on larger/longer datasets.

Tasks:

- Train on larger selector label datasets.
- Calibrate fallback to uniform.
- Evaluate on held-out videos.
- Compare against oracle/calibrated upper bound without claiming oracle as final.

Deliverables:

- Long-training selector curves.
- Held-out all-frame PSNR/RD table.
- Stability/failure analysis.
- Stage record.

## Stage67: Optional Side-information Codec

Goal: explore whether small non-keyframe payloads can improve quality enough to justify rate.

Candidate side information:

| Side Info | Purpose | Rate Rule |
|---|---|---|
| low-resolution depth | geometry/occlusion hint | count side-info rate |
| sparse depth/keypoints | compact geometry hint | count side-info rate |
| motion hint | dynamic interpolation help | count side-info rate |
| Gaussian residual correction | correct hard non-keyframes | count side-info rate; may become P-frame residual codec |
| importance map | guide selective refinement | count side-info rate |

Method naming:

- `Mono-DFCGS-KG`: keyframe Gaussian only, main method.
- `Mono-DFCGS-SI`: keyframe Gaussian plus side information, enhancement method.

Deliverables:

- anchor rate, side-info rate, total rate.
- all-frame PSNR table.
- RD curve.
- Stage record.

## Stage68: Final Ablation And RD Package

Goal: package final paper-ready evidence.

Required tables:

| Method | Compression | Adapter | Selector | Side info | Rate MiB/frame | All PSNR |
|---|---|---|---|---|---:|---:|
| uniform baseline | zlib/compact anchor | linear | uniform | none | x | x |
| trained adapter | compact anchor | trained | uniform | none | x | x |
| teacher/baseline-style adapter | compact anchor | improved | uniform | none | x | x |
| feed-forward adaptive | compact anchor | improved | feed-forward | none | x | x |
| side-info variant | compact anchor | improved | feed-forward | depth/motion/residual | x | x |
| oracle upper bound | compact anchor | improved | oracle | none | x | x |

Required figures:

- Main all-frame PSNR RD curve.
- Compression ablation RD.
- Adapter ablation RD.
- Selector ablation RD.
- Optional side-info RD.
- Long-training curves.

## Immediate Execution Order

1. Record this plan and stage-recording convention.
2. Run Stage56 Protocol Lock.
3. Implement Stage57 Compact Anchor Codec.
4. Run Stage58 Compression RD Ablation.
5. Build Stage59 adapter long-training infrastructure.
6. Continue through medium/long adapter and selector training, with user input only for major irreversible decisions such as final adapter family selection.
