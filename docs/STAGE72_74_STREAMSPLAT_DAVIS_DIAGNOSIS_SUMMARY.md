# Stage72-74 StreamSplat DAVIS Diagnosis Summary

Date: 2026-06-27

## Purpose

This document records the current user-facing outputs and conclusions for the DAVIS StreamSplat baseline and low-PSNR diagnosis.

It summarizes:

- Stage72 original StreamSplat scoped DAVIS baseline.
- Stage73 Stage70 low-PSNR diagnosis.
- Stage74 analysis of why Stage72 differs from the paper/actual StreamSplat numbers.

## Key Output Paths

Stage72 original scoped baseline:

```text
experiments/stage72_original_davis_baseline/
scripts/run_stage72_original_davis_baseline.py
```

Stage73 low-PSNR diagnosis:

```text
experiments/stage73_low_psnr_diagnosis/
scripts/run_stage73_low_psnr_diagnosis.py
```

Stage74 Stage72-vs-actual diagnosis:

```text
experiments/stage74_stage72_vs_actual_gap_diagnosis_stage72_control/
experiments/stage74_stage72_vs_actual_gap_diagnosis_sliding_per_frame/
experiments/stage74_stage72_vs_actual_gap_diagnosis_full_val_sliding_per_frame/
scripts/run_stage74_stage72_vs_actual_gap_diagnosis.py
```

Stage records:

```text
logs/stage_records/72_original_davis_baseline_and_low_psnr_diagnosis.md
logs/stage_records/73_low_psnr_diagnosis.md
logs/stage_records/74_stage72_vs_actual_gap_diagnosis.md
```

## Stage72 Result

Stage72 was a scoped internal baseline, not an exact reproduction of the StreamSplat paper benchmark.

It used:

- 4 DAVIS val sequences: `bmx-trees`, `car-shadow`, `goat`, `soapbox`.
- Disjoint sparse keyframes with tail completion.
- Gaps `4`, `8`, `16`.
- Metric computed directly at `512x288` and quantized to uint8.
- All-frame PSNR as the primary reported metric.

Stage72 mean results:

| gap | all PSNR | middle PSNR | given PSNR |
|---:|---:|---:|---:|
| 4 | 23.2446 | 20.8816 | 29.7452 |
| 8 | 20.6827 | 19.1782 | 29.7189 |
| 16 | 18.3535 | 17.3366 | 29.6893 |

## Stage73 Result

Stage73 diagnosed why Stage70 Gaussian-anchor-only RD was lower than Stage72.

Main conclusion:

- Stage70 q8 adapter uniform values are exactly reproducible.
- No RGB/resize/frame-index/render-bridge/checkpoint-loading bug was found.
- Stage70 is low mainly because it is a q8 static-anchor-only method, while Stage72 uses original full dynamic StreamSplat `pred_gs`.
- q8 quantization itself is also aggressive on DAVIS keyframe anchors.

Stage73 decomposition:

| gap | original all | float static adapter all | q8 static adapter all | q8 loss |
|---:|---:|---:|---:|---:|
| 4 | 23.2446 | 21.3473 | 20.6085 | 0.7387 |
| 8 | 20.6827 | 18.9365 | 18.5425 | 0.3940 |
| 16 | 18.3535 | 17.2445 | 17.0130 | 0.2315 |

## Stage74 Result

Stage74 explains why Stage72 looked much lower than the StreamSplat paper/actual numbers.

The main reasons are protocol differences:

- Paper metrics are evaluated at `256x256`; Stage72 used `512x288` uint8 metrics.
- Paper dynamic interpolation reports non-input frames; Stage72 emphasized all-frame sparse reconstruction.
- Paper uses fixed sparse intervals, including 5-frame and 8-frame settings; Stage72 used gaps `4/8/16` with a different meaning.
- Paper results are on the full DAVIS validation split; Stage72 used only 4 scoped sequences, several of which are harder than average.
- Stage72 was written as a custom Mono-DFCGS scoped baseline runner, not as an official StreamSplat benchmark script.

Checkpoint loading was audited in Stage74:

```text
missing_count = 0
unexpected_count = 0
checkpoint_tensor_count = 320
checkpoint_value_count = 183975569
```

So the discrepancy is not caused by a checkpoint/key mismatch.

### Stage72-Style Control

Same 4 scoped sequences, Stage72-style disjoint gap4, but comparing metric spaces:

| setting | all PSNR | middle PSNR | given PSNR |
|---|---:|---:|---:|
| official_256_float | 27.1753 | 22.0442 | 34.6418 |
| stage72_512_float | 24.4362 | 20.7974 | 29.7310 |
| stage72_512_uint8 | 24.4333 | 20.7966 | 29.7251 |

This shows the `256x256` metric alone raises the apparent PSNR by about `2-5 dB`, especially for given keyframes.

### Scoped 4-Sequence Official-Style Sliding

Same 4 scoped sequences, but using sliding fixed windows, per-frame depth normalization, and `256x256` metric:

| gap | all PSNR | middle PSNR | given PSNR |
|---:|---:|---:|---:|
| 5 | 25.7507 | 21.3008 | 34.6505 |
| 8 | 23.0741 | 19.7675 | 34.6472 |

### Full DAVIS Val Official-Style Sliding

Full 30-sequence DAVIS val split, sliding fixed windows, per-frame depth normalization, and `256x256` metric:

| gap | pair count | all PSNR | middle PSNR | given PSNR |
|---:|---:|---:|---:|---:|
| 5 | 1849 | 26.9945 | 23.0043 | 34.9749 |
| 8 | 1759 | 24.5349 | 21.5600 | 34.9468 |

The full-val result is much closer to the paper:

| paper setting | paper PSNR | local comparable PSNR | remaining gap |
|---|---:|---:|---:|
| Middle-4 frames | 23.66 | 23.0043 | 0.6557 |
| 8-frame interval | 22.10 | 21.5600 | 0.5400 |

The remaining gap can plausibly come from exact official aggregation details, depth preprocessing differences, code/checkpoint version differences, or whether the paper uses a slightly different evaluation wrapper. It is no longer a multi-dB mystery after protocol alignment.

## Current Interpretation

Stage72 should be treated as a scoped Mono-DFCGS diagnostic baseline, not the final StreamSplat paper reproduction.

For comparisons against published StreamSplat numbers, use the Stage74-style protocol:

- full DAVIS val split,
- `256x256` metric,
- non-input/middle frame PSNR,
- gap5 for Middle-4 and gap8 for 8-frame interval,
- clearly state the local checkpoint and preprocessing.

For Mono-DFCGS codec RD, continue using Stage70/73 as static-anchor-only/q8 diagnostics, but do not compare Stage70 directly to paper StreamSplat numbers without stating the representation and metric differences.

## Recommended Next Stages

1. Package a corrected StreamSplat paper-protocol DAVIS baseline table from Stage74 outputs.
2. Run q-bit and per-field quantization sweeps for static anchors, because q8 causes large keyframe degradation.
3. Add optional dynamic-field side information or a richer anchor representation if rate permits.
4. Build fair FCGS/D-FCGS DAVIS baselines before final RD claims.
5. Train/select a stronger fully feed-forward selector using rendered labels, not only anchor-space proxy labels.
