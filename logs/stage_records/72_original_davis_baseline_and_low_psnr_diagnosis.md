# Stage72 Original DAVIS Baseline And Low-PSNR Diagnosis

Date: 2026-06-27

## Goal

Run an original StreamSplat-style DAVIS scoped baseline first, then diagnose why the Stage70 Gaussian-anchor-only RD numbers are low.

The stage has two phases and should stop for user review after both are complete.

## Phase A Plan

Original method baseline on the same scoped DAVIS subset as Stage70:

```text
DAVIS/val/bmx-trees
DAVIS/val/car-shadow
DAVIS/val/goat
DAVIS/val/soapbox
```

Gaps:

```text
4 8 16
```

Primary reported metric: all-frame PSNR.

Diagnostic metrics: middle-frame and given-keyframe PSNR, only to locate protocol or alignment issues.

## Phase B Plan

If original StreamSplat results are reasonable but Stage70 remains low, diagnose the Gaussian-anchor-only path:

- RGB target resize, range, color, and frame-index alignment.
- Direct keyframe anchor rendering quality.
- Float-anchor versus q8-anchor loss.
- Static-anchor-to-renderer bridge correctness.
- Stage65 best checkpoint loading and hidden-dim match.
- Stage70 scoped rerun after fixes.

## Expected Outputs

```text
scripts/run_stage72_original_davis_baseline.py
experiments/stage72_original_davis_baseline/
```

Optional Phase B outputs should use:

```text
experiments/stage73_low_psnr_diagnosis/
```

## Notes

- Heavy caches or rendered frames must stay outside git, preferably under `/data/hctang/tmp/opencode/mono_dfcgs_runs/`.
- Do not continue into large-scale adapter/selector/compression training before reporting Phase A/B results.

## Phase A Result

Script:

```text
scripts/run_stage72_original_davis_baseline.py
```

Outputs:

```text
experiments/stage72_original_davis_baseline/
```

Original StreamSplat full dynamic mean all-frame PSNR on the Stage70 scoped DAVIS subset:

| gap | all PSNR | middle PSNR | given PSNR |
|---:|---:|---:|---:|
| 4 | 23.244598036349643 | 20.881582891367685 | 29.745217856431786 |
| 8 | 20.682715912446714 | 19.17823812651332 | 29.71886975750609 |
| 16 | 18.353465837484507 | 17.3365956594116 | 29.689301009427645 |

The original baseline is higher than Stage70 q8 adapter uniform by `2.636066178627917 dB`, `2.140227263020065 dB`, and `1.340433291926977 dB` for gaps `4`, `8`, and `16` respectively.

## Phase B Result

Phase B was completed as Stage73:

```text
scripts/run_stage73_low_psnr_diagnosis.py
experiments/stage73_low_psnr_diagnosis/
logs/stage_records/73_low_psnr_diagnosis.md
```

Diagnosis result:

- Stage70 q8 adapter uniform PSNR is exactly reproduced by Stage73, so Stage70 summary/join logic is not the source of the low numbers.
- Float static keyframe anchors match original given-keyframe PSNR almost exactly, which validates RGB/frame alignment and the static-anchor renderer bridge.
- Most of the original-vs-Stage70 gap is caused by using static-anchor-only Gaussian reconstruction instead of original full dynamic `pred_gs`.
- q8 quantization adds an extra lossy component, especially on keyframes.

Phase A/B are complete. Stop for user review before starting large-scale adapter, selector, compression, or FCGS/D-FCGS baseline work.
