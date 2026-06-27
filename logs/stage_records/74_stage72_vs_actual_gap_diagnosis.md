# Stage74 Stage72-vs-Actual Gap Diagnosis

Date: 2026-06-27

## Goal

Analyze why Stage72 original StreamSplat DAVIS scoped baseline differs from the expected actual/official result.

## Plan

- Inspect the Stage72 runner and identify exactly what it evaluates.
- Inspect the local official StreamSplat repo for DAVIS evaluation protocol, datasets, masks, resolution, metric scripts, and checkpoint/config assumptions.
- Compare Stage72 with the expected actual protocol across data split, sequence set, frame sampling, resolution, mask usage, depth source, metric scope, and rate/quality aggregation.
- Run lightweight checks only if static inspection is insufficient. Check `nvidia-smi` before any code run.

## Initial Hypothesis

Stage72 may be an internal scoped baseline using the StreamSplat model on custom DAVIS frames/depths, not an exact official benchmark reproduction. Protocol differences can easily dominate the PSNR gap.

## Result

Stage74 confirmed that Stage72 is not an exact paper-protocol StreamSplat benchmark.

Key protocol differences:

- Stage72 used 4 scoped DAVIS val sequences; paper-style evaluation should use the full 30-sequence DAVIS val split.
- Stage72 reported `512x288` uint8 metrics; the paper states metrics are evaluated at `256x256`.
- Stage72 emphasized all-frame sparse reconstruction with disjoint keyframes and tail completion; paper dynamic interpolation reports non-input frames under fixed interval settings.
- Stage72 used gaps `4/8/16`; the paper reports Middle-4 frames and 8-frame interval settings, which correspond more closely to local gap `5` and gap `8` sliding windows.

Checkpoint loading was audited and is not the cause:

```text
missing_count = 0
unexpected_count = 0
checkpoint_tensor_count = 320
checkpoint_value_count = 183975569
```

Stage72-style control on the 4 scoped sequences, gap4:

| metric space | all PSNR | middle PSNR | given PSNR |
|---|---:|---:|---:|
| official_256_float | 27.175326918003805 | 22.04421501448969 | 34.64175257247626 |
| stage72_512_float | 24.43616869678078 | 20.79743947198787 | 29.730986225600738 |
| stage72_512_uint8 | 24.433301145796477 | 20.796620327049567 | 29.72513797820384 |

Full DAVIS val official-style sliding windows, per-frame depth normalization, `256x256` metric:

| gap | pair count | all PSNR | middle PSNR | given PSNR |
|---:|---:|---:|---:|---:|
| 5 | 1849 | 26.994540075591946 | 23.004337221027775 | 34.97494578472027 |
| 8 | 1759 | 24.534872014837706 | 21.56004909948801 | 34.94675221856166 |

Compared with the paper values, local full-val middle-frame PSNR is close:

| setting | paper PSNR | local PSNR | gap |
|---|---:|---:|---:|
| Middle-4 frames | 23.66 | 23.004337221027775 | 0.6556627789722242 |
| 8-frame interval | 22.10 | 21.56004909948801 | 0.5399509005119906 |

Conclusion: Stage72 looked much lower mostly because it used a different scoped diagnostic protocol. After matching the paper-style metric resolution, interval, non-input frame scope, and full DAVIS val split, the remaining gap is about `0.5-0.7 dB` for middle-frame interpolation.

## Standalone User Summary

The standalone summary requested by the user is saved at:

```text
docs/STAGE72_74_STREAMSPLAT_DAVIS_DIAGNOSIS_SUMMARY.md
```
