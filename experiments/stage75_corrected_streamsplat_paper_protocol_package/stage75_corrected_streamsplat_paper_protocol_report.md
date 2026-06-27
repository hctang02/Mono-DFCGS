# Stage75 Corrected StreamSplat Paper-Protocol DAVIS Package

## Scope

- Full DAVIS val split: 30 sequences.
- Sliding fixed intervals.
- Paper-style `256x256` float metric space.
- Main quality metric: middle/non-input PSNR.

## Corrected Baseline

| paper setting | local gap | pair count | all PSNR | middle PSNR | given PSNR | paper PSNR | local - paper |
|---|---:|---:|---:|---:|---:|---:|---:|
| Middle-4 frames | 5 | 1849 | 26.994540075591946 | 23.004337221027775 | 34.97494578472027 | 23.66 | -0.6556627789722249 |
| 8-frame interval | 8 | 1759 | 24.534872014837706 | 21.56004909948801 | 34.94675221856166 | 22.1 | -0.5399509005119931 |

## Stage72 Versus Corrected Stage75

| comparison | Stage72 all | Stage72 middle | Stage75 all | Stage75 middle |
|---|---:|---:|---:|---:|
| Stage72 gap4 scoped vs corrected Middle-4 | 23.244598036349643 | 20.881582891367685 | 26.994540075591946 | 23.004337221027775 |
| Stage72 gap8 scoped vs corrected 8-frame | 20.682715912446714 | 19.17823812651332 | 24.534872014837706 | 21.56004909948801 |

## Interpretation

Stage75 should be used when comparing local StreamSplat against paper-style DAVIS interpolation numbers. Stage72 remains useful only as a scoped Mono-DFCGS diagnostic baseline.
