# Stage148 Rate-Counted Side-Info Rendered Revalidation

## Configuration

- task count: `120`
- codecs: `['q12']`
- gaps: `[4, 8]`
- keep fraction: `0.1`
- side bits: `6`
- zlib level: `9`

## Summary

| base | codec | gap | tasks | fixed bytes | entropy bytes | ratio | entropy MiB/intermediate | max decode diff | entropy PSNR | delta | positives |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | q12 | 4 | 60 | 43381.0 | 29934.55 | 0.6900382656001475 | 0.02854781150817871 | 0.0 | 23.189376901693723 | 3.1136959670226054 | 60 |
| linear | q12 | 8 | 60 | 43381.0 | 30209.35 | 0.6963728360342085 | 0.028809881210327147 | 0.0 | 22.147360893983887 | 3.40840563316142 | 60 |
| stage65_adapter | q12 | 4 | 60 | 43381.0 | 34785.51666666667 | 0.8018606455975347 | 0.03317405382792155 | 0.0 | 22.850143675432175 | 2.5860101860450535 | 60 |
| stage65_adapter | q12 | 8 | 60 | 43381.0 | 35153.03333333333 | 0.810332480425378 | 0.033524545033772786 | 0.0 | 21.965723744155056 | 2.9033707572416314 | 60 |

## Notes

- Entropy payload uses sorted-index deltas and zlib-compressed metadata/index/residual components.
- Decode is compared against Stage91 fixed decode before rendering.
- Residual values are encoder-side side-info payload values; decoder safety requires transmitting and counting the encoded payload.
