# Stage149 Full Gap4/Gap8 Side-Info Rendered Validation

## Configuration

- task count: `3170`
- codecs: `['q12']`
- gaps: `[4, 8]`
- base methods: `['stage65_adapter']`
- keep fraction: `0.1`
- side bits: `6`
- zlib level: `9`

## Summary

| base | codec | gap | tasks | fixed bytes | entropy bytes | ratio | entropy MiB/intermediate | max decode diff | entropy PSNR | delta | positives |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage65_adapter | q12 | 4 | 1463 | 43381.0 | 34943.74094326726 | 0.8055079630083977 | 0.03332494825674749 | 0.0 | 22.768595216050993 | 2.55782458013036 | 1463 |
| stage65_adapter | q12 | 8 | 1707 | 43381.0 | 35172.31634446397 | 0.8107769840359608 | 0.03354293474623105 | 0.0 | 21.857517703953395 | 2.790419745485695 | 1707 |

## Notes

- Entropy payload uses sorted-index deltas and zlib-compressed metadata/index/residual components.
- Decode is compared against Stage91 fixed decode before rendering.
- Residual values are encoder-side side-info payload values; decoder safety requires transmitting and counting the encoded payload.
