# Stage150 Full Linear-Base Side-Info Rendered Validation

## Configuration

- task count: `3170`
- codecs: `['q12']`
- gaps: `[4, 8]`
- base methods: `['linear']`
- keep fraction: `0.1`
- side bits: `6`
- zlib level: `9`

## Summary

| base | codec | gap | tasks | fixed bytes | entropy bytes | ratio | entropy MiB/intermediate | max decode diff | entropy PSNR | delta | positives |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | q12 | 4 | 1463 | 43381.0 | 30062.867395762132 | 0.6929961825629228 | 0.028670184512865193 | 0.0 | 23.104893423851635 | 3.120929241560934 | 1463 |
| linear | q12 | 8 | 1707 | 43381.0 | 30203.919156414762 | 0.6962476465829451 | 0.028804701954283488 | 0.0 | 22.020188948523128 | 3.253760121312108 | 1707 |

## Notes

- Entropy payload uses sorted-index deltas and zlib-compressed metadata/index/residual components.
- Decode is compared against Stage91 fixed decode before rendering.
- Residual values are encoder-side side-info payload values; decoder safety requires transmitting and counting the encoded payload.
