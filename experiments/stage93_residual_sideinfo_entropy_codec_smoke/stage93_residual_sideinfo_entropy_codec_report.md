# Stage93 Entropy-Coded Residual Side-Info Codec Smoke

## Configuration

- task count: `12`
- codecs: `['q12']`
- gaps: `[4, 8, 16]`
- keep fraction: `0.1`
- side bits: `6`
- zlib level: `9`

## Summary

| base | codec | gap | tasks | fixed bytes | entropy bytes | ratio | entropy MiB/intermediate | max decode diff | entropy PSNR | delta | positives |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | q12 | 4 | 3 | 43381.0 | 29763.666666666668 | 0.6860991371030328 | 0.028384844462076824 | 0.0 | 23.327021755289916 | 3.3372252270957077 | 3 |
| linear | q12 | 8 | 5 | 43381.0 | 30815.4 | 0.71034323782301 | 0.029387855529785158 | 0.0 | 20.94541228168298 | 2.8242945281834873 | 5 |
| linear | q12 | 16 | 4 | 43381.0 | 29559.5 | 0.6813927756391047 | 0.028190135955810547 | 0.0 | 23.589545045225233 | 4.476530074672733 | 4 |
| stage65_adapter | q12 | 4 | 3 | 43381.0 | 35026.0 | 0.8074041631128835 | 0.03340339660644531 | 0.0 | 23.352831018389125 | 2.870092812012075 | 3 |
| stage65_adapter | q12 | 8 | 5 | 43381.0 | 36109.8 | 0.8323874507272769 | 0.03443698883056641 | 0.0 | 20.943699759081163 | 2.41134126509652 | 5 |
| stage65_adapter | q12 | 16 | 4 | 43381.0 | 34910.25 | 0.8047359443074157 | 0.03329300880432129 | 0.0 | 22.48232041707169 | 3.407310328486104 | 4 |

## Notes

- Entropy payload uses sorted-index deltas and zlib-compressed metadata/index/residual components.
- Decode is compared against Stage91 fixed decode before rendering.
- Residuals are still teacher-derived; this is a codec smoke, not a deployable residual predictor.
