# Stage91 Fixed Residual Side-Info Bitstream Smoke

## Configuration

- task count: `12`
- codecs: `['q12']`
- gaps: `[4, 8, 16]`
- keep fraction: `0.1`
- side bits: `6`
- payload: header + float16 min/max metadata + bit-packed indices + bit-packed q residual values

## Summary

| base | codec | gap | tasks | payload MiB/intermediate | theoretical MiB | overhead bytes | base PSNR | bitstream PSNR | delta | positives |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | q12 | 4 | 3 | 0.04137134552001953 | 0.041353702545166016 | 18.5 | 19.98979652819421 | 23.327021755289916 | 3.3372252270957077 | 3 |
| linear | q12 | 8 | 5 | 0.04137134552001953 | 0.041353702545166016 | 18.5 | 18.12111775349949 | 20.94541228168298 | 2.8242945281834873 | 5 |
| linear | q12 | 16 | 4 | 0.04137134552001953 | 0.041353702545166016 | 18.5 | 19.1130149705525 | 23.589545045225233 | 4.476530074672733 | 4 |
| stage65_adapter | q12 | 4 | 3 | 0.04137134552001953 | 0.041353702545166016 | 18.5 | 20.48273820637705 | 23.352831018389125 | 2.870092812012075 | 3 |
| stage65_adapter | q12 | 8 | 5 | 0.04137134552001953 | 0.041353702545166016 | 18.5 | 18.532358493984642 | 20.943699759081163 | 2.41134126509652 | 5 |
| stage65_adapter | q12 | 16 | 4 | 0.04137134552001953 | 0.041353702545166016 | 18.5 | 19.075010088585586 | 22.48232041707169 | 3.407310328486104 | 4 |

## Notes

- The theoretical MiB column matches Stage87-90 accounting without packet header or byte-alignment padding.
- The payload MiB column is measured from actual encoded bytes.
- This is fixed-length bit packing, not entropy coding.
- Residuals are still teacher-derived from the dense target anchor.
