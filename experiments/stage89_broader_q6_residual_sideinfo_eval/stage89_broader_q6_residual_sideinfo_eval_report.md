# Stage89 Broader q6 Residual Side-Info Eval

## Configuration

- task count: `60`
- codecs: `['q12']`
- gaps: `[4, 8, 16]`
- keep fractions: `[0.1]`
- side bits: `[6]`
- metadata bits per min/max value: `16`

## Summary

| base | codec | gap | keep | bits | side MiB/intermediate | base PSNR | side PSNR | delta | positives |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | q12 | 4 | 0.1 | 6 | 0.041353702545166016 | 19.984796422852106 | 23.403666365261454 | 3.4188699424093465 | 18 |
| linear | q12 | 8 | 0.1 | 6 | 0.041353702545166016 | 18.457589807006293 | 21.513744759131683 | 3.056154952125395 | 19 |
| linear | q12 | 16 | 0.1 | 6 | 0.041353702545166016 | 17.08421543617396 | 20.344221350298344 | 3.2600059141243887 | 23 |
| stage65_adapter | q12 | 4 | 0.1 | 6 | 0.041353702545166016 | 20.041732308843873 | 22.84099865483423 | 2.7992663459903557 | 18 |
| stage65_adapter | q12 | 8 | 0.1 | 6 | 0.041353702545166016 | 18.706547218120296 | 21.398942062762643 | 2.6923948446423474 | 19 |
| stage65_adapter | q12 | 16 | 0.1 | 6 | 0.041353702545166016 | 17.32823013943833 | 20.29200458015274 | 2.9637744407144093 | 23 |

## Notes

- Residual values are quantized per frame and per attribute over the kept Gaussian set.
- Rate includes Gaussian indices, quantized residual attrs, and per-attribute min/max metadata.
- This is still a smoke test; entropy coding and full-video RD are not implemented here.
