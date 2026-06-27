# Stage76 Static Anchor Quantization Sweep

## Summary

| codec | MiB/anchor | PSNR 512 | delta 512 | PSNR 256 | delta 256 |
|---|---:|---:|---:|---:|---:|
| q6 | 0.3428230285644531 | 16.964416756584846 | -12.770973560085498 | 17.739751819785756 | -16.925497206905096 |
| q8 | 0.4570808410644531 | 27.044935397848864 | -2.69045491882148 | 30.39796551711433 | -4.267283509576522 |
| q10 | 0.5713386535644531 | 29.35456348600337 | -0.3808268306669724 | 34.144040488489495 | -0.5212085382013569 |
| q12 | 0.6855964660644531 | 29.708204125401867 | -0.02718619126847699 | 34.6285618759872 | -0.03668715070364925 |
| float16 | 0.9141120910644531 | 29.735390316670344 | 0.0 | 34.66524902669085 | 0.0 |
| q16 | 0.9141120910644531 | 29.73518080346308 | -0.00020951320726325662 | 34.665082254118616 | -0.00016677257223562947 |

## Notes

- Rate is a per-keyframe static anchor payload estimate for 13 attributes per Gaussian.
- Metadata includes fp16 per-channel min/scale for q-bit codecs and is negligible at this Gaussian count.
- This stage measures direct keyframe render quality only; middle-frame predictor quality is evaluated separately.
