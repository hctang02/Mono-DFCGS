# Stage76 Static Anchor Quantization Sweep

Date: 2026-06-27

## Goal

Measure direct render quality of static keyframe anchors under different quantization bit depths.

Stage73 showed that q8 static anchors lose about `2.75 dB` on given-keyframe PSNR. Stage76 checks whether q10/q12/q16 recover keyframe quality at acceptable rate.

## Plan

- Use Stage61 DAVIS gap1 anchors.
- Scope: `DAVIS/val/bmx-trees`, `DAVIS/val/car-shadow`, `DAVIS/val/goat`, `DAVIS/val/soapbox`.
- Codecs: `float16`, `q6`, `q8`, `q10`, `q12`, `q16`.
- Metrics: direct keyframe render PSNR at `512x288` and `256x256`.
- Rate: estimated static anchor MiB per keyframe.

## Expected Outputs

```text
scripts/run_stage76_static_anchor_quantization_sweep.py
experiments/stage76_static_anchor_quantization_sweep/
```

## Result

Stage76 completed on the scoped DAVIS val sequences.

Outputs:

```text
experiments/stage76_static_anchor_quantization_sweep/stage76_static_anchor_quantization_summary.json
experiments/stage76_static_anchor_quantization_sweep/stage76_static_anchor_quantization_summary.csv
experiments/stage76_static_anchor_quantization_sweep/stage76_static_anchor_quantization_rows.csv
experiments/stage76_static_anchor_quantization_sweep/stage76_static_anchor_quantization_report.md
```

Summary:

| codec | MiB/anchor | PSNR 512 | delta 512 | PSNR 256 | delta 256 |
|---|---:|---:|---:|---:|---:|
| q6 | 0.3428230285644531 | 16.964416756584846 | -12.770973560085498 | 17.739751819785756 | -16.925497206905096 |
| q8 | 0.4570808410644531 | 27.044935397848864 | -2.69045491882148 | 30.39796551711433 | -4.267283509576522 |
| q10 | 0.5713386535644531 | 29.35456348600337 | -0.3808268306669724 | 34.144040488489495 | -0.5212085382013569 |
| q12 | 0.6855964660644531 | 29.708204125401867 | -0.02718619126847699 | 34.6285618759872 | -0.03668715070364925 |
| float16 | 0.9141120910644531 | 29.735390316670344 | 0.0 | 34.66524902669085 | 0.0 |
| q16 | 0.9141120910644531 | 29.73518080346308 | -0.00020951320726325662 | 34.665082254118616 | -0.00016677257223562947 |

Conclusion:

- q8 is too aggressive for direct keyframe anchors on DAVIS.
- q10 recovers most direct keyframe quality with `0.5713386535644531 MiB/anchor`, about `25%` larger than q8.
- q12 is nearly lossless relative to float16 direct rendering while still smaller than float16.
- Next RD packages should include q10/q12 operating points, not only q8.
