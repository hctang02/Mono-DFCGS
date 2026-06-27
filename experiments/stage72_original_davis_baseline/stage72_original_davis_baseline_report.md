# Stage72 Original DAVIS Baseline

## Scope

- DAVIS root: `/data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS`
- Sequences: `bmx-trees, car-shadow, goat, soapbox`
- Gaps: `4, 8, 16`
- Primary metric: all-frame PSNR.

## Gap Summary

| gap | original all PSNR | original middle PSNR | original given PSNR | Stage70 linear uniform | Stage70 adapter uniform | Stage70 adapter predicted |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 23.244598036349643 | 20.881582891367685 | 29.745217856431786 | 20.137413494810616 | 20.608531857721726 | 20.634207426626656 |
| 8 | 20.682715912446714 | 19.17823812651332 | 29.71886975750609 | 18.063459460774446 | 18.54248864942665 | 18.525169717464813 |
| 16 | 18.353465837484507 | 17.3365956594116 | 29.689301009427645 | 16.59656669447801 | 17.01303254555753 | 17.096890478737578 |

## Per-Sequence Comparison

| sample | gap | original all PSNR | Stage70 adapter uniform | original - adapter uniform |
|---|---:|---:|---:|---:|
| `DAVIS/val/bmx-trees` | 4 | 24.04302939633874 | 20.070097715317623 | 3.972931681021116 |
| `DAVIS/val/car-shadow` | 4 | 23.820680846587777 | 21.10255897802341 | 2.718121868564367 |
| `DAVIS/val/goat` | 4 | 20.42551732006098 | 19.73679707382765 | 0.6887202462333306 |
| `DAVIS/val/soapbox` | 4 | 24.68916458241108 | 21.524673663718215 | 3.1644909186928665 |
| `DAVIS/val/bmx-trees` | 8 | 19.889383170223805 | 17.822039666501944 | 2.0673435037218617 |
| `DAVIS/val/car-shadow` | 8 | 22.66411232871394 | 19.18506378945881 | 3.479048539255132 |
| `DAVIS/val/goat` | 8 | 18.20087330775574 | 17.986255403638253 | 0.21461790411748538 |
| `DAVIS/val/soapbox` | 8 | 21.976494843093363 | 19.176595738107597 | 2.7998991049857658 |
| `DAVIS/val/bmx-trees` | 16 | 16.961003092038144 | 16.10083832134726 | 0.8601647706908828 |
| `DAVIS/val/car-shadow` | 16 | 20.56648844037132 | 18.12073768533537 | 2.4457507550359487 |
| `DAVIS/val/goat` | 16 | 16.475211046092024 | 16.74081067318615 | -0.2655996270941259 |
| `DAVIS/val/soapbox` | 16 | 19.41116077143655 | 17.089743502361333 | 2.3214172690752157 |

## Notes

- This is an original StreamSplat pretrained baseline under the current DAVIS preprocessing and Stage70 subset.
- `raw_pred_gs_mib_per_frame` is diagnostic tensor payload size, not a transmitted codec bitstream.
