# Stage73 Low-PSNR Diagnosis

## Main Diagnosis

The Stage70 numbers are low mostly because the current anchor-only path renders static anchors and discards original StreamSplat dynamic Gaussian components, not because of a large q8 quantization loss.

## Gap Summary

| gap | original all | float adapter all | q8 adapter all | q8 loss | original - float adapter | original given - float given |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 23.244598036349643 | 21.347280410784887 | 20.608531857721726 | 0.7387485530631608 | 1.8973176255647592 | 0.020452617603702095 |
| 8 | 20.682715912446714 | 18.93647863322668 | 18.54248864942665 | 0.39398998380002936 | 1.7462372792200318 | 0.011830362977251596 |
| 16 | 18.353465837484507 | 17.244538002072254 | 17.01303254555753 | 0.23150545651472498 | 1.1089278354122554 | -0.026037018245909316 |

## Notes

- `float_*` uses the saved fp16 static anchors without the additional q8 simulation.
- `q8_*` matches the Stage70 q8 static-anchor protocol.
- If `q8_loss` is small but `original - float adapter` is large, the main loss is static-anchor-only modeling, not quantization.
- If `original given - float given` is large, even keyframe rendering loses quality when dynamic Gaussian fields are discarded.

## Outputs

- Summary JSON: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage73_low_psnr_diagnosis/stage73_low_psnr_diagnosis_summary.json`
- Diagnosis CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage73_low_psnr_diagnosis/stage73_low_psnr_diagnosis.csv`
- Gap summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage73_low_psnr_diagnosis/stage73_low_psnr_gap_summary.csv`
