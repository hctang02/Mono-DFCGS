# Stage97 Residual Predictor Task Manifest

## Scope

- task count: `15554`
- missing dense target count: `0`
- potential base-method labels: `31108`
- codecs: `['q12']`
- gaps: `[4, 8, 16]`
- base methods: `['linear', 'stage65_adapter']`
- residual codec: `entropy_q6_top10_sorted_delta_zlib`
- no heavy labels or tensors are stored

## Summary

| split | codec | gap | tasks | sequences | potential labels |
|---|---|---:|---:|---:|---:|
| eval | q12 | 4 | 1463 | 30 | 2926 |
| eval | q12 | 8 | 1707 | 30 | 3414 |
| eval | q12 | 16 | 1830 | 30 | 3660 |
| train | q12 | 4 | 3087 | 60 | 6174 |
| train | q12 | 8 | 3604 | 60 | 7208 |
| train | q12 | 16 | 3863 | 60 | 7726 |

## Outputs

- tasks CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage97_residual_predictor_task_manifest/stage97_residual_predictor_tasks.csv`
- summary CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage97_residual_predictor_task_manifest/stage97_residual_predictor_task_summary.csv`
- missing dense targets CSV: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage97_residual_predictor_task_manifest/stage97_residual_predictor_missing_dense_targets.csv`

## Notes

- This package prepares supervised residual predictor data but does not train a model.
- Target dense anchors are training/encoder-side label sources, not decoder-side inputs.
- Any transmitted residual side-info remains part of total RD rate.
