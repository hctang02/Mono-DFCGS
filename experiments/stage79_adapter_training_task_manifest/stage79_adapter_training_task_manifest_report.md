# Stage79 Adapter Training Task Manifest

## Summary

| split | codec | gap | sequences | tasks | tasks / sequence |
|---|---|---:|---:|---:|---:|
| eval | q10 | 4 | 30 | 1463 | 48.766666666666666 |
| eval | q10 | 8 | 30 | 1707 | 56.9 |
| eval | q10 | 16 | 30 | 1830 | 61.0 |
| eval | q12 | 4 | 30 | 1463 | 48.766666666666666 |
| eval | q12 | 8 | 30 | 1707 | 56.9 |
| eval | q12 | 16 | 30 | 1830 | 61.0 |
| train | q10 | 4 | 60 | 3087 | 51.45 |
| train | q10 | 8 | 60 | 3604 | 60.06666666666667 |
| train | q10 | 16 | 60 | 3863 | 64.38333333333334 |
| train | q12 | 4 | 60 | 3087 | 51.45 |
| train | q12 | 8 | 60 | 3604 | 60.06666666666667 |
| train | q12 | 16 | 60 | 3863 | 64.38333333333334 |

## Sequence Coverage

| split | sequences | frames |
|---|---:|---:|
| train | 60 | 4209 |
| val | 30 | 1999 |

## Notes

- This manifest does not copy anchor tensors; it references Stage61 gap1 `.pt` items and source sides.
- `task_split=train` maps to DAVIS train, and `task_split=eval` maps to DAVIS val.
- Codecs are planned input quantization settings for Stage80 training/evaluation.
