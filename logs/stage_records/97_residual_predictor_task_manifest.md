# Stage97 Residual Predictor Task Manifest

Date: 2026-06-28

## Goal

Prepare a train/eval task manifest for future learned or deployable residual predictor experiments.

## Implementation

Added:

```text
scripts/run_stage97_residual_predictor_task_manifest.py
```

The script reads Stage79 tasks and Stage61 dense anchors, then writes q12 residual-predictor task rows. It stores references only and does not copy heavy labels, payloads, anchors, or tensors.

## Run

GPU check was performed before execution. The first run failed because the script indirectly imported a torch-dependent Stage85 module under system Python. The script was changed to use a self-contained CSV parser and then rerun successfully.

```text
python scripts/run_stage97_residual_predictor_task_manifest.py
```

## Outputs

```text
experiments/stage97_residual_predictor_task_manifest/stage97_residual_predictor_tasks.csv
experiments/stage97_residual_predictor_task_manifest/stage97_residual_predictor_task_summary.csv
experiments/stage97_residual_predictor_task_manifest/stage97_residual_predictor_missing_dense_targets.csv
experiments/stage97_residual_predictor_task_manifest/stage97_residual_predictor_task_manifest_summary.json
experiments/stage97_residual_predictor_task_manifest/stage97_residual_predictor_task_manifest_report.md
```

## Results

| split | codec | gap | tasks | sequences | potential labels |
|---|---|---:|---:|---:|---:|
| eval | q12 | 4 | 1463 | 30 | 2926 |
| eval | q12 | 8 | 1707 | 30 | 3414 |
| eval | q12 | 16 | 1830 | 30 | 3660 |
| train | q12 | 4 | 3087 | 60 | 6174 |
| train | q12 | 8 | 3604 | 60 | 7208 |
| train | q12 | 16 | 3863 | 60 | 7726 |

Total tasks: `15554`.

Missing dense targets: `0`.

Potential base-method labels: `31108`.

## Conclusion

- Stage97 creates the data entry point for residual predictor experiments.
- Target dense anchors are training/encoder-side teacher label sources, not decoder-side inputs.
- Future stages should start with a small residual importance or payload predictor smoke before attempting full residual value prediction.
