# Stage79 Adapter Training Task Manifest

Date: 2026-06-28

## Goal

Prepare a structured adapter training task manifest for stronger Gaussian adapter training.

## Scope

- Use Stage61 DAVIS dense gap1 anchors.
- Splits: DAVIS train and val.
- Codecs: `q10`, `q12`.
- Reference gaps: `4`, `8`, `16`.
- Tasks: each intermediate target frame between selected keyframes.

## Expected Outputs

```text
scripts/run_stage79_adapter_training_task_manifest.py
experiments/stage79_adapter_training_task_manifest/
```

Expected tables:

- adapter task rows.
- sequence summary.
- codec/gap summary.
- train/eval scoped summary.

## Result

Stage79 completed as a CPU-only manifest generation stage.

Outputs:

```text
experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_task_manifest_summary.json
experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_tasks.csv
experiments/stage79_adapter_training_task_manifest/stage79_adapter_sequence_summary.csv
experiments/stage79_adapter_training_task_manifest/stage79_adapter_task_summary.csv
experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_task_manifest_report.md
```

Coverage:

| split | sequences | frames |
|---|---:|---:|
| train | 60 | 4209 |
| val | 30 | 1999 |

Task summary:

| split | codec | gap | sequences | tasks |
|---|---|---:|---:|---:|
| eval | q10 | 4 | 30 | 1463 |
| eval | q10 | 8 | 30 | 1707 |
| eval | q10 | 16 | 30 | 1830 |
| eval | q12 | 4 | 30 | 1463 |
| eval | q12 | 8 | 30 | 1707 |
| eval | q12 | 16 | 30 | 1830 |
| train | q10 | 4 | 60 | 3087 |
| train | q10 | 8 | 60 | 3604 |
| train | q10 | 16 | 60 | 3863 |
| train | q12 | 4 | 60 | 3087 |
| train | q12 | 8 | 60 | 3604 |
| train | q12 | 16 | 60 | 3863 |

Total tasks: `31108`.

Conclusion:

- Stage79 provides a clean task table for Stage80 adapter training.
- The manifest references Stage61 gap1 `.pt` items and source sides without copying anchor tensors.
- Stage80 can now implement task loading, q10/q12 quantized anchors, rendered RGB loss, and validation splits using this manifest.
