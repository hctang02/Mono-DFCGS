# Stage80 Adapter Training Smoke

Date: 2026-06-28

## Goal

Validate the Stage79 adapter task manifest inside a real rendered-RGB adapter training loop.

## Scope

- Input manifest: `experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_tasks.csv`.
- Codecs: `q10`, `q12`.
- Reference gaps: `4`, `8`, `16`.
- Training mode: smoke-only rendered RGB loss.
- Checkpoints are stored outside git.

## Implementation

Added:

```text
scripts/run_stage80_adapter_training_smoke.py
```

The script loads Stage61 gap1 `.pt` anchors through the source item and side recorded in Stage79, applies q-bit static-anchor quantization, renders predicted Gaussian anchors, trains with RGB MSE, and compares against linear anchor interpolation.

## Run

GPU check was performed before execution. GPU4 was idle, so the smoke used:

```text
CUDA_VISIBLE_DEVICES=4
```

Final smoke command covered q10/q12 and all planned gaps with a small task count:

```text
--codecs q10 q12 --gaps 4 8 16 --max_train_tasks 6 --max_eval_tasks 6 --steps 6 --eval_interval 3
```

## Outputs

Repository outputs:

```text
experiments/stage80_adapter_training_smoke/stage80_adapter_training_smoke_summary.json
experiments/stage80_adapter_training_smoke/stage80_adapter_training_smoke_report.md
experiments/stage80_adapter_training_smoke/stage80_train_log.csv
experiments/stage80_adapter_training_smoke/stage80_validation_log.csv
experiments/stage80_adapter_training_smoke/stage80_best_eval_rows.csv
experiments/stage80_adapter_training_smoke/stage80_final_eval_rows.csv
experiments/stage80_adapter_training_smoke/stage80_reference_eval_rows.csv
```

External checkpoints:

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage80_adapter_training_smoke/best_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage80_adapter_training_smoke/final_adapter.safetensors
```

## Result

| checkpoint | model PSNR | linear PSNR | margin |
|---|---:|---:|---:|
| initial | 17.669089783599734 | 17.669089783599734 | 0.0 |
| best/final smoke | 17.669478613164078 | 17.669089783599734 | 0.00038882956435060123 |
| Stage65 reference adapter | 17.82809583904211 | 17.669089783599734 | 0.15900605544237814 |

## Conclusion

- Stage80 validates the task-manifest loader, q10/q12 anchor quantization path, rendered RGB loss, baseline evaluation, and checkpoint export.
- The 6-step run is only a smoke test and should not be interpreted as final adapter performance.
- Stage65 reference remains stronger on this tiny eval slice, so the next step should be controlled medium training from the Stage80 manifest and then full Stage78/Stage77-style evaluation.
