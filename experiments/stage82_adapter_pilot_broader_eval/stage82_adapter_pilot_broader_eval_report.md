# Stage82 Adapter Pilot Broader Eval

## Configuration

- task manifest: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_tasks.csv`
- codecs: `['q10', 'q12']`
- gaps: `[4, 8, 16]`
- train tasks: `1`
- eval tasks: `60`
- steps: `0`
- heavy root: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage82_adapter_pilot_broader_eval`
- init checkpoint: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage81_adapter_training_pilot/best_adapter.safetensors`

## Evaluation

| checkpoint | step | model PSNR | linear PSNR | margin |
|---|---:|---:|---:|---:|
| initial | 0 | 19.360426943660283 | 19.259310564194283 | 0.10111637946599329 |
| best | 0 | 19.360426943660283 | 19.259310564194283 | 0.10111637946599329 |
| final | 0 | 19.360426943660283 | 19.259310564194283 | 0.10111637946599329 |
| reference adapter | -1 | 19.369958827711727 | 19.259310564194283 | 0.11064826351743612 |

## Notes

- This is an evaluation-only run, not additional training.
- Training uses target RGB only as offline supervision; transmitted test-time inputs remain endpoint Gaussian anchors plus normalized time.
- Checkpoints are stored outside git under the heavy root.
