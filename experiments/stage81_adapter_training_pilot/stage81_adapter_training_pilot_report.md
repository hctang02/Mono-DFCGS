# Stage81 Adapter Training Pilot

## Configuration

- task manifest: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_tasks.csv`
- codecs: `['q10', 'q12']`
- gaps: `[4, 8, 16]`
- train tasks: `48`
- eval tasks: `18`
- steps: `48`
- heavy root: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage81_adapter_training_pilot`
- init checkpoint: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors`

## Evaluation

| checkpoint | step | model PSNR | linear PSNR | margin |
|---|---:|---:|---:|---:|
| initial | 0 | 18.43683802603116 | 18.37495762954318 | 0.061880396487980015 |
| best | 24 | 18.481773239223003 | 18.37495762954318 | 0.10681560967982436 |
| final | 48 | 18.48140534613068 | 18.37495762954318 | 0.10644771658749658 |
| reference adapter | -1 | 18.43683802603116 | 18.37495762954318 | 0.061880396487980015 |

## Notes

- This is a controlled pilot run, not a final long-training result.
- Training uses target RGB only as offline supervision; transmitted test-time inputs remain endpoint Gaussian anchors plus normalized time.
- Checkpoints are stored outside git under the heavy root.
