# Stage80 Adapter Training Smoke

## Configuration

- task manifest: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_tasks.csv`
- codecs: `['q10', 'q12']`
- gaps: `[4, 8, 16]`
- train tasks: `6`
- eval tasks: `6`
- steps: `6`
- heavy root: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage80_adapter_training_smoke`

## Evaluation

| checkpoint | step | model PSNR | linear PSNR | margin |
|---|---:|---:|---:|---:|
| initial | 0 | 17.669089783599734 | 17.669089783599734 | 0.0 |
| best | 6 | 17.669478613164078 | 17.669089783599734 | 0.00038882956435060123 |
| final | 6 | 17.669478613164078 | 17.669089783599734 | 0.00038882956435060123 |
| reference adapter | -1 | 17.82809583904211 | 17.669089783599734 | 0.15900605544237814 |

## Notes

- This is a smoke run, not a final long-training result.
- Training uses target RGB only as offline supervision; transmitted test-time inputs remain endpoint Gaussian anchors plus normalized time.
- Checkpoints are stored outside git under the heavy root.
