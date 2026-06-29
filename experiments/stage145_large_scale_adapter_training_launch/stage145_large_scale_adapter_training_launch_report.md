# Stage145 Large-Scale Adapter Training Launch

## Configuration

- task manifest: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_tasks.csv`
- codecs: `['q12']`
- gaps: `[4, 8]`
- available train rows: `6691`
- selected train rows: `6691`
- selected eval rows: `32`
- steps: `80`
- eval interval: `40`
- init checkpoint: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors`
- heavy root: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage145_large_scale_adapter_training_launch`
- gap loss weights: `{4: 2.0, 8: 1.5}`
- best metric: `min_gap_margin`

## Evaluation

| checkpoint | step | model PSNR | linear PSNR | margin | min gap margin |
|---|---:|---:|---:|---:|---:|
| initial | 0 | 19.30193946735575 | 19.200317738313426 | 0.10162172904232497 | 0.03760697804374987 |
| best | 80 | 19.313366675051686 | 19.200317738313426 | 0.11304893673826172 | 0.05164998002879758 |
| final | 80 | 19.313366675051686 | 19.200317738313426 | 0.11304893673826172 | 0.05164998002879758 |

## Gap Evaluation

| gap | initial model | best model | final model | best margin |
|---:|---:|---:|---:|---:|
| 4 | 20.05304674887969 | 20.067089750864742 | 20.067089750864742 | 0.05164998002879758 |
| 8 | 18.45068454829528 | 18.459147189130217 | 18.459147189130217 | 0.18263442100898775 |

## Notes

- This stage starts model-side quality rescue; it is not a q-bit tuning stage.
- Target RGB is used only for offline training supervision.
- Decoder-side inputs remain endpoint Gaussian anchors plus normalized time.
- Heavy checkpoints and optimizer state are outside git.
