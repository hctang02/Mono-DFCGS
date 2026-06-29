# Stage146 Gap-Balanced Adapter Training V2

## Configuration

- task manifest: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_tasks.csv`
- codecs: `['q12']`
- gaps: `[4, 8]`
- available train rows: `6691`
- selected train rows: `6691`
- selected eval rows: `64`
- steps: `240`
- eval interval: `80`
- init checkpoint: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage145_large_scale_adapter_training_launch/best_adapter.safetensors`
- heavy root: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage146_gap_balanced_adapter_training_v2`
- gap loss weights: `{4: 2.0, 8: 1.5}`
- best metric: `min_gap_margin`

## Evaluation

| checkpoint | step | model PSNR | linear PSNR | margin | min gap margin |
|---|---:|---:|---:|---:|---:|
| initial | 0 | 19.697228869272262 | 19.5588739709682 | 0.13835489830407682 | 0.12251526389602437 |
| best | 0 | 19.697228869272262 | 19.5588739709682 | 0.13835489830407682 | 0.12251526389602437 |
| final | 240 | 19.677331542393684 | 19.5588739709682 | 0.11845757142549357 | 0.10497032571835363 |

## Gap Evaluation

| gap | initial model | best model | final model | best margin |
|---:|---:|---:|---:|---:|
| 4 | 20.05174662536213 | 20.05174662536213 | 20.03420168718447 | 0.12251526389602437 |
| 8 | 19.31983899988628 | 19.31983899988628 | 19.297437517293815 | 0.155216444609423 |

## Notes

- This stage starts model-side quality rescue; it is not a q-bit tuning stage.
- Target RGB is used only for offline training supervision.
- Decoder-side inputs remain endpoint Gaussian anchors plus normalized time.
- Heavy checkpoints and optimizer state are outside git.
