# Stage83 Gap-Aware Adapter Training

## Configuration

- task manifest: `/mnt/hdd2tC/haocheng/Mono-DFCGS/experiments/stage79_adapter_training_task_manifest/stage79_adapter_training_tasks.csv`
- codecs: `['q10', 'q12']`
- gaps: `[4, 8, 16]`
- train tasks: `72`
- eval tasks: `60`
- steps: `72`
- heavy root: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage83_gap_aware_adapter_training`
- init checkpoint: `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/best_adapter.safetensors`
- gap loss weights: `{'4': 3.0, '8': 1.0, '16': 1.0}`
- best metric: `protected_gap4_margin`
- best selection score: `0.08919446622176253`

## Evaluation

| checkpoint | step | model PSNR | linear PSNR | margin |
|---|---:|---:|---:|---:|
| initial | 0 | 19.369958827711727 | 19.259310564194283 | 0.11064826351743612 |
| best | 0 | 19.369958827711727 | 19.259310564194283 | 0.11064826351743612 |
| final | 72 | 19.372171148779863 | 19.259310564194283 | 0.11286058458557878 |
| reference adapter | -1 | 19.369958827711727 | 19.259310564194283 | 0.11064826351743612 |

## Gap Margins

| gap | initial | best | final | reference |
|---:|---:|---:|---:|---:|
| 4 | -0.010726898647836793 | -0.010726898647836793 | -0.044278793981359484 | -0.010726898647836793 |
| 8 | 0.05364375708512303 | 0.05364375708512303 | 0.06219831435034278 | 0.05364375708512303 |
| 16 | 0.3343528251045902 | 0.3343528251045902 | 0.3725368725315138 | 0.3343528251045902 |

## Notes

- This is a gap-aware pilot run, not a final long-training result.
- Training uses target RGB only as offline supervision; transmitted test-time inputs remain endpoint Gaussian anchors plus normalized time.
- Checkpoints are stored outside git under the heavy root.
