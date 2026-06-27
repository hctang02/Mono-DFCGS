# Stage65 RGB-H256 Medium Adapter Training

Date: 2026-06-27

## Goal

Escalate from the Stage64 adapter architecture study to a stronger RGB-render-loss adapter training run.

Stage64 selected `rgb_h256` as the best short-run route by rendered PSNR, so Stage65 starts with that configuration:

- Loss: RGB render loss.
- Adapter hidden dim: `256`.
- Test-time inputs: q8 endpoint Gaussian anchors plus timestamp.
- No dense teacher anchors as transmitted side information.

## Plan

Run a 1024-step sanity training first before a longer 5k-style medium run.

Sanity configuration:

| item | value |
|---|---:|
| script | `scripts/run_stage64_adapter_teacher_study.py` |
| variant | `rgb_h256` |
| frame gaps | `2, 4, 8, 16` |
| max train rows per gap | `16` |
| max eval rows per gap | `4` |
| targets per row | `1` |
| steps | `1024` |
| eval interval | `256` |
| quant bits | `8` |

Expected tracked sanity outputs:

```text
experiments/stage65_rgb_h256_medium_sanity/
```

Expected external checkpoints:

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_sanity/
```

## Decision Rule

- If the 1024-step sanity run remains positive over q8 linear interpolation and does not show severe validation collapse, continue to a longer medium run.
- If validation collapses or becomes strongly negative, stop and inspect learning rate/data coverage before launching a longer run.

## Notes

The Stage64 script still writes filenames prefixed with `stage64_` because it is reused as the common adapter architecture/training harness. The Stage65 summary directory and this record are the authoritative stage context.

## Sanity Run Result

Command:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage64_adapter_teacher_study.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage64_adapter_teacher_study.py --device cuda --summary_root experiments/stage65_rgb_h256_medium_sanity --heavy_root /data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_sanity --steps 1024 --eval_interval 256 --frame_gaps 2 4 8 16 --max_train_rows_per_gap 16 --max_eval_rows_per_gap 4 --targets_per_row 1 --variants rgb_h256
```

Results:

| item | value |
|---|---:|
| available train rows | `3926` |
| available eval rows | `1853` |
| selected train rows | `64` |
| selected eval rows | `16` |
| train tasks | `64` |
| eval tasks | `16` |
| best step | `512` |
| linear PSNR | `19.655442148062438` |
| best model PSNR | `19.70705320602079` |
| best margin over linear | `+0.05161105795835397 dB` |
| final margin over linear | `+0.04578667635357192 dB` |

Validation margins:

| step | margin over linear |
|---:|---:|
| 0 | `0.0` |
| 256 | `+0.036730886749946734` |
| 512 | `+0.05161105795835397` |
| 768 | `+0.04825765918922187` |
| 1024 | `+0.04578667635357192` |

Decision: sanity is positive and the late decline is small, so continue to a longer medium run with broader train/eval row coverage and validation checkpoint selection.

## Medium Run Plan

Configuration:

| item | value |
|---|---:|
| variant | `rgb_h256` |
| frame gaps | `2, 4, 8, 16` |
| max train rows per gap | `32` |
| max eval rows per gap | `8` |
| targets per row | `1` |
| steps | `5000` |
| eval interval | `500` |
| quant bits | `8` |

Expected tracked medium outputs:

```text
experiments/stage65_rgb_h256_medium_training/
```

Expected external medium checkpoints:

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/
```

## Medium Run Result

Command:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage64_adapter_teacher_study.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage64_adapter_teacher_study.py --device cuda --summary_root experiments/stage65_rgb_h256_medium_training --heavy_root /data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training --steps 5000 --eval_interval 500 --frame_gaps 2 4 8 16 --max_train_rows_per_gap 32 --max_eval_rows_per_gap 8 --targets_per_row 1 --variants rgb_h256
```

Results:

| item | value |
|---|---:|
| available train rows | `3926` |
| available eval rows | `1853` |
| selected train rows | `128` |
| selected eval rows | `32` |
| train tasks | `128` |
| eval tasks | `32` |
| parameter count | `402445` |
| best step | `4000` |
| linear PSNR | `18.518044832601554` |
| best model PSNR | `18.79151802978449` |
| best margin over linear | `+0.2734731971829376 dB` |
| final margin over linear | `+0.21907000507035335 dB` |

Validation curve:

| step | model PSNR | linear PSNR | margin over linear |
|---:|---:|---:|---:|
| 0 | `18.518044832601554` | `18.518044832601554` | `0.0` |
| 500 | `18.554830863633462` | `18.518044832601554` | `+0.036786031031908806` |
| 1000 | `18.562608326576196` | `18.518044832601554` | `+0.044563493974642654` |
| 1500 | `18.558602085328477` | `18.518044832601554` | `+0.040557252726923565` |
| 2000 | `18.55889179850922` | `18.518044832601554` | `+0.04084696590766512` |
| 2500 | `18.574883881889722` | `18.518044832601554` | `+0.0568390492881683` |
| 3000 | `18.610383604713636` | `18.518044832601554` | `+0.09233877211208252` |
| 3500 | `18.7588553269502` | `18.518044832601554` | `+0.2408104943486471` |
| 4000 | `18.79151802978449` | `18.518044832601554` | `+0.2734731971829376` |
| 4500 | `18.76831341004217` | `18.518044832601554` | `+0.2502685774406146` |
| 5000 | `18.737114837671907` | `18.518044832601554` | `+0.21907000507035335` |

Gap-wise best-step margins at step `4000`:

| gap | model PSNR | linear PSNR | margin |
|---:|---:|---:|---:|
| 2 | `20.233952928421175` | `19.99038570685019` | `+0.24356722157098484` |
| 4 | `18.769304245873077` | `18.501980123357967` | `+0.26732412251510996` |
| 8 | `18.312973958897846` | `17.98723191690113` | `+0.3257420419967144` |
| 16 | `17.849840985945853` | `17.59258158329692` | `+0.2572594026489324` |

Output sizes:

| path | size |
|---|---:|
| `experiments/stage65_rgb_h256_medium_sanity` | `92K` |
| `experiments/stage65_rgb_h256_medium_training` | `372K` |
| `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_sanity` | `3.1M` |
| `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training` | `3.1M` |

Disk state after Stage65:

| mount | free |
|---|---:|
| `/data` | `1015G` |
| `/mnt/hdd2tC` | `3.7G` |

## Conclusion

- The `rgb_h256` route scales beyond Stage64/Stage65 sanity and gives a substantially larger validation gain in the 5000-step medium run.
- Best checkpoint selection matters: the best step is `4000`; the final checkpoint at step `5000` remains positive but is lower than the best checkpoint.
- All tested gaps are positive at the best step, with the largest gain on gap `8` in this eval subset.
- Teacher-anchor MSE becomes much worse during RGB-only training, so dense-anchor MSE should not be used as the selection criterion for this RGB route.
- Stage65 is still an intermediate eval-task training result, not final all-frame RD evaluation.

## Next Step

Use the Stage65 best `rgb_h256` checkpoint as the adapter candidate for the next full-video/all-frame evaluation or for selector-training label generation. The next planned stage is the feed-forward selector dataset/training path, keeping oracle rendered selection only as an upper bound or training label.
