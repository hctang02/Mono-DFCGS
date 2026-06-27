# Stage64 Adapter Architecture / Teacher Study

Date: 2026-06-27

## Goal

Run a small ablation before medium/long adapter training:

- RGB-loss residual adapter, hidden dim 128.
- RGB-loss residual adapter, hidden dim 256.
- Dense-gap1 anchor teacher distillation, hidden dim 128.
- Dense-gap1 anchor teacher distillation, hidden dim 256.

## Design

Teacher distillation uses dense gap1 static anchors exported in Stage61 as offline training targets for intermediate frames. This does not change test-time inputs: the adapter still receives only quantized left/right keyframe Gaussian anchors plus timestamp.

## Planned Scope

- Manifest: `experiments/stage61_davis_anchor_export_data_full/stage61_davis_anchor_export_manifest.csv`.
- Train split: DAVIS train.
- Eval split: DAVIS val.
- Gaps: `2, 4, 8, 16`.
- Selected train rows: up to `4` per gap.
- Selected eval rows: up to `2` per gap.
- Targets per row: `1`.
- Steps: small ablation only, not final medium/long claim.

## Expected Outputs

Tracked outputs under:

```text
experiments/stage64_adapter_teacher_study/
```

External checkpoints under:

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage64_adapter_teacher_study/
```

## Execution

运行前按要求使用 `nvidia-smi` 检查 GPU。GPU1/GPU3/GPU4/GPU5 空闲，Stage64 使用 GPU1。

Command:

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage64_adapter_teacher_study.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage64_adapter_teacher_study.py --device cuda --steps 48 --eval_interval 24 --frame_gaps 2 4 8 16 --max_train_rows_per_gap 4 --max_eval_rows_per_gap 2 --targets_per_row 1
```

## Outputs

Tracked outputs:

```text
experiments/stage64_adapter_teacher_study/stage64_adapter_teacher_study_summary.json
experiments/stage64_adapter_teacher_study/stage64_variant_summary.csv
experiments/stage64_adapter_teacher_study/stage64_validation_log.csv
experiments/stage64_adapter_teacher_study/stage64_train_log.csv
experiments/stage64_adapter_teacher_study/stage64_best_eval_rows.csv
```

External checkpoints:

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage64_adapter_teacher_study/<variant>/best_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage64_adapter_teacher_study/<variant>/final_adapter.safetensors
```

Output sizes:

| path | size |
|---|---:|
| `experiments/stage64_adapter_teacher_study` | `48K` |
| `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage64_adapter_teacher_study` | `7.8M` |

Disk state after run:

| mount | free |
|---|---:|
| `/data` | `1011G` |
| `/mnt/hdd2tC` | `3.7G` |

## Results

Scope:

| item | value |
|---|---:|
| available train rows | `3926` |
| available eval rows | `1853` |
| selected train rows | `16` |
| selected eval rows | `8` |
| train tasks | `16` |
| eval tasks | `8` |
| steps | `48` |
| eval interval | `24` |
| quant bits | `8` |

Variant summary:

| variant | loss | hidden dim | params | best step | best margin over linear PSNR | best teacher MSE |
|---|---|---:|---:|---:|---:|---:|
| `rgb_h128` | RGB render loss | `128` | `102925` | `48` | `+0.012173706030985443 dB` | `0.005200807470828295` |
| `rgb_h256` | RGB render loss | `256` | `402445` | `48` | `+0.017721457863302703 dB` | `0.005200803148909472` |
| `teacher_h128` | dense gap1 teacher | `128` | `102925` | `48` | `+0.002455631876273401 dB` | `0.005200357045396231` |
| `teacher_h256` | dense gap1 teacher | `256` | `402445` | `48` | `+0.005105871063040723 dB` | `0.005198333790758625` |

Best by rendered PSNR:

```text
rgb_h256, best margin +0.017721457863302703 dB
```

Best by teacher-anchor MSE:

```text
teacher_h256, best teacher MSE 0.005198333790758625
```

## Conclusion

- RGB render loss with hidden dim `256` is the best short-run route for rendered PSNR and should be used for the next medium adapter training run.
- Dense-gap1 teacher distillation improves teacher-anchor MSE, but in this 48-step ablation it does not translate into the best rendered PSNR.
- Teacher targets remain offline training supervision only; test-time inputs remain q8 endpoint Gaussian anchors plus timestamp.
- Stage64 is a small ablation, not a final adapter quality claim.

## Next Step

Run a longer Stage65 medium adapter training job using the `rgb_h256` route first. Teacher variants can remain as a secondary ablation if longer RGB training saturates.
