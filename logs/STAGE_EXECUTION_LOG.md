# Mono-DFCGS 阶段执行日志

## 2026-06-25：执行基线

### 当前仓库

```text
/mnt/hdd2tC/haocheng/Mono-DFCGS
```

### Git 远端

```text
git@github.com:hctang02/Mono-DFCGS.git
```

### 阶段计划文件

```text
docs/MONO_DFCGS_STAGE1_5_EXECUTION_PLAN.md
```

### 外部已知资源

```text
/mnt/hdd2tC/tmp/opencode/StreamSplat
/mnt/hdd2tC/tmp/opencode/streamsplat_repro
/mnt/hdd2tC/tmp/opencode/streamsplat_venv
/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/streamsplat.safetensors
/mnt/hdd2tC/tmp/opencode/StreamSplat/checkpoints/depth_anything_v2_vitl.pth
```

### 执行状态

- 已保存阶段 1-5 完整执行方案。
- 阶段 1 已完成：补 StreamSplat fair metrics baseline。

## 2026-06-25：阶段 1 StreamSplat Fair Metrics Baseline

### 执行计划

阶段 1 的目标是重新评估 StreamSplat baseline，不只统计 all-frame 指标，而是同时统计：

- all-frame PSNR / SSIM
- middle-only PSNR / SSIM
- given-keyframe PSNR / SSIM
- gap 2 / 4 / 8 / 16

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 被占用约 23.7GB，GPU 1/3/4/5/6/7 基本空闲，因此阶段 1 使用 `CUDA_VISIBLE_DEVICES=1` 执行。

### 新增脚本

```text
scripts/run_stage1_streamsplat_fair_metrics.py
```

### 输出文件

```text
experiments/stage1_streamsplat_fair_metrics/stage1_streamsplat_fair_metrics_summary.json
experiments/stage1_streamsplat_fair_metrics/stage1_streamsplat_fair_metrics_summary.csv
```

重型缓存和 per-frame 细节保存在 git 外部：

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage1_streamsplat_fair_metrics
```

### 关键结果

| sample | gap | keyframe ratio | raw pred_gs MiB/frame | all PSNR | middle PSNR | given PSNR |
|---|---:|---:|---:|---:|---:|---:|
| n3dv | 2 | 0.5062 | 3.0556 | 34.0026 | 33.5896 | 34.4054 |
| n3dv | 4 | 0.2593 | 1.5278 | 33.3961 | 33.0521 | 34.3788 |
| n3dv | 8 | 0.1358 | 0.7639 | 32.5026 | 32.2074 | 34.3810 |
| n3dv | 16 | 0.0741 | 0.3819 | 30.8818 | 30.6031 | 34.3644 |
| meetroom | 2 | 0.5062 | 3.0556 | 34.6654 | 33.1828 | 36.1118 |
| meetroom | 4 | 0.2593 | 1.5278 | 32.4263 | 31.1640 | 36.0328 |
| meetroom | 8 | 0.1358 | 0.7639 | 29.6881 | 28.6872 | 36.0580 |
| meetroom | 16 | 0.0741 | 0.3819 | 26.7990 | 26.0567 | 36.0775 |
| driving | 2 | 0.5062 | 3.0556 | 32.7780 | 28.5295 | 36.9229 |
| driving | 4 | 0.2593 | 1.5278 | 30.1950 | 27.8362 | 36.9345 |
| driving | 8 | 0.1358 | 0.7639 | 27.8651 | 26.4415 | 36.9245 |
| driving | 16 | 0.0741 | 0.3819 | 25.1163 | 24.1721 | 36.9199 |
| robot | 2 | 0.5065 | 3.0536 | 30.3682 | 24.6836 | 35.9070 |
| robot | 4 | 0.2597 | 1.5268 | 26.4895 | 23.1786 | 35.9256 |
| robot | 8 | 0.1429 | 0.8036 | 23.8811 | 21.8721 | 35.9357 |
| robot | 16 | 0.0779 | 0.4018 | 21.8207 | 20.6381 | 35.8144 |

### 结论

- all-frame 指标确实会受到 given-keyframes 的高 PSNR 影响。
- driving / robot 的 middle-only PSNR 明显低于 given-keyframe PSNR，说明动态预测难度更高。
- gap 增大时，raw pred_gs size 近似按 keyframe pair 数下降，但 middle-only 质量同步下降。
- 阶段 2 需要从这些 pair-level Gaussian 中建立 keyframe Gaussian anchor size 口径。

## 2026-06-25：阶段 2 Keyframe Gaussian Anchor Baseline

### 执行计划

阶段 2 的目标是从 StreamSplat pair-level `pred_gs` 中抽取唯一 keyframe Gaussian anchors，并估计只传关键帧 Gaussian 时的 transmitted size。

统计两种 profile：

- `static_anchor`：传 `rgb + base opacity + scale + xyz_static + rot_static`，共 13 values / Gaussian，更符合后续 codec 设定。
- `full_half_anchor`：传 pair half 中全部字段，作为保守上界，共 22 values / Gaussian。

统计 codec：

- `float32`
- `float16`
- `q8`
- `q6`
- `q4`

统计 opacity pruning threshold：

- `0.0`
- `0.05`
- `0.1`
- `0.2`

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 被占用约 23.7GB，GPU 6 有运行进程，其余 GPU 基本空闲。阶段 2 使用 `CUDA_VISIBLE_DEVICES=1` 执行。

### 新增脚本

```text
scripts/run_stage2_keyframe_gaussian_anchor_baseline.py
```

### 输出文件

```text
experiments/stage2_keyframe_gaussian_anchor/stage2_keyframe_gaussian_anchor_summary.json
experiments/stage2_keyframe_gaussian_anchor/stage2_keyframe_gaussian_anchor_summary.csv
```

### 主配置结果

主配置为 `profile=static_anchor, codec=q8, opacity_threshold=0.0`。

| sample | gap | keyframes | total MiB | avg MiB/frame |
|---|---:|---:|---:|---:|
| n3dv | 2 | 41 | 18.738 | 0.231337 |
| meetroom | 2 | 41 | 18.738 | 0.231337 |
| driving | 2 | 41 | 18.738 | 0.231337 |
| robot | 2 | 39 | 17.824 | 0.231483 |
| n3dv | 4 | 21 | 9.598 | 0.118490 |
| meetroom | 4 | 21 | 9.598 | 0.118490 |
| driving | 4 | 21 | 9.598 | 0.118490 |
| robot | 4 | 20 | 9.141 | 0.118709 |
| n3dv | 8 | 11 | 5.027 | 0.062066 |
| meetroom | 8 | 11 | 5.027 | 0.062066 |
| driving | 8 | 11 | 5.027 | 0.062066 |
| robot | 8 | 11 | 5.027 | 0.065290 |
| n3dv | 16 | 6 | 2.742 | 0.033854 |
| meetroom | 16 | 6 | 2.742 | 0.033854 |
| driving | 16 | 6 | 2.742 | 0.033854 |
| robot | 16 | 6 | 2.742 | 0.035613 |

### 结论

- 只传 static keyframe Gaussian anchors 后，估算码率远低于阶段 1 的 raw pair-level `pred_gs` size。
- gap=4 的 static q8 transmitted size 约为 `0.1185 MiB/frame`，适合作为阶段 3 的 keyframe-Gaussian-only codec 默认点。
- opacity threshold 到 `0.1` 基本不剪枝，说明当前 StreamSplat static opacity 大多较高；简单 opacity pruning 对码率帮助有限，需要后续考虑 top-K / learned importance。
- 阶段 3 将把阶段 1 的重建质量和阶段 2 的 transmitted keyframe Gaussian size 合并成 uniform keyframe Gaussian codec RD baseline。

## 2026-06-25：阶段 3 Uniform Keyframe Gaussian Codec Baseline

### 执行计划

阶段 3 的目标是合并阶段 1 的 sparse-keyframe reconstruction quality 和阶段 2 的 transmitted keyframe Gaussian anchor size，得到第一个 RD-compatible Mono-DFCGS baseline。

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。该阶段为 CPU 汇总脚本，但仍按要求检查 GPU。GPU 2 被占用，其他 GPU 基本空闲。

### 新增脚本

```text
scripts/run_stage3_uniform_keyframe_gaussian_codec.py
```

### 输出文件

```text
experiments/stage3_uniform_keyframe_gaussian_codec/stage3_uniform_keyframe_gaussian_codec_summary.json
experiments/stage3_uniform_keyframe_gaussian_codec/stage3_uniform_keyframe_gaussian_codec_summary.csv
experiments/stage3_uniform_keyframe_gaussian_codec/stage3_uniform_keyframe_gaussian_codec_main_q8.csv
```

### 主配置 RD 结果

主配置为 `profile=static_anchor, codec=q8, opacity_threshold=0.0`。

| sample | gap | avg transmitted MiB/frame | all PSNR | middle PSNR | given PSNR |
|---|---:|---:|---:|---:|---:|
| n3dv | 2 | 0.231337 | 34.0026 | 33.5896 | 34.4054 |
| n3dv | 4 | 0.118490 | 33.3961 | 33.0521 | 34.3788 |
| n3dv | 8 | 0.062066 | 32.5026 | 32.2074 | 34.3810 |
| n3dv | 16 | 0.033854 | 30.8818 | 30.6031 | 34.3644 |
| meetroom | 2 | 0.231337 | 34.6654 | 33.1828 | 36.1118 |
| meetroom | 4 | 0.118490 | 32.4263 | 31.1640 | 36.0328 |
| meetroom | 8 | 0.062066 | 29.6881 | 28.6872 | 36.0580 |
| meetroom | 16 | 0.033854 | 26.7990 | 26.0567 | 36.0775 |
| driving | 2 | 0.231337 | 32.7780 | 28.5295 | 36.9229 |
| driving | 4 | 0.118490 | 30.1950 | 27.8362 | 36.9345 |
| driving | 8 | 0.062066 | 27.8651 | 26.4415 | 36.9245 |
| driving | 16 | 0.033854 | 25.1163 | 24.1721 | 36.9199 |
| robot | 2 | 0.231483 | 30.3682 | 24.6836 | 35.9070 |
| robot | 4 | 0.118709 | 26.4895 | 23.1786 | 35.9256 |
| robot | 8 | 0.065290 | 23.8811 | 21.8721 | 35.9357 |
| robot | 16 | 0.035613 | 21.8207 | 20.6381 | 35.8144 |

### 结论

- 阶段 3 已形成可与 FCGS / D-FCGS / CWGS 对齐的 RD 表：横轴为 transmitted keyframe Gaussian size，纵轴为 StreamSplat sparse-keyframe reconstruction quality。
- 当前质量仍来自 StreamSplat RGB/depth-conditioned reconstruction，不是最终 Gaussian-anchor-only predictor；因此阶段 3 是 uniform keyframe Gaussian codec baseline / upper-reference，而非最终方法。
- 阶段 4 需要开始实现 Gaussian-anchor-conditioned dynamic predictor 的最小闭环，逐步替代对 RGB/depth features 的依赖。

## 2026-06-25：阶段 4 Gaussian-Anchor Dynamic Predictor Smoke

### 执行计划

阶段 4 的目标是实现 Gaussian-anchor-conditioned predictor 的最小闭环，而不是直接训练最终模型。该阶段验证：

- static Gaussian anchor flatten / unflatten
- payload estimate / q8 size estimate
- 轻量 GaussianAnchorDynamicPredictor
- 输入左右关键帧 anchors 和 normalized time，输出中间 Gaussian anchor fields

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 被占用约 23.7GB，GPU 1 空闲。阶段 4 使用 `CUDA_VISIBLE_DEVICES=1` 执行 smoke。

### 新增文件

```text
mono_dfcgs/gaussian_codec.py
mono_dfcgs/anchor_predictor.py
scripts/run_stage4_gaussian_anchor_predictor_smoke.py
```

### 输出文件

```text
experiments/stage4_gaussian_anchor_predictor_smoke/stage4_gaussian_anchor_predictor_smoke_summary.json
```

### Smoke 结果

| item | value |
|---|---:|
| batch | 2 |
| gaussians | 1024 |
| hidden dim | 128 |
| parameters | 102925 |
| device | cuda |
| mean MSE to linear target | 0.089763 |
| q8 payload MiB for left anchor | 0.025391 |

所有时间点 `0.0 / 0.25 / 0.5 / 0.75 / 1.0` 的输出字段形状均为：

```text
rgb: [2, 1024, 3]
opacity: [2, 1024, 1]
scale: [2, 1024, 2]
xyz: [2, 1024, 3]
rot: [2, 1024, 4]
```

### 结论

- Gaussian-anchor predictor 的最小接口已经跑通。
- 当前 predictor 只是 stage-4 smoke model，输出是 static anchor field format，不包含真实 renderer 和动态 3DGS lifecycle。
- 阶段 5 将基于这个接口实现 smoke training，验证反向传播和 loss 下降。

## 2026-06-25：阶段 5 Training / Evaluation Smoke Pipeline

### 执行计划

阶段 5 的目标是建立训练工程 smoke pipeline：

- synthetic left/right keyframe Gaussian anchors
- random normalized time
- optional q8 quantization-aware anchor perturbation
- GaussianAnchorDynamicPredictor training
- eval loss before/after training

该阶段不做真实视频大规模训练，也不声称最终重建质量，只验证训练路径、反向传播和 loss 下降。

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 被占用约 23.7GB，GPU 1 空闲。阶段 5 使用 `CUDA_VISIBLE_DEVICES=1` 执行。

### 新增脚本

```text
scripts/run_stage5_training_smoke.py
```

### 输出文件

```text
experiments/stage5_training_smoke/stage5_training_smoke_summary.json
```

### Smoke 训练配置

| item | value |
|---|---:|
| batch | 4 |
| gaussians | 512 |
| hidden dim | 128 |
| steps | 80 |
| lr | 0.001 |
| quant bits | 8 |
| parameters | 102925 |
| device | cuda |

### Smoke 训练结果

| metric | value |
|---|---:|
| initial eval loss | 0.089216 |
| final eval loss | 0.014656 |
| eval loss ratio | 0.164271 |
| first train loss | 0.090444 |
| last train loss | 0.013557 |
| train loss ratio | 0.149896 |

### 结论

- Gaussian-anchor predictor 的训练路径已跑通。
- q8 quantization-aware synthetic anchors 下，80-step smoke training 能显著降低 proxy loss。
- 下一步应从 synthetic anchors 过渡到真实导出的 keyframe Gaussian anchors，并接入 renderer / RGB supervision。
