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

## 2026-06-25：阶段 6 Real Keyframe Gaussian Anchor Dataset Export

### 执行计划

阶段 6 的目标是从 StreamSplat checkpoint 导出真实 keyframe Gaussian anchor dataset，为后续真实 Gaussian-anchor predictor 训练和 renderer/RGB supervision 做准备。

每个 dataset item 对应一个 keyframe pair，包含：

- `left_anchor`: 左关键帧 static Gaussian anchor，float16
- `right_anchor`: 右关键帧 static Gaussian anchor，float16
- pair metadata: sample / gap / left index / right index / segment length
- intermediate frame metadata: 中间帧 index、normalized time、RGB/depth path

实际 `.pt` tensor 文件较大，保存到 git 外部；仓库内只保存 manifest 和 summary。

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 被重度占用，GPU 5 有小进程，GPU 1/3/4/6/7 空闲。阶段 6 使用 `CUDA_VISIBLE_DEVICES=1` 执行。

### 新增脚本

```text
scripts/run_stage6_export_real_anchor_dataset.py
```

### 仓库内输出文件

```text
experiments/stage6_real_anchor_dataset/stage6_real_anchor_dataset_summary.json
experiments/stage6_real_anchor_dataset/stage6_real_anchor_dataset_manifest.csv
experiments/stage6_real_anchor_dataset/stage6_real_anchor_dataset_manifest.json
```

### git 外部重型数据目录

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage6_real_anchor_dataset
```

目录大小：

```text
546M
```

### 导出结果

| sample | gap | pairs | middle frames | total anchor MiB | avg anchor MiB/pair |
|---|---:|---:|---:|---:|---:|
| n3dv | 2 | 40 | 40 | 73.125 | 1.828125 |
| meetroom | 2 | 40 | 40 | 73.125 | 1.828125 |
| driving | 2 | 40 | 40 | 73.125 | 1.828125 |
| robot | 2 | 38 | 38 | 69.469 | 1.828125 |
| n3dv | 4 | 20 | 60 | 36.562 | 1.828125 |
| meetroom | 4 | 20 | 60 | 36.562 | 1.828125 |
| driving | 4 | 20 | 60 | 36.562 | 1.828125 |
| robot | 4 | 19 | 57 | 34.734 | 1.828125 |
| n3dv | 8 | 10 | 70 | 18.281 | 1.828125 |
| meetroom | 8 | 10 | 70 | 18.281 | 1.828125 |
| driving | 8 | 10 | 70 | 18.281 | 1.828125 |
| robot | 8 | 10 | 66 | 18.281 | 1.828125 |
| n3dv | 16 | 5 | 75 | 9.141 | 1.828125 |
| meetroom | 16 | 5 | 75 | 9.141 | 1.828125 |
| driving | 16 | 5 | 75 | 9.141 | 1.828125 |
| robot | 16 | 5 | 71 | 9.141 | 1.828125 |

总计：

- exported pair count: `297`
- anchor fields: `rgb / opacity / scale / xyz / rot`
- anchor dtype: `float16`
- gaussians per anchor: `36864`

### 结论

- 阶段 6 已形成真实 keyframe Gaussian anchor dataset。
- 后续阶段 7 可以直接从 manifest 读取 `.pt` item，用真实左右 anchors 替代 synthetic anchors 训练 predictor。
- 当前 dataset item 保存的是 static keyframe anchors 和 RGB/depth supervision 路径；尚未保存 intermediate teacher Gaussian。阶段 7 可先做 anchor interpolation/proxy training，阶段 8 再接入 renderer 和 RGB loss。

## 2026-06-25：阶段 7 Dataset Inventory

### 执行计划

阶段 7 的目标是整理当前可用数据集和 StreamSplat/DAVIS 相关候选路径，明确后续扩展到 DAVIS / YouTube-VOS / RE10K / CO3D 时哪些数据已存在、哪些需要下载或挂载。

### 新增脚本

```text
scripts/run_stage7_dataset_inventory.py
```

### 输出文件

```text
experiments/stage7_dataset_inventory/stage7_dataset_inventory_summary.json
experiments/stage7_dataset_inventory/stage7_dataset_inventory_candidates.csv
experiments/stage7_dataset_inventory/stage7_current_samples.csv
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 被占用，其余 GPU 基本空闲。该阶段为 CPU inventory 脚本，不占用 GPU。

### 当前样本状态

| sample | path | readable | frames | resolution | fps |
|---|---|---:|---:|---|---:|
| n3dv | `/mnt/hdd2tC/tmp/opencode/gt_reference_videos/n3dv_input_81f_560x336.mp4` | true | 81 | 560x336 | 15 |
| meetroom | `/mnt/hdd2tC/tmp/opencode/gt_reference_videos/meetroom_input_81f_560x336.mp4` | true | 81 | 560x336 | 15 |
| driving | `/mnt/ssd2tB/haocheng/NeoVerse/examples/videos/driving.mp4` | true | 81 | 560x336 | 16 |
| robot | `/mnt/ssd2tB/haocheng/NeoVerse/examples/videos/robot.mp4` | true | 79 | 560x336 | 16 |

### DAVIS / StreamSplat 数据状态

默认候选路径下未检测到完整可用数据根目录：

- DAVIS: 0 available candidates
- YouTube-VOS: 0 available candidates
- RE10K: 0 available candidates
- CO3D: 0 available candidates

已检查候选数量：

- available candidates: `0`
- missing candidates: `20`

### 结论

- 当前机器上可直接使用的是四个工作样本和已导出的 stage6 anchor dataset。
- DAVIS / YouTube-VOS / StreamSplat 原始训练数据尚未在默认路径下就绪。
- 下一步应先下载或挂载 DAVIS 2017 数据，并为 Mono-DFCGS 建立 frame/depth manifest，再扩展真实 anchor export。

## 2026-06-25：阶段 8 DAVIS / YouTube-VOS Manifest Preflight

### 执行计划

阶段 8 的目标是在 DAVIS / YouTube-VOS 原始数据尚未就绪的情况下，先建立可复用的 sequence manifest/preflight 入口。用户后续只要下载或挂载数据 root，即可通过脚本枚举序列、统计 RGB/depth 是否齐全，并判断是否可进入 depth preprocessing 和 anchor export。

### 新增脚本

```text
scripts/run_stage8_davis_vos_manifest.py
```

### 输出文件

```text
experiments/stage8_davis_vos_manifest/stage8_preflight_summary.json
experiments/stage8_davis_vos_manifest/stage8_root_preflight.csv
experiments/stage8_davis_vos_manifest/stage8_sequence_manifest.csv
experiments/stage8_davis_vos_manifest/stage8_dataset_setup_requirements.md
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 被占用，其余 GPU 基本空闲。该阶段为 CPU 文件系统 preflight，不占用 GPU。

### 执行结果

```text
sequence_count: 0
ready_for_depth_count: 0
ready_for_anchor_export_count: 0
```

已检查 root 数量：

- DAVIS: 7
- YouTube-VOS: 5
- total: 12

### 目录要求

DAVIS 推荐目录：

```text
DAVIS/
  ImageSets/2017/train.txt
  ImageSets/2017/val.txt
  JPEGImages/Full-Resolution/<sequence>/*.jpg
  Annotations_unsupervised/Full-Resolution/<sequence>/*.png
  depthImages/Full-Resolution/<sequence>/*.png
```

YouTube-VOS 推荐目录：

```text
YouTube-VOS/
  train/JPEGImages/<sequence>/*.jpg
  valid/JPEGImages/<sequence>/*.jpg
  train/Annotations/<sequence>/*.png
  train/depthImages/<sequence>/*.png
```

### 结论

- 当前默认路径仍未检测到可用 DAVIS / YouTube-VOS sequence。
- 阶段 8 产物提供了明确的数据挂载/下载检查入口。
- 后续如果用户提供 DAVIS root，可运行 `scripts/run_stage8_davis_vos_manifest.py --davis_roots /path/to/DAVIS` 生成 sequence manifest，然后接 depth preprocessing 和 anchor export。

## 2026-06-25：阶段 9 Real Anchor Proxy Training

### 执行计划

阶段 9 的目标是在 stage6 已导出的真实 keyframe Gaussian anchors 上训练 `GaussianAnchorDynamicPredictor`，验证真实数据加载、q8 keyframe anchor 输入模拟、held-out sample 评估和训练闭环。该阶段只使用 raw Gaussian attribute proxy loss，不接 renderer/RGB loss。

### 新增/修改文件

```text
scripts/run_stage9_real_anchor_proxy_training.py
mono_dfcgs/anchor_predictor.py
```

`anchor_predictor.py` 保留默认输出约束行为，同时新增可选关闭输出约束的接口。阶段 9 默认关闭输出约束，在 StreamSplat raw attribute 空间拟合线性 teacher。

### 运行命令

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage9_real_anchor_proxy_training.py --device cuda --frame_gap 4 --steps 80 --eval_batches 8 --gaussians 2048 --batch_items 2
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 被占用；GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage9_real_anchor_proxy_training/stage9_real_anchor_proxy_training_summary.json
experiments/stage9_real_anchor_proxy_training/stage9_train_losses.csv
```

### 训练设置

| item | value |
|---|---:|
| frame_gap | 4 |
| total rows | 79 |
| train rows | 60 |
| eval rows | 19 |
| held-out eval sample | robot |
| batch_items | 2 |
| gaussians_per_item | 2048 |
| steps | 80 |
| eval_batches | 8 |
| quant_bits | 8 |
| parameters | 102925 |

### 结果

| metric | value |
|---|---:|
| initial_eval_loss | 6.341110747598577e-06 |
| final_eval_loss | 4.971821141452892e-07 |
| eval_loss_ratio | 0.07840615531491475 |
| initial_baseline_loss | 5.393901716388427e-07 |
| final_baseline_loss | 5.092729402633722e-07 |
| first_train_loss | 6.465598744398449e-06 |
| last_train_loss | 4.793312768924807e-07 |
| train_loss_ratio | 0.07413563628701135 |

### 结论

- 阶段 9 已完成真实 anchor proxy training smoke。
- 训练后的 eval loss 接近 q8 linear baseline，说明真实 anchor 加载、quantized input simulation 和 predictor 训练闭环可运行。
- 该结果不是最终视频重建质量；下一步需要接 Gaussian renderer / RGB loss，或先实现能把 predicted raw attributes 转回 renderer 所需格式的 adapter。

## 2026-06-25：阶段 10 Renderer Smoke

### 执行计划

阶段 10 的目标是打通 Gaussian anchor 到 StreamSplat renderer 的最小闭环。先不做完整 RGB loss 训练，而是把单帧 static anchor 包装成 renderer 所需的 zero-dynamic Gaussian 格式，渲染一个中间帧，并和对应 RGB target 计算 MSE/PSNR。

### 新增文件

```text
mono_dfcgs/render_adapter.py
scripts/run_stage10_renderer_smoke.py
```

### 运行命令

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage10_renderer_smoke.py --device cuda --sample n3dv --frame_gap 4 --row_index 0
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 被占用；GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage10_renderer_smoke/stage10_renderer_smoke_summary.json
```

### 测试样本

| item | value |
|---|---|
| sample | n3dv |
| frame_gap | 4 |
| pair | 0 -> 4 |
| target frame | 2 |
| normalized_time | 0.5 |
| Gaussian count | 36864 |
| resolution | 512x288 |

### 结果

| metric | value |
|---|---:|
| render_shape | [1, 1, 3, 288, 512] |
| depth_shape | [1, 1, 1, 288, 512] |
| alpha_shape | [1, 1, 1, 288, 512] |
| rgb_mse | 0.0010116836056113243 |
| rgb_psnr | 29.94955287715034 |
| alpha_mean | 0.8632223010063171 |
| alpha_max | 0.9993191957473755 |

### 结论

- 阶段 10 已验证 static keyframe/intermediate anchor 可以通过 zero-dynamic adapter 调用 StreamSplat renderer。
- 当前 PSNR 只是 linear raw anchor 的 renderer smoke，不是最终 learned decoder 的 RGB reconstruction quality。
- 下一步可以把阶段 9 predictor 输出接到 renderer smoke 中，形成 differentiable RGB loss smoke；随后再做小规模 RGB fine-tuning。

## 2026-06-25：阶段 10b Predictor-Renderer RGB Loss Smoke

### 执行计划

阶段 10b 的目标是把 `GaussianAnchorDynamicPredictor` 的输出接入阶段 10 的 renderer adapter，执行少量 RGB MSE 反向传播 step，验证 predictor -> static anchor -> zero-dynamic Gaussian -> renderer -> RGB loss 的 differentiable training 闭环。

### 新增文件

```text
scripts/run_stage10b_predictor_renderer_rgb_smoke.py
```

### 运行命令

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage10b_predictor_renderer_rgb_smoke.py --device cuda --sample n3dv --frame_gap 4 --row_index 0 --steps 5
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 和 GPU 6 有进程；GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage10b_predictor_renderer_rgb_smoke/stage10b_predictor_renderer_rgb_smoke_summary.json
```

### 测试设置

| item | value |
|---|---:|
| sample | n3dv |
| frame_gap | 4 |
| pair | 0 -> 4 |
| target frame | 2 |
| normalized_time | 0.5 |
| Gaussian count | 36864 |
| resolution | 512x288 |
| steps | 5 |
| lr | 0.0001 |
| quant_bits | 8 |
| parameters | 102925 |

### 结果

| metric | value |
|---|---:|
| initial_rgb_mse | 0.0053450739942491055 |
| final_rgb_mse | 0.004406371619552374 |
| rgb_mse_ratio | 0.8243799102301101 |
| initial_rgb_psnr | 22.720462782809005 |
| final_rgb_psnr | 23.559188786071505 |

Per-step RGB MSE：

```text
0.0053450739942491055
0.005080688279122114
0.004836901556700468
0.004612587857991457
0.004406371619552374
```

### 结论

- 阶段 10b 已验证 renderer RGB loss 可以反向传播到 `GaussianAnchorDynamicPredictor`。
- 当前只是单 pair / 单中间帧 / 5 step smoke，不能代表最终重建质量。
- 初始 predictor 带随机 residual，因此初始 PSNR 低于阶段 10 的纯 linear-anchor smoke；后续应先用阶段 9 proxy checkpoint 或 linear-residual 初始化，再做 RGB fine-tuning。

## 2026-06-25：阶段 11 Keyframe Selection Baseline

### 执行计划

阶段 11 的目标是先生成 keyframe selection baseline，而不是立即重跑完整重建。脚本基于 stage1 frame cache 计算 motion score，基于 stage6 gap=2 anchors 计算 Gaussian attribute change score，然后输出 uniform / motion-aware / Gaussian-aware / RD-aware 的 keyframe indices 和 q8 static-anchor rate 估算。

### 新增文件

```text
scripts/run_stage11_keyframe_selection_baseline.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage11_keyframe_selection_baseline.py
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。该阶段为 CPU selection/statistics 脚本，不占用 GPU。

### 输出文件

```text
experiments/stage11_keyframe_selection/stage11_keyframe_selection_summary.csv
experiments/stage11_keyframe_selection/stage11_keyframe_selection_summary.json
```

### 覆盖范围

| item | value |
|---|---:|
| samples | n3dv, meetroom, driving, robot |
| reference gaps | 4, 8, 16 |
| methods | uniform, motion_aware, gaussian_aware, rd_aware |
| output rows | 48 |

### 示例 rate

| sample | gap | keyframes | estimated_q8_static_mib_per_frame |
|---|---:|---:|---:|
| n3dv | 4 | 21 | 0.11848958333333333 |
| n3dv | 8 | 11 | 0.062065972222222224 |
| n3dv | 16 | 6 | 0.033854166666666664 |
| robot | 4 | 20 | 0.11870887418831169 |
| robot | 8 | 11 | 0.0652924788961039 |
| robot | 16 | 6 | 0.03561444886363636 |

### Score summary

| sample | frames | motion_score_max | gaussian_score_max | gaussian_score_rows |
|---|---:|---:|---:|---:|
| n3dv | 81 | 0.004708124324679375 | 0.0009974667336791754 | 40 |
| meetroom | 81 | 0.007059135008603334 | 0.001674364204518497 | 40 |
| driving | 81 | 0.036478396505117416 | 0.005265535321086645 | 40 |
| robot | 77 | 0.06726385653018951 | 0.013615783303976059 | 38 |

### 结论

- 阶段 11 已形成 keyframe selection baseline 和统一 rate 估算入口。
- 当前只输出 keyframe indices，不代表 quality；后续需要把 selected indices 接入 reconstruction/evaluation pipeline，比较 selected-keyframe RD 曲线。
- Gaussian-aware 分数目前来自 stage6 gap=2 pair anchor MSE，是轻量 proxy，不是最终 entropy/RD-aware optimization。

## 2026-06-25：阶段 12 Selected-Keyframe Reconstruction Smoke

### 执行计划

阶段 12 的目标是把阶段 11 输出的非均匀 keyframe indices 接入 StreamSplat reconstruction/evaluation pipeline，验证 selected-keyframe quality 评估闭环。为避免直接运行完整 4 samples × methods × gaps 大实验，本阶段先实现通用 evaluator，并只运行 `robot + rd_aware + reference_gap=4` smoke。

### 新增文件

```text
scripts/run_stage12_selected_keyframe_reconstruction.py
```

### 运行命令

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage12_selected_keyframe_reconstruction.py --sample robot --method rd_aware --reference_gap 4 --batch_size 1 --max_segment_length 20 --save_per_frame
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 被占用；GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage12_selected_keyframe_reconstruction/stage12_selected_keyframe_reconstruction_summary.json
experiments/stage12_selected_keyframe_reconstruction/stage12_selected_keyframe_reconstruction_summary.csv
experiments/stage12_selected_keyframe_reconstruction/robot_rd_aware_gap4_summary.json
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage12_selected_keyframe_reconstruction/robot_rd_aware_gap4/per_frame_metrics.json
```

### 测试设置

| item | value |
|---|---:|
| sample | robot |
| selection_method | rd_aware |
| reference_gap | 4 |
| total_frames | 77 |
| keyframe_count | 20 |
| keyframe_ratio | 0.2597402597402597 |
| max_segment_length | 16 |
| pair_count | 19 |
| estimated_q8_static_mib_per_frame | 0.11870941558441558 |

Selected keyframes:

```text
0 8 10 11 12 22 23 24 36 38 40 41 42 43 44 54 58 74 75 76
```

Segment lengths:

```text
8 2 1 1 10 1 1 12 2 2 1 1 1 1 10 4 16 1 1
```

### 结果

| metric | value |
|---|---:|
| all_psnr_avg | 24.733323284443408 |
| all_ssim_avg | 0.8312911770560525 |
| middle_psnr_avg | 20.742726407113338 |
| middle_ssim_avg | 0.7831615586029855 |
| given_psnr_avg | 36.106524384834096 |
| given_ssim_avg | 0.9684605896472931 |
| raw_pred_gs_mib_per_frame | 1.5267857142857142 |

### 结论

- 阶段 12 已完成非均匀 selected-keyframe reconstruction/evaluation smoke。
- 该结果仍使用 StreamSplat RGB/depth-conditioned pair inference，不是最终 Gaussian-anchor-only decoder。
- 在相同 keyframe count/rate 下，当前 `rd_aware` proxy 选择低于此前 robot uniform gap4 baseline，说明无间隔约束的 top-k selection 会聚簇并造成长 segment，后续 Stage 13 应增加 minimum temporal spacing / segment length penalty 后再做 RD evaluation。

## 2026-06-25：阶段 13 Spacing-Constrained Keyframe Selection

### 执行计划

阶段 13 的目标是修复 Stage 11 top-k selection 容易聚簇的问题。脚本复用 motion / Gaussian / RD 分数，但使用 coverage-first constrained greedy：先保证所有 segment length 不超过 `max_segment_multiplier * reference_gap`，再用剩余 budget 按分数填充。默认 `max_segment_multiplier=2`。

### 新增文件

```text
scripts/run_stage13_spaced_keyframe_selection.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage13_spaced_keyframe_selection.py
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。该阶段为 CPU selection/statistics 脚本，不占用 GPU。

### 输出文件

```text
experiments/stage13_spaced_keyframe_selection/stage13_spaced_keyframe_selection_summary.csv
experiments/stage13_spaced_keyframe_selection/stage13_spaced_keyframe_selection_summary.json
```

### 覆盖范围

| item | value |
|---|---:|
| samples | n3dv, meetroom, driving, robot |
| reference gaps | 4, 8, 16 |
| methods | uniform, motion_spaced, gaussian_spaced, rd_spaced |
| max_segment_multiplier | 2 |
| output rows | 48 |

### 示例结果

| sample | method | gap | keyframes | max_segment_length | rate MiB/frame |
|---|---|---:|---:|---:|---:|
| robot | uniform | 4 | 20 | 4 | 0.11870941558441558 |
| robot | rd_spaced | 4 | 20 | 8 | 0.11870941558441558 |
| robot | rd_spaced | 8 | 11 | 16 | 0.06529017857142858 |
| robot | rd_spaced | 16 | 6 | 32 | 0.03561282467532467 |
| n3dv | rd_spaced | 4 | 21 | 8 | 0.11848958333333333 |
| n3dv | rd_spaced | 8 | 11 | 16 | 0.062065972222222224 |
| n3dv | rd_spaced | 16 | 6 | 32 | 0.033854166666666664 |

### 结论

- 阶段 13 已生成 spacing-constrained keyframe selection baseline，保持与 uniform 相同 keyframe budget/rate。
- 对 Stage 12 中 `robot rd_aware gap4` 的问题，新的 `robot rd_spaced gap4` 将 max segment length 从 16 限制到 8，适合下一步重新跑 selected-keyframe reconstruction 对比。
- 该阶段仍只输出 indices，不代表最终质量；后续 Stage 14 应用 Stage 12 evaluator 跑 `rd_spaced` reconstruction。

## 2026-06-25：阶段 14 Spaced Selected-Keyframe Reconstruction Smoke

### 执行计划

阶段 14 的目标是评估 Stage 13 spacing-constrained selection 是否修复 Stage 12 中的聚簇问题。复用 Stage 12 evaluator，扩展其 method choices 支持 `motion_spaced` / `gaussian_spaced` / `rd_spaced`，然后运行 `robot + rd_spaced + reference_gap=4` reconstruction smoke。

### 修改文件

```text
scripts/run_stage12_selected_keyframe_reconstruction.py
```

### 运行命令

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage12_selected_keyframe_reconstruction.py --sample robot --method rd_spaced --reference_gap 4 --selection_csv experiments/stage13_spaced_keyframe_selection/stage13_spaced_keyframe_selection_summary.csv --summary_root experiments/stage14_spaced_selected_keyframe_reconstruction --heavy_root /mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage14_spaced_selected_keyframe_reconstruction --batch_size 1 --max_segment_length 20 --save_per_frame
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2/3/7 有进程；GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage14_spaced_selected_keyframe_reconstruction/stage12_selected_keyframe_reconstruction_summary.json
experiments/stage14_spaced_selected_keyframe_reconstruction/stage12_selected_keyframe_reconstruction_summary.csv
experiments/stage14_spaced_selected_keyframe_reconstruction/robot_rd_spaced_gap4_summary.json
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage14_spaced_selected_keyframe_reconstruction/robot_rd_spaced_gap4/per_frame_metrics.json
```

### 结果

| metric | Stage 12 rd_aware | Stage 14 rd_spaced |
|---|---:|---:|
| keyframe_count | 20 | 20 |
| estimated_q8_static_mib_per_frame | 0.11870941558441558 | 0.11870941558441558 |
| max_segment_length | 16 | 8 |
| all_psnr_avg | 24.733323284443408 | 25.893064910908713 |
| all_ssim_avg | 0.8312911770560525 | 0.8525844734984559 |
| middle_psnr_avg | 20.742726407113338 | 22.329879626430465 |
| middle_ssim_avg | 0.7831615586029855 | 0.8120484801760891 |
| given_psnr_avg | 36.106524384834096 | 36.04814297167172 |

### 结论

- Stage 14 证明 spacing-constrained selection 在相同 rate 下改善了 selected-keyframe reconstruction smoke。
- 对 `robot gap4`，`rd_spaced` 相比无约束 `rd_aware`：all PSNR 提升约 1.16 dB，middle PSNR 提升约 1.59 dB。
- 结果仍是 StreamSplat RGB/depth-conditioned inference，不是最终 Gaussian-anchor-only decoder；但它验证了 keyframe selection 的评估闭环和 segment constraint 的必要性。

## 2026-06-25：阶段 15 Selected-Keyframe RD Curve Batch

### 执行计划

阶段 15 的目标是把 Stage 14 的单点 smoke 扩展成更完整的 selected-keyframe RD curve。评估范围为 `n3dv/robot × uniform/rd_spaced × gap4/8/16`，共 12 组。该阶段复用 Stage 12 evaluator，仍是 StreamSplat RGB/depth-conditioned selected-pair inference，不是最终 Gaussian-anchor-only decoder。

### 新增文件

```text
scripts/run_stage15_selected_keyframe_rd_curve.py
```

### 运行命令

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage15_selected_keyframe_rd_curve.py --batch_size 1 --max_segment_length 40
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 被占用；GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage15_selected_keyframe_rd_curve/stage15_selected_keyframe_rd_curve_summary.csv
experiments/stage15_selected_keyframe_rd_curve/stage15_selected_keyframe_rd_curve_summary.json
experiments/stage15_selected_keyframe_rd_curve/*_summary.json
```

### RD Summary

| sample | method | gap | rate MiB/frame | all PSNR | middle PSNR | max segment |
|---|---|---:|---:|---:|---:|---:|
| n3dv | uniform | 4 | 0.11848958333333333 | 33.39609679886218 | 33.05215015387749 | 4 |
| n3dv | uniform | 8 | 0.062065972222222224 | 32.50256438589778 | 32.2073772502555 | 8 |
| n3dv | uniform | 16 | 0.033854166666666664 | 30.881761283241584 | 30.60315188772239 | 16 |
| n3dv | rd_spaced | 4 | 0.11848958333333333 | 33.170324376459305 | 32.74286668259031 | 8 |
| n3dv | rd_spaced | 8 | 0.062065972222222224 | 31.644474346834887 | 31.21410477178391 | 16 |
| n3dv | rd_spaced | 16 | 0.033854166666666664 | 30.6905517407433 | 30.39447527528243 | 32 |
| robot | uniform | 4 | 0.11870941558441558 | 26.48952399862992 | 23.178617517791157 | 4 |
| robot | uniform | 8 | 0.06529017857142858 | 23.881147029593635 | 21.872060070351942 | 8 |
| robot | uniform | 16 | 0.03561282467532467 | 21.820691934634624 | 20.638129159906573 | 16 |
| robot | rd_spaced | 4 | 0.11870941558441558 | 25.893064910908713 | 22.329879626430465 | 8 |
| robot | rd_spaced | 8 | 0.06529017857142858 | 23.17668274155252 | 21.04240930250378 | 16 |
| robot | rd_spaced | 16 | 0.03561282467532467 | 21.50606959393489 | 20.28240369488664 | 32 |

### 结论

- Stage 15 完成了 12 组 selected-keyframe RD curve smoke。
- `rd_spaced` 修复了无约束 `rd_aware` 的聚簇问题，但在 n3dv/robot 的当前 RD curve 上仍低于 uniform。
- 当前 motion/Gaussian/RD score 只是轻量 proxy，下一步需要改进 selection objective，例如引入 per-segment interpolation difficulty、minimum spacing 与 entropy/rate proxy 的联合优化，而不是直接 top-k score。

## 2026-06-25：Stage 15 RD Curve Plot

### 执行计划

根据用户要求，将 Stage 15 RD curve CSV 绘制为可直接查看的 PNG 图。横轴为 transmitted Gaussian MiB/frame，纵轴分别为 PSNR/SSIM。

### 新增脚本

```text
scripts/plot_stage15_rd_curve.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/plot_stage15_rd_curve.py
```

### 输出文件

```text
experiments/stage15_selected_keyframe_rd_curve/stage15_rd_curve_all_psnr.png
experiments/stage15_selected_keyframe_rd_curve/stage15_rd_curve_middle_psnr.png
experiments/stage15_selected_keyframe_rd_curve/stage15_rd_curve_all_ssim.png
experiments/stage15_selected_keyframe_rd_curve/stage15_rd_curve_middle_ssim.png
```

### 结论

- 已生成 Stage 15 的 RD 曲线图。
- 最建议优先查看 `stage15_rd_curve_middle_psnr.png`，因为 middle-only 更能反映非关键帧重建质量。

## 2026-06-25：阶段 16 Segment-Error-Aware Keyframe Selection

### 执行计划

阶段 16 的目标是改进 Stage 13 的 frame-wise selection objective。新的 selection 不再直接选择高分 frame，而是对当前 selected-keyframe segmentation 做 greedy split，每次选择能最大降低估计 segment difficulty 的 split point。该阶段覆盖当前 4 个样本，为后续更多数据的 RD 曲线对比准备统一 selection CSV。

### 新增文件

```text
scripts/run_stage16_segment_error_keyframe_selection.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage16_segment_error_keyframe_selection.py
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。该阶段为 CPU selection/statistics 脚本，不占用 GPU。

### 输出文件

```text
experiments/stage16_segment_error_keyframe_selection/stage16_segment_error_keyframe_selection_summary.csv
experiments/stage16_segment_error_keyframe_selection/stage16_segment_error_keyframe_selection_summary.json
```

### 覆盖范围

| item | value |
|---|---:|
| samples | n3dv, meetroom, driving, robot |
| reference gaps | 4, 8, 16 |
| methods | uniform, segment_motion, segment_gaussian, segment_rd |
| output rows | 48 |
| max_segment_multiplier | 2 |
| motion_weight for segment_rd | 0.7 |

### 示例 selection stats

| sample | method | gap | keyframes | max segment | rate MiB/frame |
|---|---|---:|---:|---:|---:|
| n3dv | segment_rd | 4 | 21 | 6 | 0.11848958333333333 |
| n3dv | segment_rd | 8 | 11 | 12 | 0.062065972222222224 |
| robot | segment_rd | 4 | 20 | 8 | 0.11870941558441558 |
| robot | segment_rd | 8 | 11 | 11 | 0.06529017857142858 |
| driving | segment_rd | 4 | 21 | 6 | 0.11848958333333333 |
| meetroom | segment_rd | 4 | 21 | 6 | 0.11848958333333333 |

### 结论

- Stage 16 已生成 segment-error-aware keyframe selection baseline，覆盖当前 4 个样本。
- 相比 Stage 13 的 frame-wise top-k / spaced fill，Stage 16 更直接优化 segment difficulty，选帧分布更像“切分难 segment”。
- 下一步 Stage 17 应复用 selected-keyframe evaluator，对 `uniform` 与 `segment_rd` 进行 RD reconstruction 对比，并优先扩展到更多样本。

## 2026-06-25：阶段 17 Four-Sample Segment-Error RD Curve

### 执行计划

阶段 17 的目标是响应“最终 RD 曲线对比要包括更多数据”的要求，将 Stage 16 的 `segment_rd` selection 接入 selected-keyframe evaluator，并覆盖当前 4 个样本 `n3dv/meetroom/driving/robot`。评估范围为 `4 samples × 2 methods × 3 gaps = 24` 组。

### 新增/修改文件

```text
scripts/run_stage17_segment_error_rd_curve.py
scripts/run_stage12_selected_keyframe_reconstruction.py
```

`run_stage12_selected_keyframe_reconstruction.py` 扩展支持 `segment_motion` / `segment_gaussian` / `segment_rd` 方法。

### 运行命令

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage17_segment_error_rd_curve.py --batch_size 1 --max_segment_length 40
```

首次运行完成 24 组重建后，绘图阶段发现字段读取错误；修复后复用已生成的单组 JSON 重新汇总/绘图，未重复跑 24 组推理。

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 被占用；GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage17_segment_error_rd_curve/stage17_segment_error_rd_curve_summary.csv
experiments/stage17_segment_error_rd_curve/stage17_segment_error_rd_curve_summary.json
experiments/stage17_segment_error_rd_curve/stage17_rd_curve_all_psnr.png
experiments/stage17_segment_error_rd_curve/stage17_rd_curve_middle_psnr.png
experiments/stage17_segment_error_rd_curve/*_summary.json
```

### RD Summary

| sample | method | gap | rate MiB/frame | all PSNR | middle PSNR | max segment |
|---|---|---:|---:|---:|---:|---:|
| n3dv | uniform | 4 | 0.11848958333333333 | 33.39609679886218 | 33.05215015387749 | 4 |
| n3dv | segment_rd | 4 | 0.11848958333333333 | 33.428578143327144 | 33.084130548722 | 6 |
| n3dv | uniform | 16 | 0.033854166666666664 | 30.881761283241584 | 30.60315188772239 | 16 |
| n3dv | segment_rd | 16 | 0.033854166666666664 | 31.401522780566143 | 31.160727202005425 | 20 |
| meetroom | uniform | 4 | 0.11848958333333333 | 32.426306092422514 | 31.164020411576498 | 4 |
| meetroom | segment_rd | 4 | 0.11848958333333333 | 32.23470027048153 | 30.87014915711624 | 6 |
| driving | uniform | 4 | 0.11848958333333333 | 30.194992473308346 | 27.836169561169633 | 4 |
| driving | segment_rd | 4 | 0.11848958333333333 | 30.455521453965307 | 28.160586286253864 | 6 |
| robot | uniform | 4 | 0.11870941558441558 | 26.48952399862992 | 23.178617517791157 | 4 |
| robot | segment_rd | 4 | 0.11870941558441558 | 26.282488669614075 | 22.881804558757004 | 8 |

### 结论

- Stage 17 已完成 4 个当前样本的 expanded RD curve，对比范围比 Stage 15 更完整。
- `segment_rd` 在 n3dv 的 gap4/gap16、driving 的 gap4 上超过 uniform，说明 segment-level objective 比 Stage 15 的 `rd_spaced` 更有潜力。
- `segment_rd` 在 meetroom/robot 和部分低码率点仍低于 uniform，说明 selection objective 需要数据类型自适应，不能直接声明优于 uniform。
- 当前结果仍是 StreamSplat RGB/depth-conditioned selected-pair inference，不是最终 Gaussian-anchor-only decoder。

## 2026-06-25：阶段 18 Decoder Freeze Policy Report

### 执行计划

阶段 18 的目标是在开始 long-GOP Dynamic Decoder 微调前，明确 StreamSplat 模型的模块参数切分和推荐 freeze policy。该阶段只做参数审计，不训练。

### 新增文件

```text
scripts/run_stage18_decoder_freeze_policy_report.py
```

### 运行命令

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage18_decoder_freeze_policy_report.py --device cuda
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 和 7 有进程；GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage18_decoder_freeze_policy/stage18_decoder_freeze_policy_summary.json
experiments/stage18_decoder_freeze_policy/stage18_module_parameter_summary.csv
experiments/stage18_decoder_freeze_policy/stage18_freeze_policy_parameters.csv
```

### 参数统计

| module | parameters | percent |
|---|---:|---:|
| TOTAL | 268272782 | 100.0 |
| model.gs_predictor.encoder | 71023104 | 26.474211610479365 |
| model.gs_predictor.gaussian_upsampler | 5907648 | 2.2021048710040216 |
| model.gs_predictor.predictor | 1737 | 0.0006474752999728464 |
| model.condition_encoder | 86582784 | 32.274158919334575 |
| model.decoder | 98588928 | 36.74950819274689 |
| model.gaussian_upsampler | 5907648 | 2.2021048710040216 |
| model.gs_dynamic_predictor | 224069 | 0.0835228226768081 |
| model.encoder_proj | 36864 | 0.013741237454346003 |

### 推荐 freeze policy

首轮 long-GOP decoder adaptation 推荐训练：

```text
model.decoder
model.gs_dynamic_predictor
model.encoder_proj
```

首轮推荐冻结：

```text
model.gs_predictor
model.condition_encoder
model.gaussian_upsampler
```

按该策略：

| item | value |
|---|---:|
| train parameter tensors | 150 |
| freeze parameter tensors | 340 |
| train parameters | 98849861 |
| freeze parameters | 169422921 |
| train parameter percent | 36.84677225287804 |
| missing checkpoint keys | 0 |
| unexpected checkpoint keys | 0 |

### 结论

- Stage 18 已明确后续微调不应从头训练，而应从 StreamSplat checkpoint 初始化。
- 首轮 Stage 20 long-GOP 微调应冻结 static feature/Gaussian extraction，只训练 temporal/dynamic decoder path。
- 下一步 Stage 19 应整理 original decoder variable-GOP baseline，作为微调前对照。

## 2026-06-25：阶段 19 Original Decoder Variable-GOP Baseline

### 执行计划

阶段 19 的目标是建立 long-GOP 微调前的原始 StreamSplat decoder baseline。该阶段不重新推理，而是复用：

- Stage 1 pretrained StreamSplat fair metrics，作为原始 decoder 的 RGB/depth-conditioned reconstruction quality。
- Stage 2 q8 static-anchor payload estimate，作为 transmitted keyframe Gaussian rate 口径。

### 新增文件

```text
scripts/run_stage19_original_decoder_variable_gop_baseline.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage19_original_decoder_variable_gop_baseline.py
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。该阶段只读取已有 CSV/JSON 并绘图，没有新 GPU 推理。

### 输出文件

```text
experiments/stage19_original_decoder_variable_gop_baseline/stage19_original_decoder_variable_gop_baseline_summary.json
experiments/stage19_original_decoder_variable_gop_baseline/stage19_original_decoder_variable_gop_baseline.csv
experiments/stage19_original_decoder_variable_gop_baseline/stage19_original_decoder_variable_gop_gap_averages.csv
experiments/stage19_original_decoder_variable_gop_baseline/stage19_original_decoder_all_psnr.png
experiments/stage19_original_decoder_variable_gop_baseline/stage19_original_decoder_middle_psnr.png
experiments/stage19_original_decoder_variable_gop_baseline/stage19_original_decoder_given_psnr.png
```

### 覆盖范围

| item | value |
|---|---:|
| samples | n3dv, meetroom, driving, robot |
| gaps | 2, 4, 8, 16 |
| rows | 16 |
| rate profile | static_anchor |
| rate codec | q8 |
| opacity threshold | 0.0 |

### Gap 平均结果

| gap | q8 static MiB/frame | raw pred_gs MiB/frame | all PSNR | middle PSNR | given PSNR |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.23137344426406925 | 3.0550595238095237 | 32.953542199364946 | 29.996391443587946 | 35.83678978896073 |
| 4 | 0.1185445413961039 | 1.5275297619047619 | 30.626730087064544 | 28.807739474456714 | 35.81793379119585 |
| 8 | 0.06287202380952381 | 0.7738095238095237 | 28.484245511008048 | 27.302025920513756 | 35.82481680540889 |
| 16 | 0.03429383116883117 | 0.38690476190476186 | 26.15445374823089 | 25.367519709128025 | 35.79403535099706 |

### 结论

- Stage 19 已给出 Stage 20 微调前的原始 decoder baseline，后续 fine-tune 结果应优先和该表对齐比较。
- `given_keyframes` PSNR 基本稳定在 35.8 dB 左右，质量下降主要来自 non-keyframe middle reconstruction。
- `raw_pred_gs_mib_per_frame` 是 decoder 输出诊断量，不是 transmitted bitstream；论文主 rate 应使用 `estimated_q8_static_mib_per_frame` 或后续真实 entropy-coded anchor payload。

## 2026-06-25：阶段 20 Long-GOP Dynamic Decoder Fine-Tune Smoke

### 执行计划

阶段 20 的目标是验证“从 StreamSplat checkpoint 初始化，冻结 static path，只微调 temporal/dynamic decoder path”的训练链路是否可运行。该阶段是 smoke，不追求 RD 提升。

### 新增文件

```text
scripts/run_stage20_long_gop_decoder_finetune_smoke.py
```

### 运行命令

先运行了 1-step 最小 smoke，确认 backward/renderer 链路可用；随后运行正式 smoke：

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage20_long_gop_decoder_finetune_smoke.py --samples n3dv robot --train_gaps 2 4 8 12 16 --eval_gaps 4 8 16 --steps 3 --lr 1e-6 --max_eval_pairs 6
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 有进程，GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage20_long_gop_decoder_finetune_smoke/stage20_long_gop_decoder_finetune_smoke_summary.json
experiments/stage20_long_gop_decoder_finetune_smoke/stage20_train_losses.csv
```

外部 checkpoint：

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage20_long_gop_decoder_finetune_smoke/stage20_trainable_state.safetensors
```

外部 checkpoint 大小约 378M，只保存 trainable tensors，不提交到 git。

### Freeze policy

训练参数前缀：

```text
model.decoder
model.gs_dynamic_predictor
model.encoder_proj
```

冻结参数前缀：

```text
model.gs_predictor
model.condition_encoder
model.gaussian_upsampler
```

| item | value |
|---|---:|
| train parameter tensors | 150 |
| freeze parameter tensors | 340 |
| train parameters | 98849861 |
| freeze parameters | 169422921 |
| train parameter percent | 36.84677225287804 |
| missing checkpoint keys | 0 |
| unexpected checkpoint keys | 0 |

### 训练记录

| step | sample | gap | segment | loss | PSNR |
|---:|---|---:|---:|---:|---:|
| 1 | n3dv | 8 | 8 | 0.0020479541271924973 | 26.886797754941995 |
| 2 | robot | 2 | 2 | 0.0020044066477566957 | 26.980141655587456 |
| 3 | robot | 2 | 2 | 0.005418718792498112 | 22.661033863626244 |

### Eval 结果

| metric | initial | final | delta |
|---|---:|---:|---:|
| all PSNR avg | 26.427965599243006 | 26.414150103438597 | -0.013815495804408803 |
| middle PSNR avg | 25.639764120943642 | 25.6235457820023 | -0.01621833894134192 |
| given PSNR avg | 35.080338836905376 | 35.078974144427214 | -0.0013646924781619812 |

### 结论

- Stage 20 证明 pretrained checkpoint loading、freeze policy、variable-GOP forward/backward、RGB loss、trainable-state 保存链路可运行。
- 3-step smoke 没有带来质量提升，且略微降低 eval PSNR；这符合 smoke 预期，不能作为有效 fine-tune 结论。
- 后续 Stage 21/22 前应把训练扩展为更稳定的多-step/multi-sample schedule，并考虑 middle-frame weighted loss 或保留更多 gap16 样本。

## 2026-06-25：阶段 21 Gaussian-Anchor-Only Adapter RGB Smoke

### 执行计划

阶段 21 的目标是把 Stage 10b 的单 pair renderer differentiability smoke 扩展成最小 Gaussian-anchor-only adapter smoke。输入只包含 q8 keyframe static anchors 和 normalized timestamp；不输入 non-keyframe RGB、non-keyframe Gaussian、motion field、deformation payload 或 residual payload。

### 新增文件

```text
scripts/run_stage21_gaussian_anchor_only_adapter_smoke.py
```

### 运行命令

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage21_gaussian_anchor_only_adapter_smoke.py
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 有进程，GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage21_gaussian_anchor_only_adapter_smoke/stage21_gaussian_anchor_only_adapter_smoke_summary.json
experiments/stage21_gaussian_anchor_only_adapter_smoke/stage21_train_rgb_losses.csv
```

外部 adapter checkpoint：

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage21_gaussian_anchor_only_adapter_smoke/stage21_anchor_adapter.safetensors
```

外部 checkpoint 大小约 404K，不提交到 git。

### 配置

| item | value |
|---|---:|
| train samples | n3dv |
| eval samples | robot |
| frame_gap | 4 |
| train tasks | 2 |
| eval tasks | 2 |
| steps | 4 |
| hidden_dim | 128 |
| lr | 5e-05 |
| quant_bits | 8 |
| parameter_count | 102925 |

### 训练记录

| step | sample | target frame | RGB MSE | PSNR |
|---:|---|---:|---:|---:|
| 1 | n3dv | 2 | 0.002875811653211713 | 25.41239560795898 |
| 2 | n3dv | 6 | 0.0025512068532407284 | 25.932543271295398 |
| 3 | n3dv | 2 | 0.0027694874443113804 | 25.576005994209424 |
| 4 | n3dv | 6 | 0.0024442733265459538 | 26.118502315008936 |

### Eval 结果

| metric | initial | final | delta |
|---|---:|---:|---:|
| train model PSNR avg | 25.625938109386198 | 25.97498316781515 | 0.3490450584289518 |
| eval model PSNR avg | 20.602936330073554 | 20.65887117994607 | 0.05593484987251577 |
| eval linear PSNR avg | 21.07847795611209 | 21.07847795611209 | 0.0 |

### 结论

- Stage 21 证明 q8 keyframe anchors + timestamp 的 Gaussian-anchor-only RGB adapter 链路可训练并可保存 adapter checkpoint。
- 4-step smoke 在 n3dv train tasks 和 robot eval tasks 上均有小幅提升。
- 当前 MLP adapter 仍低于 linear anchor rendering baseline，说明需要更强初始化、更多训练、residual-zero init 或更贴近 StreamSplat dynamic Gaussian 格式的 adapter。

## 2026-06-25：阶段 21b Residual-Zero Anchor Adapter Training

### 执行计划

阶段 21b 的目标是修复 Stage 21 的主要问题：随机 residual 初始化导致 adapter 初始输出低于 linear anchor baseline。本阶段新增 residual-zero 初始化，使 adapter 在训练前严格等于 q8 linear anchor interpolation，然后用 RGB renderer loss 做更长的小规模训练。

### 修改/新增文件

```text
mono_dfcgs/anchor_predictor.py
scripts/run_stage21b_residual_zero_anchor_adapter_training.py
```

### 运行命令

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage21b_residual_zero_anchor_adapter_training.py
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 有进程，GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage21b_residual_zero_anchor_adapter_training/stage21b_residual_zero_anchor_adapter_training_summary.json
experiments/stage21b_residual_zero_anchor_adapter_training/stage21b_train_rgb_losses.csv
experiments/stage21b_residual_zero_anchor_adapter_training/stage21b_initial_eval.csv
experiments/stage21b_residual_zero_anchor_adapter_training/stage21b_final_eval.csv
```

外部 adapter checkpoint：

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage21b_residual_zero_anchor_adapter_training/stage21b_residual_zero_anchor_adapter.safetensors
```

外部 checkpoint 大小约 404K，不提交到 git。

### 配置

| item | value |
|---|---:|
| samples | n3dv, meetroom, driving, robot |
| eval samples | robot |
| selected train samples | driving, meetroom, n3dv |
| selected eval samples | robot |
| frame_gap | 4 |
| selected train rows | 8 |
| selected eval rows | 4 |
| targets per row | 3 |
| train tasks | 24 |
| eval tasks | 12 |
| steps | 24 |
| hidden_dim | 128 |
| lr | 1e-05 |
| quant_bits | 8 |
| parameter_count | 102925 |

### 结果

| metric | initial | final | delta |
|---|---:|---:|---:|
| train model PSNR avg | 27.96271836837856 | 28.001963714263283 | 0.03924534588472239 |
| train linear PSNR avg | 27.96271836837856 | 27.96271836837856 | 0.0 |
| eval model PSNR avg | 21.300156836809595 | 21.301945998526975 | 0.0017891617173795282 |
| eval linear PSNR avg | 21.300156836809595 | 21.300156836809595 | 0.0 |
| eval margin over linear | 0.0 | 0.0017891617173795282 | 0.0017891617173795282 |

### 结论

- residual-zero 初始化有效：初始 model PSNR 与 q8 linear anchor baseline 完全一致。
- 24-step 小规模训练后，train 集提升约 0.039 dB，robot eval 比 linear baseline 高约 0.0018 dB。
- 这是方向正确但收益极小的结果，还不足以作为最终 RD curve 的 fine-tuned adapter。下一步需要扩大 steps、balanced rows、gap 覆盖，并加入更强 regularization/learning-rate schedule。

## 2026-06-25：阶段 21c Medium Anchor Adapter Training

### 执行计划

阶段 21c 在 Stage21b 的 residual-zero 初始化基础上扩大训练规模：覆盖 GOP gap `2/4/8/16`，每个 gap 做 sample-balanced row selection，并增加到 96 steps。该阶段仍是 anchor-only：输入只包含 q8 keyframe anchors 和 timestamp，不输入 non-keyframe payload。

### 新增文件

```text
scripts/run_stage21c_medium_anchor_adapter_training.py
```

### 运行命令

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage21c_medium_anchor_adapter_training.py
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 有进程，GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage21c_medium_anchor_adapter_training/stage21c_medium_anchor_adapter_training_summary.json
experiments/stage21c_medium_anchor_adapter_training/stage21c_train_rgb_losses.csv
experiments/stage21c_medium_anchor_adapter_training/stage21c_initial_eval.csv
experiments/stage21c_medium_anchor_adapter_training/stage21c_final_eval.csv
experiments/stage21c_medium_anchor_adapter_training/stage21c_gap_eval_summary.csv
```

外部 adapter checkpoint：

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage21c_medium_anchor_adapter_training/stage21c_medium_anchor_adapter.safetensors
```

外部 checkpoint 大小约 404K，不提交到 git。

### 配置

| item | value |
|---|---:|
| frame gaps | 2, 4, 8, 16 |
| train samples | n3dv, meetroom, driving |
| eval samples | robot |
| selected train rows | 24 |
| selected eval rows | 12 |
| train tasks | 42 |
| eval tasks | 21 |
| steps | 96 |
| hidden_dim | 128 |
| lr | 1e-05 |
| quant_bits | 8 |
| parameter_count | 102925 |

### 总体结果

| metric | initial | final | delta |
|---|---:|---:|---:|
| train model PSNR avg | 27.961215027303925 | 28.06389696855266 | 0.10268194124873324 |
| train linear PSNR avg | 27.961215027303925 | 27.961215027303925 | 0.0 |
| eval model PSNR avg | 21.88713238848381 | 21.903100985613587 | 0.015968597129777606 |
| eval linear PSNR avg | 21.88713238848381 | 21.88713238848381 | 0.0 |
| eval margin over linear | 0.0 | 0.015968597129777606 | 0.015968597129777606 |

### Gap-wise Eval

| gap | initial model PSNR | final model PSNR | linear PSNR | delta | margin over linear |
|---:|---:|---:|---:|---:|---:|
| 2 | 22.898814876944613 | 22.916134779744898 | 22.898814876944613 | 0.017319902800284837 | 0.017319902800284837 |
| 4 | 21.02887941205421 | 21.02850200761415 | 21.02887941205421 | -0.0003774044400586263 | -0.0003774044400586263 |
| 8 | 21.673428963334427 | 21.69548755972806 | 21.673428963334427 | 0.02205859639363439 | 0.02205859639363439 |
| 16 | 22.453247545832397 | 22.478796492432902 | 22.453247545832397 | 0.025548946600505218 | 0.025548946600505218 |

### 结论

- Stage21c 相比 Stage21b 更稳定：整体 robot eval 比 q8 linear anchor baseline 高约 0.016 dB。
- gap2/gap8/gap16 为正收益，gap4 略降 0.0004 dB，说明训练方向可行但还没有足够稳定。
- 这仍不是 paper-level 大规模实验；下一步应继续扩大 rows/steps 或引入 validation-based checkpoint selection，再考虑 Stage22 RD curve。

## 2026-06-25：阶段 21d Validated Anchor Adapter Training

### 执行计划

阶段 21d 在 Stage21c 的基础上加入 validation-based checkpoint selection。训练覆盖 GOP gap `2/4/8/16`，每个 gap 使用更多 sample-balanced rows，训练 384 steps，每 96 steps 在 robot eval tasks 上评估并保存 best checkpoint。

### 新增文件

```text
scripts/run_stage21d_validated_anchor_adapter_training.py
```

### 运行命令

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage21d_validated_anchor_adapter_training.py
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 有进程，GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage21d_validated_anchor_adapter_training/stage21d_validated_anchor_adapter_training_summary.json
experiments/stage21d_validated_anchor_adapter_training/stage21d_train_rgb_losses.csv
experiments/stage21d_validated_anchor_adapter_training/stage21d_validation_log.csv
experiments/stage21d_validated_anchor_adapter_training/stage21d_initial_eval.csv
experiments/stage21d_validated_anchor_adapter_training/stage21d_final_eval.csv
experiments/stage21d_validated_anchor_adapter_training/stage21d_best_eval.csv
experiments/stage21d_validated_anchor_adapter_training/stage21d_best_gap_eval_summary.csv
```

外部 checkpoint：

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage21d_validated_anchor_adapter_training/stage21d_best_anchor_adapter.safetensors
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage21d_validated_anchor_adapter_training/stage21d_final_anchor_adapter.safetensors
```

每个 checkpoint 约 404K，不提交到 git。

### 配置

| item | value |
|---|---:|
| frame gaps | 2, 4, 8, 16 |
| train samples | n3dv, meetroom, driving |
| eval samples | robot |
| selected train rows | 48 |
| selected eval rows | 20 |
| train tasks | 84 |
| eval tasks | 35 |
| steps | 384 |
| eval interval | 96 |
| hidden_dim | 128 |
| lr | 8e-06 |
| quant_bits | 8 |
| parameter_count | 102925 |

### Validation Log

| step | model PSNR | linear PSNR | margin |
|---:|---:|---:|---:|
| 0 | 22.359687957915238 | 22.359687957915238 | 0.0 |
| 96 | 22.378299586064394 | 22.359687957915238 | 0.018611628149155734 |
| 192 | 22.387509907452376 | 22.359687957915238 | 0.027821949537138124 |
| 288 | 22.392827616765334 | 22.359687957915238 | 0.03313965885009651 |
| 384 | 22.396716910658334 | 22.359687957915238 | 0.037028952743096255 |

Best step: 384.

### Gap-wise Best Eval

| gap | initial/linear PSNR | best adapter PSNR | margin |
|---:|---:|---:|---:|
| 2 | 22.645976203464308 | 22.67349864905237 | 0.027522445588061828 |
| 4 | 22.0418426521134 | 22.065700074084212 | 0.02385742197081342 |
| 8 | 21.2065569120507 | 21.230463662292394 | 0.023906750241692976 |
| 16 | 23.68752018680706 | 23.755596126401358 | 0.06807593959429781 |

### 结论

- Stage21d 解决了 Stage21c 的 gap4 略负问题，robot eval 上所有 gap 均超过 q8 linear anchor baseline。
- 平均 margin over linear 达到 0.0370 dB，仍是小幅提升，但比 Stage21b/21c 更稳定。
- 该结果仍是当前四样本开发集和 robot eval tasks，不是最终大规模实验。

## 2026-06-25：阶段 22 Anchor-Only RD Curve

### 执行计划

阶段 22 将 Stage21d best checkpoint 的 robot intermediate-target quality 与 Stage2 q8 static-anchor transmitted rate 对齐，生成 anchor-only RD CSV 和曲线图。该阶段只做 CPU 汇总和绘图，不重新 GPU 推理。

### 新增文件

```text
scripts/run_stage22_anchor_only_rd_curve.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage22_anchor_only_rd_curve.py
```

### 输出文件

```text
experiments/stage22_anchor_only_rd_curve/stage22_anchor_only_rd_curve_summary.json
experiments/stage22_anchor_only_rd_curve/stage22_anchor_only_rd_curve.csv
experiments/stage22_anchor_only_rd_curve/stage22_anchor_only_rd_curve.png
experiments/stage22_anchor_only_rd_curve/stage22_anchor_only_delta_psnr.png
```

### RD 结果

| gap | q8 static MiB/frame | linear PSNR | adapter PSNR | delta |
|---:|---:|---:|---:|---:|
| 16 | 0.03561282467532467 | 23.68752018680706 | 23.755596126401358 | 0.06807593959429781 |
| 8 | 0.06529017857142858 | 21.2065569120507 | 21.230463662292394 | 0.023906750241692976 |
| 4 | 0.11870941558441558 | 22.0418426521134 | 22.065700074084212 | 0.02385742197081342 |
| 2 | 0.2314833603896104 | 22.645976203464308 | 22.67349864905237 | 0.027522445588061828 |

### 结论

- Stage22 anchor-only RD 中，Stage21d adapter 在 4 个 robot rate 点上都超过 q8 linear anchor baseline。
- mean delta PSNR: 0.03584063934871651 dB。
- 该 RD 是 intermediate-target anchor-only RD，不是 full-video all-frame PSNR/SSIM；后续 paper-level RD 还需要完整视频评估和更多数据集。

## 2026-06-25：阶段 23 Full-Video Anchor-Only Evaluator

### 执行计划

阶段 23 的目标是把 Stage22 的 robot intermediate-target RD 扩展成 full-video anchor-only evaluator。使用 Stage21d best adapter checkpoint，对 `n3dv/meetroom/driving/robot × gap 2/4/8/16` 做逐帧重建评估。

关键口径：

- 输入 payload 仍仅为 q8 keyframe static anchors + timestamp。
- given keyframes 直接由 transmitted q8 anchors 渲染。
- adapter 只用于 non-keyframe middle frames，不改写已传输 keyframes。
- 对比方法为 q8 linear anchor interpolation vs Stage21d adapter。

### 新增文件

```text
scripts/run_stage23_full_video_anchor_only_evaluator.py
```

### 运行命令

先做 smoke：

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage23_full_video_anchor_only_evaluator.py --samples robot --gaps 4 --summary_root experiments/stage23_full_video_anchor_only_evaluator_smoke
```

再跑全量：

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage23_full_video_anchor_only_evaluator.py
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 0/2 有大进程，GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage23_full_video_anchor_only_evaluator/stage23_full_video_anchor_only_evaluator_summary.json
experiments/stage23_full_video_anchor_only_evaluator/stage23_full_video_anchor_only_evaluator.csv
```

smoke 输出：

```text
experiments/stage23_full_video_anchor_only_evaluator_smoke/stage23_full_video_anchor_only_evaluator_summary.json
experiments/stage23_full_video_anchor_only_evaluator_smoke/stage23_full_video_anchor_only_evaluator.csv
```

### 总体结果

| metric | value |
|---|---:|
| rows | 16 |
| mean delta all PSNR | 0.08745696841177764 |
| mean delta middle PSNR | 0.12360522208180025 |
| negative all-PSNR points | 0 |
| negative middle-PSNR points | 0 |

### Per-Sample 平均收益

| sample | mean delta all PSNR | mean delta middle PSNR |
|---|---:|---:|
| n3dv | 0.1592318809129889 | 0.21560375144838417 |
| meetroom | 0.08321870692386568 | 0.11864301430132773 |
| driving | 0.08352391673319559 | 0.12321783919904483 |
| robot | 0.02385336907706037 | 0.036956283378444255 |

### 代表性结果

| sample | gap | rate MiB/frame | linear all PSNR | adapter all PSNR | delta all | delta middle |
|---|---:|---:|---:|---:|---:|---:|
| n3dv | 4 | 0.11848958333333333 | 30.262746842193057 | 30.433338876385193 | 0.1705920341921363 | 0.2302992461593938 |
| meetroom | 4 | 0.11848958333333333 | 29.545271806231266 | 29.645102902246254 | 0.09983109601498796 | 0.13477197962024334 |
| driving | 4 | 0.11848958333333333 | 27.918186053702325 | 28.023854321394463 | 0.10566826769213833 | 0.1426521613843903 |
| robot | 4 | 0.11870941558441558 | 25.154367560029165 | 25.19099172008088 | 0.03662416005171565 | 0.049474742525994486 |

### 结论

- Stage23 表明 Stage21d adapter 的收益能从 intermediate-target eval 转化到 full-video anchor-only evaluation。
- 16 个 sample-gap 点的 all/middle PSNR 均为正收益。
- given-keyframe delta 为 0，这是预期行为，因为 keyframes 直接使用 transmitted q8 anchors 渲染。
- 注意：Stage23 的 full-video anchor-only PSNR 不等同于 Stage19 原 StreamSplat RGB/depth-conditioned full-video PSNR；二者输入 payload 和 decoder 条件不同。

## 2026-06-25：阶段 24 Full-Video Anchor-Only RD Plots

### 执行计划

阶段 24 将 Stage23 的 full-video anchor-only CSV 转换成 RD 曲线图和 per-sample aggregate 表。该阶段只读 Stage23 CSV，不重新渲染或训练。

### 新增文件

```text
scripts/run_stage24_full_video_anchor_only_rd_plots.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage24_full_video_anchor_only_rd_plots.py
```

### 输出文件

```text
experiments/stage24_full_video_anchor_only_rd_plots/stage24_full_video_anchor_only_rd_plots_summary.json
experiments/stage24_full_video_anchor_only_rd_plots/stage24_full_video_anchor_only_rd_aggregate.csv
experiments/stage24_full_video_anchor_only_rd_plots/stage24_full_video_all_psnr_rd.png
experiments/stage24_full_video_anchor_only_rd_plots/stage24_full_video_middle_psnr_rd.png
experiments/stage24_full_video_anchor_only_rd_plots/stage24_full_video_all_ssim_rd.png
experiments/stage24_full_video_anchor_only_rd_plots/stage24_full_video_middle_ssim_rd.png
experiments/stage24_full_video_anchor_only_rd_plots/stage24_full_video_delta_all_psnr.png
experiments/stage24_full_video_anchor_only_rd_plots/stage24_full_video_delta_middle_psnr.png
```

### Aggregate 结果

| sample | mean delta all PSNR | mean delta middle PSNR | min delta all PSNR | min delta middle PSNR |
|---|---:|---:|---:|---:|
| driving | 0.08352391673319559 | 0.12321783919904483 | 0.0465389497943427 | 0.05026206577788983 |
| meetroom | 0.08321870692386568 | 0.11864301430132773 | 0.060858978680400355 | 0.06572769697483949 |
| n3dv | 0.1592318809129889 | 0.21560375144838417 | 0.12101240598846985 | 0.17531502713124425 |
| robot | 0.02385336907706037 | 0.036956283378444255 | 0.0032667066510931875 | 0.003542766368092032 |

### 总结

- Stage24 生成了 full-video anchor-only RD 图和 delta 图。
- 16 个点均保持正收益；每个样本的 min delta all/middle PSNR 也为正。
- 该图可作为当前开发集上的 anchor-only full-video RD 参考，但仍不是 paper-level 多数据集结果。

## 2026-06-25：阶段 25 Leave-One-Sample-Out Adapter Training

### 执行计划

阶段 25 的目标是验证 Stage21d adapter 的跨样本泛化。轮流 hold out `n3dv/meetroom/driving/robot`，每折用其余 3 个样本训练，在 held-out 样本上进行 validation-based checkpoint selection。

### 新增文件

```text
scripts/run_stage25_leave_one_out_adapter_training.py
```

### 运行命令

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage25_leave_one_out_adapter_training.py
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 有进程，GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage25_leave_one_out_adapter_training/stage25_leave_one_out_adapter_training_summary.json
experiments/stage25_leave_one_out_adapter_training/stage25_leave_one_out_adapter_training_aggregate.csv
experiments/stage25_leave_one_out_adapter_training/<heldout>/stage25_fold_summary.json
experiments/stage25_leave_one_out_adapter_training/<heldout>/stage25_validation_log.csv
experiments/stage25_leave_one_out_adapter_training/<heldout>/stage25_best_gap_eval_summary.csv
```

外部 checkpoint：

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage25_leave_one_out_adapter_training/<heldout>/stage25_best_anchor_adapter.safetensors
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage25_leave_one_out_adapter_training/<heldout>/stage25_final_anchor_adapter.safetensors
```

每个 checkpoint 约 404K，不提交到 git。

### 配置

| item | value |
|---|---:|
| folds | n3dv, meetroom, driving, robot |
| frame gaps | 2, 4, 8, 16 |
| train samples per fold | 3 |
| selected train rows per fold | 48 |
| selected eval rows per fold | 20 |
| train tasks per fold | 84 |
| eval tasks per fold | 35 |
| steps per fold | 384 |
| eval interval | 96 |
| hidden_dim | 128 |
| lr | 8e-06 |
| quant_bits | 8 |

### Aggregate 结果

| heldout sample | best step | best margin over linear PSNR |
|---|---:|---:|
| n3dv | 384 | 0.22715940587644212 |
| meetroom | 384 | 0.15021969118397394 |
| driving | 384 | 0.06700045546002897 |
| robot | 384 | 0.03766598947441224 |

| metric | value |
|---|---:|
| mean best margin over linear PSNR | 0.12051138549871432 |
| min best margin over linear PSNR | 0.03766598947441224 |
| negative heldout-gap points | 0 |

### Gap-wise 检查

- n3dv: gap2 +0.2375, gap4 +0.2301, gap8 +0.2345, gap16 +0.2117。
- meetroom: gap2 +0.1820, gap4 +0.1595, gap8 +0.1223, gap16 +0.1530。
- driving: gap2 +0.0586, gap4 +0.0405, gap8 +0.0669, gap16 +0.0978。
- robot: gap2 +0.0281, gap4 +0.0243, gap8 +0.0245, gap16 +0.0689。

### 结论

- Stage25 leave-one-out intermediate validation 全部 held-out sample 和全部 gap 都超过 q8 linear anchor baseline。
- 最弱泛化点仍是 robot，但 best margin 仍为正，约 +0.0377 dB。
- 下一步应做 Stage26：用每折 best checkpoint 做 held-out full-video anchor-only evaluation 和 RD plots。

## 2026-06-25：阶段 26 Leave-One-Out Full-Video Anchor-Only RD

### 执行计划

Stage26 使用 Stage25 每个 held-out fold 的 best checkpoint，对相同 held-out sample 执行 full-video anchor-only RD 评估。该阶段比 Stage23/24 更严格：Stage23/24 使用开发集统一 adapter checkpoint，而 Stage26 每个样本只使用未在该样本上训练的 adapter。

### 新增文件

```text
scripts/run_stage26_leave_one_out_full_video_rd.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage26_leave_one_out_full_video_rd.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage26_leave_one_out_full_video_rd.py
```

### GPU 检查

运行前使用 `nvidia-smi` 检查 GPU。GPU 2 仍有进程，GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输入 checkpoint

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage25_leave_one_out_adapter_training/n3dv/stage25_best_anchor_adapter.safetensors
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage25_leave_one_out_adapter_training/meetroom/stage25_best_anchor_adapter.safetensors
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage25_leave_one_out_adapter_training/driving/stage25_best_anchor_adapter.safetensors
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage25_leave_one_out_adapter_training/robot/stage25_best_anchor_adapter.safetensors
```

### 输出文件

```text
experiments/stage26_leave_one_out_full_video_rd/stage26_leave_one_out_full_video_rd.csv
experiments/stage26_leave_one_out_full_video_rd/stage26_leave_one_out_full_video_rd_summary.json
experiments/stage26_leave_one_out_full_video_rd/stage26_sample_aggregate.csv
experiments/stage26_leave_one_out_full_video_rd/stage26_gap_aggregate.csv
experiments/stage26_leave_one_out_full_video_rd/stage26_per_sample_all_psnr_rd.png
experiments/stage26_leave_one_out_full_video_rd/stage26_per_sample_middle_psnr_rd.png
experiments/stage26_leave_one_out_full_video_rd/stage26_per_sample_all_ssim_rd.png
experiments/stage26_leave_one_out_full_video_rd/stage26_per_sample_middle_ssim_rd.png
experiments/stage26_leave_one_out_full_video_rd/stage26_mean_all_psnr_rd.png
experiments/stage26_leave_one_out_full_video_rd/stage26_mean_middle_psnr_rd.png
experiments/stage26_leave_one_out_full_video_rd/stage26_mean_all_ssim_rd.png
experiments/stage26_leave_one_out_full_video_rd/stage26_mean_middle_ssim_rd.png
experiments/stage26_leave_one_out_full_video_rd/stage26_delta_all_psnr.png
experiments/stage26_leave_one_out_full_video_rd/stage26_delta_middle_psnr.png
```

### Aggregate 结果

| metric | value |
|---|---:|
| rows | 16 |
| mean delta all PSNR | 0.07979649123330468 |
| mean delta middle PSNR | 0.1125053472768538 |
| negative all PSNR points | 0 |
| negative middle PSNR points | 0 |

### Per-sample full-video gains

| sample | mean delta all PSNR | mean delta middle PSNR | min delta all PSNR | min delta middle PSNR |
|---|---:|---:|---:|---:|
| driving | 0.05776937799917814 | 0.08509640721785772 | 0.03592102212179782 | 0.03879470389153994 |
| meetroom | 0.0821721833379172 | 0.1171542035614479 | 0.06158812455336715 | 0.06651517451764022 |
| n3dv | 0.15504572925967608 | 0.21038068394586507 | 0.11930499638465619 | 0.16722150693787086 |
| robot | 0.02419867433644729 | 0.0373900943822445 | 0.003860517927481766 | 0.004186758879111352 |

### Per-gap mean gains

| gap | mean q8 MiB/frame | delta all PSNR | delta middle PSNR |
|---:|---:|---:|---:|
| 2 | 0.23137344426406925 | 0.07792147105782465 | 0.15780251136266443 |
| 4 | 0.1185445413961039 | 0.09291990502712721 | 0.1254499815718253 |
| 8 | 0.06287202380952381 | 0.08429349053606128 | 0.09758936011638486 |
| 16 | 0.03429383116883117 | 0.06405109831220557 | 0.06917953605654059 |

### 结论

- Stage26 confirms Stage25 leave-one-out checkpoints also improve full-video anchor-only RD, not only sampled intermediate validation.
- 16/16 sample-gap full-video points have positive all-frame and middle-only PSNR gain.
- Weakest point is robot gap16, but it is still positive: all +0.0039 dB, middle +0.0042 dB.
- This is currently stronger evidence than Stage23/24 for cross-sample generalization, but still limited to the 4 available StreamSplat samples.

## 2026-06-25：阶段 27 Anchor-Available Selector RD

### 执行计划

Stage27 将 keyframe selector 接入 anchor-only full-video RD。Stage16 的 unconstrained selector 会选择奇数帧，但 Stage6 当前只导出了偶数 endpoint anchors，因此不能直接传输和渲染奇数帧 keyframe anchors。为避免伪造 anchors，本阶段实现 anchor-available constrained selector：只允许在 Stage6 已有 q8 anchor 的帧上选择 keyframes。

### 新增文件

```text
scripts/run_stage27_anchor_available_selector_rd.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage27_anchor_available_selector_rd.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage27_anchor_available_selector_rd.py
```

### GPU 检查

运行前使用 `nvidia-smi`。GPU 2 有进程，GPU 6 有较小进程，GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage27_anchor_available_selector_rd/stage27_anchor_available_selector_rd.csv
experiments/stage27_anchor_available_selector_rd/stage27_anchor_available_selector_rd_summary.json
experiments/stage27_anchor_available_selector_rd/stage27_selector_comparison.csv
experiments/stage27_anchor_available_selector_rd/stage27_adapter_all_psnr_rd.png
experiments/stage27_anchor_available_selector_rd/stage27_adapter_middle_psnr_rd.png
```

### 方法

- `uniform`: standard uniform keyframes for reference gaps 4/8/16。
- `anchor_segment_rd`: Stage16-style segment RD greedy split, but constrained to available anchor indices from Stage6。
- 两者保持相同 keyframe budget，因此 `estimated_q8_static_mib_per_frame` 相同。
- 每个 sample 使用 Stage25 leave-one-out best checkpoint。

### Aggregate 结果

| metric | value |
|---|---:|
| rows | 24 |
| selector comparisons | 12 |
| mean selector delta adapter all PSNR | -0.20779707265874 |
| mean selector delta adapter middle PSNR | -0.2406135349164599 |
| positive selector all points | 4 / 12 |
| positive selector middle points | 4 / 12 |

### Selector-vs-uniform adapter PSNR delta

| sample | gap | all PSNR delta | middle PSNR delta |
|---|---:|---:|---:|
| driving | 4 | -0.15197060325413148 | -0.22026790420463982 |
| driving | 8 | -0.1802804287917823 | -0.22121484799248847 |
| driving | 16 | -0.2876096778301154 | -0.31447525772228957 |
| meetroom | 4 | -0.11145015024705174 | -0.15895369488205446 |
| meetroom | 8 | -0.29388987531305943 | -0.3411743742070712 |
| meetroom | 16 | -0.4522215409554988 | -0.4884656047377298 |
| n3dv | 4 | 0.007775012386190383 | 0.007723013074944163 |
| n3dv | 8 | 0.005460986504893128 | 0.005656976020276261 |
| n3dv | 16 | 0.1545368997013732 | 0.1661179274118325 |
| robot | 4 | 0.08645680981740256 | 0.10042260480366139 |
| robot | 8 | -0.41179171363082645 | -0.4907283482920093 |
| robot | 16 | -0.8585805902922736 | -0.9320029082699506 |

### 结论

- Stage27 selector integration is functional but the current `anchor_segment_rd` heuristic is not a reliable improvement for anchor-only full-video RD。
- It improves n3dv all gaps and robot gap4, but hurts driving/meetroom and robot long gaps.
- This negative result suggests the selector objective is mismatched with anchor-only rendering quality, and Stage16-style motion/RD edge scores are insufficient.
- 下一步应做 selector objective upgrade，而不是报告 Stage27 selector as win。

## 2026-06-25：阶段 28 Enhanced Rate Model Report

### 执行计划

Stage28 升级 rate accounting，但不改变主 rate 口径。主 rate 仍为 transmitted q8 static Gaussian anchors MiB/frame；enhanced report 额外估算 quant params、keyframe indices/timestamps、metadata，以及 q8 symbol zero-order entropy estimate。

### 新增文件

```text
scripts/run_stage28_enhanced_rate_model_report.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage28_enhanced_rate_model_report.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage28_enhanced_rate_model_report.py
```

该阶段是 CPU-only rate accounting，没有训练或渲染。

### 输出文件

```text
experiments/stage28_enhanced_rate_model_report/stage28_enhanced_rate_model_report.csv
experiments/stage28_enhanced_rate_model_report/stage28_enhanced_rate_model_report_summary.json
```

### Assumptions

- Anchor payload: q8 static anchor values only，`36864 Gaussians * 13 values * 8 bits/keyframe`。
- Quant params: per-keyframe per-field min/scale，`13 * 2 * float32 = 104 bytes/keyframe`。
- Keyframe indices: `uint16` per keyframe。
- Timestamps: `uint16` per keyframe。
- Metadata: fixed header 128 bytes + field table 64 bytes per video。
- Entropy estimate: zero-order entropy of q8 symbols after per-anchor uniform quantization；不包含真实 entropy model/header 开销。

### Aggregate 结果

| group | count | mean primary MiB/frame | mean container MiB/frame | mean entropy-container MiB/frame | mean overhead vs anchor | mean entropy bits/value | entropy savings vs raw q8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| stage26/uniform | 16 | 0.11177096015963203 | 0.11179843884697896 | 0.08875509970880423 | 0.025846283880480058% | 6.351033181558638 | 20.612085230517025% |
| stage27/anchor_segment_rd | 12 | 0.07190346545815296 | 0.07192195958389348 | 0.05710426629956228 | 0.026619792244792242% | 6.351353170965801 | 20.608085362927486% |
| stage27/uniform | 12 | 0.07190346545815296 | 0.07192195958389348 | 0.05710230238847568 | 0.026619792244792242% | 6.351305247366445 | 20.60868440791943% |

### 结论

- Metadata/index/timestamp/quant-param overhead is tiny relative to q8 anchor payload under the current fixed-size anchor format: around 0.026% vs anchor bytes.
- A simple zero-order q8 entropy estimate suggests roughly 20.6% anchor-byte savings, around 6.35 bits/value.
- This is still an estimate, not an implemented entropy-coded bitstream. Paper/report should label it as auxiliary rate analysis.

## 2026-06-25：阶段 29 Anchor-Attribute Oracle Selector RD

### 执行计划

Stage27 表明 Stage16-style motion/RD selector 在 anchor-only full-video RD 上不可靠。Stage29 改用 anchor-only quality proxy：对候选 segment 的内部 available anchors，计算 Stage25 leave-one-out adapter 预测 anchor 与真实 q8 intermediate anchor 的 attribute MSE，并用 DP 在同等 keyframe budget 下选择 keyframes。最后仍用 full-video renderer 评估。

该方法使用 held-out sample 的 intermediate anchors 做选择信号，因此是 oracle/proxy upper-bound，不是可部署在线 selector。

### 新增文件

```text
scripts/run_stage29_anchor_attribute_oracle_selector_rd.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage29_anchor_attribute_oracle_selector_rd.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage29_anchor_attribute_oracle_selector_rd.py
```

### GPU 检查

运行前使用 `nvidia-smi`。GPU 2 和 GPU 7 有进程，GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage29_anchor_attribute_oracle_selector_rd/stage29_anchor_attribute_oracle_selector_rd.csv
experiments/stage29_anchor_attribute_oracle_selector_rd/stage29_selector_comparison.csv
experiments/stage29_anchor_attribute_oracle_selector_rd/stage29_anchor_attribute_oracle_selector_rd_summary.json
experiments/stage29_anchor_attribute_oracle_selector_rd/stage29_delta_all_psnr.png
experiments/stage29_anchor_attribute_oracle_selector_rd/stage29_delta_middle_psnr.png
```

### Aggregate 结果

| metric | value |
|---|---:|
| rows | 24 |
| selector comparisons | 12 |
| mean selector delta adapter all PSNR | 0.10438711548504376 |
| mean selector delta adapter middle PSNR | 0.11370489576308233 |
| positive selector all points | 10 / 12 |
| positive selector middle points | 10 / 12 |

### Selector-vs-uniform adapter PSNR delta

| sample | gap | all PSNR delta | middle PSNR delta |
|---|---:|---:|---:|
| driving | 4 | -0.11853250341916066 | -0.1723524841032642 |
| driving | 8 | -0.05872209625836078 | -0.06481953233621596 |
| driving | 16 | 0.013726181865212794 | 0.013643069392585971 |
| meetroom | 4 | 0.02046354546268958 | 0.02281636358922512 |
| meetroom | 8 | 0.13670219109375736 | 0.15059972979703318 |
| meetroom | 16 | 0.1622869984894848 | 0.17138314537946542 |
| n3dv | 4 | 0.030343155369731534 | 0.03756564046165067 |
| n3dv | 8 | 0.10804327417126558 | 0.12227120250641477 |
| n3dv | 16 | 0.32269455633467103 | 0.34695686574367457 |
| robot | 4 | 0.16593669099086839 | 0.22049183076908108 |
| robot | 8 | 0.2550596166060153 | 0.28782964571153613 |
| robot | 16 | 0.21464377511435018 | 0.2280732722458012 |

### 结论

- Stage29 strongly suggests keyframe selection can improve anchor-only RD if the objective is aligned with adapter/anchor reconstruction quality.
- Stage27 negative result should be interpreted as heuristic-objective mismatch, not as proof that keyframe selection is useless.
- Driving gap4/8 are still negative, so even oracle/proxy anchor-MSE selection is not universally reliable.
- Next step: convert oracle/proxy signal into a deployable selector, likely by training a lightweight segment-cost predictor or deriving costs from transmitted keyframe anchors and low-cost video features.

## 2026-06-25：阶段 31 Q8 Static Anchor Bitstream Prototype

### 执行计划

Stage31 将 Stage28 的 rate estimate 推进为真实 binary container prototype。该阶段编码 Stage26 uniform keyframe anchors：container 包含 magic/version、JSON header、frame indices、timestamps、per-anchor q8 min/scale 和 q8 payload。另输出 zlib-compressed payload 版本，用作 generic compression baseline。

### 新增文件

```text
mono_dfcgs/anchor_bitstream.py
scripts/run_stage31_anchor_bitstream_prototype.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile mono_dfcgs/anchor_bitstream.py scripts/run_stage31_anchor_bitstream_prototype.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage31_anchor_bitstream_prototype.py
```

该阶段是 CPU-only encode/decode roundtrip，不需要 GPU。

### 输出文件

仓库内轻量结果：

```text
experiments/stage31_anchor_bitstream_prototype/stage31_anchor_bitstream_prototype.csv
experiments/stage31_anchor_bitstream_prototype/stage31_anchor_bitstream_prototype_summary.json
```

外部 bitstreams，不提交到 git：

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage31_anchor_bitstream_prototype
```

外部 bitstreams 总大小约 237M。

### Aggregate 结果

| metric | value |
|---|---:|
| rows | 16 |
| mean raw overhead bytes vs Stage26 anchor bytes | 11492.4375 |
| mean zlib savings percent vs raw bitstream | 34.75332282244801 |
| max roundtrip abs diff vs direct q8 dequantized anchors | 0.0 |
| mean roundtrip MSE vs direct q8 dequantized anchors | 0.0 |

### 结论

- Stage31 provides a real q8 static anchor container prototype and verifies lossless roundtrip relative to direct q8 dequantized anchors.
- Raw bitstream is slightly larger than Stage26 primary anchor bytes because the prototype stores qparams/indices/timestamps in a JSON header.
- zlib reduces raw bitstream size by about 34.75% on average, stronger than Stage28 zero-order entropy estimate because zlib exploits additional structure/repetition.
- zlib is a generic compressor, not the final entropy model; report it as a practical compression baseline, not as the learned codec result.

## 2026-06-25：阶段 30 Dense Anchor Export Preflight

### 执行计划

Stage30 统计 dense/all-frame anchor export 的必要性和空间成本。当前 Stage6 只覆盖偶数 endpoint anchors，因此 Stage16 unconstrained selector 选择奇数 keyframes 时不能直接传输对应 anchors。该阶段只做 preflight，不生成新的大 anchor dataset。

### 新增文件

```text
scripts/run_stage30_dense_anchor_export_preflight.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage30_dense_anchor_export_preflight.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage30_dense_anchor_export_preflight.py
```

该阶段是 CPU-only metadata/statistics preflight。

### 输出文件

```text
experiments/stage30_dense_anchor_export_preflight/stage30_dense_anchor_coverage.csv
experiments/stage30_dense_anchor_export_preflight/stage30_stage16_selection_compatibility.csv
experiments/stage30_dense_anchor_export_preflight/stage30_dense_anchor_export_preflight_summary.json
```

### Coverage 结果

| metric | value |
|---|---:|
| total frames | 320 |
| current unique anchors | 162 |
| missing anchors for dense coverage | 158 |
| total dense unique fp16 MiB | 292.5 |
| total dense unique q8 MiB | 146.25 |
| additional unique fp16 MiB | 144.421875 |
| additional unique q8 MiB | 72.2109375 |
| gap1 pair fp16 MiB if no dedup | 577.6875 |

### Selector compatibility

| metric | value |
|---|---:|
| Stage16 selection rows | 48 |
| unsupported rows with current Stage6 anchors | 34 |

### 结论

- Current Stage6 anchor coverage explains why Stage27 had to constrain selector to available even anchors.
- Exporting adjacent gap1 pairs for all four samples is feasible in disk terms, around 577.7 MiB if stored as non-deduplicated fp16 pair records.
- A deduplicated unique per-frame anchor store would be smaller, around 292.5 MiB fp16 total, with only about 144.4 MiB additional over current unique coverage.
- Next implementation step should be a dense/gap1 anchor exporter that writes either deduplicated per-frame anchors or gap1 pair records under external storage.

## 2026-06-25：阶段 32 Actual Bitstream RD Report

### 执行计划

Stage32 将 Stage26 leave-one-out full-video quality 与 Stage31 actual q8 anchor bitstream sizes 合并。输出 raw container 和 zlib container 两套真实码流 rate 下的 RD 表和图，同时保留 estimated q8 static anchor MiB/frame 作为对照。

### 新增文件

```text
scripts/run_stage32_actual_bitstream_rd_report.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage32_actual_bitstream_rd_report.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage32_actual_bitstream_rd_report.py
```

该阶段只做 CSV 合并和绘图，不需要 GPU。

### 输出文件

```text
experiments/stage32_actual_bitstream_rd_report/stage32_actual_bitstream_rd.csv
experiments/stage32_actual_bitstream_rd_report/stage32_gap_aggregate.csv
experiments/stage32_actual_bitstream_rd_report/stage32_actual_bitstream_rd_summary.json
experiments/stage32_actual_bitstream_rd_report/stage32_raw_*_rd.png
experiments/stage32_actual_bitstream_rd_report/stage32_zlib_*_rd.png
experiments/stage32_actual_bitstream_rd_report/stage32_mean_*_rd.png
```

### Aggregate 结果

| metric | value |
|---|---:|
| rows | 16 |
| mean delta all PSNR | 0.07979649123330468 |
| mean delta middle PSNR | 0.1125053472768538 |
| mean zlib rate / estimated q8 anchor rate | 0.6532718959778985 |
| mean zlib savings vs raw bitstream | 34.75332282244801% |

### Per-gap mean actual rates

| gap | estimated q8 MiB/frame | raw bitstream MiB/frame | zlib bitstream MiB/frame | adapter all PSNR | adapter middle PSNR |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.23137344426406925 | 0.23165553396321917 | 0.15108759154374352 | 29.94152342429438 | 28.25286654810653 |
| 4 | 0.1185445413961039 | 0.11868976438123867 | 0.07742046583572297 | 28.31306297056608 | 27.172461353061053 |
| 8 | 0.06287202380952381 | 0.06294975617919304 | 0.04109464166484354 | 26.654635281827648 | 25.862070399193193 |
| 16 | 0.03429383116883117 | 0.03433686593260466 | 0.022424008958286556 | 25.087539219111537 | 24.55959121535453 |

### 结论

- Stage32 turns the primary RD evidence into an actual q8 anchor bitstream RD report.
- Raw prototype container rate is very close to estimated q8 anchor rate, around 1.001x, because JSON/qparam/index overhead is small relative to payload.
- Zlib container rate is around 65.3% of estimated q8 anchor rate on average, while preserving the same decoded q8 anchors and full-video quality.
- Zlib is still a generic compression baseline, not the final learned entropy coder.

## 2026-06-25：阶段 33 Dense Gap1 Anchor Export

### 执行计划

Stage33 根据 Stage30 preflight 实际导出 gap1 adjacent pair anchors，让每个视频帧都至少作为某个 pair endpoint 出现，从而解除 Stage27/Stage16 selector 只能选择现有偶数 anchors 的限制。

### 新增文件

```text
scripts/run_stage33_dense_gap1_anchor_export.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage33_dense_gap1_anchor_export.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage33_dense_gap1_anchor_export.py
```

### GPU 检查

运行前使用 `nvidia-smi`。GPU 2 有大进程，GPU 7 有小进程，GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

仓库内轻量 manifest/summary：

```text
experiments/stage33_dense_gap1_anchor_export/stage33_dense_gap1_anchor_manifest.csv
experiments/stage33_dense_gap1_anchor_export/stage33_dense_gap1_anchor_manifest.json
experiments/stage33_dense_gap1_anchor_export/stage33_dense_gap1_anchor_summary.csv
experiments/stage33_dense_gap1_anchor_export/stage33_dense_gap1_anchor_summary.json
```

外部 anchor dataset，不提交到 git：

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage33_dense_gap1_anchor_export
```

外部目录大小约 581M。

### Summary 结果

| sample | total frames | exported gap1 pairs | unique anchor count | unique anchor ratio | total anchor MiB |
|---|---:|---:|---:|---:|---:|
| n3dv | 81 | 80 | 81 | 1.0 | 146.25 |
| meetroom | 81 | 80 | 81 | 1.0 | 146.25 |
| driving | 81 | 80 | 81 | 1.0 | 146.25 |
| robot | 77 | 76 | 77 | 1.0 | 138.9375 |

| metric | value |
|---|---:|
| total pair rows | 316 |
| total anchor MiB | 577.6875 |

### 结论

- Stage33 successfully provides dense endpoint coverage for all 320 frames across the four samples.
- This dataset is non-deduplicated gap1 pair storage, so adjacent anchors are duplicated across neighboring pair files.
- It is now possible to rerun unconstrained Stage16/Stage29 selectors using actual anchors for odd frames.

## 2026-06-25：阶段 34 Dense-Anchor Stage16 Selector RD

### 执行计划

Stage34 使用 Stage33 dense gap1 anchors 重新评估 Stage16 的 unconstrained `segment_rd` selections。与 Stage27 不同，本阶段所有奇数帧 keyframes 都有真实 exported anchors，不需要 snapping，也不需要 anchor-available constraint。

### 新增文件

```text
scripts/run_stage34_dense_stage16_selector_rd.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage34_dense_stage16_selector_rd.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage34_dense_stage16_selector_rd.py
```

### GPU 检查

运行前使用 `nvidia-smi`。GPU 2 有大进程，GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage34_dense_stage16_selector_rd/stage34_dense_stage16_selector_rd.csv
experiments/stage34_dense_stage16_selector_rd/stage34_selector_comparison.csv
experiments/stage34_dense_stage16_selector_rd/stage34_dense_stage16_selector_rd_summary.json
experiments/stage34_dense_stage16_selector_rd/stage34_delta_all_psnr.png
experiments/stage34_dense_stage16_selector_rd/stage34_delta_middle_psnr.png
```

### Aggregate 结果

| metric | value |
|---|---:|
| rows | 24 |
| comparisons | 12 |
| mean selector delta adapter all PSNR | -0.20844626209557107 |
| mean selector delta adapter middle PSNR | -0.23827828367891582 |
| positive selector all points | 4 / 12 |
| positive selector middle points | 4 / 12 |

### Selector-vs-uniform adapter PSNR delta

| sample | gap | all PSNR delta | middle PSNR delta |
|---|---:|---:|---:|
| driving | 4 | -0.04081701674233429 | -0.08122249036710372 |
| driving | 8 | -0.17610964099429793 | -0.21006903285362455 |
| driving | 16 | -0.3701326258593731 | -0.4024097767248733 |
| meetroom | 4 | -0.14344631811249542 | -0.2087883543335245 |
| meetroom | 8 | -0.25901919279537466 | -0.30222039669111567 |
| meetroom | 16 | -0.4515902929372686 | -0.48782180927879537 |
| n3dv | 4 | 0.020734923720947762 | 0.02170452108476084 |
| n3dv | 8 | 0.009707321878057229 | 0.00956914383604257 |
| n3dv | 16 | 0.16337144121226643 | 0.17505980150671618 |
| robot | 4 | 0.07922042169743904 | 0.09803821429976622 |
| robot | 8 | -0.35994816705443 | -0.41630651822905307 |
| robot | 16 | -0.9733259991599894 | -1.0548727063961856 |

### 结论

- Stage34 confirms Stage27's negative result is not primarily caused by missing odd-frame anchors.
- Even with dense Stage33 anchors, Stage16 `segment_rd` remains worse than uniform on average and only improves 4/12 points.
- Stage29 remains the stronger evidence: selector can help when the objective is aligned with anchor/adapter quality, but Stage16 motion/RD heuristic should not be used as final selector.

## 2026-06-25：阶段 35 Dense Anchor-Attribute Oracle Selector RD

### 执行计划

Stage35 使用 Stage33 dense gap1 anchors 作为全帧 candidate pool，重跑 Stage29 的 anchor-attribute oracle/proxy selector。选择目标是最小化 segment 内部 adapter-predicted anchor 与 dense q8 anchor 的 attribute MSE，然后用 full-video renderer 评估 RD。

该方法仍使用 intermediate anchors 作为 oracle/proxy target，因此不是可部署 encoder-side selector。

### 新增文件

```text
scripts/run_stage35_dense_anchor_attribute_oracle_selector_rd.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage35_dense_anchor_attribute_oracle_selector_rd.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage35_dense_anchor_attribute_oracle_selector_rd.py
```

### GPU 检查

运行前使用 `nvidia-smi`。GPU 2 有大进程，GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage35_dense_anchor_attribute_oracle_selector_rd/stage35_dense_anchor_attribute_oracle_selector_rd.csv
experiments/stage35_dense_anchor_attribute_oracle_selector_rd/stage35_selector_comparison.csv
experiments/stage35_dense_anchor_attribute_oracle_selector_rd/stage35_dense_anchor_attribute_oracle_selector_rd_summary.json
experiments/stage35_dense_anchor_attribute_oracle_selector_rd/stage35_delta_all_psnr.png
experiments/stage35_dense_anchor_attribute_oracle_selector_rd/stage35_delta_middle_psnr.png
```

### Aggregate 结果

| metric | value |
|---|---:|
| rows | 24 |
| comparisons | 12 |
| mean selector delta adapter all PSNR | 0.19715194312405782 |
| mean selector delta adapter middle PSNR | 0.22543596674489544 |
| positive selector all points | 12 / 12 |
| positive selector middle points | 12 / 12 |

### Selector-vs-uniform adapter PSNR delta

| sample | gap | all PSNR delta | middle PSNR delta |
|---|---:|---:|---:|
| driving | 4 | 0.10271658333038047 | 0.11828421815953405 |
| driving | 8 | 0.07420764607518393 | 0.07862427317779819 |
| driving | 16 | 0.04696688605787713 | 0.047731645026328096 |
| meetroom | 4 | 0.08472378864621177 | 0.09548642832174536 |
| meetroom | 8 | 0.1634962721689739 | 0.17821243977365953 |
| meetroom | 16 | 0.1456696036510401 | 0.15254014616091638 |
| n3dv | 4 | 0.06615256078153564 | 0.08225455855895447 |
| n3dv | 8 | 0.15626101230313694 | 0.17673748652441645 |
| n3dv | 16 | 0.32445355363339345 | 0.34806274149884686 |
| robot | 4 | 0.35480000997070604 | 0.47451667271218767 |
| robot | 8 | 0.544173011655797 | 0.6282561146693126 |
| robot | 16 | 0.3022023892144574 | 0.3245248763550457 |

### 结论

- Stage35 is the strongest selector upper-bound so far: dense anchor-attribute oracle improves all 12 sample-gap points.
- Dense candidate coverage fixes Stage29's driving gap4/8 negative points.
- Stage34 vs Stage35 cleanly shows the decisive factor is selector objective alignment, not just dense anchor availability.
- Next step is Stage36: encode Stage35 selected keyframes with actual q8 bitstreams and report actual-bitstream RD.

## 2026-06-25：阶段 36 Dense Oracle Actual-Bitstream RD

### 执行计划

Stage36 将 Stage35 的 `uniform` 和 `dense_anchor_attr_oracle` selected keyframes 编码为真实 q8 static anchor bitstream，分别输出 raw container 和 zlib container sizes。质量指标继承 Stage35 full-video rendering，rate 使用 Stage36 actual bitstream sizes。

### 新增文件

```text
scripts/run_stage36_dense_oracle_actual_bitstream_rd.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage36_dense_oracle_actual_bitstream_rd.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage36_dense_oracle_actual_bitstream_rd.py
```

该阶段是 CPU-only bitstream encode/decode 和 plotting。

### 输出文件

仓库内轻量结果：

```text
experiments/stage36_dense_oracle_actual_bitstream_rd/stage36_dense_oracle_actual_bitstream_rd.csv
experiments/stage36_dense_oracle_actual_bitstream_rd/stage36_raw_selector_comparison.csv
experiments/stage36_dense_oracle_actual_bitstream_rd/stage36_zlib_selector_comparison.csv
experiments/stage36_dense_oracle_actual_bitstream_rd/stage36_dense_oracle_actual_bitstream_rd_summary.json
experiments/stage36_dense_oracle_actual_bitstream_rd/stage36_*_rd.png
```

外部 bitstreams，不提交到 git：

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage36_dense_oracle_actual_bitstream_rd
```

外部大小约 229M。

### Aggregate 结果

| metric | value |
|---|---:|
| rows | 24 |
| mean selector delta adapter all PSNR | 0.19715194312405782 |
| mean selector delta adapter middle PSNR | 0.22543596674489544 |
| positive selector all points | 12 / 12 |
| positive selector middle points | 12 / 12 |
| mean zlib savings vs raw bitstream | 34.75464768969244% |
| max roundtrip abs diff | 0.0 |

### Rate observations

- Raw bitstream rates for uniform and dense oracle are effectively equal at the same keyframe budget; differences are only header/JSON-length noise.
- Zlib rates can differ slightly by selected keyframe content, but the differences are tiny compared with the PSNR gains.
- Example: robot gap8 zlib rate decreases from 0.04393036 to 0.04386477 MiB/frame while adapter middle PSNR increases by +0.6283 dB.

### 结论

- Stage36 converts Stage35's oracle selector upper bound into actual-bitstream RD evidence.
- Dense oracle improves all 12 points while preserving essentially the same keyframe-count bitstream budget.
- This is the strongest selector upper-bound artifact so far, but still not a deployable selector because the keyframe decisions use dense intermediate anchors as oracle targets.

## 2026-06-25：阶段 37 Deployable Selector Cost Dataset

### 执行计划

Stage37 构建用于训练 deployable selector cost predictor 的 segment-level dataset。Label 仍使用 dense intermediate anchors 计算 Stage35-style oracle/proxy cost，但 features 只使用 encoder-side 可获得信息：endpoint q8 anchors、segment length、endpoint RGB/motion features。

### 新增文件

```text
scripts/run_stage37_deployable_selector_cost_dataset.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage37_deployable_selector_cost_dataset.py
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage37_deployable_selector_cost_dataset.py
```

### GPU 检查

运行前使用 `nvidia-smi`。GPU 1 有小进程，GPU 2 有大进程，GPU 3 空闲，因此使用 `CUDA_VISIBLE_DEVICES=3`。

### 输出文件

```text
experiments/stage37_deployable_selector_cost_dataset/stage37_deployable_selector_cost_dataset.csv
experiments/stage37_deployable_selector_cost_dataset/stage37_deployable_selector_cost_dataset_summary.json
```

### Dataset 规模

| metric | value |
|---|---:|
| max segment length | 32 |
| base segment rows | 7812 |
| expanded leave-one-out rows | 31248 |

### Per-sample label statistics and correlations

| sample | rows | label mean | label min | label max | corr endpoint anchor MSE | corr RGB motion mean | corr segment length |
|---|---:|---:|---:|---:|---:|---:|---:|
| n3dv | 1984 | 0.00532244782212021 | 1.6982678062049672e-05 | 0.021853812329936773 | 0.7910826029069399 | 0.017256400691612793 | 0.9570222166451825 |
| meetroom | 1984 | 0.021771396093475733 | 0.00016252839122898877 | 0.06116097897756845 | 0.7162381539481427 | 0.06799304584099193 | 0.9806131694966483 |
| driving | 1984 | 0.054026661228231056 | 0.0007965295226313174 | 0.15516445587854832 | 0.7915354029010068 | 0.012845690433336424 | 0.9822683493581169 |
| robot | 1860 | 0.08738319363159638 | 0.0005001415847800672 | 0.3412707191891968 | 0.6147167428950462 | 0.22350813869480543 | 0.955821501089398 |

### 结论

- Segment length and endpoint anchor MSE are strong simple predictors for dense oracle cost.
- RGB motion alone is weakly correlated with the oracle cost on these samples, especially n3dv/driving.
- Stage37 provides the training data needed for Stage38 deployable selector cost predictor.

## 2026-06-25：阶段 38 Deployable Cost Predictor Validation

### 执行计划

Stage38 在 Stage37 leave-one-out dataset 上训练简单 ridge regression cost predictor。目标是预测 `log_label_anchor_attr_mse`。比较两个 baseline：只使用 segment length 的 `length_only_ridge`，以及使用全部 deployable features 的 `full_feature_ridge`。

### 新增文件

```text
scripts/run_stage38_deployable_cost_predictor_validation.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage38_deployable_cost_predictor_validation.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage38_deployable_cost_predictor_validation.py
```

该阶段是 CPU-only regression validation。

### 输出文件

```text
experiments/stage38_deployable_cost_predictor_validation/stage38_predictor_metrics.csv
experiments/stage38_deployable_cost_predictor_validation/stage38_predictor_predictions.csv
experiments/stage38_deployable_cost_predictor_validation/stage38_deployable_cost_predictor_validation_summary.json
```

### Aggregate 结果

| metric | length-only ridge | full-feature ridge |
|---|---:|---:|
| mean Spearman log-cost | 0.9810832482322229 | 0.7734269107632606 |
| mean RMSE log-cost | 0.6321678273962144 | 1.3141291584888863 |

### Fold-wise 结果

| heldout | model | Spearman log | RMSE log | Pearson cost |
|---|---|---:|---:|---:|
| driving | length_only | 0.989720327238339 | 0.49236610749417725 | 0.915799105951025 |
| driving | full_feature | 0.9820209157992446 | 1.538212462701852 | 0.7691172517622008 |
| meetroom | length_only | 0.9900150548370009 | 0.27755951764524434 | 0.9159853987410811 |
| meetroom | full_feature | 0.9803012906569413 | 1.1961997428836078 | 0.8909175137169681 |
| n3dv | length_only | 0.978034469454981 | 1.0152720329878608 | 0.9057962566716188 |
| n3dv | full_feature | 0.9843223362668158 | 0.4210671086362759 | 0.938952241825634 |
| robot | length_only | 0.966563141398571 | 0.7434736514575749 | 0.8800013214396364 |
| robot | full_feature | 0.14706310033004102 | 2.10103731973381 | 0.19128289118889527 |

### 结论

- Length-only ridge is surprisingly strong because oracle segment cost is highly length-correlated.
- Full-feature ridge does not robustly generalize across samples and collapses on robot held-out.
- Do not use naive full-feature ridge as final deployable selector yet.
- Next step should test predicted selector RD with length-only and/or improved normalized feature models, but expect length-only to behave close to uniform.

## 2026-06-25：阶段 39 Predicted Selector RD

### 执行计划

Stage39 将 Stage38 产生的 predicted segment costs 用于 dynamic programming keyframe selection，再用 Stage33 dense keyframe anchors 和 Stage25 leave-one-out adapter 进行 full-video anchor-only RD 评估。该阶段检验 deployable predictor 是否能真正带来 selector-level RD 增益。

### 新增文件

```text
scripts/run_stage39_predicted_selector_rd.py
```

### 运行命令

```text
python -m py_compile scripts/run_stage39_predicted_selector_rd.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage39_predicted_selector_rd.py
```

### GPU 检查

运行前使用 `nvidia-smi`。GPU 2 满载，GPU 1 基本空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage39_predicted_selector_rd/stage39_predicted_selector_rd.csv
experiments/stage39_predicted_selector_rd/stage39_selector_comparison.csv
experiments/stage39_predicted_selector_rd/stage39_predicted_selector_rd_summary.json
experiments/stage39_predicted_selector_rd/stage39_length_only_ridge_delta_all_psnr.png
experiments/stage39_predicted_selector_rd/stage39_length_only_ridge_delta_middle_psnr.png
experiments/stage39_predicted_selector_rd/stage39_full_feature_ridge_delta_all_psnr.png
experiments/stage39_predicted_selector_rd/stage39_full_feature_ridge_delta_middle_psnr.png
```

### Aggregate 结果

| method | points | positive all PSNR | positive middle PSNR | mean delta all PSNR | mean delta middle PSNR |
|---|---:|---:|---:|---:|---:|
| length_only_ridge | 12 | 0 | 0 | -0.035285203146952604 | -0.038178305693805946 |
| full_feature_ridge | 12 | 1 | 1 | -0.29239474971647805 | -0.3641505090023867 |

### 结论

- Stage39 是关键负结果：Stage38 predicted cost 直接驱动 DP selection 后，没有超过 uniform keyframe selection。
- `length_only_ridge` 在 Stage38 ranking metrics 上很强，但转成 fixed-budget selection 后 12/12 点低于 uniform，说明高相关不等价于 RD-optimal selector。
- `full_feature_ridge` 受跨样本 domain shift 影响更严重，只在 1/12 点上超过 uniform。
- 下一步应改用 sample-normalized / rank-normalized / candidate-normalized predictor 或直接训练 selector objective，而不是继续使用 raw-cost ridge。

## 2026-06-25：阶段 40 Normalized Cost Predictor Validation

### 执行计划

Stage40 针对 Stage38/39 的 negative result，改用 sample/candidate 内归一化的 deployable features，并尝试 z-score log target 与 rank target。目标是验证 predictor 是否能减少跨样本 scale mismatch，尤其避免 Stage38 full-feature ridge 在 robot held-out 上崩溃。

### 新增文件

```text
scripts/run_stage40_normalized_cost_predictor_validation.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage40_normalized_cost_predictor_validation.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage40_normalized_cost_predictor_validation.py
```

Stage40 是 CPU-only regression validation。运行前仍检查 `nvidia-smi`，GPU 2 满载，GPU 1/4/5/6/7 空闲。

### 输出文件

```text
experiments/stage40_normalized_cost_predictor_validation/stage40_predictor_metrics.csv
experiments/stage40_normalized_cost_predictor_validation/stage40_predictor_predictions.csv
experiments/stage40_normalized_cost_predictor_validation/stage40_normalized_cost_predictor_validation_summary.json
```

### Aggregate 结果

| model | mean Spearman log | mean Spearman transformed | mean RMSE transformed |
|---|---:|---:|---:|
| length_sample_z_zlog | 0.9810832482322229 | 0.9810832482322229 | 0.4479778496500245 |
| full_sample_z_zlog | 0.8634807769805042 | 0.8634807769805042 | 0.5124079104179997 |
| length_sample_z_rank | 0.9810832482322229 | 0.9810832482322229 | 0.05718025637542119 |
| full_sample_z_rank | 0.9855087922482462 | 0.9855087922482462 | 0.05203336788251095 |

### 结论

- Rank target + sample-normalized full features gives the best predictor ranking: `full_sample_z_rank` mean Spearman log `0.9855087922482462`.
- This substantially improves over Stage38 raw full-feature ridge mean Spearman `0.7734269107632606` and suggests scale normalization is necessary.
- Stage40 predictor metrics alone are not enough; next step must run full-video RD selection using Stage40 predictions.

## 2026-06-25：阶段 41 Normalized Predicted Selector RD

### 执行计划

Stage41 将 Stage40 的 normalized/rank predictor scores 作为 relative DP segment costs，重跑 full-video anchor-only RD。比较 uniform、`length_sample_z_rank`、`full_sample_z_rank`、`full_sample_z_zlog`。

### 新增文件

```text
scripts/run_stage41_normalized_predicted_selector_rd.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage41_normalized_predicted_selector_rd.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage41_normalized_predicted_selector_rd.py
```

### GPU 检查

运行前使用 `nvidia-smi`。GPU 2 满载，GPU 3 有运行进程，GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage41_normalized_predicted_selector_rd/stage41_normalized_predicted_selector_rd.csv
experiments/stage41_normalized_predicted_selector_rd/stage41_selector_comparison.csv
experiments/stage41_normalized_predicted_selector_rd/stage41_normalized_predicted_selector_rd_summary.json
```

### Aggregate 结果

| method | points | positive all PSNR | positive middle PSNR | mean delta all PSNR | mean delta middle PSNR |
|---|---:|---:|---:|---:|---:|
| length_sample_z_rank | 12 | 0 | 0 | -0.5054370606610267 | -0.5918630285707692 |
| full_sample_z_rank | 12 | 0 | 0 | -0.847949115968393 | -0.9945000333189684 |
| full_sample_z_zlog | 12 | 0 | 0 | -0.913330142784306 | -1.082226324905095 |

### 结论

- Stage41 是更强的负结果：即使 Stage40 predictor ranking 明显改善，直接把 predicted relative cost 放入 DP selection 仍然 0/12 点超过 uniform。
- 当前 proxy label / predictor 与最终 rendered RD objective 存在明显 mismatch；rank/normalized cost 可能导致 DP 选择过度非均匀的 segment layout。
- 下一步不应继续只优化 predictor correlation，而应加入 selector-level calibration，例如 spacing/fairness constraints、uniform-prior penalty、或直接学习/验证 keyframe layout 的 rendered RD objective。

## 2026-06-25：阶段 42 Calibrated Selector Prior RD

### 执行计划

Stage42 在 Stage40 `full_sample_z_rank` predictor score 上加入 uniform-layout prior：

```text
cost = pred_cost + alpha * ((segment_length - target_gap) / target_gap)^2
```

扫 `alpha = 0, 0.05, 0.1, 0.3, 1, 3, 10`，评估是否可以避免 Stage41 的过度非均匀 layout，同时保持可能的 selector gain。

### 新增文件

```text
scripts/run_stage42_calibrated_selector_prior_rd.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage42_calibrated_selector_prior_rd.py
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage42_calibrated_selector_prior_rd.py
```

### GPU 检查

运行前使用 `nvidia-smi`。GPU 2 满载，GPU 1/4/5/6/7 有其他进程，GPU 3 空闲，因此使用 `CUDA_VISIBLE_DEVICES=3`。

### 输出文件

```text
experiments/stage42_calibrated_selector_prior_rd/stage42_calibrated_selector_prior_rd.csv
experiments/stage42_calibrated_selector_prior_rd/stage42_selector_comparison.csv
experiments/stage42_calibrated_selector_prior_rd/stage42_calibrated_selector_prior_rd_summary.json
```

### Aggregate 结果

| method | positive all PSNR | exact uniform points | mean delta all PSNR | mean delta middle PSNR |
|---|---:|---:|---:|---:|
| prior 0.0 | 0 | 0 | -0.847949115968393 | -0.9945000333189684 |
| prior 0.05 | 0 | 0 | -0.6041559056558018 | -0.7002204555279228 |
| prior 0.1 | 0 | 0 | -0.44743080235738564 | -0.5125116619406539 |
| prior 0.3 | 1 | 0 | -0.21756857982868696 | -0.24895227588894642 |
| prior 1.0 | 3 | 6 | -0.09932737838737005 | -0.10902505140314893 |
| prior 3.0 | 1 | 9 | -0.014864631968911782 | -0.014910339549485249 |
| prior 10.0 | 1 | 10 | -0.01123851571938476 | -0.01083193660040808 |

### 结论

- Uniform prior 可以显著缓解 predicted selector 的过度非均匀 layout，`alpha=10` 几乎回到 uniform。
- 但最佳 mean 仍为负，且高 alpha 下多数点 exact uniform，说明当前 deployable predicted selector 尚未产生真实 RD gain。
- 当前结论应写成：dense oracle selector 有明确上界收益，但 deployable predictor/selector 仍未超过 uniform，需要更贴近 rendered RD 的 selector-level training 或更多数据。

## 2026-06-26：阶段 43 Selector Evidence Synthesis

### 执行计划

Stage43 汇总当前 selector/adapter 证据，避免将 oracle upper bound、deployable selector、adapter gain 混为一谈。该阶段生成机器可读 JSON/CSV 和可直接汇报的 Markdown 表格。

### 新增文件

```text
scripts/run_stage43_selector_evidence_synthesis.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage43_selector_evidence_synthesis.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage43_selector_evidence_synthesis.py
```

Stage43 是 CPU-only 汇总脚本。运行前仍检查 `nvidia-smi`。

### 输出文件

```text
experiments/stage43_selector_evidence_synthesis/stage43_selector_evidence_synthesis.csv
experiments/stage43_selector_evidence_synthesis/stage43_selector_evidence_synthesis_summary.json
experiments/stage43_selector_evidence_synthesis/stage43_selector_evidence_synthesis.md
```

### 关键汇总

| stage | evidence | points | positive all | mean delta all PSNR | interpretation |
|---|---|---:|---:|---:|---|
| 26 | leave-one-out anchor adapter vs linear | 16 | 16 | 0.079796 | best current adapter generalization evidence; not selector gain |
| 36 | dense anchor-attribute oracle selector vs uniform | 12 | 12 | 0.197152 | strong selector upper bound; non-deployable |
| 39 | raw ridge predicted selectors | 12 each | 0-1 | negative | deployable selector negative |
| 41 | normalized/rank predicted selectors | 12 each | 0 | negative | deployable selector negative |
| 42 | uniform-prior calibrated selector | 12 each | 0-3 | best -0.011239 | mostly fallback to uniform |

### 结论

- 当前 strongest deployable codec evidence 是 Stage26：leave-one-out adapter 在 uniform keyframes 下 16/16 点优于 linear interpolation。
- 当前 strongest selector evidence 是 Stage36：dense oracle/proxy selector 12/12 正收益，并有 actual zlib q8 anchor bitstream rate，但不是 deployable selector。
- 当前 deployable predicted selector 尚未超过 uniform，后续必须诚实报告为负结果，并把它作为改进 selector objective / 扩数据的动机。

## 2026-06-26：阶段 44 Rendered Segment Distortion Dataset

### 执行计划

Stage44 构建新的 Adaptive Keyframe Selection 训练/分析数据集。与 Stage37 的 anchor-attribute proxy label 不同，Stage44 直接使用 frozen Stage25 leave-one-out adapter 渲染候选 segment 的中间帧，并用 RGB MSE 形成 rendered distortion label。

### 新增文件

```text
scripts/run_stage44_rendered_segment_distortion_dataset.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage44_rendered_segment_distortion_dataset.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage44_rendered_segment_distortion_dataset.py
```

### GPU 检查

运行前使用 `nvidia-smi`。GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 重要修正

首轮运行发现 target RGB tensor 是 `[1, 3, H, W]`，而 renderer output 是 `[1, 1, 3, H, W]`，PyTorch 会 broadcasting 并给出 warning。已修正 target tensor 为 `[1, 1, 3, H, W]` 后重新运行，覆盖错误输出。

### 输出文件

```text
experiments/stage44_rendered_segment_distortion_dataset/stage44_rendered_segment_distortion_dataset.csv
experiments/stage44_rendered_segment_distortion_dataset/stage44_rendered_segment_distortion_dataset_summary.json
```

### Dataset 规模

| metric | value |
|---|---:|
| max segment length | 32 |
| max sampled targets per segment | 3 |
| base segment rows | 8128 |
| expanded leave-one-out rows | 32512 |
| rendered target evaluations | 22504 |

### Per-sample label statistics

| sample | rows | rendered target evals | label mean | label max | corr endpoint anchor MSE | corr RGB motion mean | corr segment length |
|---|---:|---:|---:|---:|---:|---:|---:|
| n3dv | 2064 | 5716 | 0.014895839439636203 | 0.04597786716961612 | 0.7870665258926746 | -0.05354426673864032 | 0.9862524358485771 |
| meetroom | 2064 | 5716 | 0.03329636638431375 | 0.09466822538524866 | 0.7107057762991904 | 0.08907297566321296 | 0.9703062757077143 |
| driving | 2064 | 5716 | 0.05687899679995837 | 0.2161321323364973 | 0.8078750784857266 | 0.08271829957254055 | 0.9566332944035877 |
| robot | 1936 | 5356 | 0.13955990905128163 | 0.7303019681324561 | 0.5965624603408157 | 0.23088073370512416 | 0.8824542788008414 |

### 结论

- Stage44 生成了第一版 rendered-distortion segment cost labels，可直接用于 Stage45 rendered-oracle DP selector。
- 当前 label 仍与 segment length 高相关，但它来自实际 adapter-rendered RGB distortion，目标比 Stage37 anchor-MSE proxy 更接近最终 RD。
- 该版本为快速 RD 曲线使用 sampled targets；大规模精确版本可用 `--max_targets_per_segment=0` 计算所有中间帧。

## 2026-06-26：阶段 45 Rendered-Oracle Adaptive Selector RD

### 执行计划

Stage45 使用 Stage44 `adapter_mse_sum_est` 作为 rendered-distortion oracle segment cost，通过 DP 在固定 keyframe budget 下选择 adaptive keyframes，并与 uniform keyframes 做 full-video anchor-only RD 对比。

### 新增文件

```text
scripts/run_stage45_rendered_oracle_adaptive_selector_rd.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage45_rendered_oracle_adaptive_selector_rd.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage45_rendered_oracle_adaptive_selector_rd.py
```

### GPU 检查

运行前使用 `nvidia-smi`。GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage45_rendered_oracle_adaptive_selector_rd/stage45_rendered_oracle_adaptive_selector_rd.csv
experiments/stage45_rendered_oracle_adaptive_selector_rd/stage45_selector_comparison.csv
experiments/stage45_rendered_oracle_adaptive_selector_rd/stage45_rendered_oracle_adaptive_selector_rd_summary.json
experiments/stage45_rendered_oracle_adaptive_selector_rd/stage45_adapter_all_psnr_rd.png
experiments/stage45_rendered_oracle_adaptive_selector_rd/stage45_adapter_middle_psnr_rd.png
experiments/stage45_rendered_oracle_adaptive_selector_rd/stage45_adapter_all_ssim_rd.png
experiments/stage45_rendered_oracle_adaptive_selector_rd/stage45_adapter_middle_ssim_rd.png
```

### Aggregate 结果

| metric | value |
|---|---:|
| points | 12 |
| positive all PSNR points | 8 |
| positive middle PSNR points | 8 |
| mean delta all PSNR | 0.04996005587468311 |
| mean delta middle PSNR | 0.06134006884884169 |

### Per-point delta all PSNR

| sample | gap4 | gap8 | gap16 |
|---|---:|---:|---:|
| n3dv | 0.06333816052883279 | 0.12809375694453706 | 0.2627643201104988 |
| meetroom | 0.09904846893396524 | 0.16468754240036532 | -0.14085700247952815 |
| driving | 0.030211703498583375 | -0.15137123254497098 | -0.08618988760293078 |
| robot | 0.17423209197442446 | 0.2969394033681816 | -0.2413766546357614 |

### 结论

- Stage45 给出了第一版 Adaptive Keyframe Selection RD 曲线：rendered-distortion oracle 明显优于之前 predicted selector，8/12 点正收益。
- 该结果说明把 selector label 从 anchor-attribute proxy 改为 rendered distortion 是有效方向。
- 但 gap16 的 meetroom/robot 和 driving gap8/16 仍为负，当前 sampled-label oracle 还不足以作为最终成功结果。
- 下一步建议优先做 Stage45b/Stage46：对 rendered oracle 加 min-gap/uniform-prior calibration 或用 all-middle-frame label 重算关键负点，再生成 actual bitstream RD。

## 2026-06-26：阶段 45b Calibrated Rendered-Oracle Selector RD

### 执行计划

Stage45b 在 Stage45 rendered-oracle segment cost 上增加 layout calibration：

- uniform segment-length prior：`cost += alpha * ((length - gap) / gap)^2`
- minimum segment length：`min2` 和 `minhalf`

目标是减少 Stage45 的大负点，同时保留 adaptive selection 的正收益。

### 新增文件

```text
scripts/run_stage45b_calibrated_rendered_oracle_selector_rd.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage45b_calibrated_rendered_oracle_selector_rd.py
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage45b_calibrated_rendered_oracle_selector_rd.py
```

### GPU 检查

运行前使用 `nvidia-smi`。GPU 1/2 有运行进程，GPU 3 空闲，因此使用 `CUDA_VISIBLE_DEVICES=3`。

### 输出文件

```text
experiments/stage45b_calibrated_rendered_oracle_selector_rd/stage45b_calibrated_rendered_oracle_selector_rd.csv
experiments/stage45b_calibrated_rendered_oracle_selector_rd/stage45b_selector_comparison.csv
experiments/stage45b_calibrated_rendered_oracle_selector_rd/stage45b_selector_aggregates.csv
experiments/stage45b_calibrated_rendered_oracle_selector_rd/stage45b_calibrated_rendered_oracle_selector_rd_summary.json
```

### Aggregate 结果

| method | positive all | mean delta all PSNR | min delta all PSNR | exact uniform points |
|---|---:|---:|---:|---:|
| rendered_raw | 8/12 | 0.04996005587468311 | -0.2413766546357614 | 0 |
| rendered_prior_0p1 | 7/12 | 0.06034847377870989 | -0.018963597365331708 | 4 |
| rendered_prior_0p3 | 5/12 | 0.028495041208740208 | 0.0 | 7 |
| rendered_prior_1p0 | 1/12 | 0.0013992107446666087 | -0.0177131536988 | 10 |
| rendered_min2 | 8/12 | 0.04582177347634975 | -0.2413766546357614 | 0 |
| rendered_minhalf | 8/12 | 0.023650366089849644 | -0.5249054201293362 | 0 |

### 结论

- `rendered_prior_0p1` 是目前最稳的 rendered-oracle selector variant：mean Δall PSNR 提高到 `+0.06035 dB`，最坏点收敛到约 `-0.019 dB`。
- 代价是 4/12 点回退到 exact uniform，positive count 从 raw 的 8/12 变成 7/12。
- 这说明 uniform prior 是必要的 layout regularization；下一步可以用该 variant 生成 actual q8/zlib bitstream RD，同时继续研究 all-middle-frame labels 或 predictor training。

## 2026-06-26：阶段 46 Calibrated Adaptive Actual-Bitstream RD

### 执行计划

Stage46 使用 Stage45b 的 `rendered_prior_0p1` adaptive selector，与 uniform 在相同 keyframe budget 下生成真实 q8 anchor bitstreams，并输出 raw/zlib MiB/frame RD 曲线。

### 新增文件

```text
scripts/run_stage46_calibrated_adaptive_actual_bitstream_rd.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage46_calibrated_adaptive_actual_bitstream_rd.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage46_calibrated_adaptive_actual_bitstream_rd.py
```

Stage46 是 CPU bitstream 生成/汇总脚本。运行前仍按要求检查 `nvidia-smi`。

### 输出文件

仓库内：

```text
experiments/stage46_calibrated_adaptive_actual_bitstream_rd/stage46_calibrated_adaptive_actual_bitstream_rd.csv
experiments/stage46_calibrated_adaptive_actual_bitstream_rd/stage46_raw_selector_comparison.csv
experiments/stage46_calibrated_adaptive_actual_bitstream_rd/stage46_zlib_selector_comparison.csv
experiments/stage46_calibrated_adaptive_actual_bitstream_rd/stage46_calibrated_adaptive_actual_bitstream_rd_summary.json
experiments/stage46_calibrated_adaptive_actual_bitstream_rd/stage46_raw_adapter_all_psnr_rd.png
experiments/stage46_calibrated_adaptive_actual_bitstream_rd/stage46_raw_adapter_middle_psnr_rd.png
experiments/stage46_calibrated_adaptive_actual_bitstream_rd/stage46_zlib_adapter_all_psnr_rd.png
experiments/stage46_calibrated_adaptive_actual_bitstream_rd/stage46_zlib_adapter_middle_psnr_rd.png
```

外部 bitstreams：

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage46_calibrated_adaptive_actual_bitstream_rd
```

### 关键结果

| metric | value |
|---|---:|
| adaptive method | rendered_prior_0p1 |
| mean selector delta all PSNR | 0.06034847377870989 |
| mean selector delta middle PSNR | 0.07079248329613523 |
| positive selector all points | 7/12 |
| mean zlib rate delta MiB/frame | -7.57253900776373e-06 |
| mean zlib savings vs raw | 34.75185096242701% |
| max roundtrip abs diff | 0.0 |

### Zlib RD point observations

- Exact-uniform fallback points: driving gap4, meetroom gap4, n3dv gap4, n3dv gap8.
- Positive adaptive points: driving gap8/16, meetroom gap8/16, n3dv gap16, robot gap4/8.
- Only negative point: robot gap16, all PSNR `-0.01896 dB`, middle PSNR `-0.01899 dB`.
- Mean zlib rate is essentially unchanged and slightly lower than uniform on average.

### 结论

- Stage46 提供了第一版 actual-bitstream Adaptive Keyframe Selection RD 曲线。
- 该方法仍是 rendered-oracle calibrated selector，不是最终 feed-forward predictor，但已经显示 adaptive-or-uniform fallback 可以在真实 bitstream rate 下取得正平均 RD gain。
- 下一步需要把 `rendered_prior_0p1` 的 oracle cost 学成 feed-forward predictor，并在更大数据上训练/验证。

## 2026-06-26：阶段 47 Feed-Forward Rendered Cost Predictor Validation

### 执行计划

Stage47 使用 Stage44 rendered segment distortion labels 训练 leave-one-sample-out feed-forward segment cost predictor。Features 只包含 encoder-side 信息：segment length、normalized time、endpoint q8 Gaussian attribute differences、RGB endpoint/motion features。

### 新增文件

```text
scripts/run_stage47_rendered_cost_predictor_validation.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage47_rendered_cost_predictor_validation.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage47_rendered_cost_predictor_validation.py
```

Stage47 是 CPU-only regression validation。运行前仍按要求检查 `nvidia-smi`。

### 输出文件

```text
experiments/stage47_rendered_cost_predictor_validation/stage47_predictor_metrics.csv
experiments/stage47_rendered_cost_predictor_validation/stage47_predictor_predictions.csv
experiments/stage47_rendered_cost_predictor_validation/stage47_rendered_cost_predictor_validation_summary.json
```

### Aggregate 结果

| model | mean Spearman log | mean Pearson log | mean Pearson cost |
|---|---:|---:|---:|
| length_raw_log | 0.9705666891537404 | 0.4973834017016663 | 0.7248967616006826 |
| full_raw_log | 0.5683448234806368 | 0.4261146695061767 | 0.3921679030153338 |
| length_sample_z_rank | 0.9705666891537404 | 0.497383401701666 | 0.9516328490520204 |
| full_sample_z_rank | 0.9760206600953013 | 0.5200300873753056 | 0.9556364554249628 |
| full_sample_z_zlog | 0.516585818211339 | 0.557012364009426 | 0.47937910569214154 |

### 结论

- `full_sample_z_rank` 是当前最好的 feed-forward rendered-cost predictor candidate，mean Spearman log `0.9760`，mean Pearson cost `0.9556`。
- Length-only 仍然很强，说明 segment length 是 rendered distortion 的主导因素；Stage48 必须验证 predictor 是否能真正产生 RD gain，而不能只看预测相关性。
- Stage47 输出的 predictions 将用于 Stage48 fully feed-forward adaptive selector RD。

## 2026-06-26：阶段 48 Predicted Adaptive Selector RD

### 执行计划

Stage48 使用 Stage47 feed-forward predictions 驱动 DP keyframe selection。测试以下 variants：

- `length_raw_log_prior_0p1`
- `full_raw_log_prior_0p1`
- `length_sample_z_rank_prior_0p1`
- `full_sample_z_rank`
- `full_sample_z_rank_prior_0p1`
- `full_sample_z_rank_prior_0p3`

该阶段不使用 rendered oracle cost，是第一版 fully feed-forward predicted adaptive selector RD。

### 新增文件

```text
scripts/run_stage48_predicted_adaptive_selector_rd.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage48_predicted_adaptive_selector_rd.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage48_predicted_adaptive_selector_rd.py
```

### GPU 检查

运行前使用 `nvidia-smi`。GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 输出文件

```text
experiments/stage48_predicted_adaptive_selector_rd/stage48_predicted_adaptive_selector_rd.csv
experiments/stage48_predicted_adaptive_selector_rd/stage48_selector_comparison.csv
experiments/stage48_predicted_adaptive_selector_rd/stage48_selector_aggregates.csv
experiments/stage48_predicted_adaptive_selector_rd/stage48_predicted_adaptive_selector_rd_summary.json
experiments/stage48_predicted_adaptive_selector_rd/stage48_best_adapter_all_psnr_rd.png
experiments/stage48_predicted_adaptive_selector_rd/stage48_best_adapter_middle_psnr_rd.png
```

### Aggregate 结果

| method | positive all | mean delta all PSNR | min delta all PSNR | exact uniform points |
|---|---:|---:|---:|---:|
| full_raw_log_prior_0p1 | 0/12 | -0.21070429130013965 | -0.4152974337125741 | 4 |
| full_sample_z_rank | 0/12 | -1.051880916423393 | -1.7778923913320988 | 0 |
| full_sample_z_rank_prior_0p1 | 1/12 | -0.34686386133972275 | -1.7752712058738709 | 0 |
| full_sample_z_rank_prior_0p3 | 2/12 | -0.2135740809898404 | -1.690961134155529 | 0 |
| length_raw_log_prior_0p1 | 0/12 | -0.0409291686850605 | -0.3857680749438721 | 10 |
| length_sample_z_rank_prior_0p1 | 0/12 | -0.038394562647086815 | -0.3857680749438721 | 10 |

### 结论

- Stage48 是关键负结果：当前第一版 fully feed-forward predicted adaptive selector 尚未超过 uniform。
- Stage47 predictor 的 high correlation 没有转化为 RD gain，再次说明 selector objective 需要 decision-aware training 或更大规模数据。
- 目前 adaptive selector 可作为主创新路线继续推进，但严谨表述应区分：Stage46 是 rendered-oracle calibrated actual-bitstream RD 正平均收益；Stage48 deployable predicted selector 仍未成功。

## 2026-06-26：阶段 49 Extended Adaptive RD

### 执行计划

Stage49 扩展 adaptive RD 曲线点数和码率范围。使用：

```text
gap = 1, 2, 3, 4, 8, 16
method = uniform, rendered_prior_0p1
rate = estimated q8, actual raw q8, actual zlib q8
```

`gap1` 作为全关键帧高码率点进入 all-frame RD；middle-only RD 排除 `gap1`。

### 新增文件

```text
scripts/run_stage49_extended_adaptive_rd.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage49_extended_adaptive_rd.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage49_extended_adaptive_rd.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage49_extended_adaptive_rd.py --reuse_existing_csv
```

### GPU 检查

运行前使用 `nvidia-smi`。GPU 1 空闲，因此渲染阶段使用 `CUDA_VISIBLE_DEVICES=1`。`--reuse_existing_csv` 是 CPU-only finalize。

### 重要修正

- `gap1` 没有 middle frames，旧 evaluator 会计算 `None - None` 并报错；Stage49 增加 keyframe-only evaluation path，all-frame 可用、middle-only 自动跳过。
- Stage49 首次写 CSV 时漏了 `checkpoint` 字段，已修复。
- Stage49 comparison helper 也需要支持 `gap1` 的 empty middle，已改用支持 `None` 的 actual comparison helper。
- 为避免重复长时间渲染，Stage49 增加 `--reuse_existing_csv` finalize 模式，用已写出的 CSV 生成 comparison、summary 和 plots。

### 输出文件

仓库内：

```text
experiments/stage49_extended_adaptive_rd/stage49_extended_adaptive_rd.csv
experiments/stage49_extended_adaptive_rd/stage49_estimated_selector_comparison.csv
experiments/stage49_extended_adaptive_rd/stage49_raw_selector_comparison.csv
experiments/stage49_extended_adaptive_rd/stage49_zlib_selector_comparison.csv
experiments/stage49_extended_adaptive_rd/stage49_extended_adaptive_rd_summary.json
experiments/stage49_extended_adaptive_rd/stage49_estimated_adapter_all_psnr_rd.png
experiments/stage49_extended_adaptive_rd/stage49_estimated_adapter_middle_psnr_rd.png
experiments/stage49_extended_adaptive_rd/stage49_raw_adapter_all_psnr_rd.png
experiments/stage49_extended_adaptive_rd/stage49_raw_adapter_middle_psnr_rd.png
experiments/stage49_extended_adaptive_rd/stage49_zlib_adapter_all_psnr_rd.png
experiments/stage49_extended_adaptive_rd/stage49_zlib_adapter_middle_psnr_rd.png
```

外部 bitstreams：

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage49_extended_adaptive_rd
```

### Zlib Aggregate 结果

| metric | value |
|---|---:|
| all-frame points | 24 |
| middle-only points | 20 |
| positive all points | 11 |
| positive middle points | 11 |
| mean delta all PSNR | 0.05422388616846652 |
| mean delta middle PSNR | 0.08540786689496276 |
| min delta all PSNR | -0.018963597365331708 |
| min delta middle PSNR | -0.01899001734383532 |
| mean rate delta MiB/frame | -3.998343664604002e-06 |
| mean zlib savings vs raw | 34.76776063305336% |
| max roundtrip abs diff | 0.0 |

### 码率范围

- zlib `gap1` 高码率点约 `0.29-0.31 MiB/frame`。
- zlib `gap2` 约 `0.147-0.156 MiB/frame`。
- zlib `gap3` 约 `0.100-0.108 MiB/frame`。
- 现有低码率 `gap4/8/16` 保留。

### 结论

- Stage49 已解决“RD 点太少”和“只有极低码率点”的第一步问题：all-frame 6 点，middle-only 5 点。
- `rendered_prior_0p1` adaptive 在扩展码率范围下保持正平均收益，但许多高码率点与 uniform 相同或接近，说明高码率区主要提升来自更密 keyframes。
- 下一步需要 Stage50/51 支持 q10/q12/q16/fp16，进一步拓展高码率和更高 PSNR 区域。

## 2026-06-26：阶段 50 Multi-Bit Anchor Bitstream Prototype

### 执行计划

Stage50 将 `mono_dfcgs/anchor_bitstream.py` 从 q8-only 扩展到 q1-q16。当前实现：

- `bits <= 8`：uint8 payload
- `bits > 8`：uint16 payload
- 不做 bit-packing

该阶段读取 Stage49 selections，统计 q6/q8/q10/q12/q16 raw/zlib rate 和 decode roundtrip，不写出所有 bitstream 文件，避免外部目录过大。

### 修改/新增文件

```text
mono_dfcgs/anchor_bitstream.py
scripts/run_stage50_multibit_anchor_bitstream_prototype.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage50_multibit_anchor_bitstream_prototype.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage50_multibit_anchor_bitstream_prototype.py
```

Stage50 是 CPU-only bitstream prototype。运行前仍按要求检查 `nvidia-smi`。

### 输出文件

```text
experiments/stage50_multibit_anchor_bitstream_prototype/stage50_multibit_anchor_bitstream_prototype.csv
experiments/stage50_multibit_anchor_bitstream_prototype/stage50_multibit_anchor_bitstream_prototype_summary.json
```

### Aggregate by bits

| bits | payload dtype | mean raw MiB/frame | mean zlib MiB/frame | mean theoretical bitpacked MiB/frame | zlib savings | max roundtrip abs diff |
|---:|---|---:|---:|---:|---:|---:|
| 6 | uint8 | 0.1773370041424699 | 0.07782511427989783 | 0.1328336377164502 | 56.10827191340124% | 0.0 |
| 8 | uint8 | 0.17733660490188816 | 0.11566781190783303 | 0.17711151695526697 | 34.76602168045661% | 0.0 |
| 10 | uint16 | 0.35445224868958825 | 0.17832858230791024 | 0.22138939619408368 | 49.68088952479985% | 0.0 |
| 12 | uint16 | 0.3544541104180982 | 0.2018438725365482 | 0.2656672754329004 | 43.04832569609213% | 0.0 |
| 16 | uint16 | 0.3544558074255042 | 0.2372980050774083 | 0.35422303391053395 | 33.0456680053366% | 0.0 |

### 结论

- Multi-bit bitstream roundtrip 已验证，所有 bits 的 max abs diff vs direct dequantized quantization 都是 `0.0`。
- q10/q12/q16 现在能提供更高 zlib 码率点，适合 Stage51 生成 high-rate RD 曲线。
- q6 当前不是 bitpacked raw storage，因此 raw rate 与 q8 接近；但 theoretical bitpacked rate 已报告，后续如要做低码率 compact bitstream 需实现真正 bit-packing。

## 2026-06-26：阶段 51 High-Rate Multi-Bit RD

### 执行计划

Stage51 使用 Stage49 selections 和 Stage50 q8/q10/q12/q16 actual rates，重新渲染对应 bit-depth dequantized anchors 的 full-video quality，生成 high-rate RD 曲线。

### 新增文件

```text
scripts/run_stage51_high_rate_multibit_rd.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage51_high_rate_multibit_rd.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage51_high_rate_multibit_rd.py
```

### GPU 检查

运行前使用 `nvidia-smi`。GPU 1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

### 重要修正

首次 Stage51 渲染完成后写 CSV 失败，因为字段列表漏了 `estimated_q8_static_mib_per_frame`。已修复后重跑完成。

### 输出文件

```text
experiments/stage51_high_rate_multibit_rd/stage51_high_rate_multibit_rd.csv
experiments/stage51_high_rate_multibit_rd/stage51_raw_selector_comparison.csv
experiments/stage51_high_rate_multibit_rd/stage51_zlib_selector_comparison.csv
experiments/stage51_high_rate_multibit_rd/stage51_high_rate_multibit_rd_summary.json
experiments/stage51_high_rate_multibit_rd/stage51_uniform_zlib_all_psnr_rd.png
experiments/stage51_high_rate_multibit_rd/stage51_uniform_zlib_middle_psnr_rd.png
experiments/stage51_high_rate_multibit_rd/stage51_rendered_prior_0p1_zlib_all_psnr_rd.png
experiments/stage51_high_rate_multibit_rd/stage51_rendered_prior_0p1_zlib_middle_psnr_rd.png
```

### Uniform aggregate by bits

| bits | mean zlib MiB/frame | mean all PSNR | mean middle PSNR |
|---:|---:|---:|---:|
| 8 | 0.11566981107966534 | 28.42804749723145 | 26.686021035212185 |
| 10 | 0.17833053263585294 | 30.34450288854624 | 27.62072153468886 |
| 12 | 0.2018442118125191 | 30.68936631000986 | 27.73308939548495 |
| 16 | 0.23730023701348568 | 30.717664385570107 | 27.742100338359098 |

### Adaptive aggregate by bits

| bits | mean zlib MiB/frame | mean all PSNR | mean middle PSNR |
|---:|---:|---:|---:|
| 8 | 0.11566581273600074 | 28.48227138339992 | 26.771428902107147 |
| 10 | 0.17832663197996754 | 30.402345435674107 | 27.71204335713491 |
| 12 | 0.20184353326057725 | 30.748004954163406 | 27.82601646966399 |
| 16 | 0.237295773141331 | 30.776381462437445 | 27.83515966997416 |

### 结论

- Stage51 明确解决“高码率点缺失”和“PSNR 太一般”的一部分问题：q10/q12/q16 让 mean all PSNR 从 q8 约 `28.4 dB` 提升到 q16 约 `30.7 dB`。
- q12 到 q16 的提升已经很小，说明当前 quality 主要进入 anchor/model 上限区间，而不是纯量化瓶颈。
- Adaptive `rendered_prior_0p1` 在各 bit-depth 下平均仍略高于 uniform，且 zlib rate 基本一致或略低。

## 2026-06-26：阶段 52 FCGS/D-FCGS Baseline Preflight

### 执行计划

Stage52 解析本机已有 FCGS / D-FCGS 候选结果，生成 availability report、CSV records 和 summary JSON。该阶段只做文本/JSON 解析，不运行 GPU 渲染或训练。

### 新增文件

```text
scripts/run_stage52_fcgs_dfcgs_baseline_preflight.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage52_fcgs_dfcgs_baseline_preflight.py
```

### GPU 检查

运行前使用 `nvidia-smi`。GPU 上已有若干 Python 进程，但 Stage52 是 CPU-only 解析，未占用 CUDA。

### 输出文件

```text
experiments/stage52_fcgs_dfcgs_baseline_preflight/stage52_dfcgs_log_records.csv
experiments/stage52_fcgs_dfcgs_baseline_preflight/stage52_baseline_summary_records.csv
experiments/stage52_fcgs_dfcgs_baseline_preflight/stage52_fcgs_dfcgs_baseline_preflight_summary.json
experiments/stage52_fcgs_dfcgs_baseline_preflight/stage52_fcgs_dfcgs_baseline_preflight_report.md
```

### 解析结果

| item | value |
|---|---:|
| D-FCGS log records | 199 |
| D-FCGS log records with all key fields | 199 |
| baseline summary records | 199 |
| summary records with rate | 199 |
| summary records with input-video quality | 173 |
| Stage53 candidate summary records | 173 |

### Codec mode coverage

| codec mode | records |
|---|---:|
| `fcgs_i_plus_dfcgs_p_gop2` | 47 |
| `fcgs_per_frame` | 140 |
| `raw_i_plus_dfcgs_p_gop2` | 12 |

### 关键结论

- 本机已有足够的 FCGS / D-FCGS 候选 summary 用于 Stage53 生成 baseline comparison scaffold。
- 可直接优先使用 video-level GOP summary rows，而不是单个 D-FCGS P-frame logs。
- 这些 baseline 的 rate 是完整 FCGS/D-FCGS codec MiB/frame，包含 FCGS 或 raw I-frame 加 D-FCGS P-frame payload，不等同于 Mono-DFCGS 的 transmitted keyframe Gaussian anchor MiB/frame。
- `dummy_reference_images=true` 的 rows 不能用于 input-video PSNR/SSIM 主对比，只能保留 `codec_psnr` 作为 compression fidelity 诊断。

## 2026-06-26：阶段 53 Baseline Comparison Scaffold

### 执行计划

Stage53 将 Stage51 的 Mono-DFCGS rows 和 Stage52 的 FCGS/D-FCGS candidate rows 统一到一个 comparison scaffold。目标是生成字段完整、口径明确的对比表框架，而不是把当前外部 baseline 直接声明为 fair apples-to-apples 主结果。

### 新增文件

```text
scripts/run_stage53_baseline_comparison_scaffold.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage53_baseline_comparison_scaffold.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage53_baseline_comparison_scaffold.py
```

### GPU 检查

运行前使用 `nvidia-smi`。Stage53 是 CPU-only CSV/JSON 汇总，未占用 CUDA。

### 输出文件

```text
experiments/stage53_baseline_comparison_scaffold/stage53_baseline_comparison_scaffold.csv
experiments/stage53_baseline_comparison_scaffold/stage53_mono_dfcgs_rows.csv
experiments/stage53_baseline_comparison_scaffold/stage53_external_baseline_rows.csv
experiments/stage53_baseline_comparison_scaffold/stage53_method_sample_aggregate.csv
experiments/stage53_baseline_comparison_scaffold/stage53_baseline_comparison_scaffold_summary.json
experiments/stage53_baseline_comparison_scaffold/stage53_baseline_comparison_scaffold_report.md
```

### 结果摘要

| item | value |
|---|---:|
| unified rows | 391 |
| Mono-DFCGS rows | 192 |
| external baseline rows | 199 |
| external rows with input-video quality and rate | 173 |
| fair external apples-to-apples rows | 0 |

### Method coverage

| method | rows |
|---|---:|
| FCGS | 140 |
| FCGS-I + D-FCGS-P | 47 |
| Raw-I + D-FCGS-P | 12 |
| Mono-DFCGS uniform | 96 |
| Mono-DFCGS adaptive | 96 |

### 结论

- Stage53 已提供统一字段 scaffold，包含 method、variant、sample、protocol、rate unit、quality fields、local/literature status、fairness flag 和 notes。
- Mono-DFCGS rows 使用 `actual zlib Gaussian-anchor bitstream MiB/frame`，外部 baseline rows 使用 `full FCGS/D-FCGS codec MiB/frame`。
- 当前外部 baseline 的 `fair_local_run=false`，因为 rate scope、source Gaussian generation 和 protocol 仍不匹配；可作为 local protocol-reference，不应混入主 anchor-only RD 曲线而不标注。
- Adaptive Mono-DFCGS rows 仍标注为 rendered-oracle calibrated selector，不是最终 fully feed-forward selector。

## 2026-06-26：阶段 54 Decision-Aware Selector Analysis

### 执行计划

Stage54 复用已有 Stage48/49 RD 结果，不重新渲染。目标是诊断 fully feed-forward predicted selector 的负结果：当前问题是候选 adaptive layouts 本身缺乏上界，还是缺少一个 reliable adaptive-or-uniform decision policy。

### 新增文件

```text
scripts/run_stage54_decision_aware_selector_analysis.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage54_decision_aware_selector_analysis.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage54_decision_aware_selector_analysis.py
```

### GPU 检查

运行和 compile 前使用 `nvidia-smi`。Stage54 是 CPU-only CSV/JSON 分析，未占用 CUDA。

### 输出文件

```text
experiments/stage54_decision_aware_selector_analysis/stage54_decision_records.csv
experiments/stage54_decision_aware_selector_analysis/stage54_policy_choices.csv
experiments/stage54_decision_aware_selector_analysis/stage54_policy_summary.csv
experiments/stage54_decision_aware_selector_analysis/stage54_decision_aware_selector_analysis_summary.json
experiments/stage54_decision_aware_selector_analysis/stage54_decision_aware_selector_analysis_report.md
```

### 结果摘要

| item | value |
|---|---:|
| common points | 12 |
| candidate records | 72 |

### Policy Summary

| policy | mean all delta PSNR | positive all | min all delta | accepted adaptive |
|---|---:|---:|---:|---:|
| uniform | 0.0 | 0/12 | 0.0 | 0 |
| oracle_best_candidate_pool | 0.0063023570604731445 | 3/12 | 0.0 | 3 |
| oracle_layout_imitation | 0.0 | 0/12 | 0.0 | 0 |
| loocv_layout_threshold | -0.004029258659210555 | 1/12 | -0.05991704624753069 | 2 |
| fixed_length_raw_log_prior_0p1 | -0.0409291686850605 | 0/12 | -0.3857680749438721 | 12 |
| fixed_full_raw_log_prior_0p1 | -0.21070429130013965 | 0/12 | -0.4152974337125741 | 12 |
| fixed_length_sample_z_rank_prior_0p1 | -0.038394562647086815 | 0/12 | -0.3857680749438721 | 12 |
| fixed_full_sample_z_rank | -1.051880916423393 | 0/12 | -1.7778923913320988 | 12 |
| fixed_full_sample_z_rank_prior_0p1 | -0.34686386133972275 | 1/12 | -1.7752712058738709 | 12 |
| fixed_full_sample_z_rank_prior_0p3 | -0.2135740809898404 | 2/12 | -1.690961134155529 | 12 |

### 结论

- `oracle_best_candidate_pool` 在 Stage48 candidate layouts 加 uniform fallback 的上界也只有 `+0.0063 dB` mean all PSNR，说明当前 predicted candidate layout pool 本身很弱。
- `oracle_layout_imitation` 全部 fallback 到 uniform，说明简单模仿 Stage49 rendered-oracle layout 的 overlap/Jaccard 不是有效的 deployable decision rule。
- `loocv_layout_threshold` 仍为负平均收益，说明当前 12 个点样本太少且 layout-only threshold fallback 不够稳健。
- 下一步 selector 方向应优先做 decision-aware objective / adaptive-or-uniform classifier / DP-aware calibration，而不是只继续提升 segment-cost correlation。

## 2026-06-26：阶段 55 Large-Scale Data Preflight

### 执行计划

Stage55 做只读数据扩展 preflight。目标是明确下一步是否能直接扩展到 DAVIS / YouTube-VOS / RE10K / CO3D，并记录 StreamSplat 官方 provider 期望路径、Mono-DFCGS anchor export readiness、以及单目 codec claim 的使用边界。

### 新增文件

```text
scripts/run_stage55_large_scale_data_preflight.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage55_large_scale_data_preflight.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage55_large_scale_data_preflight.py
```

### GPU 检查

运行前使用 `nvidia-smi`。Stage55 是 CPU-only 文件系统 preflight，未占用 CUDA。

### 输出文件

```text
experiments/stage55_large_scale_data_preflight/stage55_root_preflight.csv
experiments/stage55_large_scale_data_preflight/stage55_sequence_preflight_sample.csv
experiments/stage55_large_scale_data_preflight/stage55_current_anchor_assets.csv
experiments/stage55_large_scale_data_preflight/stage55_protocol_matrix.csv
experiments/stage55_large_scale_data_preflight/stage55_large_scale_data_preflight_summary.json
experiments/stage55_large_scale_data_preflight/stage55_large_scale_data_preflight_report.md
```

### 结果摘要

| item | value |
|---|---:|
| root candidates checked | 20 |
| provider-layout-ready roots | 0 |
| anchor-export-ready roots | 0 |
| DAVIS/YouTube-VOS sampled sequence rows | 0 |
| current local anchor samples | 4 |

### Current Anchor Assets

| sample | gap1 pair rows | unique anchor frames | external paths exist |
|---|---:|---:|---:|
| driving | 80 | 81 | 80 |
| meetroom | 80 | 81 | 80 |
| n3dv | 80 | 81 | 80 |
| robot | 76 | 77 | 76 |

### 结论

- 默认路径下仍未发现可直接用于 StreamSplat provider 的 DAVIS / YouTube-VOS / RE10K / CO3D root。
- DAVIS 和 YouTube-VOS 是最适合 Mono-DFCGS 下一步大规模单目扩展的数据集，但需要用户先 mount/download，并生成 `depthImages/*_pred.png`。
- RE10K/CO3D 可以作为潜在 pretraining source，但不能在最终单目 codec claim 中使用多视角/camera 信息，除非先定义单目序列抽取协议。
- 当前 Stage33 dense anchors 覆盖 4 个本地样本，可继续用于 selector/adapter 开发，但不足以作为 paper-scale dataset evidence。

## 2026-06-26：阶段 51b Clean All-PSNR RD Plots

### 执行计划

Stage51b 不重跑任何渲染，只读取 Stage51 high-rate RD CSV，重画更清晰的 all-frame PSNR RD 曲线。新版图避免把 q 和 gap 全部画成同色折线：颜色表示 quant bits，点标签表示 reference gap。

### 新增文件

```text
scripts/run_stage51b_clean_all_psnr_rd_plots.py
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage51b_clean_all_psnr_rd_plots.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage51b_clean_all_psnr_rd_plots.py
```

### GPU 检查

运行前使用 `nvidia-smi`。Stage51b 是 CPU-only CSV plotting，未占用 CUDA。

### 输出文件

```text
experiments/stage51_high_rate_multibit_rd/stage51_clean_mean_all_psnr_rd.csv
experiments/stage51_high_rate_multibit_rd/stage51_clean_adaptive_delta_all_psnr.csv
experiments/stage51_high_rate_multibit_rd/stage51_clean_adaptive_all_psnr_by_bits.png
experiments/stage51_high_rate_multibit_rd/stage51_clean_uniform_all_psnr_by_bits.png
experiments/stage51_high_rate_multibit_rd/stage51_clean_mean_all_psnr_by_bits.png
experiments/stage51_high_rate_multibit_rd/stage51_clean_adaptive_delta_all_psnr_heatmap.png
experiments/stage51_high_rate_multibit_rd/stage51_clean_all_psnr_rd_plots_summary.json
```

### 结论

- 新图只展示 all-frame PSNR，符合后续默认汇报口径。
- `stage51_clean_adaptive_all_psnr_by_bits.png` 按样本展示 adaptive RD，每条线对应一个 q bit-depth，点标签为 gap。
- `stage51_clean_mean_all_psnr_by_bits.png` 给出 uniform 和 adaptive 的跨样本均值 RD。
- `stage51_clean_adaptive_delta_all_psnr_heatmap.png` 直接展示 adaptive 相比 uniform 的 all-frame PSNR 增益。
- `rendered_prior_0p1` 仍只能作为 oracle/calibrated selector 上限参考；最终 selector 必须改为 fully feed-forward。

## 2026-06-26：近期训练设计与创新点问答整理

### 执行计划

用户要求将最近关于训练过程、创新点、Stage3 PSNR 口径、RD 术语和未来 selector/side-information 设计的解释单独落盘。

### 新增文件

```text
logs/RECENT_TRAINING_INNOVATION_AND_QA_2026-06-26.md
```

### 记录内容

- Stage3 PSNR 为什么比当前 anchor-only pipeline 高。
- Stage3 使用 StreamSplat RGB/depth-conditioned reconstruction，不是严格 Gaussian-anchor-only 解码。
- Stage51 当前最佳 all-frame PSNR / rate 表和 clean RD 图路径。
- `bits`、`gap`、`zlib` 的含义。
- `uniform` 和 adaptive `rendered_prior_0p1` 的区别。
- 当前主码流传输内容：关键帧 quantized Gaussian anchors + metadata，不传 non-keyframe RGB/depth/Gaussian/motion/residual。
- 已完成训练：Gaussian adapter 小规模训练、leave-one-out、full-video anchor-only evaluation。
- 当前 selector 训练状态：oracle/calibrated 有正结果，fully feed-forward selector 未解决。
- 后续必须探索：fully feed-forward keyframe selection 和可计入码率的 optional non-keyframe side information。
- 当前创新点和主要限制。

## 2026-06-26：Stage56-68 计划落盘与 Stage Records 目录

### 执行计划

根据用户要求，将最新完整后续计划落盘，并建立后续每个 stage 的独立记录目录。后续每次 plan 完成后也需要整理记录。

### 新增文件

```text
logs/FUTURE_WORK_PLAN_STAGE56_68.md
logs/stage_records/README.md
logs/stage_records/00_previous_rounds_summary.md
```

### 结论

- Stage56-68 后续计划正式固定为三条核心贡献线：compression contribution、Gaussian adapter contribution、feed-forward keyframe selector contribution。
- Optional side information 作为增强线探索，所有传输信息必须计入 rate。
- 后续每个 stage 都需要在 `logs/stage_records/` 下记录过程、输出、结果、结论、caveat 和 commit。

## 2026-06-26：阶段 56 Protocol Lock

### 执行计划

Stage56 锁定后续实验协议和报告口径，为 compression、adapter、selector、side-info 后续贡献线提供统一评价标准。

### 新增文件

```text
scripts/run_stage56_protocol_lock.py
logs/stage_records/56_protocol_lock.md
```

### 运行命令

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage56_protocol_lock.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage56_protocol_lock.py
```

### GPU 检查

运行前使用 `nvidia-smi`。Stage56 是 CPU-only 协议生成，未占用 CUDA。

### 输出文件

```text
experiments/stage56_protocol_lock/stage56_protocol_lock_summary.json
experiments/stage56_protocol_lock/stage56_protocol_lock_report.md
experiments/stage56_protocol_lock/stage56_rate_accounting_rules.csv
experiments/stage56_protocol_lock/stage56_method_deployability_rules.csv
experiments/stage56_protocol_lock/stage56_standard_table_schemas.csv
```

### 结论

- 默认质量指标锁定为 all-frame PSNR。
- 主 rate 锁定为 transmitted Gaussian anchor bitstream MiB/frame。
- side information 如传输，必须报告 side-info rate 和 total rate。
- `rendered_prior_0p1` 锁定为 oracle/calibrated upper bound，不可作为最终 deployed selector claim。
- 最终 adaptive selector 必须是 frozen feed-forward predictor + deterministic selection/DP，不能使用 rendered oracle、PSNR lookahead 或 test-time reconstruction optimization。
- 强 adapter/selector claim 必须依赖 medium/long training，短训只作 smoke。

## 2026-06-26：Stage56-70 计划更新

### 执行计划

根据用户新要求，将 DAVIS 和 YouTube-VOS 下载/准备/预处理/anchor export 纳入后续大规模训练计划，然后继续 Stage57。

### 新增文件

```text
logs/FUTURE_WORK_PLAN_STAGE56_70.md
logs/stage_records/01_updated_plan_stage56_70.md
```

### 关键更新

- 新计划加入 DAVIS：`https://davischallenge.org/`。
- 新计划加入 YouTube-VOS：`https://youtube-vos.org/`。
- Stage59 变为 dataset download/prepare preflight。
- Stage60 变为 DAVIS/YouTube-VOS depth preprocessing。
- Stage61 变为 large-scale anchor export。
- Adapter / selector / side-info / final package 后移到 Stage62-70。

## 2026-06-26：阶段 57 Compact Anchor Codec

### 目标

实现真正 q1-q16 bit-packing，替代 Stage50 中 q6 用 uint8、q10/q12 用 uint16 的 storage prototype。

### 代码变更

- `mono_dfcgs/anchor_bitstream.py` 新增 bitpacked payload encode/decode，`encode_anchor_bitstream` 默认 `payload_encoding="bitpack"`。
- `payload_encoding="dtype"` 保留 legacy storage，方便 size ablation 和旧 header decode。
- `scripts/run_stage50_multibit_anchor_bitstream_prototype.py` 显式固定 `payload_encoding="dtype"`，保留历史 Stage50 语义。
- 新增 `scripts/run_stage57_compact_anchor_codec.py` 输出 legacy vs compact raw/zlib size table 和 roundtrip correctness。

### 验证

- 运行前已检查 `nvidia-smi`。
- `compileall` 通过。
- smoke：`n3dv/uniform/gap16`，q1/q6/q8/q10/q12/q16，`--verify_encodings all`，max roundtrip abs diff `0.0`。
- formal：4 samples、uniform、gap16、q1/q2/q4/q6/q8/q10/q12/q16，共 `32` rows，max roundtrip abs diff `0.0`。

### 输出

```text
experiments/stage57_compact_anchor_codec/stage57_compact_anchor_codec.csv
experiments/stage57_compact_anchor_codec/stage57_compact_anchor_codec_summary.json
logs/stage_records/57_compact_anchor_codec.md
```

### 关键结果

- q1 compact raw mean：`0.004331399600475631 MiB/frame`，legacy raw mean：`0.03433817985330386 MiB/frame`。
- q10 compact+zlib mean：`0.029996156310240045 MiB/frame`，legacy+zlib mean：`0.034567841089018385 MiB/frame`。
- q12 compact+zlib mean：`0.035456150641172995 MiB/frame`，legacy+zlib mean：`0.039116560122667454 MiB/frame`。
- q6 compact raw payload saving 为 `25%`，但 compact+zlib 比 legacy+zlib 更大，说明 bit-packing 后不一定更利于 generic zlib。

### 注意

全默认 coverage 和一个 128-row 代表性 run 在当前 CPU 时间限制下超时；Stage57 记录 compact codec correctness 和代表性 size trends，Stage58 再做 RD 集成和更有针对性的 ablation。

## 2026-06-26：阶段 58 Compression RD Ablation

### 目标

将 Stage57 compact anchor codec 接入 all-frame PSNR RD 报告，不重新渲染，复用 Stage51 all-frame PSNR 和 Stage57 actual compact rate。

### 代码

新增：

```text
scripts/run_stage58_compression_rd_ablation.py
```

### 执行

- 运行前已检查 `nvidia-smi`。
- `compileall` 通过。
- Stage58 仅读 CSV 和画图，不使用 CUDA。

### 输出

```text
experiments/stage58_compression_rd_ablation/stage58_compression_rd_ablation.csv
experiments/stage58_compression_rd_ablation/stage58_mean_compression_rd.csv
experiments/stage58_compression_rd_ablation/stage58_codec_summary.csv
experiments/stage58_compression_rd_ablation/stage58_actual_compact_savings.csv
experiments/stage58_compression_rd_ablation/stage58_actual_compact_savings_by_bits.csv
experiments/stage58_compression_rd_ablation/stage58_compression_rd_ablation_summary.json
experiments/stage58_compression_rd_ablation/stage58_full_mean_compression_rd.png
experiments/stage58_compression_rd_ablation/stage58_actual_compact_subset_rd.png
logs/stage_records/58_compression_rd_ablation.md
```

### 关键结果

- Stage51 input rows：`192`。
- Stage57 compact input rows：`32`。
- RD rows：`608`。
- Mean RD rows：`152`。
- Actual Stage57 compact subset 中 q10 compact+zlib 比 legacy dtype+zlib 平均节省 `13.21108787307025%` rate。
- Actual Stage57 compact subset 中 q12 compact+zlib 比 legacy dtype+zlib 平均节省 `9.3464515290433%` rate。
- q8/q16 基本持平，因 byte-aligned payload 本身不变，差异主要来自 v2 metadata/header。

### 限制

- `compact_bitpack_raw_payload_estimate` 是 payload-only estimate，不含 metadata/header。
- actual compact raw/zlib 目前只覆盖 Stage57 formal subset：uniform gap16 q8/q10/q12/q16 joined with Stage51 PSNR。

## 2026-06-26：阶段 59 DAVIS/YouTube-VOS Prepare Preflight

### 目标

执行 DAVIS 和 YouTube-VOS 下载/准备 preflight，检查 StreamSplat provider layout、候选数据 root、缺失项和下一步 checklist。

### 代码

新增：

```text
scripts/run_stage59_davis_vos_prepare_preflight.py
```

### 执行

- 运行前已检查 `nvidia-smi`。
- `compileall` 通过。
- Stage59 只做 filesystem/provider preflight，不下载数据、不使用 CUDA。

### 输出

```text
experiments/stage59_davis_vos_prepare_preflight/stage59_dataset_root_status.csv
experiments/stage59_davis_vos_prepare_preflight/stage59_expected_provider_layout.csv
experiments/stage59_davis_vos_prepare_preflight/stage59_streamsplat_provider_check.csv
experiments/stage59_davis_vos_prepare_preflight/stage59_download_prepare_checklist.csv
experiments/stage59_davis_vos_prepare_preflight/stage59_davis_vos_prepare_report.md
experiments/stage59_davis_vos_prepare_preflight/stage59_davis_vos_prepare_preflight_summary.json
logs/stage_records/59_davis_vos_prepare_preflight.md
```

### 结果

- DAVIS provider-ready roots：`0`。
- DAVIS anchor-export-ready roots：`0`。
- YouTube-VOS provider-ready roots：`0`。
- YouTube-VOS anchor-export-ready roots：`0`。
- StreamSplat `provider_davis.py` 存在。
- StreamSplat `provider_vos.py` 存在。

### 阻塞

Stage60 depth preprocessing 和 Stage61 large-scale anchor export 需要用户下载或挂载 DAVIS / YouTube-VOS 数据到 expected roots，或后续通过脚本参数提供等价 root。

## 2026-06-27：Stage59 后续数据下载和准备状态更新

### 执行背景

用户要求后续大规模训练纳入 DAVIS 和 YouTube-VOS。Stage59 初始 preflight 发现默认候选 root 均未 ready 后，继续尝试官方数据下载和准备。

### 已完成下载/解压

- DAVIS Full-Resolution unsupervised trainval 已下载并解压到 `/mnt/hdd2tC/tmp/opencode/datasets/DAVIS`。
- DAVIS zip `Content-Length` 为 `2957815900`，解压统计约 `12607` files / `2994587439` bytes uncompressed。
- DAVIS zip 已删除，避免占用额外空间。
- YouTube-VOS 2019 `valid.tar` 已通过 Google Drive id `1bw8KcpzfrT08HYbuROZmY0bp4TkYl4_g` 下载、解压到 `/mnt/hdd2tC/tmp/opencode/datasets/YouTube-VOS/valid`，随后删除 tar。
- `gdown` 已安装在 `/mnt/hdd2tC/tmp/opencode/streamsplat_venv`。

### 被空间阻塞的部分

- YouTube-VOS 2019 `train.tar` Google Drive id 为 `1lU9jCX-H0ntwh87tt2cA0xEPeWOJzD6S`。
- `gdown` 探测 `train.tar` 大小约 `9.26G`，而当前 `/mnt/hdd2tC` 剩余空间不足，未继续完整下载。
- partial `.part` 文件已删除。
- 当前数据集目录占用约：DAVIS `4.8G`，YouTube-VOS valid `1.3G`，downloads `4.0K`。

### 重跑 Stage59 状态

重跑命令：

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage59_davis_vos_prepare_preflight.py
```

最新结果：

- DAVIS provider-ready roots：`1`。
- DAVIS anchor-export-ready roots：`1`。
- YouTube-VOS provider-ready roots：`0`。
- YouTube-VOS anchor-export-ready roots：`0`。
- YouTube-VOS primary root 存在，但缺少 `train/JPEGImages` 和 `train/Annotations`。

## 2026-06-27：阶段 60 DAVIS Depth Preprocess

### 目标

实现并运行 DepthAnything V2 preprocessing，把 DAVIS / YouTube-VOS RGB frames 转为 StreamSplat provider 需要的 `*_pred.png` depth images。本次正式运行只覆盖 DAVIS，因为 YouTube-VOS train split 尚未下载完成。

### 代码

新增：

```text
scripts/run_stage60_depth_preprocess.py
```

脚本支持：

- DAVIS `JPEGImages/Full-Resolution/<sequence>/*.jpg` -> `depthImages/Full-Resolution/<sequence>/*_pred.png`。
- YouTube-VOS `train/valid/JPEGImages/<sequence>/*.jpg` -> `train/valid/depthImages/<sequence>/*_pred.png`。
- `--skip_existing`、`--max_frames`、`--datasets`、`--include_vos_train`、`--include_vos_valid`。
- StreamSplat `DepthAnythingWrapper`，默认 `model_name=vitl`、`target_short_side=518`。

### 执行

运行前按要求使用 `nvidia-smi` 检查 GPU。GPU1 基本空闲，因此 smoke 和 full DAVIS run 使用：

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage60_depth_preprocess.py --datasets davis --max_frames 2 --device cuda --log_every 1
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage60_depth_preprocess.py --datasets davis --device cuda --log_every 500
```

Stage60 收尾时又执行了脚本编译和 diff whitespace 检查：

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage59_davis_vos_prepare_preflight.py scripts/run_stage60_depth_preprocess.py
git diff --check
```

验证通过。Stage59/60 CSV writer 均固定 `lineterminator="\n"`，避免生成 CRLF 后被 `git diff --check` 标记为 trailing whitespace。

### 输出

仓库内只保存小型统计文件：

```text
experiments/stage60_depth_preprocess/stage60_depth_preprocess_frames.csv
experiments/stage60_depth_preprocess/stage60_depth_preprocess_summary.json
logs/stage_records/60_depth_preprocess.md
```

实际 depth PNG 写入 git 外部：

```text
/mnt/hdd2tC/tmp/opencode/datasets/DAVIS/depthImages/Full-Resolution/<sequence>/*_pred.png
```

### 结果

- DAVIS candidate frames：`6208`。
- Processed frames：`6206`。
- Skipped frames：`2`，来自 smoke run 已生成的 bear 前两帧。
- Failed frames：`0`。
- 重跑 Stage59 后，DAVIS 已 ready for Stage61 anchor export。

### 当前限制

- YouTube-VOS 只下载了 valid split；train split 因空间不足未下载。
- `/mnt/hdd2tC` 当前仅约 `4.1G` free，Stage61 大规模 anchor export 前必须先评估输出规模或释放空间。

## 2026-06-27：阶段 61 DAVIS Anchor Export Preflight And Smoke

### 目标

在 DAVIS 已 provider-ready / anchor-export-ready 后，先安全评估大规模 Gaussian anchor export 的空间需求，并验证 DAVIS RGB/depth 到 StreamSplat Gaussian anchor `.pt` 的导出闭环。

### 代码

新增：

```text
scripts/run_stage61_davis_anchor_export_preflight.py
scripts/run_stage61_davis_anchor_export.py
```

`run_stage61_davis_anchor_export_preflight.py` 只做 CPU filesystem/size preflight，不导出 `.pt`。`run_stage61_davis_anchor_export.py` 是 full-capable DAVIS anchor exporter，但默认限制为 `max_sequences=1` 和 `max_pairs_per_sequence=1`，防止误触发大规模输出。

### 执行

运行前按要求使用 `nvidia-smi` 检查 GPU。

Preflight command：

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage61_davis_anchor_export_preflight.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage61_davis_anchor_export_preflight.py
```

Smoke export command：

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage61_davis_anchor_export.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage61_davis_anchor_export.py --splits train --gaps 16 --max_sequences 1 --max_pairs_per_sequence 1 --batch_size 1 --device cuda:0
```

### 输出

Repository-tracked preflight outputs：

```text
experiments/stage61_davis_anchor_export_preflight/stage61_davis_sequences.csv
experiments/stage61_davis_anchor_export_preflight/stage61_davis_anchor_export_plan.csv
experiments/stage61_davis_anchor_export_preflight/stage61_davis_anchor_export_totals.csv
experiments/stage61_davis_anchor_export_preflight/stage61_davis_anchor_export_preflight_summary.json
experiments/stage61_davis_anchor_export_preflight/stage61_davis_anchor_export_preflight_report.md
```

Repository-tracked smoke outputs：

```text
experiments/stage61_davis_anchor_export/stage61_davis_anchor_export_manifest.csv
experiments/stage61_davis_anchor_export/stage61_davis_anchor_export_manifest.json
experiments/stage61_davis_anchor_export/stage61_davis_anchor_export_summary.csv
experiments/stage61_davis_anchor_export/stage61_davis_anchor_export_summary.json
```

External heavy output, not tracked by git：

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage61_davis_anchor_export/DAVIS/train/bear/gap16/pair_000000_000016.pt
```

### Preflight Results

- DAVIS ready sequences：`90 / 90`。
- DAVIS frames：`6208`。
- Estimated pair-pt output for gaps `[1, 2, 4, 8, 16]`：`21950.296875 MiB`。
- Estimated deduplicated static-anchor payload for the same gaps：`11386.4765625 MiB`。
- Current free space at heavy root mount：`4034.95703125 MiB`。
- Needed with `2048 MiB` reserve：`23998.296875 MiB`。
- Full all-gap export safe：`false`。

Key per-gap estimates：

| gap | pairs | pair-pt MiB | dedup static-anchor MiB |
|---:|---:|---:|---:|
| 1 | 6118 | 11184.46875 | 5674.5 |
| 2 | 3089 | 5647.078125 | 2905.8046875 |
| 4 | 1568 | 2866.5 | 1515.515625 |
| 8 | 807 | 1475.296875 | 819.9140625 |
| 16 | 425 | 776.953125 | 470.7421875 |

### Smoke Export Results

- Sequence：DAVIS `train/bear`。
- Gap：`16`。
- Exported pair：`0 -> 16`。
- Rows：`1`。
- Anchor tensor payload：`1.828125 MiB`。
- Gaussians per anchor：`36864`。
- External heavy root size after smoke：about `1.9M`.

### Gap16 Partial Large-Scale Export Update

After the smoke run, Stage61 preflight showed that DAVIS `gap16` alone was small enough to fit on the current mount, while all-gap export remained unsafe. A partial large-scale export was run with:

```text
CUDA_VISIBLE_DEVICES=4 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage61_davis_anchor_export.py --splits train val --gaps 16 --max_sequences 0 --max_pairs_per_sequence 0 --batch_size 2 --device cuda:0
```

Results:

- Splits：`train val`。
- Sequences：`90`.
- Gap：`16`。
- Exported pair rows：`425`。
- Anchor tensor payload：`776.953125 MiB`。
- Heavy root disk usage after export：about `781M`。
- Repository summary size：about `360K`。
- `/mnt/hdd2tC` free space after export：about `3.3G`。

Updated tracked outputs:

```text
experiments/stage61_davis_anchor_export/stage61_davis_anchor_export_manifest.csv
experiments/stage61_davis_anchor_export/stage61_davis_anchor_export_manifest.json
experiments/stage61_davis_anchor_export/stage61_davis_anchor_export_summary.csv
experiments/stage61_davis_anchor_export/stage61_davis_anchor_export_summary.json
```

External heavy outputs remain outside git under:

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage61_davis_anchor_export/DAVIS/{train,val}/<sequence>/gap16/*.pt
```

### 结论

- DAVIS 数据和 depth 已满足 anchor export 输入条件。
- Stage61 full all-gap export 当前被磁盘空间阻塞，不应在 `/mnt/hdd2tC` 只剩几 GiB 时启动。
- DAVIS-aware anchor export code path 已通过 smoke，并已完成 DAVIS train+val gap16 partial large-scale export。
- 后续若要导出 gap1/2/4/8 或 YouTube-VOS，需要先释放更多空间或提供其他外部挂载点。

## 2026-06-27：完整 DAVIS 数据下载与 /data 迁移计划

### 执行背景

用户要求查看其他磁盘空间，继续下载更完整 DAVIS 数据集，并暂时不处理 YouTube-VOS。随后用户要求使用新数据集继续后续 stages。

### 空间检查

- `/data`：约 `1.1T` free。
- `/mnt/hdd2tC`：约 `3.4G` free，继续作为代码仓库位置，不适合再放大数据。
- `/mnt/hdd2tB`：约 `54G` free。
- `/mnt/pool-sdd/containers/haocheng`：约 `208G` free。

### 下载结果

DAVIS 官方 Full-Resolution 数据已下载并解压：

```text
/data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS
```

zip 保留：

```text
/data/hctang/tmp/opencode/datasets/DAVIS_official_zips
```

已下载 zip：

- `DAVIS-2017-trainval-Full-Resolution.zip`
- `DAVIS-2017-test-dev-Full-Resolution.zip`
- `DAVIS-2017-test-challenge-Full-Resolution.zip`
- `DAVIS-2017-Unsupervised-trainval-Full-Resolution.zip`
- `DAVIS-2019-Unsupervised-test-dev-Full-Resolution.zip`
- `DAVIS-2019-Unsupervised-test-challenge-Full-Resolution.zip`
- `DAVIS-2017_semantics-Full-resolution.zip`
- `DAVIS-2017-scribbles-trainval.zip`

Zip integrity test passed for all 8 files.

### New Root Layout Summary

`/data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS` contains:

- `JPEGImages/Full-Resolution`
- `Annotations/Full-Resolution`
- `Annotations_unsupervised/Full-Resolution`
- `Annotations_semantics/Full-Resolution`
- `ImageSets/2017/{train,val,test-dev,test-challenge}.txt`
- `ImageSets/2019/{test-dev,test-challenge}.txt`
- `Scribbles`
- `categories.json`
- `davis_semantics.json`

Initial split statistics before depth generation:

| year | split | sequences | frames | depth | unsup masks | semi masks |
|---|---|---:|---:|---:|---:|---:|
| 2017 | train | 60 | 4209 | 0 | 4209 | 4209 |
| 2017 | val | 30 | 1999 | 0 | 1999 | 1999 |
| 2017 | test-dev | 30 | 2086 | 0 | 0 | 30 |
| 2017 | test-challenge | 30 | 2180 | 0 | 0 | 30 |
| 2019 | test-dev | 30 | 2294 | 0 | 0 | 0 |
| 2019 | test-challenge | 30 | 2229 | 0 | 0 | 0 |

### Updated Execution Plan

- Treat `/data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS` as the main DAVIS root for subsequent stages.
- Put new large anchors/checkpoints under `/data/hctang/tmp/opencode/mono_dfcgs_runs/`.
- Reuse existing train/val depth when safe, or run Stage60 depth preprocessing on `/data`.
- Export train/val multi-gap anchors on `/data` before Stage62 adapter training infra.
- Keep YouTube-VOS paused per user request.

## 2026-06-27：Stage61 /data DAVIS Train/Val All-Gap Anchor Export

### 目标

把新下载的 `/data` DAVIS official root 接入现有 Stage61 anchor export pipeline，并在 `/data` 上完成 DAVIS train/val gaps `1/2/4/8/16` 的大规模 Gaussian anchor export，为 Stage62 adapter training infra v2 提供大规模 anchor manifest。

### Depth Reuse

`/data` official root 初始没有 depth。比较 `/mnt/hdd2tC/tmp/opencode/datasets/DAVIS` 和 `/data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS` 的 2017 train/val split 后确认：

- train：`60` sequences，frame/depth/frame-name mismatch `0`。
- val：`30` sequences，frame/depth/frame-name mismatch `0`。

因此直接复制已生成的 train/val depth：

```text
cp -a /mnt/hdd2tC/tmp/opencode/datasets/DAVIS/depthImages /data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS/
```

复制后 `/data` DAVIS train/val depth readiness：

| split | sequences | frames | depth | ready |
|---|---:|---:|---:|---|
| train | 60 | 4209 | 4209 | true |
| val | 30 | 1999 | 1999 | true |

test-dev/test-challenge depth remains absent and is not used for Stage61 supervised train/val export.

### Code Update

`scripts/run_stage61_davis_anchor_export.py` 新增 `--skip_existing/--no-skip_existing`，默认 `true`。长时间导出第一次在 gap8 阶段触发 shell timeout；续跑时跳过已存在 `.pt` 并补齐剩余 pairs。脚本还优化了完全已存在的 sequence/gap，不再加载 RGB/depth arrays。

### Preflight

Command：

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage61_davis_anchor_export_preflight.py --davis_root /data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS --heavy_root /data/hctang/tmp/opencode/mono_dfcgs_runs/stage61_davis_anchor_export_full --summary_root experiments/stage61_davis_anchor_export_data_full_preflight
```

Result：

- ready sequences：`90 / 90`。
- frames：`6208`。
- free MiB at `/data` heavy root：`1059217.48828125`。
- needed MiB including 2GiB reserve：`23998.296875`。
- safe to full export：`true`。

### Full Export

Command：

```text
CUDA_VISIBLE_DEVICES=3 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage61_davis_anchor_export.py --davis_root /data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS --heavy_root /data/hctang/tmp/opencode/mono_dfcgs_runs/stage61_davis_anchor_export_full --summary_root experiments/stage61_davis_anchor_export_data_full --splits train val --gaps 1 2 4 8 16 --max_sequences 0 --max_pairs_per_sequence 0 --batch_size 2 --device cuda:0
```

The first run timed out after generating `11311` `.pt` files. Resume command with the same arguments completed the remaining pairs.

Final results：

- splits：`train val`。
- sequences：`90`。
- gaps：`1, 2, 4, 8, 16`。
- exported pair rows：`12007`。
- total anchor tensor payload：`21950.296875 MiB`。
- external heavy root disk usage：about `22G`。
- tracked summary/manifest size：about `8.7M`。
- `/data` free after export：about `1013G`。

Tracked outputs：

```text
experiments/stage61_davis_anchor_export_data_full_preflight/
experiments/stage61_davis_anchor_export_data_full/
```

External heavy outputs, not tracked by git：

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage61_davis_anchor_export_full/DAVIS/{train,val}/<sequence>/gap*/pair_*.pt
```

## 2026-06-27：Stage62 Adapter Training Infra v2 Smoke

### 目标

实现 Stage62 adapter training infrastructure v2，支持 DAVIS Stage61 manifest、train/val split、best checkpoint、resume state、外部 checkpoint root 和 storage-safe small summaries。

### 代码

新增：

```text
scripts/run_stage62_adapter_training_infra_v2.py
```

核心能力：

- 读取 Stage61 DAVIS manifest，并将 `dataset/split/sequence` 映射为 Stage21-compatible `sample`。
- 默认使用 `train` split 训练、`val` split 验证。
- 支持多 gap selection、balanced row sampling、q8 anchor input、RGB render loss。
- 保存 best/final adapter checkpoint 到 `/data`。
- 保存 latest training state `.pt` 以支持 resume。
- CSV writer 固定 LF，避免 `git diff --check` CRLF 问题。

### 执行

运行前按要求使用 `nvidia-smi` 检查 GPU。GPU1 空闲，因此使用：

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage62_adapter_training_infra_v2.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage62_adapter_training_infra_v2.py --device cuda --steps 4 --eval_interval 2 --frame_gaps 2 4 --max_train_rows_per_gap 1 --max_eval_rows_per_gap 1 --targets_per_row 1
```

Resume validation command：

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage62_adapter_training_infra_v2.py --device cuda --steps 5 --eval_interval 1 --frame_gaps 2 4 --max_train_rows_per_gap 1 --max_eval_rows_per_gap 1 --targets_per_row 1 --resume_state /data/hctang/tmp/opencode/mono_dfcgs_runs/stage62_adapter_training_infra_v2/stage62_latest_training_state.pt
```

### 输出

Tracked outputs：

```text
experiments/stage62_adapter_training_infra_v2/stage62_adapter_training_infra_v2_summary.json
experiments/stage62_adapter_training_infra_v2/stage62_train_rgb_losses.csv
experiments/stage62_adapter_training_infra_v2/stage62_validation_log.csv
experiments/stage62_adapter_training_infra_v2/stage62_selected_rows.csv
experiments/stage62_adapter_training_infra_v2/stage62_initial_eval.csv
experiments/stage62_adapter_training_infra_v2/stage62_final_eval.csv
experiments/stage62_adapter_training_infra_v2/stage62_best_eval.csv
experiments/stage62_adapter_training_infra_v2/stage62_best_gap_eval_summary.csv
```

External outputs：

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage62_adapter_training_infra_v2/stage62_best_anchor_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage62_adapter_training_infra_v2/stage62_final_anchor_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage62_adapter_training_infra_v2/stage62_latest_training_state.pt
```

### Results

- Available train rows for gaps `2,4`：`3103`。
- Available eval rows for gaps `2,4`：`1469`。
- Smoke selected train rows：`2`。
- Smoke selected eval rows：`2`。
- Train tasks：`2`。
- Eval tasks：`2`。
- Resume start step：`4`。
- Final/best step after resume：`5`。
- Best eval model PSNR avg：`23.256934739494863`。
- Linear PSNR avg：`23.252508732675636`。
- Best margin over linear：`0.004426006819226558 dB`。
- External checkpoint root size：about `2.0M`。

### 结论

Stage62 infrastructure is functional: DAVIS manifest reading, train/val split, q8 anchor input, RGB rendering loss, best checkpoint, final checkpoint, and resume state all work. This is a smoke/infrastructure validation, not a medium/long adapter quality result.

## 2026-06-27：Stage63 DAVIS Medium Adapter Training Pilot

### 目标

使用 Stage62 infrastructure 在 DAVIS all-gap anchors 上运行一个 medium-training pilot，验证更大 row/task 覆盖和更长 step 后的训练曲线，再决定是否进入 5k+ step 中长训。

### 执行

运行前按要求使用 `nvidia-smi` 检查 GPU。GPU1 空闲，因此使用：

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage62_adapter_training_infra_v2.py --device cuda --summary_root experiments/stage63_medium_adapter_training_pilot --heavy_root /data/hctang/tmp/opencode/mono_dfcgs_runs/stage63_medium_adapter_training_pilot --steps 128 --eval_interval 32 --frame_gaps 2 4 8 16 --max_train_rows_per_gap 4 --max_eval_rows_per_gap 2 --targets_per_row 1
```

### 输出

Tracked outputs：

```text
experiments/stage63_medium_adapter_training_pilot/
```

External outputs：

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage63_medium_adapter_training_pilot/stage62_best_anchor_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage63_medium_adapter_training_pilot/stage62_final_anchor_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage63_medium_adapter_training_pilot/stage62_latest_training_state.pt
```

### Results

- Available train rows：`3926`。
- Available eval rows：`1853`。
- Selected train rows：`16`。
- Selected eval rows：`8`。
- Steps：`128`。
- Best step：`128`。
- Initial eval model/linear PSNR：`20.90521679961685`。
- Final/best eval model PSNR：`20.93635879151969`。
- Best margin over linear：`0.031141991902842392 dB`。
- External checkpoint root size：about `2.0M`。

Validation curve：

| step | model PSNR | linear PSNR | margin over linear |
|---:|---:|---:|---:|
| 0 | 20.90521679961685 | 20.90521679961685 | 0.0 |
| 32 | 20.913996462661025 | 20.90521679961685 | 0.008779663044176544 |
| 64 | 20.92152115695736 | 20.90521679961685 | 0.01630435734051261 |
| 96 | 20.929940697543877 | 20.90521679961685 | 0.02472389792702856 |
| 128 | 20.93635879151969 | 20.90521679961685 | 0.031141991902842392 |

Gap-wise best margins：

| gap | margin over linear |
|---:|---:|
| 2 | 0.04881351193215622 |
| 4 | 0.03418816116420231 |
| 8 | 0.024005080795515 |
| 16 | 0.017561213719488933 |

### 结论

Stage63 pilot shows a small but monotonic validation gain across all tested gaps. It remains a pilot with only 128 steps and 8 eval tasks; it supports running a longer Stage63/65 training job, but should not be presented as the final medium/long adapter result.

## 2026-06-27：Stage64 Adapter Architecture / Teacher Study

### 目标

在进入更长 medium/long adapter training 前，先做小型 adapter architecture / supervision ablation：比较 RGB render loss 和 dense-gap1 anchor teacher distillation，并分别测试 hidden dim `128` 和 `256`。

Dense-gap1 anchors 只作为 offline teacher target，不改变 codec test-time 输入。Stage64 所有 variant 的 test-time 输入仍为 q8 endpoint Gaussian anchors plus timestamp。

### 执行

运行前按要求使用 `nvidia-smi` 检查 GPU。GPU1/GPU3/GPU4/GPU5 空闲，因此使用 GPU1。

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage64_adapter_teacher_study.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage64_adapter_teacher_study.py --device cuda --steps 48 --eval_interval 24 --frame_gaps 2 4 8 16 --max_train_rows_per_gap 4 --max_eval_rows_per_gap 2 --targets_per_row 1
```

### 输出

Tracked outputs：

```text
experiments/stage64_adapter_teacher_study/stage64_adapter_teacher_study_summary.json
experiments/stage64_adapter_teacher_study/stage64_variant_summary.csv
experiments/stage64_adapter_teacher_study/stage64_validation_log.csv
experiments/stage64_adapter_teacher_study/stage64_train_log.csv
experiments/stage64_adapter_teacher_study/stage64_best_eval_rows.csv
```

External checkpoints：

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage64_adapter_teacher_study/<variant>/best_adapter.safetensors
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage64_adapter_teacher_study/<variant>/final_adapter.safetensors
```

Output sizes：

| path | size |
|---|---:|
| `experiments/stage64_adapter_teacher_study` | `48K` |
| `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage64_adapter_teacher_study` | `7.8M` |

### Results

- Available train rows：`3926`。
- Available eval rows：`1853`。
- Selected train rows：`16`。
- Selected eval rows：`8`。
- Train tasks：`16`。
- Eval tasks：`8`。
- Steps：`48`。
- Eval interval：`24`。

Variant summary：

| variant | loss | hidden dim | params | best step | best margin over linear PSNR | best teacher MSE |
|---|---|---:|---:|---:|---:|---:|
| `rgb_h128` | RGB render loss | `128` | `102925` | `48` | `+0.012173706030985443 dB` | `0.005200807470828295` |
| `rgb_h256` | RGB render loss | `256` | `402445` | `48` | `+0.017721457863302703 dB` | `0.005200803148909472` |
| `teacher_h128` | dense gap1 teacher | `128` | `102925` | `48` | `+0.002455631876273401 dB` | `0.005200357045396231` |
| `teacher_h256` | dense gap1 teacher | `256` | `402445` | `48` | `+0.005105871063040723 dB` | `0.005198333790758625` |

Best by rendered PSNR：`rgb_h256`，margin over linear `+0.017721457863302703 dB`。

Best by teacher-anchor MSE：`teacher_h256`，teacher MSE `0.005198333790758625`。

### 结论

- 短训下，hidden dim `256` 明显好于 `128`。
- RGB render loss route 的 rendered PSNR 优于 teacher route，适合作为 Stage65 medium run 的首选配置。
- Teacher distillation route 能更明显降低 dense-anchor teacher MSE，但 48-step ablation 没有转化为最佳 rendered PSNR。
- Stage64 仍是小型 architecture/supervision ablation，不作为最终 medium/long adapter claim。

## 2026-06-27：Stage65 RGB-H256 Medium Adapter Training

### 目标

基于 Stage64 的 architecture / supervision ablation 结论，选择 rendered PSNR 最好的 `rgb_h256` route，执行更长的 RGB render-loss adapter training。

Stage65 仍保持 test-time 输入为 q8 endpoint Gaussian anchors plus timestamp；不把 dense teacher anchors 作为 transmitted side information。

### Sanity Run

运行前按要求使用 `nvidia-smi` 检查 GPU。GPU1 空闲，因此使用：

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage64_adapter_teacher_study.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage64_adapter_teacher_study.py --device cuda --summary_root experiments/stage65_rgb_h256_medium_sanity --heavy_root /data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_sanity --steps 1024 --eval_interval 256 --frame_gaps 2 4 8 16 --max_train_rows_per_gap 16 --max_eval_rows_per_gap 4 --targets_per_row 1 --variants rgb_h256
```

Sanity result：

- Train/eval tasks：`64 / 16`。
- Best step：`512`。
- Linear PSNR：`19.655442148062438`。
- Best model PSNR：`19.70705320602079`。
- Best margin over linear：`+0.05161105795835397 dB`。
- Final margin over linear：`+0.04578667635357192 dB`。

The sanity run stayed positive without severe validation collapse, so Stage65 continued to the longer medium run.

### Medium Run

运行前再次使用 `nvidia-smi` 检查 GPU。GPU1 仍空闲，因此使用：

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage64_adapter_teacher_study.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage64_adapter_teacher_study.py --device cuda --summary_root experiments/stage65_rgb_h256_medium_training --heavy_root /data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training --steps 5000 --eval_interval 500 --frame_gaps 2 4 8 16 --max_train_rows_per_gap 32 --max_eval_rows_per_gap 8 --targets_per_row 1 --variants rgb_h256
```

Medium result：

- Train/eval tasks：`128 / 32`。
- Parameter count：`402445`。
- Best step：`4000`。
- Linear PSNR：`18.518044832601554`。
- Best model PSNR：`18.79151802978449`。
- Best margin over linear：`+0.2734731971829376 dB`。
- Final margin over linear：`+0.21907000507035335 dB`。

Validation curve：

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

Gap-wise best-step margins：

| gap | margin over linear |
|---:|---:|
| 2 | `+0.24356722157098484` |
| 4 | `+0.26732412251510996` |
| 8 | `+0.3257420419967144` |
| 16 | `+0.2572594026489324` |

### 输出

Tracked outputs：

```text
experiments/stage65_rgb_h256_medium_sanity/
experiments/stage65_rgb_h256_medium_training/
```

External checkpoints：

```text
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_sanity/rgb_h256/{best_adapter.safetensors,final_adapter.safetensors}
/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training/rgb_h256/{best_adapter.safetensors,final_adapter.safetensors}
```

Output sizes：

| path | size |
|---|---:|
| `experiments/stage65_rgb_h256_medium_sanity` | `92K` |
| `experiments/stage65_rgb_h256_medium_training` | `372K` |
| `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_sanity` | `3.1M` |
| `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage65_rgb_h256_medium_training` | `3.1M` |

### 结论

- Stage65 `rgb_h256` medium run 显著扩大了 Stage64/Stage65 sanity 的 validation gain，best margin 达到 `+0.2734731971829376 dB`。
- Best checkpoint selection 很重要：step `4000` 最好，step `5000` 仍为正但已回落。
- Best-step 下 gaps `2/4/8/16` 全部为正，gap `8` 在该 eval subset 上 gain 最大。
- RGB-only route 会显著恶化 dense-anchor teacher MSE，因此 teacher-anchor MSE 不能作为该 route 的模型选择指标。
- Stage65 仍是 intermediate eval-task training 结果，不是最终 all-frame RD evaluation。

## 2026-06-27：Stage66 DAVIS Feed-Forward Selector Dataset

### 目标

构建 DAVIS feed-forward selector segment-cost dataset，为 Stage67 selector predictor / deterministic DP training 做准备。

Stage66 只生成数据集，不做最终 selector quality claim。Features 限制为 encoder-side 信息：segment metadata、endpoint Gaussian-anchor statistics 和 original RGB motion statistics。Labels 使用 Stage65 best `rgb_h256` adapter 和 dense gap1 anchors 离线生成，不作为 test-time input。

### 代码

新增：

```text
scripts/run_stage66_davis_feedforward_selector_dataset.py
```

### 执行

运行前按要求使用 `nvidia-smi` 检查 GPU。GPU1 空闲，因此使用：

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage66_davis_feedforward_selector_dataset.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage66_davis_feedforward_selector_dataset.py --device cuda
```

### 输出

Tracked outputs：

```text
experiments/stage66_davis_feedforward_selector_dataset/stage66_davis_selector_dataset.csv
experiments/stage66_davis_feedforward_selector_dataset/stage66_davis_selector_sequence_summary.csv
experiments/stage66_davis_feedforward_selector_dataset/stage66_davis_selector_feature_correlations.csv
experiments/stage66_davis_feedforward_selector_dataset/stage66_davis_selector_dataset_summary.json
```

Output size：`2.4M`。

### Results

- Selected sequences：`12`。
- Train sequences：`8`。
- Eval sequences：`4`。
- Dataset rows：`4608`。
- Train rows：`3072`。
- Eval rows：`1536`。
- Max segment length：`16`。
- Max segments per sequence：`384`。
- Adapter-better-than-linear in anchor space：`0 / 4608`。
- Mean adapter anchor MSE label：`0.031690189455081855`。
- Mean linear anchor MSE label：`0.011579127648414848`。

Strongest all-scope feature correlations with `log10(adapter_anchor_mse_mean)`：

| feature | Pearson |
|---|---:|
| `endpoint_anchor_l1` | `0.7819601460791495` |
| `endpoint_rgb_mse` | `0.7630952706790444` |
| `endpoint_anchor_mse` | `0.7528988639175856` |
| `rgb_motion_max` | `0.7304595140192872` |
| `rgb_motion_mean` | `0.7021857588764513` |

### 结论

- Stage66 建立了 DAVIS selector dataset 的最小闭环：encoder-side features + offline labels。
- Endpoint-anchor/RGB-motion features 对 anchor-space adapter error 有明显相关性，说明可训练 feed-forward segment-cost predictor。
- 但 Stage65 RGB adapter 在 anchor-space MSE 上全段都差于 linear，这再次说明 dense-anchor MSE 不是 rendered PSNR 的可靠代理。
- Stage67 若直接训练 selector predictor，应明确它是 anchor-space difficulty proxy；更强路线是在 Stage67/68 增加 rendered-distortion label subset，再做 deployable selector RD。

## 2026-06-27：Stage67 DAVIS Selector Predictor Training

### 目标

在 Stage66 DAVIS selector dataset 上训练和验证 feed-forward segment-cost predictor。该阶段只预测 Stage66 offline anchor-space proxy label，不做 selector RD claim。

### 代码

新增：

```text
scripts/run_stage67_davis_selector_predictor_training.py
```

### 执行

运行前按要求使用 `nvidia-smi` 检查 GPU。Stage67 是 CPU ridge training，但仍完成 GPU 状态检查。

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage67_davis_selector_predictor_training.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage67_davis_selector_predictor_training.py
```

### Results

| model | features | eval RMSE log | eval Pearson | eval Spearman |
|---|---:|---:|---:|---:|
| `length_only_ridge` | `2` | `0.03748165925098028` | `0.3094398002609641` | `0.3391438570632329` |
| `rgb_motion_ridge` | `7` | `0.032271284148152724` | `0.6528770726740828` | `0.6576988074032709` |
| `anchor_endpoint_ridge` | `15` | `0.020229501422918364` | `0.8664301098563049` | `0.8671254647839715` |
| `full_feature_ridge` | `18` | `0.01746532908957139` | `0.9103771172408511` | `0.9017146144293104` |

Best model：`full_feature_ridge`。

Rows：train `3072`，eval `1536`。

### 输出

Tracked outputs：

```text
experiments/stage67_davis_selector_predictor_training/stage67_selector_predictor_metrics.csv
experiments/stage67_davis_selector_predictor_training/stage67_selector_predictor_predictions.csv
experiments/stage67_davis_selector_predictor_training/stage67_selector_predictor_model_params.json
experiments/stage67_davis_selector_predictor_training/stage67_selector_predictor_training_summary.json
```

Output size：`3.2M`。

### 结论

- Full-feature ridge 能较好预测 Stage66 anchor-space proxy label，eval Spearman `0.9017146144293104`。
- Endpoint-anchor features 明显强于 length-only / RGB-motion-only。
- Stage67 仍不是最终 selector RD；Stage68 需要 deterministic DP selection 和 rendered/full-video validation。

## 2026-06-27：Stage68 DAVIS Feed-Forward Selector Rendered Validation

### 目标

使用 Stage67 `full_feature_ridge` feed-forward cost predictor，对 Stage66 DAVIS eval sequences 做 deterministic DP keyframe selection，并用 Stage65 best `rgb_h256` adapter 做 rendered validation。

Selection 过程不使用 rendered oracle、PSNR labels、dense-anchor labels 或 reconstruction lookahead。

### 代码

新增：

```text
scripts/run_stage68_davis_feedforward_selector_rendered_validation.py
```

### 执行

运行前按要求使用 `nvidia-smi` 检查 GPU。GPU1 空闲，因此使用：

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage68_davis_feedforward_selector_rendered_validation.py
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage68_davis_feedforward_selector_rendered_validation.py --device cuda
```

### 输出

Tracked outputs：

```text
experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_rendered_validation.csv
experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_comparison.csv
experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_selections.csv
experiments/stage68_davis_feedforward_selector_rendered_validation/stage68_davis_selector_rendered_validation_summary.json
```

Output size：`28K`。

### Results

Aggregate all-frame PSNR selector result：

- Comparison points：`12`。
- Positive adapter all-frame PSNR points：`7`。
- Mean selector delta adapter all-frame PSNR：`+0.030738190041048163 dB`。
- Positive linear all-frame PSNR points：`7`。
- Mean selector delta linear all-frame PSNR：`+0.02925622579667782 dB`。

Per-point adapter all-frame PSNR delta：

| sample | gap | predicted - uniform all PSNR |
|---|---:|---:|
| `DAVIS/val/bmx-trees` | `4` | `+0.06357661189149155` |
| `DAVIS/val/bmx-trees` | `8` | `-0.03615984112334303` |
| `DAVIS/val/bmx-trees` | `16` | `0.0` |
| `DAVIS/val/car-shadow` | `4` | `-0.000555582328463089` |
| `DAVIS/val/car-shadow` | `8` | `-0.006733966014870418` |
| `DAVIS/val/car-shadow` | `16` | `+0.04549131375734561` |
| `DAVIS/val/goat` | `4` | `+0.0055252224122490645` |
| `DAVIS/val/goat` | `8` | `-0.10978492809701024` |
| `DAVIS/val/goat` | `16` | `+0.02716329180723065` |
| `DAVIS/val/soapbox` | `4` | `+0.03415602364444936` |
| `DAVIS/val/soapbox` | `8` | `+0.08340300738787576` |
| `DAVIS/val/soapbox` | `16` | `+0.26277712715562274` |

### 结论

- Stage68 完成了 DAVIS eval subset 上的 fully feed-forward deterministic-DP selector rendered validation。
- 结果平均为正，但不稳定：`7/12` 点为正，`5/12` 点非正。
- `goat gap8` 是明显负例，说明 anchor-space proxy predictor 仍与 rendered all-frame objective 存在 mismatch。
- 下一步需要 rendered-distortion labels 或 fallback calibration，再做更稳健的 selector validation。

## 2026-06-27：Stage69 Selector Fallback Calibration Analysis

### 目标

分析 Stage68 mixed-positive selector 结果能否通过简单 fallback-to-uniform policy 稳定下来。Stage69 不重渲染，复用 Stage68 rendered outcomes 作为离线 calibration labels。

### 代码

新增：

```text
scripts/run_stage69_selector_fallback_calibration.py
```

### 执行

运行前按要求使用 `nvidia-smi` 检查 GPU。Stage69 是 CPU analysis，但仍完成 GPU 状态检查。

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage69_selector_fallback_calibration.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage69_selector_fallback_calibration.py
```

### Results

| policy | accepted predicted | positive points | mean all PSNR delta | min all PSNR delta | status |
|---|---:|---:|---:|---:|---|
| `uniform` | `0 / 12` | `0 / 12` | `0.0` | `0.0` | deployable baseline |
| `fixed_predicted` | `12 / 12` | `7 / 12` | `+0.030738190041048163` | `-0.10978492809701024` | deployable but unstable |
| `oracle_positive_fallback` | `7 / 12` | `7 / 12` | `+0.04350771650468873` | `0.0` | oracle upper bound, not deployable |
| `same_data_threshold_fallback` | `5 / 12` | `5 / 12` | `+0.03745316604097404` | `0.0` | same-data upper bound, not deployable |
| `loocv_threshold_fallback` | `4 / 12` | `1 / 12` | `-0.01170162890067535` | `-0.10978492809701024` | deployable-style small-sample calibration |

### 输出

Tracked outputs：

```text
experiments/stage69_selector_fallback_calibration/stage69_selector_decision_records.csv
experiments/stage69_selector_fallback_calibration/stage69_selector_policy_choices.csv
experiments/stage69_selector_fallback_calibration/stage69_selector_policy_summary.csv
experiments/stage69_selector_fallback_calibration/stage69_selector_fallback_calibration_summary.json
```

Output size：`28K`。

### 结论

- Oracle/same-data fallback 说明避开负点有潜在收益，但不是 deployable claim。
- Leave-one-sequence-out threshold fallback 在当前 4 个 eval sequences 上反而变差，mean all PSNR delta `-0.01170162890067535 dB`。
- 简单 layout/cost threshold 不足以稳定 selector；下一步需要更多 rendered labels 或 decision-aware fallback classifier。

## 2026-06-27：Stage70 Scoped DAVIS RD Package

### 目标

将当前 Stage68/69 DAVIS eval-subset 结果整理为 scoped RD package：rate table、all-frame PSNR table、selector delta table、method summary、baseline status 和 RD curve。

Stage70 不重渲染，不声明最终完整 benchmark。

### 代码

新增：

```text
scripts/run_stage70_scoped_davis_rd_package.py
```

### 执行

运行前按要求使用 `nvidia-smi` 检查 GPU。Stage70 是 report/plot generation。

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage70_scoped_davis_rd_package.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage70_scoped_davis_rd_package.py
```

### 输出

Tracked outputs：

```text
experiments/stage70_scoped_davis_rd_package/stage70_rate_table.csv
experiments/stage70_scoped_davis_rd_package/stage70_all_psnr_table.csv
experiments/stage70_scoped_davis_rd_package/stage70_selector_delta_table.csv
experiments/stage70_scoped_davis_rd_package/stage70_method_summary.csv
experiments/stage70_scoped_davis_rd_package/stage70_baseline_status.csv
experiments/stage70_scoped_davis_rd_package/stage70_scoped_davis_rd_curve.png
experiments/stage70_scoped_davis_rd_package/stage70_scoped_davis_rd_package_summary.json
```

Output size：`408K`。

### Results

Mean all-frame PSNR summary：

| method | selector | gap | mean rate MiB/frame | mean all PSNR |
|---|---|---:|---:|---:|
| `linear_anchor` | `uniform` | `4` | `0.12188942649147727` | `20.137413494810616` |
| `linear_anchor` | `uniform` | `8` | `0.06551069779829546` | `18.063459460774446` |
| `linear_anchor` | `uniform` | `16` | `0.03811479048295455` | `16.59656669447801` |
| `stage65_rgb_h256_adapter` | `uniform` | `4` | `0.12188942649147727` | `20.608531857721726` |
| `stage65_rgb_h256_adapter` | `uniform` | `8` | `0.06551069779829546` | `18.54248864942665` |
| `stage65_rgb_h256_adapter` | `uniform` | `16` | `0.03811479048295455` | `17.01303254555753` |
| `stage65_rgb_h256_adapter` | `predicted_full_feature_dp` | `4` | `0.12188942649147727` | `20.634207426626656` |
| `stage65_rgb_h256_adapter` | `predicted_full_feature_dp` | `8` | `0.06551069779829546` | `18.525169717464813` |
| `stage65_rgb_h256_adapter` | `predicted_full_feature_dp` | `16` | `0.03811479048295455` | `17.096890478737578` |

Selector aggregate：`7 / 12` positive，mean adapter all-frame PSNR delta `+0.030738190041048163 dB`。

Baseline status：FCGS/D-FCGS/CWGS are not yet locally evaluated apples-to-apples.

### 结论

- Stage70 完成当前 scoped DAVIS RD package。
- 该 package 可用于内部进度汇总和展示，但不能作为最终 benchmark。
- 最终报告仍需要补 FCGS/D-FCGS local fair baselines、selector robustness 和更完整数据覆盖。

## 2026-06-27：Stage71 Baseline Availability Preflight

### 目标

盘点本机 FCGS/D-FCGS/CWGS baseline 可用性、旧结果是否能接入 DAVIS scoped RD comparison、以及距离 apples-to-apples local DAVIS baseline 还缺哪些字段/步骤。

Stage71 不运行重型训练或渲染，不产出大文件。

### 计划输出

```text
scripts/run_stage71_baseline_availability_preflight.py
experiments/stage71_baseline_availability_preflight/stage71_baseline_code_inventory.csv
experiments/stage71_baseline_availability_preflight/stage71_baseline_artifact_inventory.csv
experiments/stage71_baseline_availability_preflight/stage71_baseline_fairness_status.csv
experiments/stage71_baseline_availability_preflight/stage71_baseline_missing_fields.csv
experiments/stage71_baseline_availability_preflight/stage71_baseline_availability_preflight_summary.json
experiments/stage71_baseline_availability_preflight/stage71_baseline_availability_preflight_report.md
```

### 执行前状态

- `git status --short`：clean。
- `/data` free：约 `1010G`。
- `/mnt/hdd2tC` free：约 `3.7G`，仍只适合源码和小 summary。
- 已按要求运行 `nvidia-smi`；GPU 上有其它任务，因此 Stage71 保持 CPU-only 轻量扫描。

### 预期结论口径

- 若旧 FCGS/D-FCGS artifacts 数据集、rate scope 或质量口径与 DAVIS scoped eval 不一致，只作为 diagnostic/reference，不作为 final apples-to-apples baseline。
- 后续若要形成 final comparison，应新增或改造 baseline runner，使输入、frame set、rate accounting 和 all-frame PSNR 口径与 Stage70 DAVIS package 对齐。

### 代码

新增：

```text
scripts/run_stage71_baseline_availability_preflight.py
```

### 执行

运行前按要求重新使用 `nvidia-smi` 检查 GPU。GPU 上有其它任务，Stage71 是 CPU-only 文件扫描。

```text
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage71_baseline_availability_preflight.py
/mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage71_baseline_availability_preflight.py
```

### 输出

Tracked outputs：

```text
experiments/stage71_baseline_availability_preflight/stage71_baseline_code_inventory.csv
experiments/stage71_baseline_availability_preflight/stage71_baseline_artifact_inventory.csv
experiments/stage71_baseline_availability_preflight/stage71_baseline_fairness_status.csv
experiments/stage71_baseline_availability_preflight/stage71_baseline_missing_fields.csv
experiments/stage71_baseline_availability_preflight/stage71_baseline_availability_preflight_summary.json
experiments/stage71_baseline_availability_preflight/stage71_baseline_availability_preflight_report.md
```

Output size：`32K`。

### Results

- DAVIS root exists：`true`。
- Stage61 full DAVIS manifest exists：`true`。
- Stage70 scoped samples：`DAVIS/val/bmx-trees`, `DAVIS/val/car-shadow`, `DAVIS/val/goat`, `DAVIS/val/soapbox`。
- Stage70 gaps：`4`, `8`, `16`。
- Fairness-ready methods：none。
- Not-ready methods：`FCGS`, `D-FCGS`, `CWGS`。
- High-priority missing counts：FCGS `5`, D-FCGS `5`, CWGS `3`。

Code inventory：

| method | code status | DAVIS mentions | checkpoints | missing tools |
|---|---|---:|---:|---|
| `FCGS` | `present` | `0` | `5` | `tmc3` |
| `D-FCGS` | `present` | `0` | `1` | `tmc3` |
| `CWGS` | `missing_optional` | `0` | `0` | `code_checkout` |

Artifact inventory：

| group | records | DAVIS-related | rate rows | quality rows | fair rows |
|---|---:|---:|---:|---:|---:|
| `stage52_fcgs_dfcgs_summary_records` | `199` | `0` | `199` | `173` | `0` |
| `stage53_external_baseline_rows` | `199` | `0` | `199` | `173` | `0` |
| `stage70_baseline_status` | `3` | `3` | `0` | `0` | `0` |
| `legacy_cwgs_rd_summaries` | `264` | `0` | `264` | `264` | `0` |

### 结论

- FCGS/D-FCGS 代码可用作后续 baseline implementation 起点，但当前没有 DAVIS scoped apples-to-apples 结果。
- FCGS 输入是 static 3DGS `.ply` 和 Gaussian-Splatting Scene，相比 Stage61 DAVIS `.pt` anchors 仍缺 adapter/wrapper。
- D-FCGS README/runner 假设多视角 per-frame Gaussian sequence / 3DGStream-style layout，不能直接用于单目 DAVIS claim。
- 旧 Stage52/53 和 CWGS artifacts 都是 diagnostic/reference；不能进入最终 DAVIS RD 对比。

## 2026-06-27：Stage72 Original StreamSplat DAVIS Baseline And Low-PSNR Diagnosis

### 目标

先验证原 StreamSplat 方法在当前 DAVIS 数据/预处理/metric 口径下是否正常，再诊断并修复 Stage70 Gaussian-anchor-only RD 指标偏低的问题。

### 操作计划

Phase A：原方法 DAVIS scoped baseline。

- 使用当前 `/data` DAVIS root 和已生成 depth。
- 覆盖 Stage70 相同 eval subset：`DAVIS/val/bmx-trees`, `DAVIS/val/car-shadow`, `DAVIS/val/goat`, `DAVIS/val/soapbox`。
- 覆盖 gaps：`4`, `8`, `16`。
- 输出 all-frame PSNR，并保留 middle/given PSNR 作为诊断，不主动作为主指标。
- 对比 Stage70 的 linear/adapter all-frame PSNR，判断低指标是否来自数据/metric/原方法复现问题。

Phase B：修复 Stage70 低 PSNR。

- 检查 target RGB resize/range/颜色空间和 frame index alignment。
- 检查直接渲染 keyframe anchor 的 PSNR，判断 anchor 本身质量。
- 对比 float anchor 与 q8 anchor，分离量化损失。
- 检查 static anchor 到 renderer dynamic format 的 bridge 字段、domain、`scale/rot/opacity/rgb/xyz` 处理。
- 确认 Stage65 best checkpoint、hidden dim 和 state dict 加载无误。
- 修复后重跑相同 Stage70 scoped subset，再输出修复前后对比。

### 停止条件

完成 Phase A 和 Phase B 后先暂停，向用户汇报，不继续大规模训练、selector 长训或压缩大实验。

### Phase A 执行结果

新增脚本：

```text
scripts/run_stage72_original_davis_baseline.py
```

运行前按要求使用 `nvidia-smi` 检查 GPU，GPU1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1` 执行 smoke 和 full run。

输出文件：

```text
experiments/stage72_original_davis_baseline/stage72_original_davis_baseline_summary.json
experiments/stage72_original_davis_baseline/stage72_original_davis_baseline_rows.csv
experiments/stage72_original_davis_baseline/stage72_original_davis_baseline_per_frame.csv
experiments/stage72_original_davis_baseline/stage72_original_vs_stage70_comparison.csv
experiments/stage72_original_davis_baseline/stage72_original_vs_stage70_gap_summary.csv
experiments/stage72_original_davis_baseline/stage72_original_davis_baseline_report.md
```

原 StreamSplat full dynamic baseline mean all-frame PSNR：

| gap | all PSNR | middle PSNR | given PSNR |
|---:|---:|---:|---:|
| 4 | 23.244598036349643 | 20.881582891367685 | 29.745217856431786 |
| 8 | 20.682715912446714 | 19.17823812651332 | 29.71886975750609 |
| 16 | 18.353465837484507 | 17.3365956594116 | 29.689301009427645 |

与 Stage70 q8 adapter uniform 的 all-frame PSNR 对比：

| gap | original full dynamic | Stage70 adapter uniform | gap |
|---:|---:|---:|---:|
| 4 | 23.244598036349643 | 20.608531857721726 | 2.636066178627917 |
| 8 | 20.682715912446714 | 18.54248864942665 | 2.140227263020065 |
| 16 | 18.353465837484507 | 17.01303254555753 | 1.340433291926977 |

结论：原方法在当前 DAVIS 数据、depth、resize 和 metric 口径下正常；Stage70 偏低不是因为 DAVIS 数据或原方法复现整体失效。

## 2026-06-27：Stage73 Low-PSNR Diagnosis

### 目标

分解 Stage70 Gaussian-anchor-only RD 偏低来源，确认是否存在可修复的 evaluator/bridge/quantization/checkpoint bug。

### 新增脚本

```text
scripts/run_stage73_low_psnr_diagnosis.py
```

### 执行记录

运行前按要求使用 `nvidia-smi` 检查 GPU。GPU1 空闲，因此 Stage73 smoke 和 full scoped diagnosis 均使用 `CUDA_VISIBLE_DEVICES=1`。

先运行单序列 smoke：

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python -m py_compile scripts/run_stage73_low_psnr_diagnosis.py && CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage73_low_psnr_diagnosis.py --device cuda --sequences bmx-trees --gaps 16 --summary_root experiments/stage73_low_psnr_diagnosis_smoke
```

随后运行完整 scoped diagnosis：

```text
CUDA_VISIBLE_DEVICES=1 /mnt/hdd2tC/tmp/opencode/streamsplat_venv/bin/python scripts/run_stage73_low_psnr_diagnosis.py --device cuda
```

### 输出文件

```text
experiments/stage73_low_psnr_diagnosis/stage73_low_psnr_diagnosis_summary.json
experiments/stage73_low_psnr_diagnosis/stage73_low_psnr_diagnosis.csv
experiments/stage73_low_psnr_diagnosis/stage73_low_psnr_gap_summary.csv
experiments/stage73_low_psnr_diagnosis/stage73_low_psnr_diagnosis_report.md
```

### 关键结果

| gap | original all | float static adapter all | q8 static adapter all | q8 loss | original - float static | original given - float given | original given - q8 given |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 23.244598036349643 | 21.347280410784887 | 20.608531857721726 | 0.7387485530631608 | 1.8973176255647592 | 0.020452617603702095 | 2.762040748859361 |
| 8 | 20.682715912446714 | 18.93647863322668 | 18.54248864942665 | 0.39398998380002936 | 1.7462372792200318 | 0.011830362977251596 | 2.74592976460412 |
| 16 | 18.353465837484507 | 17.244538002072254 | 17.01303254555753 | 0.23150545651472498 | 1.1089278354122554 | -0.026037018245909316 | 2.745552680459971 |

### 结论

- Stage73 q8 adapter uniform 完全复现 Stage70 adapter uniform all-frame PSNR，说明 Stage70 汇总表和 scoped selector/rate join 没有发现口径错误。
- `float_given` 与原始 StreamSplat `given` 基本一致，说明 target RGB resize/range/color、frame index alignment、static-anchor-to-renderer bridge 和 Stage61 keyframe anchor export 是对齐的。
- Stage70 低于原 StreamSplat full dynamic baseline，主要因为当前 transmitted representation 只保留 static anchors，并用 zero-dynamic wrapper 渲染/预测 middle frames，丢弃原 StreamSplat `pred_gs` 的 dynamic components。
- q8 量化不是唯一主因，但会显著降低 keyframe render quality：given-keyframe PSNR 平均约下降 `2.75 dB`；对 all-frame PSNR 的额外损失随 keyframe 占比从 gap4 `0.7387485530631608 dB` 降到 gap16 `0.23150545651472498 dB`。
- 因未发现 evaluator bug，不直接改写 Stage70 指标；后续应把 Stage70 标注为 q8 static-anchor-only lower-quality point，并新增 q-bit/per-field quantization、动态字段保留或从原 StreamSplat checkpoint 继续训练的实验，而不是把 Stage70 作为最终质量结论。

### 停止状态

Phase A 和 Phase B 已完成。按用户要求暂停并汇报，不继续大规模训练或新的 baseline runner。

## 2026-06-27：Stage74 Stage72-vs-Actual Gap Diagnosis

### 目标

分析为什么 Stage72 的“原 StreamSplat 方法”DAVIS scoped baseline 与用户认知中的实际/官方结果相差较大。

### 操作计划

- 核查 Stage72 runner 是否等价于官方 StreamSplat evaluation，而不是仅使用 StreamSplat model 的一个自定义 DAVIS path。
- 对比 Stage72 数据/metric/protocol 与官方/实际结果：dataset split、sequence set、resolution、mask、frame sampling、input gap、depth、checkpoint/config、all-frame/middle/given 口径。
- 如果需要运行验证，先按要求检查 `nvidia-smi`，再运行小规模诊断脚本。
- 输出原因排序和下一步修复路径；若产生脚本或结果，写入 stage record 并提交推送。

### Stage74 执行结果

新增脚本：

```text
scripts/run_stage74_stage72_vs_actual_gap_diagnosis.py
```

按要求运行前多次使用 `nvidia-smi` 检查 GPU，GPU1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1`。

一次完整多组合 scoped run 超时，原因是组合过多且脚本在末尾写文件。随后缩小为关键诊断组合并完成：

```text
experiments/stage74_stage72_vs_actual_gap_diagnosis_stage72_control/
experiments/stage74_stage72_vs_actual_gap_diagnosis_sliding_per_frame/
experiments/stage74_stage72_vs_actual_gap_diagnosis_full_val_sliding_per_frame/
```

checkpoint 加载审计：

```text
missing_count = 0
unexpected_count = 0
checkpoint_tensor_count = 320
checkpoint_value_count = 183975569
```

Stage72-style control 说明 `256x256` metric 会显著抬高 PSNR：

| metric space | all PSNR | middle PSNR | given PSNR |
|---|---:|---:|---:|
| official_256_float | 27.175326918003805 | 22.04421501448969 | 34.64175257247626 |
| stage72_512_float | 24.43616869678078 | 20.79743947198787 | 29.730986225600738 |
| stage72_512_uint8 | 24.433301145796477 | 20.796620327049567 | 29.72513797820384 |

Full DAVIS val official-style sliding windows：

| gap | pair count | all PSNR | middle PSNR | given PSNR |
|---:|---:|---:|---:|---:|
| 5 | 1849 | 26.994540075591946 | 23.004337221027775 | 34.97494578472027 |
| 8 | 1759 | 24.534872014837706 | 21.56004909948801 | 34.94675221856166 |

论文数值对齐后，中间帧 PSNR 剩余差距约 `0.5-0.7 dB`：

| setting | paper PSNR | local PSNR | gap |
|---|---:|---:|---:|
| Middle-4 frames | 23.66 | 23.004337221027775 | 0.6556627789722242 |
| 8-frame interval | 22.10 | 21.56004909948801 | 0.5399509005119906 |

### 单独总结文档

按用户要求，Stage72/73/74 的当前输出和总结已单独记录在：

```text
docs/STAGE72_74_STREAMSPLAT_DAVIS_DIAGNOSIS_SUMMARY.md
```

## 2026-06-27：Stage75 Corrected StreamSplat Paper-Protocol DAVIS Package

### 目标

把 Stage74 full DAVIS val official-style outputs 打包成正式可引用的 corrected StreamSplat baseline package，避免后续报告继续误用 Stage72 scoped protocol 数值。

### 操作计划

- 读取 `experiments/stage74_stage72_vs_actual_gap_diagnosis_full_val_sliding_per_frame/` 的 aggregate 和 per-sequence CSV。
- 输出 paper-protocol summary：full DAVIS val、sliding fixed intervals、`256x256` metric、middle/non-input PSNR。
- 对照论文 Table1 Middle-4 PSNR `23.66` 和 Table2 8-frame interval PSNR `22.10`。
- 记录 Stage72 scoped 数值与 Stage75 corrected 数值的区别。
- Stage75 为 CPU-only 汇总，不运行 GPU inference。

### Stage75 执行结果

运行前按要求使用 `nvidia-smi` 检查 GPU；该阶段是 CPU-only 汇总脚本。

新增脚本：

```text
scripts/run_stage75_corrected_streamsplat_paper_protocol_package.py
```

输出：

```text
experiments/stage75_corrected_streamsplat_paper_protocol_package/stage75_corrected_streamsplat_paper_protocol_summary.json
experiments/stage75_corrected_streamsplat_paper_protocol_package/stage75_corrected_streamsplat_paper_protocol_summary.csv
experiments/stage75_corrected_streamsplat_paper_protocol_package/stage75_corrected_streamsplat_per_sequence.csv
experiments/stage75_corrected_streamsplat_paper_protocol_package/stage75_stage72_vs_corrected_comparison.csv
experiments/stage75_corrected_streamsplat_paper_protocol_package/stage75_corrected_streamsplat_paper_protocol_report.md
```

Corrected StreamSplat paper-protocol DAVIS baseline：

| paper setting | local gap | pair count | all PSNR | middle PSNR | given PSNR | paper PSNR | local - paper |
|---|---:|---:|---:|---:|---:|---:|---:|
| Middle-4 frames | 5 | 1849 | 26.994540075591946 | 23.004337221027775 | 34.97494578472027 | 23.66 | -0.6556627789722249 |
| 8-frame interval | 8 | 1759 | 24.534872014837706 | 21.56004909948801 | 34.94675221856166 | 22.1 | -0.5399509005119931 |

Stage75 应作为本地 StreamSplat DAVIS paper-protocol reference；Stage72 仅作为 scoped Mono-DFCGS diagnostic baseline。

## 2026-06-27：Stage76 Static Anchor Quantization Sweep

### 目标

量化 Stage73 中 q8 static anchor 导致 keyframe render PSNR 下降的问题，找更合理的 quantization operating point。

### 操作计划

- 使用 Stage61 DAVIS gap1 anchors，覆盖 Stage70 scoped DAVIS val sequences：`bmx-trees`, `car-shadow`, `goat`, `soapbox`。
- 对 `float16`, `q6`, `q8`, `q10`, `q12`, `q16` static anchors 直接渲染 keyframes。
- 输出 `512x288` 和 paper-style `256x256` PSNR。
- 估算单个 static anchor payload MiB，并保留 bit-depth/quality tradeoff 表。
- 运行前按要求检查 `nvidia-smi`。

### Stage77 执行结果

运行前按要求使用 `nvidia-smi` 检查 GPU。GPU1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1` 运行。

新增脚本：

```text
scripts/run_stage77_qbit_full_video_anchor_only_rd_sweep.py
```

输出：

```text
experiments/stage77_qbit_full_video_anchor_only_rd_sweep/stage77_qbit_full_video_anchor_only_rd_summary.json
experiments/stage77_qbit_full_video_anchor_only_rd_sweep/stage77_qbit_full_video_anchor_only_rd_summary.csv
experiments/stage77_qbit_full_video_anchor_only_rd_sweep/stage77_qbit_full_video_anchor_only_rd_rows.csv
experiments/stage77_qbit_full_video_anchor_only_rd_sweep/stage77_qbit_full_video_anchor_only_rd_report.md
```

Adapter key results：

| codec | gap | MiB/frame | all PSNR | middle PSNR | given PSNR | delta all vs q8 | delta middle vs q8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| q8 | 4 | 0.12129653387470925 | 20.57270098931695 | 18.247303018014392 | 27.02327983720894 | 0.0 | 0.0 |
| q10 | 4 | 0.1516173773213112 | 21.191095902140322 | 18.25601918552348 | 29.327984245556006 | 0.6183949128233728 | 0.00871616750908899 |
| q12 | 4 | 0.18193822076791313 | 21.284133638556813 | 18.256196169477683 | 29.678129037019875 | 0.7114326492398639 | 0.008893151463290394 |
| q8 | 8 | 0.06508594500594155 | 18.480296729121694 | 17.068303922946175 | 27.01806414475214 | 0.0 | 0.0 |
| q10 | 8 | 0.08135566587972795 | 18.809757721047486 | 17.06974217895638 | 29.320664241574747 | 0.32946099192579226 | 0.0014382560102035313 |
| q12 | 8 | 0.09762538675351436 | 18.85917872255155 | 17.06969395261803 | 29.668274733935373 | 0.3788819934298573 | 0.0013900296718567517 |
| q8 | 16 | 0.036980650571557694 | 16.865547380070296 | 15.978911066116073 | 26.997379801213054 | 0.0 | 0.0 |
| q10 | 16 | 0.046224810158936334 | 17.052739413791674 | 15.97580514912888 | 29.328475996349997 | 0.18719203372137727 | -0.0031059169871934245 |
| q12 | 16 | 0.055468969746314975 | 17.081329746423872 | 15.975915248513115 | 29.679767860200425 | 0.21578236635357584 | -0.002995817602958084 |

结论：q10/q12 改善 all-frame PSNR 主要来自 given/keyframe 质量恢复；middle-frame PSNR 基本不变。后续瓶颈应转向 stronger predictor / dynamic side-info / rendered-label selector。

## 2026-06-28：Stage78 Integrated DAVIS RD Package

### 目标

整合 Stage75 corrected StreamSplat baseline 与 Stage77 q8/q10/q12 anchor-only RD sweep，形成新的 DAVIS RD package，作为后续 adapter training、selector 和 compression ablation 的统一对照基线。

### 操作计划

- 读取 `experiments/stage75_corrected_streamsplat_paper_protocol_package/` 的 corrected StreamSplat paper-style reference。
- 读取 `experiments/stage77_qbit_full_video_anchor_only_rd_sweep/` 的 q8/q10/q12 linear/adapter RD rows。
- 输出 rate table、all/middle/given PSNR table、method summary、gap-to-StreamSplat reference table 和 RD curve。
- 明确 Stage78 的 StreamSplat reference 是 paper-protocol interpolation reference，不直接等同 Stage70 scoped all-frame protocol。
- Stage78 为 CPU-only 汇总；运行前仍按用户要求检查 `nvidia-smi`。

### Stage78 执行结果

运行前按要求使用 `nvidia-smi` 检查 GPU。该阶段为 CPU-only package generation。

新增脚本：

```text
scripts/run_stage78_integrated_davis_rd_package.py
```

输出：

```text
experiments/stage78_integrated_davis_rd_package/stage78_integrated_davis_rd_summary.json
experiments/stage78_integrated_davis_rd_package/stage78_integrated_davis_rd_report.md
experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_rate_table.csv
experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_psnr_table.csv
experiments/stage78_integrated_davis_rd_package/stage78_method_summary.csv
experiments/stage78_integrated_davis_rd_package/stage78_reference_gap_table.csv
experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_all_rd_curve.png
experiments/stage78_integrated_davis_rd_package/stage78_anchor_only_middle_rd_curve.png
```

Method averages：

| method | codec | mean rate | mean all PSNR | mean middle PSNR | mean given PSNR |
|---|---|---:|---:|---:|---:|
| linear | q8 | 0.0744543764840695 | 18.196689289875984 | 16.561587772918216 | 27.01290792772471 |
| linear | q10 | 0.09306595111999183 | 18.589141793730672 | 16.581582825535197 | 29.32570816116025 |
| linear | q12 | 0.11167752575591416 | 18.648513030391765 | 16.58455938471112 | 29.67539054371856 |
| adapter | q8 | 0.0744543764840695 | 18.639515032836314 | 17.098172669025544 | 27.01290792772471 |
| adapter | q10 | 0.09306595111999183 | 19.017864345659827 | 17.100522171202915 | 29.32570816116025 |
| adapter | q12 | 0.11167752575591416 | 19.074880702510743 | 17.100601790202944 | 29.67539054371856 |

诊断结论：q12 adapter 是当前 anchor-only scoped RD 中最好点，但 middle-frame PSNR 与 corrected StreamSplat reference 仍差约 `4.5-4.8 dB`。后续优先做 stronger adapter training、rendered-label selector 和 dynamic side-info。

### Stage76 执行结果

运行前按要求使用 `nvidia-smi` 检查 GPU。GPU1 空闲，因此使用 `CUDA_VISIBLE_DEVICES=1` 运行 scoped sweep。

新增脚本：

```text
scripts/run_stage76_static_anchor_quantization_sweep.py
```

输出：

```text
experiments/stage76_static_anchor_quantization_sweep/stage76_static_anchor_quantization_summary.json
experiments/stage76_static_anchor_quantization_sweep/stage76_static_anchor_quantization_summary.csv
experiments/stage76_static_anchor_quantization_sweep/stage76_static_anchor_quantization_rows.csv
experiments/stage76_static_anchor_quantization_sweep/stage76_static_anchor_quantization_report.md
```

Key result：

| codec | MiB/anchor | PSNR 512 | delta 512 | PSNR 256 | delta 256 |
|---|---:|---:|---:|---:|---:|
| q6 | 0.3428230285644531 | 16.964416756584846 | -12.770973560085498 | 17.739751819785756 | -16.925497206905096 |
| q8 | 0.4570808410644531 | 27.044935397848864 | -2.69045491882148 | 30.39796551711433 | -4.267283509576522 |
| q10 | 0.5713386535644531 | 29.35456348600337 | -0.3808268306669724 | 34.144040488489495 | -0.5212085382013569 |
| q12 | 0.6855964660644531 | 29.708204125401867 | -0.02718619126847699 | 34.6285618759872 | -0.03668715070364925 |
| float16 | 0.9141120910644531 | 29.735390316670344 | 0.0 | 34.66524902669085 | 0.0 |
| q16 | 0.9141120910644531 | 29.73518080346308 | -0.00020951320726325662 | 34.665082254118616 | -0.00016677257223562947 |

结论：q8 是 Stage70/73 keyframe quality 下降的重要原因；后续 RD 应加入 q10/q12。q10 只比 q8 单 anchor payload 增加约 `0.1142578125 MiB`，但 512 PSNR loss 从 `-2.69045491882148 dB` 收敛到 `-0.3808268306669724 dB`。

## 2026-06-27：Stage77 Q-Bit Full-Video Anchor-Only RD Sweep

### 目标

在 Stage76 direct keyframe quantization 之后，检查 q10/q12 是否能改善 full-video anchor-only RD，而不仅仅改善 keyframe direct render。

### 操作计划

- 使用 Stage61 DAVIS gap1 anchors。
- 覆盖 Stage70 scoped DAVIS val sequences：`bmx-trees`, `car-shadow`, `goat`, `soapbox`。
- 覆盖 uniform gaps：`4`, `8`, `16`。
- 覆盖 codecs：`q8`, `q10`, `q12`。
- 方法：linear anchor interpolation 与 Stage65 `rgb_h256` adapter。
- 输出 all/middle/given PSNR、rate MiB/frame、method/codec/gap summary。
- 运行前按要求检查 `nvidia-smi`。
