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
