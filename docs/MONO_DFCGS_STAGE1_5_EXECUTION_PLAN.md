# Mono-DFCGS 阶段 1-5 完整执行方案

## 0. 项目定位

Mono-DFCGS 的目标是基于 StreamSplat-style decoder 设计一个 keyframe-Gaussian-only dynamic 3D Gaussian video codec。

核心设定：

```text
完整单目视频
-> 编码端选择 sparse keyframes
-> 编码端只传输 keyframe Gaussian anchors
-> 解码端根据 keyframe Gaussians 预测中间 dynamic Gaussians
-> Gaussian renderer 渲染完整视频 / 输出 dynamic 3DGS
```

码流中包含：

- 压缩后的关键帧 Gaussian anchors
- keyframe indices / timestamps
- resolution / fps / frame count / GOP / quantization metadata

码流中不包含：

- 非关键帧 RGB
- 非关键帧 Gaussian parameters
- optical flow
- explicit deformation field
- residual payload
- decoder 生成的 intermediate Gaussians

主码率口径与 FCGS / D-FCGS / CWGS 对齐：

```text
average transmitted Gaussian size per frame
= transmitted keyframe Gaussian bytes / total video frames
```

默认不计入：

- decoder model weights
- Depth Anything model weights
- decoder 端内部生成的 dynamic Gaussians

## 1. 阶段 1：StreamSplat Fair Metrics Baseline

目标：先把 StreamSplat 在当前四个样本上的公平指标测清楚，不再只看 all-frame PSNR。

需要实现：

- all-frame PSNR / SSIM
- middle-only PSNR / SSIM，即排除每个 keyframe pair 的输入端点，只看非输入帧
- given-keyframe PSNR / SSIM，即只看关键帧端点
- 不同 gap 的评估，优先 gap 2 / 4 / 8 / 16
- 每个样本的 per-frame metric 曲线和 summary JSON / CSV

样本：

- n3dv
- meetroom
- driving
- robot

输出：

- `experiments/stage1_streamsplat_fair_metrics/`
- 阶段结果写入 `logs/STAGE_EXECUTION_LOG.md`

阶段 1 的作用：

- 明确当前 StreamSplat baseline 的真实中间帧预测能力
- 避免 all-frame 指标被输入关键帧抬高
- 为后续 Gaussian-only codec 提供 upper-bound / teacher 参考

## 2. 阶段 2：Keyframe Gaussian Anchor Baseline

目标：建立“只传 keyframe Gaussian anchors”的基础表示和 size 统计。

需要实现：

- 固定 gap 选择关键帧
- 从 StreamSplat forward 中导出每个 keyframe pair 的 `pred_gs`
- 从 pair-level `pred_gs` 中拆出 keyframe-side Gaussian anchors
- 统计 Gaussian 字段大小：`rgb`、`opacity`、`scale`、`xyz`、`rot`
- 实现基础压缩估计：float32 / float16 / uniform quantization bit-depth
- 输出每个 gap 的 keyframe Gaussian payload size

第一版 anchor source：

- StreamSplat predicted Gaussian fields

后续可扩展 anchor source：

- FCGS per-frame anchors
- optimized static 3DGS anchors

输出：

- `experiments/stage2_keyframe_gaussian_anchor/`
- 每样本 anchor size summary
- 每样本 field-wise size breakdown

阶段 2 的作用：

- 建立与 FCGS / D-FCGS / CWGS 对齐的 Gaussian-size 口径
- 确定 keyframe Gaussian anchors 的基础码率范围

## 3. 阶段 3：Uniform Keyframe Gaussian Codec Baseline

目标：实现第一个完整 codec baseline：固定间隔 keyframe Gaussian 传输 + decoder 生成完整视频。

需要实现：

- fixed-gap keyframe selection
- 只统计 transmitted keyframe Gaussian bytes
- 非关键帧由 decoder 预测，不计入传输 size
- 输出 full-video reconstruction metrics
- 输出 RD-compatible CSV

核心码率：

```text
avg_transmitted_mib_per_frame = transmitted_keyframe_gaussian_mib / total_video_frames
```

实验 sweep：

- gap 2 / 4 / 8 / 16
- float32 / float16 / q8 / q6 / q4 anchor payload estimate

输出：

- `experiments/stage3_uniform_keyframe_gaussian_codec/`
- RD rows：size vs all/middle/given metrics

阶段 3 的作用：

- 获得第一个可画到 FCGS/D-FCGS/CWGS 对比图上的 Mono-DFCGS baseline
- 明确只传 keyframe Gaussian 的理论码率优势和质量代价

## 4. 阶段 4：Gaussian-Anchor Dynamic Predictor 最小闭环

目标：开始把原始 RGB/depth-conditioned StreamSplat dynamic decoder 改造成 Gaussian-anchor-conditioned predictor。

最小可运行版本先不追求最终质量，先建立接口和数据流：

```text
G_a, G_b, normalized_time t
-> GaussianAnchorEncoder
-> DynamicPredictor
-> predicted intermediate Gaussian fields / deformation
-> renderer / proxy evaluator
```

第一版模型可以是轻量 proxy：

- Gaussian attributes MLP encoder
- bidirectional endpoint interpolation baseline
- opacity lifecycle proxy
- optional residual MLP

需要实现：

- `mono_dfcgs/gaussian_codec.py`：Gaussian payload size / quantization 工具
- `mono_dfcgs/anchor_predictor.py`：Gaussian-anchor dynamic predictor 最小模型
- `scripts/run_stage4_gaussian_anchor_predictor_smoke.py`：smoke pipeline

输出：

- smoke JSON
- tensor shape check
- payload / prediction interface check

阶段 4 的作用：

- 从“复用 StreamSplat 输出”过渡到“我们自己的 Gaussian-anchor-conditioned predictor”
- 为阶段 5 训练脚本提供最小模型接口

## 5. 阶段 5：Training / Evaluation Smoke Pipeline

目标：建立训练框架，而不是立即进行大规模训练。

训练原则：

- 输入只能包含 keyframe Gaussians
- 中间帧只能用于 loss，不作为模型输入
- 先做 smoke training，验证代码路径和 loss 能下降/能反向传播

需要实现：

- synthetic / exported Gaussian sequence dataset loader
- variable gap sampling interface
- quantization-aware training option
- smoke training script
- evaluation script

第一版训练内容：

- 用 stage 2/3 导出的 Gaussian fields 或 synthetic Gaussians 做最小训练
- loss 使用 Gaussian field reconstruction/proxy loss
- 若 renderer 可用，再加入 RGB render loss

输出：

- `experiments/stage5_training_smoke/`
- train loss JSON
- eval JSON
- 阶段日志

阶段 5 的作用：

- 确认 Mono-DFCGS 的训练工程可以跑通
- 后续再扩展到真实数据、多 gap、大规模训练和 RD-aware keyframe selection

## 6. Git 与日志要求

每个阶段完成后必须：

- 更新 `logs/STAGE_EXECUTION_LOG.md`
- 如有踩坑，更新 `logs/PITFALLS.md`
- 提交 git
- 推送到 `git@github.com:hctang02/Mono-DFCGS.git`

每次运行 GPU 相关代码前必须：

- 检查 `nvidia-smi`
- 如果默认 GPU 被占用，选择空闲 GPU
- 能并行的机械重复评估优先考虑 2-3 张空卡并行，但不要影响已有进程
