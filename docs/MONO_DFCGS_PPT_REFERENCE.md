# Mono-DFCGS PPT 文字参考

## Slide 1：标题页

标题建议：

```text
Mono-DFCGS: Keyframe-Gaussian-Only Dynamic 3D Gaussian Video Codec
```

中文标题建议：

```text
Mono-DFCGS：仅传输关键帧 Gaussian 的动态 3DGS 视频压缩方法
```

副标题：

```text
From monocular video to sparse keyframe Gaussian anchors and decoder-side dynamic 3D reconstruction
```

讲解重点：

- 我们关注的是单目动态视频的 3D-aware 压缩。
- 与传统视频编码不同，方法不传输所有帧，也不传输非关键帧 motion/residual。
- 码流只包含 sparse keyframe Gaussian anchors。
- 解码端利用动态 3DGS 预测模型补全中间帧和中间 dynamic Gaussians。

## Slide 2：研究动机

标题：

```text
Motivation: Low-rate Dynamic 3D Scene Transmission
```

可放 bullet：

- 传统视频编码主要面向 2D frame reconstruction，缺少显式 3D dynamic representation。
- 动态 3DGS 方法可以表达时变场景，但逐帧传输 Gaussian 参数代价较高。
- 单目视频场景下，我们希望以极低码率传输动态场景，同时保留 3D-aware rendering 能力。
- 核心问题：能否只传输少量关键帧 Gaussian，让 decoder 生成完整动态 3DGS？

讲解备注：

```text
我们的目标不是传统意义上的逐像素无损恢复，而是面向低码率、3D-aware、动态场景重建的神经预测式 codec。
```

## Slide 3：核心设定

标题：

```text
Keyframe-Gaussian-Only Codec Setting
```

核心流程：

```text
full monocular video
-> keyframe selection
-> keyframe Gaussian anchors
-> transmitted Gaussian bitstream
-> decoder-side dynamic Gaussian prediction
-> reconstructed video / dynamic 3DGS
```

码流包含：

- keyframe Gaussian anchors
- keyframe indices / timestamps
- metadata: resolution, fps, frame count, GOP, quantization settings

码流不包含：

- non-keyframe RGB
- non-keyframe Gaussian parameters
- optical flow
- deformation field
- residual payload

讲解备注：

```text
这里的“关键帧”不是传统视频 codec 的 I-frame RGB，而是关键帧对应的 Gaussian anchor。非关键帧完全由 decoder 预测得到。
```

## Slide 4：与传统视频编码/已有 Gaussian 编码的区别

标题：

```text
Difference from Existing Codecs
```

表格建议：

| Method type | Transmitted payload | Decoder output | 3D-aware |
|---|---|---|---|
| H.264 / H.265 | frames + motion + residual | RGB video | No |
| FCGS | per-frame Gaussians | Gaussian-rendered frames | Yes |
| D-FCGS | I/P Gaussian or motion payload | dynamic Gaussian/video | Yes |
| Mono-DFCGS | sparse keyframe Gaussian anchors only | full dynamic Gaussians + video | Yes |

讲解备注：

```text
我们的方法更极端：只传关键帧 Gaussian，不传中间帧 motion 或 residual。它牺牲一定逐像素精确性，换取更低传输码率和 decoder-side 3D dynamic generation。
```

## Slide 5：方法总览

标题：

```text
Method Overview
```

模块：

1. Keyframe selection
2. Keyframe Gaussian anchor extraction
3. Keyframe Gaussian compression
4. Gaussian-anchor-conditioned dynamic prediction
5. Gaussian rendering and reconstruction

可以放的文字：

```text
Given a monocular video, the encoder selects sparse keyframes and converts them into compact Gaussian anchors. Only these anchors are transmitted. At the decoder, a Gaussian-anchor-conditioned predictor estimates intermediate dynamic Gaussians between adjacent keyframes, which are then rendered into RGB frames.
```

中文说明：

```text
编码端只负责选择并压缩关键帧 Gaussian；解码端负责根据相邻关键帧 Gaussian 预测中间动态 Gaussian，并通过 Gaussian renderer 重建完整视频。
```

## Slide 6：当前已完成的实验阶段

标题：

```text
Current Implementation Progress
```

表格建议：

| Stage | Status | Output |
|---|---|---|
| Stage 1 | Done | StreamSplat all/middle/given metrics |
| Stage 2 | Done | keyframe Gaussian anchor size baseline |
| Stage 3 | Done | uniform keyframe Gaussian RD baseline |
| Stage 4 | Done | Gaussian-anchor predictor smoke model |
| Stage 5 | Done | synthetic training smoke |
| Stage 6 | Done | real keyframe Gaussian anchor dataset |

讲解备注：

```text
目前我们已经完成了从 baseline evaluation 到真实 keyframe Gaussian dataset 导出的基础链路。下一步是用真实 anchors 训练 predictor，并接入 renderer/RGB loss。
```

## Slide 7：Stage 1 Fair Metrics

标题：

```text
Fair Evaluation: All vs Middle-only vs Given-keyframe
```

关键点：

- all-frame PSNR 会包含输入 keyframes，因此可能偏乐观。
- middle-only PSNR 更能反映非关键帧预测能力。
- given-keyframe PSNR 反映输入关键帧重建质量。

代表结果：

| sample | gap | all PSNR | middle PSNR | given PSNR |
|---|---:|---:|---:|---:|
| n3dv | 4 | 33.3961 | 33.0521 | 34.3788 |
| meetroom | 4 | 32.4263 | 31.1640 | 36.0328 |
| driving | 4 | 30.1950 | 27.8362 | 36.9345 |
| robot | 4 | 26.4895 | 23.1786 | 35.9256 |

讲解备注：

```text
driving 和 robot 的 middle-only PSNR 明显低于 given-keyframe PSNR，说明动态预测和遮挡/运动建模才是真正难点。
```

## Slide 8：Stage 2/3 Gaussian-size RD Baseline

标题：

```text
Uniform Keyframe Gaussian Codec Baseline
```

主配置：

```text
static_anchor + q8 + opacity_threshold=0
```

代表结果：

| sample | gap | transmitted MiB/frame | all PSNR | middle PSNR |
|---|---:|---:|---:|---:|
| n3dv | 4 | 0.11849 | 33.3961 | 33.0521 |
| meetroom | 4 | 0.11849 | 32.4263 | 31.1640 |
| driving | 4 | 0.11849 | 30.1950 | 27.8362 |
| robot | 4 | 0.11871 | 26.4895 | 23.1786 |

讲解备注：

```text
这里的横轴不是 raw StreamSplat pred_gs size，而是 estimated transmitted keyframe Gaussian anchor size，和 FCGS / D-FCGS / CWGS 的 Gaussian-size 口径更一致。
```

## Slide 9：Stage 6 Real Anchor Dataset

标题：

```text
Real Keyframe Gaussian Anchor Dataset
```

导出结果：

- 297 keyframe-pair items
- external dataset size: 546M
- supports gap 2 / 4 / 8 / 16
- samples: n3dv, meetroom, driving, robot

每个 item 包含：

- left keyframe Gaussian anchor
- right keyframe Gaussian anchor
- intermediate frame timestamps
- RGB/depth supervision paths

anchor shape：

```text
rgb:     [36864, 3], float16
opacity: [36864, 1], float16
scale:   [36864, 2], float16
xyz:     [36864, 3], float16
rot:     [36864, 4], float16
```

讲解备注：

```text
这是后续真实 Gaussian-anchor predictor 训练的关键数据基础。之前的 smoke training 使用 synthetic anchors，现在可以过渡到真实 StreamSplat anchors。
```

## Slide 10：当前 Predictor Smoke

标题：

```text
Gaussian-Anchor Dynamic Predictor: Initial Smoke Result
```

模型输入：

```text
left_anchor, right_anchor, normalized_time
```

模型输出：

```text
intermediate Gaussian anchor fields
```

smoke 结果：

| item | value |
|---|---:|
| parameters | 102925 |
| synthetic initial eval loss | 0.089216 |
| synthetic final eval loss | 0.014656 |
| eval loss ratio | 0.164271 |

讲解备注：

```text
这个结果只说明训练路径、反向传播和接口是可行的，还不是最终视频重建质量。下一步要用真实 anchors 和 renderer supervision 替代 synthetic proxy。
```

## Slide 11：当前限制

标题：

```text
Current Limitations
```

可放 bullet：

- Stage 3 的重建质量仍来自 StreamSplat RGB/depth-conditioned reconstruction，不是最终 Gaussian-anchor-only decoder。
- Stage 4/5 predictor 仍是 smoke/proxy 模型。
- 还没有用 Stage 6 的真实 anchors 训练 predictor。
- 还没有接入 Gaussian renderer 和 RGB reconstruction loss。
- 还没有实现 adaptive / RD-aware keyframe selection。
- 还没有和 FCGS / D-FCGS / CWGS 做最终统一 RD 图。

讲解备注：

```text
当前工作已经完成了方法定义、码率口径、baseline 和数据准备，但真正的模型训练和最终 RD 对比还在下一阶段。
```

## Slide 12：下一步计划

标题：

```text
Next Steps
```

建议顺序：

1. Train predictor on real keyframe Gaussian anchors
2. Add renderer and RGB reconstruction loss
3. Add variable-gap training
4. Implement motion-aware / Gaussian-aware keyframe selection
5. Implement RD-aware keyframe selection
6. Compare with FCGS / D-FCGS / CWGS under unified Gaussian-size rate

中文版本：

- 用真实 keyframe Gaussian anchors 训练 predictor。
- 接入 renderer，让模型直接优化中间帧 RGB 重建质量。
- 加入 variable GOP 训练，提高长间隔预测能力。
- 设计 motion-aware / Gaussian-aware / RD-aware keyframe selection。
- 和 FCGS、D-FCGS、CWGS 在统一 Gaussian-size 口径下做 RD 对比。

## 一句话方法总结

英文：

```text
Mono-DFCGS is a keyframe-Gaussian-only dynamic 3DGS codec that transmits only sparse keyframe Gaussian anchors and reconstructs non-keyframes through decoder-side dynamic Gaussian prediction.
```

中文：

```text
Mono-DFCGS 是一种仅传输稀疏关键帧 Gaussian anchors 的动态 3DGS 视频压缩方法，解码端通过动态 Gaussian 预测生成所有非关键帧和完整动态 3D 表示。
```

## 30 秒口头介绍版本

```text
我们的工作目标是设计一个面向单目动态视频的 3D-aware codec。和传统视频编码不同，我们不传输所有帧，也不传输非关键帧的 motion 或 residual；和逐帧 Gaussian 编码不同，我们也不传输每一帧的 Gaussian。Mono-DFCGS 只传输少量关键帧对应的 Gaussian anchors，解码端根据相邻关键帧 anchors 预测中间时刻的 dynamic Gaussians，并通过 Gaussian renderer 重建完整视频。当前我们已经完成了 StreamSplat fair metrics、keyframe Gaussian size 统计、uniform RD baseline、Gaussian-anchor predictor smoke、training smoke，以及真实 keyframe Gaussian anchor dataset 导出。下一步会用真实 anchors 训练 predictor，并接入 renderer/RGB loss，最终和 FCGS、D-FCGS、CWGS 在统一 Gaussian-size 口径下做 RD 对比。
```

## 1 分钟口头介绍版本

```text
Mono-DFCGS 关注的是低码率动态 3D 场景传输。现有视频 codec 通常只恢复 2D RGB，而逐帧 3DGS 或动态 Gaussian 方法虽然有 3D 表示，但如果每帧都传 Gaussian，码率会比较高。我们的核心想法是只传 sparse keyframe Gaussian anchors。编码端从完整单目视频中选择关键帧，并把这些关键帧表示成压缩 Gaussian anchors；解码端不接收非关键帧 RGB、motion、residual 或 Gaussian，而是根据相邻 keyframe anchors 预测中间 dynamic Gaussians，再渲染出完整视频。

目前我们已经建立了完整的基础链路。首先，我们重新评估了 StreamSplat 的 all-frame、middle-only 和 given-keyframe 指标，避免 all-frame PSNR 被输入关键帧抬高。其次，我们统计了 keyframe Gaussian anchor 的 transmitted size，并形成了 uniform keyframe Gaussian codec 的 RD baseline。例如在 gap=4、q8 static anchors 下，n3dv 的 transmitted size 约为 0.118 MiB/frame，middle-only PSNR 约为 33.05 dB。最后，我们实现了 Gaussian-anchor predictor 的 smoke model 和训练 smoke，并导出了 297 个真实 keyframe-pair anchor 数据项。接下来会用这些真实 anchors 训练 predictor，接入 renderer 和 RGB loss，并进一步研究 RD-aware keyframe selection。
```
