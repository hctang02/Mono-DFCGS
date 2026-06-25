# 用户问题与执行要求记录

## 2026-06-25：阶段 1-5 执行要求

### 用户原始要求摘要

用户要求：

- 先把完整 Mono-DFCGS 方案保存到一个 Markdown 日志/方案文件中。
- 然后完整运行阶段 1 到阶段 5，全部做完后再汇报。
- 每个阶段执行之后都要提交并推送到 git。
- 后续 git 目标仓库是 `https://github.com/hctang02/Mono-DFCGS`，不是 NeoVerse。
- 本地项目目录使用 `/mnt/hdd2tC/haocheng/Mono-DFCGS`。

### 执行和操作要求

- 每次打算执行操作之前，尽量先复述要做的事情和计划。
- 每次运行代码前，查看当前默认 GPU 是否被占用。
- 如果默认 GPU 被占用，选择空闲 GPU。
- 如果有能够并行在多张 GPU 上执行的机械重复操作，可以尝试使用 2-3 张空卡并行。
- 可以调用 subagent，subagent 标题名称也尽量用中文。

### 输出和对话要求

- 操作过程和对话输出尽量使用中文。
- 每次完整回答前，先完整重复用户问题。
- 如果遇到中断，回答时重复中断前要求执行的操作。
- 每次回答完后，总结当前情况，并说明下一步计划。
- 如果产生文件、指标或表格，给出路径和结果。
- 如果有明确数据对比，尽量用表格展示。

### 内容保存要求

- 新建 Markdown 文件记录用户每次问题，并在回答前更新。
- 操作过程和结果保存到 Markdown 文件，可用中文描述。
- 每次更新之后上传到 git。
- 如果上下文过长、压缩可能困难，提前申请是否新开 session，并同步当前 session 信息。
- 如果有踩坑点，记录到主文件夹下新的 Markdown 或 `logs/PITFALLS.md`。

### 当前执行决策

阶段 1-5 会按“先可运行最小闭环，再逐步扩展”的方式执行。阶段 4/5 涉及模型接口和训练，不直接启动大规模训练；先实现可运行的 Gaussian-anchor predictor smoke pipeline 和训练 smoke pipeline，避免资源失控。

## 2026-06-25：阶段 6 执行要求

### 用户原始要求摘要

用户要求：

- 按照前面设计，先执行阶段 6。
- 阶段 6 的目标是导出真实 keyframe Gaussian anchor dataset，为后续真实 predictor 训练、renderer/RGB loss 和 keyframe selection 做准备。

### 当前执行决策

阶段 6 会导出真实 StreamSplat keyframe Gaussian anchors。实际 `.pt` tensor 文件较大，保存到 git 外部：

```text
/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage6_real_anchor_dataset
```

仓库内只保存脚本、manifest summary、CSV 和日志：

```text
experiments/stage6_real_anchor_dataset
```

## 2026-06-25：当前进度询问

### 用户原始问题

用户询问：现在已经做到的是什么，以及目前的结果是怎么样的。

### 回答要求

需要用中文总结当前 Mono-DFCGS 已完成阶段、关键输出文件、主要实验结果、当前限制和下一步建议。

## 2026-06-25：PPT 内容参考

### 用户原始问题

用户询问：现在想基于最新的 Mono-DFCGS 方法做一个 PPT，希望给出一些文字参考和介绍内容。

### 回答要求

需要给出可直接放入 PPT 的中文 slide 结构、每页标题、核心 bullet、讲解备注和方法定位表述。

## 2026-06-25：transmitted MiB/frame 数值相近的问题

### 用户原始问题

用户询问：`transmitted MiB/frame` 为什么在几个数据集上的结果都差不多。

### 回答要求

需要解释当前阶段 2/3 的 size 统计口径：固定分辨率、固定 Gaussian 数量、固定字段数、固定 q8 估算，因此主要由 keyframe 数量和总帧数决定，而不是由内容复杂度决定。

## 2026-06-25：方法框图绘制 prompt

### 用户原始问题

用户要求：写一段话，用于描述给 GPT/绘图模型画 Mono-DFCGS 新方法的大致框图，不需要很详细，但要表达清楚整个 pipeline。

### 回答要求

需要给出一段可直接复制给绘图模型的中文 prompt，重点表达 full video -> keyframe selection -> keyframe Gaussian anchors -> transmitted bitstream -> decoder dynamic Gaussian prediction -> rendering/reconstruction 的 pipeline。

## 2026-06-25：继续执行后续阶段

### 用户原始问题

用户表示：如果不打断，就继续一直往下做。系统模式已从 plan 切换到 build，可以进行文件修改、运行命令和工具调用。

### 当前执行决策

从阶段 7 开始继续推进，优先整理 DAVIS / StreamSplat 数据集 inventory 和 manifest，然后再基于可用数据推进真实 anchor dataset 扩展和真实 predictor 训练。

## 2026-06-25：继续推进下一步

### 用户原始问题

用户表示：如果有下一步就继续执行；如果不确定如何推进，再停下来询问。

### 当前执行决策

阶段 7 已完成并推送。由于默认路径下未检测到 DAVIS / YouTube-VOS / RE10K / CO3D 原始数据，阶段 8 先实现 DAVIS / YouTube-VOS sequence manifest 和 preflight 脚本，支持用户之后挂载或下载数据后直接生成可用于 depth preprocessing 和 anchor export 的 manifest。

## 2026-06-25：阶段 9 真实 anchor proxy 训练

### 用户原始问题

用户表示：如果有下一步就继续执行；如果不确定如何推进，再停下来询问。

### 当前执行决策

阶段 8 已完成并推送。由于 DAVIS / YouTube-VOS 数据缺失但 stage6 真实 keyframe anchor dataset 已可用，下一步执行阶段 9：在 stage6 真实 anchors 上训练 `GaussianAnchorDynamicPredictor` 的 proxy loss。该阶段只验证真实 anchor 加载、q8 输入模拟、训练闭环和 held-out sample 评估，不引入 renderer/RGB loss。

## 2026-06-25：阶段 10 renderer smoke

### 用户原始问题

用户表示：如果有下一步就继续执行；如果不确定如何推进，再停下来询问。

### 当前执行决策

阶段 9 已完成并推送。下一步执行阶段 10 的最小 renderer smoke：先不做完整 RGB loss 训练，而是把单帧 static anchor 包装成 StreamSplat renderer 可接受的 zero-dynamic Gaussian 格式，验证 renderer 调用、RGB target 对齐和 MSE/PSNR 计算闭环。

## 2026-06-25：阶段 10b predictor-renderer RGB smoke

### 用户原始问题

用户表示：如果有下一步就继续执行；如果不确定如何推进，再停下来询问。

### 当前执行决策

阶段 10 已完成并推送。下一步执行 predictor 到 renderer 的 RGB loss smoke：用 q8 左右 keyframe anchors 作为输入，predictor 生成中间 raw anchor，经 zero-dynamic adapter 渲染后对单个 RGB target 执行少量反向传播 step，验证 renderer/RGB loss 可训练闭环。
