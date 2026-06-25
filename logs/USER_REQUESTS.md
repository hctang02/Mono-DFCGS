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
