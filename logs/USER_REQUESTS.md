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

## 2026-06-25：阶段 11 keyframe selection baseline

### 用户原始问题

用户表示：如果有下一步就继续执行；如果不确定如何推进，再停下来询问。

### 当前执行决策

阶段 10b 已完成并推送。下一步实现 keyframe selection baseline，先输出 uniform / motion-aware / Gaussian-aware / RD-aware 的 keyframe indices 和 q8 static-anchor rate 估算；重建质量评估留到后续阶段运行 selected-keyframe reconstruction。

## 2026-06-25：阶段 12 selected-keyframe reconstruction smoke

### 用户原始问题

用户表示：好的，可以继续做。

### 当前执行决策

阶段 11 已完成并推送。下一步执行 Stage 12 selected-keyframe reconstruction/evaluation。为避免直接启动完整大批量 GPU 实验，先实现通用非均匀 selected-keyframe evaluator，并只运行 `robot + rd_aware + reference_gap=4` 的 smoke；该组合最大 segment length 为 16，风险低于 gap8/gap16 的聚簇选择。

## 2026-06-25：阶段 13 spaced keyframe selection

### 用户原始问题

用户表示：好的，可以继续做。

### 当前执行决策

阶段 12 已完成并推送，且暴露了 top-k motion/RD keyframe selection 容易聚簇、造成长 segment 的问题。下一步执行 Stage 13：实现 spacing-constrained keyframe selection，在保持与 uniform 相同 keyframe budget 的同时约束最大 segment length 不超过 reference gap。

## 2026-06-25：阶段 14 spaced selected-keyframe reconstruction

### 用户原始问题

用户表示：好的，可以继续做。

### 当前执行决策

阶段 13 已完成并推送。下一步扩展 Stage 12 evaluator 支持 `motion_spaced` / `gaussian_spaced` / `rd_spaced` 方法，并运行 `robot + rd_spaced + reference_gap=4` reconstruction smoke，用于和 Stage 12 的 `robot + rd_aware + reference_gap=4` 对比。

## 2026-06-25：阶段 15 selected-keyframe RD curve

### 用户原始问题

用户表示：好的，做吧。

### 当前执行决策

阶段 14 已完成并推送。下一步扩展评估到 `n3dv/robot × uniform/rd_spaced × gap4/8/16`，形成一个更完整的 selected-keyframe RD curve smoke。该阶段仍复用 Stage 12 evaluator，结果口径仍是 StreamSplat RGB/depth-conditioned selected-pair inference，不是最终 Gaussian-anchor-only decoder。

## 2026-06-25：绘制 Stage 15 RD 曲线

### 用户原始问题

用户要求：先画一下 Stage 15 的 RD 曲线。

### 当前执行决策

基于 `experiments/stage15_selected_keyframe_rd_curve/stage15_selected_keyframe_rd_curve_summary.csv` 绘制 RD 曲线。横轴使用 `estimated_q8_static_mib_per_frame`，纵轴分别绘制 all-frame PSNR、middle-only PSNR、all-frame SSIM 和 middle-only SSIM，并按 `n3dv` / `robot` 分子图展示 `uniform` 与 `rd_spaced`。

## 2026-06-25：继续推进 Stage 16/17 并扩展数据集对比

### 用户原始问题

用户表示：可以往下做，并提醒最后的数据集 RD 曲线等对比要包括更多数据。

### 当前执行决策

下一步先执行 Stage 16：实现 segment-error-aware keyframe selection，覆盖当前 4 个样本 `n3dv/meetroom/driving/robot`，为后续更多数据的 RD 曲线对比准备统一 selection CSV。随后执行 Stage 17：在可控范围内复用 selected-keyframe evaluator 跑扩展 RD reconstruction，并逐步扩展到更多样本。

## 2026-06-25：Stage 18 long-GOP decoder 微调前参数审计

### 用户原始问题

用户要求：按照“先审计参数切分，再做 variable-GOP baseline，再做 long-GOP Dynamic Decoder 微调”的顺序继续操作。

### 当前执行决策

Stage 18 先不训练，只加载 StreamSplat checkpoint 并统计模块参数量、checkpoint 加载状态和推荐 freeze policy。首轮 long-GOP 微调建议冻结 static encoder / static Gaussian predictor / DINO condition encoder，只训练 temporal decoder、dynamic Gaussian predictor 和小的 projection adapter。

## 2026-06-25：继续执行后续阶段

### 用户原始问题

用户要求：如果有下一步就继续执行；如果不确定如何推进，再停止并提问。

### 当前执行决策

Stage 18 已完成并推送后，继续执行 Stage 19：整理 original StreamSplat decoder 的 variable-GOP pre-finetune baseline。该阶段复用已有 Stage1 推理质量和 Stage2 q8 static-anchor rate，不重新运行昂贵 GPU 推理。

## 2026-06-25：按下一步计划继续做 Stage21b

### 用户原始问题

用户确认：按照设计的下一步计划继续做。

### 当前执行决策

优先执行 Stage21b residual-zero Gaussian-anchor-only adapter training，而不是直接跑 Stage22 RD curve。原因是 Stage21 随机 residual 初始化仍低于 linear anchor baseline；Stage21b 先让初始 adapter 严格等于 q8 linear anchor interpolation，再用 RGB renderer loss 做更稳定的小规模训练。
