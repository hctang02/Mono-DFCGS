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

## 2026-06-25：继续推进到 Stage21c

### 用户原始问题

用户确认：继续按下一步计划执行。

### 当前执行决策

Stage21b 收益为正但极小，因此继续执行 Stage21c medium anchor adapter training。Stage21c 扩大到多 GOP gap `2/4/8/16`，使用 sample-balanced rows，并保持输入 payload 只包含 q8 keyframe anchors + timestamp。

## 2026-06-25：完整推进后续阶段

### 用户原始问题

用户要求：完整地继续做，后续由模型自行分析并执行。

### 当前执行决策

继续执行 Stage21d validated anchor adapter training 和 Stage22 anchor-only RD curve。Stage21d 解决 Stage21c 缺少 validation-based checkpoint selection 的问题；Stage22 将 Stage21d best checkpoint 的 robot intermediate-target quality 接入 Stage2 q8 static-anchor rate，形成 anchor-only RD 汇总和曲线图。

## 2026-06-25：继续执行 Stage23 full-video evaluator

### 用户原始问题

用户要求：可以继续往后做后面的 stages。

### 当前执行决策

执行 Stage23 full-video anchor-only evaluator，将 Stage21d best adapter 从 intermediate-target RD 扩展到完整视频逐帧评估。优先验证 adapter 收益是否在 `n3dv/meetroom/driving/robot × gap 2/4/8/16` 的 all/middle PSNR 上持续为正。

## 2026-06-25：继续执行更多后续 stages

### 用户原始问题

用户要求：按照设计继续往后做，多做一些。

### 当前执行决策

继续执行 Stage25 leave-one-sample-out adapter training。该阶段轮流 hold out `n3dv/meetroom/driving/robot`，每折使用其余三个样本训练并在 held-out 样本上 validation-based checkpoint selection，用于验证 adapter 的跨样本泛化。

## 2026-06-25：继续 Stage26 held-out full-video RD

### 用户原始问题

用户要求：如果有下一步就继续，如果不确定再问。

### 当前执行决策

Stage25 已完成并推送后，继续执行 Stage26。该阶段使用 Stage25 每个 held-out fold 的 best checkpoint，在对应 held-out sample 上做 full-video anchor-only RD 评估和绘图。

## 2026-06-25：继续 Stage27 selector integration

### 用户原始问题

用户要求继续做后续步骤。

### 当前执行决策

继续执行 Stage27 anchor-only keyframe selector integration。由于 Stage6 当前只导出了偶数 endpoint anchors，而 Stage16 unconstrained selector 会选奇数帧，Stage27 采用 anchor-available constrained selector：只允许选择已有 transmitted anchor 的帧，再比较 uniform 和 constrained `anchor_segment_rd`。

## 2026-06-25：继续 Stage28 enhanced rate model

### 用户原始问题

用户要求继续做后续步骤。

### 当前执行决策

继续执行 Stage28 rate model upgrade。保持主 rate 口径为 transmitted q8 static Gaussian anchors MiB/frame，同时额外报告 quant params、keyframe indices/timestamps、metadata 和 q8 symbol entropy estimate。

## 2026-06-25：继续 Stage29 selector objective upgrade

### 用户原始问题

用户要求：按照建议继续做一些步骤，并且尽量多做一些。

### 当前执行决策

先执行 Stage29，用 anchor-only/anchor-attribute quality 目标替代 Stage27 的 motion/RD heuristic。该阶段使用 held-out sample 的 intermediate q8 anchors 作为 oracle/proxy signal，通过 DP 选择 keyframes，再用 full-video renderer 验证是否优于 uniform。

## 2026-06-25：继续 Stage31 bitstream prototype

### 用户原始问题

用户要求继续多做一些步骤。

### 当前执行决策

继续执行 Stage31，实现 q8 static anchor bitstream prototype。该阶段将 Stage26 uniform keyframe anchors 编码为真实 binary container，并验证 decode roundtrip。raw/zlib bitstreams 写到外部 run 目录，仓库只提交轻量 CSV/JSON。

## 2026-06-25：补充 Stage30 dense anchor preflight

### 用户原始问题

用户要求继续多做一些步骤。

### 当前执行决策

补充执行 Stage30 dense/all-frame anchor export preflight。该阶段不导出大文件，只统计当前 Stage6 anchor coverage、Stage16 selector 与现有 anchors 的兼容性，以及导出 dense anchors 的空间成本。

## 2026-06-25：继续 Stage32 actual bitstream RD

### 用户原始问题

用户要求：按照设计继续往下做。

### 当前执行决策

先执行 Stage32，将 Stage26 leave-one-out full-video quality 与 Stage31 raw/zlib q8 anchor bitstream sizes 合并，生成 actual-bitstream RD 表和图。该阶段把之前的 estimated q8 anchor rate 推进到真实 container rate 报告。

## 2026-06-25：继续 Stage33 dense gap1 anchor export

### 用户原始问题

用户要求继续往下做。

### 当前执行决策

执行 Stage33，实际导出 gap1 adjacent pair anchors，覆盖所有帧作为 keyframe anchor candidates。外部 `.pt` 数据写入 `/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage33_dense_gap1_anchor_export`，仓库只保存 manifest 和 summary。

## 2026-06-25：继续 Stage34 dense Stage16 selector RD

### 用户原始问题

用户要求继续往下做。

### 当前执行决策

执行 Stage34，用 Stage33 dense gap1 anchors 重新评估 Stage16 unconstrained `segment_rd` selector，比较 uniform vs segment_rd。该阶段用于判断 Stage27 的负结果是否只是因为奇数帧 anchors 不可用。

## 2026-06-25：继续 Stage35 dense anchor-attribute oracle selector

### 用户原始问题

用户准备睡觉，要求按照顺序继续多执行实验。

### 当前执行决策

执行 Stage35，用 Stage33 dense anchors 重跑 anchor-attribute oracle/proxy selector。该阶段验证全帧 keyframe candidates 下，Stage29 的 oracle/proxy selector 上界是否进一步提升，并检查 driving gap4/8 负点是否消失。

## 2026-06-25：继续 Stage36 dense oracle actual-bitstream RD

### 用户原始问题

用户睡觉前要求继续按顺序多执行实验。

### 当前执行决策

执行 Stage36，将 Stage35 的 uniform 和 dense oracle selected keyframes 编成真实 q8 anchor bitstream，输出 raw/zlib actual-bitstream RD。外部 bitstreams 写入 `/mnt/hdd2tC/tmp/opencode/mono_dfcgs_runs/stage36_dense_oracle_actual_bitstream_rd`。

## 2026-06-25：继续 Stage37 deployable selector cost dataset

### 用户原始问题

用户睡觉前要求继续按顺序多执行实验。

### 当前执行决策

执行 Stage37，构建 deployable selector cost predictor 数据集。标签来自 dense oracle anchor-attribute segment cost，特征只使用 encoder 可获得的信息：endpoint anchors、segment length、RGB motion 和 endpoint RGB difference。输出 leave-one-out split 标记。

## 2026-06-25：继续 Stage38 deployable cost predictor validation

### 用户原始问题

用户睡觉前要求继续按顺序多执行实验。

### 当前执行决策

执行 Stage38，对 Stage37 数据集训练 leave-one-out ridge cost predictor。比较 `length_only_ridge` 和 `full_feature_ridge`，先验证 deployable features 是否能泛化预测 dense oracle cost，再决定是否进入 predicted selector RD。

## 2026-06-25：继续 Stage39 predicted selector RD

### 用户原始问题

用户要求继续执行下一步，如果有下一步就继续推进，不确定时再询问。

### 当前执行决策

执行 Stage39，把 Stage38 的 predicted segment costs 直接用于 DP keyframe selection，然后用 Stage25 leave-one-out anchor adapter 和 Stage33 dense anchors 做 full-video anchor-only RD 评估。比较 `uniform`、`length_only_ridge` predicted selector 和 `full_feature_ridge` predicted selector。

## 2026-06-25：继续 Stage40 normalized cost predictor validation

### 用户原始问题

用户要求如果有下一步就继续推进。

### 当前执行决策

执行 Stage40，针对 Stage38/39 暴露的 raw cross-sample scale mismatch，验证 sample/candidate 内归一化特征、z-score target 和 rank target 的 deployable cost predictor。若 normalized/rank predictor 明显改善，再进入 Stage41 full-video predicted selector RD。

## 2026-06-25：继续 Stage41 normalized predicted selector RD

### 用户原始问题

用户要求如果有下一步就继续推进。

### 当前执行决策

执行 Stage41，将 Stage40 的 normalized/rank predicted costs 用于 DP keyframe selection，并用 Stage25 leave-one-out adapter + Stage33 dense anchors 进行 full-video RD 评估。重点检验 Stage40 的 predictor ranking 改善是否能转化为实际 RD 增益。

## 2026-06-25：继续 Stage42 calibrated selector prior RD

### 用户原始问题

用户要求如果有下一步就继续推进。

### 当前执行决策

执行 Stage42，在 Stage41/40 的 `full_sample_z_rank` predictor score 上加入 uniform segment-length prior，扫多个 alpha，验证是否可以避免过度非均匀 keyframe layout 并恢复或超过 uniform RD。

## 2026-06-26：继续 Stage43 selector evidence synthesis

### 用户原始问题

用户要求如果有下一步就继续推进。

### 当前执行决策

执行 Stage43，汇总当前 selector 相关证据，明确区分 Stage26 adapter codec 正收益、Stage36 dense oracle/proxy selector upper bound，以及 Stage39/41/42 deployable predicted selector 负结果。输出 JSON/CSV/Markdown，便于后续汇报和论文 artifact 使用。

## 2026-06-26：继续 Stage44 rendered segment distortion dataset

### 用户原始问题

用户同意新的 Adaptive Keyframe Selection 研究方案，要求继续往下做，并强调：保留我们自己方法的创新性、后续要推广到更大规模训练、尽快看到 RD 曲线。

### 当前执行决策

执行 Stage44，先在当前四个样本上构建 rendered segment distortion dataset。该数据集用 frozen Stage25 leave-one-out adapter 实际渲染候选 segment 的中间帧，得到 adapter rendered distortion label，作为 Stage45 rendered-oracle selector 和后续 Stage47 feed-forward selector predictor 的训练/分析基础。为尽快得到 RD 曲线，默认每个候选 segment 采样最多 3 个中间帧估计整段误差，脚本保留 `--max_targets_per_segment=0` 以支持后续全中间帧精确版本。

## 2026-06-26：继续 Stage45 rendered-oracle adaptive selector RD

### 用户原始问题

用户要求按照新的 Adaptive Keyframe Selection 设计继续执行，并希望尽快看到 RD 曲线。

### 当前执行决策

执行 Stage45，用 Stage44 的 rendered segment distortion label 作为 oracle DP cost，在与 uniform 相同 keyframe budget 下选择 adaptive keyframes，并用 Stage25 leave-one-out adapter + Stage33 dense anchors 做 full-video RD evaluation，输出第一版 adaptive-vs-uniform RD 曲线。
