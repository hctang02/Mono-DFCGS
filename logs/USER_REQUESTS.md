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

## 2026-06-26：继续 Stage45b calibrated rendered-oracle selector RD

### 用户原始问题

用户要求继续推进 Adaptive Keyframe Selection，尽快看到并改进 RD 曲线。

### 当前执行决策

执行 Stage45b，在 Stage45 rendered-oracle cost 基础上扫描 uniform segment-length prior 和 minimum segment length calibration，目标是减少 Stage45 gap8/gap16 的大负点，找到更稳的 adaptive-or-uniform fallback 策略。

## 2026-06-26：继续 Stage46 calibrated adaptive actual-bitstream RD

### 用户原始问题

用户要求继续推进，尽快看到 RD 曲线，并且后续 adaptive keyframe selection 要作为主要创新点之一。

### 当前执行决策

执行 Stage46，将 Stage45b 当前最稳的 `rendered_prior_0p1` adaptive selector 结果转成真实 q8 raw/zlib Gaussian anchor bitstream RD。质量沿用 Stage45b full-video evaluation，rate 使用实际编码文件大小，heavy bitstreams 写入 git 外部目录。

## 2026-06-26：继续 Stage47 feed-forward rendered cost predictor validation

### 用户原始问题

用户要求继续推进 Adaptive Keyframe Selection，并保证最终方法具备完整前馈性。

### 当前执行决策

执行 Stage47，用 Stage44 rendered segment distortion labels 训练/验证 leave-one-sample-out feed-forward segment cost predictors。输入只使用 encoder-side temporal/RGB/Gaussian endpoint features，输出 prediction CSV 供 Stage48 的 fully feed-forward DP keyframe selection 使用。

## 2026-06-26：继续 Stage48 predicted adaptive selector RD

### 用户原始问题

用户要求继续推进 Adaptive Keyframe Selection，并保证最终方案完整前馈。

### 当前执行决策

执行 Stage48，将 Stage47 前馈 predictor 输出的 segment costs 输入 DP keyframe selection，再用 Stage25 leave-one-out adapter + Stage33 dense anchors 做 full-video RD 评估。该阶段是当前第一版 fully feed-forward predicted adaptive selector 测试，不使用 rendered oracle cost 作为 selection 输入。

## 2026-06-26：继续 Stage49 extended adaptive RD

### 用户原始问题

用户要求继续按新顺序往下做，并指出当前 RD 曲线点太少、码率过低，希望先增加 5 个以上 RD 点、加入更高码率点，后续再做更高 bit-depth 和 FCGS/D-FCGS 对比。

### 当前执行决策

执行 Stage49，扩展当前 adaptive RD 曲线到 `gap=1/2/3/4/8/16`。all-frame RD 使用 6 个点，middle-only RD 自动排除没有中间帧的 `gap1`，使用 5 个点。方法包括 `uniform` 和当前最稳的 rendered-oracle calibrated adaptive selector `rendered_prior_0p1`。同时输出 estimated q8 rate、actual raw q8 bitstream rate 和 actual zlib q8 bitstream rate。

## 2026-06-26：继续 Stage50 multi-bit anchor bitstream prototype

### 用户原始问题

用户要求在扩展 RD 点之后，继续增加更高码率点，解决当前码率分布偏低和 PSNR 一般的问题。

### 当前执行决策

执行 Stage50，将 q8-only anchor bitstream 扩展到 q6/q8/q10/q12/q16 multi-bit prototype。该阶段只做编码/解码 roundtrip 和 rate 统计，不重新渲染。为快速支持高码率，`bits<=8` 使用 uint8 payload，`bits>8` 使用 uint16 payload；同时报告 theoretical bitpacked MiB/frame，方便后续区分 prototype storage rate 和 compact bitpacked estimate。

## 2026-06-26：继续 Stage51 high-rate multi-bit RD

### 用户原始问题

用户要求继续补充更高码率点，让 RD 曲线覆盖更完整码率范围，并改善当前 PSNR 偏一般的问题。

### 当前执行决策

执行 Stage51，用 Stage50 的 q8/q10/q12/q16 actual raw/zlib bitstream rate，重新以对应 bit-depth dequantized anchors 渲染 Stage49 的 uniform 和 `rendered_prior_0p1` selections，生成 high-rate multi-bit RD 曲线。该阶段质量是真实重新渲染得到，不只是替换 rate 轴。

## 2026-06-26：继续 Stage52 FCGS/D-FCGS baseline preflight

### 用户原始问题

用户要求：如果有下一步就继续执行；如果不确定如何推进，再停下来询问。

### 当前执行决策

执行 Stage52，解析本机已有 FCGS / D-FCGS 候选日志和 summary，先形成 baseline availability report、字段解析状态和缺字段清单。该阶段不做新训练、不复制外部大文件，重点是给 Stage53 的我们 vs FCGS / D-FCGS 对比表提供可筛选的本地候选记录，并明确标注不同 protocol/source group，避免把不公平口径误当主对比。

## 2026-06-26：继续 Stage53 baseline comparison scaffold

### 用户原始问题

用户要求继续推进下一步，补齐 FCGS / D-FCGS 对比框架。

### 当前执行决策

执行 Stage53，将 Stage51 的 Mono-DFCGS high-rate RD rows 和 Stage52 的 FCGS/D-FCGS baseline candidate rows 归一化到统一 comparison scaffold。该阶段不生成最终 fair 主表，而是明确记录 method、dataset/protocol、rate unit、quality fields、local/literature status、fairness flag 和 protocol mismatch notes，为后续 Stage57 final paper RD package 做准备。

## 2026-06-26：继续 Stage54 decision-aware selector analysis

### 用户原始问题

用户先询问当前已经完成的工作；在获得阶段汇总后，要求如果有下一步就继续执行，如果不确定再停下来询问。

### 当前执行决策

执行 Stage54，复用 Stage48 predicted selector RD 和 Stage49 rendered-oracle RD，不重新渲染，分析当前 feed-forward selector 失败是否主要来自候选布局池、fallback 决策或 oracle-layout imitation。该阶段重点输出 decision-aware selector diagnosis，为下一步 adaptive-or-uniform fallback classifier / DP-aware selector training 提供依据。

## 2026-06-26：继续 Stage55 large-scale data preflight

### 用户原始问题

用户要求如果有下一步就继续执行；完成 Stage54 后，下一步是为后续大规模训练/评估检查可用数据和 StreamSplat 原始 protocol readiness。

### 当前执行决策

执行 Stage55，只做只读 preflight，不下载、不预处理、不导出 anchors。该阶段检查默认 DAVIS / YouTube-VOS / RE10K / CO3D 候选根路径、StreamSplat provider 所需目录、Mono-DFCGS 当前 Stage33 dense anchor assets，以及各数据源在最终单目 codec claim 中的使用边界。

## 2026-06-26：Stage51 RD 图重绘和汇报口径更新

### 用户原始问题

用户指出 `rendered_prior_0p1` 不能作为最终部署方案，后续必须改成直接前馈版本；同时认为 Stage51 RD 图太乱，不同 q 和 gap 都用同一颜色点线显示，需要重画。用户还询问当前压缩传输内容是否只是关键帧 GS，并要求后续汇报默认只讲 all-frame PSNR，不讲 middle-only PSNR。

### 当前执行决策

新增 Stage51b clean plotting 脚本，复用已有 Stage51 CSV，不重跑渲染，只重画 all-frame PSNR RD 图。新版图用颜色区分 q8/q10/q12/q16，点标签标注 gap，并额外输出 adaptive-uniform all-frame PSNR delta heatmap。后续对话默认只汇报 all-frame PSNR，除非用户明确要求 middle-only。

## 2026-06-26：整理训练设计、创新点和近期问答日志

### 用户原始问题

用户要求将上一条关于训练过程设计、当前创新点、feed-forward selector 和 optional side-information 探索的回答单独整理到一个日志文件中；同时补记前面若未记录日志的解释性问答。

### 当前执行决策

新增独立日志 `logs/RECENT_TRAINING_INNOVATION_AND_QA_2026-06-26.md`。该文件记录最近关于 Stage3 PSNR 口径、Stage51 最新结果、bits/gap/zlib 含义、uniform/adaptive 区别、当前传输内容、已完成训练、后续训练设计、创新点、feed-forward selector 要求和 side-information 探索边界的整理说明。

## 2026-06-26：Stage56-68 后续计划与持续执行要求

### 用户原始问题

用户要求将完整后续计划记录到日志，并要求后续每次 plan 完都整理记录。用户还要求新建一个文件夹，把每轮 stage 的重要过程、结果和结论都记录为独立文件；先在该文件夹下建立一个文件总结之前轮次结果。随后按计划一直往下执行，除非遇到重大决定，例如最终 adapter 路线选择。

### 当前执行决策

新增 `logs/FUTURE_WORK_PLAN_STAGE56_68.md` 保存 Stage56-68 完整后续计划。新增 `logs/stage_records/` 作为后续逐 stage 记录目录，并创建 `00_previous_rounds_summary.md` 总结 Stage1-55/51b 的关键结果、当前状态和下一步。

## 2026-06-26：执行 Stage56 protocol lock

### 用户原始问题

用户要求按照已记录的后续计划持续执行，并在每个 stage 的独立记录文件中保存重要过程、结果和结论。

### 当前执行决策

执行 Stage56，锁定后续实验协议：默认只汇报 all-frame PSNR，主 rate 为 transmitted Gaussian anchor bitstream MiB/frame，side information 如传输必须计入 total rate，`rendered_prior_0p1` 只作为 oracle/calibrated upper bound，最终 selector 必须 fully feed-forward。

## 2026-06-26：更新 Stage56-70 计划并继续执行

### 用户原始问题

用户补充要求后续大规模训练必须纳入 StreamSplat 使用的数据集，特别是 DAVIS 和 YouTube-VOS，并给出官网链接。随后用户要求按照最新计划开始执行，继续保持每个 stage 独立记录并持续推进。

### 当前执行决策

新增最新计划 `logs/FUTURE_WORK_PLAN_STAGE56_70.md`，补充 DAVIS / YouTube-VOS 下载、路径检查、depth preprocessing、anchor export 和大规模训练阶段。旧 `Stage56-68` 计划保留为历史版本并标注 superseded。继续执行 Stage57 compact anchor codec。

## 2026-06-26：执行 Stage57 compact anchor codec

### 用户原始问题

用户要求如果有下一步就继续执行，不确定时再询问。

### 当前执行决策

继续执行 Stage57：实现 q1-q16 true bit-packing，保留 legacy dtype storage 对比，生成 compact codec size table 和 roundtrip verification。Stage57 完成后继续进入 Stage58 compression RD ablation。

## 2026-06-26：执行 Stage58 compression RD ablation

### 用户原始问题

用户要求如果有下一步就继续执行，不确定时再询问。

### 当前执行决策

继续执行 Stage58：复用 Stage51 all-frame PSNR 和 Stage57 compact rate 表，生成 compression RD ablation CSV、summary JSON 和 RD plots。默认只使用 all-frame PSNR。

## 2026-06-26：执行 Stage59 DAVIS/YouTube-VOS prepare preflight

### 用户原始问题

用户要求如果有下一步就继续执行，不确定时再询问。此前用户明确要求后续数据扩展纳入 DAVIS 和 YouTube-VOS。

### 当前执行决策

执行 Stage59：检查 DAVIS / YouTube-VOS expected roots、StreamSplat provider layout、缺失项和下载/准备 checklist。该阶段不下载数据；若 root 缺失，则 Stage60/61 被数据可用性阻塞。

## 2026-06-27：继续 DAVIS/YouTube-VOS 下载准备与 Stage60 depth preprocessing

### 用户原始问题

用户先询问当前已经完成的内容；随后要求如果有下一步就继续执行，如果不确定如何推进再停下来询问。

### 当前执行决策

继续按 Stage56-70 计划推进数据准备。已尝试下载 DAVIS 和 YouTube-VOS：DAVIS Full-Resolution unsupervised trainval 已下载、解压并完成 DepthAnything V2 depth preprocessing；YouTube-VOS 2019 valid 已下载、解压，但 train split 约 `9.26G`，当前 `/mnt/hdd2tC` 剩余空间不足，未继续下载。随后重跑 Stage59 preflight，DAVIS 已变为 provider-ready 和 anchor-export-ready，YouTube-VOS 仍因缺少 train split 不 ready。Stage60 记录和小型实验输出需要提交并推送，大型数据集、depth PNG、tar/zip 和 checkpoint 保持在 git 外部。

## 2026-06-27：继续 Stage60 收尾和 Stage61 预检查

### 用户原始问题

用户要求：如果有下一步就继续执行；如果不确定如何推进，再停下来询问。

### 当前执行决策

继续收尾 Stage60：补齐 `logs/stage_records/60_depth_preprocess.md`、更新全局日志和 Stage59 post-download 状态，验证脚本和 preflight 输出，然后提交并推送。提交后再评估 Stage61 DAVIS large-scale anchor export 的入口脚本和空间需求；当前 `/mnt/hdd2tC` 只剩约 `4.1G`，不能贸然启动大输出任务。

## 2026-06-27：Stage61 DAVIS anchor export preflight/smoke

### 用户原始问题

用户要求如果有下一步就继续执行；如果不确定如何推进，再停下来询问。

### 当前执行决策

执行 Stage61 的安全版本：先做 DAVIS large-scale anchor export preflight，估算 full gap1/2/4/8/16 输出规模和当前磁盘可行性；由于 `/mnt/hdd2tC` 空间不足，不启动 full all-gap export。随后新增 DAVIS-aware anchor export 脚本，默认只运行一条 sequence 的一个 pair，并用 `train/bear gap16 pair_000000_000016` 完成 smoke export，验证 DAVIS RGB/depth -> StreamSplat Gaussian anchor `.pt` 保存闭环。

### 后续执行更新

根据 Stage61 preflight，DAVIS 单独 gap16 全量估算约 `776.953125 MiB`，在当前空间下可安全尝试。继续运行 DAVIS train+val gap16 partial large-scale export，完成 `90` 个序列、`425` 个 pair 的 anchor 导出；all-gap export 仍保持 blocked。

## 2026-06-27：使用 /data 上完整 DAVIS 继续后续 stages

### 用户原始问题

用户要求检查其他磁盘空间，继续下载更完整的 DAVIS 数据集；YouTube-VOS 暂时不用。下载完成后用户要求先告知路径。随后用户确认使用该数据集继续后续 stages，按之前计划一步步推进。

### 当前执行决策

已检查磁盘空间，`/data` 约有 `1.1T` 可用，适合承载完整 DAVIS 和后续大规模 anchor/checkpoint 输出。DAVIS 官方 Full-Resolution 扩展包已下载并解压到 `/data/hctang/tmp/opencode/datasets/DAVIS_official_downloads/DAVIS`，zip 保留在 `/data/hctang/tmp/opencode/datasets/DAVIS_official_zips`。后续 stages 将优先使用 `/data` 作为 heavy root，避免继续写满 `/mnt/hdd2tC`。YouTube-VOS 暂不处理。

下一步计划：先将 `/data` DAVIS root 接入 Stage60/61 pipeline，生成或复用 depth，导出 DAVIS train/val 多 gap anchors 到 `/data/hctang/tmp/opencode/mono_dfcgs_runs/`，然后进入 Stage62 adapter training infra v2。

### 后续执行更新

已将 `/data` DAVIS train/val root 接入 Stage61：复用旧 root 的 train/val depth，完成 `/data` 上 DAVIS train+val gaps `1/2/4/8/16` 全量 anchor export。最终导出 `12007` pair rows，anchor tensor payload `21950.296875 MiB`，heavy root 为 `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage61_davis_anchor_export_full`。下一步进入 Stage62 adapter training infra v2。

## 2026-06-27：执行 Stage62 adapter training infra v2

### 用户原始问题

用户要求使用 `/data` 上的新 DAVIS 数据集继续后面的 stages，按照之前计划一步步推进。

### 当前执行决策

执行 Stage62：新增 adapter training infrastructure v2，读取 `/data` DAVIS all-gap anchor manifest，支持 train/val split、best checkpoint、final checkpoint、resume training state 和 storage-safe external checkpoint root。先运行小规模 smoke 和 resume 验证，不将其作为 medium/long adapter 质量结果。

## 2026-06-27：执行 Stage63 DAVIS medium adapter training pilot

### 用户原始问题

用户要求继续按照之前计划逐步推进后续 stages。

### 当前执行决策

Stage62 infra 已验证并推送。下一步执行 Stage63 的 medium-training pilot：使用 DAVIS train/val all-gap anchors，覆盖 gaps `2/4/8/16`，扩大 selected rows 和 steps，但先限制为 `128` steps，验证训练曲线、best checkpoint、validation 开销和是否值得继续扩展到 5k+ steps。该 pilot 仍不作为最终 medium/long adapter claim。

### 后续执行更新

Stage63 pilot 已完成：使用 `16` train rows、`8` eval tasks、`128` steps，best eval margin over linear 为 `+0.031141991902842392 dB`，validation curve 单调提升。结果支持后续扩展到更长训练，但目前仍只是 pilot。

## 2026-06-27：执行 Stage64 adapter architecture / teacher study

### 用户原始问题

用户要求按之前说明的顺序继续后续 stages。

### 当前执行决策

Stage63 pilot 已验证更长训练有正向趋势。下一步执行 Stage64 小型 adapter architecture / teacher study：比较 RGB-loss residual adapter 与 dense-gap1 anchor teacher distillation，分别测试 hidden_dim `128` 和 `256`。训练和评估都使用 DAVIS train/val anchors；teacher distillation 只作为训练监督，test-time 输入仍限制为 keyframe anchors + timestamp。

### 后续执行更新

Stage64 ablation 已完成：`rgb_h256` 在 rendered PSNR 上最好，best margin over linear 为 `+0.017721457863302703 dB`；`teacher_h256` 在 dense-anchor teacher MSE 上最好，teacher MSE 为 `0.005198333790758625`。结论是下一步 Stage65 medium adapter training 优先使用 RGB render loss + hidden dim `256`，teacher route 暂作为 secondary ablation。

## 2026-06-27：Stage64 收尾并继续推进

### 用户原始问题

用户要求：如果有下一步就继续执行；如果不确定如何推进，再停下来询问。

### 当前执行决策

Stage64 已完成运行，下一步收尾：更新 Stage64 stage record、全局执行日志和 pitfalls，检查 diff 与 formatting，然后提交并推送 Stage64 script、tracked experiment summaries 和日志。提交后再进入 Stage65 medium adapter training。

### 后续执行更新

Stage64 已提交并推送，commit 为 `0f7dfc2 Run adapter teacher study`。提交时本机没有 git identity 配置，未修改 git config，而是复用最近提交的 `OpenCode <opencode@local>` 作为一次性 author/committer 环境变量完成提交。

## 2026-06-27：执行 Stage65 rgb_h256 medium sanity training

### 用户原始问题

用户要求：如果有下一步就继续执行；如果不确定如何推进，再停下来询问。

### 当前执行决策

Stage64 结论显示 `rgb_h256` 的 rendered PSNR 最好。Stage65 先执行一个 1024-step sanity run：使用 RGB render loss、hidden dim `256`、DAVIS train/val gaps `2/4/8/16`、每 gap 最多 `16` 个 train rows 和 `4` 个 eval rows。若 sanity run 保持正向且无明显 validation collapse，再继续考虑更长 medium run。

### 后续执行更新

Stage65 sanity run 已完成：64 train tasks、16 eval tasks、1024 steps，best step 为 `512`，best margin over q8 linear 为 `+0.05161105795835397 dB`，final margin 为 `+0.04578667635357192 dB`。结果仍为正且没有严重 validation collapse，因此继续执行更长 medium run：`5000` steps、每 gap `32` 个 train rows、每 gap `8` 个 eval rows。

### Medium run 执行更新

Stage65 medium run 已完成：128 train tasks、32 eval tasks、5000 steps，best checkpoint 在 step `4000`，best eval margin over q8 linear 为 `+0.2734731971829376 dB`，final step `5000` margin 为 `+0.21907000507035335 dB`。Best-step 下 gaps `2/4/8/16` 全部为正。下一步应使用 Stage65 best checkpoint 做 full-video/all-frame evaluation，或进入 feed-forward selector dataset/training，并继续保持 rendered oracle 仅作为 upper bound 或 training label。

## 2026-06-27：执行 Stage66 DAVIS feed-forward selector dataset

### 用户原始问题

用户要求：如果有下一步就继续执行；如果不确定如何推进，再停下来询问。

### 当前执行决策

Stage65 已完成并推送。按 Stage56-70 计划进入 Stage66：构建 DAVIS feed-forward selector segment-cost dataset。该阶段只生成 selector training/eval 数据，不训练最终 selector。Features 限制为 encoder-side 信息，包括 segment metadata、endpoint Gaussian-anchor statistics 和 original RGB motion statistics；labels 离线使用 Stage65 best adapter 与 dense gap1 anchors 计算，不作为 test-time input。

### 后续执行更新

Stage66 dataset 已完成：选取 DAVIS train `8` 个序列、val `4` 个序列，共生成 `4608` 个 segment rows，其中 train rows `3072`、eval rows `1536`。Endpoint-anchor/RGB-motion features 与 offline anchor-space adapter error label 有明显相关性，最高 Pearson 为 `endpoint_anchor_l1=0.7819601460791495`。同时发现 Stage65 RGB adapter 在 anchor-space MSE 上 `0/4608` 段优于 linear，说明该 label 只能作为 difficulty proxy，不能直接代表 rendered PSNR。下一步 Stage67 需要训练 feed-forward segment-cost predictor，并最好补 rendered-distortion label subset 后再做 selector RD claim。

## 2026-06-27：执行 Stage67 DAVIS selector predictor training

### 用户原始问题

用户要求：如果有下一步就继续执行；如果不确定如何推进，再停下来询问。

### 当前执行决策

Stage66 已完成并推送。继续 Stage67：在 Stage66 DAVIS selector dataset 上训练和验证 feed-forward segment-cost predictor。先做 ridge baselines，包括 length-only、RGB-motion、endpoint-anchor 和 full-feature variants。该阶段只预测 Stage66 anchor-space proxy label，不做最终 selector RD claim。

### 后续执行更新

Stage67 predictor validation 已完成：`full_feature_ridge` 最好，eval RMSE log `0.01746532908957139`，eval Pearson `0.9103771172408511`，eval Spearman `0.9017146144293104`。Endpoint-anchor features 明显强于 length-only / RGB-motion-only。该结果说明 feed-forward proxy cost predictor 可训练，但仍只针对 Stage66 anchor-space proxy label；下一步 Stage68 需要用预测 cost 做 deterministic DP selection，并进行 rendered/full-video validation，不能直接声称 selector 提升 all-frame PSNR。

## 2026-06-27：执行 Stage68 DAVIS feed-forward selector rendered validation

### 用户原始问题

用户要求：好的，继续一直往下做。

### 当前执行决策

继续 Stage68：使用 Stage67 `full_feature_ridge` 的 feed-forward segment-cost predictor，对 Stage66 DAVIS eval sequences 生成 deterministic DP keyframe selections，并用 Stage65 best `rgb_h256` adapter 做 rendered validation。对比 `uniform` 与 `predicted_full_feature_dp`，主指标为 all-frame PSNR；selection 过程不使用 rendered oracle、PSNR labels、dense-anchor labels 或 reconstruction lookahead。

### 后续执行更新

Stage68 rendered validation 已完成：在 DAVIS eval sequences `bmx-trees/car-shadow/goat/soapbox` 和 gaps `4/8/16` 上共 `12` 个对比点，`predicted_full_feature_dp` 相对 uniform 的 adapter all-frame PSNR 平均提升 `+0.030738190041048163 dB`，其中 `7/12` 个点为正。结果是 mixed-positive，不足以作为最终 selector claim；`goat gap8` 为明显负例，下一步需要 rendered-distortion labels 或 fallback calibration。

## 2026-06-27：执行 Stage69 selector fallback calibration analysis

### 用户原始问题

用户要求：好的，继续一直往下做。

### 当前执行决策

Stage68 暴露 selector 不稳定问题，因此 Stage69 先做 fallback calibration analysis，而不是直接进入 side-info。该阶段不重渲染，复用 Stage68 rendered outcomes 作为离线 calibration labels，比较 always-uniform、always-predicted、oracle-positive fallback 和 leave-one-sequence-out threshold fallback。目标是判断简单 fallback 是否能降低负点风险。

### 后续执行更新

Stage69 fallback calibration analysis 已完成：fixed predicted mean all-frame PSNR delta 为 `+0.030738190041048163 dB`，但最差点 `-0.10978492809701024 dB`；oracle-positive fallback 上界为 `+0.04350771650468873 dB` 且无负点；same-data threshold 上界为 `+0.03745316604097404 dB` 且无负点；leave-one-sequence-out threshold fallback 反而为 `-0.01170162890067535 dB`。结论是简单 threshold fallback 不稳定，下一步需要更多 rendered labels 或 decision-aware fallback classifier。

## 2026-06-27：执行 Stage70 scoped DAVIS RD package

### 用户原始问题

用户要求：好的，继续一直往下做。

### 当前执行决策

Stage69 已完成并推送。继续 Stage70：把当前 Stage68/69 DAVIS eval-subset 结果整理成 scoped RD package，包括 rate table、all-frame PSNR table、selector delta table、method summary、RD curve 和 FCGS/D-FCGS baseline status。该阶段不重渲染，也不声明最终完整 benchmark。

### 后续执行更新

Stage70 scoped DAVIS RD package 已完成：生成 rate table、all-frame PSNR table、selector delta table、method summary、baseline status 和 RD curve。当前 selector aggregate 为 `7/12` positive，mean adapter all-frame PSNR delta `+0.030738190041048163 dB`。Stage70 明确标记 FCGS/D-FCGS 尚未做 local apples-to-apples baseline，因此它是 scoped progress package，不是最终 benchmark。

## 2026-06-27：执行 Stage71 baseline availability preflight

### 用户原始问题

用户询问当前进度后，要求：如果有下一步就继续执行；如果不确定如何推进，再停下来询问。

### 当前执行决策

继续 Stage71：对本机 FCGS/D-FCGS/CWGS baseline 代码、旧实验结果和 DAVIS scoped eval 适配状态做轻量 preflight。该阶段不运行重型训练/渲染，只输出 baseline candidate inventory、protocol/fairness status、missing-fields checklist 和 summary，用于判断后续能否复用旧 artifacts 或需要新增 DAVIS local fair baseline runner。

### 后续执行更新

Stage71 baseline availability preflight 已完成：本机存在 FCGS 和 D-FCGS 代码，但二者没有 DAVIS/Mono-DFCGS anchor adapter；旧 FCGS/D-FCGS/CWGS artifacts 均为非 DAVIS 或非 Stage70 scoped protocol。当前没有任何 external baseline 可直接作为 DAVIS apples-to-apples RD 点。下一步应优先实现 FCGS DAVIS wrapper 或明确 D-FCGS 是否能在不引入多视角信息的前提下安全适配。

## 2026-06-27：执行 Stage72 原 StreamSplat DAVIS baseline 与低 PSNR 诊断

### 用户原始问题

用户要求：按照最新计划继续操作，并明确操作前先把 plan 记录到日志，操作后也要记录。当前优先级是：先在 DAVIS 上跑原有方法 baseline，看看原方法结果是否正常；再修复我们 Stage70 结果偏低的问题；这两步做完之后先找用户汇报。

### 当前执行决策

Stage72 先执行 Phase A：复现原 StreamSplat 在 Stage70 相同 DAVIS val subset 上的 scoped baseline，优先覆盖 `bmx-trees/car-shadow/goat/soapbox` 和 gaps `4/8/16`。若原方法 baseline 也低，优先诊断 DAVIS 数据、depth、resize、metric 或 frame alignment；若原方法 baseline 正常，再进入 Phase B：排查并修复 Stage70 Gaussian-anchor-only pipeline 的低 PSNR，包括 target RGB resize、keyframe direct render、float-vs-q8 anchor、anchor-to-renderer bridge、checkpoint 加载和 frame index/timestamp 对齐。完成 Phase A/B 后暂停并汇报，不继续大规模训练。

### 操作后记录

Stage72 Phase A 已完成：原 StreamSplat full dynamic Gaussian baseline 在相同 DAVIS scoped subset 上正常，mean all-frame PSNR 为 gap4 `23.244598036349643`、gap8 `20.682715912446714`、gap16 `18.353465837484507`。

Stage73 Phase B 诊断已完成：Stage70 q8 adapter uniform 数值可被完全复现，未发现 RGB/resize/frame index/render bridge/checkpoint 加载口径 bug。Stage70 低于原方法主要来自两部分：当前 Gaussian-anchor-only path 只传/渲染 static anchors，丢弃原 StreamSplat full dynamic Gaussian fields；q8 anchor 量化又额外降低 keyframe render quality。完成 Phase A/B 后按用户要求暂停并汇报，不继续大规模训练。

## 2026-06-27：分析 Stage72 原方法结果与实际结果差距

### 用户原始问题

用户指出：Stage72 原方法结果为什么和实际结果相差这么多，需要想办法分析原因。

### 当前执行决策

新增 Stage74 诊断：先静态核查 Stage72 runner 与官方/实际 StreamSplat evaluation protocol 的差异，包括数据集 split、frame sampling、DAVIS 分辨率、resize、mask/foreground evaluation、depth 输入、checkpoint/config、metric 口径和 keyframe/gap 定义。若静态核查不足，再运行轻量验证脚本，运行代码前先检查 `nvidia-smi`。

### 后续执行更新

Stage74 已完成关键诊断：Stage72 与实际/论文结果相差大的主因是 protocol 不一致，而不是 checkpoint 加载错误。改成 full DAVIS val、sliding fixed windows、`256x256` metric 和 non-input middle-frame scope 后，gap5 middle PSNR 为 `23.004337221027775`，接近论文 Middle-4 `23.66`；gap8 middle PSNR 为 `21.56004909948801`，接近论文 8-frame interval `22.10`。用户要求把当前输出和总结单独记录到文档，并继续后续 stages；总结文档路径为 `docs/STAGE72_74_STREAMSPLAT_DAVIS_DIAGNOSIS_SUMMARY.md`。

## 2026-06-27：继续后续 Stage75

### 用户原始问题

用户要求：把当前输出和总结单独记录到文档，并给出路径，然后继续做后面的 stages。

### 当前执行决策

已创建总结文档 `docs/STAGE72_74_STREAMSPLAT_DAVIS_DIAGNOSIS_SUMMARY.md` 并完成 Stage74 提交推送。下一步先执行 Stage75：从 Stage74 full DAVIS val official-style outputs 中生成 corrected StreamSplat paper-protocol baseline package，用于后续报告/对比时替代容易误解的 Stage72 scoped 数值。Stage75 是 CPU-only 汇总，不启动新的 GPU 渲染。

### 后续执行更新

Stage75 已完成 corrected StreamSplat paper-protocol DAVIS package：Middle-4 local middle PSNR `23.004337221027775` vs paper `23.66`，8-frame interval local middle PSNR `21.56004909948801` vs paper `22.10`。输出目录为 `experiments/stage75_corrected_streamsplat_paper_protocol_package/`。

## 2026-06-27：继续 Stage76 static anchor quantization sweep

### 用户原始问题

用户要求继续做后面的 stages。

### 当前执行决策

Stage76 先做轻量但关键的 static anchor quantization sweep：针对 Stage73 发现的 q8 keyframe render quality 大幅下降，评估 float16/q6/q8/q10/q12/q16 static anchors 的 direct keyframe render PSNR 与估算 anchor payload。该阶段先覆盖 Stage70 scoped DAVIS val sequences，不启动长训。

### 后续执行更新

Stage76 已完成：q8 direct keyframe render 在 scoped DAVIS val 上相对 float16 掉 `-2.69045491882148 dB`（512 metric），q10 仅掉 `-0.3808268306669724 dB`，q12 基本无损 `-0.02718619126847699 dB`。后续 RD 应加入 q10/q12 operating points，而不能只用 q8 代表 anchor codec。

## 2026-06-27：继续 Stage77 q-bit full-video RD sweep

### 用户原始问题

用户要求继续做后面的 stages。

### 当前执行决策

Stage77 在 Stage76 direct keyframe quantization sweep 之后，进一步评估 q8/q10/q12 对 full-video anchor-only RD 的影响。该阶段覆盖 Stage70 scoped DAVIS val sequences、uniform gaps `4/8/16`、linear anchor 和 Stage65 adapter 两种方法，输出 all/middle/given PSNR 与 q-bit static anchor rate。

### 后续执行更新

Stage77 已完成：q10/q12 明显提升 all-frame PSNR 和 given-keyframe PSNR，但 Stage65 adapter 的 middle-frame PSNR 几乎不变。例如 gap4 adapter 从 q8 `20.57270098931695` all PSNR 提升到 q12 `21.284133638556813`，但 middle PSNR 只从 `18.247303018014392` 到 `18.256196169477683`。结论是 q10/q12 应加入 RD operating points，但主要剩余瓶颈是 static-anchor-only dynamic prediction/modeling。

## 2026-06-28：继续 Stage78 integrated DAVIS RD package

### 用户原始问题

用户要求：按照后续计划继续往下做，一点点把这些步骤做好；当前先不急着做 FCGS/D-FCGS 对比实验。

### 当前执行决策

从 Stage78 开始继续执行：整合 Stage75 corrected StreamSplat paper-protocol baseline 和 Stage77 q8/q10/q12 anchor-only RD sweep，生成新的 DAVIS RD package。该阶段聚焦我们自己的创新线，输出 anchor-only RD table、rate table、method summary、与 corrected StreamSplat reference 的差距表和 RD curve。Stage78 为 CPU-only package generation，不启动训练或重渲染。

### 后续执行更新

Stage78 已完成 integrated DAVIS RD package，输出目录为 `experiments/stage78_integrated_davis_rd_package/`。当前最佳 anchor-only scoped 点为 q12 adapter，mean all PSNR `19.074880702510743`、mean middle PSNR `17.100601790202944`、mean rate `0.11167752575591416 MiB/frame`。与 corrected StreamSplat reference 的 middle-frame gap 仍约 `4.5-4.8 dB`，后续应转向 stronger adapter training、rendered-label selector 和 dynamic side-info。

## 2026-06-28：继续 Stage79 adapter training task manifest

### 用户原始问题

用户要求继续按计划推进我们自己的创新点和训练部分。

### 当前执行决策

Stage79 开始 Gaussian adapter 训练线：先构建训练任务 manifest，而不是立即长训。任务覆盖 DAVIS train/val dense gap1 anchors，codecs `q10/q12`，reference gaps `4/8/16`，每行描述 left/right keyframe anchors、target frame、normalized time 和 target RGB path。Stage79 为 CPU-only manifest generation，为 Stage80 adapter smoke/long training 做输入准备。

### 后续执行更新

Stage79 已完成 adapter training task manifest：覆盖 DAVIS train `60` sequences / `4209` frames 和 val `30` sequences / `1999` frames，共 `31108` 个 q10/q12、gap4/8/16 中间帧训练/评估 tasks。输出目录为 `experiments/stage79_adapter_training_task_manifest/`。

## 2026-06-28：继续 Stage80 adapter training smoke

### 用户原始问题

用户要求：如果有下一步就继续执行；如果不确定如何推进，再停下来询问。

### 当前执行决策

Stage80 继续 Gaussian adapter 训练线，但先做小规模 smoke，而不是直接长训。计划新增 Stage80 task-manifest adapter training smoke：从 Stage79 task CSV 读取 q10/q12、gap4/8/16 的训练/评估任务，加载 Stage61 gap1 anchor `.pt` 的指定 side，模拟 q-bit static anchor 输入，用 rendered RGB loss 训练 `GaussianAnchorDynamicPredictor`，并与 linear anchor baseline 做 eval 对比。Heavy checkpoint 写入 `/data/hctang/tmp/opencode/...`，仓库内只保存脚本、小型 CSV/JSON/report 和日志。

### 后续执行更新

Stage80 smoke 已完成：新增 `scripts/run_stage80_adapter_training_smoke.py`，最终默认输出覆盖 q10/q12 和 gaps `4/8/16` 的小规模训练闭环。运行前检查 `nvidia-smi`，GPU4 空闲，因此使用 `CUDA_VISIBLE_DEVICES=4`。最终 smoke 使用 `6` 个 train tasks、`6` 个 eval tasks、`6` 个 training steps。best/final eval PSNR 为 `17.669478613164078`，linear baseline 为 `17.669089783599734`，margin 为 `+0.00038882956435060123 dB`；Stage65 reference adapter 在同一小 eval 上 margin 为 `+0.15900605544237814 dB`。该结果只验证 loader/render/loss/checkpoint 闭环，不代表长训效果。输出目录为 `experiments/stage80_adapter_training_smoke/`，checkpoint 在 `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage80_adapter_training_smoke/`。

## 2026-06-28：继续 Stage81 adapter pilot fine-tuning

### 用户原始问题

用户要求继续执行下一步。

### 当前执行决策

Stage81 在 Stage80 smoke 通过后做 controlled adapter pilot fine-tuning。先对 Stage80 trainer 做最小扩展：支持从已有 Stage65 `rgb_h256` adapter checkpoint 初始化，并支持自定义 stage/run 输出前缀，避免 Stage81 输出仍使用 Stage80 文件名。随后用 Stage79 manifest 跑小规模但比 smoke 更有意义的 q10/q12、gap4/8/16 pilot：`48` train tasks、`18` eval tasks、`48` steps、每 `12` steps eval。运行前继续检查 `nvidia-smi`，heavy checkpoint 仍写到 `/data/hctang/tmp/opencode/...`。

### 后续执行更新

Stage81 pilot 已完成：新增 Stage80 trainer 的 `--init_checkpoint` 和自定义输出前缀能力，并从 Stage65 `rgb_h256` checkpoint 初始化运行 q10/q12、gap4/8/16 pilot。运行前检查 `nvidia-smi`，GPU4 空闲，因此使用 `CUDA_VISIBLE_DEVICES=4`。最终配置为 `48` train tasks、`18` eval tasks、`48` steps，best step 为 `24`。reference/initial margin 为 `+0.061880396487980015 dB`，best margin 为 `+0.10681560967982436 dB`，final margin 为 `+0.10644771658749658 dB`。按 gap 看，best gap16 `+0.403032388467782 dB`、gap8 `+0.14113360277949102 dB`，但 gap4 仍为 `-0.18006936459308479 dB`。输出目录为 `experiments/stage81_adapter_training_pilot/`，heavy checkpoint 在 `/data/hctang/tmp/opencode/mono_dfcgs_runs/stage81_adapter_training_pilot/`。

## 2026-06-28：继续 Stage82 broader adapter evaluation

### 用户原始问题

用户要求继续执行下一步。

### 当前执行决策

Stage82 不继续训练，先检查 Stage81 best checkpoint 是否只是在 18-task 小 eval 上过拟合。使用 Stage80/81 trainer 的 `steps=0` eval-only 路径，在更宽的 DAVIS val held-out slice 上评估 Stage81 best，并在同一 slice 上保留 Stage65 reference adapter 与 linear anchor baseline。计划覆盖 q10/q12、gaps `4/8/16`，使用 `60` eval tasks。运行前继续检查 `nvidia-smi`，heavy checkpoint copy 写到 `/data/hctang/tmp/opencode/...`，仓库内保存 summary/report/CSV。

### 后续执行更新

Stage82 broader eval 已完成：在 60-task DAVIS val slice 上，Stage81 best 的 mean margin 为 `+0.10111637946599329 dB`，Stage65 reference adapter 的 mean margin 为 `+0.11064826351743612 dB`，因此 Stage81 pilot 没有在更宽 eval 上超过 Stage65。按 gap 看，Stage81 best 相对 linear 为 gap16 `+0.3698363934848712 dB`、gap8 `+0.06382222406404825 dB`、gap4 `-0.0773488024139178 dB`；Stage65 reference 为 gap16 `+0.3343528251045902 dB`、gap8 `+0.05364375708512303 dB`、gap4 `-0.010726898647836793 dB`。结论：Stage81 pilot 可能改善长 gap，但损害 gap4，暂不能替代 Stage65 作为 current best adapter。输出目录为 `experiments/stage82_adapter_pilot_broader_eval/`。
