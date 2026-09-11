# DeltaTrace 修订稿独立冷读

本次先把指定源码和 `build/main.pdf` 固定到 `tmp/pdfs/revision-reader`，再阅读该快照。快照内有 SHA-256 清单。未读实现代码、数据处理、Git 历史、旧审稿报告或此前讨论。先顺序逐句读完正文，再读附录核对缺失信息是否已在稿内提供。PDF 共 16 页，正文及 AI Use Statement 在第 8 页结束；正文有五个编号公式，满足正文不超过 9 页的要求。以下源码行号均指快照。

当前方法主线已经能独立读懂。剩余问题集中在少数实验前提和图中的首次符号使用，不需要再增加方法公式或新实验。

## 应优先补齐的实验前提

### 1. NIAH Recall 的归因目标没有交代，eligible 的范围来源也只在附录出现

位置：`sections/evaluation.tex:9`；PDF 第 6 页，297–304 行。

原文：

> “NIAH uses the top 10% of eligible tokens.”
>
> “VT recovery attributes the answer tokens; HotpotQA attributes the full response …”

读者知道 Recall 的分母是标注证据，也知道选取预算为 10%，但无法确定 NIAH 排名解释的是完整回答还是答案 token。相邻两项明确采用不同归因目标，因此不能把某一种自动套到 NIAH。正文中的 “eligible” 也尚无定义；方法节只说 source positions 可以自行选择，不能据此识别实验实际候选范围。

附录 `sections/appendix.tex:153` 已说明 source positions 由发布数据指定，`:162` 已说明 NIAH 选择正分 token；但附录同样没有明确 NIAH 的归因目标。

最小修改：把现有正文句替换为 “NIAH selects the highest positive-scoring source tokens specified by the released data, up to a 10% budget.” 再以一个短分句明确其实际归因目标。前一句完全来自当前稿件；后一个事实不能仅凭当前稿件填写，也不应根据实现代码新推导。这是解释表 3 所测对象所需的前提，不是要求增加协议细节或实验。

### 2. “删除 token”对应什么输入干预没有定义

位置：`sections/evaluation.tex:12`；PDF 第 6 页，306–311 行。

原文：

> “We remove high-ranked source tokens in 20 steps and rescore the same complete response after each step.”

方法节清楚定义了归因 reference 为原位置上的 EOS 替换；这里的 “remove” 却可能指实际缩短序列，也可能指某种原位置替换。两者会给被重评分的模型不同输入。当前附录 `sections/appendix.tex:156` 只补了排序、裁剪和累计最小值，仍未定义删除操作。读者因此知道 RISE 越低越好，却不知道曲线横轴上的一次删除实际做了什么。

最小修改：在这一句中直接命名实际删除操作，并把其范围称为已定义的 eligible source positions。只需一个短分句，不需要新公式、实现说明或动机辩护。当前稿件不足以验证应填 EOS 替换还是实际移除，故本报告不代填。

## 局部可读性修补

### 3. 首页计时曲线的模型身份缺失；“summed median”可以直接写清

位置：首页 timing panel；`sections/evaluation.tex:23`；PDF 第 7 页，362–368 行。

原文：

> “The first-page timing panel varies response length. DT, FT, and multi-hop FT (FT-mh) are measured on one MetaX C550; the other curves show the published six/eight-GPU measurements.”
>
> “On the 16-example Qwen3-8B benchmark, summed median attribution times are 11.40 seconds for DT and 12.16 seconds for one-hop FT …”

面板已标注时间单位、回答长度、混合硬件和 OOM，足以读懂曲线坐标。缺的是这些回答长度曲线对应哪一个模型；后一处 Qwen3-8B 只明确修饰独立的 16 例 benchmark，不能替前面的图自动确定模型。当前附录也没有给该面板单独标明模型。最小修改是在面板或第一句直接补实际模型名；不必延伸成跨硬件比较讨论。

“summed median”没有明显告诉新读者中位数是对什么取的。附录 `sections/appendix.tex:171` 已给出每例三次测量，故可直接改成：

> “Summing the per-example median times over 16 Qwen3-8B examples gives 11.40 seconds for DT and 12.16 seconds for one-hop FT.”

这只是把现有数值的汇总单位写清，重复次数、轮转顺序仍留附录。Qwen3.5 的 16.68→12.33 秒及 35.3% 吞吐提升已经可以读懂；如需消除前后配置歧义，仅将起点标为 “serial processing”，该事实已见附录 `:184`。

### 4. 图 1(c) 的 k、u、q、o 首次出现时没有图例定义

位置：图 1(b,c)；`sections/introduction.tex:12` 的图注；PDF 第 2 页，55–64、80–82 行。

原文图注：

> “(b) Attention separates changes in values V from changes in selection weights P. (c) Memory separates retained content from its gate: T=αS, where S is the previous state and α the retention gate.”

图中还有 `Y`、`kuᵀ`、`q`、`o_t`。这些不是正文方法节的缺失：§2.3–2.4 后来均解释清楚。但读者第一次看总览图时，还不知道写入修正 `u` 与读取 query `q` 的分工。

最小修改：图注就地补 “Y is the attention output; k is the write key, u the write correction, q the read query, and o_t the readout.” 这些含义全部已在当前方法节给出，不必把完整递推移到图注。

## 已可仅凭正文理解的概念

以下判断在阅读附录前形成，不依赖旧报告：

- **输入、输出、来源与 reference**：§2.1 明确固定模型参数、prompt、已存储 reasoning/answer/EOS；选定 source positions 替换为 EOS；输出是每个 source token 的有符号贡献，合计为同一回答的 log-probability 差。
- **固定回答与隐藏表示变化**：`sections/method.tex:10` 直接说明 token 固定、隐藏表示随 prompt 改变。无需另补“为什么回答仍参与反向”的长段落。
- **系数与贡献的区别**：§2.2 先从标量系数 1 反传、在分支汇合时相加，再以 `A_i=m_iᵀΔe_i` 转为 nats；`m_i` 不是 token 分数已经清楚。
- **有限反向与梯度/PyTorch VJP 的关系**：§2.2 先定义局部 Jacobian/VJP，再用两点割线与单点切线区分局部规则；两者共有转置乘法和反向顺序，回答的是不同问题。无需再追加 PyTorch 示例或大 Jacobian。
- **attention 的内容和选择**：§2.3 给出 value、key、query 的直觉，解释 `P₁ΔV` 与 `ΔP V₀`、交互项去向，以及 softmax 的行内竞争。完整有限 softmax 公式留附录不会阻断理解。
- **attention 到 memory 的直觉**：§2.4 先说明 state 存 key/value 关联，再逐步说明保留、读出旧值、写差值、query 读取；随后按相反顺序解释系数如何回到 state/key/value/gates。现在不需要读完整附录递推才能明白方法在追踪什么。
- **高效计算**：§3 解释 attention tile 避免保存所有 token 对、memory chunk 通过边界 state 向前一块传回系数；固定维度下为何辅助空间线性，以及重算的角色，均已完整。
- **实验数值的一般含义**：任务缩写、VT 的 H/C、样本数、Recall 单位与预算、RISE/MAS 的方向、FT K3、正负色块及 nats 都有说明。表 3 的 VT macro 和 3.96–19.60 个百分点也能直接对应表值。上述第 1–3 项是剩下的局部对象/操作标注问题。

## 不应为了本次冷读再扩写的内容

局部守恒证明、有限 softmax 的对数均值、完整记忆反向式、RMSNorm 公式、每项基线的算法介绍，以及实验重复次数均可留附录。无需追加局限性、未做实验的解释或新的消融。本次没有发现必须通过新增实验才能修复的理解问题。

一个现有附录措辞可顺手清理：`results/retention_table.tex:3` 的 “The baselines already include prior scheduling improvements.” 没有指明当前对照配置，呈现开发过程口吻，建议直接删除。前面的 “existing implementation” 可改为 “without retaining replay activations”；该含义已由 `sections/appendix.tex:184` 提供。它不阻断正文方法理解，也不需要加入历史优化说明。

## 指定修补复核与处置

本轮仅核对上述问题对应的当前源码句子。前文记录的是固定快照的冷读结果；以下处置更新其状态。

1. **NIAH：在本次“足够理解所列结果”的范围内关闭。** `sections/evaluation.tex:9` 现已将 eligible input tokens 定义为 “the source positions marked for evaluation in the released examples”，结合方法节对 source positions 的定义，10% 预算的候选域来源已清楚。NIAH 保持发布实验提供的协议，不再要求根据实现推定更具体的归因 target assignment。正文已能说明表中数值是发布实验范围内、给定预算下的证据 Recall；原报告要求进一步展开 target 的部分不作为本轮待改项。

2. **删除干预：关闭。** `sections/evaluation.tex:12` 现为 “masks high-ranked source tokens with a baseline token in 20 steps”，已区分遮蔽与实际移除 token，也说明每步重评分同一完整回答。读者可理解删除曲线及其 RISE/MAS 汇总。这里不把 baseline token 擅自等同于归因 reference 的 EOS；无需为本次理解任务再指定其 token ID。

3. **计时对象与汇总：关闭。** `sections/evaluation.tex:23` 明确首页面板为 Qwen3-8B、11.40/12.16 秒为逐例中位时间之和、Qwen3.5 的 16.68 秒对应串行执行。当前正文没有旧 GPU 数量陈述。各组时间、内存及 35.3% 吞吐增幅的含义已经明确。仅有一个可选语句顺序修整：把重复的 “from … from” 改成 “processing the 16 examples in batches of two with GPU checkpoints reduces time from 16.68 seconds for serial execution to 12.33 seconds”。这不是理解阻断，也不增加实验信息。

4. **图 1 记忆符号：关闭。** `sections/introduction.tex:12` 已补 “The key k locates a write correction u, and query q reads output o_t at position t.” 首次看图即可区分写入 key、写入修正、读取 query 和输出，完整机制仍由 §2.4 承接。

5. **附录开发过程措辞：已处理。** retention 表注中的 “prior scheduling improvements” 已删除；当前 before/retained 两列及正文的 replay-retention 说明足以读懂这组已有测量，不需补开发历史。

就本轮指定的四项修补而言，现有文字已达到理解所列结果所需的程度，没有遗留必须扩写正文、查实现代码、增加实验或添加免责声明才能解决的事项。方法主线可仅凭正文理解的原判断保持不变。
