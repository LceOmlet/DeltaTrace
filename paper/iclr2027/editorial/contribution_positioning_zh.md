# DeltaTrace：贡献定位与论证依据

论文的中心判断是：**输入既提供证据，也通过选择与记忆改变证据的使用；DeltaTrace 将这两类作用统一追溯为对完整固定回答的有符号贡献。**

这是一篇方法论文。任务是回答归因；内容与控制的联合有限传播是核心方法；解析算子、贡献守恒和分块实现共同解释它的优势。引言以剧作家／作曲家的实际样例贯穿任务与机制。

## 相对于相关工作的具体定位

| 研究脉络 | 相关工作的研究对象 | DeltaTrace 集中的推进方向 |
|---|---|---|
| 参考输入与预测差分 | Integrated Gradients 将参考路径上的敏感度积分为输入贡献；DeepLIFT 传播参考激活差分。 | 将有限差分传播具体化为完整回答的 log-probability 归因，并为归一化注意力和 GDN 构造可组合的解析规则。 |
| 相关性传播 | Conservative Propagation、AttnLRP 和 MambaLRP 将守恒归因用于 Transformer 或选择性状态空间计算。 | 同时传播实际内容变化与输入相关的控制变化，使查询、键、保留门与写入门进入同一个有限回答差分的解释。 |
| 表示分解与信息流 | ALTI、DecompX、Information Flow Routes 与 FlashTrace 分析表示混合、预测路径及多 token 目标的来源；LaTIM 分解 Mamba 的 token 交互。 | 用明确的回答分数差作为共同计量对象，将输入文字的内容路径与控制路径累加到相同的有符号分数；词段贡献由其 token 分数直接求和。 |
| 高效序列计算 | FlashAttention 以 tile 组织注意力；Flash Linear Attention 提供硬件高效的线性注意力实现；Gated Delta Networks 以衰减和 delta-rule 更新组织状态。 | 通过 FA 风格 tile 归约、原生 FLA chunk 状态伴随、矩阵乘法和 scan，实现完整有限传播对加速注意力／线性注意力路径的支持。 |
| 反事实信用与策略学习 | 策略梯度中的动作无关基线保留期望更新；COMA 用其他智能体动作固定时的单动作平均构造反事实优势。 | 将有限干预信用视角接到精确优势信号，提出借助结构估计奖励信用的后续研究方向；附录明确等式成立的参考分布和轨迹平均条件。 |

## 机制如何导出优势

| 机制 | 直接产生的解释能力或计算性质 | 论文证据 |
|---|---|---|
| 原输入与参考输入共享同一固定回答；以回答 log-probability 差作为标量目标 | 多个回答位置共享同一归因目标，全部来源分数具有明确的符号和 nats 单位。 | `sections/method.tex` 的目标定义；正式评测目标；完整回答加 EOS 的保存记录。 |
| 每个算子将有限输出变化分配给输入变化；反向系数在共享来源相加 | 局部解释组合成全局解释，全部输入贡献之和等于回答分数差。 | `sections/appendix.tex` 的有限链式法则与贡献守恒证明。 |
| Attention 采用原执行的选择权重传递内容变化，同时传播权重变化；解析 softmax 保留行内竞争 | 来源同时获得内容作用和选择作用的贡献，联合变化中的交互项有固定归属。 | PV 分解、对数均值 softmax 恒等式、共享 Q/K 规则；图1(b)。 |
| GDN 对保留、检索、误差写入和读取使用同一有限分配原则 | 较早输入经状态记忆到后续回答的贡献，以及改变读取／保留／写入的控制贡献，均有明确传播路径。 | 完整 GDN 有限递推；`deltatrace/clean/` 固定入口；图1(c)。 |
| tile 内处理配对交互、chunk 边界传递状态系数 | 固定维度和 chunk 大小时，辅助存储随序列长度线性增长。 | `sections/computation.tex`；附录存储账目；可追溯 FA／FLA 实现。 |
| 完整有符号分数与真实文本对齐；固定任务上的配对评估 | 图1显示剧作家／作曲家的正负名字贡献；配对开发实验显示证据召回和 MAS 的改善。 | 图1及 `figures/data/overview_role_case.json`；`research/temporary/development16_20260909/summary.json`。 |

## 论文贡献的组织

1. **有限干预定义下的有符号信用。** 固定模型与完整回答，比较参考和原输入，明确被分配的干预总效应；有限链式法则将局部分配组合为全局守恒，符号表达支持与反向作用。
2. **统一的内容与控制机制。** Attention 与 GDN 的解析规则同时追踪信息本身及其选择、保留、写入和读取，将来源的不同作用汇入相同的回答差分。
3. **支持加速计算的实现。** FA 风格 tile 和原生 FLA chunk 状态伴随实现同一套有限规则，兼容注意力与混合架构，并给出序列线性辅助存储界。

这三条由同权重、同输入、同回答的配对开发评价和真实有符号文本图支撑。RL 放在引言结尾形成研究视角：准确平均的动作反事实回报差等于优势函数，从而产生与精确动作价值相同的期望策略梯度信号；有限传播用于奖励信用估计、方差与时间稳定性的研究由此获得明确目标。

审稿说服力来自“定义—机制—实现—证据”的连续关系。因果含义落实到固定模型和回答下的输入干预；有符号落实到可加的总效应分配；效率落实到实际 FA／FLA 操作和存储界。相关工作按其解释对象组织，FlashTrace 在引言中只点名一次。概括采用可以由定义、代码、证明和表格核查的具体判断。

完整主结果现已接入 Qwen3-8B 的 13 个任务、1,243 例，比较 Perturbation、REAGENT、CLP、IFR、AttnLRP、FT 和 DT 七种方法。DT 的 RISE 在 9/13 个任务最低，MAS 在 11/13 个任务最低，六个检索任务的 recovery 均最高。按要求不展示 VT 与 HotpotQA 的 recovery，保留两者 RISE/MAS。

附录开发实验使用两个模型，每个模型各 8 个检索与 8 个 MoreHopQA 案例。检索 Recall@10% 在 Qwen3-8B 上为 77.52%，配对基线为 56.45%；Qwen3.5-9B 上为 73.35%，配对基线为 63.92%。对应增量为 21.07 与 9.43 个百分点。四个模型／任务组合的 MAS 均值均更低。附录表 5 保持这一明确开发样本范围，并列出 signed RISE、positive MAS 与召回数据；这些开发数据不与完整主表混合。

## 经检索核对的主要文献

- [Integrated Gradients，ICML 2017](https://proceedings.mlr.press/v70/sundararajan17a.html)
- [DeepLIFT，ICML 2017](https://proceedings.mlr.press/v70/shrikumar17a.html)
- [Conservative Propagation，ICML 2022](https://proceedings.mlr.press/v162/ali22a.html)
- [AttnLRP，ICML 2024](https://proceedings.mlr.press/v235/achtibat24a.html)
- [ALTI，EMNLP 2022](https://aclanthology.org/2022.emnlp-main.595/)
- [DecompX，ACL 2023](https://aclanthology.org/2023.acl-long.149/)
- [Information Flow Routes，EMNLP 2024](https://aclanthology.org/2024.emnlp-main.965/)
- [MambaLRP，NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/hash/d6d0e41e0b1ed38c76d13c9e417a8f1f-Abstract-Conference.html)
- [LaTIM，ACL 2025](https://aclanthology.org/2025.acl-long.1194/)
- [FlashTrace，2026](https://arxiv.org/abs/2602.01914)
- [FlashAttention，2022](https://arxiv.org/abs/2205.14135)
- [Gated Delta Networks，ICLR 2025](https://arxiv.org/abs/2412.06464)
- [Flash Linear Attention，官方仓库与引用信息](https://github.com/fla-org/flash-linear-attention)
- [Policy Gradient Methods，NeurIPS 1999](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
- [Counterfactual Multi-Agent Policy Gradients，AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11794)

写作采用 Supervisor Skills 的 `tech-paper-template`、`intro-drafter` 与 `paper-writer`：先确定贡献与证据，再安排段落，最后撰写摘要。正式段落以任务和机制组织论证；本文件保存作者查看用的定位依据。
