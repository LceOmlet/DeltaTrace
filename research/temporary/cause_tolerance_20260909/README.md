# 两模型差距及容差加速：新一轮研究

状态：初始核对，尚无新修复或加速收益。Goal见[当前目标](../../../docs/current_goal.md)。本目录是临时机制及性能研究，不能替代冻结的Qwen3论文表格。

相同NI8/MH8的干净有符号RISE显示：最明确的待解释问题是NI的DT相对FT优势反转。两模型MAS都优于各自FT，MH的RISE也都优于FT；不能先验归纳为Qwen3.5所有方面失败。

[初始核验](initial_evidence.json)重新核对3份原报告、向量文件、全部32条输入及向量身份，未运行新模型或指标。[逐例数据](baseline_cases.csv)和[均值](baseline_means.csv)显示：NI的DT−FT RISE由Qwen3的−0.011089变为Qwen3.5的+0.017283；逐例领先数量从7/8变为1/8。这定位了待解释现象，尚不能把架构差异与实现、精度和模型行为分离。

历史对照中的FT needle使用K=3；原论文CSV的配套轨迹为n1。这里保持原开发对照身份，不替换或混合这两组数值。32例有符号视图由已验证原曲线及有证明的复用组装，不能宣称重新运行32例。

本轮第一项算子核对直接调用当前原生FLA前向和冻结`finite_fla_pullback`，检查纯保留路径中的连续门顺序分配，并覆盖跨64-token分块边界。四种固定算子情形只用于核对数学机制，不是新质量基准，不测RISE/MAS，不改任何框架。对称门的真实删除作用、有限分配及舍入余项分别记录；即使反例成立，也不能据此认定它解释了真实NI差距。

计算预算：长度4和67，各检查0.1→0.9及反向两种门变化；每种情形一次原生双端点前向、一次原生删除组合前向及一次冻结有限FLA（内含两项原生伴随）。共8次局部前向、4次有限调用，无模型加载、无答案生成或指标评分。

另一工作树的[连续门/复合运算审查](https://github.com/LceOmlet/DeltaTrace/blob/920c5d2/research/temporary/method_invariance_audit_20260909/RESULTS.md)与[已停止的统一归一化修法](https://github.com/LceOmlet/DeltaTrace/blob/a5ca921/research/temporary/deletion_audit_20260909/README.md)作为历史证据引用；其未提交实验不在本分支改动或接管。
