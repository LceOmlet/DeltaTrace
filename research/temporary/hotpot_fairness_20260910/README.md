# HotpotQA evidence v2

**后续 [v3 48 例结果](../hotpot_context_v3_20260910/RESULTS.md) 已完成。** 新实验保留原推理和完整输入、计入全部正文词元成本、按固定排序前缀选句，同时报告有符号总和与旧正值均值。两种答案条件视图均为 DT 低于 FT K3；完整回答两项差值区间跨零。本目录的 v2 原数值、向量和复审记录继续保留。

**请先读 [方法学复审](REVIEW.md)。** 两个定位修正确实成立；答案重建改变了
条件前缀，整句预算只计算过滤后的有效词元，聚合也未证明中立。撤回将整套
v2 称作“公平性已经修好”的结论，数值保留为探索性对照。

完整结果、明确错误、全部正负对照和复现命令见 [RESULTS.md](RESULTS.md)。

`prepare_inputs.py` 先构建不依赖 gold 或归因值的原生候选句，再写独立标签。
`analyze_fairness.py` 核验归档和 48 例重跑，从原始向量计算两个目标、两种标签、
三个方法和六种预算。`build_report.py` 生成汇总表；`verify_outputs.py`
从实际检索集合独立核算所有支持事实指标与成本。

共享评分模块位于 [`experiments/official/hotpot_evidence.py`](../../../experiments/official/hotpot_evidence.py)。
运行前协议和 GPU 驱动固定在提交 `9bbce4f`；实现中的三个不可分词元跨句问题
在评分前按照既定首内容字符规则处理，见 [TOKEN_BOUNDARY_NOTE.md](TOKEN_BOUNDARY_NOTE.md)。
旧 448 例结果、原模型方法和历史评分入口保留，新版必须显式选择。
