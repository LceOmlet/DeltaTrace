# HotpotQA evidence v2

完整结果、明确错误、全部正负对照和复现命令见 [RESULTS.md](RESULTS.md)。

`prepare_inputs.py` 先构建不读取 gold 或归因的原生候选句，再写独立标签。
`analyze_fairness.py` 核验归档和 48 例重跑，从原始向量计算两个目标、两种标签、
三个方法和六种预算。`build_report.py` 生成汇总表；`verify_outputs.py`
从实际检索集合独立核算所有支持事实指标与成本。

共享评分模块位于 [`experiments/official/hotpot_evidence.py`](../../../experiments/official/hotpot_evidence.py)。
运行前协议和 GPU 驱动固定在提交 `9bbce4f`；实现中的三个不可分词元跨句问题
在评分前按照既定首内容字符规则处理，见 [TOKEN_BOUNDARY_NOTE.md](TOKEN_BOUNDARY_NOTE.md)。
旧 448 例结果、原模型方法和历史评分入口保留，新版必须显式选择。
