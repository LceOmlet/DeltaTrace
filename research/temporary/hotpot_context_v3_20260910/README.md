# HotpotQA context and cost v3

本目录完成了 48 例保留原推理的答案归因重跑，并将整句检索改为全部正文词元计费、固定排序前缀选择。
两种目标与两种池化都已固定并完整报告。它修正具体的上下文/成本问题，不声称已证明测量中立。

请读 [结果与复现命令](RESULTS.md)、[运行前协议](PROTOCOL.md) 和 [独立核验回执](verification.json)。
10% 全部正文词元预算下，保留原推理的答案条件中，DT/FT K3 有符号总和 Recall 为
35.59%/51.04%，正值有效词元均值为 48.96%/64.76%；完整回答两项均值略高但调整区间均跨零。

新 GPU 归档见 [清单](raw/conditioned_v3/manifest.json)，旧完整回答由
[原完整基准](../source_v2_gpu_20260910/full_recall/analysis.json) 的逐例已验证来源读取。
原方法与先前 v2 数值不改动；[v2 复审](../hotpot_fairness_20260910/REVIEW.md) 保留历史问题与最小定位修正依据。
