# 读者问题的处理结果

在当前稿件与 [FlashTrace 论文](https://arxiv.org/html/2602.01914v4) 的基础上，按本轮授权，仅查阅以下三项对应的测试代码。未运行测试或实验，未查阅实验日志。

## EOS 的直觉

EOS 是本文在文本替换操作中约定的参考零点。将源文本统一替换为 EOS 后，得到作为归因起点的参考文本；原文本的贡献相对于该参考衡量。零点指这个比较基准，不要求参考响应分数或 EOS 嵌入本身等于零。方法段先解释文本层面的零点，再引出嵌入差分。

## 三项定义已明确

1. **删除 baseline。** 测试使用 EOS 替换被删除的 eligible tokens，全部替换后的 prompt 与 DT 归因参考一致。正文将 baseline token 明确为 EOS，附录用一句话说明删除终点与参考的关系。
2. **NIAH recovery 的目标。** DT 使用完整固定 response 加 EOS。正文与附录均补充这一点，限定为 DT，不将 FT 的最终输出目标套用到 DT。
3. **MAS。** 实际比较的是归一化响应分数与剩余正贡献比例。后者是未删除 token 的正贡献之和除以 eligible tokens 的初始正贡献总和。将两者的绝对差加到响应曲线上，裁剪到 [0,1]，按曲线最小值与最大值重新归一化，再计算梯形面积。正文更正“已删除贡献比例”，附录补充简短计算说明。

## 核对依据

以下文件均只读自提交 `93665b6`，仅使用对应的输入构造、归因目标和指标计算片段。

- `experiments/official/evaluate.py`，第 248–255、278、303–327、344–388 行。完整响应加 EOS 的构造，eligible 位置及 EOS 参考，指标调用。
- `research/temporary/all_baselines_20260910/source_snapshot/exp/exp2/run_exp.py` 的 `load_model` 与删除指标函数。padding token 设置为 EOS，MAS 使用剩余 attribution density。
- `research/temporary/all_baselines_20260910/source_snapshot/ft_ifr_improve.py` 的 `faithfulness_test_skip_tokens`。MAS 的差值校正、裁剪、最小最大归一化及面积。
- `research/temporary/qwen35_niah_causal_20260913/raw/validation_v1/run_validation.py`，第 74–85、116–119、159–193 行。完整 response 的目标位置、NIAH recovery 使用的向量、EOS 删除及 MAS 计算。
- `research/third_party/flashtrace_qwen35_e81b3be/flashtrace/improved.py`，第 116–129、199–225 行。删除 token、固定目标及 MAS 计算。

这些定义已落实到稿件，无需新增实验或扩写其他章节。
