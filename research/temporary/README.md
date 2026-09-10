# 临时研究与诊断

最新：[Qwen3完整热调用与固定几何显存双验收、更新rollout曲线](qwen3_stream_memory_20260910/README.md)。

这里保存独立效率研究、诊断及未完成或被撤回的草稿，研究文件本身不是干净方法入口。

[Qwen3短输入B1效率研究](qwen3_short_efficiency_20260910/README.md)已完成四档、两版固定FT的完整热调用验收。通过核验的显式实现位于[deltatrace/accelerated](../../deltatrace/accelerated/GRAPHED_QWEN3.md)；全部失败候选、重复计时、原始向量和复核脚本在研究目录保留。

`unaligned_qwen35_raw_prompt_20260909.py`是未执行、已撤回的raw-prompt草稿，不再推进。正式入口为`experiments/official/evaluate.py`，其输入、评分与FT跳数按作者原实验设置。

旧`research/runtime`、`research/prototypes`、`research/reproduction_templates`与`evidence`保留为历史材料；当前干净方法仅从`deltatrace/clean`导入。进入正式版本前须核验来源与实际调用，不能凭文件名或局部测试升级。
