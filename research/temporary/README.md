# 临时研究与诊断

这里保存未完成、被撤回或尚未对齐正式协议的草稿。它们不是干净方法入口，也不是正式评测。

`unaligned_qwen35_raw_prompt_20260909.py`是未执行、已撤回的raw-prompt草稿，不再推进。正式入口为`experiments/official/evaluate.py`，其输入、评分与FT跳数按作者原实验设置。

旧`research/runtime`、`research/prototypes`、`research/reproduction_templates`与`evidence`保留为历史材料；当前干净方法仅从`deltatrace/clean`导入。进入正式版本前须核验来源与实际调用，不能凭文件名或局部测试升级。
