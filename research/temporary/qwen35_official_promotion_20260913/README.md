# Qwen3.5 官方实现切换

2026-09-13，按用户要求将 `gdn-symmetric-v1` 设为官方默认，并采用对应的 8 任务、72 例数据。当前入口、配置和清单在 `deltatrace/profiles/` 与 `configs/official_dt_profiles.json`。旧 clean-v1 的代码、依赖清单、13 任务数据和示例图源均保留。

正文数据来自已冻结的六个 NIAH 任务各 10 例新键值模板，以及 MATH/MoreHopQA 各 6 例固定原始样本。FT K1 用于 RISE/MAS，K3 用于 NIAH Recall；DT 使用 signed RISE、positive MAS/Recall。所有 44 个显示值已核对到原始逐例记录，并从 PDF 再次提取核验。Qwen3 主数据未改动。

另外重算了论文原有的三个 Qwen3.5 示例：MQ-Q2 第 0 例、MoreHopQA 第 0/1 例；它们不计入 72 例正文均值。运行实际调用官方默认 factory，首例与冻结原型逐位一致。独立 CPU 审计验证了 204 次原生评分输入和 15 条曲线。代码与评分适配器的 32 项测试通过。

`raw/figures_v1/` 保存原生输出、源码、协议和验证；`figure_import_verification.json` 保存图源导入记录；`promotion_receipt.json` 保存完成状态与哈希。论文方法、双端点记忆公式、示例图和正文表已同步。最终 PDF 共 17 页，新表位于第 7 页，结论结束于第 8 页；没有 Qwen3.5 效率汇报。
