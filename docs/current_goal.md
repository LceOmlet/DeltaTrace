# 当前官方实现与结果

2026-09-13，按用户要求将 GDN 对称版设为 Qwen3.5 的官方实现，并采用对应的已确认数据。

- 当前配置：[official_dt_profiles.json](../configs/official_dt_profiles.json)；默认入口：[make_qwen35_runner](../deltatrace/profiles/official.py)。
- Qwen3.5 的全部 24 个 GDN 层统一使用对称输出门和两个端点顺序的完整记忆系数平均。注意力与 MLP 规则沿用原实现。
- 当前 Qwen3.5 正文比较覆盖 8 个任务、72 例：六个 NIAH 任务各 10 例新键值模板，MATH/MoreHopQA 各 6 例固定原始样本。signed RISE、positive MAS/Recall；FT K1 忠实度、K3 Recall。
- 新版结果：[数据来源](../paper/iclr2027/results/data/qwen35_official_quality_sources.json)；[正文表](../paper/iclr2027/results/qwen35_quality_table.tex)。该数据不代表旧版 13 任务全量实验。
- Qwen3-8B 的方法及完整 1,243 例比较不变。旧 Qwen3.5 clean-v1 代码、清单和结果保留作历史复现；新运行需显式指定 `--qwen35-profile clean-v1` 才使用旧方法。
- 论文同步更新公式、示例图和质量表；Qwen3.5 正文不报告效率结果。

此前的阶段说明保存在 [2026-09-09 历史目标](goal_clean_baselines_20260909.md)，不再控制当前默认实现。
