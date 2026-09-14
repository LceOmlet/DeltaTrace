# Qwen 平均对称实现全量复测数据

当前入口为 `latest_results.json`，最新 CSV 均位于 `hotpot_complete_support_30_v1/`。下方原始运行记录及 `final_results/`、`hotpot_positive_sum_v1/` 是历史版本，不作为当前 HotpotQA 表。

| 文件 | 内容 |
|---|---|
| `hotpot_complete_support_30_v1/task_metrics.csv` | Qwen3-8B 全任务及 Qwen3.5-9B VT、HotpotQA，200 行汇总 |
| `hotpot_complete_support_30_v1/per_case.csv` | 对应 18,217 行逐例指标 |
| `hotpot_complete_support_30_v1/hotpotqa_all_methods.csv` | 两模型目前测试的全部方法，14 行，每行 48 例 |
| `hotpot_complete_support_30_v1/hotpotqa_per_case.csv` | 全方法 HotpotQA，672 行逐例结果 |
| `hotpot_complete_support_30_v1/selections.json` | 全方法选句、token、官方支持事实及完整支持判定 |
| `hotpot_complete_support_30_v1/protocol.json` | 最新指标定义、预算、数据来源与哈希 |
| `final_results/actual_run_costs.csv` | 本次实际运行成本，包含编译，不是重复热运行基准 |

Qwen3-8B 使用 `pv-layer-symmetric-v1`，包含 NIAH 六组（600 例）、VT 四组（400 例）、HotpotQA（48 例）、MATH（100 例）、MoreHopQA（95 例）。Qwen3.5-9B 使用 `gdn-symmetric-v1`，包含 VT 四组（400 例）和 HotpotQA（48 例）。两模型另有各 448 例的 VT/HotpotQA 固定目标恢复评测；它们不是新增独立样本。MATH、MoreHopQA 没有相应 gold，不生成 Recall。

HotpotQA 当前表名为 **Full-support Recall@30%**，机器字段仍为 `complete_support`：全部官方支持事实均被选中记 1，否则记 0，再对 48 例求均值。各方法均先将 token 贡献截为正值，再按原生句子求和；预算为完整正文 token 的 30%，按排序前缀选择，遇到首个放不下的句子即停止。CSV 比例数值为 0–1。普通 supporting-fact Recall 与该指标不同。该指标和预算是在查看已有曲线后选定，不能当作新留出集或预先注册的优势证据。

Qwen3 的 `DT_original` 与平均对称 `DT`、Qwen3.5 的 `DT_legacy` 与本次 `DT` 分行保留。旧版结果和低于对照的结果均保留；FT 使用本次对应协议结果。同一算法的历史归一化诊断不重复计为新方法。

Git 保存最新和历史 CSV、逐例选句证据、生成脚本、评分源文件及交付/备份哈希回执；`git_data_manifest.json` 列出本次数据文件的准确字节哈希。大型原始归因向量、运行缓存和机器访问辅助脚本不纳入 Git。完整原始归档 `qwen_symmetric_full_complete.zip`（185,730,198 字节，SHA-256 `6acd36198314b9cd50580b8cc51fa4759e12d413e6dc0e26de1b51a1ec73af60`）已在本机 `cross_machine_backup/mirror/` 与远端 `/tmp/codex_source_v2_gpu_20260910_v1/qwen_symmetric_full_20260913/` 保存。`final_results/delivery_receipt.json` 记录完成交付；`cross_machine_backup/latest_verified.json` 记录后续完整备份；最新 HotpotQA 修订的异机回执为 `hotpot_complete_support_30_v1_labels.backup.json`。完整重算需这些原始归档及既有基线向量，单独读取本次 CSV 无此依赖。

给 main 智能体：在 main 上 fetch `codex/needle-gap-audit`，cherry-pick 本次数据提交即可。当前任务仅同步数据及评分来源；不要重跑模型、修改指标、扩写论文或执行额外检查。若按用户此前授权同步论文，只能使用已批准的 Qwen3.5 表数据及由表格直接得到的分析；实现仅保留在仓库。不要把本分支此前其他提交一并整支 merge 到 main。

## 原始运行记录（历史口径）

**2026-09-14 用户修正：HotpotQA Recall 只使用正贡献。** 两模型各 48 例已使用保存的向量重算，公式为 `sentence_score = sum(max(token_attribution, 0))`，DT 与 FT K1/K3 相同；预算、目标和选句规则不变。最新完整表为 `hotpot_positive_sum_v1/RESULTS_zh.md`，机器可读入口为 `latest_results.json`；旧 signed-sum 结果保留供追溯，不能继续作为最新 HotpotQA Recall。下方原始协议描述是历史记录。本次未重跑模型，也未修改论文。

本轮仅产生实现核验和评测数据，不修改论文。正式任务范围由 `protocol.json` 固定，结果未完整时不得标为全量结论。

| 模型 | 被测实现 | 全量任务 | 主评测样本 | 论文 Recall 另测 |
|---|---|---|---:|---:|
| Qwen3-8B | 每层平均对称 PV | NIAH 六组、VT 四组、HotpotQA、MATH、MoreHopQA | 1,243 | 448 |
| Qwen3.5-9B | gdn-symmetric-v1 | VT 四组、HotpotQA | 448 | 448 |

Qwen3 的 PV 分配为 `Delta(PV)=Pbar*DeltaV+DeltaP*Vbar`，其中两个平均值都取实际端点的算术平均。每层对同一个上游系数调用两次现有有限注意力算子，平均 `dq`、`dk`、`dv`，然后继续向前一层传播。交换端点后的系数不另取负号；这不是两个完整归因向量的事后平均。全模型入口断言每例 36 层、72 次有限算子调用。仓库正式默认配置不在本轮改动范围内。

实现检查已完成：

- CPU 100 组直接 PV 恒等式及系数收缩检查，最大误差约 `5.33e-15`。
- GPU 执行现有 `symmetric_secant.py` 中实际平均分支，使用真实 FP16 有限算子和原生 FA LSE，与直接平均 P/V 的 FP64 参考比较。
- 5 组 GPU 情形覆盖长度 17、65、127 以及相同 Q/K、相同 V；最大相对 L2 误差 `0.000314184`（约 0.0314%）。交换端点后的系数逐位相同。
- 数值证据见 `pv_cpu_verification.json`、`pv_gpu_verification.json`。这是实现一致性证据，任务效果由本次全量结果确定。

完整响应的忠实度沿用 released-v1 的固定输入和 20 步删除：signed RISE、positive MAS、positive RISE+AP，使用本次重新运行的 FT K1。NIAH 为原始 Recall@10%，对照 FT K3。没有 gold 的 MATH 和 MoreHopQA 不虚构 Recall。

VT 与 HotpotQA Recall 沿用 DT 论文的单独协议：VT 的固定最终答案目标和正文 token 排序，H2/H4/H6/H10 主预算分别 10%/10%/20%/30%；HotpotQA 完整响应目标，原生句子 signed sum 排序，10% 完整正文词元预算，报告 supporting-fact Recall。双方目标排除相同停止词元与 EOS。另存 5%、10%、20%、30%、40%、50% 预算结果。旧全响应 VT/HotpotQA token Recall 不是论文主 Recall。

`runtime_snapshot/` 保存此前已执行的动态运行时；`runtime/` 是本次隔离入口。`prepare_runtime.py` 记录 Qwen3 PV 和 Qwen3.5 GDN 的明确接线，`prepare_recovery_driver.py` 仅泛化原论文输入准备的模型选择。主入口、恢复入口与原始输入均保留哈希。

远端目录：`/tmp/codex_source_v2_gpu_20260910_v1/qwen_symmetric_full_20260913`。`run_suite.py` 顺序执行两模型 Recall 入口检查、两模型完整主评测、两模型完整 Recall。已完成 Recall 个例在恢复执行时按来源哈希保留；失败记录不得丢弃。

## 2026-09-13 保存瓶颈修复

原主评测每例两次重写全部历史 JSON，378 例时文件约 469 MB，一次离线序列化约 15.8 秒，而最近 20 例归因和指标阶段平均合计约 10.0 秒。诊断原始数字与测量限制见 `performance_diagnosis.json`。

用户要求修好续跑后，旧队列在第 **400** 例 `niah_mv_v2_99` 完整落盘后停止；旧 `qwen3_main_v1` 的结果和向量保留。新 `run_suite_incremental.py` 将两个主评测切换到 `evaluate_incremental.py` 和 `qwen3_main_v2`/`qwen35_main_v2`，逐例原子保存全部结果、成本、来源和向量，只在最后统一生成兼容的完整 `results.json`/`vectors.npz`。恢复严格校验来源并跳过完整案例；归因、输入、指标、精度和两个 Recall driver 不变。

`incremental_driver_receipt.json` 保存源码哈希与精确替换清单，并验证归因和指标 helper AST 不变。`test_incremental_store.py` 验证旧完整案例迁移、案例提交后中断恢复、来源与损坏文件拒绝，以及最终结果和向量无损合并。监控、离线汇总和收尾入口分别为 `status_incremental.py`、`analyze_incremental.py`、`finalize_incremental.py`。具体状态和恢复说明见 `MONITOR_HANDOFF.md` 顶部升级章节。
