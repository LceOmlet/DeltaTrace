# Qwen3-8B：FT 原表完整 DT 实验

**已完成作者Table 1全部13个任务、1,243条样本。** 全部原发布样本保留，无跳过、重新采样或重新生成；干净方法及评测源码在运行期间保持冻结。

制表直接使用[完整CSV](table.csv)或[LaTeX表格](table.tex)。[逐例指标](per_case.csv)、[逐例DT/FT配对](paired/paired_cases.csv)、[差值分布](paired/paired_summary.csv)及[来源核验](paired/paired_audit.json)同时保存。原始有符号向量、正值视图、删除曲线及输入映射位于`raw/`；[summary.json](summary.json)和[run.json](run.json)保存版本、哈希及全部完成记录。此前不完整表格可从Git历史恢复，当前目录仅保留完整表格入口。

| 指标 | DT均值较优的任务数 | DT均值较差的任务 |
|---|---:|---|
| RISE↓ | 10/13 | MQ-Q4、MQ-Q8、VT-H10-C1 |
| MAS↓ | 12/13 | MQ-Q8 |
| Recall@10%↑ | 6/11 | 全部4项VT及HotpotQA |

这些是各任务完整样本的均值方向，不是显著性声明。6项NIAH的needle均领先，而4项VT及HotpotQA均落后；不概括为全面领先。MATH、MoreHopQA原表没有needle，保持空值。所有逐例退步均保留，没有根据本轮成绩调整方法。

- 方法：冻结的 `clean-v1-20260909` Qwen3 DT，`content_P1`，不采用后续逐层修补。
- 评测：`clean-v1-eval3-native-batch-20260909` 的正式入口；Qwen3 使用默认 clean 后端，真实样本 batch=1，内部两个端点共用原生 FA batch=2。Qwen3.5 的加速后端没有混入本次实验。
- 输入：作者 `table1-data-v1` 的完整发布缓存；使用原输入包装、完整 target + EOS、原 keep/gold 映射，不重新生成、采样或过滤。
- 模型与评分：原 Qwen3-8B 权重逐文件校验，FP16；原作者 eager 评分，20 次删除、21 个响应点。
- 指标：RISE 使用有符号排序；MAS 和 needle 使用正值部分。原始有符号向量、正值向量、实际删除输入哈希和原响应曲线均保存。
- 对照：FT 直接引用作者发布的完整任务 CSV，needle 为 Recall@10%；不重跑或修改 FT。配套逐例轨迹记录为`n1`，且汇总值与相应CSV一致，见下方来源校正。
- 范围：10 个 RULER 任务各 100 条、HotpotQA 48 条、MATH 100 条、MoreHopQA 95 条。MATH/MoreHopQA 原表没有 needle，对应单元格为空。

运行与恢复：

```bash
MACA_PATH=/opt/maca TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS=0 \
python experiments/official/run_qwen3_paper.py \
  --environment /path/environment.json --output /path/qwen3-paper-run
```

控制器逐任务调用现有 `evaluate.py --selection paper --ft published`。任务输出独立，完成任务经过现有 `summarize.py` 校验后可复用；失败尝试保留，不覆盖或按成绩筛选。原作者源码、FA/模型实现及冻结 DT 传播没有修改。计时包含冷调用，不能当作预热后纯归因效率表。

完整表格导出：

```bash
python experiments/official/export_qwen3_paper.py \
  --run /path/qwen3-paper-run --output /path/publication
```

计算期间加 `--allow-partial` 只导出已经完整完成的任务，输出显式命名为 `table_progress`。CSV 保留完整浮点精度，LaTeX 显示四位小数；RISE/MAS 越低越好，Recall@10% 越高越好。Recall 保存为 [0,1] 比例，百分数展示时乘 100。粗体只表示观测均值较优，不表示显著性。

需要计算期间自动更新导出时，在独立 CPU 进程运行：

```bash
python experiments/official/watch_qwen3_paper_exports.py \
  --run /path/qwen3-paper-run --output /path/publication
```

该进程每 30 秒检查任务状态，只在新任务完成时调用同一个导出器；全表完成后导出正式文件并退出。它不执行模型或指标评分。实际 [依赖与权重记录](environment_receipt.json) 随结果保存。

论文接入时，以这里的固定指标口径为准：**RISE 使用有符号排序，只有 MAS 和 needle 使用正值部分**。截至启动本次实验时，论文草稿 `paper/iclr2027/sections/evaluation.tex` 中“Both metrics use the positive part”仍是旧口径，不能作为本次实验设置。

原始结果中的 `metrics.DT` 是最终入表标量。绘制 RISE 曲线时，有 `metrics.DT_signed` 则使用其曲线；没有时，`signed_RISE_reuse_proof` 证明正值曲线产生相同 RISE。MAS 使用 `metrics.DT_positive`。`scores` 是作者函数返回时保存的真实 21 点响应；有符号视图下附带的 MAS 已明确标为无效，不能填表。

协议与原表来源见 [protocol.json](../../protocol.json) 和作者 [REPRODUCTION.md](../../reference/REPRODUCTION.md)。

### 原始逐例配对及来源校正

[配对结果](paired/paired_cases.csv)、[差值分布](paired/paired_summary.csv)及[来源核验](paired/paired_audit.json)通过作者发布的`exp2_table1_traces.tar.gz`补充，不重跑FT。脚本验证两个原档案SHA256、所有原缓存文本、target、用户token/keep/gold映射，以及每项轨迹汇总和实际FT CSV完全对应。作者轨迹的`prompt_len`是归因矩阵中的用户token数；DT报告的`prompt_length`含聊天模板，两字段不能直接相等比较，实际token位置映射已逐例核验。

**更正此前的“FT needle为K=3”说明：**该说法来自作者发布README；本次实际引用的recovery CSV与其`ifr_multi_hop_both_n1_*`逐例轨迹吻合。原脚本`_trace_run_tag`直接由`n_hops`产生`n1`目录名。因此，这里按已发布CSV及其实际轨迹标识来源，不把这些复用数值称为已验证的K=3执行。现有表格一直使用同一批作者CSV，分数没有因这项说明校正而改变；作者文件和正在运行的冻结评测源码保持原样。

逐例配对是描述性分析，严格保留全部案例，不构成新的测试集或方法选择过程。以已完成的MQ-Q8为例，DT的RISE有78条落后、MAS有95条落后，但needle全部100条领先；VT-H2-C3的RISE/MAS各99条领先、needle96条落后。这些任务差异并非仅由个别极端案例导致，不据此调整正在运行的冻结方法。

复现配对分析：

```bash
python experiments/official/analyze_qwen3_paper_pairs.py \
  --publication experiments/official/results/qwen3_8b_table1_20260909 \
  --traces /path/exp2_table1_traces.tar.gz \
  --caches /path/exp2_table1_caches.tar.gz \
  --output experiments/official/results/qwen3_8b_table1_20260909/paired
```

两个档案均来自作者固定的`table1-data-v1`发布。配对分析覆盖范围以`paired_audit.json`中的任务数和publication摘要哈希为准。
