# VT 与 HotpotQA 的 Recall 修复验证

**最新全量 448/448 例已完成并填表。** 四组 VT 各 100 例，HotpotQA 48 例，采用同一冻结目标策略。VT 原始 token Recall 宏平均为 DT **60.20%**、FT K3 **56.14%**，差值 **+4.05 点**，描述性调整区间 **[+3.59, +4.51]**；双方句聚合均为 **63.80%**，每例达到预算上限。HotpotQA 原始排序为 **40.63% 对 45.35%**，DT 落后 **4.71 点**，调整区间 **[−8.60, −1.07]**；句聚合为 **73.81% 对 72.09%**，差值 **+1.73 点**，调整区间 **[−6.54, +9.48]**，仍不确定。见 [全量表与图](full_recall/RESULTS.md)、[Recall@10% CSV](full_recall/table_recall10.csv) 和 [多预算表](full_recall/table_budgets.csv)。全量数据包含开发及验证样本，区间用于描述，不作为新的独立留出证据。

此前 **80 例内部留出验证**单独保留：VT 四任务等权原始 Recall 为 DT 59.79%、FT K3 55.70%，差值 +4.09 点，四项比较校正后区间 [+2.78, +5.56]。双方句聚合后均为 63.58%。HotpotQA 句聚合为 75.94% 对 69.22%，差值 +6.72 点，校正区间 [−4.55, +17.92]，尚不能确认稳定优势。见 [保留验证结果](target_scope/RESULTS.md) 与 [目标构造修复说明及运行入口](TARGET_CONSTRUCTION_FIX.md)。

本轮在运行前固定 VT 只解释生成的最终答案、HotpotQA 解释完整回答，双方在各任务内使用相同输入和目标；原始排序与共同最佳句均值排序都报告。它给出的是任务与排序视图限定的结论，并未通过或替代此前失败的统一目标门槛。

当前聚焦 Recall。首轮 **16 例开发 + 80 例留出验证**已完成，**没有通过共同优势判据**：句聚合后的 VT H2-C3 为 DT 48.11%、FT 74.67%；三个单链 VT 任务打平；HotpotQA 为 72.63% 对 70.53%，差值区间 [−11.38, +14.67] 点，仍不确定。见 [完整的 80 例验证](recall_validation/RESULTS.md)。全部负向结果保留，没有在验证集上另选方案。

解释目标对照已定位多链 VT 的主要失败：完整响应讨论无关链，而 gold 仅覆盖被问到的链。只保留生成的最终答案后，开发集 VT H2-C3 的 DT/FT 都达到 100%；HotpotQA 为 71.85% 对 73.35%，尚未胜出。更改参照、句子聚合与逐层对称分配都未通过共同开发门槛。见 [开发对照报告](answer_development/RESULTS.md)、[目标对照计划](ANSWER_PILOT.md)、[FT 回溯范围校验](ANSWER_HOPS_ADDENDUM.md)。这些失败方案未进入第二组验证；最新验证只采用预先固定的任务目标，并明确区分原始排序与共同句聚合。

**新 reference 在 VT H4-C1 全部 100 例上使 needle 变差。** 共同正文范围、10% 预算下，旧 reference DT 为 71.24%，新 reference DT 为 66.69%，live FT K3 为 71.33%；新−旧为 −4.55 个百分点，95% 配对区间 [−5.20, −3.93]。100 例中 84 例下降、1 例提高、15 例持平。RISE/MAS 虽然改善，不能代替 needle 改善证据。见 [第一个完整任务的中途报告](interim_1/RESULTS.md)。

HotpotQA 全部 48 例也已完成：共同正文范围与 10% 预算下，原 reference DT 为 40.56%，新 reference 为 31.87%，live FT K3 为 45.35%。更改 reference 使 Recall 下降 8.69 点。见 [两个完整任务的配对报告](interim_focus_2/RESULTS.md)。

固定设计见 [PROTOCOL.md](PROTOCOL.md)：每例重新执行新、旧 reference 的 DT，并在相同正文候选集、相同预算和相同删除规则下评分；FT K1/K3 同例 live 运行。主比较是 Recall@10% 配对差，次要指标包括多预算 recovery、precision、RISE、MAS。全部任务和失败结果都保留。

## 当前执行状态

- 固定目标的全量运行已完成全部 448 例：复用已验证的 80 例，新增 368 例，并额外重跑 5 个衔接控制。所有 26 批结果、全部索引覆盖、输入/目标/预算与 15 份控制向量逐位一致检查通过；原始归档和复现入口见 [全量报告](full_recall/RESULTS.md)。
- 三例 GPU 执行检查通过；本地从向量重算 recovery，并重建两个 reference 及所有实际删除输入哈希。模型操作耗时合计 141.58 秒，包含三次模型加载。
- 原 `full_v1` 队列完成 40 个 NI MQ-Q2 样本后，按用户要求调整优先级。已完成和未完成数据均保留，该前缀不标作完整任务结果。
- 完成 VT H4-C1 全部 100 例、HotpotQA 全部 48 例并通过独立复核。后续 VT H2-C3 前缀在用户要求改为 Recall 小样本研究时停止；已有完整任务与前缀均保留。
- `recall_dev_v1` 的 16 例、64 次 DT 和 32 次 live FT 归因已完成。原方向对照与此前运行逐位一致；全部正负候选都保留。固定方案的 `recall_validation_v1` 已完成 80 例、通过独立向量/输入/预算复核，质量判据失败。
- 新增执行开关和汇总校验通过 28 项测试；13 个旧任务、1,243 例的汇总保持兼容，27 项冻结方法依赖哈希一致，见 [实现回执](implementation_validation.json)。

## 三例执行检查

下表是相同正文范围与 10% 预算的 token recall。每个任务只有一例，不能据此声称质量提升。全 prompt reference 对照也是本次重新归因。

| 样本 | 旧 reference DT | 新 reference DT | live FT K3 |
| --- | ---: | ---: | ---: |
| NI MQ-Q2，第 0 例 | 60.00% | 55.00% | 55.00% |
| VT H4-C1，第 0 例 | 66.67% | 64.71% | 66.67% |
| HotpotQA，第 0 例 | 72.73% | 75.76% | 90.91% |

VT 这例的 RISE 从 0.07941 降至 0.05729、MAS 从 0.09206 降至 0.07646；HotpotQA 的 RISE 均为 0.025、MAS 从 0.07046 降至 0.05507。删除指标与 recovery 的方向并不一致，完整任务的配对验证用于判断覆盖范围与不确定性。

本地验证结果见 [执行检查分析](smoke_analysis/analysis.json) 和 [逐例表](smoke_analysis/paired_cases.csv)。旧冻结运行向量与本次旧 reference 重算存在小幅数值差异；主比较使用本次新旧两份向量，不用旧向量替代配对对照。

## 运行与复核

`run_suite.py` 逐任务执行完整缓存，记录代码、协议、环境和向量哈希。`--task-order` 只接受预定 11 个任务的完整排列，改变顺序不改变样本集合。每个完成的任务独立保存结果，未完成任务不会进入完整任务汇总。

`analyze.py` 独立复核输入、gold、向量、全部预算点和实际删除输入，计算每任务及固定任务等权的 95% 配对 bootstrap 区间（10,000 次，seed=73）。默认要求全量完成；`--completed-tasks-only` 仅允许中途汇总已经完成的整个任务，并明确标记剩余任务。

各任务按固定任务索引取得独立的 seed=73 随机流，因此同一任务在中途和最终报告中使用相同的重采样。截止处并列导致的 recall 上下界单列为诊断，不改变原定位置升序规则。

```bash
python research/temporary/source_v2_gpu_20260910/analyze.py \
  --run /path/full_v2 \
  --publication ../DeltaTrace-paper-qwen3/experiments/official/results/qwen3_8b_table1_20260909 \
  --data ../audit/published_flashtrace/table1-data-v1/extracted/data \
  --tokenizer ../audit/deltatrace-figures-20260909/qwen3-tokenizer.json \
  --output research/temporary/source_v2_gpu_20260910/analysis
```

最初的 1,048 例队列按用户收窄任务的要求停止，不再把它列为待完成目标。研究基于已经定位的多链失败，已完成开发对照与未用于首轮验证的第二组保留样本；所有 GPU 运行均已结束。

原始文件以确定性 gzip 保存在 `raw/`，清单同时记录压缩前后的 SHA-256。恢复到一个新目录后即可用上面的分析入口复核，例如：

```bash
python research/temporary/source_v2_gpu_20260910/archive_results.py --restore \
  --source research/temporary/source_v2_gpu_20260910/raw/smoke_v1 \
  --output /path/restored-smoke
```
