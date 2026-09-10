# 新 reference 候选的 GPU 配对验证

**新 reference 在 VT H4-C1 全部 100 例上使 needle 变差。** 共同正文范围、10% 预算下，旧 reference DT 为 71.24%，新 reference DT 为 66.69%，live FT K3 为 71.33%；新−旧为 −4.55 个百分点，95% 配对区间 [−5.20, −3.93]。100 例中 84 例下降、1 例提高、15 例持平。RISE/MAS 虽然改善，不能代替 needle 改善证据。见 [第一个完整任务的中途报告](interim_1/RESULTS.md)。

预定的 1,048 例对照仍在执行，HotpotQA 已接续启动。结论覆盖范围目前限于 VT H4-C1，不把这一任务外推为全部任务结果。

固定设计见 [PROTOCOL.md](PROTOCOL.md)：每例重新执行新、旧 reference 的 DT，并在相同正文候选集、相同预算和相同删除规则下评分；FT K1/K3 同例 live 运行。主比较是 Recall@10% 配对差，次要指标包括多预算 recovery、precision、RISE、MAS。全部任务和失败结果都保留。

## 当前执行状态

- 三例 GPU 执行检查通过；本地从向量重算 recovery，并重建两个 reference 及所有实际删除输入哈希。模型操作耗时合计 141.58 秒，包含三次模型加载。
- 原 `full_v1` 队列完成 40 个 NI MQ-Q2 样本后，按用户要求调整优先级。已完成和未完成数据均保留，该前缀不标作完整任务结果。
- `full_v2` 已完成 VT H4-C1 全部 100 例并通过独立复核，正在运行 HotpotQA，之后运行其余 VT 与全部 NI。顺序变更见 [PRIORITY.md](PRIORITY.md)，正式对照参数未变。
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

完整结果尚未产生。任何“修复有效”的结论必须由共同评测规则下的配对收益支持。

原始文件以确定性 gzip 保存在 `raw/`，清单同时记录压缩前后的 SHA-256。恢复到一个新目录后即可用上面的分析入口复核，例如：

```bash
python research/temporary/source_v2_gpu_20260910/archive_results.py --restore \
  --source research/temporary/source_v2_gpu_20260910/raw/smoke_v1 \
  --output /path/restored-smoke
```
