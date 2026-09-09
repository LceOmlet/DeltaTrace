# Qwen3-8B：FT 原表完整 DT 实验

状态：已完成 3/13 个完整任务，共 300/1,243 条；其余任务继续运行。实验范围固定为作者 Table 1 的完整发布缓存。

目前完成 `niah_mq_q2` 的全部 100 条：DT/FT 的 RISE 为 0.064538/0.068136，MAS 为 0.092454/0.162782，Recall@10% 为 0.667608/0.483435。它是该任务完整结果，不能替代其余任务或全表均值。

`niah_mq_q4` 也已完成全部 100 条：DT/FT 的 RISE 为 0.130963/0.113366，MAS 为 0.182525/0.193532，Recall@10% 为 0.478293/0.413260。该任务 DT 的 RISE 落后，MAS 和 needle 领先；原样保留所有结果，不根据任务成绩修改冻结方法。

`niah_mq_q8` 已完成全部 100 条：DT/FT 的 RISE 为 0.375735/0.351848，MAS 为 0.544922/0.426987，Recall@10% 为 0.132395/0.074983。该任务 DT 的 RISE、MAS 均落后，needle 领先；这些差距同样完整入表，不能根据前两项任务推广为全面领先。

已完成任务见 [table_progress.csv](table_progress.csv) 和 [LaTeX](table_progress.tex)；[per_case.csv](per_case.csv) 保存逐例指标，[summary.json](summary.json) 保存输入、源码、权重和原始结果的校验记录。`raw/` 中保留原结果 JSON 的无损 gzip 和原始归因 NPZ。所有任务完成后导出 `table.csv` / `table.tex`；导出器默认拒绝不完整全表。

- 方法：冻结的 `clean-v1-20260909` Qwen3 DT，`content_P1`，不采用后续逐层修补。
- 评测：`clean-v1-eval3-native-batch-20260909` 的正式入口；Qwen3 使用默认 clean 后端，真实样本 batch=1，内部两个端点共用原生 FA batch=2。Qwen3.5 的加速后端没有混入本次实验。
- 输入：作者 `table1-data-v1` 的完整发布缓存；使用原输入包装、完整 target + EOS、原 keep/gold 映射，不重新生成、采样或过滤。
- 模型与评分：原 Qwen3-8B 权重逐文件校验，FP16；原作者 eager 评分，20 次删除、21 个响应点。
- 指标：RISE 使用有符号排序；MAS 和 needle 使用正值部分。原始有符号向量、正值向量、实际删除输入哈希和原响应曲线均保存。
- 对照：FT 直接引用作者发布的完整任务 CSV。RISE/MAS 为 K=1，needle 为 K=3、Recall@10%；不重跑或修改 FT。
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
