# Qwen3-8B：FT 原表完整 DT 实验

状态：已启动，完整结果尚未产出。实验计划固定为作者 Table 1 的 13 个任务、1,243 条样本；任务全部完成后才生成正式表格。

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

协议与原表来源见 [protocol.json](../../protocol.json) 和作者 [REPRODUCTION.md](../../reference/REPRODUCTION.md)。
