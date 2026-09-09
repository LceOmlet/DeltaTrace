# FlashTrace 原实验对齐入口

依据作者发布的[REPRODUCTION.md](reference/REPRODUCTION.md)和校验后的原始CSV固定协议，[protocol.json](protocol.json)记录原缓存哈希、完整样本数和对应结果文件。

加速分支的`eval4-compact-json-20260909`只将报告保存切换为Python标准库紧凑JSON，其他入口源码字节完全保留；真实报告解析后数据相同，见[采用记录](../../research/temporary/acceleration_20260909/compact_report_adoption.json)。已有数值执行记录仍固定在eval3，进行中的Qwen3原表及后续Qwen3.5成本任务未切换版本。

- 原主表模型是Qwen3-8B，原模型加载器使用FP16；评分使用eager。
- 直接调用作者`075e7e4`的`_ensure_generation`、原evaluator和原RISE/MAS，不覆盖`format_prompt`。前导空格、Context、chat模板、完整target及EOS保留；每条实际评分输入都核对。
- `--ft live`保留既定开发对照：`ifr_multi_hop_both, K=1`忠实度及`K=3, Recall@10%`。`--ft published`直接读取原CSV；配套原始记录为`n1`且汇总一致，不能将这些复用数值标成已验证的K=3运行。作者发布README在此处与保存的轨迹标签不同，见[原表来源核验](https://github.com/LceOmlet/DeltaTrace/blob/d09b73c1b38291e9fe89032c04aa8dd6aa2e9c27/experiments/official/results/qwen3_8b_table1_20260909/README.md)。不从FT变体中择优。
- DT的RISE使用有符号排序，MAS与needle使用正值视图，完整有符号向量另存。两种视图都调用作者原函数；仅当正分严格排序相同、且作者响应在删到非正分之前已归零时，复用已返回的相同RISE并保存证明。其余情况额外执行一次原评分，不改指标公式。完整target的logprob为DT有限差目标；FT使用作者原聚合与递归规则。
- `paper`使用所选任务的完整发布缓存。MoreHopQA为95条，对应`n1_ifr_multi_hop_both_95_examples.csv`；HotpotQA为48条，其余为100条。
- `development16`仅NI0–7/MH0–7；`smoke`仅NI0/MH0。两者不得拿均值与完整论文CSV均值比较，也不作为正式表格完成声明。
- 可用`--datasets morehopqa`分任务执行同一固定范围。静态有限图的官方Dynamo缓存上限按所选样本数设置，防止第9种长度触发默认8次重编译上限；原方法、全图编译和原FT不改，不启用失败回退。每次运行记录实际缓存配置与首次编译成本。
- 默认`--dt-backend clean --sample-batch 1`保持干净方法。Qwen3.5另提供显式`accelerated_qwen35`后端，按每个任务的输入长度组成真实样本batch；只扩展输入打包与执行调度，调用已固定的加速控制器，不改作者评分。新入口已完成Qwen3.5原16例B2及Qwen3两例默认入口验证。

## 运行

按[environment.example.json](environment.example.json)填写权重、原作者代码和已编译有限FA库的路径与校验信息。使用对应模型已验证的依赖环境，不修改作者代码。

```bash
# 正式入口的小规模执行检查，输出到临时测试目录
python experiments/official/evaluate.py --family qwen3 --environment /path/environment.json \
  --selection smoke --ft live --output /path/temporary/qwen3-smoke

# 既定16例开发回归：原FT K=1忠实度与K=3 needle
python experiments/official/evaluate.py --family qwen3 --environment /path/environment.json \
  --selection development16 --ft live --output /path/development/qwen3-16

# 正式原表全部13个任务：只运行DT，直接引用原论文FT结果（默认行为）
python experiments/official/evaluate.py --family qwen3 --environment /path/environment.json \
  --selection paper --output /path/formal/qwen3-table

# 也可分任务运行；每个任务使用全部发布样本
python experiments/official/evaluate.py --family qwen3 --environment /path/environment.json \
  --selection paper --datasets niah_mq_q2 morehopqa --ft published --output /path/formal/qwen3-ni-mh

# 汇总完成的运行；只在完整同模型任务上附作者FT原表均值
python experiments/official/summarize.py /path/formal/qwen3-ni-mh/results.json

# 可选加速后端：真实B2，Qwen3.5原16例；指标仍逐例调用作者实现
TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS=0 python experiments/official/evaluate.py \
  --family qwen35 --environment /path/environment.json --selection development16 \
  --dt-backend accelerated_qwen35 --sample-batch 2 --ft live --output /path/development/qwen35-b2
```

Qwen3.5沿用同一原作者输入包装、数据和eager评分流程，`--family qwen35 --ft live`在同一模型上调用固定e81b3be原FT；给原FT传入已包装输入，并检查其真实模型输入与DT相同。`--ft published`明确拒绝Qwen3.5，因为原论文发布的数字来自Qwen3-8B。不得把不同权重的结果直接填入原模型对照。

输出包含逐例输入身份、原始有符号/正值归因、原删除曲线与指标、needle、调用耗时及峰值显存。正式结果保存为独立产物；此目录不接收临时实验结果。入口状态与实际已测范围见[validation.json](validation.json)。

加速后端的完整批次诊断与守恒总量存于`DT_batches`，逐例只引用对应批次及样本位置，不把批次总量伪装成每例总量。RISE/MAS评分和FT对照仍逐例进行；`sample_batch`描述DT归因吞吐，不描述指标评分的batch。当前实际验收为B2，不把它外推任意batch或长上下文。所有质量退步和运行成本见[记录](../../research/temporary/acceleration_20260909/official_batch_entry_summary.json)。

当前厂商Triton的持久化自动调优开关在原生前向有效，但完整DT的原生伴随会触发过长缓存文件名错误。上述运行明确关闭该开关；不修改框架或另写缓存实现。普通Triton/Inductor编译缓存照常使用。

## Explicit diagnostic scheduling backends

The existing driver also accepts `--dt-backend deferred_qwen3 --sample-batch 1`
and `--dt-backend deferred_qwen35 --sample-batch 2` with the matching family.
They preserve the frozen target and metric protocol; clean remains the default.
See [backend provenance and measured limits](../../deltatrace/accelerated/DEFERRED.md).
