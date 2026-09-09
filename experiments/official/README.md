# FlashTrace 原实验对齐入口

依据作者发布的[REPRODUCTION.md](reference/REPRODUCTION.md)和校验后的原始CSV固定协议，[protocol.json](protocol.json)记录原缓存哈希、完整样本数和对应结果文件。

- 原主表模型是Qwen3-8B，原模型加载器使用FP16；评分使用eager。
- 直接调用作者`075e7e4`的`_ensure_generation`、原evaluator和原RISE/MAS，不覆盖`format_prompt`。前导空格、Context、chat模板、完整target及EOS保留；每条实际评分输入都核对。
- 忠实度对照固定为`ifr_multi_hop_both, K=1`；needle对照固定为`K=3, Recall@10%`。不从FT0–3择优，不用FT0/FT3替代论文主表方法。
- DT只在进入指标时取正值部分，完整有符号向量另存。完整target的logprob为DT有限差目标；FT使用作者原聚合与递归规则。
- `paper`使用所选任务的完整发布缓存。MoreHopQA为95条，对应`n1_ifr_multi_hop_both_95_examples.csv`；HotpotQA为48条，其余为100条。
- `development16`仅NI0–7/MH0–7；`smoke`仅NI0/MH0。两者不得拿均值与完整论文CSV均值比较，也不作为正式表格完成声明。
- 可用`--datasets morehopqa`分任务执行同一固定范围。静态有限图的官方Dynamo缓存上限按所选样本数设置，防止第9种长度触发默认8次重编译上限；原方法、全图编译和原FT不改，不启用失败回退。每次运行记录实际缓存配置与首次编译成本。

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
```

Qwen3.5沿用同一原作者输入包装、数据和eager评分流程，`--family qwen35 --ft live`在同一模型上调用固定e81b3be原FT；给原FT传入已包装输入，并检查其真实模型输入与DT相同。`--ft published`明确拒绝Qwen3.5，因为原论文发布的数字来自Qwen3-8B。不得把不同权重的结果直接填入原模型对照。

输出包含逐例输入身份、原始有符号/正值归因、原删除曲线与指标、needle、调用耗时及峰值显存。正式结果保存为独立产物；此目录不接收临时实验结果。入口状态与实际已测范围见[validation.json](validation.json)。
