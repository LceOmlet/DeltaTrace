# 版本化评测入口

默认协议为 [`source-v2`](source_protocol.json)：只在证据正文内构造 DT reference、做 RISE/MAS 删除和计算 recovery。旧的全 prompt 协议显式使用 `--evaluation-protocol released-v1`；原 [`protocol.json`](protocol.json)、原始结果和冻结方法源码保留。

**效果状态：正文 reference 使 Recall 变差，不能作为 needle 修复推荐。** VT H4-C1 全部 100 例下降 4.55 点，HotpotQA 全部 48 例下降 8.69 点。保留原 reference、正文内句聚合的 80 例留出验证也没有建立共同优势，主要失败来自多链 VT。后续开发对照中，只解释生成的最终答案让多链 VT 的两种方法均恢复到 100%，HotpotQA 仍未胜出。更改参照、聚合与逐层对称分配的负向结果均已保留。见 [GPU 补跑](../../research/temporary/source_v2_gpu_20260910/README.md)。

## source-v2 的评分规则

- NI 使用指令与问题之间的正文；VT 使用当前题正文，排除已解示例；HotpotQA 使用文档区。区间仅从固定 prompt 模板解析，不读取 gold、答案或归因值。未知模板报错。
- 用真实 tokenizer offset 选取完全位于正文内的 token，再与作者原 `keep` 相交。保留边界空白以免漏掉带前导空格的 BPE token；若丢失任何原有效 gold，立即报错，不缩小 gold 分母。
- DT 与 FT 使用相同候选集，在过滤后重新计算 `k = max(1, ceil(f * N_source))`。同时输出 5%、10%、20%、30%、50% 的 recall、precision、预算上限、随机期望、上限归一化和机会校正值。分数取正值部分，并列按 prompt token 位置升序。原始有符号 DT 向量单独保存。
- DT reference 仅把正文候选 token 换为 EOS。原作者 RISE/MAS 删除同一候选集；每次实际模型输入都检查范围、完整 target 和 EOS，最后一个删除输入的哈希必须等于 reference。继续使用原作者的 20 步指标函数；候选不足 20 时按原函数减少步数。
- FT 必须在相同模型和原始输入上 live 执行：K=1 用于忠实度，K=3 用于主要 recovery 对照，另报 K=1 recovery。新协议拒绝复用旧论文 CSV 均值。
- `--sentence-recovery` 可另报句子/行级检索：按固定标点及换行切分，以单元内候选 token 正值均分排序，选中后返回该单元全部候选 token。预算单位是**句子/行单元数**，另报实际 token 成本，不能与同百分比 token 预算等同。

`paper` 对所选任务使用全部发布样本；默认选择 11 个有证据标注的任务（NI 6、VT 4、HotpotQA 1），共 1,048 例。`smoke` 对每个所选任务只执行第 0 例。MATH、MoreHopQA 和历史 `development16` 仅支持 `released-v1`。`paper` 表示完整缓存范围，source-v2 结果不属于旧论文协议结果。

## 运行新协议

按 [`environment.example.json`](environment.example.json) 填写模型权重、原作者代码和有限 FA 库路径及校验信息。使用对应模型已验证的依赖环境。输入仍来自作者发布的 [REPRODUCTION.md](reference/REPRODUCTION.md) 和缓存，保留原输入包装、完整固定 target 与 EOS。

```bash
# 三类正文解析各执行一例；默认 source-v2 和 live FT
python experiments/official/evaluate.py --family qwen3 --environment /path/environment.json \
  --evaluation-protocol source-v2 --selection smoke \
  --datasets niah_mq_q2 vt_h4_c1 hotpotqa_long --output /path/temporary/source-v2-smoke

# 11 任务、1,048 例；包含独立标记的句子/行级附加指标
python experiments/official/evaluate.py --family qwen3 --environment /path/environment.json \
  --evaluation-protocol source-v2 --selection paper --sentence-recovery \
  --output /path/formal/source-v2-qwen3

python experiments/official/summarize.py /path/formal/source-v2-qwen3/results.json
```

可用 `--datasets` 分任务执行，每个 `paper` 任务必须完整。Qwen3.5 使用 `--family qwen35` 和对应环境，同样 live 运行 FT。汇总拒绝混合协议、缺失 FT、错误预算、丢失 gold、部分 paper 任务及 reference/删除终点不一致的结果。每次运行保存协议及代码哈希。

`--paired-reference-audit` 会另行重新计算全 prompt reference 的 DT，再用同一正文范围、预算及删除规则评分，输出 `DT_full_reference` 对照。它用于隔离 reference 改变的作用；需要两次 DT 归因及额外一组删除曲线。

## 重放旧协议

```bash
# 历史 NI0–7/MH0–7 开发回归
python experiments/official/evaluate.py --family qwen3 --environment /path/environment.json \
  --evaluation-protocol released-v1 --selection development16 --ft live \
  --output /path/development/qwen3-16

# 旧协议的全部 13 任务，只运行 DT 并引用发布的 FT 均值
python experiments/official/evaluate.py --family qwen3 --environment /path/environment.json \
  --evaluation-protocol released-v1 --selection paper --ft published \
  --output /path/formal/released-v1-qwen3
```

`released-v1` 保留本入口原有的正值评分适配器与作者 recovery 函数。完整已发表实验快照使用独立的 signed-RISE 适配器，应按仓库 [README](../../README.md#run-model-attribution-and-evaluation) 指定的冻结 revision 重放；不能把两种 RISE 视图混称为同一实验。旧结果缺少新版本字段时，汇总按 released-v1 读取。

发布 FT 数值逐例匹配 `ifr_multi_hop_both_n1_*` trace 及原 CSV；发布说明将 recovery 描述为 K=3，但记录目录标为 n1。K=3 是 live 调用配置，不表示已验证复用 CSV 的实际执行为 K=3。Qwen3.5 拒绝 `--ft published`，因为原数字来自 Qwen3-8B。

## 已验证范围

新协议实现与 CPU 全量检查见 [source-v2 验证记录](../../research/temporary/source_protocol_v2_20260910/README.md)。1,048 例真实输入的正文映射、gold 保留、预算和 reference 已检查。后续已完成 VT H4-C1 100 例、HotpotQA 48 例的新旧 reference GPU 归因、live FT 和完整删除评测；独立复核与负向 Recall 结果见 [GPU 记录](../../research/temporary/source_v2_gpu_20260910/interim_focus_2/RESULTS.md)。

历史入口执行范围见 [`validation.json`](validation.json)，该文件是旧版本的执行记录。旧向量离线原因审计见 [needle 报告](../../research/temporary/needle_gap_20260910/README.md)。[`retrieval_views.py`](retrieval_views.py) 是该审计的辅助视图，不是 source-v2 的生产实现。
