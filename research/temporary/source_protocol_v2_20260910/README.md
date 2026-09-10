# source-v2 测法修复与验证

2026-09-10。在隔离工作树 `DeltaTrace-needle-gap-audit`、分支 `codex/needle-gap-audit` 实现。默认评测从全 prompt 改为证据正文范围；FT 与 DT 使用相同候选集和重新计算的预算，DT reference 与忠实度删除使用同一范围。冻结的 27 个方法依赖文件全部校验通过，未改模型、FT 或有限传播方法。

实现入口与运行命令见 [评测说明](../../../experiments/official/README.md)，固定参数见 [source_protocol.json](../../../experiments/official/source_protocol.json)。旧协议显式使用 `released-v1`，原发布结果保留。本目录是实现验证记录，不是新协议质量结果。

## 完成的修复

1. 从 NI、VT、HotpotQA 的固定模板解析正文，完全不依赖 gold 或归因值。排除指令、已解示例、当前问题与答案前缀；通过真实 tokenizer offset 映射到作者原候选 token。
2. 对两种方法同时使用 `k = max(1, ceil(f * N_source))`，报告 5%、10%、20%、30%、50% 预算曲线。保存 recall、precision、理论上限、随机期望和归一化诊断。
3. 用同一正文候选集构造 EOS reference 和原 RISE/MAS 删除路径。运行时校验每条实际模型输入，保留其他上下文和完整固定 target，强制 reference 与最终删除输入哈希相同。
4. 可选句子/行级指标使用独立的单元预算，选中整个单元的候选 token，同时报告实际 token 成本。
5. 默认要求 live FT K1/K3，拒绝在 source-v2 表格中引用旧 CSV。汇总检查版本、全部预算点、gold 保留、完整任务范围和删除端点，避免新旧结果混用。

正文字符区间保留边界空白。全量检查曾发现 208 个 NI 样本的首个 gold token 含前导空格，裁掉空白会错误删除该 token；已修复并添加回归用例。生产入口遇到任何原有效 gold 丢失都会报错。

## 验证结果

| 检查 | 结果 | 记录 |
| --- | --- | --- |
| 生产正文解析、实际 token 对齐、gold 保留、预算及 reference | 11 任务，1,048/1,048 例通过 | [全量回执](source_preflight.json)、[逐例映射和预算](source_preflight_cases.csv) |
| 回归测试 | 26 项通过 | [测试输出](unit_tests.txt) |
| 旧版汇总兼容性 | 13 任务、1,243 例的实际报告，除新增版本元数据外，与修复前汇总完全相同 | [实现回执](implementation_validation.json) |
| CLI、语法和冻结方法身份 | 通过，27 个方法文件哈希一致 | [实现回执](implementation_validation.json) |
| 新 reference 的 GPU 归因和完整删除曲线 | 尚未执行，模型调用数为 0 | [全量回执](source_preflight.json) |

全量检查使用哈希匹配的原缓存、真实 Qwen3 tokenizer 和旧运行的输入/向量。它验证新生产函数的区间、映射、预算、reference 构造和附加指标计算；没有把旧归因重新排名当作 source-v2 的 DT 质量结果。实际 GPU 删除输入与 reference 的匹配目前由生产入口的运行时检查约束，尚无新 GPU 执行回执。Qwen3.5 的真实 tokenizer 与模型执行也未在本次检查中运行。

## 预算变化的含义

下表是逐例候选数和预算的均值，不是模型质量分数。

| 任务 | 原候选数 | 正文候选数 | 原 10% token 预算 | 新 10% token 预算 |
| --- | ---: | ---: | ---: | ---: |
| NI MQ-Q8 | 346.8 | 235.4 | 35.1 | 24.0 |
| NI MV-V8 | 538.4 | 482.0 | 54.3 | 48.7 |
| VT H4-C1 | 812.4 | 332.5 | 81.7 | 33.9 |
| VT H10-C1 | 829.0 | 330.9 | 83.3 | 33.5 |
| HotpotQA | 1,327.4 | 1,258.3 | 133.2 | 126.3 |

过滤后，10% 对应的绝对 token 数会下降。对于高 gold 密度任务，raw recall 的理论上限仍可能很低，因此必须连同多预算曲线解释。逐例上限是 `min(1, k/G)`，随机期望是 `k/N_source`，上限归一化值是 `hits/min(k,G)`；机会校正值是 `(recall-random)/(ceiling-random)`，分母为零时记录 null，并在均值旁保留有效例数。

## 复现 CPU 验证

在仓库根目录，使用 Python 3.12、NumPy 和 tokenizers 0.22.2。以下路径对应本工作区现有的已冻结输入；在其他机器上替换为同哈希文件的位置。

```bash
python experiments/official/validate_source_protocol.py \
  --publication ../DeltaTrace-paper-qwen3/experiments/official/results/qwen3_8b_table1_20260909 \
  --data ../audit/published_flashtrace/table1-data-v1/extracted/data \
  --tokenizer ../audit/deltatrace-figures-20260909/qwen3-tokenizer.json \
  --output research/temporary/source_protocol_v2_20260910

python research/temporary/source_protocol_v2_20260910/verify_implementation.py \
  --publication ../DeltaTrace-paper-qwen3/experiments/official/results/qwen3_8b_table1_20260909
```

第一条命令检查全部 1,048 例，失败即返回非零退出码。第二条执行回归测试、CLI/语法检查、方法身份验证，并使用修复前固定 revision 的汇总代码验证 13 个旧报告。回执记录所测代码和输入的 SHA-256。
