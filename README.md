# DeltaTrace

带符号的双端点有限归因研究，以 FlashTrace 原基准检验质量、方向定义和完整成本。

当前实现采用真实 Qwen3/厂商 FlashAttention 的 EOS 与原输入端点，在捕获的实际张量上执行明确的有限传播规则。**FA 框架有限扩展已完成首个真实层核验，尚未整网接入。** 不把独立分块原型、显式概率重建或本项目传播规则称为模型的原生梯度。

|内容|状态|入口|
|---|---|---|
|原生双端点捕获与层重算|公开FA入口已核验，旧私有路径保留为对照|[公开捕获结果](docs/history/公开FA捕获_P1数值与成本_20260907.md)|
|整网有限传播、三种 PV 分配|原16条开发样本、176条原曲线已独立核验；P1进入下一步开发|[完整结果](docs/history/PV交互分配_三规则完整16条结果_20260907.md)|
|已有编译器融合|已在原样本核验|`core/compiled_*.py`|
|历史独立分块原型|仅局部数值/成本试验；较省显存但较慢，未整网接入|`research/prototypes/`|
|FA 双端点有限传播加速|真实单层局部数值通过；2.106ms/64MB，尚未整网接入|[首验结果](docs/history/可追溯FA有限传播_真实层首验_20260907.md)|
|原评估器的批量删除曲线|已实现；B4在三个原样本提速1.19—1.52x，数值差在冻结容差内|[批量结果与后端契约](docs/history/默认FA兼容边界与原评估批处理_20260907.md)|

当前研究配置为 [`content_P1`](configs/pv_content_P1_development.json)。NI 恢复率77.52%（对称77.90%，最佳 FT59.33%）；NI RISE0.05742（最佳 FT0.06193），MH RISE0.05848（最佳 FT0.11013），均为越低越好。NI/MH MAS也均改善。用户明确接受约0.38个百分点的恢复下降，原冻结“对称无退步”附加筛选仍如实记为未通过；对 FT 原联合质量门槛通过。完整耗时/FT逐样本比值中位数1.057，最慢1.463；16条显存均低于普通原生 FA 反向参照。仅开发结果，未完成独立或长输入确认。

后续效率优先，并把双端点算法固定在稳定的公开接口/张量契约层，避免随 FA 私有版本反复移植。真实模型默认 FA 不替换。公开捕获已在三个原样本通过：P1完整向量相同，额外耗时1.05%—1.50%、峰值最多增加0.38MB；每层新增一次原FA调用，完整计费。多样本归因尚未完成，不能把评估批处理称为批量归因。

当前主瓶颈是有限传播中显式 N×N 矩阵随 rollout 长度增长的平方级存储；公开捕获与评估批处理均未解决它。优先[保持 P1 并消除这些中间矩阵](docs/history/finite_attention_storage_priority_20260907.md)，批处理服从显存和效率预算。`research/prototypes` 中新增的异长批量源码尚未运行，不能当作已完成批量归因。

## 运行核心代码

将 `core` 和 `research/runtime` 加入 `PYTHONPATH`，使用实际配置好的 Qwen3-8B 模型和完整目标 token 轨迹：

```python
from qwen_signed_secant_paired_public_fa import (
    capture_checkpoint_pair, propagate_paired_secant,
)

# 模型为 eval、FP16、flash_attention_2；仅 eligible 输入替换为 EOS。
# baseline_ids 与 input_ids 的固定生成轨迹完全相同。
before, after = capture_checkpoint_pair(
    model, baseline_ids, input_ids, attention_mask, prompt_len,
)
result = propagate_paired_secant(model, before, after, pv_rule="content_P1")
signed_scores = result["signed_full_sequence"]
```

`pv_rule` 只能为 `symmetric`、`content_P1`、`content_P0`，默认对称。后两者分别为 ΔP·V0 + P1·ΔV、ΔP·V1 + P0·ΔV。混合乘积只是归因恒等式，不是额外模型反事实前向。其他有限规则保持不变。

复现环境记录：MetaX C550，PyTorch `2.8.0+metax3.5.3.9`、Transformers `4.57.3`、FlashAttention `2.6.3+metax3.5.3.9torch2.8`、Triton `3.0.0`。不假定不同设备/上游版本有相同 ABI 或表现，不自动覆盖安装的模型、FA 或 autograd。

原曲线批调度可以直接接受原评估器及真实删除状态，按精确长度分桶：

```python
from research.runtime.original_ft_batched_evaluation import evaluate_requests

# evaluator 是 FlashTrace 原 LLMAttributionEvaluator。
# requests: 唯一标识 -> (prompt_ids[1,N], fixed_response_ids[1,M])。
scores, cost = evaluate_requests(evaluator, requests, batch_size=4)
```

`cost` 同时返回物理前向次数和评估轨迹数。该函数不生成新删除顺序、不生成答案，也不改变评分器；原基准使用的实际后端应保持一致。当前 B4 数值/性能核验限于报告中的三个原样本。

## 证据与维护

- `evidence/export_manifest.json` 记录原始与公开副本哈希。核心源码按字节保存。
- `research/reproduction_templates/` 是包含 `${...}` 路径占位符的公开复现模板，**不是原始冻结脚本的相同哈希副本，也不可直接当作原实验重新执行**；原始哈希保留。
- `evidence/` 包含已核验摘要，以及176条原始数值曲线、评分、删除分组和带符号向量；模型权重、输入文本/token IDs、私有路径和凭据不进公开 Git。
- `third_party/metax_fa_2_5_3/` 保存 MetaX 官方仓库固定提交的78个原始文件及逐文件哈希。该公开版本是2.5.3，不能冒充当前安装2.6.3二进制的同版源码；构建与后续扩展状态另行记录。
- `docs/history/` 保留阶段结果，失败结果不删除或改称成功。
- 新实验先提交公式、源码、协议和调用预算；完成后追加原曲线来源、逐样本结果、完整成本与决定。任何未验证结果必须标明状态。
- 基准采用 FlashTrace 原作者代码和数据：`075e7e44ae4d5acd2ed76e0d2aced57107d02736`、`table1-data-v1`。开发 NI0—7/MH0—7 不等于独立确认或完整跨任务胜出。

符号表示指定 EOS 基线和有限规则下的带符号分配；原 RISE/MAS 与 needle 恢复不能单独证明单 token 删除的因果符号。完整成本应计入捕获、重算、传播、输入准备和分数返回，并另列研究作业开销。
