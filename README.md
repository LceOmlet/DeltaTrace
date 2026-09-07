# DeltaTrace

**最新优先级：[FA 主线优先](docs/history/FA主线优先_20260907.md)。先完成 FA 配套开销与集成，暂停 MLP 和输出头优化。已移除 GQA 输入头复制，并通过原 NI2 的 B1 和真实四例 B4 全向量核验；整网峰值未下降，尚无稳定加速结论。继续处理 FA 周边布局、端点平均与传播重复，旧 FA 证据不自动继承。**

最新实现：[原生 MLP 复用与剩余优化](docs/history/原生MLP复用_两例小预算结果_20260907.md)。公开 PyTorch SAC 复用真实 MLP 输出，两例共 12 次归因；NI2 相对原 P1 耗时下降 7.7%，峰值显存增加 5.12 GB；MH5 单次配对下降 2.4%，尚不稳定。同次完整新旧分数相同，但未继承不同分数的历史质量曲线，未重新比较 FT。**按用户更新取消一次反向加固定 72.5 MB 的硬上限，按匹配 FA 的速度、显存与 batch/长度容量共同判断。**

当前执行约束：[官方数据与小预算验证](docs/history/官方数据优先与小预算验证_20260907.md)。使用作者处理后原缓存；本地生成已停止并排除。现有证据足以先改P1重复计算，初筛限两个原开发样本、最多12次归因和0条新删除曲线；有依据再扩大，不默认重跑全量。

最新公平性修正：[原FT seq入口](docs/history/原FT_seq入口公平性修正_20260907.md)。用作者原函数省去无关row/rec视图后，16例完整FT分数相同、耗时约降30.4%；P1相对该入口反而慢37.7%（配对中位数，16/16均慢）。此前0.928仅为相对作者总入口，不构成效率胜出。下一步按最新用户要求先完成FA主线；非注意力优化后移。

最新成本核验：[FA有限P1与原FT同次成本](docs/history/FA有限P1_原FT同次完整成本_20260907.md)。16例9方法各自完整重跑；P1/FTboth1配对耗时比中位数0.928，仍慢的样本全部保留。原质量仅在完整分数相同后关联，非独立确认；NIq2-100与MH95均已历史使用，新MH预检通过但缺原API配置。

最新符号核验：[P1条件有限效应](docs/history/P1有限分配_条件符号诊断_20260907.md)。384个开发token中201个随干预条件反号；不能把P1负分直接当作原输入删除方向。原质量候选保持，符号与独立确认尚未完成。

最新资源核验：[原长输入与同批反向参照](docs/history/FA有限传播_原长输入与同批反向参照_20260907.md)。原长输入归因耗时下降43%—44%，真实B4显存低于同批普通FA反向；不等于长输入质量确认。新增原MoreHopQA来源与采样协议已准备，尚未生成新确认缓存。

带符号的双端点有限归因研究，以 FlashTrace 原基准检验质量、方向定义和完整成本。

当前实现采用真实 Qwen3/厂商 FlashAttention 的 EOS 与原输入端点，在捕获的实际张量上执行明确的有限传播规则。**FA 框架有限扩展已完成原16条整网开发核验；四个原样本的真实 B2/B4 归因也已完成。** 不把独立分块原型、显式概率重建或本项目传播规则称为模型的原生梯度。

|内容|状态|入口|
|---|---|---|
|原生双端点捕获与层重算|公开FA入口已核验，旧私有路径保留为对照|[公开捕获结果](docs/history/公开FA捕获_P1数值与成本_20260907.md)|
|整网有限传播、三种 PV 分配|原16条开发样本、176条原曲线已独立核验；P1进入下一步开发|[完整结果](docs/history/PV交互分配_三规则完整16条结果_20260907.md)|
|已有编译器融合|已在原样本核验|`core/compiled_*.py`|
|历史独立分块原型|仅局部数值/成本试验；较省显存但较慢，未整网接入|`research/prototypes/`|
|FA 双端点有限传播加速|原16条配对耗时中位数下降11.2%，原开发门槛通过|[完整核验](docs/history/FA有限传播_完整16条与真实批处理_20260907.md)|
|真实多样本归因|四例B4吞吐1.430×，完整峰值21.70GB，needle恢复不变|[批处理核验](docs/history/FA有限传播_完整16条与真实批处理_20260907.md)|
|原评估器的批量删除曲线|已实现；B4在三个原样本提速1.19—1.52x，数值差在冻结容差内|[批量结果与后端契约](docs/history/默认FA兼容边界与原评估批处理_20260907.md)|

当前研究配置为 [`content_P1`](configs/pv_content_P1_development.json)。FA有限版本在原16条开发集的NI恢复率77.52%（最强FT历史对照59.33%），NI/MH RISE为0.05736/0.05839（最强FT分别0.06193/0.11013），MAS也较低。RISE/MAS越低越好。微小浮点变化不算新的算法提升；原对称无退步附加门槛仍记录失败，用户已接受该取舍。仍需独立质量和原长输入确认。

新有限路径已移除全局attention N×N中间矩阵，实际默认模型FA保持不变。可追溯有限扩展在 `research/prototypes/qwen_signed_secant_paired_vendor_fa.py`，多样本入口在 `research/prototypes/qwen_signed_secant_batched_vendor_fa_public.py`；明确使用固定厂商FA框架的独立库，未宣称任意设备/FA版本即插即用。

原16条采用单样本归因、B1评分。后续四例测试才是真实B2/B4归因和B4原评分，不能混称。评分的原生行间端点差异已直接复现，原曲线保留各自真实值；[完整报告](docs/history/FA有限传播_完整16条与真实批处理_20260907.md)说明数值、内存、吞吐与证据边界。短输入下一轮使用B4需保持全方法相同调度，长输入batch仍需实际资源确认。

## 运行核心代码

以下是保留的显式有限传播对照入口。将 `core` 和 `research/runtime` 加入 `PYTHONPATH`，使用实际配置好的 Qwen3-8B 模型和完整目标 token 轨迹：

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

`cost` 同时返回物理前向次数和评估轨迹数。该函数不生成新删除顺序、不生成答案，也不改变评分器；原基准使用的实际后端应保持一致。原调度器的三例核验保留；新增四例的归因及完整原曲线批处理见[最新报告](docs/history/FA有限传播_完整16条与真实批处理_20260907.md)。

## 证据与维护

- `evidence/export_manifest.json` 记录原始与公开副本哈希。核心源码按字节保存。
- `research/reproduction_templates/` 是包含 `${...}` 路径占位符的公开复现模板，**不是原始冻结脚本的相同哈希副本，也不可直接当作原实验重新执行**；原始哈希保留。
- `evidence/` 包含已核验摘要，以及176条原始数值曲线、评分、删除分组和带符号向量；模型权重、输入文本/token IDs、私有路径和凭据不进公开 Git。
- `third_party/metax_fa_2_5_3/` 保存 MetaX 官方仓库固定提交的78个原始文件及逐文件哈希。该公开版本是2.5.3，不能冒充当前安装2.6.3二进制的同版源码；构建与后续扩展状态另行记录。
- `docs/history/` 保留阶段结果，失败结果不删除或改称成功。
- 新实验先提交公式、源码、协议和调用预算；完成后追加原曲线来源、逐样本结果、完整成本与决定。任何未验证结果必须标明状态。
- 基准采用 FlashTrace 原作者代码和数据：`075e7e44ae4d5acd2ed76e0d2aced57107d02736`、`table1-data-v1`。开发 NI0—7/MH0—7 不等于独立确认或完整跨任务胜出。

符号表示指定 EOS 基线和有限规则下的带符号分配；原 RISE/MAS 与 needle 恢复不能单独证明单 token 删除的因果符号。完整成本应计入捕获、重算、传播、输入准备和分数返回，并另列研究作业开销。
