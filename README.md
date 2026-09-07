# DeltaTrace

带符号的双端点有限归因研究，以 FlashTrace 原基准检验质量、方向定义和完整成本。

当前实现采用真实 Qwen3/厂商 FlashAttention 的 EOS 与原输入端点，在捕获的实际张量上执行明确的有限传播规则。**还未完成生产 FA 分块后端的双端点扩展。** 不把独立分块原型、显式概率重建或本项目传播规则称为模型的原生梯度。

|内容|状态|入口|
|---|---|---|
|原生双端点捕获与层重算|已有，实际端点检查保留|`core/qwen_signed_secant_native_paired_pv_rules.py`|
|整网有限传播、三种 PV 分配|已有，PV 16条完整结果待本地核验归档|`core/qwen_signed_secant_pv_rules.py`|
|已有编译器融合|已在原样本核验|`core/compiled_*.py`|
|历史独立分块原型|仅局部数值/成本试验；较省显存但较慢，未整网接入|`research/prototypes/`|
|基于官方／厂商生产 FA 框架的有限传播扩展|待完成；不以小幅变慢为理由改用影子实现|[设计与来源要求](docs/fa_two_endpoint_extension.md)|

## 运行核心代码

将 `core` 加入 `PYTHONPATH`，使用实际配置好的 Qwen3-8B 模型和完整目标 token 轨迹：

```python
from qwen_signed_secant_native_paired_pv_rules import (
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

## 证据与维护

- `evidence/export_manifest.json` 记录原始与公开副本哈希。核心源码按字节保存。
- `research/reproduction_templates/` 是包含 `${...}` 路径占位符的公开复现模板，**不是原始冻结脚本的相同哈希副本，也不可直接当作原实验重新执行**；原始哈希保留。
- `evidence/` 是经过机器路径删减的已核验摘要，原始大数组、模型权重、私有路径和凭据不进公开 Git。
- `docs/history/` 保留阶段结果，失败结果不删除或改称成功。
- 新实验先提交公式、源码、协议和调用预算；完成后追加原曲线来源、逐样本结果、完整成本与决定。任何未验证结果必须标明状态。
- 基准采用 FlashTrace 原作者代码和数据：`075e7e44ae4d5acd2ed76e0d2aced57107d02736`、`table1-data-v1`。开发 NI0—7/MH0—7 不等于独立确认或完整跨任务胜出。

符号表示指定 EOS 基线和有限规则下的带符号分配；原 RISE/MAS 与 needle 恢复不能单独证明单 token 删除的因果符号。完整成本应计入捕获、重算、传播、输入准备和分数返回，并另列研究作业开销。
