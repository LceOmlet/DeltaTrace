# Recall 开发试测：全部候选

16 例（VT H4-C1、HotpotQA 各 8 例），用于选择方案。下表均为相对 live FT K3、相同排序方式下的 Recall@10% 差值，单位为百分点。

| Reference / 方向 / 排序 | VT H4-C1 差值 | HotpotQA 差值 |
| --- | ---: | ---: |
| full/forward/density | +0.00 | +6.01 |
| full/symmetric/density | +0.00 | +3.02 |
| full/forward/raw | +0.00 | -8.01 |
| full/symmetric/raw | +0.00 | -9.33 |
| full/reverse/density | +0.00 | -10.08 |
| full/reverse/raw | -0.28 | -11.66 |
| body/symmetric/density | -9.27 | -12.53 |
| body/reverse/raw | -12.48 | -14.29 |
| body/symmetric/raw | -7.75 | -14.48 |
| body/forward/density | -4.06 | -14.97 |
| body/forward/raw | -4.04 | -16.88 |
| body/reverse/density | -11.31 | -21.82 |

选择 `full/forward/density`：原 full-prompt reference、原方向、句内正分数均值排序。

VT 两者 8/8 例达到同一 token 预算上限，均为 70.8357%；HotpotQA DT 86.1688%，FT 80.1550%。VT 上限使严格正差值不可能，因此仅允许每例均已到上限的精确平局通过开发准入；没有改动样本、候选排序规则或验证集。见 [上限修正说明](../RECALL_CEILING_ADDENDUM.md)。

这是开发结果，不能作为验证优势。80 例验证前已固定方案；完整负向结果、原始向量和计算成本均保留。

原方向的 DT/FT 对照与前轮 GPU 运行逐位一致。交换端点的实验在部分原生批次上有数值不对称，最大目标差偏差 0.480 nats；不能把它的变化全部归因于单一有限交互规则。所选方案不使用端点交换。

[全部统计](analysis.json) · [逐例表](cases.csv) · [固定方案](choice.json) · [对照复现](control_verification.json)
