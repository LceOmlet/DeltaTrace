# DeltaTrace RL

本分支唯一的方法计划是 [PLAN.md](PLAN.md)：DT 提供逐 token 的
\(Q^{DT}\)、\(V^{DT}\) 和 \(A^{DT}=Q^{DT}-V^{DT}\)，接入上游 PPO
clipped objective。实现和测试应先核对该文件。

旧 README 中与该计划冲突的方法说明和验收表述已删除；本文件只记录状态和入口。

## 当前实现状态

**固定计划尚未完成实现与验收。** 当前工作区仍有偏离该计划的实验代码，不能将
`METHOD=dt` 的名称、已有归因向量或一次 optimizer step 视为计划已经实现。
当前信用适配代码尚未构造并验证计划要求的逐 token \(Q^{DT}/V^{DT}\)。

2026-09-21 的等价性复核已补入同一份 [PLAN.md](PLAN.md)：精确文本 attribution
与回报守恒不足以推出精确环境 advantage；完整回报 oracle 的条件等价命题仍成立。
当前缺少从现有 DT 输出到该目标的已证明计算，完整方法的实现可行性和一次 DT
开销均未建立，不能把补充定义或接口连通当作完成。

2026-09-21 的检查发现，旧适配代码仍在进行归因缩放及观测位置映射；这些代码
不构成本计划的实现依据。后续修改应直接对照 [PLAN.md](PLAN.md)，不沿用旧说明。
本次文档整理没有修改训练算法，也没有新增训练通过记录。

## 环境与历史记录

已有 A6000 环境、权重、上游版本、缓存和任务资产的复用方式见
[REMOTE_ENVIRONMENT.md](REMOTE_ENVIRONMENT.md)。恢复工作从该记录开始，
不要重装已经可用的环境。

`results_*.json` 是历史运行记录，保留其原始数据；其中的 `passed` 等状态只适用
于记录当时的实现与测试范围，**均不能直接作为 PLAN.md 的验收结果**：

- [环境 smoke](results_env_smoke.json)记录官方任务环境的历史 reset/step 检查。
- [旧 32k 记录](results_counterfactual_32k_a6000.json)与
  [旧多步记录](results_counterfactual_multistep.json)使用已被替代的信用方案。
- `results_dt_action_*.json` 记录归因接口检查，不代表完整 RL 信号等价性。
- 其他旧训练和探测记录也不替代三个任务上对当前固定方法的验证。

只有当前固定实现的对应代码、运行配置、原始日志和明确测试范围才能形成新的验收记录。
