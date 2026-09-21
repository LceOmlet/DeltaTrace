# DeltaTrace RL

本分支唯一的方法计划是 [PLAN.md](PLAN.md)：DT 提供逐 token 的
\(Q^{DT}\)、\(V^{DT}\) 和 \(A^{DT}=Q^{DT}-V^{DT}\)，接入上游 PPO
clipped objective。实现和测试应先核对该文件。

旧 README 中与该计划冲突的方法说明和验收表述已删除；本文件只记录状态和入口。

## 当前实现状态

**固定计划尚未完成实现与验收。** 当前工作区仍有偏离该计划的实验代码，不能将
`METHOD=dt` 的名称、已有归因向量或一次 optimizer step 视为计划已经实现。
当前信用适配代码尚未构造并验证计划要求的逐 token \(Q^{DT}/V^{DT}\)。

用户最新明确的 token 反事实/价值假设、固定 Q/V 组合与近似部分已写入
同一份 [PLAN.md](PLAN.md)。实际 DT 估计允许近似；估计误差本身不阻止开发，
也不否定已经正确的组合与接口。当前需要继续核对正式 DT 输出的语义对应，并
完成三个任务的训练与资源验证；不能把估计误差和已有适配代码的错误混为一谈。
“每次只处理一个未定部分”是此前自行添加的约束，已按用户纠正撤下。

2026-09-21 的检查发现，旧适配代码仍在进行归因缩放及观测位置映射；这些代码
不构成本计划的实现依据。后续修改应直接对照 [PLAN.md](PLAN.md)，不沿用旧说明。
本次修订固定上述边界，没有修改训练算法，也没有新增训练通过记录。

同日已补充奖励根节点的代数推导：将正式 DT 的 log-prob 差通过概率的
对数平均精确转换为 reward 加权的概率差，并证明在原有条件价值与旧策略参照
下，其奖励结果求和对应 Q-V。正负奖励和等端点的 CPU 数值检查已完成；这只
验证该恒等式，未验证真实模型归因或任务训练。三个任务的奖励结果与正式文本
目标的具体对应，以及其结果期望的实际计算，仍未完成，不能用此推导冒充接通。

已从本地 Git 提交核对正式入口 `deltatrace/profiles/official.py` 中的
`make_qwen35_runner`，以及 `deltatrace/clean/qwen35/qwen35_answer_finite.py`
中的 `PackedAnswerTargets`、`FiniteAnswerOps`。目标行选择、原模型输出与有限
传播 seed 已有归属，后续接入复用这些实现。当前未定部分是正式 DT 输出到固定
token 价值/策略参照的对应；上游 rollout、环境、PPO 损失和 optimizer 不因此重写。

2026-09-21 经新 A6000 地址核对了实际任务奖励接口：WebShop worker 保留
`info['task_score']`，并将终局满分映射为训练 reward 10、其他为 0；AppWorld
在结束时使用官方 `evaluate().success`，reward 为 10 或 0；Sokoban 原环境
包含步惩罚 -0.1、箱子离开目标 -1、进入目标 +1 和完成 +10。环境已有奖励，
不需要复制其评分器。正式 DT 当前 `PackedAnswerTargets` 接收文本 token IDs
及目标位置，`FiniteAnswerOps` 计算这些文本的 log-prob。将这些文本目标对应到
上述任务奖励结果、再计算同前缀旧策略参照的具体实现仍未闭合；未启动旧适配
训练来替代这项工作，也未声称当前固定算法已经无缺口。

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
