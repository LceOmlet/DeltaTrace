# DeltaTrace RL

本分支唯一的方法计划是 [PLAN.md](PLAN.md)：DT 提供逐 token 的
\(Q^{DT}\)、\(V^{DT}\) 和 \(A^{DT}=Q^{DT}-V^{DT}\)，接入上游 PPO
clipped objective。实现和测试应先核对该文件。

旧 README 中与该计划冲突的方法说明和验收表述已删除；本文件只记录状态和入口。

## 当前实现状态

**终局／过程奖励的采样组合已实现；真实 DT 概率比生产和三个任务的训练尚未接通。**

- `counterfactual.py::reward_event_token_credit` 接收逐 token、逐奖励事件的
  DT log ratio 和真实环境奖励，按 PLAN 的采样修正构造 Q/V 估计与 advantage。
  保留奖励的正负号、过程事件、原折扣及 policy mask；过去事件不回传到后续 token。
- 旧的归因总和归一化、整局奖励缩放、观测归因位置搬运已移除。`dt_action_smoke.py`
  只检查正式 DT 文本归因，不作为 RL 信号验收。
- `test_counterfactual.py` 检查有限结果分布下的期望 Q/V/advantage、过程奖励、
  终局单事件、观测/过去奖励屏蔽、数值稳定性，以及 minibatch 4 × 32768 的 CPU
  张量边界。`test_verl_counterfactual.py` 直接调用现有 VERL PPO loss 检查 clipping
  与 actor 梯度，没有重写 PPO。结果记录见 `results_reward_events.json`。
- 正式 DT 当前提供固定目标文本及两个联合输入端点的 log-prob 与 signed attribution；
  它们尚未接成同前缀旧策略边缘下的逐 token／逐奖励事件 log ratio。训练入口
  `METHOD=dt` 在加载模型前明确停止，避免继续执行旧算法。

这些测试使用 A6000 主机已有 Python/VERL 的 CPU 运算；没有安装依赖、下载数据、
清空缓存、占用 GPU 或运行模型。CPU 32k 张量检查不代表 32k 模型训练成功。
WebShop、Sokoban、AppWorld 当前固定方法均未取得新的训练／评估通过记录。

现有任务的实际奖励已核对：WebShop 使用 worker 的训练奖励；AppWorld 使用官方
`evaluate().success` 对应的训练奖励；Sokoban 使用官方环境每步返回的奖励。适配层
不复制评分器。环境、rollout、模型 forward、有限传播与 PPO optimizer 均保留原归属。

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
