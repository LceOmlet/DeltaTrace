# DeltaTrace RL

唯一方法规范是 [PLAN.md](PLAN.md)。环境复用见
[REMOTE_ENVIRONMENT.md](REMOTE_ENVIRONMENT.md)；本页只记录实现与测试状态。

## 当前实现：EOS DT → 奖励事件 → token PPO

2026-09-22 已移除额外参考 token 采样和逐 token 前后奖励询问。
`reward_readout.py` 现在调用正式 DT `attribute`，对每个 response / 非零未来
奖励事件返回各 token 的 signed 归因，再由 `counterfactual.py` 分事件完成
`r * (-expm1(-d))` 和 Q/V 组合。没有复制有限传播、环境评分或 PPO。

原始 token IDs、奖励事件身份及 mask 由固定 VERL collector 提供。
O/padding 不参与 actor loss；实际 EOS action 保留 action 身份，其 EOS 替换
对比可以为零。各事件分别变换后相加，不广播 span，不归一化 credit。

归因阶段通过 HF 公共接口临时切换到 FA；正常和异常退出都恢复原 actor
后端。复用 A6000 已记录的动态 shape 执行配置和持久缓存。启动脚本按实际
tokenizer 为事件询问及 target 预留空间，总上限仍为 32768，不截断已生成 action。

## 当前验证范围

- CPU：54 项测试通过（含新增 12 项原生 BF16 head 舍入边界回归）。
  覆盖数学奖励组合、终局/过程奖励、真实上游 collector/PPO loss、token 身份、
  O/padding mask、EOS 端点构造、逐事件 nonlinear 变换、后端恢复等检查。
  正式 runner 在 adapter 单元测试中使用明确标注的 test double；不能据此声称
  真实模型归因已通过。最新结果见 `results_eos_events.json`。
- `verify_positive_task_rewards.py`：三项真实官方 task worker 的正奖励轨迹。
  使用脚本/官方答案和显式数值夹具，只验证 reward 接线；不计为模型任务成功率。
- `verify_reward_readout.py`：准备了真实 Qwen + 正式 EOS DT 的三任务检查，并将
  部分 token 的有限归因与原模型单 token EOS 替换前向对照。该对照只测文本
  端点数值误差，不冒充真实环境反事实。GPU 运行状态见结果记录。
- Head 数值修复：正式 `FiniteAnswerOps` 现在使用已捕获的原生隐藏行，补上
  BF16 输出舍入的有限差分，并对少量类别行保留 FP32 系数。原 head 权重、
  logits、奖励公式与 PPO 不变；没有全轨迹信用归一化或额外模型前向。
- MetaX 三任务首个真实 head 边界误差均小于 `8e-8`；A6000 的 9 项编译/eager
  对照通过，最大误差约 `1.2e-7`。这是 head 修复的证据。
- 三任务完整 DT 仍在后续层累计误差上未通过原阈值；A6000 原生整网检查同样
  未通过。失败和逐层诊断保留在结果索引中；尚未运行本修订的 PPO 更新。

**当前 EOS 路径的三任务模型训练、32k 实机容量、稳定多步更新和吞吐/显存验收
尚未完成。旧生成前后读出探针、旧 32k 训练及 CPU 张量检查均不能替代。**

## 入口与历史结果

- `run_verl_agent.sh`：固定上游 VERL 训练入口，DT/PPO/GRPO 共用原训练基建。
- `patch_verl_agent2.py`：固定上游的薄接口补丁与共有左 padding 裁剪；默认上游路径
  保持原行为。环境、rollout、optimizer 和 PPO clipping 都由上游实现。
- `results_env_smoke.json`：历史环境 readiness。
- `results_reward_events.json`、`results_event_transport.json`：奖励代数和传输历史检查。
- `results_reward_readout.json`：已撤下的逐前缀读出历史结果，非当前 EOS 方法结果。
- `results_counterfactual_32k_a6000.json`、其他旧归因/训练记录：只适用于各自历史
  实现，不能作为当前 PLAN 的验收。

实际累计长度、DT 调用次数、首轮/后续时间及显存必须来自新日志；不将配置上限
当作真实任务长度，不将所有非零 reward 项的多个 DT 调用报告为一次。
