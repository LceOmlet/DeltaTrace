# DeltaTrace RL

本分支唯一的方法计划是 [PLAN.md](PLAN.md)：DT 提供逐 token 的
\(Q^{DT}\)、\(V^{DT}\) 和 \(A^{DT}=Q^{DT}-V^{DT}\)，接入上游 PPO
clipped objective。实现和测试应先核对该文件。

旧 README 中与该计划冲突的方法说明和验收表述已删除；本文件只记录状态和入口。

## 当前实现状态

**用户批准的奖励事件目标读出已实现并接入生产器；真实 Qwen 读出与有限传播已执行，
三个任务的完整训练仍在验证，尚未验收。**

- `counterfactual.py::reward_event_token_credit` 接收逐 token、逐奖励事件的
  DT log ratio 和真实环境奖励，按 PLAN 的采样修正构造 Q/V 估计与 advantage。
  保留奖励的正负号、过程事件、原折扣及 policy mask；过去事件不回传到后续 token。
- `policy_marginal_log_ratio` 已实现同前缀候选反事实差的旧策略边缘计算：
  对概率比加权求和后取 log，候选维可分块处理；不将截断的策略质量重新归一化。
  它消费候选反事实差，不是已经完成的正式 DT 反事实差生产器。
- `reward_event_credit_for_episode` 按原始 rollout 行及 `env_step` 对齐每步奖励，
  使用原始 response IDs 和 attention mask，不通过最终 prompt 反找历史 token。
  collector 在 DT 路径保留实际 next observation、info 和 done 的快照；生成 token
  不因环境 action parser 的改写而被替换。原始非 DT 收集路径不增加这些快照。
- trainer 现分别接收 `dt_token_advantages`、`dt_q_estimates`、`dt_v_estimates`；
  `returns` 使用采样 Q，修正了旧接线将 advantage 复制成 return 的错误。
  启动配置以 `clip_ratio_c=inf` 关闭上游可选的 dual clipping，保持固定计划的
  原始 PPO clipped objective；PPO loss、环境评分及 rollout 状态机仍使用原上游实现。
- 旧的归因总和归一化、整局奖励缩放、观测归因位置搬运已移除。`dt_action_smoke.py`
  只检查正式 DT 文本归因，不作为 RL 信号验收。
- `test_counterfactual.py` 检查有限结果分布下的期望 Q/V/advantage、过程奖励、
  终局单事件、观测/过去奖励屏蔽、数值稳定性，以及 minibatch 4 × 32768 的 CPU
  张量边界。`test_verl_counterfactual.py` 直接调用现有 VERL PPO loss 检查 clipping
  与 actor 梯度，没有重写 PPO。结果记录见 `results_reward_events.json`。
- `reward_readout.py` 使用原始 rollout token 前缀，对每个 action 和旧策略采样
  参照构建单位置反事实。正式 owner 的新增 categorical target 读出复用同一个
  Qwen head；其事件类别归一化与旧策略概率边缘分别计算，没有训练额外 head。
  `deltatrace_rollout.py` 已将这些比值送入既定逐事件 Q/V 组合，`METHOD=dt` 已开放
  实际联调。开放入口不代表训练已通过。
- 前缀使用 HF 原生缓存，分支复制防止污染事实轨迹；同长度事件采用原生 cache
  `reorder_cache` 批量查询，重复参考、相同端点和零奖励项避免重复读出。
  Actor 保留 SDPA 以使用现有可用反向；读出不把 actor 切换成前向专用 FA 构建。

这些测试使用 A6000 主机已有 Python/VERL 的 CPU 运算；没有安装依赖、下载数据、
清空缓存、占用 GPU 或运行模型。CPU 32k 张量检查不代表 32k 模型训练成功。
WebShop、Sokoban、AppWorld 当前固定方法均未取得新的训练／评估通过记录。

2026-09-22 的新增结果见 [results_reward_readout.json](results_reward_readout.json)。
40 项 CPU 测试通过，覆盖新增读出边界、真实 HF 混合 cache 分支与上游 padding
修复。真实 Qwen/DT 探针输出了非零类别 log ratio，并完成单 action 变化的有限传播
对照：目标差 0.07384、action 归因 0.06987；这记录实际数值残差，不声称零误差。
缓存与整段前向的最大 log-prob 差为 0.11676，仍需单独评估其影响。
探针在同一个 Sokoban 状态前缀上检查了三套奖励类别，实际上下文为 225--252 tokens，
不是三个任务的完整轨迹或 32k 满长训练。后续训练状态以对应日志验收为准。

已在上游 HF rollout/actor 加入显式启用的共有左 padding 裁剪：实际前向仅处理
非共有 padding 的列，rollout 返回时恢复原始宽度，token IDs、mask 和 response
位置不变。`VERL_TRIM_SHARED_PADDING=1` 由本训练入口启用；上游默认路径不变。
32k 是容量上限，不能强迫原本很短的任务每一步都计算 32k padding。

2026-09-21 新增接线验证见 [results_event_transport.json](results_event_transport.json)：
30 项测试通过，含真实 VERL collector、advantage 入口和 PPO loss；原始固定版本
补丁首装通过、二次执行无变化，现有远端重复执行也无变化。测试没有改动已安装的
Transformers，也没有重新配置环境。4 × 32768 的 collector 检查是 CPU 输入夹具，
不代表该长度的模型运行或显存验收。

`verify_task_reward_events.py` 另调用三个真实官方 task worker。Sokoban 的 seed-7
棋盘实际经历无效动作、普通移动和完成，奖励为 `[-0.1,-0.1,-0.1,-0.1,10.9]`，
首行采样 Q 为 10.5；WebShop 三步购买轨迹及 AppWorld 两步本地隔离环境均取得
实际终止事件，本次奖励为零。AppWorld 直接复用官方本地环境及 worker.step，未
触碰已有 HTTP 服务。**该脚本使用脚本动作和明确的比值测试夹具；只验证奖励及
Q/V 传递，不验证 DT 估计准确性、任务成功率或学习效果。** 总长度上限为 32768，
各步实际文本长度单独记录，不将脚本轨迹长度报告为模型真实 rollout 长度。

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
