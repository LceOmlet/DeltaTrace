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
后端。当前按用户指令只使用 MetaX，复用其已记录的动态 shape 执行配置和持久缓存。启动脚本按实际
tokenizer 为事件询问及 target 预留空间，总上限仍为 32768，不截断已生成 action。

## 当前验证范围

- 短链数值对拍已覆盖真实训练的共有 padding 裁剪路径。发现安装版 Qwen3.5
  在 batch=1 时跳过线性注意力 mask，回补 Transformers `59eed1a6` 的官方
  条件修复；Qwen 裁剪保留原 FLA 64-token 分块边界，最多留 63 个 padding。
  固定 VERL 原 actor、相同初值/输入/DT 优势/optimizer、两次更新，对照独立
  FP32 原生模型与 BF16 math 基线，按 FA 官方的 **2 倍基线最大误差** 判据
  通过。旧 log-prob、更新 log-prob、梯度误差倍数为 1.310、0.621、0.639，
  两次参数增量为 1.000、0.997；PPO loss/clipping 也通过。True/False 两种
  FSDP 设置都检查；没有把 FP32 配置用于训练。关闭裁剪的默认路径另有
  零容差对拍，不能拿该结果代替开启裁剪的验收。
  原先两条 BF16 路径差值的 1x 检查与严格零误差连续性失败均保留，未改写；
  它们不是 FA 的独立 FP32 判据。14 项 mask/裁剪接口检查通过。
  见 [results_short_owner_parity.json](results_short_owner_parity.json)。
- 修复前 v3 原生模型训练已记录：WebShop 两轮成功率 0/25%，第二轮 11 次
  DT、非零梯度；AppWorld 两轮 50%/0，首轮 26 次 DT、非零梯度；Sokoban
  首轮 100%、140 次 DT、非零梯度，第二轮因本次数值修复留存日志后主动停止。
  这些是小规模训练接线记录，不是测试集指标或修复后验收。新版本的三任务
  连续训练需单独记录；详见 [results_native_training_metax.json](results_native_training_metax.json)。
- 发现并修复 Qwen3.5-9B 聊天停止边界：checkpoint 没有 generation_config.json，
  原模型回退只用 `<|endoftext|>`（248044），漏掉 tokenizer 的 `<|im_end|>`
  （248046），导致模型在动作后继续编造后续轮次。固定 VERL worker 仅在这个
  模型的回退配置中加入聊天 EOS，保留显式配置，仍用 HF 原生成器与 VERL
  原 mask。11 项停止符/共有 padding 接口回归通过；修复后的真实训练另测。
  旧配置下 Sokoban 第一轮四条真实轨迹均成功、DT 优势非零且原 PPO grad_norm
  为 0.148；WebShop/AppWorld 第一轮实际奖励与梯度为零。三者旧日志均保留，
  不将零梯度当作学习，也不将旧停止配置结果冒充修复后的训练验收。
- 完整生成成本对照：原 HF worker 同一 32256-token 前文、batch=4，各生成
  512 token，总宽度 32768；FSDP reshard=True/False 的生成 IDs 完全一致。
  预热后 93.355/77.174 秒，allocated 40.317/54.308 GiB，reserved
  44.588/59.293 GiB。关闭每步 reshard 省时 17.3%，代价是保留参数占用。
  1024-action 容量也已独立验证：两个 DT 端点均为 32768，minibatch=4，
  四次正式 DT 加两次非零原 PPO 更新，allocated 60.596 GiB；True/False
  的 reserved 为 62.289/62.250 GiB、总耗时 748.51/743.78 秒。该输入含
  显式合成 action 和前文 filler，奖励只作容量检查系数，不算自然任务成功。
  详见 [results_runtime_efficiency.json](results_runtime_efficiency.json)。
- Ray CPU 配置从未被 trainer 读取的 `ray_kwargs.ray_init.num_cpus` 改为固定
  上游的 `ray_init.num_cpus`；真实 Hydra compose→原 run_ppo→ray.init 参数
  对照通过。没有新调度器；当前已经开始的 v3 作业保留原启动资源，后续生效。
- MetaX 按用户最新要求以物理 64 GB、不 OOM 验收。显式容量夹具把 DT
  两个端点各固定为 **32768 个有效 token**（actor 输入 32597，加事件询问
  170、目标 1），32769 被原读出接口拒绝。正式 DT 连续四次归因后，原 VERL
  minibatch=4 完成两次非零 PPO 更新，1,448,475 个可训练参数元素改变；
  peak allocated **60.627 GiB** / reserved **62.328 GiB**。
  夹具使用官方脚本动作的 14 个 action token 和显式长前文，不是自然任务
  长度或 512 个模型生成 token 的端到端吞吐。记录见
  [results_dt_context_capacity.json](results_dt_context_capacity.json)。
- FSDP2 归因按当前层使用公共 unshard/reshard，不再整模型驻留所有 gathered
  参数。相同 16384-token 输入、非零 LoRA 下，完整 signed 向量与 Q/V/A
  逐元素相等（零容差）；预热后中位耗时 40.521→40.698 秒（+0.44%），
  allocated 54.187→39.806 GiB。保留修复前 32k 的 OOM 与 reshard 接口
  失败记录；这不是用 offload 或改变有限公式换容量。50 项接口回归通过。
- 完整 chat collector 已保留历史，环境此前又在每轮重复插入旧历史与任务
  说明。现通过上游 manager 的 opt-in `full_chat_observations` 只发送新
  observation，WebShop 保留当前 admissible actions；初始 prompt 和默认
  路径与固定上游完全一致，4 项对照通过。DT 启动显式关闭上游默认熵正则
  和额外无效动作罚分，避免混入固定计划外的梯度或奖励。三任务正在以此
  配置重跑，任务结果另按真实 rollout 记录，容量夹具不代替任务验收。
- FSDP1 分层 LoRA 的参数恢复错误已定位到本地补丁：它把上游
  `use_orig_params=False` 无条件改成了 LoRA 下的 `True`。现对有 auto-wrap
  policy 的路径恢复上游设置。真实 Qwen 的旧 log-prob 和连续两次非零更新，
  在同卡、同初值及库的确定性选项下与上游参数配置逐元素一致（零容差）；
  20 项 owner 接口回归通过。32k 显式长输入夹具也完成两次非零更新，
  peak reserved 36.16 GiB。详见 [results_fsdp_owner.json](results_fsdp_owner.json)。
  这些是 actor 接入与容量检查，不代表三任务 DT 训练通过。
- MetaX 训练环境：固定 VERL 已安装并应用现有接口补丁，42 项奖励组合、
  collector、PPO 接口回归通过；另已在当地官方 Sokoban worker 验证真实
  `[-0.1,-0.1,-0.1,-0.1,10.9]` 奖励轨迹。WebShop 官方 1k 索引已构建，
  购买成功返回 10；AppWorld 官方数据和 20 个服务已就绪，原远程 worker
  执行官方训练答案后返回 10 并终止。上述均为明确标注的脚本/答案验证。
  这些是环境及接口结果，尚不表示完整 DT 训练通过。
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
  未通过。失败和逐层诊断保留在结果索引中；本修订的 DT→PPO 训练尚未验收。
- MetaX 三个任务的完整官方奖励→正式 DT→逐事件 Q/V 组合已实际执行：
  Sokoban 15 次 trace / 65 个非零 token 优势，WebShop 6 / 46，AppWorld 1 / 341。
  使用官方脚本/训练答案作接口验证，不能当作模型 rollout 或训练成功率。
  Q−V、mask 和有限值检查通过；原数值阈值仍有 6/15、3/6、1/1 项失败。
- 正式 runner 把原生舍入残差作为诊断，PLAN 允许数值估计。RL 适配层现将
  原 `0.02 * max(1, abs(root))` 检查结果传到日志和验收，而不阻断原始 DT
  向量进入既定 Q/V 组合。没有提高阈值、重缩放或替换向量。数值 verifier
  仍以失败状态退出；非有限值、标量端点/seed 不一致仍阻断执行。新增 3 项
  回归覆盖该区别，相关 38 项组合/collector/PPO/读出测试通过。

- 非零 LoRA 接口修复：原 PEFT `.weight` 只包含基础权重，正式 DT 现使用
  PEFT `get_delta_weight` 取得实际增量，分别通过已有线性转置，避免 BF16
  预合并抹去小更新。没有复制 LoRA 实现或更改 actor 参数。17 项线性/head
  回归通过，原生非零 LoRA 的 Sokoban 15 次 DT/Q/V 组合完成；数值审计
  仍失败 8/15，不能当作三任务训练或整网精度通过。

**当前数值修复的短链对拍已通过；修复后的三任务连续训练仍需完成。
精确 32k 容量夹具已通过；它不能替代真实任务训练与成功率评估。**

## 入口与历史结果

- `run_verl_agent.sh`：固定上游 VERL 训练入口，DT/PPO/GRPO 共用原训练基建。
- `verify_short_owner_parity.py` / `compare_short_owner_parity.py`：固定原 actor
  的真实短链对拍，独立 FP32 校准；只用于测试，不接管训练。
- `patch_verl_agent2.py`：固定上游的薄接口补丁与共有左 padding 裁剪；默认上游路径
  保持原行为。环境、rollout、optimizer 和 PPO clipping 都由上游实现。
- `verify_upstream_actor_update.py`：复用真实 worker 的非零更新回归，可复现旧
  FSDP1 参数配置、对照上游参数配置，并保存初始/每次更新后的可训练权重。
  明确使用测试优势与脚本动作；不产生 DT 估计，不作为任务训练数据。
- `results_env_smoke.json`：历史环境 readiness。
- `results_reward_events.json`、`results_event_transport.json`：奖励代数和传输历史检查。
- `results_reward_readout.json`：已撤下的逐前缀读出历史结果，非当前 EOS 方法结果。
- `results_counterfactual_32k_a6000.json`、其他旧归因/训练记录：只适用于各自历史
  实现，不能作为当前 PLAN 的验收。

实际累计长度、DT 调用次数、首轮/后续时间及显存必须来自新日志；不将配置上限
当作真实任务长度，不将所有非零 reward 项的多个 DT 调用报告为一次。
