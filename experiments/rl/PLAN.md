# DeltaTrace token 信用：唯一固定计划

更新：2026-09-29。用户明确恢复固定 VERL 官方损失默认设置（entropy=0.001、
dual-clip=3），DT Q/V/A 不变。依据用户既定约束，禁止 O(n²) 的归因调用。直接使用 EOS DT
对完整未来累计回报结果的 log-prob 变化归因，每个当前 response 至多一个请求，
不再展开 response × 未来奖励事件。不逐 token 询问，不额外采样参考 token。
本文件是唯一方法规范；README 只记录实现状态，环境记录只规定复用。

**2026-09-30 当前范围：按用户最新指令，仅 SkyRL-SQL、TextCraft、AppWorld 三组 DTPO，
每组两卡，只使用 GPU 0–5；GPU 6/7 的 SQL GRPO 对照组停止，不再提交。
只复用任务作者的环境、数据、采样/评测接口；不引入 SkyRL、
AgentGym-RL 或 LOOP 的训练器及推理服务。AppWorld 调用 LOOP f14107a 的训练基准
环境，不混入历史 VERL AppWorld 奖励。训练由已验证的 VERL-agent / vLLM 承接；
数值版本以 verified_runtime.json 的官方对照回执、哈希及运行开关为准。
以下 DT Q/V/A 不变。环境比较、启动、容量检查分别报告，不能代替训练验收。
Sokoban、WebShop、SQL GRPO 不属于当前三组。**

**同日最新用户指令优先：本文件只固定信用分配；信用以外使用官方实现与配置，
不得用旧 PLAN 阻止恢复官方行为。** 原 PPO loss、无效动作惩罚、任务历史窗口、
采样与更新由固定 VERL-agent 的原代码负责。训练所需的机型、模型、内存兼容
改动须逐项记录，不能称为原论文同配置。DT 容差复用对应 FA/FLA 原参考和断言，
不把整网守恒、短链逐值一致或自己设置的倍率变成官方验收标准。

## 前提与固定接口

保留用户的理想化假设：**DT 估计的 token 反事实价值完全准确时，就是对应
真实世界变量的反事实价值。** 实际有限传播允许估计误差，不要求新的环境
反事实模拟器，也不要求开发前达到零误差。组合规则、估计误差、接口错误、
资源效率分别核验；不因其中一项近似而撤回其他已经成立的部分。

每个 reasoning、工具调用、final token 都是独立 policy action。工具观测 O
属于状态，不是 action；O/padding 的 ratio 和 actor loss mask 为零。不把
O 的 token 链当作工具的自回归生成过程，不给调用 span 广播一个数，不做
O credit 路由，不引入 value model/value loss 或第二套 PPO。

## 完整未来回报到 token 的计算

官方环境给出原始每步 reward；当前官方 PPO 配置 gamma=1。对第 t 个 response，
先取从本次 action 执行到实际结束的环境 reward 之和。DT 在原 trainer 奖励处理后
读取其 `token_level_rewards`，加上该 response 的原生训练分数与原 episode score
之差，得到 G_t。这使官方当前无效动作惩罚计入一次；不复制惩罚公式，也不额外
累计过去或未来 response 的惩罚。原 EpisodeRewardManager 将 episode score
放在每条 response 的最后一个有效 token，原 `apply_invalid_action_penalty`
负责调整它。历史环境 reward 不计入 G_t，
终止后没有新 reward，预算终止不补造终局 reward。整段 response 的 token 在
同一次环境结算前，观察到同一个 G_t，但各有独立的反事实比值。DT 估计：

\[
d_i(g)=\log p_i(G_t=g)-\log p_i^{\setminus i}(G_t=g),
\]

其中第二项是 EOS 删除 token i 后完整未来回报的反事实概率。精确假设作用于
这个逐 token 的量；不会把一条轨迹归因向量的总和守恒当作逐 token
反事实准确性的证明。

实际观察到的 g=G_t 来自事实 rollout，计算：

\[
\widehat Q_i=g,\qquad \widehat V_i=g e^{-d_i(g)},
\]
\[
\boxed{\widehat A_i=g\bigl[-\operatorname{expm1}(-d_i(g))\bigr]}.
\]

这里读取的是累计回报的分布，**不是把旧的各步 log-prob 归因相加后取指数**。
同一回报可以来自不同未来轨迹，其概率包含这些轨迹的总概率；不假设各步奖励
独立。不能用 g*d 替代，不能再次乘 reward、取绝对值、归一化或裁剪指数。

在参考回报分布被事实回报分布覆盖时，精确 d 满足：

\[
\mathbb E_{g\sim p_i}\left[g(1-e^{-d_i(g)})\right]
=\sum_g g\left[p_i(g)-p_i^{\setminus i}(g)\right].
\]

外部累计 reward 是数值系数，无需对环境求导。实际 G_t=0 的样本项为零，
可省去 DT；零类别仍保留在回报分布中。过程奖励、提前停止和预算终止均影响
G_t，不能只预测成功奖励或丢弃步罚。该组合与 return-conditional HCA 的
采样形式对应（[原文定理2、式6](https://papers.neurips.cc/paper/9413-hindsight-credit-assignment.pdf)）。
全回报读出与逐事件读出在精确分布下有相同的期望差，不宣称单样本结果或方差相同。

## 与 PPO 的对应及等价范围

| 项目 | PPO | 本方案 |
| --- | --- | --- |
| 状态/action | token 前缀 / 生成 token | 相同 |
| Q | 条件未来累加回报 | 事实回报的采样估计 Q_hat |
| V | 当前状态的价值 | 由 DT 反事实比值得到的 V_hat，不训练 value model |
| Advantage | Q-V 或其采样估计 | A_hat=Q_hat-V_hat，稳定实现用 expm1 |
| 工具调用 | 价值含调用结果对后续奖励的影响 | 同一事件反事实定义包含该链路 |
| O | 状态，不参与 actor loss | 相同 |
| Actor loss | 原 token PPO ratio/clipping、原熵项 | 完全复用固定 VERL 的原实现与默认设置 |

这里的 A_hat（也记作 C_i^{DT}）已经汇总该 token 之后的奖励事件，直接进入
它自己的 PPO ratio 和逐 token clipping。**不再对 A_hat 做 GAE、第二次
return-to-go 累加或 span 平均。** O/padding 仍不参与策略损失。

保留既定精确 PPO 对照目标：

\[
Q^\pi(h_i,a_i)=\mathbb E[G_i\mid h_i,a_i],\qquad
V^\pi(h_i)=\mathbb E[G_i\mid h_i],\qquad A_i^\pi=Q^\pi-V^\pi.
\]

当 DT 提供的事实和反事实 token 价值分别对应这里的 Q 和 V 时，上述采样
组合在条件期望上等于精确 Q-V；相同 advantage 进入相同 PPO loss，给出
相同更新信号。**单条蒙特卡洛样本不被宣称为精确期望值；无偏 advantage
也不自动证明有限采样的 clipped loss 无偏。**

EOS 删除端点首先定义删除反事实。仅“删除反事实精确”不能额外证明该端点
天然等于 PPO 的动作边缘 V；两者的价值语义必须对齐，才使用 PPO 等价结论。
这项范围说明不引入额外参考采样、逐前缀预测或替代算法；实现保留用户选择的
EOS DT 和上述 Q/V 数值接口，验证中分别记录组合正确性与 DT 估计语义。

**同一期望策略梯度不要求删除参照等于 PPO 的 V。** 若同一前缀的统一删除
参照价值 B(h_i) 不依赖实际选中的 token，则精确 DT 组合的条件期望为
Q(h_i,a_i)-B(h_i)。因为 E_pi[grad log pi(a|h_i) B(h_i)]=0，它与减去
精确 V(h_i) 的期望 actor 梯度相同。这是当前 EOS 方法的基线消去性质，
不是另增一个 B 模块，也不需要额外计算或采样策略参照。等价范围是采样策略
处、相同状态/token 权重下的期望梯度；不能扩大成所有 clipping 更新相同。
不能固定由实际 token 导致的事实后缀，再把该删除参照宣称为动作无关。
实际 DT 分解对这个理想干预的近似质量仍单独核验。

## 回报结果目标与正式 DT 接入

用户此前批准的同模型事件读出继续作为目标编码：使用当前 Qwen 原 head 的
类别 logits，不新增参数或训练头。取消的只是每写一个 token 都重新询问
事件概率的路径。当前选择：

1. 复用官方 rollout 的原始 prompt、response IDs、attention mask、traj_uid、
   env_step 和实际每步 reward；在官方 reward/penalty 之后读取训练奖励。
   原始 action 不经过 decode/encode 重建。原 `adjust_batch` 可能复制行，
   balance 可能改变顺序；以原 traj_uid/env_step 对齐，每个实际 response
   归因一次，再按原训练行顺序返回，不把复制行当成新环境奖励。
2. 每个当前 response 行至多附加一次完整未来回报询问，
   指明任务、当前交互、固定最大步数及回报类别定义。询问不能透露
   事实未来 O、未来 action、实际 reward 或实际停止时间。
3. 目标为实际观察到的累计回报类别的单 token 标签；其概率由原 head 在完整
   回报类别上的 log-softmax 定义。实际类别只作为被评分目标，不放进它的
   因果前缀。目标编码是明确的事件概率估计，不将任意工具文本概率冒充 reward。
4. 事实端点为原 prompt + 当前完整 response + 相同询问/目标；参考端点仅
   将当前 response 的源 token 换成 EOS。prompt（含已发生的 O）、询问、
   目标和长度在两端相同。不额外采样 token，不逐 token 执行完整前向。
5. 调用正式 `PackedAnswerTargets` / `FiniteAnswerOps` / `runner.attribute`，
   一次有限归因返回当前 response 每个 token 各自的 signed log-prob 贡献。
   该向量是 DT 对理想逐 token 删除效应的实际估计，不是已经逐 token
   穷举测量的删除差。保留 signed 值，按原 action 索引接入前述采样公式。
6. 每行只有一个回报目标、一个 signed token 向量，调用既有 Q/V/A 稳定组合
   接口的单事件维度。没有逐未来事件矩阵，没有多目标 seed 合并。

对调用之前/之内的 token，未来工具返回不作为固定事实输入读出；其影响由
该 token 对未来奖励事件的 DT 估计承载。对下一轮 token，已经收到的 O
包含在新行的原 prompt 中。没有遗漏未来事件，也没有另造 O→A 分账模块。
当前同 response 内固定后缀和联合 EOS 有限分解的近似质量需要单独测量；
其输出守恒、接口形状正确，均不能代替单 token 反事实精度检查。

一条 n 步轨迹至多 n 个归因请求，全局按 minibatch4 打包；15步至多15个请求，
单独打包为4次 runner 调用，禁止平方展开。回报通过一次反向累计求和取得，
不在每行扫描整个未来。每次调用仍产生各 token 独立的 signed 值，不能广播
一个 span advantage。调用数线性不代表总 FLOPs 线性：单次上下文变长仍增加成本。

## 三个任务的真实奖励

以下环境奖励来自固定上游 worker，不在适配层重新计算评分：

- WebShop：固定 VERL worker 将购买成功（原 task_score=1）映射为 10，其他为 0。
- AppWorld：按用户后来选定的 LOOP 训练基准，直接读取原生 rollout 的 `ret`，
  即终止时的官方测试通过比例；类别分母读取原 `eval_result.num_tests`。
  不再使用已退役 VERL AppWorld worker 的 0/10。独立评估仍保留 LOOP 的
  原生稀疏成功评分，不能用评估分数覆盖训练回报。
- Sokoban：当前 6×6、单箱、成功即终止。有效期内每次交互奖励 −0.1，
  解出时为 −0.1+1+10=10.9；未发生的未来交互记 0。官方每步值域是
  `{-0.1,0,10.9}`。15步内的累计回报类别为0、`-0.1*N`、`11-0.1*N`
  （N=1..15），共31类；使用互异的单token标签，复用原Qwen head。
  不能把这个值域用于未经支持的多箱配置。

固定 VERL 默认额外对**当前 response**的无效动作扣 0.1。训练目标类别由上述
累计环境回报集合与其减去 0.1 的集合取并集；WebShop 为4类，
15步 Sokoban 为33类。惩罚数值直接读取官方处理结果，类别编码只说明目标
分布的支持集，不自行判定 action 有效性。未启用官方惩罚的独立旧夹具仍使用
原2/31类，不能拿该夹具声称新接口已经通过。LOOP AppWorld 原生配置的
两项动作/执行失败惩罚均为0，不混入 VERL 的无效动作惩罚；其训练目标类别为
`k/num_tests`（k=0..num_tests）。这只是官方结果编码，Q/V/A 组合不变。

Sokoban 使用每个实际过程 reward，不遗漏步罚。不重复加终局成功值。
在 gamma=1 下，当前单箱配置的完整未来回报可写作 11*S-0.1*N（尚未终止的
源response之后只可能解出一次）。实际 G_t 始终取官方 rewards 的累计值，
公式仅用于完整类别编码，不新增成功概率/剩余步数预测器。
配置固定 gamma=1；若以后更改折扣，须同时更新回报时钟与类别编码，
不能只改 trainer 参数而保留不折扣的 DT 回报。

## 实现与验收约束

- 2026-09-29 用户澄清：存在数值残差本身正常，主要判断是否超出对应官方容差。
  整网守恒残差只作定位诊断，不另设归零或百分比门槛；已通过的算子不因
  此诊断再加补偿。实际超差、非有限值与普通低精度误差分别记录处理。
- 2026-09-29 用户再次要求：DT 有数值问题时，补齐保存的实际操作数 dtype
  对照并定位残差来源；记录 q/k/v、beta、g、缓存状态、上游系数及参考的
  dtype，不能用不同 dtype 的随机测试冒充实际路径通过。FA/FLA 原参考和
  原容差保持不变，不添加强制守恒缩放、裁剪或其他“纠偏”来让结果通过。
- 2026-09-29 用户要求：PPO 除信用来源外，须对齐同一个强化学习框架的官方
  行为，并采用该框架对应测试原有的容差。此处 owner 仍为固定 VERL-agent，
  不用 LOOP 的轨迹单位、预算解释或测试冒充 VERL 行为。对照传入完全相同的
  token IDs、mask、advantages、初始模型/optimizer 状态及配置，只检查信用
  接口之后的原 PPO 路径。原测试未覆盖的量须明确说明，不能自创或放宽容差，
  也不能把 FA/FLA 算子阈值扩大成整条 PPO 阈值。以下 FA/FLA 要求继续只用于
  各自对应的算子；本条不改变上面的 EOS DT Q/V/A、reward 或 token mask。
- 同日用户明确选择“恢复 VERL 官方默认损失设置”：删除启动器对
  `entropy_coeff=0`、`clip_ratio_c=inf` 的覆盖，继承固定 owner 的
  `entropy_coeff=0.001`、`clip_ratio_c=3.0`。PPO 对照与 DT 使用相同设置。
  熵项不是 DT 信用；报告更新时分别保留原 actor 的 policy/entropy 指标，
  不能将仅由熵项造成的参数变化称为非零 DT 更新。环境 reward 与 Q/V/A
  继承官方奖励处理，包括原生无效动作惩罚；由上述信用接口读取结果。
- 正式 DT 负责有限传播，现有 VERL 负责 rollout、PPO clipping、optimizer，
  官方 task worker 负责 reward。禁止复制这些实现。
- 2026-09-30 用户固定四组 LoRA rank=8、alpha=16，不得更改。旧rank=1/alpha=2
  的容量记录不代表当前配置通过；资源调整只能保留该LoRA设置后处理。
- 2026-09-29 最新资源安排：三个 DT 任务各用两张卡，剩余两张卡运行 GRPO
  对照。Qwen3.5-9B、总长度上限32768；实际 minibatch 通过官方分布式配置安排，
  初始每卡 actor microbatch4/DT batch4。信用公式不依赖单卡假设。原单卡
  验收不能冒充双卡请求分发、FSDP collective、奖励/优势对齐已通过。
  PPO optimizer minibatch 恢复固定作者任务脚本的64，由原更新循环累计梯度，
  撤销旧 PLAN 对 optimizer minibatch4 的覆盖；GRPO 对照
  group size 4。smoke 同样保持 32768 上限，实际长度另报，不能用 padding
  冒充原始任务上下文。读出询问和 target 占用也计入上限。
- 2026-09-23 用户明确要求 DT 本身使用 minibatch，效率应接近一次反向或更快。
  调用正式 runner 的 batch 接口，保留每个样本/奖励事件的独立归因。
  用同模型、同卡、同长度、同 batch 的原生反向作为实测参照；分别报告完整
  DT（含端点前向、重算、传播）和各阶段成本，不能用 PPO 更新耗时代替纯反向。
  没有完成该对照、32k 容量和数值校验前，不报告效率要求已达标。
  加速优先参考已验证的官方反向设置及执行路径，包括卸载、重算、保存张量、
  FA/FLA 内核与批处理；先比较已暴露的主要阶段，尽早确认 DT 多出的计算和
  搬运是否必要。原版可行反向保持为主基准，不用新的较慢变体替换它，也不
  为统一内部卸载方式而削弱官方路径；确需不同策略时记录实际成本与原因。
- 2026-09-22 用户指定后续全部改用 MetaX，不再使用 A6000。先读环境记录并
  复用 Python/权重/持久缓存；先确认空闲显卡。平台变更不改变上述 Q/V、
  信用或上下文约束。2026-09-22 用户进一步明确：MetaX 按物理 64 GB
  显存、不 OOM 验收，不再要求 48 GB；物理容量与实测占用仍分别报告。
  32k 容量须用已知长度的输入主动测试：DT 输入连同事件询问和目标恰好
  32768 tokens，并验证原始 PPO 更新；不能等待真实任务偶然接近上限。
  容量夹具与原始任务长度分开报告，越界必须明确拒绝而非静默截断。
  2026-09-24 用户批准 AppWorld 的轨迹预算处理：下一轮 prompt 无法在预留
  response 和 DT 读出后放入上限时，结束该条轨迹，保留已经执行的 action
  和官方 reward，不再生成或执行工具，不补造终局 reward。其他轨迹继续。
  2026-09-29 最新指令恢复官方 AppWorld history_length=2、10000字符历史窗口；
  撤销旧全历史覆盖。预算终止保护只在官方渲染后的输入仍超限时生效，保留
  已执行动作及官方奖励，不补造奖励；该32k边界适配不冒充作者默认配置。
  显存修复必须同时对比相同输入的预热后耗时，不能以严重拖慢训练换取通过。
  归因阶段使用正式 runner 要求的 FA，结束后恢复原 actor attention 后端，
  包括异常路径；不能让前向专用 FA wheel 接管 PPO backward。
- 复用已有动态 shape 执行配置，记录首轮/后续耗时和编译；不能清缓存假装恢复。
- 测试必须分清：数学组合、正式 DT 调用及单 token 对照、真实任务 reward、
  上游 actor 更新、32k 容量、完整训练/评估。历史结果不自动覆盖当前实现。
- 相同权重/输入/随机种子下的 PPO 短链保留为对照诊断；不再用 FA/FLA
  算子误差替整网梯度或多步更新设阈值。信用以外调用同一框架原代码与原测试，
  原测试未覆盖的量如实记录；既不据差值自动判错，也不据差值小就宣称通过。
  用户进一步确认：数值容差直接采用固定版本 FA/FLA 官方测试的计算方式，
  比较同一初值下单步前向/反向；允许与这些低精度实现相同的多步累计误差，
  不另设多步逐位相同、更新方向或 clipping 分支完全一致的验收门槛。
  FA 2.6.3 `test_flash_attn_output`/`test_flash_attn_varlen_output` 比较同一
  Q/K/V 的 FP32 参考与普通低精度基线，输出最大绝对误差为基线的 2 倍、
  dQ/dK/dV 为 3 倍，不能笼统写成所有项目 2 倍；其他测试的例外须按原函数。
  FLA 0.4.1 `test_chunk` 调用原 `assert_close`：误差 RMS / (参考 RMS+1e-8)，
  o/ht 阈值 0.005，dq/dk/dv/dh0 为 0.008，db/dg 为 0.02；绝对误差 <=1e-6
  直接通过。不启用 `FLA_CI_ENV` 的 warning 豁免冒充严格通过。上述断言在
  各自对应的算子量上复用，不把权重梯度任意改名成 dq，不虚构官方 PPO 阈值。
  已有多步结果保留为诊断，三任务仍须实际连续运行且无 OOM/非有限值。
- 同步、提交时保留他人改动。遇到实际无法继续的问题，停止受影响部分并给出
  具体事实，等待用户处理；不得悄悄换用被否决的方法。
