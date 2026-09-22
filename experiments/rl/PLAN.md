# DeltaTrace token 信用：唯一固定计划

更新：2026-09-22。依据用户最新确认，直接使用 EOS DT 的 log-prob 变化归因，
按奖励事件转换为 token 信用。取消逐 token 的生成前后奖励询问，以及额外参考
 token 采样。本文件是唯一方法规范；README 只记录实现状态，环境记录只规定复用。

## 前提与固定接口

保留用户的理想化假设：**DT 估计的 token 反事实价值完全准确时，就是对应
真实世界变量的反事实价值。** 实际有限传播允许估计误差，不要求新的环境
反事实模拟器，也不要求开发前达到零误差。组合规则、估计误差、接口错误、
资源效率分别核验；不因其中一项近似而撤回其他已经成立的部分。

每个 reasoning、工具调用、final token 都是独立 policy action。工具观测 O
属于状态，不是 action；O/padding 的 ratio 和 actor loss mask 为零。不把
O 的 token 链当作工具的自回归生成过程，不给调用 span 广播一个数，不做
O credit 路由，不引入 value model/value loss 或第二套 PPO。

## 奖励事件到 token 的计算

对生成 token i 和其后的奖励事件 k，y_k 表示足以确定该事件奖励的结果。
真实 reward r_k(y_k) 只由现有官方环境提供。DT 估计：

\[
d_{ik}(y_k)=\log p_{ik}(y_k)-\log p_{ik}^{\setminus i}(y_k),
\]

其中第二项是 EOS 删除 token i 的反事实事件概率。精确假设作用于这个
逐 token、逐事件的量；不会把一条轨迹归因向量的总和守恒当作逐 token
反事实准确性的证明。

实际观察到的 y_k 来自事实 rollout。令 w_ik 是与回报时钟一致的折扣，计算：

\[
\widehat Q_i=\sum_{k\in\mathcal F_i}w_{ik}r_k(y_k),\qquad
\widehat V_i=\sum_{k\in\mathcal F_i}w_{ik}r_k(y_k)e^{-d_{ik}(y_k)},
\]
\[
\boxed{\widehat A_i=\sum_{k\in\mathcal F_i}
 w_{ik}r_k(y_k)\bigl[-\operatorname{expm1}(-d_{ik}(y_k))\bigr]}.
\]

F_i 只含该 token 之后结算的事件；过去奖励不会分给后来的 token。
终局奖励是只有一个事件的特例。**先对各个事件做 expm1，再相加**；不能先
合并不同事件的 log-prob 归因再做指数，不能用 r*d 替代，也不能乘第二次
终局奖励、取绝对值、归一化全轨迹 credit 或悄悄裁剪指数。

在参考事件分布被事实事件分布覆盖时，精确 d 满足：

\[
\mathbb E_{y\sim p_{ik}}\left[r_k(y)(1-e^{-d_{ik}(y)})\right]
=\sum_y r_k(y)\left[p_{ik}(y)-p_{ik}^{\setminus i}(y)\right].
\]

外部 reward 是数值系数，无需对环境求导。零实际 reward 的样本项为零，
可省去该项 DT 计算；零类别仍保留在事件概率分布中。过程奖励和停止后的
事件缺席必须都在事件定义中，不能只条件化到“这一步一定会发生”。

## 与 PPO 的对应及等价范围

| 项目 | PPO | 本方案 |
| --- | --- | --- |
| 状态/action | token 前缀 / 生成 token | 相同 |
| Q | 条件未来累加回报 | 事实回报的采样估计 Q_hat |
| V | 当前状态的价值 | 由 DT 反事实比值得到的 V_hat，不训练 value model |
| Advantage | Q-V 或其采样估计 | A_hat=Q_hat-V_hat，稳定实现用 expm1 |
| 工具调用 | 价值含调用结果对后续奖励的影响 | 同一事件反事实定义包含该链路 |
| O | 状态，不参与 actor loss | 相同 |
| Actor loss | 原 token PPO ratio/clipping | 完全复用固定 VERL 的原实现 |

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

## 已批准的奖励事件目标与正式 DT 接入

用户此前批准的同模型事件读出继续作为目标编码：使用当前 Qwen 原 head 的
类别 logits，不新增参数或训练头。取消的只是每写一个 token 都重新询问
事件概率的路径。当前选择：

1. 复用官方 rollout 的原始 prompt、response IDs、attention mask、traj_uid、
   env_step 和实际每步 reward。原始 action 不经过 decode/encode 重建。
2. 每个当前 response 行、每个非零未来奖励事件，附加一次固定事件询问，
   指明任务、当前交互、目标交互、固定最大步数及类别定义。询问不能透露
   事实未来 O、未来 action、实际 reward 或实际停止时间。
3. 目标为实际观察到的奖励类别的单 token 标签；其概率由原 head 在完整
   事件类别上的 log-softmax 定义。实际类别只作为被评分目标，不放进它的
   因果前缀。目标编码是明确的事件概率估计，不将任意工具文本概率冒充 reward。
4. 事实端点为原 prompt + 当前完整 response + 相同询问/目标；参考端点仅
   将当前 response 的源 token 换成 EOS。prompt（含已发生的 O）、询问、
   目标和长度在两端相同。不额外采样 token，不逐 token 执行完整前向。
5. 调用正式 `PackedAnswerTargets` / `FiniteAnswerOps` / `runner.attribute`，
   一次有限归因返回当前 response 每个 token 各自的 signed log-prob 贡献。
   该向量是 DT 对理想逐 token 删除效应的实际估计，不是已经逐 token
   穷举测量的删除差。保留 signed 值，按原 action 索引接入前述采样公式。
6. 不同奖励事件的 signed 向量保持分离，直到完成各自的 expm1 变换。
   没有将多目标 seed 先合并后再非线性变换的捷径。

对调用之前/之内的 token，未来工具返回不作为固定事实输入读出；其影响由
该 token 对未来奖励事件的 DT 估计承载。对下一轮 token，已经收到的 O
包含在新行的原 prompt 中。没有遗漏未来事件，也没有另造 O→A 分账模块。
当前同 response 内固定后缀和联合 EOS 有限分解的近似质量需要单独测量；
其输出守恒、接口形状正确，均不能代替单 token 反事实精度检查。

一条轨迹的正式 DT 次数是“当前 response 行 × 非零未来事件”的有效组合数，
不是“生成 token 数 × 未来事件数”。这是当前实际调用边界；不能把多事件
情况提前报告为恰好一次 DT。后续优化必须保持逐事件与逐 token 语义。

## 三个任务的真实奖励

以下来自已安装固定上游 worker，不在适配层重新计算评分：

- WebShop：固定 VERL worker 将购买成功（原 task_score=1）映射为 10，其他为 0。
- AppWorld：固定 worker 在终止时用官方 `evaluate().success` 返回 10 或 0。
- Sokoban：当前 6×6、单箱、成功即终止。有效期内每次交互奖励 −0.1，
  解出时为 −0.1+1+10=10.9；未发生的未来交互记 0。当前类别是
  `{-0.1,0,10.9}`。不能把这个值域用于未经支持的多箱配置。

Sokoban 使用每个实际过程 reward，不遗漏步罚。不重复加终局成功值。
在 gamma=1 下，当前单箱配置的完整未来回报可写作 11*S-B_i-0.1*N；
这只是核对逐事件总和的恒等式，不新增成功概率/剩余步数预测器。
配置中固定 gamma=1；若以后更改折扣，须把同一回报时钟的折扣传入已有
组合接口，不能只改 trainer 参数而保留不折扣的 DT 回报。

## 实现与验收约束

- 正式 DT 负责有限传播，现有 VERL 负责 rollout、PPO clipping、optimizer，
  官方 task worker 负责 reward。禁止复制这些实现。
- 保持单卡、Qwen3.5-9B、minibatch 4、总长度上限 32768；GRPO 对照
  group size 4。smoke 同样保持 32768 上限，实际长度另报，不能用 padding
  冒充原始任务上下文。读出询问和 target 占用也计入上限。
- 2026-09-22 用户指定后续全部改用 MetaX，不再使用 A6000。先读环境记录并
  复用 Python/权重/持久缓存；先确认空闲显卡。平台变更不改变上述 Q/V、
  minibatch 或上下文约束。2026-09-22 用户进一步明确：MetaX 按物理 64 GB
  显存、不 OOM 验收，不再要求 48 GB；物理容量与实测占用仍分别报告。
  32k 容量须用已知长度的输入主动测试：DT 输入连同事件询问和目标恰好
  32768 tokens，并验证原始 PPO 更新；不能等待真实任务偶然接近上限。
  容量夹具与原始任务长度分开报告，越界必须明确拒绝而非静默截断。
  显存修复必须同时对比相同输入的预热后耗时，不能以严重拖慢训练换取通过。
  归因阶段使用正式 runner 要求的 FA，结束后恢复原 actor attention 后端，
  包括异常路径；不能让前向专用 FA wheel 接管 PPO backward。
- 复用已有动态 shape 执行配置，记录首轮/后续耗时和编译；不能清缓存假装恢复。
- 测试必须分清：数学组合、正式 DT 调用及单 token 对照、真实任务 reward、
  上游 actor 更新、32k 容量、完整训练/评估。历史结果不自动覆盖当前实现。
- 修复后还须在短链、相同权重/输入/随机种子下对拍正式 DT 和固定上游 PPO。
  比较归因、log-prob、梯度及参数更新；容差以同机 FA 加速的基准误差为依据，
  不因新实现出现偏差而放宽，不用 shape、总和或 loss 单项替代数值对照。
  短链可借鉴 FA 测试的独立 FP32 参考与低精度基线误差比较；不能把两条
  BF16 路径之间的差值当成 FP32 参考误差，也不能称为 FA 官方为 PPO 训练
  背书。整网误差界不能单独证明训练可靠；还检查实际概率比、更新方向和
  代表性长序列。高精度配置仅用于校验，不改变训练算法。
- 同步、提交时保留他人改动。遇到实际无法继续的问题，停止受影响部分并给出
  具体事实，等待用户处理；不得悄悄换用被否决的方法。
