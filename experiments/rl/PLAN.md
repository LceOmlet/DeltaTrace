# DeltaTrace 替代 PPO 价值信号：唯一固定计划

固定日期：2026-09-21。依据：用户在当前任务中完整贴出并要求固定的最终方案。
本文件是本分支 RL 方法的唯一规范；实现、测试和其他说明必须对应本文件。
实现进展单独记录在 [README.md](README.md)，环境复用记录在
[REMOTE_ENVIRONMENT.md](REMOTE_ENVIRONMENT.md)。二者不另行定义方法。

**按用户最新澄清固定：理想条件下，DT 估计的 token 价值就是该 token 对应的
真实世界变量价值。以此检查组合规则的正确性；实际实现允许 DT 估计、数值计算和
策略参照估计存在近似。近似误差不自动否定组合规则，也不是启动开发的禁止条件。**

## 前提

只以 token 价值作为 DT 与策略学习之间的对象。用户给定的理想化假设是：
**DT 的 token 反事实就是对应真实世界变量的精确反事实；模型估计的 token
价值完全准确，并等于对应真实世界变量的价值。**
推导采用这一前提，不重新质疑它，也不将其改成新增环境反事实模拟器或外部逐
token 打分器的要求。它用于检查精确极限下的组合；不声称现实估计已经没有误差。

下面的 \(Q/V\) 规定这些价值相对于既定奖励目标、token 前缀及旧策略参照的含义。
实际输出仍须与这个含义对齐；既不能把任意张量改名当作完成，也不能因估计有误差
就撤回已经成立的组合规则。

## 与 PPO 的逐项对应

| 项目 | 标准 PPO | 最终 DT 方案 |
| --- | --- | --- |
| 状态 | 每个 token 前缀 \(h_i\)，包含历史 \(O\) | 相同 |
| Action | 每个模型生成 token | 相同；reasoning、工具调用、final 都是独立 action |
| 工具观测 \(O\) | 状态，不是 action | 相同；不计算 ratio，不单独接收 credit |
| \(Q\) | \(Q^{\pi_{\rm old}}(h_i,a_i)\) | DT 提供该 token 的对应价值估计 |
| \(V\) | critic 学习 \(V^{\pi_{\rm old}}(h_i)\) | DT 计算策略边缘的 \(V^{\pi_{\rm old}}(h_i)\)，不建 value model |
| Advantage | \(A_i=Q^{\pi_{\rm old}}(h_i,a_i)-V^{\pi_{\rm old}}(h_i)\) | \(A_i^{DT}=Q_i^{DT}-V_i^{DT}\) |
| 工具调用 token 的价值 | 包含 \(A_t\rightarrow O_t\rightarrow\) 后续回报 | 完整包含同一链路 |
| 后续 token | 状态中包含工具返回 \(O_t\)，独立计算 advantage | 相同 |
| Token credit | 每个 token 一个 \(A_i\) | 每个 token 一个 \(A_i^{DT}\)，不广播、不平均 |
| Reward | 通过 \(Q/V\) 进入 advantage | 通过 DT 的 token 价值及既定 Q/V 关系进入 advantage |
| PPO ratio | \(\rho_i=\pi_\theta/\pi_{\rm old}\) | 完全相同 |
| PPO clipping | 原始 clipped objective | 完全相同 |
| Value loss | 通常训练 critic | 没有额外 value loss |
| 训练目标 | \(\min(\rho_i A_i,\operatorname{clip}(\rho_i)A_i)\) | 同一个目标，只替换 \(A_i\) 的来源 |

## 固定的 Q、V 和目标函数

令 \(G_i\) 为与 PPO 对照使用相同回报及折扣约定的环境 return-to-go，
\(\mathcal E\) 表示真实环境转移。DT 要计算的对象是：

\[
Q_i^{DT}
=\mathbb E_{\pi_{\rm old},\mathcal E}[G_i\mid h_i,a_i],
\qquad
V_i^{DT}
=\mathbb E_{a\sim\pi_{\rm old}(\cdot\mid h_i)}
   [Q^{DT}(h_i,a)],
\qquad
A_i^{DT}=Q_i^{DT}-V_i^{DT}.
\]

这里的策略边缘只定义 \(V\)；它不把同一调用 span 内的 token advantage 平均。
每个调用 token 仍使用自己的前缀 \(h_i\) 和 action \(a_i\)。

\[
\rho_i=\frac{\pi_\theta(a_i\mid h_i)}{\pi_{\rm old}(a_i\mid h_i)},
\qquad
\mathcal L=-\sum_{i\in\text{所有模型生成 token}}
\min\left[
\rho_i A_i^{DT},
\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i^{DT}
\right].
\]

在给定的 token 价值准确、并按上述旧策略参照正确组合时：

\[
Q_i^{DT}=Q^{\pi_{\rm old}}(h_i,a_i),\qquad
V_i^{DT}=V^{\pi_{\rm old}}(h_i),\qquad
A_i^{DT}=A_i^{\pi_{\rm old}}.
\]

因此，在同一策略、回报/折扣约定和 action mask 下，学习信号与精确
\(Q-V\) 的 token-level PPO 对应。工具返回 \(O\) 的影响包含在产生它的
调用 token 及后续 token 的 \(Q\) 中；\(O\) 自身不参与策略损失。

## 固定部分与近似部分

| 部分 | 固定的含义 | 实现和验证范围 |
| --- | --- | --- |
| Token 价值 | 理想条件下与对应真实变量价值相同 | 现实 DT 估计允许误差；单独衡量估计质量 |
| Q/V 组合 | 同一前缀、同一奖励目标、同一旧策略参照，相减得到 advantage | 组合正确性与上游张量接口分别检查，不以模型零误差作为验收前提 |
| 策略参照 | 旧策略的条件边缘 | 数值或采样近似应保持这个目标；不能无依据换成 EOS/null、跨状态组均值 |
| 工具与观测 | O 是状态；调用及后续 token 的价值保留其影响 | 使用真实工具事件和准确 token 身份，不凭最近位置推断归属 |
| Actor 更新 | 所有生成 token 的原始 PPO ratio、mask、clipping | 复用上游实现；不用近似估计为接口错位、漏 token 或广播开脱 |
| 资源 | 已有环境、48 GB 单卡、minibatch 4、32k 总长度上限 | 直接在该配置检查实际长度、显存、重复计算和编译，保留可复用结果 |

“模型近似”和“实现错误”分开处理。一个近似模块误差较大，只更新该模块的结论；
不据此否定已验证的 mask、Q/V 代数组合或上游损失。反过来，接口跑通不证明
估计已经准确，也不证明训练收益。

在同一前缀下，将真实 token 动作价值记为 \(q(a)\)，DT 估计记为
\(\widehat q(a)=q(a)+e(a)\)。按既定旧策略组合：

\[
\widehat A(a)=\widehat q(a)-\mathbb E_{\pi_{\rm old}}\widehat q,
\qquad
\widehat A(a)-A(a)=e(a)-\mathbb E_{\pi_{\rm old}}e.
\]

这是固定 Q/V 规则的代数性质，不增加一个估计模块：token 价值误差为零时，
组合误差也为零；实际 token 价值存在误差时，组合仍有明确含义。
若策略边缘的数值近似另有误差，应单独记录，不把它扩大成整个方案无效。

同一个 Q/V 组合也可以直接使用 DT 的反事实价值差。对固定前缀，若 DT 以同一个
参考值 \(B_i\) 给出各候选 token 的差 \(d_i(a)=q_i(a)-B_i\)，则：

\[
d_i(a_i)-\mathbb E_{a\sim\pi_{\rm old}}d_i(a)
=q_i(a_i)-\mathbb E_{a\sim\pi_{\rm old}}q_i(a)
=Q_i-V_i.
\]

参考值在相减中消去，不需要额外学习或求出它。这是已有 Q/V 规则的等价计算，
不是另一种算法。所有差值必须对应同一前缀、同一价值定义和同一个参考；不能把
跨前缀的值混在一起，也不能将某个参考 token 本身直接视为策略平均价值。
该恒等式说明如何使用精确 DT 反事实；当前具体输出与这些量的对应仍需据正式
接口核对，不能仅靠变量命名认定。数值近似只近似这些既定量，不改其含义。

## 每次推进的规则

1. 不再质疑用户给定的精确 token 反事实假设；不擅自增加“每次只能改一个部分”等
   工作约束。保留已经确定的结论，除非相关证据或用户指令确实要求修正。
2. 每个改动说明它改变了哪个计算、为何符合既定语义，以及准确性或成本上要改善什么。
   尚未证明的选择保持为待核对项，不因命名相近就固定成实现。
3. 不以穷举所有未来、真实模型零误差或重新训练一个完美环境模型作为开发门槛。
   有限采样与数值近似按实际预算处理，并保留它们所估计的原目标。
4. 一个反例只否定它实际触及的计算。常数奖励反例排除了“把守恒份额直接当作
   advantage”，没有否定正确构造 Q、V 后相减的计划。
5. 不将 HCA、COCOA、奖励事件概率预测或其他讨论过的候选改写自动升级为本计划。
   不把用户对 Gibbs 采样逻辑的类比扩写成新的算法或强制工作顺序。
6. 遇到真实的接口冲突或无法继续的实际错误时，停止受影响部分并报告具体事实；
   “DT 是近似”本身不构成这样的错误。

当前代码中已有的归因缩放与位置映射仍需对照固定计划修正。该项属于具体计算的
语义核对；不能用它推翻用户给定的 token 价值假设，也不能用假设替它证明正确。

## 实现必须守住的范围

- 复用正式 DeltaTrace 和固定上游 RL/任务接口。rollout、环境、策略损失、
  optimizer 等已有能力由其原有实现负责，不复制或另写影子实现。
- 只替换 token advantage 的来源；不增加 value model、value loss、span credit、
  macro-action、独立 O credit 或独立奖励分账模块。
- 不将旧实现、历史通过记录或张量守恒检查当作上述 \(Q/V\) 构造及等价性的证明。
- 发现实现偏离本文件时，回到本文件修正；遇到实际卡点时停止相关实现并向用户报告，
  等待处理，不自行换用备选方法或改写本计划。

## 已确定的实验范围

使用已有 A6000 环境及 Qwen3.5-9B 权重，复用 WebShop、Sokoban、AppWorld
三个官方任务环境，测试多轮工具调用。单卡 48 GB、minibatch 4、总上下文上限
32,768；GRPO 对照的 group size 为 4。smoke 也保持该总长度上限。

实际累计长度包含初始输入、历史 reasoning、工具调用、工具返回和 final；
必须报告实际长度，不能把配置上限或人工填充量报告为任务原生长度。
完成标准包含当前固定方法的语义验证、三个任务的训练验证，以及实测速度、
显存和重复计算/编译情况。历史其他方法的结果不替代这些验证。
