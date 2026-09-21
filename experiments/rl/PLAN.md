# DeltaTrace 替代 PPO 价值信号：唯一固定计划

固定日期：2026-09-21。依据：用户在当前任务中完整贴出并要求固定的最终方案。
本文件是本分支 RL 方法的唯一规范；实现、测试和其他说明必须对应本文件。
实现进展单独记录在 [README.md](README.md)，环境复用记录在
[REMOTE_ENVIRONMENT.md](REMOTE_ENVIRONMENT.md)。二者不另行定义方法。

**2026-09-21 等价性复核：下面的完整回报 oracle 等价命题成立；仅凭现有文本
token attribution 精确，不能推出该命题的前提已经满足。本文定义了要实现的
目标，不代表从现有 DT 输出得到该目标的计算方法已经构造完成。**

## 前提

DT 能准确计算**完整环境反事实**，包括工具调用改变 \(O_t\) 后对后续生成和
最终奖励的影响。这是下面精确等价命题的前提，不是对现有代码完成情况的声明。

## 与 PPO 的逐项对应

| 项目 | 标准 PPO | 最终 DT 方案 |
| --- | --- | --- |
| 状态 | 每个 token 前缀 \(h_i\)，包含历史 \(O\) | 相同 |
| Action | 每个模型生成 token | 相同；reasoning、工具调用、final 都是独立 action |
| 工具观测 \(O\) | 状态，不是 action | 相同；不计算 ratio，不单独接收 credit |
| \(Q\) | \(Q^{\pi_{\rm old}}(h_i,a_i)\) | DT 计算同一个完整反事实回报 |
| \(V\) | critic 学习 \(V^{\pi_{\rm old}}(h_i)\) | DT 计算策略边缘的 \(V^{\pi_{\rm old}}(h_i)\)，不建 value model |
| Advantage | \(A_i=Q^{\pi_{\rm old}}(h_i,a_i)-V^{\pi_{\rm old}}(h_i)\) | \(A_i^{DT}=Q_i^{DT}-V_i^{DT}\) |
| 工具调用 token 的价值 | 包含 \(A_t\rightarrow O_t\rightarrow\) 后续回报 | 完整包含同一链路 |
| 后续 token | 状态中包含工具返回 \(O_t\)，独立计算 advantage | 相同 |
| Token credit | 每个 token 一个 \(A_i\) | 每个 token 一个 \(A_i^{DT}\)，不广播、不平均 |
| Reward | 通过 \(Q/V\) 进入 advantage | 通过 DT 的完整反事实回报进入 advantage |
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

在上述完整反事实及策略边缘计算准确时：

\[
Q_i^{DT}=Q^{\pi_{\rm old}}(h_i,a_i),\qquad
V_i^{DT}=V^{\pi_{\rm old}}(h_i),\qquad
A_i^{DT}=A_i^{\pi_{\rm old}}.
\]

因此，在同一策略、回报/折扣约定和 action mask 下，学习信号与精确
\(Q-V\) 的 token-level PPO 对应。工具返回 \(O\) 的影响包含在产生它的
调用 token 及后续 token 的 \(Q\) 中；\(O\) 自身不参与策略损失。

## 文本反事实精确究竟能推出什么

目标仍为上面的逐 token \(Q^{DT}-V^{DT}\)，不改变训练接口或增加替代模块。
此前说明把“文本反事实精确”直接提升成“完整环境回报及策略边缘计算精确”，
这一步没有推导依据，现予以纠正。

当前 DT 接入对同一个固定 target 文本比较两个输入端点的 log-prob，并返回
源 token attribution。真实环境 reward 是 rollout 已提供的；事实工具返回
\(O\) 也已经存在。缺少的不是这两项事实数据，而是从该文本量计算上面环境
条件期望及策略边缘的、已证明正确的转换。

一个直接反例：有限回合、无折扣，所有 action 和后续轨迹的终局 reward 都为 1。
不同输入仍可以改变 target 的文本概率，产生非零且完全精确的文本 attribution。
此时每个 token 的精确 \(Q=V=1\)，所以 \(A=0\)。如果把文本 attribution
按终局 reward 归一分配，得到的 credit 总和为 1，不能逐 token 等于全零的
advantage。**回报守恒不是 advantage 等价性的充分条件。**

准确的奖励投影关系可以直接由条件期望展开。令 \(z\) 为包含奖励所需信息的
后续结果，\(P\) 包含旧策略生成及真实工具转移，则离散情形下：

\[
A_i=\sum_z G_i(z)\left[
P(z\mid h_i,a_i)-
\sum_b\pi_{\rm old}(b\mid h_i)P(z\mid h_i,b)
\right].
\]

这不是新方案，而是原 \(Q-V\) 定义的展开，说明需要精确到哪个量。计算它不
要求 reward 可导；但一次事实回报乘上固定 target 的 log-prob attribution，
没有自动给出这个奖励加权的条件分布差。精确的 EOS/null 对照也不会自动变成
\(V\) 所要求的旧策略边缘对照。

工具部分同样不能只由 \(A_t\rightarrow O_t\) 的事实归属推断等价性：例如同一
前缀下两个单 token 调用各以 1/2 概率选择，事实调用回报为 1。另一个调用的
回报若为 0，事实调用的 advantage 为 1/2；若也为 1，advantage 为 0。
事实调用、事实 \(O\)、事实最终文本及回报可以完全相同。只处理这些事实文本的
DT 归因不会仅因文本计算精确，就区分这两种环境。这个例子不要求增加环境重跑
模块；它只说明因果归属本身没有确定精确 advantage 的数值。

因此，现阶段可以确认上游 token advantage 接口的计算位置，不能确认完整方法
已经可实现于“一次文本 DT”预算内。现有代码的归因缩放与位置映射不满足上述
证明要求，不得作为固定方法继续宣称等价或验收通过。

核查依据：[PPO 原论文](https://arxiv.org/abs/1707.06347)的式 (7)、(10)–(12)
分别定义 clipped objective 和 advantage 估计；使用精确 value 的采样估计
不应与逐样本精确 \(Q-V\) 混称。
[RUDDER 原论文](https://proceedings.neurips.cc/paper_files/paper/2019/file/16105fb9cc614fc29e1bda00dab60d41-Paper.pdf)
的 Theorem 1–3 区分回报等价与最优奖励重分配，后者有额外的条件期望要求。
这里仅核对性质，不引入该论文的预测模型或训练模块。

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
