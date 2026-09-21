# DeltaTrace 替代 PPO 价值信号：唯一固定计划

固定日期：2026-09-21。依据：用户在当前任务中完整贴出并要求固定的最终方案。
本文件是本分支 RL 方法的唯一规范；实现、测试和其他说明必须对应本文件。
实现进展单独记录在 [README.md](README.md)，环境复用记录在
[REMOTE_ENVIRONMENT.md](REMOTE_ENVIRONMENT.md)。二者不另行定义方法。

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
