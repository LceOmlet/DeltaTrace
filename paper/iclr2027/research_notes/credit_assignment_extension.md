# 反事实信用与策略学习

研究动机：精确平均的动作反事实信用与精确优势函数给出相同的学习信号。利用计算或环境中的结构估计贡献，可能提供一种有效的信用估计路径；方差以及无关时间延长下的稳定性是后续值得研究的性质。

固定历史 h 和后续策略 π。令 Gπ(h,a;ξ) 表示当前动作设为 a、以后依 π 行动的回报，ξ 包含环境与策略随机性。参考动作 a′ 从 π(·|h) 采样，条件于 h 独立于当前选中动作。两个反事实执行可共享随机性，各自保持正确的边际轨迹分布。对于可积回报，

$$C^\pi(h,a)=\mathbb E_{a',\xi}[G^\pi(h,a;\xi)-G^\pi(h,a';\xi)]=Q^\pi(h,a)-V^\pi(h).$$

把 C 或 Q 作为 score-function 更新的权重，则动作无关的 V 项在动作平均后为零：

$$\mathbb E_a[\nabla_\theta\log\pi_\theta(a|h)C^{\pi_\theta}(h,a)]=\mathbb E_a[\nabla_\theta\log\pi_\theta(a|h)Q^{\pi_\theta}(h,a)].$$

这是策略梯度的基线关系在反事实形式下的表达。正文在引言结尾提出这一连接，附录给出条件和直接证明。它为后续的奖励目标有限传播提供一个可检验目标：平均动作信用恢复优势信号。进一步研究包括结构化信用估计、参考与随机性耦合对方差的作用，以及在指定奖励、折扣和历史条件下对无关轨迹延长的稳定性。

当前 DeltaTrace 的归因目标是固定回答的 log-probability 差，其来源分数按算子规则分配联合输入干预的总效应。RL 延伸将目标转为奖励并引入动作参考与后续轨迹平均，形成独立明确的学习问题。

主要文献：[Sutton et al., 1999](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)；[Foerster et al., 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11794)。
