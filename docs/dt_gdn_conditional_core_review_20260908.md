# GDN 条件误差与计算核心审查

日期：2026-09-08。只读代码与数学审查；未运行 GPU、未修改 FT、DT 核心或评测。本文中的定位和候选条件不是已测改进，NI1/MH1 的真实粗边界定位由独立实验执行。

## 审查结论与证据边界

当前没有发现可以实证的 GDN 路由、写入或擦除漏项。已经能够严格定义的问题是：固定 EOS/输入两端点的有限算子保证两端点差，却不保证预测实际部分删除产生的条件差。条件误差可由下面的乘积残差精确展开；这些式子本身不能证明 GDN 是现有 MAS 差距的主要来源。

不能由 GDN 层数、负贡献大小或 EOS 状态与输入状态的距离，直接决定应改哪一项。尤其不能把真实模型的记忆擦除当作错误的负归因。代码正确性、默认精度数值差、跨批次端点迁移与条件分配误差是需要分别核对的问题。

审查对象：

- [有限 FLA 核心](../research/runtime/finite_fla_gpu.py)。
- [完整 GDN 有限边界](../research/runtime/qwen35_gdn_finite.py)。
- [Decoder 与标准注意力有限边界](../research/runtime/qwen35_decoder_finite.py)。
- [CPU 块内代数参考](../research/runtime/finite_fla_chunk_reference.py)。
- [Qwen3.5 原生与有限传播契约](qwen35_finite_extension_contract.md)。
- 已保存官方 `modeling_qwen3_5.py`，SHA-256 为 `cf085792cb59e5bdf9b88a3d20bd353892289d054662a9c2b662221b97caefba`。其中 `torch_recurrent_gated_delta_rule` 给出递推参考；真实模型仍调用安装的原生 FLA。

当前核心包含 endpoint1 原生状态伴随、`A1ᵀ dU_WY` 坐标转换、key 系数中的正对角修正 `beta1 * k1 * dot(u0, L)`，以及 q/k/v/beta/decay 全部分支。GDN 外围包含输出 norm 与 SiLU gate 的两个分支、Q/K L2、因果卷积与 SiLU 有限传播。这里的“未发现漏项”限于本次代码和既有代数的审查，并不等于用四例总守恒证明每个实现细节都正确。

## 当前状态递推及条件误差的定义

对一个头、一个时间步，令 H 为 key×value 矩阵，q/k 为列向量，r/u/v/o 为 value 维行向量，alpha/beta 为标量。固定 query scale 可吸收到 q 中。官方递推为：

\[
C_t=\alpha_t H_{t-1},\qquad r_t=k_t^\top C_t,
\]

\[
u_t=\beta_t(v_t-r_t),\qquad H_t=C_t+k_tu_t,\qquad o_t=q_t^\top H_t.
\]

下标 0/1 始终表示 EOS/原输入两个实际端点；下标 A 表示实际部分删除后的原生执行。记 `delta x = x1 - xA`，区别于完整端点差 `Delta x = x1 - x0`。以下先在实数算术、已正规化 q/k 与 alpha 操作数上推导。raw-g→alpha、Q/K 正规化及外围投影的有限误差需另外计入。

对标量或逐元素产品 a·b，当前 content1 有限算子用于条件方向时给出：

\[
\widehat{\delta(ab)}=a_1\delta b+b_0\delta a.
\]

真实条件差为 `a1 delta b + bA delta a`，因此：

\[
\widehat{\delta(ab)}-\delta(ab)=(b_0-b_A)\delta a.
\]

矩阵/向量产品保留乘法顺序可得同样的恒等式。其含义是：路由变化仍然乘 EOS 内容，而真实条件差需要该删除状态的内容。误差既取决于内容偏离，也取决于相应路由是否真的变化。

在当前递推中，定义五个局部条件残差：

\[
\begin{aligned}
R_{C,t}&=\delta\alpha_t(H_{0,t-1}-H_{A,t-1}),\\
R_{r,t}&=\delta k_t^\top(C_{0,t}-C_{A,t}),\\
R_{u,t}&=\delta\beta_t[(v_{0,t}-r_{0,t})-(v_{A,t}-r_{A,t})],\\
R_{H,t}&=\delta k_t(u_{0,t}-u_{A,t}),\\
R_{o,t}&=\delta q_t^\top(H_{0,t}-H_{A,t}).
\end{aligned}
\]

这不是把 A 状态替换成混合端点构造的新前向。所有 A 量在诊断中必须来自真实模型的部分删除执行；上述乘积仅描述固定有限算子对该真实方向的误差。

为了明确误差如何沿状态传播，令带帽差表示将真实操作数变化 `delta q/k/v/alpha/beta` 输入当前固定有限递推所得的预测；中间状态差也由该有限递推传播。令 `E_t = delta_hat H_t - delta H_t`，初始状态相同且为零时 `E_0 = 0`。逐式相减得到：

\[
\begin{aligned}
E_t={}&\alpha_{1,t}(I-\beta_{1,t}k_{1,t}k_{1,t}^\top)E_{t-1}\\
 &+(I-\beta_{1,t}k_{1,t}k_{1,t}^\top)R_{C,t}
 -\beta_{1,t}k_{1,t}R_{r,t}+k_{1,t}R_{u,t}+R_{H,t},\\
E_{o,t}={}&q_{1,t}^\top E_t+R_{o,t}.
\end{aligned}
\]

这里各项的符号可以互相抵消。不能只把最大的绝对残差、最大的 baseline state 或最大的负分支叫作唯一根因。实际目标还通过被冻结的输出有限系数收缩这些向量和矩阵。

### 可以排除的错误解释：V 分支天然丢失内容

固定 q/k/alpha/beta 时，整个 GDN 递推对 V 严格线性。当前 `coeff['v'] = beta1 * A1ᵀ * dU_WY` 正是同输出伴随下原生 endpoint1 的 V 系数。因此，对任意只改变 V 的条件方向，该分支在实数算术下都是正确的；它并不因未显式形成全局注意力矩阵而失去内容作用。

真实 token 删除会同时改变 q/k/alpha/beta、输出 gate、正规化和外围隐藏状态。可疑机制是这些变化与固定端点内容之间的交互分配，而不是“线性注意力所以 V 分支不准确”。

## 原生状态转移不支持指数误差放大的说法

官方代码使用 beta=sigmoid(b)，alpha=exp(-exp(A_log)·softplus(a+dt_bias))，Q/K 使用带 epsilon 的 L2 正规化。在实数算术下有 `0 <= beta <= 1`、`0 <= alpha <= 1`、`||k||2 <= 1`。因此当前状态转移：

\[
T_t=\alpha_{1,t}(I-\beta_{1,t}k_{1,t}k_{1,t}^\top)
\]

满足：

\[
\|T_t\|_2\le\alpha_{1,t}\le1.
\]

其理由是括号内沿 k 方向的特征值为 `1-beta||k||²`，正交方向特征值为 1。状态递推本身不扩张。若将前节新注入项合记为 B_t，则：

\[
\|E_t\|_F\le\sum_{j\le t}\left(\prod_{i=j+1}^{t}\alpha_{1,i}\right)\|B_j\|_F.
\]

长序列可以持续注入新的交互残差、累计多个输出读取效应，但不能由此状态算子推出误差指数增长。默认 BF16 舍入可能改变精确不等式的数值边界；该结论不是对浮点实现逐步范数的测量。

此外，外部 RMSNorm、SiLU gate、MLP、输出投影和跨 decoder 传播不在此不扩张结论内。既不能说整个网络不可能放大误差，也不能因共有 24 个 GDN decoder 就说它们必定是 MAS 大头。

## 用真实边界区分状态递推与输出门控

先执行 [粗条件误差定位](dt_conditional_error_diagnostic_design_20260908.md)。只有它指向 GDN，才有理由增加内部捕获。对于同后端 B1 clean−A 的实际激活差以及当前冻结的实际系数，可细分：

\[
\begin{aligned}
e_{outproj}&=\langle m_y,\delta y\rangle-\langle m_{norm},\delta y_{normgate}\rangle,\\
e_{normgate}&=\langle m_{norm},\delta y_{normgate}\rangle
 -\langle m_o,\delta o\rangle-\langle m_z,\delta z\rangle,\\
e_{cast}&=\langle m_o-m_o^{BF16},\delta o\rangle,\\
e_{FLA}&=\langle m_o^{BF16},\delta o\rangle
 -\sum_{x\in\{q,k,v,\beta,raw\_g\}}\langle\widetilde m_x,\delta x\rangle.
\end{aligned}
\]

其中 `m_o` 对应 `mo_before_cast`，`m_o_BF16` 对应 `mo_native`，`m_norm` 对应 `mnorm`。q/k 是原生 FLA 捕获的已正规化操作数；raw-g 配 `coeff['g']`，不能配 `coeff['alpha']`。此 FLA 项包括 raw-g 的 exp 有限处理；若继续细分，必须明确另列 exp 边界，不能混用操作数。

后续 Q/K L2、beta/decay 参数、conv/SiLU 与输入线性投影单独闭合。保留同次 DT B2 两端点残差，以及评分 B1 clean−全 EOS 上使用相同系数的残差，以识别批次/端点迁移。不能把所有 B1 条件残差归为算法自身的问题，更不能把端点残差直接减掉当作已校准答案。

`norm_output` 是原生融合 norm-gate 的实际输出；`o`、`z` 也是实际捕获。若进一步按 n=RMS(o)、s=SiLU(z) 拆乘积，重算的 n/s 必须标记为归因代数因子，不能冒充融合算子单独返回的原生内部张量。融合舍入差应保留在账本中。

## 条件修法优先级

### 1. 先修已定位的计算或映射错误；目前没有这样的 GDN 漏项证据

若定位显示输入映射、批次迁移、某个局部完整端点恒等式或操作数配对错误，首先修该明确边界，并保留原默认 FA/FLA 精度。不能在看到大条件残差之前先升精度，也不能让方法改动掩盖调用错误。当前检查没有发现可宣布“补上就会有效”的 GDN 缺失项。

### 2. 输出乘积确是主因时，才评估 output gate 对称分配

记 n=RMS(o)、s=SiLU(z)。当前采用：

\[
\Delta(ns)=s_1\Delta n+n_0\Delta s.
\]

候选是：

\[
\Delta(ns)=\bar s\Delta n+\bar n\Delta s.
\]

它把一半 `Delta n * Delta s` 从内容分支移到 gate 分支，不裁负、不改变完整端点总差。只改该乘积分配不会改变 FLA 状态转移，因此可继续使用同一次原生 endpoint1 状态伴随。现有 o0/o1/z0/z1 足够，多算 n1 和均值仅增加 O(Td) 点运算；矩阵乘法次数与有限 FLA 调用次数不增加。真实耗时和峰值仍需记录。

但“平均更稳定”不是理由。暂只检查乘积，不混入前后 RMS/SiLU 割线及融合舍入，对实际 nA/sA，当前预测误差为：

\[
E_{P1}=(n_0-n_A)(s_1-s_A).
\]

对称规则改变的预测量为：

\[
C=\tfrac12[\Delta n(s_1-s_A)-\Delta s(n_1-n_A)].
\]

对冻结标量目标，令 e=<m,E_P1>、c=<m,C>。此局部乘积的绝对预测误差严格下降，当且仅当：

\[
c(2e+c)<0.
\]

这是可以用真实条件方向预先核对的代数门槛，不是最终 MAS 改善保证。若误差主要来自 RMS/SiLU 的条件曲率，或实际条件方向令 c 接近零/同向增加误差，就没有依据启动这个候选。特别地，若 nA/sA 具有相同的端点插值比例，两种分配给出相同预测，对称化无法修复该方向。

实际端到端候选还会改变更早层的传播和最终排序；局部改善既不保证整体原始效应误差下降，也不保证 MAS 下降。禁止用上述局部条件当作已完成的原指标验证。

### 3. 状态递推被定位后，才考虑其交互分配；不能仅替换几个端点乘子

若将整个递推的每个乘积改为对称分配，状态差的线性转移变为：

\[
\bar\alpha_t(I-\bar\beta_t\bar k_t\bar k_t^\top).
\]

其中均值是相应实际操作数的端点均值，例如 `alpha_bar=(alpha0+alpha1)/2`，不是 `exp((raw_g0+raw_g1)/2)`。这个转移一般不等于当前 endpoint1 原生转移。

因此不能仅修改 `mixed_coefficients` 的某些 baseline 因子，却继续原样使用 endpoint1 的 `native_input_adjoints`，再声称得到正确的对称有限递推。那会造成真正的数学不一致。必须明确构建对应的 WY/状态归因代数，或者采用两套端点伴随等具有额外成本的设计；也不能将混合操作数冒充真实模型反事实前向。

该方向没有输出 gate 单点改动的简单复用保证，目前也没有条件优势证据。若诊断未指向状态交互，就不应启动。若确实指向状态，也应先识别 R_C/R_r/R_u/R_H/R_o 的目标收缩及抵消关系，而非统一弱化 GDN 或扫描任意权重。

## 不支持的修法与承诺

不支持裁负、改 FT/评测、因 GDN 数量多而降权、只因负量大而削减擦除、任意扫端点权重，或在没有定位时回退 ordinary gradient。也不能用小的完整端点总残差证明所有条件方向正确。

当前可保证的是清楚的有限代数、原生复用边界和复杂度推导；没有证据保证某个尚未定位的改动必然降低 MAS。正确执行顺序是：真实条件误差闭合定位 → 候选是否抵消该计算边界误差的检查 → 固定原始指标的有限预算确认。候选无依据或未改善时应拒绝，不把缺少结果包装成架构结论。
