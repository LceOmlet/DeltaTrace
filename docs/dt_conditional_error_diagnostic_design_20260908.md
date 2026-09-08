# DT 静态幅度失配的只读机制审查

2026-09-08；子任务产物。只读检查当前仓库、原始三例 results.json 和已关闭 MLP 结果；未修改 FT、DT 核心或评测，未启动 GPU，未重跑任何关闭样本。以下诊断/候选尚未执行，不是改进声明。

## 结论与证据边界

优先做“真实条件误差的逐层分解”，不能凭某层负质量增加就认定它是 MAS 问题。当前端点守恒误差小，与中间删除预测失配是两件事：NI1、NI2、MH1 的整网 relative_residual 分别为 0.1483%、−0.0548%、0.3471%，但 MAS 差距明显。此事实反对“仅因总差漏算”解释，不排除可抵消的局部错误。

四例 MAS 均落后，支持把静态分配/条件效应错配列为当前方法问题；它没有定位 FA、FLA 或 MLP 哪一个算子。现有 layer 账本只有 root_output_effect / replay_output_effect / input_effect 等标量，没有每层 token 贡献或 branch 贡献，不能从已保存标量重建抵消来源。

MLP content1 已在 NI0 使 needle、RISE、MAS 均退步，保持关闭。下面的潜在改动是 GDN **输出门控由当前 content1 改成对称分配**，不是重复给已有 content1 命名，也不涉及重新改 MLP/PV/FLA 状态规则。只有定位证据指向该边界才有启动依据。

## 当前规则实际是什么

设所有 0/1 均为 EOS/输入端点；m 是同一目标的有限系数。

1. **标准注意力的路由/内容。** PV 使用 Δ(PV)=ΔP·V0+P1·ΔV。softmax 采用端点概率的 logarithmic mean L，有限算子 J=diag(L)−LLᵀ/sum(L)；QK 产品用端点均值。J 是有限割线算子，满足 JΔz=ΔP（实数算术），不等于整条输入删除路径的 Jacobian 积分。FA 框架只实现该明确有限规则；其分块、默认精度和正端点内容路由本身不保证 MAS 对齐。输出 sigmoid 门控另用 content1。见 research/runtime/qwen35_decoder_finite.py:73、core/signed_secant_rules.py:23、research/prototypes/vendor_fa_finite_p1_bf16_d256.cu。

2. **FLA 状态的写入/擦除。** 原 GDN 递推 C=αHprev、r=kᵀC、u=β(v−r)、H=C+kuᵀ、o=qᵀH。有限传播每个乘积采用 content1，从而 ΔC=α1ΔHprev+Δα Hprev,0、Δu=β1(Δv−Δr)+Δβ(v0−r0)。状态反传复用原生 endpoint1 的 α1(I−β1 k1k1ᵀ) 转移；q/k/β/α 的局部系数使用 endpoint0 状态/写入。减去旧预测 r 是真实模型的擦除机制，负贡献不能一概叫伪差；但 EOS 旧状态与实际部分删除状态不同，可使这种分配不能预测实际条件擦除效应。24 层的数量只是优先测量的成本/覆盖理由，不是已证明其责任最大。见 research/runtime/finite_fla_gpu.py:65、:91 及 docs/qwen35_finite_extension_contract.md。

3. **GDN 输出门控。** n=RMS(o)，s=SiLU(z)，输出为 n⊙s。当前代码明确 mo=RMS_secantᵀ(m⊙s1)，mz=m⊙n0⊙SiLU_secant(z0,z1)，对应 Δ(ns)=s1Δn+n0Δs。norm 的 r0/r1 则仍采用对称有限规则。这里 n0 是用实际捕获 o0 与原参数计算的归因公式因子；不能把它说成原融合 norm-gate 内部已单独返回的原生 n 张量。更细的 n/s 分解是归因代数诊断，先以实际捕获 o/z/融合输出核验完整边界，保留融合舍入差；不要新增影子 norm-gate 前向。见 research/runtime/qwen35_gdn_finite.py:114–125。

4. **MLP。** y=u⊙SiLU(g)，当前为对称分配 Δy=mean(SiLU(g))Δu+mean(u)ΔSiLU(g)。已关闭候选才是 content1。见 core/compiled_swiglu_secant.py:10。MLP 与 RMSNorm 都是逐 token 的算子：局部恒等式成立时，对同一个 token 的总收缩守恒；它们可改变隐藏通道方向，进而改变更早 mixer 如何分配，却不会在该边界直接把总效应搬到另一个 token。

共同保证限于两端点的总有限关系及相同端点极限等已核验范围。m_l 的不同合法取值可在 Δh_l 的正交方向不同；删除引起的 h1−hA 不必沿 Δh_l。因此守恒和符号存在都不等于条件符号正确。

## 诊断 1：流式 token 与分支账本，先找可审查边界

在将来一个已授权的正常归因 pass 中，保留实际隐藏坐标，逐层归约 c_l,t=Σ_d m_l,t,d(h1_l,t,d−h0_l,t,d)。记录 P_l=Σ_t max(c_l,t,0)、N_l=Σ_t max(−c_l,t,0)、net=P_l−N_l；另保留 token 向量和 token 范围，不能把中间生成位置或非 eligible 位置的质量默认为最终输入归因。

对 mixer，可进一步对比进入它的 R_t=<m_y,t,Δy_t> 与出来的 Q_t=<m_x,t,Δx_t>，记录 Q−R。MLP/残差/norm 是逐 token 的，因而在舍入误差外，整层 token 分配变化应由 mixer 的 Q−R 解释。局部路由分支可记 FA 的 q/k/v/output-gate，FLA 的 q/k/v/β/g/output-gate；报告各支净收缩与绝对收缩，不只总和。

成本是现有 GPU 张量的 O(Td) 归约与 O(T) 每层输出，不增加模型前向/反向或全局 T² 存储。分支大张量应当场归约，不开启全部 diagnostics 后把整网中间量留在显存。所有开销仍需计入；不要合入当前 head ABBA 计时控制。

**限制：** 固定实际隐藏坐标下的通道级 P/N、分支 P/N 不能跨任意重参数化比较，线性换基也可能改变通道级抵消。token 级收缩在同 token 内、系数同步逆变换的线性换基下是不变的，但这不赋予跨 token 任意基变换不变性。大 N、N 增量或大抵消可以完全真实；该账本只定位后续诊断位置，不能据此剪层、裁负或直接定罪某层。

## 诊断 2：真实条件误差的逐层望远镜分解（优先级最高）

在未来**本来就要执行的原评测**中事前固定 1–2 个删除进度（例如第 1 步与第 10 步，不按结果选最差点）。保存原指标已有的 B1 clean、这些 B1 I_A、B1 全 EOS 前向边界值；所有差均以同后端 B1 clean 减 B1 I_A，不能用 DT 的 B2 clean 减评分器 B1 I_A。系数仍使用冻结 DT 双端点实际得到的 m_l。不要通过混合端点构造 h_l(A)，也不为此重跑已关闭样本。此处 A 是该原始曲线已删除集合；只用于诊断，绝不送回评分器调参数。

取 h_0 为原模型输入嵌入（首 decoder 的实际输入），h_L 为最后 decoder 输出，n 为实际最终 norm 输出，z 为实际原输出头的完整词表、所需目标行 logits。令 m_l、m_n、m_z 分别为冻结 DT 在这些坐标上的有限系数/种子。都要采集实际当前系数；不要另用 FP64 理论系数替代 BF16 投影路径所得系数。令 d_l(A)=h_l(B1 clean)−h_l(B1 I_A)，a_l(A)=<m_l,d_l(A)>，a_n/a_z 同理。对层 l 定义

    e_l(A) = a_(l+1)(A) − a_l(A)

令 D_native(A)=G_native(B1 clean)−G_native(B1 I_A)，其中 G_native 就是作者原指标的默认 BF16 评分；对**同一次**保存的实际 B1 logits 非侵入地计算同目标、同词表的 FP32 log-softmax 诊断 G_32，得 D_32。分开记录

    e_score_precision(A) = D_native(A) − D_32(A)
    e_logprob_seed(A) = D_32(A) − a_z(A)
    e_head_projection(A) = a_z(A) − a_n(A)
    e_final_norm(A) = a_n(A) − a_L(A)
    e_input_map(A) = a_0(A) − Σ_(t∈A) s_DT,t.

于是精确地

    D_native(A) − Σ_(t∈A) s_DT,t
      = e_score_precision + e_logprob_seed + e_head_projection
        + e_final_norm + Σ_l e_l + e_input_map.

输入 map 单独覆盖 EOS/token/eligible 映射、输入实际 B1/B2 舍入与最终归约数值差；不应先假定它严格为零。只有映射检查通过时，输入嵌入 d_0 才仅在 A 上非零，且 a_0 近似等于 Σ_A s_DT。所有内积可离线使用同一 FP64 归约以检查上述代数闭合，但必须保留原保存的默认精度系数/激活，不把归约精度提升冒充生产路径精度变化。

这个关系不依赖候选获胜、不借 gold，也不把总守恒当条件正确。更细的 mixer / MLP / norm 边界可以同样望远镜分解。e_l 的正负仅表示该条件下各段收缩差对最终**预测误差**的方向，不是 token 的正/负因果贡献。它真正测量哪一步把当前有限系数用于实际条件方向时失配，比凭 N 增量猜测有信息。

**B2→B1 与数值迁移对照不可省略：** m 来源于 DT B2 EOS/input，而 h 差来源于原评分器 B1，不能把 e_l 全算作方法本身的误差。对每一边界同时保留两类端点控制：(i) DT 同次 B2 input−EOS 上的既有局部有限残差；(ii) 原曲线本来就含有的 B1 clean−B1 all-EOS 上，用同一 m 收缩所得的残差。第二者与第一者的变化量显示端点/批次迁移及舍入影响。若某层仅在这些端点控制就明显不闭合，应先归类为迁移/数值混合问题，不声称已定位条件路径机制。若控制小而中间条件残差大，可支持“与条件方向失配一致”的结论，但依然不能无论证地从中间残差减去端点残差当成校准后的方法误差。保留全部原数值和这个识别边界。

实施可在归因时将 m_l 与上述种子系数写入 CPU，在原评分的 B1 clean 保存 clean 边界，在已计划的 I_A/全 EOS 前向按层流式加载、收缩和释放；隐藏部分额外存储 O(LTd) 在 CPU、GPU 暂存一层。输出头只保留实际所需目标行和完整词表，不裁词表；种子/logit 诊断 O(K_target Vocab) 的 CPU 保存/分块收缩成本另列，不能藏在 O(LTd) 内。零额外模型调用不等于零成本：传输、归约、hook 和 CPU 保存计入诊断；不用于声称生产加速。先粗层级再定点边界，避免全模型全中间量输出。

如果主要误差来自顶部 logprob seed / final norm，就不先动 FA/FLA；如果主要为 e_score_precision 或输入 map，不能反过来改传播方法背锅；如果集中在 GDN mixer，且 B1/B2 端点控制不足以解释，则继续区分真实记忆递推与输出门控；如果主要分散于归一化/MLP，说明仅改 P1 的动机不足。各 e_l 有正负，可互相补偿，应保留有符号与绝对量，不能只把最大绝对层当唯一原因。

**从原始效应接回 MAS 的账本也要分开：** 上式分解的是未归一化效应误差，不是最终 MAS。令 S=Σ_eligible s_DT、D*=D_native(all-EOS)>0、B_A=Σ_A s_DT，原密度 ρ=1−B_A/S，原始归一化响应 u=1−D_native(A)/D*，则

    ρ−u = [D_native(A)−B_A]/D* + B_A(1/D*−1/S).

前一项才接上述逐边界残差，后一项是实际指标端点净效应与有限分数总和的分母差。保留原函数实际绝对值/退化条件；这里只写当前 D*>0 的情形。接着独立列出原指标截断及 prefix-min 带来的 u→r 改变，最后按原 MAS 函数计算。不能把 seed 精度、B2/B1、分母差或响应截断全部记在某个 decoder 的方法误差里，也不能直接从 MAS 减去某一残差 AUC。

## 一个条件性候选：仅 GDN 输出乘积改成对称分配

保留当前 FA/PV、GDN 状态递推、MLP 及所有实际模型算子。把 GDN 的输出乘积换为

    Δ(ns) = s_bar Δn + n_bar Δs,
    mo = RMS_secantᵀ(m s_bar),
    mz = m n_bar SiLU_secant(z0,z1).

相对当前 content1，这恰好把 mΔnΔs 的一半从内容分支移到输出门控分支；不裁负、不改总差、不加额外端点。现有 o0/o1、z0/z1 足够，多算 n1 和均值是 O(Td) 点运算，可复用当前 Torch 编译器。原生 FA/FLA 完全不改，矩阵乘法次数与有限 FLA 调用次数不增加。不要预先声称 wall time 或峰值必定相同。

**为何可能有用：** 当前把整个门控×记忆内容交互送入 FLA 内容传播；部分删除后，gate 与 memory 的变化未必同步。若真实条件残差集中在这里，重新分配交互可能减少由远端 n0 因子造成的错误幅度，并把部分作用归到本地 gate 特征，而非继续沿记忆内容路由传播。

**为何这还不是证明：** 把实际中间值写成 nA/sA，暂单独检查乘积、暂不混入前后的 norm/SiLU 割线。当前 P1 对该条件差的预测误差为

    predicted_P1 − actual = (n0−nA)(s1−sA).

对称规则误差为

    predicted_sym − actual = (n_bar−nA)(s1−sA) − (s1−s0)(n1−nA)/2.

两者无统一大小关系。特别地，若 (nA,sA) 在端点共同直线比例上，两种分配对此方向给出相同预测，均不能消除乘积的路径曲率。因此不能用“平均更稳”“负量更少”宣布其应当更好。只有诊断 2 定位这里，并显示实际条件方向存在能被此次交互重分配修正的误差，才冻结一次小预算候选对比。若残差主要来自记忆状态递推或输出种子，这一候选应先不执行。

该改动与已经失败的 MLP content1 是不同边界、不同方向；既有 MLP 失败不直接否定它，但同样提醒“恒等式正确且复杂度不增加”远远不够证明质量。已有 NI0–2/MH1 曲线不足以指明它会改善 MAS，更不支持对称混合权重或逐层权重扫描。

## 建议交接

官方 logits_to_keep 配对测试已完成，保持纯工程控制，未加入上述诊断。下一质量任务优先实现可选、流式的诊断 1，并将诊断 2 绑定于下一次事前冻结的原评测调用；不额外扩数据，不重开已关闭消融。根据条件误差分解确定要改哪个边界，再决定是否执行上述单一候选。所有判断仍基于当前 DT 和固定 FT，未提出修改 FT。
