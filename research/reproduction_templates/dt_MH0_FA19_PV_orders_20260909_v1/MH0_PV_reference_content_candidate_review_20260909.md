# MH0：PV 参考内容错配与同 GEMM 候选审查

2026-09-09；只读源码、原结果与数学审查。未运行 GPU、模型、FT、评分或新规则。本记录没有认定任何候选已改善完整输入归因或 MAS。

当前 MH0 第 19 层确有大的 **PV 参考内容错配**，也同时有大的路由预测误差。前者有必要检验分配规则，但不能直接推出换 P0 或对称分配更好。现有后端可以单次调用得到 P0；生产对称规则也存在不增加 GEMM 数的实现路径，但需要最小归因内核适配，不能把两次后端调用平均当作免费生产方法。

## 1. 已有实际证据能说明什么

令 C 为实际 B1 clean，A 为原评分删除输入，V0 为原 B2 EOS 端点内容，m 为保存的生产 FP32 mcontent。固定这些实际坐标，原生 FA 算子对照给出：

\[
R_0=\langle m,\operatorname{FA}(Q_C,K_C,V_0)-\operatorname{FA}(Q_A,K_A,V_0)\rangle,
\]
\[
B_V=\langle m,O_C-\operatorname{FA}(Q_C,K_C,V_A)\rangle,\qquad
R_A=\langle m,\operatorname{FA}(Q_C,K_C,V_A)-O_A\rangle.
\]

预测减实际的完整 core 误差严格拆成

\[
E=(D_{qk}-R_0)+(D_v-B_V)+(R_0-R_A).
\]

|原删除步|完整 core E|路由在 V0 上的预测误差|内容分支误差|PV 参考内容交互|
|---|---:|---:|---:|---:|
|3（early）|15.003021|6.702619|0.079080|8.221321|
|10（middle）|18.072608|7.863335|0.029794|10.179479|
|20（allEOS）|0.073491|0.016757|0.041921|0.014813|

其中 early 的 R0 / RA 为 +1.994513 / −6.226808，middle 为 +4.277872 / −5.901607：换参考内容后，实际原生路由算子投影发生反号。理想实数下第三项是 \(\langle m,(P_C-P_A)(V_0-V_A)\rangle\)，实际账本还含默认 FA 舍入。这不是 EOS 的独立因果贡献，也不是混合端点模型反事实。11 次原生 FA 中四次实际输出重放漂移均为零；FP32/BF16 seed 差另列，未吞入守恒残差。

原 content-P1 分支在这两个条件方向上的误差很小；改成 P0 或 Pmean 会同时改变该分支。因此不能只看路由改善，更不能从 PV 项大就断言总误差下降。第一项仍未区分 QK 交互与 softmax 条件斜率；那是已安排的独立现有诊断的职责。

## 2. 三种规则与现有后端的交换对称性

0 / 1 表示原 B2 EOS / 原输入端点，\(\Delta X=X_1-X_0\)。三种规则都是精确双线性恒等式：

\[
\begin{aligned}
\text{P1: }&\Delta(PV)=\Delta P\,V_0+P_1\Delta V,\\
\text{P0: }&\Delta(PV)=\Delta P\,V_1+P_0\Delta V,\\
\text{对称: }&\Delta(PV)=\Delta P\,(V_0+V_1)/2+(P_0+P_1)\Delta V/2.
\end{aligned}
\]

现 kernel 路由乘子为 \(W=\ell(p_0,p_1)(g-c)\)，其中 \(g=UV_0^T\)，\(c=\sum_j\ell_jg_j/\sum_j\ell_j\)。`vendor_fa_finite_p1_bf16_d256.cu:155–166` 的 logmean 用 `max(lp0,lp1)`、`abs(lp1-lp0)` 和 `expm1` 计算，对交换两端对称；`tau` 不变，center 对 g 线性。`:132–136` 的 Q/K 平均同样交换对称。`:166,191` 的内容权重明确为原 `actual_p1`。

所以沿现有 wrapper 交换 q0/q1、k0/k1、lse0/lse1，并把 v0 槽换为真实 v1，u / scale 不变，即得到 P0 规则的系数：QK 中点、logmean 不变，路由改用 g1，内容改用原 P0；**不需要给结果乘负号**。这只是归因规则，模型原生 FA 没有改变。

理想实数下两次系数的平均对应对称规则，因为上述路由算子对 g 线性；实际 BF16 输出的 CPU 平均只是诊断代理，不保证逐位等于未来单次 symmetric kernel。只平均 dq/dk/dv 等线性系数；不能把 center/tau 等调试状态一概平均后冒充重新运行。

对称分配的功能理由是对“路由变化”和“内容变化”的交互不偏向一方。在理想条件路由预测准确的简化下，P1 改对称对 C−A 预测的变化为

\[
\tfrac12\langle m,(P_C-P_A)(V_1-V_0)-(P_1-P_0)(V_C-V_A)\rangle.
\]

第一项可以减少 EOS 内容参照偏差，第二项却改变原输入的内容路由。两者符号均需实测；在原 1−0 弦方向上恰好抵消，故端点守恒不能认证条件改善。

## 3. 当前三 phase 的真实 GEMM 成本

源码 `vendor_fa_finite_p1_bf16_d256.cu:141–148,175–192` 每个有效因果 tile 的乘法如下。Phase 2 以 key 为输出行，矩阵方向转置；这里数的是模板 GEMM 调用，不是整个设备 launch 数或延迟比例。

|phase|已有乘法|每 tile GEMM 数|
|---|---|---:|
|0|Q0K0、Q1K1、UV0；归约 tau/center|3|
|1|同三项，再 W Kmean|4|
|2|转置同三项，再 Wᵀ Qmean、P1ᵀ U|5|

当前 wrapper 只传 v0，既没有读 v1，也没有隐藏的 g1。若三个 phase 都显式再计算 UV1，则总数由 12 增到 15，约多 25% tile GEMM 调用；不能据此直接预测延迟，也不能称为免费。

但只实现一条对称规则不必同时形成 g0/g1：先形成 Vmean，复用原一次 U Vmeanᵀ，并把 Phase 2 的 P1 换成 Pmean。GEMM 次数、三 phase 与线性全局存储阶数可以保留。V1 的实际加载、平均、默认 BF16 舍入、可能的线性暂存及性能变化都必须计入。仅给现 wrapper 传 Vmean 而不改 dv 权重仍是混合错误规则。最小实现应延续可追溯 vendor `flash::copy/gemm/gemm_rs` 框架，不新增一套 attention forward。

P0 的一次原后端调用本身无需增加 GEMM；两次 P1/P0 后端后平均需要六 phase，只用于当前受控诊断，不能据此主张生产成本不变。

## 4. 旧 Qwen3 的反证与本次最小判别

旧原 NI0–7 / MH0–7 共 16 个已使用开发样本结果如下；是当时 Qwen3 FP16、显式平方矩阵归因版本，不能当作当前 Qwen3.5 BF16 的直接指标。

|规则|NI needle ↑|NI RISE ↓|NI MAS ↓|MH RISE ↓|MH MAS ↓|
|---|---:|---:|---:|---:|---:|
|对称|0.778991|0.064765|0.108313|0.070935|0.162727|
|P1|0.775200|0.057423|0.080668|0.058475|0.117276|
|P0|0.726675|0.072000|0.124782|0.092238|0.197932|

P0 相对 P1 的 NI needle 低约 4.85 个百分点；P1 相对对称只低约 0.38 个百分点，且当时用户允许该取舍。旧证据明确反对未经验证直接把所有层统一换回 P0 或对称规则；不能以当前单例 PV 机制解释抹去这些退步。

当前最小判别已由 root 安排：同一实际 B2 端点、同一保存上游 m，**1 次公共 B2 FA 取得真实 LSE + 2 次当前有限后端（control / reverse，共六 phase）**；零模型、整网 DT、scorer、FT 和新排序。原 MH0 internal / hybrid 私有保存没有可直接复用的 LSE，不能漏计这次公共调用。新 control 对旧保存系数的漂移单列；固定 step 3/10/20 与 B2 方向比较 dq/dk/dv 完整 core 收缩，以及 P0 与 CPU 平均代理的有符号分项，不只挑有利分支。

这个局部试验只能判断具体条件方向上的相互补偿，不能直接输出新 MAS/RISE/needle，更不能把所有局部绝对误差下降当作全网改善的充分或必要条件。下一阶段若有依据，应先完成相应输入传播并保留原 NI needle 回归；本记录不自动授权追加测试、全层扫描或连续权重搜索。

## 源证据指纹

- `audit/snapshot${ARTIFACT_ROOT}/codex_dt_MH0_FA19_native_hybrid_20260909_v1/results.json`：`3fde131fe10487d66c11e21d8fe1616785eb71a662f3f1e0f381ebf3dab21f89`。
- `DeltaTrace/research/prototypes/vendor_fa_finite_p1_bf16_d256.cu`：`8116d27efdd9f41c5eab84d2a062fe9e5c8ebed58fd21c87ef3d52d26749b19a`。
- `DeltaTrace/research/runtime/vendor_fa_finite_bf16_d256.py`：`645736f68bf2f8169feff7218d30de20d3ed2a0b75e321603e6503a514763fc2`。
- `audit/pv_interaction_development16_summary_20260907.json`：`96beaf39a814274ddaf847b2c469beeef4c844387162392dd66d9d7b25fbadeb`；完整解释见 `DeltaTrace/docs/history/PV交互分配_三规则完整16条结果_20260907.md`。
- 当前 FA 调用端点来源：`DeltaTrace/research/runtime/qwen35_decoder_finite.py:130`；原 internal LSE 生命周期：`DeltaTrace/research/reproduction_templates/dt_MH0_FA19_internal_20260909.py:175`。
