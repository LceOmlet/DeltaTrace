# Qwen3.5 native execution and finite-propagation contract

This contract now has a locally evaluated CPU reference and an official-compiler
GPU implementation of mixed FLA coefficients on actual EOS/input endpoints.
It is not a complete FLA attribution method. The existing Qwen3-8B quality/cost results do not
transfer to this model. Native execution and fixed-text mappings are recorded in
the dated evidence below; remaining finite pullbacks and whole-model checks are
still required.

## Native boundaries that differ from Qwen3

The pinned Qwen3.5-9B configuration has 24 gated-delta layers and 8 full-attention
layers. Full attention has BF16 head dimension256, a query/output gate, and
RMSNorm weights used as `1 + weight`; it cannot use the old FP16/D128 propagation
wrapper unchanged. Linear layers additionally contain depthwise causal
convolution, SiLU, Q/K L2 normalization, exponential decay, beta gating, and fused
RMSNorm with an output gate. The actual model forward must continue to use the
official Transformers model and its installed FA/FLA/causal-conv operators.

## An exact content-anchored recurrence

For one head, let the normalized/scaled query be q, normalized key be k, value be
v, decay α=exp(g), and write strength β. Let H have key-by-value shape. The
following is the recurrence in the pinned official Transformers reference:

\[
C_t=\alpha_t H_{t-1},\quad r_t=k_t^T C_t,\quad
u_t=\beta_t(v_t-r_t),\quad H_t=C_t+k_tu_t^T,\quad o_t=q_t^T H_t.
\]

Subscripts0/1 below refer to the actual EOS/input endpoints, not time. Applying
the same content-anchored product identity as the current P1 method gives:

\[
\begin{aligned}
\Delta C_t &= \alpha_{1,t}\Delta H_{t-1}+\Delta\alpha_t H_{0,t-1},\\
\Delta r_t &= k_{1,t}^T\Delta C_t+\Delta k_t^T C_{0,t},\\
\Delta u_t &= \beta_{1,t}(\Delta v_t-\Delta r_t)
                 +\Delta\beta_t(v_{0,t}-r_{0,t}),\\
\Delta H_t &= \Delta C_t+k_{1,t}\Delta u_t^T+\Delta k_tu_{0,t}^T,\\
\Delta o_t &= q_{1,t}^T\Delta H_t+\Delta q_t^T H_{0,t}.
\end{aligned}
\]

Each line follows by subtracting the two endpoint products. Substitution proves
the recurrence exactly in real arithmetic; finite-precision residuals must still
be measured. This allocation uses the actual input's memory dynamics to carry
content changes, while retaining key, query, write-gate and decay contributions.
Dropping those routing/gating terms would break the identity. The gate and
normalization differences still need explicit finite pullbacks; ordinary point
derivatives cannot silently stand in for them.

## What can reuse the native backward

Let Λ_t denote the cotangent for H_t, including the output contribution
q1,t times the output cotangent. Reverse the equations above:

\[
\lambda_u=k_{1,t}^T\Lambda_t,\quad
\lambda_r=-\beta_{1,t}\lambda_u,\quad
\lambda_C=\Lambda_t+k_{1,t}\lambda_r^T,\quad
\Lambda_{t-1}=\alpha_{1,t}\lambda_C.
\]

Thus the state-to-state cotangent transition is exactly

\[
\Lambda_{t-1}=\alpha_{1,t}(I-\beta_{1,t}k_{1,t}k_{1,t}^T)\Lambda_t,
\]

which is the native endpoint1 state transition. This offers a concrete reason to
reuse FLA's existing reverse state propagation rather than reimplement an entire
attention operator. The local parameter cotangents are different:

\[
\begin{aligned}
\tilde q_t &= H_{0,t}\lambda_{o,t}, &
\tilde v_t &= \beta_{1,t}\lambda_u,\\
\tilde k_t &= \Lambda_tu_{0,t}+C_{0,t}\lambda_r, &
\tilde\beta_t &= (v_{0,t}-r_{0,t})^T\lambda_u,\\
\tilde\alpha_t &= \langle H_{0,t-1},\lambda_C\rangle.
\end{aligned}
\]

These are coefficients of finite differences, not standard gradients. In
particular, feeding every operand from endpoint1 to an unchanged backward would
use the wrong state/value factors. A traceable extension must combine endpoint1
state-adjoint propagation with endpoint0 state reconstruction and the stated
mixed local contractions. The mixed factors are attribution algebra, not a new
counterfactual model forward. Reuse of the native adjoint transition is proved
here algebraically; reuse of a particular chunked kernel has not yet been proved
or implemented.

## Storage and acceptance conditions

The FLA0.4.1 chunk implementation uses64-token chunks. Its forward state helper
stores chunk-boundary states with shape `[B, ceil(T/64), H, K, V]`; it does not
need a global T-by-T matrix. Preserve that structure. Do not retain a K-by-V
state for every token across the entire network, or introduce a global dense
transition matrix just to simplify finite propagation. Reconstruct needed
endpoint states within the official chunk framework and release each layer's
temporary state after its pullback. The memory constant K*V is still material,
so linear sequence complexity alone does not establish acceptable memory.

Before integration, verify actual endpoints, the complete local finite identity,
state initialization, padding and chunk boundaries, GQA/head expansion, and
the official model's gate/normalization semantics. Then compare complete
same-trajectory attribution vectors and original FlashTrace metrics under a
frozen small budget. No quality advantage, reliable deletion sign, efficiency
gain or completed new-model support is claimed by this derivation.

Source identities: Transformers `modeling_qwen3_5.py`
`cf085792cb59e5bdf9b88a3d20bd353892289d054662a9c2b662221b97caefba`;
FLA0.4.1 original `chunk_delta_h.py`
`be25cba5e99073465c0653f50cd7c1c96466335e541a76cdfe449992d4146e01`.

## Concrete native kernel reuse boundary, 2026-09-08

The bounded native diagnostic now has actual BF16 FA/FLA execution and a first
layer official CPU reference. Its old batch screen remains failed; see the
[diagnostic report](history/Qwen35批处理差异定位与有限传播推进_20260908.md).
That default-model discrepancy must not be charged to an attribution extension
which has not yet run, or silently treated as permission to skip finite checks.

Inspecting the pinned FLA `gated_delta_rule/chunk.py`, `wy_fast.py`, and
`common/chunk_delta_h.py` identifies the following implementation boundaries:

| Native stage | Reusable work | Constraint |
| --- | --- | --- |
| Q/K normalization, cumulative g, triangular solve and WY preparation | Use official primitives for each actual endpoint. | Native helpers consume chunk-cumulative g, not raw log decay. Keep normalization, scale and endpoint identities explicit. |
| `chunk_gated_delta_rule_fwd_h` | Reconstruct endpoint0 chunk-boundary states and its transformed updates within the official chunk framework. | These are attribution reconstruction costs; they cannot be advertised as free cached model execution. |
| `chunk_bwd_dv_local` followed by `chunk_gated_delta_rule_bwd_dhu` | Endpoint1 reverse state propagation, driven by the finite output cotangent. | q1/k1/w1/cumulative-g1 determine this transition; the helper does not consume v values. Returned `dh` is a chunk-boundary adjoint, not every-token Λ. |
| `prepare_wy_repr_bwd` | Its endpoint1 value-gradient branch gives the finite content-v coefficient. | Its other ordinary gradients are not automatically finite coefficients; copying all its outputs would be wrong. |
| Mixed local q/k/beta/decay coefficients | Use endpoint0 state/update factors with endpoint1 adjoints, following the equations above. | This part still needs a traceable extension and numerical validation. No unchanged all-endpoint1 backward implements it. |

There is one further exact reuse result. With endpoint1 routing and decay held
fixed, the recurrence is linear in v. Thus the finite content coefficient
`tilde_v = beta1 * lambda_u` is exactly the ordinary native endpoint1 value
cotangent for the same output cotangent. This does **not** make the complete
method an ordinary gradient: q/k/g/beta still require mixed finite factors, and
the surrounding nonlinear gates/normalizations require finite pullbacks.

The native WY representation changes variables within each 64-token chunk.
Writing `U = A * (beta * V)`, its backward returns an adjoint `dU`; the source
computes `dV = beta * (A^T * dU)`. Consequently the `dv2` returned by
`chunk_gated_delta_rule_bwd_dhu` cannot be directly substituted for the
single-token `lambda_u` in the recurrence above. The triangular transform must
be accounted for. Dividing the native dV by beta is also unsuitable at zero or
tiny beta; reuse the underlying contraction instead.

The next implementation should preserve native endpoint1 state-adjoint work,
reconstruct endpoint0 states within chunks, and extend local contractions. It
must not store a K-by-V state for every sequence token or use a second full
ordinary backward merely to obtain one reusable branch without accounting for
its cost. Actual local backward execution is now verified after the two-line
upstream WY layout backport; see the [backward report](history/Qwen35原生FLA反向与上游两行修复_20260908.md).
This covers one saved real prefix and one output cotangent. Complete finite
coefficients, whole-model backward and a speed advantage remain unverified.

The observed WY value output equals the native autograd V gradient. Independently
contracting its saved A, dU and beta in CPU64 differs by 0.1669% in relative L2;
the native value-linearity Euler residual is 0.0371%. These are local numerical
checks of the reuse relation, not original benchmark or deletion-sign evidence.
The passive stage interface preserves native calls; its retained storage and
threading controls are diagnostic costs, not a default production design.

## Mixed coefficients using only chunk boundaries and 64-token tiles

The [CPU algebra reference](../research/runtime/finite_fla_chunk_reference.py)
now evaluates the stated recurrence's allocation using actual native FLA
intermediates. It does not replace model forward or ordinary backward. Simply
applying a generic product allocation to FLA's rewritten chunk graph can assign
interactions differently; the formulas below preserve the recurrence above.

All matrices in this section are for a single chunk of C≤64 tokens and one head.
Q includes the fixed query scale; the implementation applies that scale to the
returned normalized-Q coefficient too. Let H be the native endpoint0 state at
the chunk start, D the native endpoint1 adjoint at the chunk end, U the native
endpoint0 `v_new` (the actual writes, not WY's `u` before subtracting `w*H`), and
Z the output cotangent. Obtain L = A1ᵀ dU_WY and W = diag(beta1)L from native
endpoint1 reverse stages. This uses two native stages, without a second full
ordinary backward or an extra local forward.

Let G0/G1 be within-chunk cumulative log decay, and define

\[
 E_0[t,j]=1_{j\le t}\exp(G_0[t]-G_0[j]),\qquad
 E_1[t,i]=1_{i\ge t}\exp(G_1[i]-G_1[t]),\qquad
 e[t]=\exp(G_1[C-1]-G_1[t]).
\]

Let E0⁻ omit the diagonal, R = -W, and let `rowdot` contract feature dimensions.
For unscaled q coordinates the first expression also receives the query scale:

\[
\begin{aligned}
\tilde Q &= \operatorname{diag}(e^{G_0})ZH^T
              + ((ZU^T)\odot E_0)K_0,\\
\tilde K &= \operatorname{diag}(e)UD^T
              +((UZ^T)\odot E_1)Q_1
              -((UW^T)\odot E_1)K_1\\
 &\quad+\operatorname{diag}(\beta_1\,\operatorname{rowdot}(U,L))K_1
              +\operatorname{diag}(e^{G_0})RH^T
              +((RU^T)\odot E_0^-)K_0,\\
 r_0 &= \operatorname{diag}(e^{G_0})K_0H
              +((K_0K_0^T)\odot E_0^-)U,\\
 \tilde V &= W,\qquad
 \tilde\beta=\operatorname{rowdot}(V_0-r_0,L).
\end{aligned}
\]

These follow by expanding H0,t as its decayed chunk-start state plus earlier
writes, and Λt as its decayed chunk-end adjoint plus later output terms and
write corrections. The positive diagonal term in the K expression removes the
current write correction, which is in λC,t but not Λt. Omitting it is incorrect.

For the decay coefficient define Gprev[t]=G0[t-1], with Gprev[0]=0, and
Eprev[t,j]=1(j<t)exp(Gprev[t]-G0[j]). Also define

\[
\begin{aligned}
 S&=\langle H,D\rangle,\\
 b_i&=Z_i\cdot(Q_{1,i}H)-W_i\cdot(K_{1,i}H),\\
 d_j&=U_j\cdot(K_{0,j}D),\\
 M_{ji}&=(K_{0,j}\cdot Q_{1,i})(U_j\cdot Z_i)
          -(K_{0,j}\cdot K_{1,i})(U_j\cdot W_i).
\end{aligned}
\]

Then the exact mixed decay contraction is

\[
\tilde\alpha_t=e_t\left(e^{Gprev_t}S+\sum_j Eprev_{tj}d_j\right)
  +e^{Gprev_t}\sum_i E1_{ti}b_i
  +\sum_{j<t,i\ge t}Eprev_{tj}E1_{ti}M_{ji}.
\]

The last term needs no per-token K×V state. For each column i, accumulate
P[j,i]=M[j,i]+alpha0[j]P[j-1,i], with zero initial P. At cut t use P[t-1,i],
then multiply by E1[t,i] and reduce i. This affine prefix operation admits an
associative scan; the current CPU reference uses a bounded serial scan. All
valid decay exponents are nonpositive. Do not form inverse cumulative decays
or evaluate unmasked positive exponentials to implement this contraction.

Finally tilde_g = tilde_alpha * exp_secant(g0,g1), with a stable `expm1`
divided difference and its equal-endpoint limit. Native cumulative sums,
BF16 states, WY tensors and output rounding introduce residuals; the real
arithmetic identity is not a claim of bitwise numerical conservation.

On actual NI0/MH1 layer0 prefixes of129 tokens, the tile expressions differ from
direct local contractions by at most2.51e-7 in relative L2. Their complete local
finite-effect residuals are0.00635% and0.01586%, respectively. These results use
normalized q/k and a fixed local output cotangent, not the model answer target.
Separate per-head effect vectors have relative L2 residuals0.3997% and0.3677%:
the smaller total residuals include numerical-error cancellation and must not
be used alone as evidence of high accuracy. All32 head effects are retained.
The direct CPU oracle temporarily retains states for one head and one chunk;
it is diagnostic only. Production coefficients must use the chunk formula and
native state helpers, not that oracle. GPU fusion, normalization/gate/convolution
pullbacks, padded finite propagation, whole-model quality and cost remain open.
See [mapping and paired finite evidence](history/Qwen35官方映射与真实双端点有限系数_20260908.md).

## GPU implementation and official compiler fusion

The [GPU mixed pullback](../research/runtime/finite_fla_gpu.py) now executes the
native input-adjoint stages, BF16/FP32 existing `torch.bmm` calls, and one new
affine-scan contraction. The scan is finite attribution algebra, not an attention
forward or a replacement native backward. All sample/head/chunk dimensions are
batched; only64×64 interaction tiles are materialized. The last incomplete tile
is extended by identity transitions, not by fabricating extra model outputs.

Official `torch.compile` fusion reduced145 to45 kernel launches on the saved
real B2,T129,H32,K128 operands. In one matched job, local wall-time medians were
2.7656ms eager finite,1.4870ms compiled finite,1.0756ms native normalized backward;
the median paired compiled/native ratio was1.3853. Complete coefficient differences
between eager/compiled have relative L2 at most1.211e-7; per-head finite-effect
residuals remain about0.4032%/0.3663%. These are local engineering results.

First compiled-call preparation took25.746s. `max_autotune=False` still allowed
default compiler tuning:51 benchmark_gpu calls were recorded, with internal
repetition counts unmeasured. Static graph compilation is currently shape-specific.
Native model/FA/FLA implementations are unchanged by this extension. See the
[GPU implementation and cost report](history/Qwen35有限FLA_GPU实现与官方编译融合_20260908.md).

One bounded next optimization can use the exact inner-product identities
`Z·(Q1 H0)=Q1·(Z H0ᵀ)`, `W·(K1 H0)=K1·(W H0ᵀ)`,
`U0·(K0 D)=K0·(U0 Dᵀ)` to reuse three contractions already needed by q/k.
That reuse has not been implemented or measured; floating-point effects and
temporary lifetimes must be checked. Whole-model nonlinear/full-attention
pullbacks, real variable-length masking, quality and full costs remain open.
