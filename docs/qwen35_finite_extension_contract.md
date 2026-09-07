# Qwen3.5 native execution and finite-propagation contract

This is a derivation and implementation contract, not an implemented or validated
FLA attribution method. The existing Qwen3-8B results do not transfer to this model.
The immediate runtime prerequisite is successful default FA/FLA execution on the
actual checkpoint and author-fixed text, with real batch and padding checks.

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
