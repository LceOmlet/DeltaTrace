# Counterfactual RL development branch

This directory is an engineering branch for Qwen3.5-9B on the existing A6000
container.  It does not install packages or alter the owner Transformers,
FlashAttention, FLA, or DeltaTrace sources.

The policy signal follows the proposition in
`paper/iclr2027/sections/appendix.tex`: for a fixed history and continuation
policy, an independently sampled reference action gives

\[
C(h,a)=\mathbb E_{a'}[G(h,a)-G(h,a')]=Q(h,a)-V(h).
\]

`counterfactual.py` computes this action-level signal and detaches it before
the score-function loss.  `deltatrace_credit.py` is deliberately strict about
the scalar endpoint: it accepts a DeltaTrace result only when the endpoint
effect agrees with the runner's compiled scalar seed.  It records any
signed-vector residual as an attribution diagnostic.  The vector itself is
never used as a per-token GRPO/PPO weight, because the owner runner explicitly
keeps that residual unassigned.

`train_counterfactual.py` is a short MATH-data smoke trainer.  Its return is a
bounded prefix-match verifier score after a forced first action, with the
continuation sampled from the same policy for the selected and reference
actions.  It uses a small LoRA adapter
(`q_proj`/`v_proj`, rank 4) because the shared environment has no PEFT/TRL and
full Qwen3.5-9B updates do not fit the requested single-card budget.  The
truncation and three-step settings are runtime checks, not final benchmark
claims.

The owner DeltaTrace evaluator must run inside the existing CUDA container:
the host venv exposes GLIBC-incompatible wheels even though the container's
same venv imports `causal-conv1d 1.5.4` and `flash-attn 2.8.3` correctly.

The first measured A6000 baseline (Qwen3.5-9B, BF16, 12-token prompt,
3-token greedy generation) loads in 7.6 s and reaches 12.5 generated tokens/s
after warm-up, with 18.9 GB peak allocation.  A real action-intervention DT
trace on a 108-token prompt and 8 target tokens takes 9.8 s cold and 1.31 s on
the second identical call in the same process, at 19.0 GB peak allocation.
With a 340-token prompt and 247 target tokens it takes 10.6 s in a fresh
process using the already populated shared compiler cache and 1.64 s on the
second identical call, at 20.1 GB peak allocation.  A clean official smoke
process measured 53.4 s for a comparable 357-token/248-target case, showing
that the cold cost includes cache population.  The gap is compiler and Triton
work concentrated in the first finite decoder calls; the runner's steady-state
memory is nearly flat for these lengths.

The LoRA compatibility check also passes with a nonzero rank-4 update: the
effective linear weight exposed by `lora.py` keeps the owner finite pullback's
transpose maps exact, while the adapter still reports the scalar endpoint
effect independently of the signed-vector residual.
