# Native Qwen3.5 root-tape scheduling audit

This is an implementation audit, not a second method plan. `experiments/rl/PLAN.md`
remains the credit specification. No production source, PPO, Q/V/A, LoRA setting,
compiler rule or numerical tolerance was changed by this audit.

The candidate is generated from current prefix-provider owner SHA
`6348ebef115ff0ea45deeee3df8b65dcfbf0d1ae86619edc67f34d16d92d69aa`.
It keeps the original finite body and captures actual operands in the original
root call, consuming them once in reverse layer order. Source identities and
the bounded local test result are recorded in
`experiments/rl/results_native_root_tape_owner_cpu_20261004.json`.

## Default path and capture ownership

The new `test_native_root_tape_owner.py` passed 14 source/dispatch contracts.
The strongest check selects the actual candidate's disabled branch, inlines
the candidate capture factory, restores the two moved range calculations to
their original positions and removes only the new context/report seam. The
resulting entire `attribute` AST is exactly the original owner AST. The factory
has a separate equality check against the original capture construction.
Numerical module helpers, all other original methods, finite calls and
parameter prepare/release expressions have their own comparisons.

`NativeQwen35RootTape` composes the original capture contexts and original Torch
hooks. It closes each layer's capture before the next sequential native layer,
checks retained tensors' version metadata before consumption, and removes its
hooks/clears unconsumed fields in cleanup. It does not call a model or layer,
copy a finite formula, implement a cache, move tensors, or supply an offloader.
The candidate uses the original `_make_layer_captures` body for root and replay.
The wrapper's enabled and disabled branches call the same attribute body once;
the enabled branch clears the tape in `finally` without replacing the exception.
An omitted replay is not presented as a replay with zero numerical error.

Four optional tests use the exact existing `NativeDecoderCapture` class AST,
the existing real CPU Torch decoder fixture, and actual Torch forward hooks:
capture identity/dtype/stride, two-layer reverse consumption, original forward
failure cleanup, and a mutated borrowed tensor. They were **not executed** here
because the existing local interpreter has no Torch. The standard-library suite
reports one skipped class; that does not mean the four CPU tests passed. They
can run with GPU hidden in the already provisioned Torch environment, using
`DT_ROOT_TAPE_OWNER_SOURCE` and `DT_ROOT_INVENTORY_DECODER_SOURCE` for the frozen
owner files. No package installation is needed.

## Bound of the actual numerical comparison

`diagnose_native_prefix_leases.py:107` loads this independent candidate. Lines
135–151 switch only the class/flag/bound method on the same initialized runner;
the previously initialized finite objects and model remain in place. Its shared
factory uses one immutable bank and the lease creates a fresh original cache
for each consumption. Lines 271–276 restore the bound method, class and flag in
`finally`. Lines 307–313 compare raw Q/V/A to the same shared-bank variant using
equality and maximum absolute difference only. They do not invent a whole-DT
threshold or turn a small difference into acceptance.

For this scheduling seam, if the actual captured operands, incoming finite
coefficients and native arguments are unchanged, the same finite operator
paths retain their existing numerical reference scope. Source equality alone
does not establish that runtime operand identity. If the actual OFF/ON values
differ, locate the first different layer and compare its original root and
replay captures before changing numerical code.

For FA, retain the actual suffix query, **complete** cached key/value tensors,
native output, causal/dropout/scale arguments and their dtypes. Run the pinned
`attention_ref` and original `test_flash_attn_output` output assertion on that
geometry. The existing reference supports unequal Q/K lengths and the causal
bottom-right alignment; do not substitute a prefix-only square problem.

For FLA, retain actual raw Q/K, V, beta, raw gate, scale, `initial_state`,
`cu_seqlens`, native output and final state, and preserve dtypes. Cached suffixes
can have a nonzero initial state. The existing component diagnostic's
`initial_state=None` prefix read at lines 420–423 was built for another scope;
its old receipt cannot certify this root-tape cached suffix. Invoke the original
`recurrent_gated_delta_rule_ref` with the actual initial state and the original
`o`/`ht` assertions. Preserve strict `FLA_CI_ENV=False`; warning-only behavior
does not count as a passed assertion. For a finite-stage difference also record
the actual incoming coefficient and stage `k/w/u/g/h` rather than relabelling a
whole-DT error as a native FLA output error.

The original FA/FLA tests remain the authority for their corresponding outputs
and gradients. This audit does not extend those assertions to entire DT, Q/V/A,
PPO updates, root-tape capacity, or training effectiveness. The original B4
warm comparison and physical resource measurement are performed by the root
agent; this subtask ran no GPU or remote operation.
