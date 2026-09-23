# Reused execution helpers

The Qwen3.5 capture helpers here come from
[`codex/qwen35-cause-and-tolerance` at e1e37bb4](https://github.com/LceOmlet/DeltaTrace/tree/e1e37bb45de19e62a60ab650e481a52337605e14/deltatrace/accelerated).
Their original byte hashes and integration edits are recorded in
[qwen35_capture_sources.json](qwen35_capture_sources.json).

`qwen35_retained_capture` retains the owner's actual native tensors;
`qwen35_code_local_capture` uses the existing CPython local-event observer.
Both inherit the current owner captures. The bridge retains current capture
filters and the stride-preserving MetaX host/device transfer fix. Neither
implements a model forward, attention, finite rule, reward, or optimizer.

The current official Qwen3.5 runner accepts these helpers through
`capture_backend`, and ports the historical `controller_deferred.py` event
timing and deferred scalar checks through `defer_diagnostics`. Default owner
execution is unchanged. This integrates with the existing RL target and FSDP
lifecycle instead of installing another copy of the historical controller.

The RL candidate selects local capture and deferred diagnostics. Same-actor
B4 controls compare all signed outputs, target scores and Q/V/A, with raw
results indexed in `experiments/rl/results_dt_minibatch_candidate.json`.
Short execution parity does not establish 32k capacity or throughput.

The optional `fa_coefficient_suffix` runner setting reuses the same finite FA
operator with an explicit output range. It derives each pair's first changed
input token once. In this causal decoder, the common input prefix has zero
displacement at every layer; earlier output coefficients also cannot feed
later input coefficients. Thus its omitted coefficients do not contribute to
the final signed attribution. The attention operator still reads the full
K/V history. This does not truncate the model input, alter EOS endpoints, or
change finite propagation formulas. Observer runs keep the complete trace.
The default remains disabled; the separately exported suffix operation must
be present in the selected, hash-checked library. Its output restriction is
checked against the full operator, and complete short attribution against
the same runner with the setting disabled.

Other audited historical capabilities are kept distinct:

- Dynamic compilation already exists in the current owner's public options.
- The old Qwen3.5 controller supports GPU checkpoints and native variable
  lengths. Those controller features have not yet been migrated here. At
  B4/32k, all 32 paired BF16 layer-input checkpoints alone require 64 GiB;
  selecting its all-GPU mode would not fit the current card.
- Qwen3 V4's graph and projection-cache controller is for a frozen FP16 Qwen3
  model, without GDN. Its parameter-version contract rejects optimizer
  updates. It is not selected for the Qwen3.5 BF16 LoRA/FSDP actor.
- Its finite FA source uses D128/FP16. Current Qwen3.5 uses D256/BF16; neither
  library identity nor historical full-model timings are interchangeable.

The sole RL method specification remains `experiments/rl/PLAN.md`.
