# Qwen3 native replay tensor retention

Use `--family qwen3 --dt-backend retained_qwen3 --sample-batch 1` with the
existing official evaluator. The clean default, frozen paper tag, previous
deferred options and Qwen3.5 backend remain separately available. This option
uses two endpoints per example; it does not add Qwen3 sample batching.

The native layer replay already produces the tensors consumed by DT. On the
pinned Qwen3 implementation, downstream native operations do not mutate these
operands. This backend retains their detached tensors instead of copying each
one. It changes four capture expressions and one DT import; original root
checkpoint copies, including the compact target-logit copy, remain. Retaining
a view of the full vocabulary output would have a different memory cost, so
that storage boundary is not changed.

The actual model/FA calls, original finite attention kernel, precision,
finite rules, target, all 828 validation predicates and 36 statistics are
unchanged. No native callable is replaced. The published source manifest
pins the measured DT files and depends on the existing deferred manifest.
The factory fails on source mismatch. The native model remains independently
pinned by the evaluation environment. This is not automatic support for
arbitrary future native implementations with different mutation semantics.

| Same-process complete warm calls | Deferred B1 | Retained B1 | Reduction | Peak GB, before → after |
|---|---:|---:|---:|---:|
| Original NI8/MH8 | 10.3595 s | 9.7587 s | 5.80% | 18.7603 → 18.6731 |
| Frozen author VT2/Hotpot4 | 7.9937 s | 7.6653 s | 4.11% | 21.8524 → 21.6220 |

The pilot, original16 and extra6 contain 156 complete DT calls; each sample's
six full signed vectors match its same-process baseline exactly. Signed RISE
ordering and positive MAS/needle inputs therefore remain identical in these
controls. No new task-score calls or quality improvement are claimed. This
observed equality is specific to removing copies, not a required standard for
future FA tolerance optimizations.

Each mode/shape is warmed before two interleaved measured passes. Full timing
includes checkpoint capture, finite propagation and diagnostics. First-shape
times share compiler caches; they are preserved in the receipts but cannot
be compared as independent cold starts. These baselines already contain the
earlier deferred optimization. Do not combine different-run timings into one
measured speedup, or treat any DT speedup as a change to FT.

Evidence: [all three controls](../../research/temporary/cause_tolerance_20260909/qwen3_retained_summary.json),
[source derivation](../../research/temporary/cause_tolerance_20260909/qwen3_retained_derivation.json),
[source manifest](retained_sources.json).

The [published factory integration check](../../research/temporary/cause_tolerance_20260909/qwen3_retained_api/summary.json)
adds 12 actual NI0/MH0 calls; all full vectors match, and both published
implementation files are byte-identical to the measured research files.
