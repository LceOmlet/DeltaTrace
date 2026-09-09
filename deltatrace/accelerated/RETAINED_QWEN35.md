# Qwen3.5 native replay retention

Use the formal evaluator with `--family qwen35 --dt-backend retained_qwen35
--sample-batch 2`. This keeps the existing real B2 examples / B4 endpoints,
right-padded default native FA/FLA, finite formulas, full cached target and
BF16 precision. It only subclasses DT's three passive capture helpers to
retain original tensors instead of copying them. Root checkpoint and output
storage copies remain. No model, attention or native backward is replaced.

Before measuring, two complete B2 calls kept an independent copy at every
capture. All 2,080 tensor snapshots across 128 capture objects matched after
the native layer returned. The GDN, decoder and FA capture paths are all
covered. This includes the actual native FLA path, not an alternative recurrence.
The extra audit copies and comparisons are separately charged, disabled for
warm timing, and not silently omitted from reported research cost.

| Same-process complete warm calls | Deferred B2 | Retained B2 | Reduction | Peak GB, before → after |
|---|---:|---:|---:|---:|
| Original NI8/MH8 | 11.4803 s | 10.8892 s | 5.15% | 23.62895 → 23.62888 |
| Original author VT2/Hotpot4 | 10.0751 s | 9.6495 s | 4.22% | 30.29720 → 30.09073 |

These are increments over the already accelerated deferred backend, not
combined multipliers from different historical runs. Every shape is warmed;
two measured passes reverse mode and group order. All warm calls have no new
compiler graphs. First-shape and loading costs remain in the receipts but
share caches, so they are not independent cold-start comparisons.

The pilot, original16, extra6 and published-factory controls contain 92 B2 DT
calls /184 sample vectors. Every sample's complete vectors match exactly
within its paired process, preserving signed RISE ordering and positive
MAS/needle inputs without repeating metric curves. No quality gain is claimed.
The six extra inputs were frozen before attribution using the original formal
input preparation function and author caches; their generated targets were
not regenerated. They are transfer checks, not new FT paper table estimates.

The measured capture and controller files are copied byte-for-byte into the
published backend. Its own manifest pins them and the factory, and depends on
the previous deferred manifest. The old Qwen3 retention manifest is untouched.
Native source checks remain those of the evaluation environment. A future
native implementation with different mutation semantics needs revalidation.

Evidence: [all controls](../../research/temporary/cause_tolerance_20260909/qwen35_retained_summary.json),
[derivation](../../research/temporary/cause_tolerance_20260909/qwen35_retained_derivation.json),
[source manifest](retained_qwen35_sources.json).
