# Deferred diagnostic backends

These are explicit acceleration options for the frozen finite method. The clean
backend remains the default, and the previous `accelerated_qwen35` remains pinned.
The Qwen3 paper tag and its results are unchanged.

Use the existing formal evaluator with `--dt-backend deferred_qwen3 --sample-batch 1`
or `--dt-backend deferred_qwen35 --sample-batch 2`, with the matching `--family`.
All other dataset, trajectory, metric and FT-reference options retain their meaning.
Qwen3 true sample batching is not implemented by this change; its endpoint batch is 2.
Qwen3.5 sample batch 2 means endpoint batch 4, using existing native padded FA/FLA.

Qwen3 collects the same 828 GPU predicates and 36 scalar diagnostics, checks them
before returning, and retains failure on invalid values. Qwen3.5 removes per-stage
device barriers and defers scalar diagnostics/32 finite checks. Stage timings are
now separately named `host_enqueue_seconds` and `stream_elapsed_seconds`; they
must not be confused with synchronized wall time. Full API time remains measured.

The original model forwards, public FA calls, finite attention extension, FLA and
finite arithmetic are unchanged. There is no framework patch or fallback. The
Qwen3 helper files and Qwen3.5 controller are copies of our measured DT plumbing,
with source-bound derivation records under the temporary research directory;
they do not implement a model or native attention/backward replacement.

The original 16 author cases were checked in both models. Qwen3: 96 complete
vectors match exactly, 10.960→10.311 seconds per warmed pass (5.9% reduction),
18.7608→18.7613 GB peak. Qwen3.5: 48 B2 calls /96 sample vectors match exactly,
12.555→11.573 seconds (7.8% reduction), 23.6281→23.6280 GB peak. Each mode has
separate shape warmup and two interleaved measured passes; Qwen3.5 has no new
Dynamo graphs in measured calls. All full-call diagnostics are included.

These improvements are relative to each same-process baseline. Qwen3.5's baseline
already includes the previous batching/dynamic acceleration. Do not multiply this
reduction by a historical run and present it as one measured speedup. Exact vector
equality preserves signed RISE ordering and positive MAS/needle inputs in these
controls; it is not a general requirement on future tolerance optimizations.

Evidence: [16-case Qwen3](../../research/temporary/cause_tolerance_20260909/qwen3_deferred16/summary.json),
[B2 Qwen3.5](../../research/temporary/cause_tolerance_20260909/qwen35_deferred16/summary.json),
[source manifest](deferred_sources.json).
