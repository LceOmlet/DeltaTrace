# HotpotQA evidence-evaluation corrections

The user requires demonstrated unfairness or implementation errors to be
identified and corrected, rather than a particular method being made to win.
Keep all 48 released cases and both methods' negative as well as positive results.
This is a retrospective correction audit, not a new holdout experiment.

Confirmed issues before scoring this protocol:

1. In case 8, the cached span for native supporting sentence 0 of
   "Why Is There Air?" points to the document title instead of its body occurrence.
   Restore coordinates by locating the complete source document and cumulatively
   indexing its original sentence array. Matching text or sentence IDs alone is
   insufficient to validate coordinates.
2. The old regular-expression pool assigns tokens using their offset start.
   A token's leading space can put its first word in the previous sentence.
   Use the first non-whitespace character and native sentence boundaries.
3. Token Recall weights each supporting fact in proportion to its token count.
   It is not the native supporting-fact metric. Add equal-fact precision, Recall,
   F1, exact set match, and complete-support hit rate, with explicit retrieval cost.
4. Full-response attribution and final-answer attribution are different questions.
   Report both fixed objectives for all 48, without selecting the better outcome.
   The answer-only input is constructed from the cached generated answer span,
   before attribution, identically for DT and live FT K1/K3. No reference answer
   or gold label is read by this extraction. Retain the same original EOS reference,
   model weights, native content-P1 propagation, and non-stop/non-EOS seed policy.

Build every candidate sentence from the original context arrays, not just gold
sentences. All 480 complete context documents must uniquely match the actual
prompts. Build and hash a source-only candidate sidecar first; supporting-fact
keys enter only the scoring sidecar. Preserve sentence IDs, even when the native
dataset has unusual punctuation splits. Titles and document numbering remain in
the model input; native supporting-fact retrieval ranks sentence bodies only.

Pool the same positive-token arithmetic mean for DT and FT. Do not try alternative
pooling powers or choose a rule from corrected scores. Under token budgets of
5%, 10%, and 20% of eligible sentence-body tokens, visit the ranked sentences in
order and retrieve a whole sentence only if it fits the remaining budget. Charge
all eligible tokens of every selected sentence; never fill a partial sentence or
grant free expansion. Ties follow source position. Report actual spent tokens,
unused budget, equal-fact Recall ceilings, and token Recall alongside fact scores.
As a separate sentence-cost view report top 2, 4, and 8 sentences and their actual
token costs. Never equate the two budget units.

Isolate fixes using the saved full-response vectors: legacy raw and regex pooling
with cached gold; raw with restored official coordinates; legacy pooling with
restored coordinates; whitespace-corrected regex pooling with restored coordinates;
and native sentence pooling. These diagnostic token views retain the same old
source candidate set and token budgets, with metadata as separate non-fact units.
The native whole-sentence comparison uses only sentence-body candidates for both.

Keep restored official supporting-fact IDs as the principal source-label view.
Review previously identified missing-relation labels in cases 36 and 40 against
the actual questions and complete supporting documents. If confirmed, report an
explicit revised-label sensitivity view for both methods, retaining original IDs.
Flag question/evidence ambiguities in cases 15, 16, and 30 and report the same
45-case sensitivity subset for both, without removing them from the full table.
Do not claim that a single-reviewer revision is official ground truth or that
agreement with one annotated proof establishes causal faithfulness.

Use paired 10,000-draw bootstrap intervals (seed 73), descriptively. Native
sentence Recall@10%-token-budget under each of the two target objectives and
the two label views gives four comparisons; report Bonferroni 98.75% intervals.
Other metrics and budgets are diagnostics. Recompute all scores from original
vectors and verify code/input/weights/reference identities; previously computed
answer-only development overlaps must reproduce bitwise. Preserve the frozen
old implementation and its full results so they remain reproducible. The corrected
entry point must explicitly select this new protocol instead of silently changing
the old scorer.
