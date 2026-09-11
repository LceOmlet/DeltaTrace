# Complete VT and HotpotQA Recall table

The user requested completion of the full evaluation and table after reviewing
the fixed target construction and the 80-case reserved comparison.

Keep the exact scope_choice.json policy and attribution computation: all four
VT tasks use only the stored generated final-answer substring; HotpotQA uses
the complete response. DT and live FT K1/K3 see the same selected native input
and non-stop, non-EOS target weights. DT uses original content-P1 rules and the
original full author-eligible EOS reference. Gold and source candidates remain
unchanged. Recall ranks positive raw tokens or identical sentence-mean scores
with exactly ceil(f*N_source) tokens, for f=5%,10%,20%.

Evaluate every released case: VT H2-C3, H4-C1, H6-C1 and H10-C1 each have 100;
HotpotQA has 48, giving 448 unique cases. Reuse the already independently
verified 80-case target_scope_v1 artifact byte-for-byte. Compute all remaining
368 cases, with no selection by quality. Do not reuse other experimental
target/reference variants. The full table includes development and validation
examples, so it is a complete benchmark description, not a new independent
holdout claim. Preserve the earlier reserved validation and every failed trial.

Use deterministic shards of at most 16 new cases to bound checkpoint size.
The first shard of each task additionally reruns its lowest-index reused case.
Require bitwise equality of DT_target, FT K1 and FT K3 vectors and identical
input/target metadata for these five overlap controls. Charge their operations
but count each benchmark case once. The same method, model and tokenization
are used in all shards. Splitting the schedule must not change the method.

Verify each completed shard's code, plan, model/reference/source and vector
identity before continuing. Resume only from complete verified shards. Preserve
failed or interrupted shards rather than treating them as complete data. The
controller writes progress separately; a full table requires exact coverage
of all indices and no missing, duplicate or conflicting cases.

Fill the final table with per-task sample counts, DT and FT K3 Recall@10% for
both views, differences and budget ceilings. Include FT K1 and 5%/20% in the
supplementary table. Report equal-task VT and five-task macros, HotpotQA, paired
95% intervals and the same four-comparison Bonferroni 98.75% intervals as the
reserved study. Interpret full-data intervals descriptively because some cases
were used for development. Retain positive, zero and negative differences.
Do not retune any setting from these complete-data results.
