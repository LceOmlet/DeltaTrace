# HotpotQA v3: preserve context and make retrieval cost literal

This continues the user's requested repair after the methodological review of
v2. Confirmed mapping fixes remain: restore official supporting-fact body
coordinates and assign leading-space tokens by the first content character.
Do not call this protocol intrinsically neutral or treat an observed method
win as its acceptance criterion. Freeze these choices before computing v3 scores.

## Attribution condition

Use every one of the 48 released cases. Keep the original complete tokenized
model input, including the entire cached reasoning prefix and final answer.
The primary new run is `answer_conditioned`: only the initial output seed is
masked to the original final-answer token span, with the existing common
comma/period/whitespace and EOS exclusions. Do not replace `ex.target`.
For each case require the complete input hash and full-prompt EOS reference hash
to match the previously verified full-response run. Also compare the saved
full-response objective; this uses exactly the same inputs and reference.

Use the unchanged weighted DT propagation and existing InitialTargetFT adapter.
FT's initial seed weights must equal DT's actual target weights. All subsequent
FT hops and observation support retain the complete original generation span.
The 8 earlier answer-conditioned development cases must reproduce DT, FT K1 and
FT K3 vectors bitwise. Keep all 48 results and all failures; do not select a
target by its new score. The removed-prefix v2 experiment stays historical.

## Source units and gold

Use all original native HotpotQA context sentence arrays, mapped through whole
documents. Candidate construction depends only on prompt, source context and
token offsets. All 480 documents must uniquely match. Do not infer candidate
boundaries from gold, generated answers or attribution scores.

Keep the official supporting-fact IDs for every case. Restore their coordinates,
including case 8's title/body error. Do not use the two assistant label revisions
or the 45-case subset as v3 primary results; those remain v2 sensitivity records.
Preserve native IDs and the three original indivisible BPE boundary crossings.
Assign each such token to its first non-whitespace character's unit once, with
its full token cost. Record that this is native-unit token assignment rather
than character-perfect independent sentence tokenization.

## Cost and nested selection

Sentence cost is the number of ALL original tokenizer tokens assigned to the
sentence, including comma, period and whitespace tokens. It does not depend on
the method's old stop-token filter. The budget denominator is the total number
of these native sentence-body tokens. Titles/document markers remain in the
model input and are provided document metadata, outside this explicitly body-
token budget; this is not a budget for the entire model input or serialized IDs.

For f = 5%, 10%, 20%, allow ceil(f * N_all_body) tokens. Retrieve the longest
initial prefix of the fixed sentence ranking whose cumulative full cost fits.
STOP at the first sentence that does not fit. Do not skip it to fill space with
lower-ranked short sentences, split it or expand a partially paid sentence.
Record unused budget, including empty selections when the first sentence does
not fit. Rankings are budget-independent, so selections must be nested and
official fact Recall nondecreasing as the token budget increases.

Also report top 2, 4 and 8 native sentences with their full body-token costs.
This is a distinct cost unit; never equate a sentence budget with a token budget.

## Two fixed attribution-to-sentence views

Report BOTH views for BOTH objectives and DT / live FT K1 / live FT K3:

1. `signed_sum`: sum the original signed token attributions over all tokens
   assigned to the native sentence. Retain negative contributions; do not clip
   each token first. Rank by this sum, with source-order ties. This is the
   additive contribution to the method's fixed attribution contrast, not a
   proof that supporting-fact labels capture causal faithfulness.
2. `positive_mean_eligible`: the previous v2 diagnostic, arithmetic mean of
   positive attributions over the original eligible tokens in each native
   sentence. Preserve its definition to expose sensitivity to aggregation.
   Retrieval nevertheless pays the same new ALL-token cost as the sum view.

Do not choose a winning view from results. Shared transformations do not prove
measurement neutrality. Verify whether tokens excluded by the old filter have
zero saved source attribution; full-token cost applies even when they do.

## Reporting and verification

The fixed principal table is official supporting-fact Recall at 10% all-body-
token budget for two objectives x two pooling views. Four DT-minus-FT-K3 paired
comparisons use descriptive Bonferroni 98.75% bootstrap intervals (10,000 draws,
seed 73); FT K1, other budgets, P/R/F1/EM, complete support, selected sentence
counts and full token costs are diagnostics. This is the repeatedly inspected
48-case released cache, not an independent holdout or a guarantee of neutrality.

Retain the unchanged full-response legacy token-recall views with coordinate /
leading-space fixes to isolate known bugs. For the positive-mean view separately
show old eligible-cost skip packing, all-token skip packing, and all-token prefix
selection on the same saved vectors. This separates a cost correction from a
selection-policy change; the two skip policies are diagnostics, not v3 outputs.

Verify every new input, target mask, FT initial aggregation and later hop span,
reference, raw vector and legacy metric. Independently recompute all selected
sets, full costs and official fact scores. Test the former non-nested packing
counterexample, punctuation costs, full-input target masking and label invariance.
Keep prior frozen protocols, snapshots and 27 core method files unchanged.
