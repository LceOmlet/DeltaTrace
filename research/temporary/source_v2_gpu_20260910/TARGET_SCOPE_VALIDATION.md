# Reserved validation of task-specific target repair and Recall scope

The first 80-case shared sentence-ranking validation failed. Subsequent
development tests of answer targets, EOS/space references, four pooling rules
and layerwise symmetric attention did not establish a shared advantage under
one universal target. Preserve those outcomes and their original criteria.
This experiment does not relabel any of those failed gates as passed.

The remaining question is narrower: does the demonstrated target repair give
DT a reproducible Recall advantage in a clearly specified task and ranking
view, and where does that advantage end? Development evidence shows that
multi-chain VT's full response credits unqueried chains. On the same eight
development cases, answer-only raw token Recall is DT 84.35% versus FT 74.80%;
shared sentence mean reaches 100% for both. For HotpotQA, full responses give
the highest sentence-mean Recall for both DT and FT among the three tested
target modes (82.05% versus 80.16%). These are selection observations, not
validation claims. HotpotQA's bridge-context explanation remains a hypothesis.

Freeze a task-specific target policy now: all four VT tasks use answer_only;
HotpotQA uses full. Both DT and FT receive the identical selected input and
non-stop, non-EOS target weights. The final-answer substring comes only from
the already stored generated target's token span and tokenizer offsets. No
gold, reference answer or metadata output enters extraction or ranking. Task
identity is used explicitly; this is not a universal input-independent method.

Use original content-P1 DT and its original full author-eligible EOS reference.
Use live FT K1/K3, with K3 as primary comparator. Keep the same evidence-body
candidate mask and exactly ceil(0.10*N_source) source tokens. Report both
positive raw token ranking and the identical sentence-mean ranking for each
method. Mean pooling was independently best for each method in development;
do not substitute RMS or max to weaken FT. Do not add another reference or
blend, target ensemble, task-specific pooling, or layer rule.

Validation sample: the second reserved split already fixed in answer_split.json,
16 cases in each of four VT tasks and HotpotQA, 80 total. It is disjoint from
development and the first 80-case validation. Candidate outcomes on these
cases have not been inspected. The original benchmark baseline was audited
earlier, so this is an internal reserved validation, not an external dataset.
All 80 cases must finish before candidate quality is examined.

Four prespecified primary comparisons at Recall@10%: DT minus FT K3 for
VT macro/raw, VT macro/sentence mean, HotpotQA/raw, and HotpotQA/sentence mean.
VT macro weights its four tasks equally. Use paired example resampling within
each task, 10,000 replicates, fixed task-specific seed=73 streams. For each
primary comparison report ordinary 95% and Bonferroni-adjusted 98.75% percentile
bootstrap intervals (quantiles 0.00625 and 0.99375; nominal family 95% coverage
for the four comparisons). Call an advantage confirmed only in a prespecified
comparison whose adjusted lower bound is strictly positive. A negative adjusted
upper bound confirms a disadvantage. Exact joint ceiling ties are parity.

Report every task and both views, all signs and failures. Individual VT tasks
and 5%/20% budgets are secondary and are not substitute primary win criteria.
Do not claim a shared VT/HotpotQA advantage unless both groups show confirmed
improvement in the same view. A scoped VT result does not prove a HotpotQA win;
a raw-ranking advantage does not imply beating FT with sentence aggregation.
Preserve the distinction between signed attribution and a retrieval ranking.

After this reserved comparison, no new candidate will be selected on these
80 validation outcomes. Archive original inputs, scores, target masks, signed
vectors, references, native input checks and actual model-operation costs.
