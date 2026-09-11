# Recall-focused pilot, fixed before new candidate measurements

The user narrowed the objective to VT and HotpotQA Recall and requested small
experiments instead of waiting for the full sweep. The sweep is stopped; all
completed data and interrupted prefixes are retained. RISE/MAS are outside this
pilot's decision criterion.

Primary metric: evidence-body eligible-token Recall at exactly ceil(0.10*N)
tokens. The same source, gold denominator, token budget, input, target, model,
and tie rule apply to every method. FT K=3 is the primary comparator; K=1 is
reported as a secondary comparator. No candidate may use gold in its selector.

Two mechanistic hypotheses are tested:

1. Sparse token allocations poorly recover a sentence-level annotation. Use
   the already defined sentence-density token ordering, applied identically to
   DT and FT. This still retrieves exactly the token budget; it is not the
   different whole-sentence-unit Recall metric.
2. The original attention product rule assigns the P/V interaction to one
   endpoint. Compare its forward allocation with the negated allocation after
   exchanging the two native endpoints. Their mean is a symmetric interaction
   allocation. Native model code and the frozen finite kernel are unchanged;
   these are explicitly new attribution variants. Keep both full-prompt-EOS
   and evidence-body-EOS references in the 2-by-2 control.

For each reference, retain forward, reversed-negated, and their equal-weight
mean. Retain both raw-token and sentence-density orders. This yields 12 DT
candidate settings, and two shared ordering settings for each FT hop count.
There is no weight or temperature search. We select one shared reference,
direction setting, and ordering using the development data only. Report all
development candidates, then freeze the selected setting before validation.
Selection maximizes the smaller of the VT-H4-C1 and HotpotQA mean differences
from FT K=3 under the same ordering; ties use the equal-task mean, then the
lexicographic setting name. If all settings fail, retain that failure and
form a new mechanistic hypothesis; do not repeatedly sample validation.

The split is determined by ascending SHA-256 of
`recall-pilot-v1|dataset|index`, independently within each released task.
Development: first 8 cases of VT-H4-C1 and HotpotQA (16 total).
Validation: next 16 cases in each of all four VT tasks and HotpotQA (80 total).
The first 8 cases in other VT tasks are unused. Record all indices before
evaluating any candidate. Validation labels will be read only after the
candidate is frozen. These are reserved cases from previously audited caches,
not a new external dataset; the old baseline statistics have already been seen.

Report all five validation tasks and an equal-task macro mean, paired bootstrap
95% intervals (10,000 draws, seed 73, fixed independent task streams), and 5/20%
budget sensitivity. A claim covering both VT and HotpotQA requires a positive
mean in each and a macro interval excluding zero. A subgroup-only advantage is
labelled as such. Count all four DT pilot passes and both live FT passes; no
claim of equal compute. Preserve negative candidates and all run costs.
