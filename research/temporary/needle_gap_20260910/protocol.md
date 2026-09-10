# Needle gap audit — fixed analysis plan

Base: main `2f80c06`; isolated branch `codex/needle-gap-audit`.
Quality inputs: frozen Qwen3-8B table1 run `9c6497c`, all 1,048 cases in
11 tasks with released recovery labels. MATH/MoreHopQA have no such labels.

1. Verify cache, DT result/vector hashes, paired FT prompt/target identities,
   token maps, and reproduce released recovery within exact top-k tie bounds.
2. Quantify eligible token count N, eligible gold count G, k=ceil(0.1N),
   ceiling min(1,k/G), recall/ceiling, random expectation, and recall-budget curves.
3. Test whether clipping negative values or boundary token mismatch explains
   the gap, and report conservation residuals as diagnostics (not a proof of
   ranking correctness).
4. Reconstruct gold offsets from the recorded tokenizer, inspect gold subtypes
   (answer text versus remaining span), and attribute selected non-gold tokens
   to demonstration/query/other regions where those boundaries exist.
5. Before inspecting pooled results, fix two retrieval-view candidates:
   sentence positive density and sentence absolute density. Split text only
   at newline or sentence-ending punctuation followed by whitespace. Pool the
   mean score across eligible tokens in each segment and rank tokens by segment
   density, then original token score, then position. No dataset-specific rules,
   gold spans, target answers, or fitted parameters enter either selector.
6. Apply the same positive-density transformation to FT. Include a deterministic
   shuffled-score control. Report every task, wins/losses, and separate even/odd
   index subsets. This is retrospective validation on an existing benchmark,
   not a genuinely unseen held-out generalization claim.

Frozen attribution and official RISE/MAS/Recall@10% are never overwritten.
Pooling is a distinct retrieval ranking: it does not inherit DT's signed
conservation meaning and requires new model deletion calls before any claim
of improved faithfulness. Hypotheses about different targets/reference inputs
require future model ablations if a compatible accelerator is unavailable.

## Follow-up after the first fixed probes

The first pass found 41–52% of DT's VT top-k in the solved demonstration,
versus 19–28% for FT. A second, explicitly retrospective rank intervention
removes the prefix before the second task instruction. Its boundary comes
from the prompt's repeated instruction string, never needle labels. Both DT
and FT retain the original k; a separately reported control recomputes k at
10% of the narrower scope. These are distinct budgets and must not be mixed.
Combine the source restriction with the already fixed sentence pooling and
report both even/odd halves and all individual cases. This cannot establish
that a model run with a different reference input would produce these scores.
