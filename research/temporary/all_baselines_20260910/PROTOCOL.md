# Frozen VT and HotpotQA comparison for all seven algorithms

The user accepted the full-response signed-sum HotpotQA view after seeing DT/FT
results and requested freezing that view together with the existing VT policy.
This is an explicitly retrospective choice, not a newly preregistered holdout.
Freeze the remaining baseline settings before examining their new quality.
Do not change the frozen v3 or previous VT artifacts, methods, or labels.

## Scope and reuse

Qwen3-8B, the same checkpoint and tokenizer, all 48 released HotpotQA cases and
all 100 cases in each of VT H2-C3, H4-C1, H6-C1, H10-C1: 448 inputs in total.
Reuse the already verified DT, FT K3 and supplementary FT K1 vectors and scores;
the user explicitly exempted them from rerunning. Newly execute Perturbation,
REAGENT, CLP, IFR and AttnLRP for all 448 inputs (2,240 method-case results).
The current scope is Recall; RISE/MAS deletion sweeps are a separate scope.
Do not copy published baseline aggregate CSV numbers into the new table.

## Input and output scope

Every baseline must construct exactly the tokenized input stored in its paired
DT/FT case. HotpotQA retains the complete cached reasoning and answer response.
VT retains the previously fixed answer-reconstruction condition: extract the
generated answer using the cached token span, replace the response with that
substring, and construct the input exactly as the old VT run did. This removes
the original reasoning prefix; do not call it context-preserving attribution.

Use the same full remaining response target positions as DT/FT, excluding
terminal EOS and the existing comma, period and pure-whitespace output filter.
The 0/1 target weights are saved per input. Never substitute the cached final
answer span for the full-response HotpotQA target. Do not read gold in the GPU
attribution runner. Native perturbation forwards may evaluate causal prefixes
and their defined interventions; verify that unperturbed prefixes and fixed
response IDs match the corresponding complete input exactly.

## Fixed retrieval policy

HotpotQA: the v3 `full` + `signed_sum` setting. Candidates are all original native
body sentences. Sum each method's signed source-token scores over every token
assigned to that sentence. Pay all assigned body tokens, including punctuation
and whitespace. At 5%, 10%, 20% allow ceil(f*N_all_body) and retrieve the longest
affordable initial ranking prefix, stopping at the first non-fitting sentence.
Official supporting-fact IDs and v3 corrected coordinates stay fixed for all
48 cases; no semantic relabeling or case exclusion. Titles/IDs are provided
metadata outside this explicitly body-only budget. Keep source-order ties.
Primary score: official supporting-fact Recall at 10%; also P/R/F1/EM, complete
support, token Recall, actual cost, unused cost, empty selection and top 2/4/8.

VT: the previously frozen `answer_only` + raw positive-token ranking setting.
The candidate body span, eligible-token filter, gold, position-order ties and
ceil(f*N_eligible) budgets remain identical to the saved DT/FT run. Primary is
token Recall at 10%, with 5/20% and the unchanged shared sentence-density token
ranking reported as diagnostics. This remains an eligible-token budget, not
HotpotQA's full-token whole-sentence budget. Do not combine their raw Recall
values into one cross-task metric. Report all four VT tasks and their equal-task
macro mean. The old sentence-density view is a token ordering, not an integer
number of retrieved complete sentences.

## Baseline implementations and aggregation

Use the authors' primitives from commit
`075e7e44ae4d5acd2ed76e0d2aced57107d02736`, with source and checkpoint hashes.
Perturbation, REAGENT and CLP retain the published fast approximation: 20 source
segments over user prompt plus generation, original generation-sentence sink
groups, and original per-token log-loss or KL-like differences. REAGENT keeps
the original Longformer replacement rule; its auxiliary model revision and
weight hashes are recorded. The 20-segment approximation is stated in reports.

IFR uses `calculate_ifr_for_all_positions`, original proximity rules,
chunk_tokens=128 and sink_chunk_tokens=32. AttnLRP uses the author's raw
`calculate_attnlrp_span_aggregate` with generated-token logits and the exact
target weights, normalize_weights=False; its LRP backward patches are restored
after every call. No multi-hop AttnLRP variant is substituted.

For per-output matrices (the perturbation methods and IFR), sum original rows
with the shared 0/1 target weights to obtain the prompt vector. For AttnLRP,
use the weighted aggregate returned by the native primitive. Preserve negative
values before the fixed HotpotQA sentence sum. Do not silently call the older
`get_all_token_attrs` Row result: it clamps negatives, normalizes every output
row and may select only the cached answer span. Save raw matrices/vectors plus
the positive row-normalized diagnostic view to make that distinction auditable.
Attribution mechanisms still differ: perturbation log-probability differences,
the author's KL-like CLP score, IFR proximity, AttnLRP generated logits and DT
finite log-probability contrast are not the same numerical quantity. Common
input, output scope and retrieval rules do not equate these quantities.

## Verification and reporting

Before quality scoring, verify all 448 input identities and the same frozen
target weights. A technical pilot uses index 0 from each task and is retained;
it tests shapes, finite values, complete target coverage, causal-prefix identity,
and native model restoration. It is not used to choose hyperparameters. No
DT/FT attribution is rerun. Preserve every successful result and all failures.
The full controller checkpoints per case, and resumes only completed records
with exact input, code, protocol and vector hashes.

Recompute every score and selection from saved raw vectors independently. Check
all budgets and HotpotQA nested sets, reproduce the reused DT/FT tables exactly,
and report all 2,240 new method-case results, failures and wall/operation costs.
All primary methods appear in one table; no winner-based target/view selection.
DT minus each of the other six methods on HotpotQA and on the VT macro is a
fixed family of 12 descriptive paired comparisons. Use 10,000 bootstrap draws,
seed 73, Bonferroni 99.583333% intervals; independently resample within each VT
task before averaging task means. Other intervals are descriptive diagnostics.
Neither the user's choice nor a method's win establishes measurement neutrality.

