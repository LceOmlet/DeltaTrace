# All seven algorithms under the frozen VT and HotpotQA policies

Completed **448 paired inputs and 2,240 new baseline method-case results** on Qwen3-8B.
The five newly computed methods are Perturbation, REAGENT, CLP, IFR and AttnLRP.
DT and FT K3 are reused as requested; FT K1 is supplementary. Every main-table
baseline value is recomputed from raw vectors under the fixed policy.

The user selected HotpotQA `full` + `signed_sum` after viewing DT/FT results.
VT retains its preceding `answer_only` + raw positive-token policy. This is a
retrospective protocol choice over the complete released cases, including
development examples; it is not an independent holdout or evidence that the
measurement is neutral. See the [frozen protocol](PROTOCOL.md).

## Primary Recall at 10%

| Method | VT H2-C3 | VT H4-C1 | VT H6-C1 | VT H10-C1 | VT macro | HotpotQA |
| --- | --- | --- | --- | --- | --- | --- |
| DeltaTrace | 85.62% | 71.33% | 51.23% | 32.61% | 60.20% | 72.05% |
| FlashTrace K3 | 72.41% | 68.32% | 51.23% | 32.61% | 56.14% | 66.67% |
| Perturbation (20 segments) | 26.69% | 18.86% | 15.78% | 11.16% | 18.12% | 37.50% |
| REAGENT (20 segments) | 26.66% | 17.58% | 15.95% | 11.17% | 17.84% | 34.72% |
| CLP (20 segments) | 29.10% | 20.53% | 17.06% | 13.12% | 19.95% | 42.88% |
| IFR | 81.16% | 70.09% | 51.23% | 32.61% | 58.77% | 75.87% |
| AttnLRP † | 83.37% | 70.54% | 51.23% | 32.61% | 59.44% | 74.83% |

VT measures eligible **body-token Recall**: remove the cached reasoning prefix,
construct the previously fixed generated-answer input, rank positive source-token
scores, and use ceil(10% × eligible body tokens). The VT macro weights the four
100-case tasks equally. This VT input is not context-preserving attribution of
the answer inside its original reasoning.

HotpotQA measures official **supporting-fact sentence Recall** over 48 cases.
Keep the entire reasoning-and-answer input and explain its full response with
the same saved output weights. Sum signed source scores over each original
body sentence, rank by that sum, and retrieve the longest initial ranking
prefix costing at most ceil(10% × all body tokens). Charge punctuation and
whitespace as well. Stop at the first sentence that does not fit. Titles and
document IDs are provided metadata outside this stated body-only budget.
The two tasks' denominators differ; no VT-plus-Hotpot raw-Recall average is used.

† AttnLRP includes an explicitly documented FP16 zero-division repair and
lossless saved-tensor offload. See the technical record below.

## Fixed family of paired comparisons

| Scope | Comparator | DT − comparator (pp) | Adjusted interval (pp) |
| --- | --- | --- | --- |
| HotpotQA | FlashTrace K3 | +5.38 | [-7.81, +19.47] |
| VT macro | FlashTrace K3 | +4.05 | [+3.56, +4.58] |
| HotpotQA | Perturbation (20 segments) | +34.55 | [+17.82, +51.24] |
| VT macro | Perturbation (20 segments) | +42.07 | [+40.98, +43.28] |
| HotpotQA | REAGENT (20 segments) | +37.33 | [+21.35, +53.30] |
| VT macro | REAGENT (20 segments) | +42.36 | [+41.27, +43.60] |
| HotpotQA | CLP (20 segments) | +29.17 | [+15.45, +42.71] |
| VT macro | CLP (20 segments) | +40.24 | [+39.04, +41.52] |
| HotpotQA | IFR | -3.82 | [-15.13, +8.19] |
| VT macro | IFR | +1.42 | [+0.96, +1.91] |
| HotpotQA | AttnLRP † | -2.78 | [-12.33, +6.97] |
| VT macro | AttnLRP † | +0.76 | [+0.19, +1.34] |

Intervals use 10,000 paired bootstrap draws, seed 73, with Bonferroni confidence
99.583333% across the fixed family of 12 comparisons (DT against six alternatives
on HotpotQA and VT macro). VT resampling is independent within each task before
the equal-task average. Intervals are descriptive for this complete benchmark.

## HotpotQA supporting-fact and budget diagnostics at 10%

| Method | Precision | Recall | F1 | Exact set | Complete support | Token Recall | Tokens used | Unused | Empty selections |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeltaTrace | 60.52% | 72.05% | 61.91% | 6.25% | 39.58% | 70.91% | 111.73 | 18.83 | 0.00% |
| FlashTrace K3 | 67.40% | 66.67% | 63.29% | 10.42% | 31.25% | 68.52% | 105.40 | 25.17 | 0.00% |
| Perturbation (20 segments) | 33.85% | 37.50% | 33.18% | 0.00% | 8.33% | 40.58% | 108.77 | 21.79 | 0.00% |
| REAGENT (20 segments) | 32.40% | 34.72% | 31.31% | 2.08% | 10.42% | 38.05% | 106.65 | 23.92 | 0.00% |
| CLP (20 segments) | 36.50% | 42.88% | 36.69% | 0.00% | 6.25% | 45.07% | 111.46 | 19.10 | 0.00% |
| IFR | 67.88% | 75.87% | 67.99% | 14.58% | 50.00% | 75.99% | 109.65 | 20.92 | 0.00% |
| AttnLRP † | 66.15% | 74.83% | 66.31% | 12.50% | 45.83% | 74.39% | 112.60 | 17.96 | 0.00% |

All values are case means. Exact set requires retrieving exactly the gold fact
set; complete support permits extra sentences. Empty selections occur when the
top-ranked whole sentence exceeds the allowed prefix budget.

## Supplementary budgets

| Task | Method | Recall @5% | Recall @10% | Recall @20% |
| --- | --- | --- | --- | --- |
| VT H2-C3 | DeltaTrace | 57.60% | 85.62% | 94.11% |
| VT H2-C3 | FlashTrace K3 | 56.17% | 72.41% | 91.71% |
| VT H2-C3 | Perturbation (20 segments) | 10.63% | 26.69% | 48.60% |
| VT H2-C3 | REAGENT (20 segments) | 10.20% | 26.66% | 47.82% |
| VT H2-C3 | CLP (20 segments) | 16.32% | 29.10% | 49.60% |
| VT H2-C3 | IFR | 58.94% | 81.16% | 92.10% |
| VT H2-C3 | AttnLRP † | 54.13% | 83.37% | 96.82% |
| VT H4-C1 | DeltaTrace | 35.84% | 71.33% | 97.55% |
| VT H4-C1 | FlashTrace K3 | 35.84% | 68.32% | 97.60% |
| VT H4-C1 | Perturbation (20 segments) | 9.27% | 18.86% | 36.38% |
| VT H4-C1 | REAGENT (20 segments) | 7.50% | 17.58% | 35.01% |
| VT H4-C1 | CLP (20 segments) | 11.72% | 20.53% | 39.21% |
| VT H4-C1 | IFR | 35.84% | 70.09% | 97.88% |
| VT H4-C1 | AttnLRP † | 35.84% | 70.54% | 96.99% |
| VT H6-C1 | DeltaTrace | 25.90% | 51.23% | 92.05% |
| VT H6-C1 | FlashTrace K3 | 25.90% | 51.23% | 82.15% |
| VT H6-C1 | Perturbation (20 segments) | 11.59% | 15.78% | 33.57% |
| VT H6-C1 | REAGENT (20 segments) | 11.51% | 15.95% | 32.47% |
| VT H6-C1 | CLP (20 segments) | 11.69% | 17.06% | 33.85% |
| VT H6-C1 | IFR | 25.90% | 51.23% | 83.74% |
| VT H6-C1 | AttnLRP † | 25.90% | 51.23% | 84.57% |
| VT H10-C1 | DeltaTrace | 16.54% | 32.61% | 64.69% |
| VT H10-C1 | FlashTrace K3 | 16.54% | 32.61% | 64.63% |
| VT H10-C1 | Perturbation (20 segments) | 8.69% | 11.16% | 26.25% |
| VT H10-C1 | REAGENT (20 segments) | 8.54% | 11.17% | 26.19% |
| VT H10-C1 | CLP (20 segments) | 9.57% | 13.12% | 27.45% |
| VT H10-C1 | IFR | 16.54% | 32.61% | 64.67% |
| VT H10-C1 | AttnLRP † | 16.54% | 32.61% | 61.86% |
| HotpotQA | DeltaTrace | 39.93% | 72.05% | 86.11% |
| HotpotQA | FlashTrace K3 | 38.54% | 66.67% | 86.11% |
| HotpotQA | Perturbation (20 segments) | 8.68% | 37.50% | 59.72% |
| HotpotQA | REAGENT (20 segments) | 8.16% | 34.72% | 56.08% |
| HotpotQA | CLP (20 segments) | 15.28% | 42.88% | 73.96% |
| HotpotQA | IFR | 43.58% | 75.87% | 91.84% |
| HotpotQA | AttnLRP † | 38.02% | 74.83% | 92.88% |

HotpotQA's integer sentence budgets use the same ranking:

| Method | Recall top 2 | Recall top 4 | Recall top 8 |
| --- | --- | --- | --- |
| DeltaTrace | 58.85% | 82.47% | 87.85% |
| FlashTrace K3 | 60.59% | 77.60% | 90.80% |
| Perturbation (20 segments) | 21.88% | 48.09% | 64.76% |
| REAGENT (20 segments) | 22.22% | 43.75% | 61.63% |
| CLP (20 segments) | 27.60% | 54.51% | 74.83% |
| IFR | 67.19% | 85.24% | 96.70% |
| AttnLRP † | 60.94% | 87.33% | 97.74% |

## Diagnostic VT sentence-density token ordering

| Method | VT H2-C3 | VT H4-C1 | VT H6-C1 | VT H10-C1 |
| --- | --- | --- | --- | --- |
| DeltaTrace | 100.00% | 71.37% | 51.23% | 32.61% |
| FlashTrace K3 | 100.00% | 71.37% | 51.23% | 32.61% |
| Perturbation (20 segments) | 25.59% | 17.74% | 15.23% | 10.94% |
| REAGENT (20 segments) | 25.59% | 16.37% | 15.56% | 11.16% |
| CLP (20 segments) | 29.07% | 19.58% | 16.28% | 12.93% |
| IFR | 100.00% | 71.37% | 51.23% | 32.61% |
| AttnLRP † | 99.42% | 71.37% | 51.23% | 32.61% |

This unchanged diagnostic ranks eligible tokens using shared newline/punctuation
segment means, then token score and position. Its 10% budget still counts tokens;
these columns do not represent an integer number of complete retrieved sentences.

## Diagnostic positive normalization of baseline output rows

| Method | VT H2-C3 | VT H4-C1 | VT H6-C1 | VT H10-C1 | HotpotQA |
| --- | --- | --- | --- | --- | --- |
| Perturbation (20 segments) | 26.57% | 22.38% | 16.97% | 13.21% | 37.85% |
| REAGENT (20 segments) | 24.77% | 20.62% | 16.78% | 12.51% | 36.11% |
| CLP (20 segments) | 31.32% | 22.25% | 17.31% | 13.47% | 43.23% |
| IFR | 79.02% | 69.79% | 51.23% | 32.61% | 76.74% |
| AttnLRP † | 83.37% | 70.54% | 51.23% | 32.61% | 72.74% |

These values retain the native-style positive row normalization before summing
selected output rows. For AttnLRP this is the positive normalized aggregate.
They do not replace the signed-sum primary table, and no winning view was chosen
after inspecting baseline scores. The complete [summary CSV](summary.csv) contains
all budgets, and [case scores](per_case.csv) and [selections](selections.json) make
the exact retrieval decisions inspectable.

FT K1 supplementary Recall@10%: VT H2-C3 73.33%, VT H4-C1 68.55%, VT H6-C1 51.23%, VT H10-C1 32.61%, HotpotQA 67.53%;
VT macro 56.43%.

## Implementation identity and technical corrections

The author source snapshot is commit `075e7e44ae4d5acd2ed76e0d2aced57107d02736`;
all nine dependencies are copied in [source_snapshot](source_snapshot) and bound
by [source hashes](baseline_source_identity.json). The main protocol was frozen
in commit `2bf98cb` before the new baseline quality scores were examined.

Perturbation, CLP and REAGENT use the authors' published **20-source-segment fast
approximation**, with generation-sentence sink groups. This is not exhaustive
single-token perturbation. Native causal-prefix and source-intervention calls
are retained and checked against the same frozen tokenized response. REAGENT
uses the unchanged Longformer replacement and 4,096-token auxiliary input limit;
the [expected auxiliary model ledger](mlm_identity.json) records revision and full hashes,
and the [uploaded-asset receipt](mlm_verification.json) confirms all five actual files.
The uploaded 597,257,159-byte weight file matched SHA-256
`06c56757f0510de87c231acd03650ca204c5ed65c281cd1081b98288183f2468`.

IFR uses the native all-position primitive with chunk sizes 128/32. AttnLRP uses
the native generated-logit weighted aggregate, with normalization disabled and
the same saved 0/1 output weights. The other methods' raw matrices are summed
over those selected output rows. The older result wrapper is bypassed because
it clips signs, normalizes rows and can restrict the sink to the cached answer.
Attribution quantities still differ: perturbation log-probability differences,
the author's KL-like CLP score, IFR proximity, AttnLRP logits and DT's finite
log-probability contrast are not numerically equivalent quantities.

The first AttnLRP pilot failed because its FP16 `output/(input+1e-10)` saved ratio
became 0/0 at zero activations. The [numeric amendment](NUMERIC_FIX.md), frozen
in commit `6f967e4`, re-evaluates only those exact zero/zero ratios with the
existing epsilon in FP32. Forward outputs and all finite ratios are untouched;
other nonfinite cases still fail. The complete run records 5,625 repaired
ratios. Original author source files are unchanged, while the runtime adapter
is explicitly amended and labeled †.

HotpotQA then exceeded 64 GiB when retaining the full AttnLRP backward graph.
The [storage amendment](STORAGE_FIX.md) offloads saved activations while preserving
their entire storage, shape, strides and offsets, retaining model parameters on
the GPU. Generic contiguous offload failed the bitwise control and was rejected.
The accepted version reproduced all six stored arrays bit for bit; its input,
target weights and attribution values matched the existing VT H2-C3 pilot.
See the [control receipt](storage_control/verification.json).

All 23 successful pre-offload pilots remain immutable and are accepted only by
their exact registered result hashes in [execution compatibility](execution_compatibility.json).
The three first-pilot Perturbation/CLP/IFR results also match their repeated pilot
vectors bit for bit. Original numerical failures, OOM and the rejected storage
control are retained. DT/FT attribution is never rerun in this experiment.

## Measured run costs

| Method | Cases | Measured operation time (h) | Median time/case (s) | Max PyTorch allocation (GiB) | Qwen forwards |
| --- | --- | --- | --- | --- | --- |
| Perturbation (20 segments) | 448 | 1.055 | 7.421 | 19.474 | 19748 |
| REAGENT (20 segments) | 448 | 1.488 | 10.507 | 19.474 | 19748 |
| CLP (20 segments) | 448 | 1.064 | 7.478 | 19.942 | 19748 |
| IFR | 448 | 0.912 | 6.715 | 48.934 | 448 |
| AttnLRP † | 448 | 0.160 | 1.006 | 28.378 | 448 |

Total successful attribution-operation time is 4.679 hours.
The accepted controller's elapsed wall time, including its pilot-resume and full
phases, is 4.800 hours. Its wall interval excludes earlier pilot
attempts and storage controls, whereas the operation total above includes all
successful records, including reused pilots. The two totals cover different
execution intervals and should not be subtracted as an overhead estimate.
These are the observed execution costs of this implementation, including CPU
transfers for offloaded AttnLRP and the auxiliary-model work inside REAGENT.
Peak allocations include models already resident in the shared process. Model
startup, failed attempts, controls, orchestration, transfer and CPU evaluation
are additional. This table is not a separately standardized efficiency benchmark;
the pre-offload VT pilots and subsequent offloaded cases use different storage.

## CPU scoring precision audit

The final independent verifier detected an unintended float32 cast of the reused
HotpotQA DT vector in the first combined CPU scoring draft. The scorer now
preserves native v3 precision before float64 sentence sums; VT retains its fixed
float32 token view. The initial draft and its hashes are preserved. This correction
changes 288 DT sentence-score arrays, while all 34,944 per-case metric rows,
390 summary rows and the primary CSV remain byte-identical. Every ranking order
and selected set is unchanged, and all 864 reused HotpotQA ranking-score records
now match v3 exactly. No attribution is rerun or changed. See the
[correction record](SCORING_PRECISION_FIX.md) and
[precision control](scoring_precision_control/verification.json).

## Artifacts and reproduction

- [Protocol JSON](protocol.json), [input audit](input_audit.json), [all-input CPU preflight](inputs_preflight.json).
- [Analysis and hashes](analysis.json), [full primary CSV](primary_recall10.csv), [raw per-case records](raw).
- [Independent verification](verification.json), [execution controller record](audit/all_baselines_logs_v3/controller.json).
- [Frozen previous VT/DT/FT run](../source_v2_gpu_20260910/full_recall/RESULTS.md), [native HotpotQA v3 evaluation](../hotpot_context_v3_20260910/RESULTS.md).

From the repository root, with NumPy and the recorded dependencies installed:

```bash
python research/temporary/all_baselines_20260910/analyze_all.py --run research/temporary/all_baselines_20260910/raw
python research/temporary/all_baselines_20260910/build_report.py
python research/temporary/all_baselines_20260910/verify_outputs.py
```

Attribution runs require the frozen author environment and checkpoint receipt:

```bash
python research/temporary/all_baselines_20260910/run_controller.py --environment /path/environment.json --preflight /path/inputs_preflight.json --output /path/raw --logs /path/new_logs --methods Perturbation CLP IFR AttnLRP REAGENT --mlm /path/longformer-base-4096
```

The driver refuses unregistered completed records, altered inputs, mismatched
target weights, changed source/checkpoint files, incomplete vectors and any new
generation call. Inputs passed to the GPU runner do not contain retrieval gold.
The independent verifier reconstructs rankings, whole-sentence prefixes, all-token
costs, official fact sets, paired intervals and the unchanged DT/FT evidence.
