# VT H6/H10: increased retrieval budgets

The user requested larger retrieval budgets after every DT, FT K3, IFR and AttnLRP
case reached the Recall ceiling at 10% on these two tasks. The previously measured
20% results distinguish H6, while H10 still shows ceiling saturation. This extension
therefore reports a common 10%, 20%, 30%, 40% grid for all methods on both tasks.
The 30%/40% grid was fixed before calculating those new results. This is a retrospective
extension, not a new holdout or a newly preregistered primary comparison.

**200 inputs; 7 main methods plus supplementary FT K1; 6,400 score rows; zero new
attribution/model calls.** All rankings use the existing verified attribution vectors.
Only the number of retrieved tokens increases. Target scope, positive-score view,
eligible body-token mask and stable position tie-breaking retain the frozen VT policy.
Gold labels are used only to measure recall and its theoretical ceiling.

The original [seven-method Recall@10% report](../all_baselines_20260910/RESULTS.md)
and its macro average remain the historical fixed-budget results. No macro average
mixing different task budgets is substituted for that Recall@10% column.

## vt_h6_c1: body-token Recall

| Method | 10% budget | 20% budget | 30% budget | 40% budget |
| --- | ---: | ---: | ---: | ---: |
| DeltaTrace | 51.23% | 92.05% | 96.66% | 97.68% |
| FlashTrace K3 | 51.23% | 82.15% | 99.94% | 100.00% |
| Perturbation (20 segments) | 15.78% | 33.57% | 46.05% | 57.15% |
| REAGENT (20 segments) | 15.95% | 32.47% | 45.73% | 57.29% |
| CLP (20 segments) | 17.06% | 33.85% | 46.70% | 57.96% |
| IFR | 51.23% | 83.74% | 99.77% | 100.00% |
| AttnLRP † | 51.23% | 84.57% | 97.26% | 98.70% |
| Theoretical ceiling (case mean) | 51.23% | 99.45% | 100.00% | 100.00% |

Number of cases at their individual budget-dependent Recall ceiling:

| Method | 10% | 20% | 30% | 40% |
| --- | ---: | ---: | ---: | ---: |
| DeltaTrace | 100/100 | 0/100 | 11/100 | 18/100 |
| FlashTrace K3 | 100/100 | 0/100 | 96/100 | 100/100 |
| Perturbation (20 segments) | 0/100 | 0/100 | 0/100 | 0/100 |
| REAGENT (20 segments) | 0/100 | 0/100 | 0/100 | 0/100 |
| CLP (20 segments) | 0/100 | 0/100 | 0/100 | 0/100 |
| IFR | 100/100 | 0/100 | 86/100 | 100/100 |
| AttnLRP † | 100/100 | 0/100 | 18/100 | 41/100 |

Precision (case mean), showing the cost of retrieving more tokens:

| Method | 10% | 20% | 30% | 40% |
| --- | ---: | ---: | ---: | ---: |
| DeltaTrace | 100.00% | 90.70% | 63.58% | 48.27% |
| FlashTrace K3 | 100.00% | 80.94% | 65.73% | 49.41% |
| Perturbation (20 segments) | 30.82% | 33.10% | 30.27% | 28.22% |
| REAGENT (20 segments) | 31.16% | 32.00% | 30.06% | 28.29% |
| CLP (20 segments) | 33.32% | 33.34% | 30.70% | 28.61% |
| IFR | 100.00% | 82.51% | 65.62% | 49.41% |
| AttnLRP † | 100.00% | 83.29% | 63.96% | 48.76% |

## vt_h10_c1: body-token Recall

| Method | 10% budget | 20% budget | 30% budget | 40% budget |
| --- | ---: | ---: | ---: | ---: |
| DeltaTrace | 32.61% | 64.69% | 87.74% | 93.33% |
| FlashTrace K3 | 32.61% | 64.63% | 83.41% | 98.97% |
| Perturbation (20 segments) | 11.16% | 26.25% | 40.70% | 53.07% |
| REAGENT (20 segments) | 11.17% | 26.19% | 42.48% | 53.45% |
| CLP (20 segments) | 13.12% | 27.45% | 44.04% | 53.70% |
| IFR | 32.61% | 64.67% | 82.66% | 94.94% |
| AttnLRP † | 32.61% | 61.86% | 77.33% | 90.90% |
| Theoretical ceiling (case mean) | 32.61% | 64.71% | 96.87% | 100.00% |

Number of cases at their individual budget-dependent Recall ceiling:

| Method | 10% | 20% | 30% | 40% |
| --- | ---: | ---: | ---: | ---: |
| DeltaTrace | 100/100 | 98/100 | 0/100 | 0/100 |
| FlashTrace K3 | 100/100 | 92/100 | 0/100 | 42/100 |
| Perturbation (20 segments) | 0/100 | 0/100 | 0/100 | 0/100 |
| REAGENT (20 segments) | 0/100 | 0/100 | 0/100 | 0/100 |
| CLP (20 segments) | 0/100 | 0/100 | 0/100 | 0/100 |
| IFR | 100/100 | 96/100 | 0/100 | 3/100 |
| AttnLRP † | 100/100 | 0/100 | 0/100 | 0/100 |

Precision (case mean), showing the cost of retrieving more tokens:

| Method | 10% | 20% | 30% | 40% |
| --- | ---: | ---: | ---: | ---: |
| DeltaTrace | 100.00% | 99.97% | 90.55% | 72.33% |
| FlashTrace K3 | 100.00% | 99.88% | 86.06% | 76.70% |
| Perturbation (20 segments) | 34.24% | 40.58% | 41.99% | 41.13% |
| REAGENT (20 segments) | 34.28% | 40.49% | 43.85% | 41.42% |
| CLP (20 segments) | 40.24% | 42.42% | 45.45% | 41.62% |
| IFR | 100.00% | 99.94% | 85.29% | 73.57% |
| AttnLRP † | 100.00% | 95.60% | 79.78% | 70.44% |

## Supplementary FT K1

| Task | 10% | 20% | 30% | 40% |
| --- | ---: | ---: | ---: | ---: |
| vt_h6_c1 | 51.23% | 82.62% | 99.96% | 100.00% |
| vt_h10_c1 | 32.61% | 64.65% | 83.82% | 98.89% |

## Interpretation and verification

Recall ceilings are computed per case as min(1, ceil(f × eligible tokens) / gold tokens),
then averaged. A ceiling count means the selected prefix attains the maximum possible
Recall at that budget; it does not mean Recall is necessarily 100%. At 40%, both tasks
permit 100% Recall in every case, so remaining misses are not forced by token capacity.
Larger budgets can increase Recall while decreasing Precision. These tables are
descriptive; no new significance claim or best-budget selection is made.

† AttnLRP retains the recorded numeric repair and lossless saved-tensor offload.
Perturbation, REAGENT and CLP retain the native 20-source-segment approximation.

The check reconstructs all selected prefixes using an independent Python sort,
recomputes every metric and tie bound, checks nested selections and monotonic Recall,
and exactly reproduces all 3,200 historical 10%/20% metric rows within 1e-12.
Parent inputs and results are bound by hashes, including all 1,000 new-baseline
raw vector files used here and the original DT/FT source archives.

- [Fixed extension protocol](protocol.json)
- [All per-case metrics](per_case.csv)
- [Summary and tie-range diagnostics](summary.csv)
- [Exact retrieved token lists](selections.json)
- [Input/vector provenance](provenance.json)
- [Verification receipt](verification.json)

Reproduce from the repository root with NumPy available:

```bash
python research/temporary/vt_budget_extension_20260911/recompute.py
```
