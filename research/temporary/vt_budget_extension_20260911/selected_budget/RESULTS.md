# Current comparison with user-selected task budgets

**VT budgets: H2-C3 10%, H4-C1 10%, H6-C1 20%, H10-C1 30%. HotpotQA remains 10%.**

The user selected H6=20% and H10=30% after seeing the complete budget grid.
This is a retrospective reporting choice. It is not an independent holdout
or a budget choice fixed before observing results. All algorithms use the same
budget within each task. The complete [10%–40% grid](../RESULTS.md) remains available.

VT reports eligible body-token Recall using its saved answer-only input and positive
token ranking. HotpotQA reports official supporting-fact Recall using full-response
signed sentence sums and the native all-body-token prefix budget.

| Method | H2-C3 @10% | H4-C1 @10% | H6-C1 @20% | H10-C1 @30% | VT macro (task-specific budgets) | HotpotQA @10% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeltaTrace | 85.62% | 71.33% | 92.05% | 87.74% | 84.18% | 72.05% |
| FlashTrace K3 | 72.41% | 68.32% | 82.15% | 83.41% | 76.57% | 66.67% |
| Perturbation | 26.69% | 18.86% | 33.57% | 40.70% | 29.96% | 37.50% |
| REAGENT | 26.66% | 17.58% | 32.47% | 42.48% | 29.80% | 34.72% |
| CLP | 29.10% | 20.53% | 33.85% | 44.04% | 31.88% | 42.88% |
| IFR | 81.16% | 70.09% | 83.74% | 82.66% | 79.41% | 75.87% |
| AttnLRP † | 83.37% | 70.54% | 84.57% | 77.33% | 78.95% | 74.83% |

The VT macro is the equal mean of the four task recalls at their stated budgets.
It is labeled **task-specific-budget macro**, not Recall@10%. It does not include
HotpotQA, whose recall denominator is different. The old all-10% table and its
[historical intervals](../../all_baselines_20260910/RESULTS.md) remain archived.
These selected-budget tables are descriptive and make no new significance claim.

## Provenance

All 3,584 selected per-case rows (448 inputs × 8 methods, including supplementary
FT K1) are copied from verified scores. No attribution or model computation is
rerun. Each of the 40 task/method means is checked against its source summary.

† AttnLRP retains its documented numeric repair and lossless saved-tensor offload.
Perturbation, REAGENT and CLP retain the author’s 20-source-segment approximation.

- [Selected policy](protocol.json)
- [Primary CSV, including supplementary FT K1](primary.csv)
- [Selected per-case metrics](per_case.csv)
- [Verification](verification.json)

Reproduce from the repository root:

```bash
python research/temporary/vt_budget_extension_20260911/build_selected_budget.py
```
