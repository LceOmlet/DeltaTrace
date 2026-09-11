# Frozen VT and HotpotQA: all seven algorithms

The complete result covers 448 paired inputs: 400 VT cases and 48 HotpotQA cases.
Five baselines were newly run for 2,240 method-case records; DT and FT K3 reuse
their verified vectors. The HotpotQA scope is full-response signed sentence sums
under full body-token prefix budgets. VT keeps generated-answer reconstruction
and raw positive eligible-token ranking. The scope choice is retrospective.

| Method | VT H2-C3 | VT H4-C1 | VT H6-C1 | VT H10-C1 | VT macro | HotpotQA |
| --- | --- | --- | --- | --- | --- | --- |
| DeltaTrace | 85.62% | 71.33% | 51.23% | 32.61% | 60.20% | 72.05% |
| FlashTrace K3 | 72.41% | 68.32% | 51.23% | 32.61% | 56.14% | 66.67% |
| Perturbation (20 segments) | 26.69% | 18.86% | 15.78% | 11.16% | 18.12% | 37.50% |
| REAGENT (20 segments) | 26.66% | 17.58% | 15.95% | 11.17% | 17.84% | 34.72% |
| CLP (20 segments) | 29.10% | 20.53% | 17.06% | 13.12% | 19.95% | 42.88% |
| IFR | 81.16% | 70.09% | 51.23% | 32.61% | 58.77% | 75.87% |
| AttnLRP † | 83.37% | 70.54% | 51.23% | 32.61% | 59.44% | 74.83% |

† AttnLRP includes the documented FP16 zero-ratio repair and lossless saved-tensor
offload. Perturbation/CLP/REAGENT use the author's 20-segment approximation.

Read [full results and technical history](RESULTS.md), [protocol](PROTOCOL.md),
[primary CSV](primary_recall10.csv), [case scores](per_case.csv),
[raw vectors](raw), and [independent verification](verification.json).
