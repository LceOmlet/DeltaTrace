# Qwen3.5-9B paired DT/FT comparison

All 1,243 released examples are included. Attribution and deletion scoring use the same Qwen3.5-9B checkpoint.

VT and HotpotQA recovery follow the current paper policy; their primary FT comparator is K3. NIAH recovery and all RISE/MAS retain the released protocol.

| Task | N | DT Recall | FT K3 Recall | FT K1 Recall | DT RISE | FT K1 RISE | DT MAS | FT K1 MAS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| niah_mq_q2 | 100 | 63.12% | 52.92% | 50.22% | 0.1149 | 0.0982 | 0.1510 | 0.2549 |
| niah_mq_q4 | 100 | 45.70% | 38.56% | 36.92% | 0.1745 | 0.1414 | 0.2468 | 0.2651 |
| niah_mq_q8 | 100 | 13.28% | 10.72% | 8.52% | 0.3752 | 0.3704 | 0.5640 | 0.4390 |
| niah_mv_v2 | 100 | 74.07% | 60.51% | 57.58% | 0.0923 | 0.0974 | 0.1219 | 0.2467 |
| niah_mv_v4 | 100 | 54.14% | 46.19% | 45.63% | 0.1361 | 0.1284 | 0.1860 | 0.2538 |
| niah_mv_v8 | 100 | 22.12% | 22.27% | 22.32% | 0.2375 | 0.2508 | 0.3504 | 0.3264 |
| vt_h2_c3 | 100 | 80.04% | 67.80% | 67.80% | 0.1212 | 0.1514 | 0.1461 | 0.2226 |
| vt_h4_c1 | 100 | 70.56% | 69.07% | 69.13% | 0.1163 | 0.1161 | 0.1499 | 0.1889 |
| vt_h6_c1 | 100 | 80.27% | 83.06% | 83.11% | 0.1285 | 0.1301 | 0.1632 | 0.2050 |
| vt_h10_c1 | 100 | 76.35% | 81.90% | 81.85% | 0.1703 | 0.1537 | 0.2244 | 0.2304 |
| hotpotqa_long | 48 | 76.56% | 57.12% | 58.51% | 0.1906 | 0.2154 | 0.3086 | 0.3123 |
| math | 100 | — | — | — | 0.2752 | 0.3779 | 0.3635 | 0.4840 |
| morehopqa | 95 | — | — | — | 0.1562 | 0.2089 | 0.2268 | 0.2896 |

VT uses cached answers without the reasoning prefix and body-token Recall at 10%/10%/20%/30% for H2/H4/H6/H10. HotpotQA retains the full cached response and reports supporting-fact Recall: rank native sentences by signed token-score sums, then select the longest ranked prefix within 10% of all sentence-body tokens. Target filtering and candidate budgets are matched across methods. NIAH uses the released 10% eligible-token budget. Higher Recall and lower RISE/MAS are better.

| Task | N | IFR Recall | IFR RISE | IFR MAS |
|---|---:|---:|---:|---:|
| niah_mq_q2 | 100 | 45.42% | 0.1068 | 0.2510 |
| niah_mq_q4 | 100 | 35.40% | 0.1448 | 0.2631 |
| niah_mq_q8 | 100 | 2.22% | 0.3544 | 0.4318 |
| niah_mv_v2 | 100 | 55.49% | 0.1013 | 0.2398 |
| niah_mv_v4 | 100 | 47.42% | 0.1385 | 0.2638 |
| niah_mv_v8 | 100 | 21.44% | 0.2624 | 0.3494 |
| vt_h2_c3 | 100 | 77.55% | 0.1629 | 0.2315 |
| vt_h4_c1 | 100 | 70.48% | 0.1140 | 0.1692 |
| vt_h6_c1 | 100 | 82.52% | 0.1262 | 0.1814 |
| vt_h10_c1 | 100 | 81.56% | 0.1484 | 0.2021 |
| hotpotqa_long | 48 | 58.51% | 0.2437 | 0.3499 |
| math | 100 | — | 0.3929 | 0.5196 |
| morehopqa | 95 | — | 0.2297 | 0.3115 |

| Task | DT seconds | FT K1 seconds | IFR seconds |
|---|---:|---:|---:|
| niah_mq_q2 | 1.2019 | 3.0166 | 3.5592 |
| niah_mq_q4 | 1.3462 | 14.5619 | 7.6433 |
| niah_mq_q8 | 1.3406 | 15.9946 | 12.4337 |
| niah_mv_v2 | 1.1295 | 10.9379 | 3.1686 |
| niah_mv_v4 | 1.3181 | 9.3715 | 5.9275 |
| niah_mv_v8 | 1.3953 | 14.3146 | 10.2899 |
| vt_h2_c3 | 1.5535 | 11.5412 | 6.0813 |
| vt_h4_c1 | 1.5011 | 4.6239 | 6.4749 |
| vt_h6_c1 | 1.5891 | 5.8115 | 7.6559 |
| vt_h10_c1 | 1.6335 | 8.3507 | 10.0126 |
| hotpotqa_long | 1.6859 | 10.5927 | 3.1358 |
| math | 0.7192 | 14.1748 | 0.6763 |
| morehopqa | 0.8527 | 12.0944 | 1.4088 |

Times above are the full-response calls used for RISE/MAS. Separate paper-matched recovery call times are in paper_recovery_per_case.csv. Times include calls that compile or tune kernels; model loading and initialization are separate. The runtime CSV also provides medians, 90th percentiles and new Torch graph counts.

The CSV files retain IFR quality results, peak memory, and paired bootstrap intervals. Raw reports retain nested compiler timers, which must not be summed as pure compilation time.
