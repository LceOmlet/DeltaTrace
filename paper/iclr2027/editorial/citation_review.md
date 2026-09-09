# Citation review · 2026-09-09

A fresh independent reviewer checked all five cited works against primary sources and reviewed their uses in the manuscript. The author, title, year, and cited conceptual claims are supported. No files were changed by that reviewer. The root author reviewed and applied the findings below.

| Key | Verified source | Supported use |
|---|---|---|
| `shrikumar2017deeplift` | [PMLR publication and paper](https://proceedings.mlr.press/v70/shrikumar17a.html) | Activation differences, multiplier composition, summation to delta |
| `sundararajan2017axiomatic` | [PMLR publication and paper](https://proceedings.mlr.press/v70/sundararajan17a.html) | Sensitivity and implementation invariance |
| `dao2022flashattention` | [Original paper](https://arxiv.org/abs/2205.14135) | Exact attention, tiling, memory traffic |
| `pan2026flashtrace` | [Original paper](https://arxiv.org/abs/2602.01914) and [authors' tagged release](https://github.com/wbopan/flashtrace/releases/tag/table1-data-v1) | Multi-token span aggregation, recursive attribution, recovery and faithfulness protocol |
| `yang2024gated` | [Original paper](https://arxiv.org/html/2412.06464v3#S3.SS1) and [author publication page](https://research.nvidia.com/labs/lpr/publication/yanggated2024/) | Memory decay and delta update; ICLR 2025 publication |

The FlashTrace paper's general experiment description says one recursive hop. The authors' `table1-data-v1` release distinguishes one hop for faithfulness from three hops for recovery. Section 4 and Appendix B now name that release explicitly, and Section 4 links to it in a footnote. The existing protocol is preserved. The prose now refers to the released RISE and MAS scoring functions.

The tagged `exp/exp2/run_exp.py` and `llm_attr_eval.py` support the recorded FP16 loading, eager scoring, EOS handling, and deletion ordering. Statements about DeltaTrace's implementation and mathematical identities are grounded separately in the evidence map and derivations.

The subsequent prose revision retains the five bibliography entries and their supported uses. FlashTrace Section 4 was read as an editorial reference for moving from computation to mechanism and benefit. The date-and-qualifier example illustrates computational roles; it is not an empirical result. Formal statements, operator definitions, and derivations now appear in the appendix.
