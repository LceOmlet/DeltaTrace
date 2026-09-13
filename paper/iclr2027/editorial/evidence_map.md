# Manuscript evidence map

This working file connects the draft to its source material. It is separate from the manuscript.

| ID | Material | Supports | Scope of support | Manuscript use |
|---|---|---|---|---|
| D1 | `deltatrace/clean/qwen3/qwen_signed_secant_vendor_fa.py` and paired runner | P1 attention, symmetric QK/SwiGLU, finite normalization and fixed-response seed | Method definition; empirical comparison comes from result records | Sections 1–3 |
| D2 | `deltatrace/profiles/official.py`, `qwen35_gdn_symmetric.py`, and frozen decoder/GDN modules | Uniform symmetric GDN output gate; average of complete memory endpoint orders | Official source definition; not a benchmark result | Sections 2–3 |
| D3 | `deltatrace/clean/qwen35/finite_fla_gpu.py` | Native chunk adjoints, mixed contractions, finite decay scan, 64-token chunks | Actual computational structure | Section 3 |
| D4 | `research/prototypes/vendor_fa_finite_p1.cu` and BF16/D256 extension | Log-mean row reductions and finite attention contractions on the vendor FA framework | Operator design; whole-method timings are separate | Section 3 |
| D5 | `experiments/official/protocol.json` and README | Author inputs, task selection, eager scoring, K=1 faithfulness and K=3 recovery | Evaluation protocol; results remain in their own records | Section 4 and Appendix B |
| D6 | `figures/data/cases.json`, `figure_manifest.json` and `research/temporary/development16_20260909/numeric.json` | Actual signed token contributions and saved deletion curves for both models | First released example of each of two development tasks; four cases, with actual input hashes and response text verified | Source-span bars in Figure 1; Figures 2--3 and signed-evidence paragraph in Section 4 |
| D7 | `results/data/qwen35_official_quality.csv` and `results/qwen35_quality_verification.json` | Qwen3.5 symmetric-GDN DT/FT task means | 8 tasks, 72 cases, 44 values; no efficiency or causal-mechanism inference | Table 4 and Qwen3.5 quality paragraph |
| T1 | Finite local identity and adjoint inner-product substitution | Contribution conservation on an acyclic real-valued graph | Exact arithmetic theorem | Section 2 states the accounting consequence; Appendix A.1 gives Theorem 1 and its proof |
| T2 | Probability normalization and logarithmic-mean identity | Finite softmax, response seed, PSD score operator | Exact arithmetic on common unmasked support | Section 2 explains competition among sources; Appendix A.2 gives Proposition 1 and its derivation |
| T3 | Bilinear expansion and reciprocal-norm identity | Product rules, GDN change recurrence and RMS rule | Local mathematical identities | Section 2 explains content, routing, retention and writing; Appendix A.3--A.5 gives the full rules |
| L1 | Shrikumar et al., PMLR 2017 | Reference activation differences, multiplier composition, summation to delta | Retrieved original paper and supplement | Related Work |
| L2 | Sundararajan et al., PMLR 2017 | IG and its axiomatic attribution framing | Retrieved PMLR entry | Related Work |
| L3 | Dao et al., arXiv:2205.14135 | FlashAttention tiling and memory-traffic organization | Retrieved paper abstract and original source | Introduction, computation, related work |
| L4 | Pan et al., arXiv:2602.01914v4 and authors' `table1-data-v1` release | FlashTrace span aggregation, recursion, original evaluation; release specifies K=1 faithfulness and K=3 recovery | Retrieved full text, tagged author code and data release, and released local protocol | Introduction, evaluation, related work |
| L5 | Yang et al., arXiv:2412.06464v3 | Gated delta state update and chunkwise computation | Retrieved full text | Gated-memory formulation and related work |

The draft includes mechanism, mathematical properties, implementation structure, evaluation protocol, and fixed-index development-case illustrations. Quality and timing comparisons attach to full aligned experiment artifacts. The positive-view patched eight-case results and earlier raw-prompt comparisons are not manuscript result sources. Figure 1 centers an exact response excerpt and actual input-token heatmap from the Qwen3.5 multi-hop example. Its single reverse arrow and local attention identity explain the method. Figures 1--3 use stored clean-v1 Qwen3 scores and recomputed official symmetric-GDN Qwen3.5 scores, with one linear color scale across both models within each task. They introduce no FlashTrace mechanism or hop-difference semantics.

Supervisor-Skills reference: commit `207bc6f7a1aa107e544099c2c7cc86816fba9628`, `paper-writer` and `intro-drafter`. The workflow uses source-linked claims and an independent citation check. The user's requested focus and affirmative narrative govern the prose; fixed paragraph lengths, reference quotas, and prescribed limitations sections are not applied.
