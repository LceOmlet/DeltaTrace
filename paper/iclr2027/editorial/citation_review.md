# Citation review · 2026-09-09

An independent reviewer verified the 15 references used in the revised manuscript through three retrieval forms per entry: exact title, author plus keywords and year, and core keywords. The review covered bibliographic metadata and each associated claim. The author applied the precise Integrated Gradients wording and added the independently verified FLA and policy-learning references.

| Key | Primary source | Supported use |
|---|---|---|
| shrikumar2017deeplift | [PMLR](https://proceedings.mlr.press/v70/shrikumar17a.html) | Reference activation differences and propagated source contributions |
| sundararajan2017axiomatic | [PMLR](https://proceedings.mlr.press/v70/sundararajan17a.html) | Integration along the straight line from baseline to input |
| ali2022conservative | [PMLR](https://proceedings.mlr.press/v162/ali22a.html) | Conservative propagation through transformer computation |
| achtibat2024attnlrp | [PMLR](https://proceedings.mlr.press/v235/achtibat24a.html) | Attention-aware relevance propagation |
| ferrando2022alti | [ACL Anthology](https://aclanthology.org/2022.emnlp-main.595/) | Token interaction and mixing across layers |
| modarressi2023decompx | [ACL Anthology](https://aclanthology.org/2023.acl-long.149/) | Decomposed token vectors propagated to prediction |
| ferrando2024information | [ACL Anthology](https://aclanthology.org/2024.emnlp-main.965/) | Prediction-specific information-flow routes |
| jafari2024mambalrp | [NeurIPS](https://papers.neurips.cc/paper_files/paper/2024/hash/d6d0e41e0b1ed38c76d13c9e417a8f1f-Abstract-Conference.html) | Relevance conservation in selective state-space computation |
| pitorro2025latim | [ACL Anthology](https://aclanthology.org/2025.acl-long.1194/) | Token interactions in Mamba-1 and Mamba-2 |
| pan2026flashtrace | [arXiv](https://arxiv.org/abs/2602.01914) | Multi-token span aggregation and recursive source tracing |
| dao2022flashattention | [arXiv](https://arxiv.org/abs/2205.14135) | Tiling, exact attention and memory traffic |
| yang2024gated | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html) | Decay gate and delta-rule state update; ICLR 2025 |
| yang2024fla | [Repository citation](https://github.com/fla-org/flash-linear-attention/blob/main/CITATION.cff) | Hardware-efficient linear attention; native chunked Gated DeltaNet implementation |
| sutton1999policy | [NeurIPS](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) | Action-independent baseline preserves the expected policy-gradient signal |
| foerster2018counterfactual | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/11794) | Counterfactual advantage averages one agent's action with the others fixed |

The three additional entries were verified against the FLA citation file and the publisher records. Sutton et al. appear in volume 12, pages 1057–1063 (1999); Foerster et al. appear in volume 32(1), pages 2974–2982 (2018). The conference bibliography prints volume 32 for the latter. The valid arXiv citations for FlashAttention and FlashTrace are retained.

The independent reviewer also checked the elementary algebra in the policy-learning outlook. Correct interventional rollout marginals and policy-distributed reference actions give C = Q − V. Shared randomness is permitted. The policy-gradient identity uses credit as the score-function weight. The manuscript presents this as the established baseline connection motivating further research on finite reward credit.

The evaluation protocol follows the authors' [table1-data-v1 release](https://github.com/wbopan/flashtrace/releases/tag/table1-data-v1): one recursive hop for faithfulness and three for recovery. The manuscript's aggregate table explicitly reports the paired development subset. DeltaTrace's operator and kernel claims are grounded in its local derivations, frozen clean implementation, and source verification.
