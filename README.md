# DeltaTrace

**Efficient Signed Attribution for Reasoning Language Models**

[Paper](paper/iclr2027/output/pdf/deltatrace-iclr2027-draft.pdf) · [Results](paper/iclr2027/results/all_methods.csv) · [Method code](deltatrace/profiles/) · [Reproducibility](#reproducibility)

DeltaTrace explains how an input contributes to a language model's complete reasoning response. It assigns signed credit to input tokens by tracing a finite change from a reference input to the original input through the model. The explanation follows both the evidence being carried and the attention or memory operations that determine how that evidence is used.

This repository contains the method implementation for **Qwen3-8B** and **Qwen3.5-9B**, evaluation records, and the editable manuscript. Qwen3.5 now defaults to **GDN symmetric propagation**; the [official profile](deltatrace/profiles/README.md) and [active configuration](configs/official_dt_profiles.json) define its uniform rules and provenance.

![DeltaTrace overview: finite reverse propagation, attention content and selection, gated delta memory, and a signed evidence example.](paper/iclr2027/figures/generated/deltatrace-mechanism.png)

## How it works

DeltaTrace compares two executions of the same model with the same fixed response:

1. **Define the response contrast.** Replace eligible source tokens with EOS to construct the reference input. Keep the stored reasoning, final answer, and terminal EOS fixed. The attribution target is the change in the log-likelihood of that entire response.
2. **Propagate the finite change.** Compose local finite propagation rules in one reverse traversal. Attention traces both transported values and changes in their selection. Gated delta memory traces retained content, writes, queries, keys, and gates.
3. **Read signed source contributions.** Project the propagated coefficients onto each input embedding difference. Contributions can be added across tokens to inspect a name, phrase, or passage.

For original input $x_1$, reference input $x_0$, and fixed response $y$, the target is

$$
\Delta F = \log p(y\mid x_1)-\log p(y\mid x_0).
$$

The finite chain rule gives $\sum_i A_i=\Delta F$ in exact arithmetic; implementation records also track numerical residuals. A positive score supports this response contrast, and a negative score opposes it. The sign is relative to the specified reference intervention; deleting one token from the original input defines a different intervention.

The implementation uses FlashAttention-style tiled propagation for dense attention and native Flash Linear Attention operations for gated delta memory. Auxiliary attribution storage grows linearly with sequence length at fixed model dimensions and chunk size. The operator rules and proofs are given in the [manuscript](paper/iclr2027/main.tex).

## Results

<!-- selected-vt-budgets:start -->
### Qwen3-8B comparison with task-specific VT budgets

VT retrieval budgets are **H2-C3 10%, H4-C1 10%, H6-C1 20%, H10-C1 30%**;
HotpotQA remains at 10%. These budgets were selected by the user after viewing
the budget sweep. The table is a retrospective comparison using the same budget
for every method within a task.

| Method | H2-C3 @10% | H4-C1 @10% | H6-C1 @20% | H10-C1 @30% | VT macro (task-specific budgets) | HotpotQA @10% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeltaTrace | 85.62% | 71.33% | 92.05% | 87.74% | 84.18% | 72.05% |
| FlashTrace K3 | 72.41% | 68.32% | 82.15% | 83.41% | 76.57% | 66.67% |
| Perturbation | 26.69% | 18.86% | 33.57% | 40.70% | 29.96% | 37.50% |
| REAGENT | 26.66% | 17.58% | 32.47% | 42.48% | 29.80% | 34.72% |
| CLP | 29.10% | 20.53% | 33.85% | 44.04% | 31.88% | 42.88% |
| IFR | 81.16% | 70.09% | 83.74% | 82.66% | 79.41% | 75.87% |
| AttnLRP † | 83.37% | 70.54% | 84.57% | 77.33% | 78.95% | 74.83% |

The VT macro uses the four stated budgets; it is not Recall@10%. All 3,584 selected
per-case rows and 40 source means are verified, with no attribution reruns.
See the [current report, policy and data](research/temporary/vt_budget_extension_20260911/selected_budget/RESULTS.md)
and the [complete budget grid](research/temporary/vt_budget_extension_20260911/RESULTS.md).
<!-- selected-vt-budgets:end -->

<!-- frozen-all-baselines:start -->
### Frozen VT and HotpotQA comparison across all seven algorithms

All five remaining baselines have been remeasured on the same **448 inputs**,
yielding **2,240 new method-case records**. DT and FT reuse their verified vectors.
The fixed VT policy reconstructs the stored generated answer and ranks positive
eligible body-token scores. The HotpotQA policy retains the entire response,
sums signed scores over native body sentences, and charges every body token in
the longest affordable ranking prefix.

| Method | VT macro Recall@10% | HotpotQA Recall@10% |
| --- | ---: | ---: |
| DeltaTrace | 60.20% | 72.05% |
| FlashTrace K3 | 56.14% | 66.67% |
| Perturbation | 18.12% | 37.50% |
| REAGENT | 17.84% | 34.72% |
| CLP | 19.95% | 42.88% |
| IFR | 58.77% | 75.87% |
| AttnLRP † | 59.44% | 74.83% |

VT macro equally weights four 100-case tasks; HotpotQA uses official
supporting-fact Recall over 48 cases. Their metric denominators differ.
Perturbation/REAGENT/CLP use the author's 20-segment approximation. † AttnLRP
includes a documented FP16 zero-ratio repair and lossless saved-tensor offload.
This reporting scope was chosen retrospectively after DT/FT results; it does not
establish measurement neutrality or independent holdout evidence. See the
[complete results, adjusted paired intervals, technical history and raw data](research/temporary/all_baselines_20260910/RESULTS.md)
and [independent verification](research/temporary/all_baselines_20260910/verification.json).
<!-- frozen-all-baselines:end -->

### HotpotQA v3: preserved context and full body-token costs

The new **48-case run** preserves the original reasoning and answer input, masks only the initial answer target weights, and charges every tokenizer token assigned to retrieved native body sentences. Selection is the longest affordable ranking prefix, with unused budget recorded. Both full-response and answer-conditioned targets, and both signed-sum and positive-mean pooling, remain reported. At 10% full body-token budget, answer-conditioned supporting-fact Recall is **DT 35.59% versus FT K3 51.04%** for signed sums and **48.96% versus 64.76%** for positive means. The two full-response differences have descriptive adjusted intervals crossing zero. See [v3 results, full costs and verification](research/temporary/hotpot_context_v3_20260910/RESULTS.md). These specific repairs do not establish measurement neutrality or independent holdout evidence.

### Prior HotpotQA mapping fixes and exploratory evaluation

The **48-case HotpotQA audit** confirms a title/body gold-position error and a leading-space token assignment error. The added sentence pooling, budget, and target choices have **not established a neutral evaluation protocol**. In particular, the experiment labeled `answer_only` removes the cached reasoning prefix, changing the conditioning context in all 48 cases; its 68.58% versus 66.15% result cannot stand for answer attribution with the original reasoning preserved. Its sentence budget also counts eligible tokens after punctuation/whitespace filtering, rather than all body tokens. See the [methodological review and minimal mapping-fix comparison](research/temporary/hotpot_fairness_20260910/REVIEW.md). The [complete exploratory tables and raw vectors](research/temporary/hotpot_fairness_20260910/RESULTS.md) remain available.

### Complete Recall evaluation with corrected VT targets and historical HotpotQA scoring

The corrected target protocol now covers **all 448 VT and HotpotQA examples**: 100 for each VT task and 48 for HotpotQA. VT attribution uses the stored generated final answer; HotpotQA uses the full response. DT and live FT K3 receive identical inputs and target weights, rank the same source tokens, and use the same 10% token budget.

The following frozen table retains the earlier HotpotQA cached gold and regex sentence grouping for reproducibility. Use the correction report above for native HotpotQA supporting-fact evaluation; the metric and budget candidate set differ, so their absolute scores should not be directly compared.

| Task | n | DT raw Recall | FT K3 raw Recall | DT sentence mean | FT K3 sentence mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| VT H2-C3 | 100 | 85.62% | 72.41% | 100.00% | 100.00% |
| VT H4-C1 | 100 | 71.33% | 68.32% | 71.37% | 71.37% |
| VT H6-C1 | 100 | 51.23% | 51.23% | 51.23% | 51.23% |
| VT H10-C1 | 100 | 32.61% | 32.61% | 32.61% | 32.61% |
| HotpotQA | 48 | 40.63% | 45.35% | 73.81% | 72.09% |

The equal-task VT raw difference is **+4.05 percentage points**, with a descriptive adjusted interval of **[+3.59, +4.51]**. Shared sentence-mean ranking reaches the per-case budget ceiling for both methods on every VT example. On HotpotQA, DT's raw difference is **−4.71 points** with interval **[−8.60, −1.07]**; the sentence-mean difference is **+1.73 points** with interval **[−6.54, +9.48]**. These full-benchmark intervals include development examples and are descriptive. The [complete table, supplementary budgets, raw artifacts and reproduction commands](research/temporary/source_v2_gpu_20260910/full_recall/RESULTS.md) preserve the separately reported 80-case reserved validation and the earlier full-response baseline below.

### Complete Qwen3-8B evaluation

The main comparison covers **1,243 examples across all 13 released tasks** from the FlashTrace `table1-data-v1` benchmark: ten RULER tasks with 100 examples each, HotpotQA with 48, MATH with 100, and MoreHopQA with 95. The released inputs and stored responses are retained without resampling or regeneration.

The tables compare **Perturbation, REAGENT, CLP, IFR, AttnLRP, FlashTrace (FT), and DeltaTrace (DT)**. Baseline numbers come from the published result CSVs; DeltaTrace results come from the frozen full-task run.

| Metric | DeltaTrace better than FT | DeltaTrace best among all seven methods |
| --- | ---: | ---: |
| RISE ↓ | 10 / 13 tasks | 9 / 13 tasks |
| MAS ↓ | 12 / 13 tasks | 11 / 13 tasks |
| Recovery@10% ↑ | 6 / 6 reported retrieval tasks | 6 / 6 reported retrieval tasks |

These counts compare observed task means. On the six reported retrieval tasks, recovery improves over FT by **3.96–19.60 percentage points**. RISE remains higher on MQ-Q4, MQ-Q8, and VT-H10-C1; MAS remains higher on MQ-Q8.

**Reporting scope:** RISE and MAS include all 13 tasks. Recovery includes six NIAH tasks, four VT tasks at 10%/10%/20%/30% budgets, and HotpotQA supporting-fact Recall at a 10% body-token budget. The manuscript table and combined CSV state each budget and metric. MATH and MoreHopQA have no recovery measure. Earlier source records remain in the [frozen evaluation snapshot](https://github.com/LceOmlet/DeltaTrace/tree/9c6497c08ac3ffa57a40189291644e5a6b99ee36/experiments/official/results/qwen3_8b_table1_20260909).

| Artifact | Contents |
| --- | --- |
| [Combined results CSV](paper/iclr2027/results/all_methods.csv) | All seven methods, full numeric precision, and explicit empty cells |
| [Manuscript tables](paper/iclr2027/results/full_table.tex) | Separate RISE, MAS, and recovery tables |
| [Result summary](paper/iclr2027/results/summary.json) | Task counts, development results, and measured cost summaries |
| [Source ledger](paper/iclr2027/results/sources.json) | Original records, selected fixtures, and SHA-256 hashes |
| [Published baseline ledger](paper/iclr2027/results/data/published_baseline_sources.json) | The 95 source CSVs and aggregation rows underlying the added baselines |

### Qwen3.5-9B official profile results

The official `gdn-symmetric-v1` profile is compared with the same-model FlashTrace on **72 cases**: ten new key/value assignments for each NIAH task and six fixed released cases each for MATH and MoreHopQA. RISE uses signed DT ranking, MAS uses its positive part, and Recall uses a 10% eligible-token budget. FT uses K1 for RISE/MAS and K3 for Recall. These descriptive subsets are separate from the complete Qwen3 benchmark.

| Task | n | Recall FT / DT (%) | RISE FT / DT | MAS FT / DT |
| --- | ---: | ---: | ---: | ---: |
| MQ-Q2 | 10 | 62.87 / 80.59 | 0.1014 / 0.1012 | 0.2535 / 0.1325 |
| MQ-Q4 | 10 | 36.86 / 52.28 | 0.1504 / 0.1585 | 0.2675 / 0.2185 |
| MQ-Q8 | 10 | 11.97 / 15.31 | 0.3393 / 0.3544 | 0.4017 / 0.5044 |
| MV-V2 | 10 | 64.30 / 87.11 | 0.0971 / 0.0798 | 0.2452 / 0.1095 |
| MV-V4 | 10 | 50.41 / 65.34 | 0.1068 / 0.1029 | 0.2471 / 0.1390 |
| MV-V8 | 10 | 24.27 / 29.50 | 0.2357 / 0.2127 | 0.3178 / 0.2965 |
| MATH | 6 | — | 0.3734 / 0.2662 | 0.4801 / 0.3460 |
| MoreHopQA | 6 | — | 0.1920 / 0.1327 | 0.2763 / 0.1845 |

DT has higher Recall on all six NIAH tasks, lower RISE on six of eight tasks, and lower MAS on seven. See the [full-precision CSV](paper/iclr2027/results/data/qwen35_official_quality.csv) and [source ledger](paper/iclr2027/results/data/qwen35_official_quality_sources.json). The older 13-task Qwen3.5 results remain archived under `qwen35_full_quality.csv` with the clean-v1 identity.

### Measured efficiency

![Paired attribution latency and allocated memory for Qwen3-8B.](paper/iclr2027/results/figures/deltatrace-efficiency.png)

| Recorded comparison | Complete attribution time for 16 examples | Peak allocated memory |
| --- | ---: | ---: |
| Qwen3-8B: one-hop FT → tiled DT, sample batch 1 | 12.16 s → 11.40 s | 21.49 GB → 18.76 GB |

These Qwen3 measurements predate the full-task quality freeze. Historical Qwen3.5 timing fixtures describe clean-v1 and do not measure the current official profile.

Times include input preparation, endpoint capture, layer replay, propagation, and returning scores. Model loading, shape warmup, and deletion scoring are measured separately. Memory is peak allocated device memory, including resident weights, in decimal GB. Real batch size counts examples; each example has two endpoints.

Additional [dense-versus-tiled measurements](paper/iclr2027/results/figures/deltatrace-tiling-cost.pdf) cover three recorded inputs of 1,241, 3,470, and 3,762 tokens. Later [replay-retention measurements](paper/iclr2027/results/retention_table.tex) reduce warm time by 5.80% on Qwen3 against its baseline, with identical paired source vectors. The [measurement notes](paper/iclr2027/results/README.md) document the scope of each comparison.

## Inspect an explanation

![A multi-hop example with signed token contributions and each model's own DT and FT deletion curves.](paper/iclr2027/figures/generated/deltatrace-multihop.png)

Teal and coral mark positive and negative DeltaTrace contributions in nats. Both model panels use the same color scale within an example. The right panel compares each model's DT curve with its own one-hop FT baseline; its vertical axis is the **normalized log-likelihood of the full response**, including EOS. Normalization uses each model's full-input and fully-deleted scores, clipping, and the released cumulative-minimum transform. These illustrative DT curves use positive-score ranking; the main RISE tables use signed ranking.

The [figure collection](paper/iclr2027/output/pdf/deltatrace-figures.pdf) includes the mechanism overview, a two-number retrieval example, and this multi-hop example. Exact token scores and the recorded deletion points are stored in the [case fixture](paper/iclr2027/figures/data/cases.json).

## Reproducibility

### Rebuild tables, figures, and the paper on CPU

The manuscript's selected data and figure fixtures are included in `main`. Rebuilding them requires no model weights, accelerator, or new evaluation calls.

```bash
git clone --branch main https://github.com/LceOmlet/DeltaTrace.git
cd DeltaTrace

python -m venv .venv
source .venv/bin/activate
python -m pip install "numpy==1.26.4" "matplotlib==3.10.9" "pypdf==6.8.0"

python paper/iclr2027/results/build_results.py
python paper/iclr2027/figures/build_figures.py
python paper/iclr2027/figures/check_case_layout.py
python paper/iclr2027/verify_source.py
```

Use Python 3.10–3.12 for this pinned artifact environment. On Windows, activate it with `.venv\Scripts\Activate.ps1`. The commands regenerate the tables and PDF/SVG/PNG figures from the included records, and check source hashes, case geometry, citations, and LaTeX input references.

With a TeX installation providing `latexmk` and `pdflatex`:

```bash
cd paper/iclr2027
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The rebuilt manuscript is `build/main.pdf`. The checked-in [reading copy](paper/iclr2027/output/pdf/deltatrace-iclr2027-draft.pdf) is 17 pages, with the main text ending on page 9. The [build receipt](paper/iclr2027/build_receipt.json) records the distributed PDF's hash and page review. Rebuilt files may differ in metadata or layout across library and TeX versions.

### Run model attribution and evaluation

The current evaluation entry point defaults to **source-v2**: DT reference, deletion, and retrieval share an evidence-body candidate set, and token budgets are recomputed after filtering. It requires live FT and reports multiple budgets. See the [versioned evaluation instructions](experiments/official/README.md) and [CPU validation record](research/temporary/source_protocol_v2_20260910/README.md). For the completed frozen VT/HotpotQA baseline comparison, use the [dedicated protocol and runner](research/temporary/all_baselines_20260910/RESULTS.md). Use `--evaluation-protocol released-v1` for this checkout's historical adapter; the exact published benchmark uses the frozen snapshot below.

Model execution uses a source-based research environment with compiled finite-propagation extensions. The complete Qwen3 run was recorded on a **MetaX C550 64 GB**, with Python 3.12, PyTorch `2.8.0+metax3.5.3.9`, Transformers `4.57.3`, vendor FlashAttention `2.6.3+metax3.5.3.9torch2.8`, and vendor Triton `3.0.0+metax3.5.3.9`. See the [environment receipt](paper/iclr2027/results/data/qwen3_environment.json) for the complete package and checkpoint identity.

Current entry points are listed below. The frozen `clean-v1-20260909` dependencies remain in [`deltatrace/clean/`](deltatrace/clean/).

| Model | Entry point | Configuration |
| --- | --- | --- |
| Qwen3-8B | [`propagate_paired_secant`](deltatrace/clean/qwen3/qwen_signed_secant_paired_vendor_fa.py) | FP16, `content_P1`, native model FlashAttention and a separate finite-propagation extension |
| Qwen3.5-9B | [`make_qwen35_runner`](deltatrace/profiles/official.py) | BF16, symmetric output gate and memory-order average in all GDN layers; native FA/FLA |

The [official profile manifest](deltatrace/profiles/sources.json) identifies the current profile. The [dependency manifest](deltatrace/clean/sources.json) retains all 27 frozen dependency files. Qwen3 and Qwen3.5 use their respective dependency environments and run in separate processes. Model weights and compiled libraries are supplied by the execution environment; their paths and identities are checked against an [environment manifest](experiments/official/environment.example.json).

For the full benchmark, use the **paper evaluation snapshot** below. It contains the signed-RISE adapter and task controller used for the reported results. The current Qwen3.5 released-v1 adapter defaults to signed RISE for the official profile; the historical clean-v1 and source-v2 adapters retain positive RISE unless explicitly overridden.

```bash
# From the repository root, check out the exact full-benchmark snapshot.
git fetch origin
git worktree add --detach ../DeltaTrace-table1 9c6497c08ac3ffa57a40189291644e5a6b99ee36
cd ../DeltaTrace-table1
```

Before running, prepare the local model checkpoint, compatible compiled finite library, and the original FlashTrace checkout at `075e7e44ae4d5acd2ed76e0d2aced57107d02736`. Place the released `table1-data-v1` caches in that checkout and fill in `experiments/official/environment.example.json` as an external environment file. The [release protocol](experiments/official/reference/REPRODUCTION.md) documents the cache layout; use its prepared responses for benchmark reproduction.

```bash
# In the configured Qwen3 execution environment: two-example execution check.
MACA_PATH=/opt/maca TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS=0 \
python experiments/official/evaluate.py \
  --family qwen3 --environment /absolute/path/environment.json \
  --selection smoke --ft live --output /absolute/path/qwen3-smoke

# All 13 tasks, retaining the released examples and published FT results.
python experiments/official/run_qwen3_paper.py \
  --environment /absolute/path/environment.json \
  --output /absolute/path/qwen3-paper-run

# Validate and export a completed run.
python experiments/official/export_qwen3_paper.py \
  --run /absolute/path/qwen3-paper-run \
  --output /absolute/path/qwen3-paper-export
```

The task controller in the frozen snapshot verifies completed tasks before reusing them when resuming. Per-example outputs include input identity, signed and positive attribution views, recorded deletion curves, metrics, and execution costs. In the current checkout, historical Qwen3.5 development runs use `--evaluation-protocol released-v1 --family qwen35 --qwen35-profile clean-v1 --selection development16 --ft live` with the matching Qwen3.5 environment; published Qwen3 FT results are not used as its control.

### Experiment snapshots

| Record | Frozen source revision | Evidence |
| --- | --- | --- |
| Complete Qwen3-8B benchmark | [`9c6497c`](https://github.com/LceOmlet/DeltaTrace/tree/9c6497c08ac3ffa57a40189291644e5a6b99ee36) | [Full results, paired cases, and source audit](https://github.com/LceOmlet/DeltaTrace/tree/9c6497c08ac3ffa57a40189291644e5a6b99ee36/experiments/official/results/qwen3_8b_table1_20260909) |
| Qwen3.5 batching and GPU checkpoints | [`2b36c4e`](https://github.com/LceOmlet/DeltaTrace/tree/2b36c4ec74ebb0c317d2ec4b8e8da83941fea747) | [Paired warm-cost records](https://github.com/LceOmlet/DeltaTrace/tree/2b36c4ec74ebb0c317d2ec4b8e8da83941fea747/research/temporary/acceleration_20260909) |
| Replay retention | [`7fa54d2`](https://github.com/LceOmlet/DeltaTrace/tree/7fa54d2b340f47aace09c528361cb62febfc496c) | [Paired implementation measurements](https://github.com/LceOmlet/DeltaTrace/tree/7fa54d2b340f47aace09c528361cb62febfc496c/research/temporary/cause_tolerance_20260909) |

### Metric conventions

These conventions describe the published benchmark snapshot. The current source-v2 protocol is specified separately in the [evaluation instructions](experiments/official/README.md).

- **RISE:** rank the complete signed DeltaTrace contributions. Lower is better.
- **MAS and recovery:** use the positive part of the contributions. Recovery@10% measures the fraction of gold evidence in the top 10% of eligible input tokens.
- **Deletion scoring:** preserve the complete stored response plus EOS, use the released eager evaluator, and record 20 deletion steps plus the initial response point.
- **FT provenance:** full-task comparisons reuse the published CSVs and their matching `n1` traces. The separately executed development control uses one hop for faithfulness and three for recovery. The [source audit](https://github.com/LceOmlet/DeltaTrace/blob/9c6497c08ac3ffa57a40189291644e5a6b99ee36/experiments/official/results/qwen3_8b_table1_20260909/README.md) explains the difference between the published recovery traces and the upstream README's hop description.

The [results notes](paper/iclr2027/results/README.md) also document each baseline's aggregation row and perturbation variant. The experiment snapshots retain the original records, including task-level regressions and numerical differences between execution configurations.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`deltatrace/clean/`](deltatrace/clean/) | Frozen method implementations and source manifest |
| [`experiments/official/`](experiments/official/) | Reference evaluation adapter, environment schema, original protocol, and baseline CSVs |
| [`paper/iclr2027/`](paper/iclr2027/) | Editable manuscript, official template, figures, and final PDFs |
| [`paper/iclr2027/results/`](paper/iclr2027/results/) | Selected result fixtures, baseline importer, and table/plot builder |
| [`evidence/`](evidence/) | Numeric evidence, source receipts, and export hashes |
| [`core/`](core/) | Earlier finite-rule implementations retained for comparison |
| [`research/`](research/) | Prototypes, execution utilities, and recorded development experiments |
| [`docs/history/`](docs/history/) | Historical analyses and implementation decisions |
| [`third_party/`](third_party/) | Pinned upstream sources and their licenses |

## Acknowledgments

The evaluation builds on [FlashTrace](https://github.com/bwopan/flashtrace/releases/tag/table1-data-v1), its released caches and baseline results, and the RULER, HotpotQA, MATH, and MoreHopQA tasks. Efficient execution builds on FlashAttention and Flash Linear Attention. Third-party source licenses and provenance are retained alongside the corresponding code; the manuscript's [bibliography](paper/iclr2027/references.bib) lists the research references.
