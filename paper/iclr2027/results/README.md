# Manuscript quality results and Qwen3 cost figures

This directory contains a self-contained selection of existing verified records and a deterministic CPU-only figure/table builder. It does not run models or recompute deletion scores.

- `data/qwen3_full.csv`: frozen 13-task / 1,243-example Qwen3-8B RISE/MAS and six-task NIAH recovery source fixture.
- `data/current_recovery.csv`: verified VT body-token Recall at H2/H4/H6/H10 budgets of 10%/10%/20%/30%, plus HotpotQA supporting-fact Recall at a 10% body-token budget. All seven methods are included.
- `data/qwen35_official_quality.csv`: current symmetric-GDN Qwen3.5 means on 8 tasks / 72 cases. `build_qwen35_quality.py` produces the 8-row, 44-value DT/FT table; FT uses K3 for Recall and K1 for RISE/MAS. `qwen35_official_cases.csv` and `qwen35_official_quality_sources.json` retain paired rows, selections and hashes.
- `data/qwen35_full_quality.csv`: archived clean-v1 13-task means; excluded from the current manuscript table.
- `data/clean_development32.json`: archived development results, superseded in the manuscript by the full quality tables.
- `data/qwen3_matched_cost16.json`: 7 September native finite-P1 implementation versus original FT, one warmup and three rotated measurements per example. This implementation benchmark predates the clean quality freeze.
- `data/qwen3_tiling_cost.json`: same-job dense/tiled finite propagation on three actual released examples selected by length, one warmup and three measurements.
- `data/qwen3_retained16.json`: paired warm replay-retention measurements. Historical Qwen3.5 cost fixtures remain archived but are not reported in the manuscript.

`sources.json` and `data/current_recovery_sources.json` record the source paths and hashes. Current recovery fixtures retain the selected policy, full-precision primary data, 3,584 per-case rows and verification receipt. Device-memory figures are allocated bytes including resident weights, converted to decimal GB. Timing snapshots are compared only within each recorded benchmark.

Rebuild with Python, NumPy, and Matplotlib:

```sh
python results/build_results.py
```

To refresh only the selected recovery results and quality tables:

```sh
python results/import_current_recovery.py
python results/import_qwen35_official.py
python results/build_results.py --tables-only
python results/verify_current_results.py
```

Outputs include `qwen35_quality_table.tex` and its source-hash verification receipt, alongside the Qwen3 tables and cost figures. `all_methods.csv` has 91 method/task rows and 259 metric values. Its `Recovery` column has explicit budget, unit, target and protocol columns; it is not a universal Recall@10% column. `recovery_summary.csv` records the VT macro at the stated task budgets. MATH and MoreHopQA recovery cells remain empty.

## Additional published baselines

`data/published_baselines.csv` supplies the five published baselines for RISE/MAS and NIAH recovery. The 35 VT/HotpotQA recovery values come from the separately verified re-evaluation. Three manuscript tables show RISE, MAS, and recovery, with task budgets listed in the recovery table.

`import_published_baselines.py --release-dir <table1-data-v1>` verifies the original result archive and imports 95 source CSVs, retained under `data/published_baselines_raw/`. All 160 added metric values are checked against the published Tables 1 and 2 within 0.0011 in the paper's fraction units (maximum actual difference 0.000543); CSV precision is preserved. The source ledger records every file and selected row. The original paper uses `Row Attr` for these baseline means except IFR MQ-Q8 faithfulness, which matches `Recursive Attr`; this exception is explicit. Fast perturbation variants follow the published results outside MQ-Q2 and MoreHopQA. These are published baseline settings, not newly rerun methods.

The case plots compare each model with its own one-hop FT curve. Qwen3.5 scores and curves were recomputed with the official symmetric-GDN factory on the same fixed illustrative inputs; Qwen3 records retain clean-v1. Each of four curves contains the original 21 recorded points. The metric is normalized full-response log-likelihood, including EOS, with the released clipping and cumulative-minimum transform; full-input and fully-deleted scores match exactly between DT and FT within each model/example.
