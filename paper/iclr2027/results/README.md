# Manuscript results and measured efficiency

This directory contains a self-contained selection of existing verified records and a deterministic CPU-only figure/table builder. It does not run models or recompute deletion scores.

- `data/qwen3_full.csv`: complete 13-task / 1,243-example Qwen3-8B RISE and MAS results. VT and HotpotQA recovery cells are deliberately omitted at the author's request; their task rows remain. Recovery is shown for all six NIAH tasks. MATH and MoreHopQA have no released recovery measure.
- `data/clean_development32.json`: signed RISE and positive MAS/recovery for 16 paired development examples per model. These are kept separate from the full task table.
- `data/qwen3_matched_cost16.json`: 7 September native finite-P1 implementation versus original FT, one warmup and three rotated measurements per example. This implementation benchmark predates the clean quality freeze.
- `data/qwen3_tiling_cost.json`: same-job dense/tiled finite propagation on three actual released examples selected by length, one warmup and three measurements.
- `data/qwen35_batch_cost16.json`: two paired warm rounds over 16 development examples, comparing CPU-checkpoint serial B1 with GPU-checkpoint real B2. B means real examples, not endpoint count.
- `data/qwen3_retained16.json` and `data/qwen35_retained16.json`: later paired warm replay-retention improvements with all complete vectors unchanged.

`sources.json` records original paths, source SHA256 values, fixture hashes, and the explicit recovery selection. The source records retain their full comparison settings; the manuscript does not interpret a before/after implementation benchmark as a speedup against FlashTrace. Device-memory figures are allocated bytes including resident weights, converted to decimal GB, not reserved memory. Loading, shape warmup, and deletion scoring are separate from the reported warm attribution calls. Timing snapshots are compared only within each recorded benchmark.

Rebuild with Python, NumPy, and Matplotlib:

```sh
python results/build_results.py
```

Outputs: `full_table.tex`, `development_table.tex`, `retention_table.tex`, `summary.json`, and vector PDF/SVG plus PNG copies in `figures/`. The main manuscript includes the full table and efficiency figure; detailed cost and development records are in the appendix. The illustrated case curves retain their recorded positive-ranking deletion protocol.

## Additional published baselines

`data/published_baselines.csv` adds Perturbation, REAGENT, CLP, IFR, and AttnLRP on all 13 tasks, preserving the requested recovery exclusions. `all_methods.csv` is the combined 91-row export for seven methods. Three manuscript tables show all methods separately for RISE, MAS, and recovery.

`import_published_baselines.py --release-dir <table1-data-v1>` verifies the original result archive and imports 95 source CSVs, retained under `data/published_baselines_raw/`. All 160 added metric values are checked against the published Tables 1 and 2 within 0.0011 in the paper's fraction units (maximum actual difference 0.000543); CSV precision is preserved. The source ledger records every file and selected row. The original paper uses `Row Attr` for these baseline means except IFR MQ-Q8 faithfulness, which matches `Recursive Attr`; this exception is explicit. Fast perturbation variants follow the published results outside MQ-Q2 and MoreHopQA. These are published baseline settings, not newly rerun methods.

The case plots now compare each model with its own one-hop FT curve. Each of four curves contains the original 21 recorded points. The metric is normalized full-response log-likelihood, including EOS, with the released clipping and cumulative-minimum transform; full-input and fully-deleted scores match exactly between DT and FT within each model/example.
