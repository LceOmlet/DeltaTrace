# Attribution cost versus rollout length

The single plot retains seven baseline curves from the [FlashTrace exp1 release](https://github.com/wbopan/flashtrace/tree/075e7e44ae4d5acd2ed76e0d2aced57107d02736/exp/exp1), replaces its FT curve with the newly measured FT curve, and adds frozen DeltaTrace (`clean-v1`). It is a standalone artifact; the manuscript is unchanged.

## Measured DT and FT

Qwen3-8B FP16, nominal input length 10 tokens, one MetaX C550 64 GB. Each method/length cell runs in a fresh process: one first call followed by three warm calls. Lines show warm medians; shading shows the observed range. Time includes the complete attribution entry point and excludes model loading. Targets are supplied by the original exp1 token-tiling functions, with identical input IDs for DT and FT.

| Output tokens | DT median (s) | FT median (s) | DT peak (GB) | FT peak (GB) |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.329 | 0.183 | 16.52 | 16.58 |
| 100 | 0.311 | 0.204 | 16.81 | 16.84 |
| 500 | 0.521 | 0.396 | 18.17 | 17.85 |
| 1,000 | 0.883 | 0.633 | 19.86 | 20.16 |
| 2,000 | 1.716 | 1.236 | 23.27 | 28.59 |
| 5,000 | 5.181 | OOM | 33.43 | — |
| 10,000 | 14.406 | OOM | 50.38 | — |

Peak memory includes resident weights in decimal GB. Every completed cell has identical score hashes across four calls. DT keeps its complete target plus EOS; FT keeps the original `ifr_multi_hop_both` default sink excluding EOS. Raw calls and the verified checkpoint receipt are in `results/`.

## Reused baselines

IG, IG × Attention, Perturbation, REAGENT, IFR, CLP, and AttnLRP use 35 successful points from `out-0` and `out-2` through `out-5`. The input-length sweep in `out-1` is excluded. Short points (10/100 tokens) average three runs on eight devices; longer points are single runs on six devices. These are different hardware settings from the new DT/FT measurements, so the overlay does not establish cross-hardware speedup ratios.

`upstream/` retains byte-preserved source logs, the original runner, and hashes. Original FT records remain available for provenance but are omitted from the plotted CSV. Failed cells have no inferred times.

## Rebuild

```bash
python experiments/efficiency/build_curve.py
```

Requires `numpy==1.26.4` and `matplotlib==3.10.9`. Outputs are `curve_data.csv`, `curve_data.json`, `verification.json`, and `figures/deltatrace-rollout-scaling.{pdf,svg,png}`. The standalone PDF is also copied to `output/pdf/deltatrace-rollout-scaling.pdf` at the repository root.

To rerun measurements in the native environment, supply the same `qwen3` environment schema as `experiments/official/evaluate.py`:

```bash
python experiments/efficiency/benchmark.py \
  --environment /path/to/environment.json \
  --author-script experiments/efficiency/upstream/run_time_curve.py \
  --output /path/to/new-run
```

The destination must be new. `--lengths` and `--methods` support smaller runs. Source, native model, and finite-attention library hashes are checked. The 600-second setup/per-call limit terminates only the controller's own worker.
