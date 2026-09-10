# Attribution cost versus rollout length

The single plot retains seven baseline curves from the [FlashTrace exp1 release](https://github.com/wbopan/flashtrace/tree/075e7e44ae4d5acd2ed76e0d2aced57107d02736/exp/exp1), replaces its FT curve with new complete-wrapper measurements, and adds frozen DeltaTrace (`clean-v1`). It is a standalone artifact; the manuscript is unchanged.

## Measured DT and FT

Qwen3-8B FP16, nominal input length 10 tokens, one MetaX C550 64 GB. Each method/length cell runs in a fresh process: one first call followed by three warm calls. Lines show warm medians; shading shows the observed range. Both timers stop after final input-token scores are returned as CPU float lists; model loading is excluded. FT calls the unmodified exp2 `run_attribution` wrapper, including per-call engine construction and all sequence, row, and recursive views. DT includes per-call construction, preparation, endpoint capture, propagation, and final score selection. Targets use the original exp1 token-tiling functions, with identical actual input IDs for DT and FT.

| Output tokens | DT median (s) | FT median (s) | DT peak (GB) | FT peak (GB) |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.281 | 0.180 | 16.52 | 16.58 |
| 100 | 0.314 | 0.255 | 16.81 | 16.84 |
| 500 | 0.517 | 1.473 | 18.17 | 17.85 |
| 1,000 | 0.868 | 5.193 | 19.86 | 20.16 |
| 2,000 | 1.691 | 19.091 | 23.27 | 28.59 |
| 5,000 | 5.149 | OOM | 33.43 | — |
| 10,000 | 14.330 | OOM | 50.38 | — |

Peak memory includes resident weights in decimal GB. Every completed cell has identical score hashes across four calls. DT keeps its complete target plus EOS; FT keeps the original `ifr_multi_hop_both` default sink excluding EOS. Raw calls, the verified checkpoint receipt, the host record, and the byte-identical native-library rebuild receipt are in `results/`. All current points come from one instance; the earlier interrupted run is preserved in `archive/interrupted-full-wrapper-v2/` and is not pooled into this curve.

## Reused baselines

IG, IG × Attention, Perturbation, REAGENT, IFR, CLP, and AttnLRP use 35 successful points from `out-0` and `out-2` through `out-5`. The input-length sweep in `out-1` is excluded. Short points (10/100 tokens) average three runs on eight devices; longer points are single runs on six devices. These are different hardware settings from the new DT/FT measurements, so the overlay does not establish cross-hardware speedup ratios.

`upstream/` retains byte-preserved source logs, the original runner, and hashes. Original FT records remain available for provenance but are omitted from the plotted CSV. Failed cells have no inferred times. `archive/entrypoint-v1/` preserves the previous FT core-entry-point measurements and their exact protocol; those records are not used in this plot. The new FT curve measures the complete evaluation wrapper, so it does not establish a core-algorithm speed ranking.

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
