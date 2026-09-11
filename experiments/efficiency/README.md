# Attribution cost versus rollout length

![Updated DeltaTrace and FlashTrace with released reference methods.](figures/deltatrace-rollout-scaling.png)

The single figure replaces DT and both FT curves with measurements from the current public v4 runtime and the unchanged author exp1 FT entry points. Qwen3-8B FP16, one C550, nominal 10-token input; rollout targets are 10, 100, 500, 1,000, 2,000, 5,000 and 10,000 tokens. Every cell has its own process and identical model input IDs across all three methods. Lines show the mean of three complete synchronized API calls after two separately retained warm calls. Error bars show the observed range. Model loading, construction, cold calls, GPU allocated/reserved peaks and host memory remain in the data.

IG, IG × Attention, Perturbation, REAGENT, IFR, CLP and AttnLRP retain 35 successful points from the [fixed FlashTrace exp1 release](https://github.com/wbopan/flashtrace/tree/075e7e44ae4d5acd2ed76e0d2aced57107d02736/exp/exp1). These historical results use six/eight devices; the overlay does not establish cross-hardware speedup ratios. Missing or failed cells have no inferred latency and split lines. The original FT rows remain in the source files but are replaced in the plotted series.

The measured DT/FT lines cross at 1,000 and 2,000 output tokens. The separately passed short-input speed and GPU-memory gate fixes output length at 32 tokens; it does not establish that DT is faster or uses less memory at every rollout length.

[Plotted JSON](curve_data.json), [CSV](curve_data.csv), [verification](verification.json), and [speed/memory scope and complete evidence](https://github.com/LceOmlet/DeltaTrace/tree/e1e37bb45de19e62a60ab650e481a52337605e14/research/temporary/qwen3_stream_memory_20260910) preserve the experiment details. Rebuild with `python experiments/efficiency/build_curve.py` (NumPy and Matplotlib). PNG, SVG and PDF are three formats of this one figure; the figure is included in the manuscript without a title or caption.

The full runtime, frozen measurement drivers and all failed candidates are pinned to [research commit e1e37bb45de1](https://github.com/LceOmlet/DeltaTrace/tree/e1e37bb45de19e62a60ab650e481a52337605e14). `verified_memory_v4/` contains the four complete final raw archives and their identities. Earlier wrapper measurements remain historical records and are not pooled into this figure.
