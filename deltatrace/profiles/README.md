# Official attribution profiles

As of 2026-09-13, **`gdn-symmetric-v1` is the official Qwen3.5 profile**. The default factory and evaluation driver invoke it. Qwen3 retains `clean-v1`. The [active configuration](../../configs/official_dt_profiles.json) and [source manifest](sources.json) record the current entry points and identities.

```python
from deltatrace.profiles.official import make_qwen35_runner

runner = make_qwen35_runner(model, finite_fa, finite_fla)
scores, diagnostics = runner.attribute(paired_ids, mask, selection)
```

Load the reviewed directory containing `qwen35_dense_finite_runner.py` and its finite backends on `sys.path`, as for the frozen runtime. The official factory uses the already validated `make_qwen35_gdn_symmetric_runner` implementation:

- Every GDN layer uses the symmetric product rule for normalized content and its SiLU output gate.
- Every GDN memory block averages coefficients from the two orders of the existing reference/input captures, after composing each complete ordered recurrence.
- Attention PV and output gating, QK, MLP, and normalization retain their existing rules. The native forward, reference and fixed response remain the caller's inputs.

The profile rejects layer-specific attribution overrides and does not inspect task labels, query positions, gold annotations, or evaluation scores. Verified runtime overlays may pass supported execution options, such as `dynamic_shapes` and `compiler_options`, through the official factory. The original frozen runtime does not support those extra execution options.

For Qwen3.5-9B, one attribution uses 48 finite memory calls, 96 native FLA adjoint stages, and 8 finite attention calls. The two local memory orders reuse existing captures; they add no native root forward or reference endpoint.

Current quality evidence contains **8 tasks and 72 examples**: six NIAH tasks with ten novel key/value assignments each, and six fixed MATH plus six fixed MoreHopQA examples. The [source ledger](../../paper/iclr2027/results/data/qwen35_official_quality_sources.json) and [DT/FT table](../../paper/iclr2027/results/qwen35_quality_table.tex) report signed RISE, positive-part MAS and NIAH Recall at a 10% token budget. FT uses K1 for faithfulness and K3 for Recall. These subsets are distinct from the archived 13-task clean-v1 benchmark.

The [controlled experiment report](../../research/temporary/qwen35_niah_causal_20260913/RESULTS_zh.md) retains candidate selection, all regressions, the frozen validation protocol, and the single-case cost measurement. The [native integration check](../../research/temporary/qwen35_official_promotion_20260913/figure_import_verification.json) verifies that the official default matches the frozen prototype bitwise and supplies three recomputed manuscript illustrations.

Use `make_qwen35_runner(..., profile='clean-v1')` or the evaluator flag `--qwen35-profile clean-v1` to reproduce the old attribution rules. The frozen `qwen35_clean_runner.py` factory and its dependency manifest are unchanged.
