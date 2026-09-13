# Official attribution profiles

As of 2026-09-13, **`gdn-symmetric-v1` is the official Qwen3.5 profile**. The default factory and evaluation driver invoke it. Qwen3 retains `clean-v1`. The [active configuration](../../configs/official_dt_profiles.json) and [source manifest](sources.json) record the current entry points and identities.

```python
from deltatrace.profiles.official import make_qwen35_runner

runner = make_qwen35_runner(model, finite_fa, finite_fla)
scores, diagnostics = runner.attribute(paired_ids, mask, selection)
```

Load the directory containing `qwen35_dense_finite_runner.py` and its finite backends on `sys.path`, as for the frozen runtime. The official factory uses `make_qwen35_gdn_symmetric_runner`:

- Every GDN layer uses the symmetric product rule for normalized content and its SiLU output gate.
- Every GDN memory block averages coefficients from the two orders of the existing reference/input captures, after composing each complete ordered recurrence.
- Attention PV and output gating, QK, MLP, and normalization retain their existing rules. The native forward, reference and fixed response remain the caller's inputs.

Use `make_qwen35_runner(..., profile='clean-v1')` or the evaluator flag `--qwen35-profile clean-v1` to reproduce the old attribution rules. The frozen `qwen35_clean_runner.py` factory and its dependency manifest are unchanged.
