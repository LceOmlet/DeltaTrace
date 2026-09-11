# CPU scoring precision correction

The complete GPU run finished before the independent CPU verifier rejected a
pooled HotpotQA DT sentence score: 7.93521027508618 from the original vector
versus 7.935210063820705 in the first combined scoring draft. The combined
scorer had converted every method's source vector to float32 before dispatching
between VT and HotpotQA. Frozen native HotpotQA v3 preserves the original
vector precision and sums sentence tokens in float64. VT explicitly uses
float32, and keeps that behavior.

The initial combined draft is preserved under `scoring_precision_control`,
including its scorer source and compressed output files. Comparison with the
frozen v3 selections located differences only in the 288 DT `ranked_scores`
arrays (48 cases times six budgets), with no differences in ranking order,
selected sentences or selected tokens. Restore the existing v3 precision rule
in the combined CPU scorer and require exact reproduction of the previous
DT/FT sentence scores as well as selections.

This restores the already frozen policy; no GPU vectors, target weights,
candidate mapping, aggregation, budget, labels or method settings are changed.
Recompute the complete CPU analysis and run both the precision control and the
independent verifier. The control requires all 34,944 per-case metric rows,
390 summary rows and the primary CSV to remain byte-identical, every ranking
and selected set to remain unchanged, and all 864 reused HotpotQA ranking-score
records to match v3 exactly. Its machine-readable verification records the
outcome and hashes.
