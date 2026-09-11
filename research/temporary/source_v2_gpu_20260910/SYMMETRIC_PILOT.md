# Layerwise symmetric attention allocation pilot

The answer-only target repaired the multi-chain VT development failure, but
HotpotQA remained below FT. Neither reference replacement nor four equally
available pooling rules passed development. Keep every failed experiment.

Hypothesis: content-P1 attention allocation favors value content over changes
in routing that can encode bridge evidence. For an attention product P V,
the current rule is deltaP V0 + P1 deltaV. Test the endpoint-symmetric identity
deltaP (V0+V1)/2 + (P0+P1)/2 deltaV. This is a different finite allocation rule,
not a change to the native model and not an assertion of unique causal credit.

Implement the candidate at every attention layer by averaging the original
finite P1 kernel's multipliers with those from swapped operands and V1. Use
the same two captured native endpoints, without exchanging native batch
positions. Average dq/dk/dv in FP32 before the unchanged layout and upstream
propagation. The Q/K symmetric product rule, non-attention operators, signed
seed, native replay and unassigned residual reporting remain unchanged.
This differs from averaging two complete attribution passes, already tested.

Fixed candidate: layer_symmetric, all 36 layers, no layer or blend search.
Reference: original full author-eligible EOS. Target: generated answer_only,
matched non-stop tokens. Ranking: sentence mean, exactly ceil(0.10*N_source).
Native input, candidates and budget are identical for DT and live FT K1/K3.
The new DT method invokes the finite attention kernel twice per layer; report
the additional operations and actual time. Never present that compute as free.

Use the same fixed 8 VT-H2-C3 and 8 HotpotQA development examples. They are
reused development data, not a fresh test. Rerun original content-P1 DT and
live FT controls, requiring bitwise equality with the answer-target pilot.
Test the symmetric product algebra separately and audit finite outputs.

Advance only if the fixed candidate has a positive DT-minus-FT-K3 mean on at
least one task and each other task improves or both methods attain their exact
per-case budget ceiling. The original content-P1 method is a diagnostic, not
an alternative validation choice. Freeze layer_symmetric before the untouched
80-case validation. Retain the reference pilot's criterion: positive macro
95% paired-bootstrap lower bound; VT and HotpotQA must each improve or have
joint exact per-case ceiling parity, with at least one group improving.
Report 5%/20% and raw ranking as secondary results. No second validation
candidate outcome has been read when this plan is written.
