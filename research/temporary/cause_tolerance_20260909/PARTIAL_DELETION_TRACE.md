# Trace actual deletion discrepancy through unchanged finite propagation

The previous count of DT/FT deletion-ranking reversals does not identify a
responsible operator. This diagnostic follows the actual partial-deletion
inputs through the native model, and contracts their hidden-state differences
with the original all-EOS-to-input DT coefficients. It neither repairs a layer
nor introduces a new attribution target or benchmark.

For a boundary h with its original DT coefficient m, define
`P_h(A) = <m_h, h(original) - h(delete A)>`.
The difference between consecutive P values is that block's contribution to
the mismatch between the fixed linear allocation and this actual deletion.
Summing these signed differences telescopes to
`sum(original DT credits of A) - actual full-target logprob drop`.
This is an accounting identity for a specific observed intervention, not proof
that the block alone is causally responsible or should be forced to fit A.

Qwen3.5 already exposes a read-only decoder observer. Use it unchanged to split
each layer into input RMSNorm, mixer, residual addition, post RMSNorm, MLP, and
second residual addition. Logit-head/seed and final RMSNorm terms are separate.
Use FP64 dot reductions of actual BF16 activations and original coefficients;
these diagnostic reductions do not change propagation or scoring outputs.
Residual rounding and any native replay-to-root boundary gap remain named
terms, not requests to modify the framework.

Freeze the exact original DT and FT K1 10% deletion sets from the already
verified author NI0..7/MH0..7 records. First run NI0/1 and MH0/1 as an instrument
pilot, retaining all directions. The rule/sets are never adjusted per result.
Each case costs two native paired partial forwards and two complete DT calls
(one plain and one observed). Endpoints keep the same B2 shape, mask, native FA,
target rows and full cached generation including EOS. Original EOS endpoint
logprobs must agree, and the observed DT vector must equal the same-process
plain vector. No FT or generation call and no new 21-point metric curve.

Inspect signed family sums and patterns across cases, not the largest isolated
layer residual or absolute intermediate mass. If the identity or native control
fails, correct this diagnostic before interpreting it. Extend to original16
and the corresponding Qwen3 trace before making a cross-model root-cause claim.
No repair is justified merely by a successful telescope check.

The pilot passed. To avoid repeating it, the extension runs only indices2..7
of NI/MH (12 additional cases); the verified summary joins these with pilot4.
Each sample's plain/observed DT and two partial forwards are within one process.
The native FA FP32-logsoftmax/FP64-sum drops are diagnostics at DT's objective;
they do not replace the author's metric backend or constitute new RISE values.
