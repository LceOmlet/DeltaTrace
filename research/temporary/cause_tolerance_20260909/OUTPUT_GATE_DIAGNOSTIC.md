# Test the shared Qwen3.5 post-attention boundary before changing it

The matched partial-deletion diagnostic identifies attention-family errors in
both models and in the successful MH controls. It does not justify changing a
rule merely because its family term is positive. Within Qwen3.5, the NI-minus-MH
excess preference has a +36.99 mixer contribution; the corresponding Qwen3
difference is -1.02. These are within-model error accounts, not architectural
causal effects. Preserve the native models, their original wrapping and targets.

One structural difference common to Qwen3.5's FA and GDN blocks is their output
gate. Both use content1 finite allocation, while Qwen3 has no such output gate.
Before proposing a uniform gate rule, split its actual boundary contribution
from the rest of the mixer on fixed NI0/1 and MH0/1. No layer is selected by size.
Reuse the original whole-DT observer and all original10%DT/FTK1 deletion sets.

Observe the actual GDN norm-gate module input/output with ordinary module hooks.
Observe the actual FA interface return plus its q_proj/o_proj boundary; no new
attention call or probability matrix. Compute the original finite coefficient
dot products against these actual partial activations. The FA projection
coefficient needs one additional call to the existing finite linear transpose;
it is charged diagnostic work, not a candidate's speed measurement.

Report four signed terms per mixer: output projection, fused norm-gate (GDN)
or sigmoid gate (FA), GDN upstream cast, and the remaining mixer. Their sum must
reconstruct the prior unsplit mixer error. GDN norm-gate is intentionally one
composite term: the fused native operator does not publish a separate normalized
output, so do not invent one and call it an actual counterfactual activation.
FA upstream cast remains inside its core remainder. Preserve every opposing
term; no clipping, calibration, layer repair, new target, or metric change.

Same-process plain/observed full vectors and EOS logprobs must agree, and the
prior four-case split must reproduce within measured native precision. The pilot
costs8completeDT+8pairedpartialnativeforwards+1nativeinitialization. There is no
candidate or 21-point metric run. If this boundary contributes little, opposes
the excess preference, or lacks the proposed NI-specific pattern, stop the
gate-only repair idea rather than expand it. A large error would still require
a uniform rule counterfactual and original-metric validation before adoption.
