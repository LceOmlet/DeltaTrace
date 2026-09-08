# Qwen3.5 answer integration status

One frozen32-layer engineering pass has now executed and been independently accounted. See [result](history/Qwen35答案目标与32层有限传播_20260908.md). It is not a quality result, fair FT timing or general production API validation.

The [protocol](../research/reproduction_templates/qwen35_whole_finite_protocol_20260908.json) permits one finite32-layer pass on the two already used author trajectories, with saved root inputs and saved original layer0/3 operands.30 original decoder replays,32 finite decoder calls, seven new public FA auxiliary calls, one packed original head/norm forward and backward. No root forward, generation or quality query.

The [answer runtime](../research/runtime/qwen35_answer_finite.py) makes target positions explicit. It selects the author's remapped closed answer sink (38/21 tokens);59 genuine target rows are packed into one original248320-vocabulary Linear head call, with both endpoints. Whole-response context is unchanged. The old8B P1 used full-response seeds, a different target scope; new results cannot be merged with old metrics. RISE/MAS remains the original separate evaluator.

The [study](../research/reproduction_templates/qwen35_whole_finite_20260908.py) records packed-head/root logprob differences, equal-endpoint head/norm native-gradient differences, per-layer replay discontinuities and finite residuals. Rounding remains unassigned. Saved per-layer coefficients permit diagnosis without repeating completed layers. This is an engineering integration run with weight IO, capture and compiler costs, not a warm FT cost comparison.

The main comparator is the official FT-Qwen3.5 variant on the same checkpoint. See [source audit](../evidence/qwen35_FT_variant_audit_20260908.json). Do not replace it with FT on Qwen3-8B or confuse the currently verified9B checkpoint name with an8B model.
