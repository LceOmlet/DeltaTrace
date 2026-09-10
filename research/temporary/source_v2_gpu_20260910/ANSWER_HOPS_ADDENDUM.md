# Preserve FT reasoning-hop support when selecting the final-answer seed

Code inspection during execution, before inspecting any answer-pilot quality
result, found that restricting both FT spans would restrict *every* hop to the
answer. That is not a clean initial-target control. The completed restricted
hop variant is retained as a diagnostic, but cannot be the primary comparator.

The primary `answer_conditioned` FT control instead keeps the original full
generation hop span and observation mask. A separately named adapter changes
only the initial aggregation weights: zero outside the same final-answer,
non-stop token positions seeded by DT. All later hops retain the original
CoT-plus-answer support and original algorithm. An all-ones initial mask must
reproduce the original FT vector exactly. The frozen FT source is unmodified.

The correction reruns FT only on the same 16 development examples and reuses
their already completed DT vectors. Original FT restricted-hop vectors, both
run identities, and all costs are retained. No sample, DT vector, ranking rule,
candidate selection rule, or reserved validation index is changed. No answer
pilot candidate has been chosen from the uncorrected results.
