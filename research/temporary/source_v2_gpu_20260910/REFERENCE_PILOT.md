# Fixed answer-only reference pilot

The answer-target development experiment restored VT-H2-C3 DT Recall to
100% (FT also 100%), but answer-only HotpotQA DT remained 1.50 points behind
FT on the eight development cases. No second validation has been run.

Hypothesis: a repeated EOS reference creates a model-state change specific to
the special end token. Compare it to a repeated ordinary space-token reference,
then average their signed allocations with equal weight. Both references replace
exactly the original author-eligible prompt positions, retain the same original
prompt scaffold, and use the same generated final answer plus EOS. The target
seed is the same non-stop, non-EOS answer token set used by live FT. Neither
reference uses gold, document labels, task-specific answer keys, or attribution
outcomes. Native models and all finite propagation rules remain unchanged.

Three reference settings only: EOS, space, and the arithmetic mean of their
signed vectors before positive clipping. This is a two-reference attribution
variant with twice the DT compute. Its signed sum describes the mean of two
explicit contrasts, not the original single-reference contrast. Keep all three
settings and their costs. Token ranking is the previously fixed sentence-density
ordering with exactly ceil(0.10*N_source); FT has the identical ordering and
budget. Raw ordering and 5%/20% budgets remain secondary.

Use exactly the second pilot's fixed development and untouched validation
indices (answer_split.json), with a separate plan identity. Select on the same
8 VT-H2-C3 plus 8 HotpotQA development examples. Advance the setting with the
largest minimum task difference, then macro mean, then name, only if one task
improves and the other improves or is jointly at its exact per-case ceiling.

Before the 80 untouched validation cases, freeze the reference setting. Claim
a shared aggregate advantage only if the equal-task macro difference has a
positive 95% paired-bootstrap lower bound and neither the VT group nor HotpotQA
has a negative mean difference. If a group ties exactly at every case's budget
ceiling, label it as saturated parity, not superiority; require a positive mean
in at least one group. This explicit ceiling condition is necessary because
both methods already attain 100% on the VT development cases. Report all five
validation tasks and preserve the first pilot's failed validation result.

For development, rerun the original EOS DT and live FT controls on identical
answer-only inputs and require bitwise equality with the completed answer-target
pilot. Verify weighted conservation against both measured endpoint contrasts.
No new validation score has been inspected or used to make this plan.
