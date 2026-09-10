# Fixed sentence pooling development probe

The answer-only target removed the irrelevant-chain failure in VT-H2-C3,
reaching 100% for both DT and FT on all eight development examples. It did
not improve HotpotQA: DT remained 1.50 percentage points below FT. The EOS,
space and two-reference mean controls all failed. Preserve those failures.

Hypothesis: averaging every eligible token in a sentence can dilute a sparse
support anchor. Compare positive-token raw ranking, sentence arithmetic mean,
sentence root-mean-square, and sentence maximum. For sentence scores, break
ties by positive token score and original position, as before. Every ranking
selects exactly ceil(0.10*N_source) tokens; it never selects whole sentences
for free. Sentence boundaries, target extraction, source scope, EOS reference,
model input and both attribution vectors remain fixed and gold independent.

This CPU-only development probe uses exactly the 8 VT-H2-C3 and 8 HotpotQA
examples from answer_split.json, answer_only mode, corrected FT K3 and the
matched non-stop target DT. No second validation example is loaded. These
development cases have supported earlier failed hypotheses; they are not a
fresh significance test. No added GPU attribution is needed for this probe.

Give each method the same four options. Select each method's single pooling
rule separately by equal-task mean Recall@10%, then minimum task Recall,
then lexicographic rule name. Compare DT against that development-optimized
FT baseline, not against whichever FT variant gives the largest DT gap.
Report every option for both methods, and the same-rule comparisons.

Advance only if DT's selected rule improves at least one task and every other
task improves or both methods attain their exact per-case budget ceiling.
Freeze both rules before running the untouched second 80-case validation.
Use the reference pilot's existing criterion there: the equal-task macro
95% paired-bootstrap lower bound must be positive; VT and HotpotQA must each
improve or have joint exact per-case ceiling parity, with at least one group
improving. Label ceiling parity as parity. Keep 5% and 20% as secondary checks.

If this probe fails, do not select another target/reference combination after
looking at pooling results. The answer-only target and EOS reference are fixed
for this experiment. Preserve the failure and its full development table.
