# Real AppWorld action-target cumulative deletion

This is a bounded diagnostic of the actual extreme source token, not a new
training implementation or method specification. `experiments/rl/PLAN.md` is
unchanged. TextCraft's first update remains held; AppWorld is terminal. No
checkpoint, rollout, DT call, backward, optimizer step or formal restart occurs.

## Owner and exact target

The unchanged author `ft_ifr_improve.faithfulness_test_skip_tokens`, SHA
`583f4b7d0426407eb9a517f173365762860a1f4382f472dffb5c07de7d3e94a1`,
owns sorting, deletion groups, default k=20, density, normalization and all
three returned metrics. Its literal-ID/scalar-score seam supplies the actual
1,266 noncontiguous prior-source positions and original scattered joint action
target Y. Original target IDs, observations, masks and original B4 other rows
are unchanged. EOS is the specified perturbation. No text round trip or new
reward label occurs.

The callback scores full joint Y with the original PackedAnswerTargets,
NativeTargetLogitRows and selected_target_log_probs. This is an explicit target
adaptation, **not** the pristine author's single-suffix LLM evaluator. The
positive-only view is for the original MAS evaluation only, never for credit.
Original signed input is FP64; the author metric receives FP32. Raw scores and
the author's normalized/running-minimum and penalized arrays are all saved.
Evaluation normalization/penalties never change training credit.

## Completed measurement

Source SHA `58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0`;
actual saved native B4 SHA
`aaa03be7fb10624918e5289aa1fc3409ae2d81ce32243a84df5539c48727be85`.
PID4098367/birth1791382409.93; GPU4/5 only. Each rank completed 42 paired
native forwards in about 450.8 seconds including initialization. Rank arrays,
scores and returned metrics agree exactly. Twin-row differences are zero;
the other three B4 rows remain constant. Fresh actor LoRA_B shards are zero.
Factual/all-EOS endpoint scores exactly match the previous native diagnostic.

Original owner returned signed-input RISE 0.31342148; the positive-only input
MAS is 0.48193712. These are one-trajectory, adapted-target measurements, with
no invented threshold or claim of overall/paper-scale DT quality.

The final signed-ranked group changes 63 actual IDs, all with negative original
DT values (sum -7.29023080). Deleting it reduces the native target log-prob by
15.72953576. The extreme newline is in this group. This group-context mismatch
complements its previously measured factual single-token sign mismatch. It does
not equate a group marginal to an individual token effect, or assign all error
to context interaction. Positive-ranked early deletion does reduce target score
strongly; the curve does not support declaring all attribution useless.

The two real extreme tokens must remain distinguished: TextCraft ` Format`
has a native factual negative effect (deleted/factual ratio 9.1575), but the
joint DT estimate implies 23.8169; AppWorld's newline supports the factual code
target, while its joint DT component is negative. `existing-numeric-scope.json`
separately preserves finite residuals and cached identity-row drift; aggregate
conservation is not an FA/FLA criterion or a bound on each component's error.

## Evidence and resource scope

`actual-results/transport.json` verifies every copied result against remote
SHA256. `analysis.json`, `curve.csv` and `action-deletion-curve.png` describe
original returned arrays; no metric is reimplemented. Two CPU tests prove only
literal ID recombination and source/target separation. They do not prove model
accuracy or numerical tolerance. The diagnostic driver has exited, GPU4/5
returned to 859MiB each, TextCraft has no release file, and container usage is
136.55GiB at the final snapshot.

The earlier AppWorld storage fix is code79922486, separately verified on the
original failing B4 with original async vLLM coexistence. It is not formally
deployed and is not exact32768/full-training verification. No capacity or
credit claim is enlarged by this curve measurement.
