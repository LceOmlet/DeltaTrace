# Saved credit tails and real high-impact token endpoints

Status: diagnostics completed; credit is not repaired and formal updates remain held.
This is an evidence index, not a method specification. `experiments/rl/PLAN.md`
remains unchanged.

The CPU population calculation invokes the original `counterfactual` owner,
SHA256 `0d3412b8ac25d7c036e7a827d4bbf19a4eb2b9984a0a0956f37e82d7d86f9d07`.
It reads all 44 saved TextCraft native batches and the 54 completed AppWorld
native batches. AppWorld did not complete formal DT. TextCraft includes six
repeated rows: these statistics weight saved request slots, not independent
episodes. No GPU, DT, whitening, model query or optimizer is used in this stage.

| Saved scope | TextCraft | AppWorld |
| --- | ---: | ---: |
| Rows / distinct trajectory UIDs | 176 / 170 | 216 / 216 |
| Prior source slots | 220,380 | 403,866 |
| Raw prior A <= -5 | 6 | 2 |
| Their square share of prior coefficients | 14.98% | 34.26% |
| Their square share of all policy coefficients | 4.51% | 2.50% |

Threshold -5 is a descriptive histogram bin, not an acceptance rule. Squares
describe coefficients, not parameter gradients or whitened training weights.

For each task, choose the real B4 with largest negative-prior square sum; test
the most negative, most positive and median negative source in each of its four
rows. Four original paired native B8 forwards per diagnostic rank: identity,
minimum, maximum, median. Both ranks retain the same full real B4, original
targets, observations, tokenizer IDs, scorer, HF/PEFT/FA/FLA owners and actor
initialization. LoRA B shards start at zero. No DT or optimizer update runs.

The selected minima cover 44.12% / 77.41% of saved negative-prior squares.
The 12 positions per task are deliberately biased toward influence; their
6/12 TextCraft and 4/12 AppWorld sign differences are not population error rates.
Every target vector agrees exactly across ranks; identity differences and
earlier-target effects are zero. Factual endpoints match across all modes.
Drift against original cached formal factual endpoints is retained in the data.
These controls are not FA/FLA acceptance tests or whole-DT accuracy certificates.

The new worst saved AppWorld source is newline token 198 after `Code:`:
UID `f0f85f5c-74b4-4670-a074-99a3a09dbb98`, response125/packed2883,
native rank1/batch16, reward0.75. Joint DT d=-4.29009095 implies raw A=-53.97984.
Original native single-EOS deletion gives d=+23.08903143, A=+0.75 under the
unchanged credit owner. The immediately following code-fence target 71093
falls from probability0.998526126 to7.3486e-10 after deletion. The source has
strong model influence; the saved joint negative direction is not supported
by this factual-context deletion. Its single square is 29.58% of saved prior
squares, 2.16% of all-policy squares, neither a gradient share.

TextCraft ` Format` remains genuinely negative under original native deletion,
but its coefficient is -8.15750 rather than saved joint DT -22.81693. Large
negative values cannot all be dismissed as impossible; direction and magnitude
must be checked separately. No clipping, multiplier, sign replacement or new
credit is introduced by these diagnostics. Original cumulative deletion / RISE /
MAS evidence remains in `../../direct-target-action-author-curve-20261007/v1`.

Memory repair is separately recorded in
`experiments/rl/results_extreme_credit_memory_20261007.json`, code79922486:
existing mixer offload and release of consumed original HF replay cache storage.
Actual failed B4 with original async vLLM lifecycle completed, physical sampled
peaks55.716/55.290GiB; signed/QVA equal exactly. It is not formally deployed;
this is not an exact32768 or entire28-batch capacity claim.

Artifacts: `population.json`, `sample-analysis.json`, `sample-points.csv`,
`sample-endpoints.png`, both `results-*` transports, effective configurations,
runtime owner inspection, process birth handles, physical/cgroup observations.
Original bytes and SHA256 are preserved. Model diagnostic script SHA256 is
`5a02e2470e9a9e617a33fd2c7c006de0f45a4f58c2a39dde78b4f317b7560d8d`.

TextCraft's first submit rejected a wrong AppWorld-specific runner hash before
worker creation; the submit check now reads each task's own frozen source hash.
The rejection is retained. `inspect.getsourcefile` of a no_grad-wrapped function
reported Torch contextlib; that metadata is retained and separately disambiguated
by original module import plus `inspect.unwrap`, without changing computation.
