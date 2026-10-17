# Six native eight-GPU runs

The user confirmed TextCraft and AppWorld, three algorithms each, with 200
complete iterations per combination and one combination on physical GPU0–7
at a time. Method ownership remains in `experiments/rl/PLAN.md`.

The existing TextCraft controller and current DTPO v6 process are preserved.
The order is TextCraft DTPO v6, PPO, GRPO, then AppWorld DTPO v6, PPO, GRPO.
The appended controller checks the original predecessor's recorded identity
and successful completion state before invoking the same process owner for
AppWorld. It never interprets a failed or missing predecessor as completion.

The default controller path remains the TextCraft path. The appended segment
uses its new `readiness` callback; there is no copied trainer, advantage,
optimizer, rollout scheduler, parser, checkpoint implementation or retry loop.
The deployed TextCraft source snapshot is retained unchanged; this extension
is a separately committed execution version.

Within each task, the native resolved data, environment and actor/rollout/ref
configurations are identical across algorithms. All runs are fresh base-model
training. PPO uses native GAE and its native critic; GRPO uses the native group
advantage; DTPO retains the approved v6 method. Native partial minibatches and
LOOP cancellation remain under their original owners.

TextCraft retains 32 groups × 8 rollouts, global optimizer minibatch 64,
PPO epochs 1 and shared save5/keep2. AppWorld retains the author's 40 groups ×
6 rollouts, global minibatch 32, PPO epochs 2, original evaluation every five
iterations and shared save2/keep2, including the actual LOOP sampler sidecar.
Per-task model, seed, sampling, B4, 32768 context, LoRA8/16 and optimizer
settings remain unchanged. AppWorld PPO retains its accepted native critic
32k memory backend. No old two-rank checkpoint is repackaged into a fresh
eight-rank run.

The controller requires native exit0, original metrics for all 200 complete
iterations, the final native completion marker, data and all eight ranks of
model/optimizer/extra state, plus the AppWorld sampler sidecar, before the
next combination. The native trainer alone decides when to save and finish.
A failed combination is recorded and is not skipped.

`plot_both_tasks.py` calls the existing plot entry once per task. Both reuse
the original collector/parser and six-panel renderer. Pending runs have no
metric points; no zero filling, smoothing, interpolation or old-run splicing
is used. Raw owner logs retain generation, DT, forward and update timings;
PPO's actor cost and additional critic cost must be reported separately.

Physical memory observation includes the environment pools, controllers,
compilers, Ray, GPU worker PSS, one host XTT pool and MemAvailable. AppWorld's
original runner count is per rank, so eight ranks increase its environment
pool as specified by that owner. Resolved configuration and single-rank 32k
receipts do not prove all-eight-rank worst-case capacity. Actual execution
and its resource observations are recorded separately.
