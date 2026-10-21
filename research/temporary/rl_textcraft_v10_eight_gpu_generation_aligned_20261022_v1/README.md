This run keeps the approved v10 token credit and original 30 epoch / 330
iteration configuration. It starts from the base model and does not resume
the stopped two-card or earlier eight-card runs.

Execution v1 stopped during actor initialization at zero complete iterations:
the process controller could not publish its row because the three audit-local
manifests had not been initialized. Its exact session was released by the
original owner, with no survivors and all GPUs empty. The receipts and source
remain preserved. Execution v2 initializes those native manifests before
launch, and the controller checks their existing schema before spawning.

The full zero-update native control used the same 32 tasks × 8 episodes:

| Explicit request seeds | Submission groups | Physical DP | Reward sum |
| --- | --- | --- | --- |
| Same task/sample/turn identities | Original two-way | 2 | 160 / 256 |
| Same task/sample/turn identities | Eight-way | 2 | 134 / 256 |
| Same task/sample/turn identities | Original eight-way | 8 | 134 / 256 |

The latter two controls had identical 256 complete raw token trajectories,
4722 actual request outputs, 786085 chosen token log-probabilities, and native
environment returns. Changing submission grouping on two cards reproduced
the observed reward difference in this workload. Uncontrolled rank RNG and
dispatch therefore confound the older learning curves. This does not identify
a faulty kernel, reconstruct the exact historical reward difference, establish
a whole-network tolerance, or demonstrate a quality improvement.

The minimal processing change passes a stable per-request seed through the
existing TextCraft owner to original vLLM SamplingParams. Its identity uses
the original data seed, native trainer iteration, task, sample, and assistant
turn. The seed conversion is the exact function exercised in the completed
native control. An opt-in trainer metadata assignment supplies the existing
iteration clock; no second clock, sampler, environment, parser, logger, DT
implementation, reward allocation, or PPO implementation is introduced.

Formal generation retains native eight-way dispatch. No seed was selected by
reward, and no kernel or cache setting is changed. The original AgentGym log
destinations receive the new run identity. The v10 credit readout and DT batch
owners retain SHA256 812ef466… and 006419cd… . The existing eight-card VERL
owner supplies global pre-clip gradient norm measurement and resource hooks;
its v8/v9 credit branches are inactive in the approved v10 mode.

`prepare_campaign.py` binds sources; `verify_prepared_campaign.py` resolves the
exact CLI with native Hydra without loading a model. `run_campaign.py`
composes the original f71ff4d8 process owner, including its exclusive lock,
failure handling, continuous full-iteration checks, and native eight-rank
checkpoint checks, with a 330 iteration budget. Initial and after-first-DT
import observations reuse the original read-only metadata owner once each.

`plot_hourly.py` composes the existing c20559e3 parser and f6ed1ff3 renderer.
Old runs remain independent historical references; the plot is not by itself
a controlled comparison of v10 quality against v5.

The first observer adapter retained the original two-rank assertion and exited
before any worker RPC. That observation failure is archived separately. The
DP8 adapter changes only that identity count and continues the original
once-per-phase observer; the running training source and process stay fixed.
