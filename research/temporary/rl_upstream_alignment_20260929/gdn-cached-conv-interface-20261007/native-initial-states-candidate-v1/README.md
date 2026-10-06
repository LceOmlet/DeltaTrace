# Cached-convolution initial_states: measured and deployed

2026-10-07: the actual long B8 comparison and exact32768 capacity check have
completed. Current production AppWorld driver1181392 uses the two tested DT owner
bytes and the minimal patch in the original installed canonical HF module,
with original VERL/LOOP unchanged. Driver1119928 failed at task service
readiness before sampling: its isolation namespace loader added about4s to
every environment interpreter startup. The current production PYTHONPATH
excludes that loader; both loader flags are removed. The private namespace
path below describes the completed isolated DT comparison, not the production
service import path. It starts from base weights without checkpoint recovery. See
`experiments/rl/results_appworld_efficiency_20261007.json` and RUNTIME_RECORD.md
for source hashes and the scope of official operator assertions. Historical
preparation receipts remain unchanged; they do not override deployment status.

The three source patches use the installed public `causal_conv1d_fn` interface.
The HF owner receives a small opt-in branch instead of a copied/wrapped GDN
forward. The existing DT capture retains the actual suffix operand and its
fixed left window. The existing finite convolution and public autograd input
VJP receive that same window. The finite secant/seed/contraction/dtype formulas
are unchanged; `prepare_sources.py` checks their ASTs.

The cache owner mutates its state with `copy_`. The small last `width-1` window
must therefore be cloned before the existing cache update. At a positive
existing coefficient start, the actual projection's preceding `width-1`
values become the fixed window; no full concatenation is reintroduced.
The default/no-cache/single-token/short-suffix/fallback/seq_idx paths and the
prefix-bank preparation retain the original code.

## Reproducible preparation

The historical preparation reads pinned before-owner sources from the c1a079f
checkout. Use that pinned checkout to reproduce it; the current repository
already contains the two tested owner changes. Run the CPU preparation scripts
in that checkout only. They assert the frozen
owners' exact SHA256 and emit the isolated source files and patch diffs. They
do not change a runtime, import torch, load a model or start a remote process.

`prepared-sources.json` lists the frozen owner files, candidate hashes,
actual source of the in-place cache update and the already completed original
real-long-operand operator comparison. `prepared-diagnostic.json` records the
diagnostic driver's source and limited changes.

## Original B8 diagnostic entry

`candidate_probe_entry.py` is generated from the exact original long-case
collector SHA `1d4270bc883d1c495de3997fe191a53495a1ef9d58b3deb408eba0c893675613`.
It retains the original ActorRolloutRefWorker initialization, effective config,
DataProto collation, saved request bank and context-sorted offsets 84:88 per
rank. It initializes base weights only, disables checkpoint recovery and
performs one original `compute_dt_token_advantages` RPC on B8 total (B4/rank).
It does not generate tasks, update parameters or create a new training path.

For a later, separately staged candidate diagnostic process, reuse the
existing long-v1 `prepared.json` launch_path/request_root/environment values,
substituting only the independently staged candidate owner paths. Keep LoRA
rank8, alpha16, per-rank microbatch4 and 32768 unchanged. Its source check
requires the exact three candidate hashes **before** enabling the runner's
default-disabled execution option. Do not change the running formal job or
shared owner files to satisfy this check.

The original execution command, after that isolated source staging and the
existing environment exports, is:

```sh
export CONV_PROBE_ROOT=<independent-receipt-directory>
"$VENV_PYTHON" -u "$CONV_PROBE_ROOT/candidate_probe_entry.py"
```

Copy `candidate_probe_entry.py` and `diagnostic_enable.py` into that receipt
directory, and put it first on the diagnostic process's PYTHONPATH, as in the
original collector. No passive tensor observer is installed by this driver.
`returned-arrays.pt` is copied to CPU only after the original RPC; exclude this
transport from phase timing. This driver is not a substitute for the original
official operator assertions or evidence-restoration tests.

## Applicable checks and scale

The existing same-bank stager now also has a default-disabled mode. Unlike
the one-RPC collector above, it schedules the original cold OFF, warm OFF and
warm ON variants on one initialized base actor and one immutable prefix bank:

```powershell
& 'C:/Users/Administrator/miniconda3/python.exe' -X utf8 research/temporary/rl_upstream_alignment_20260929/run_native_prefix_reuse_workload.py --devices 2 3 --current-formal-owner --base-model --phase-only --warm-phases --request-offset 84 --native-conv-initial-states
```

This command completed with the original actual long B8 input on 2026-10-07.
It checks GPU occupancy before staging, uses
`prepare_isolated_owner_paths.py` to link the current DT tree and copy only the
two prepared DT owners and one HF owner, and inserts the HF candidate directory
on the original qwen3_5 package's standard `__path__` before its model import.
The installed package files and formal paths are never overwritten. The
original resource JSON, caches, actor, factory and finite libraries remain in
use. The stage/source receipt includes the isolated paths and hashes; the
original diagnosis also checks the actual imported owners before its first
variant. Its new mode is mutually exclusive with prefetch and the other
candidate/observer modes. The original reported phase timing, PSS, physical
free memory and raw returned Q/V/A comparisons remain the same. It does not
invent a whole-DT tolerance or replace the original operator assertions.

The completed long operator receipts retain the actual BF16 input and frozen
BF16 weights. They reuse causal-conv1d v1.5.0's original output/dx assertion
expressions and tolerance assignments, with test-source SHA
`c15131c88911cf7e942fdd693cfbddd3e2b4d29b0acbfd28ca69b7fd6cb965bf`.
All nine expressions pass on both ranks. BF16 weights are outside that
test's original FP32-weight fixture, which is recorded explicitly. No new
tolerance, corrective scale or whole-DT/PPO guarantee is inferred.

The actual original native concat+kernel took 25.475/22.611 ms, and public
initial_states native took 1.083/0.814 ms. The public initial-state input VJP
took 7.328/5.359 ms, compared with 1.776/1.740 ms on already-concatenated input.
Counting root and replay native calls and the finite VJP increment gives
1.038/0.959 seconds conditional savings per 24-GDN-layer B4 call. Root-only
would give 0.585/0.523 seconds. These are operator-budget calculations, not
measured whole-DT speedup; capture transport, compact-layout copies, the small
window snapshot, FSDP and peer waits still need their original phase timings.

Before accepting: check the actual native/cache states and suffix output with
the same official operator assertions, including a positive original compact
start; inspect original DT operator counts and actual imported hashes; compare
original returned credit/conservation/evidence behavior under its existing
checks; record physical mx-smi and phase PSS/cgroup. Preserve the official
tolerances. No integrated candidate correctness or speed claim is made yet.


## Completed measurements and residual scope

The same immutable prefix bank and same actor yielded critical warm DT time
11.32949 to 9.79171 seconds on the actual 12k–13.5k input, saving13.57%.
Root and per-layer replay each saved about0.67s. The exact32768 capacity
branch saved4.36%/4.62%; all original returned Q/V/A tensors were finite.
This capacity branch uses the existing synthetic helper, not task efficacy or
a new actor update test. Official causal-conv assertions passed for the actual
primitive; no whole-DT tolerance was invented.

The old local0.02 conservation diagnostic is not an FA/FLA assertion. It
reports residual failures in OFFcold, OFFwarm and ON with identical counts.
These observations, raw Q/V/A differences and source hashes are retained in
`runtime-capacity-20261007-v1/read-only-capacity-analysis.json`. No corrective
scale, clipping or threshold change was introduced. Whole-DT conservation
is not reported as passing, and this diagnostic is not used to invent a new
training gate.

The historical `prepare_appworld_native_conv_initial_states.py` namespace
preparation is not the accepted production import route. The exact canonical
owner change, original backup SHA, CPU import verification and new submitted
source are recorded under `appworld-efficiency-20261007/native-conv-canonical-
owner-v2`. No LOOP wait/retry or service parameters were altered.
