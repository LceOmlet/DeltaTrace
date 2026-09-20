# RL development branch: upstream environments and trainers

This branch deliberately does not contain a local PPO, GRPO, rollout, or
LoRA implementation. Training is delegated to the pinned
[verl-agent2](https://github.com/blackkiring/verl-agent2) fork of VERL. Its
environment managers are used for WebShop, Sokoban, and AppWorld; this branch
only supplies a prompt-parquet helper, a launcher, and the DeltaTrace credit
adapter.

The upstream revision used for the A6000 setup is:

```
verl-agent2 732f37acd7684b8c24d14ba3ededfe9fab1ed472
```

The remote environment already contains the heavyweight CUDA stack and these
RL/task packages. The versions are recorded so another host can reproduce the
setup without silently replacing its existing PyTorch:

```
verl-agent2 (editable, --no-deps)
trl==1.13.0
peft==0.21.0
bitsandbytes==0.50.2
gym-sokoban==0.0.6
ray==2.55.1
torchdata==0.11.0
pyserini==0.22.1 (--no-deps)
qwen-vl-utils==0.0.14
```

WebShop uses Princeton's official text environment and its official 1k
product/instruction files plus a Pyserini Lucene index. AppWorld uses the
official `appworld install --repo` and `appworld download data` commands and
the official environment server. Sokoban uses the gym-sokoban dependency
through verl-agent2's environment package. No task environment is copied into
this repository.

After cloning the three official repositories, prepare their assets with
`experiments/rl/prepare_task_assets.sh`. It only creates symlinks into the
upstream environment and runs AppWorld's official installer/download commands;
it does not generate a replacement task implementation.

## Run

On a prepared Linux host, set `DT_ROOT` to the checkout and point
`MODEL_PATH` to a local Qwen3.5 checkpoint. The launcher uses the upstream
`verl.trainer.main_ppo` entry point. `METHOD=grpo` selects upstream GRPO
(`algorithm.adv_estimator=grpo`); `METHOD=ppo` selects upstream GAE/PPO;
`METHOD=counterfactual` (or `METHOD=dt`) selects the pinned trainer's
counterfactual estimator. The same command supports all three tasks:

```bash
DT_ROOT=/path/to/DeltaTrace \
MODEL_PATH=/data/models/Qwen3.5-9B \
CUDA_VISIBLE_DEVICES=0 \
ENV_NAME=Webshop METHOD=grpo \
bash experiments/rl/run_verl_agent.sh

DT_ROOT=/path/to/DeltaTrace \
MODEL_PATH=/data/models/Qwen3.5-9B \
CUDA_VISIBLE_DEVICES=0 \
ENV_NAME=Sokoban METHOD=ppo \
bash experiments/rl/run_verl_agent.sh

DT_ROOT=/path/to/DeltaTrace \
MODEL_PATH=/data/models/Qwen3.5-9B \
CUDA_VISIBLE_DEVICES=0 \
ENV_NAME=AppWorld METHOD=grpo \
TRAIN_SIZE=1 VAL_SIZE=1 GROUP_SIZE=1 \
bash experiments/rl/run_verl_agent.sh

DT_ROOT=/path/to/DeltaTrace \
MODEL_PATH=/data/models/Qwen3.5-9B \
CUDA_VISIBLE_DEVICES=0 \
ENV_NAME=Webshop METHOD=counterfactual \
TRAIN_SIZE=1 VAL_SIZE=1 GROUP_SIZE=4 MAX_STEPS=2 \
bash experiments/rl/run_verl_agent.sh
```

The A6000 profile is deliberately memory bounded: `GROUP_SIZE=4` gives four
environment rollouts per prompt for GRPO, `MINI_BATCH_SIZE=4` is the upstream
policy minibatch, and `MAX_TOTAL_TOKENS=32768` is the per-GPU token budget.
It uses upstream PEFT LoRA (`LORA_RANK=1` by default; raise it only after
checking peak VRAM), disables the reference/KL worker when KL is not requested,
disables dynamic batching and `torch.compile`, and keeps the single-GPU actor
parameters on CUDA (`ACTOR_OFFLOAD_POLICY=False`) while offloading only the
optimizer state. This avoids per-layer CPU/CUDA copies; set
`ACTOR_OFFLOAD_POLICY=True` only when host memory is the priority. The launcher
also serializes HF generation with `ROLLOUT_MICRO_BATCH_SIZE=1`, so the four
rollouts do not multiply the long-context activation peak. Set
`VAL_BEFORE_TRAIN=True` or `TEST_FREQ=1` when an online validation pass is
desired; they default off to avoid duplicating the rollout during a
memory/throughput run.

For a real 32k resource check, set `PROMPT_FILL_TOKENS=30000`. The helper puts
the filler in a prior assistant turn so the upstream collector preserves it
when it appends the official environment observation:

```bash
PROMPT_FILL_TOKENS=30000 MAX_PROMPT=32256 MAX_TOTAL_TOKENS=32768 MAX_RESPONSE=64 \
GROUP_SIZE=4 MINI_BATCH_SIZE=4 ROLLOUT_MICRO_BATCH_SIZE=1 \
TRAIN_SIZE=1 VAL_SIZE=1 MAX_STEPS=1 ENV_NAME=Webshop METHOD=counterfactual \
bash experiments/rl/run_verl_agent.sh
```

On the A6000/Qwen3.5-9B setup this completed one upstream optimizer step for
WebShop, Sokoban, and AppWorld with `prompt_length` 30219--31802 and no OOM;
peak allocated/reserved memory was `38.482/42.932 GiB`. The exact per-task
logs and throughput are recorded in
`results_counterfactual_32k_a6000.json`.

That boundary fixture uses one environment step. For multi-turn episodes the
official observation history grows, so leave headroom. The two-step Sokoban
semantic smoke is recorded in `results_counterfactual_multistep.json`.

The launcher uses the upstream HF rollout backend (`rollout.name=hf`) to avoid
assuming an incompatible vLLM build for Qwen3.5; switching to vLLM is an
explicit host-level choice.

For the text-only Qwen3.5 checkpoint, Sokoban defaults to the upstream
`tiny_rgb_array` ASCII observation (`SOKOBAN_MODE=tiny_rgb_array`). Set
`SOKOBAN_MODE=rgb_array` only when using a vision-capable processor and model.
For multi-turn Qwen text runs, the patch preserves decoded actions before
calling the upstream projection, whose Sokoban implementation normalizes its
input list in place.

AppWorld requires its official service before launching. The environment
manager assigns one port per training rollout and one validation port. Thus
the GRPO smoke above (`TRAIN_SIZE=1`, `GROUP_SIZE=4`, `VAL_SIZE=1`) needs five
official service instances. Start them from the AppWorld repository root so
the server resolves `./data/tasks`:

```bash
cd "$DT_ROOT/third_party/appworld"
echo 7000 > appworld_ports.ports
echo 7001 >> appworld_ports.ports
echo 7002 >> appworld_ports.ports
echo 7003 >> appworld_ports.ports
echo 7004 >> appworld_ports.ports
for port in 7000 7001 7002 7003 7004; do
  nohup "$DT_ROOT/env/bin/appworld" serve environment --port "$port" \
    > "/tmp/appworld-${port}.log" 2>&1 &
done
```

The upstream AppWorld wrapper reads `appworld_ports.ports` from its current
working directory. Run the launcher with `cd "$DT_ROOT/third_party/appworld"`
or keep the port file in the launcher working directory.

## DeltaTrace credit contract

`counterfactual.py` and `deltatrace_credit.py` implement only the policy-credit
contract: independently sampled reference actions produce
`G(selected) - mean(G(reference))`, which is the policy-average
`Q(h,a)-V(h)` signal. The patched upstream collector uses a leave-one-out mean
over the same initial-state rollout group, so the selected rollout is excluded
from its own reference set. This credit is attached only to the first
environment action; later actions are masked because the group rollouts no
longer share a history. The upstream VERL optimizer remains the optimizer and
the signed DeltaTrace input vector is never used as a token-wise advantage.

The `counterfactual` estimator is a reward-level policy signal, not the
model-log-probability `root_effect` returned by the attribution runner.
`dt_action_smoke.py` separately checks that the owner DeltaTrace runner can
produce and audit an action intervention; it is diagnostic and must not be
substituted for the environment return in the policy loss.

`patch_verl_agent2.py` applies the pinned tree's compatibility fixes and adds
only the counterfactual estimator/collector seam described above. It does not
copy the trainer, alter environment dynamics, or replace the upstream actor
optimizer.

The local invariant test is:

```bash
python -m experiments.rl.test_counterfactual
```
