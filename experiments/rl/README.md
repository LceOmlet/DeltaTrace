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
(`algorithm.adv_estimator=grpo`); `METHOD=ppo` selects upstream GAE/PPO. The
same command supports all three tasks:

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

For a real 32k resource check, set `PROMPT_FILL_TOKENS=32000`. The helper puts
the filler in a prior assistant turn so the upstream collector preserves it
when it appends the official environment observation:

```bash
PROMPT_FILL_TOKENS=32000 MAX_PROMPT=32256 MAX_RESPONSE=64 \
GROUP_SIZE=4 MINI_BATCH_SIZE=4 ROLLOUT_MICRO_BATCH_SIZE=1 \
TRAIN_SIZE=1 VAL_SIZE=1 MAX_STEPS=1 ENV_NAME=Webshop METHOD=grpo \
bash experiments/rl/run_verl_agent.sh
```

On the A6000/Qwen3.5-9B setup this completed one upstream GRPO optimizer step
with `prompt_length=32219`, `response_length=64`, peak allocated/reserved
memory `38.482/42.932 GiB`, and no OOM.

The launcher uses the upstream HF rollout backend (`rollout.name=hf`) to avoid
assuming an incompatible vLLM build for Qwen3.5; switching to vLLM is an
explicit host-level choice.

AppWorld requires its official service before launching. The command above
uses one train and one validation worker, so one service is enough for the
smoke. For a real run, start enough ports for the chosen train/validation
batch sizes:

```bash
cd "$DT_ROOT/third_party/appworld"
echo 7000 > appworld_ports.ports
nohup "$DT_ROOT/env/bin/appworld" serve environment --port 7000 \
  > /tmp/appworld7000.log 2>&1 &
```

The upstream AppWorld wrapper reads `appworld_ports.ports` from its current
working directory. Run the launcher with `cd "$DT_ROOT/third_party/appworld"`
or keep the port file in the launcher working directory.

## DeltaTrace credit contract

`counterfactual.py` and `deltatrace_credit.py` implement only the policy-credit
contract: independently sampled reference actions produce
`G(selected) - mean(G(reference))`, which is the policy-average
`Q(h,a)-V(h)` signal. The DeltaTrace signed input vector is retained for
diagnostics and is never used as a token-wise GRPO/PPO advantage. The upstream
VERL optimizer remains the optimizer; the adapter is the narrow place to add
counterfactual endpoint values once the rollout records selected/reference
traces. It does not duplicate VERL's trainer.

`patch_verl_agent2.py` applies one compatibility-only fix to the pinned
upstream tree: it makes the Sokoban factory lazy to avoid that tree's nested
package import cycle under Python 3.12. It changes no environment dynamics or
RL math.

The local invariant test is:

```bash
python -m experiments.rl.test_counterfactual
```
