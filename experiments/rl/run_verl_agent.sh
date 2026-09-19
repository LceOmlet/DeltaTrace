#!/usr/bin/env bash
set -euo pipefail

# Thin launcher around the pinned upstream verl-agent2 trainer.  It does not
# reimplement PPO/GRPO, rollout, or any environment.  Override variables for
# another machine instead of editing the command below.

METHOD="${METHOD:-grpo}"             # grpo or ppo
ENV_NAME="${ENV_NAME:-Webshop}"      # Webshop, Sokoban, or AppWorld
DT_ROOT="${DT_ROOT:-$PWD}"
MODEL_PATH="${MODEL_PATH:-/data/liangchen/models/Qwen3.5-9B}"
VERL_ROOT="${VERL_ROOT:-$DT_ROOT/third_party/verl-agent2}"
APPWORLD_ROOT="${APPWORLD_ROOT:-$DT_ROOT/third_party/appworld}"
VENV_PYTHON="${VENV_PYTHON:-$DT_ROOT/env/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TRAIN_SIZE="${TRAIN_SIZE:-4}"
VAL_SIZE="${VAL_SIZE:-4}"
GROUP_SIZE="${GROUP_SIZE:-4}"
MAX_PROMPT="${MAX_PROMPT:-4096}"
MAX_RESPONSE="${MAX_RESPONSE:-512}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"

case "$METHOD" in
  grpo) ADV_ESTIMATOR=grpo ;;
  ppo) ADV_ESTIMATOR=gae ;;
  *) echo "METHOD must be grpo or ppo" >&2; exit 2 ;;
esac
case "$ENV_NAME" in
  Webshop|Sokoban|AppWorld) ;;
  *) echo "ENV_NAME must be Webshop, Sokoban, or AppWorld" >&2; exit 2 ;;
esac

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "VENV_PYTHON does not exist: $VENV_PYTHON" >&2
  exit 2
fi
if [[ ! -d "$VERL_ROOT" ]]; then
  echo "VERL_ROOT does not exist: $VERL_ROOT" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$VERL_ROOT:${VERL_ROOT}/agent_system/environments/env_package/webshop/webshop:${APPWORLD_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"
# The pinned upstream tree requests FlashAttention 2 by default, but its
# optional CUDA extension may be unavailable on older-glibc hosts.  Hugging
# Face SDPA is the portable upstream attention backend; set this to
# flash_attention_2 on a host with a compatible flash-attn build.
export VERL_ATTN_IMPLEMENTATION="${VERL_ATTN_IMPLEMENTATION:-sdpa}"

"$VENV_PYTHON" "$DT_ROOT/experiments/rl/patch_verl_agent2.py" "$VERL_ROOT"
if [[ "$ENV_NAME" == "Webshop" && -d "$VERL_ROOT/agent_system/environments/env_package/webshop/webshop/search_engine/indexes_1k" ]]; then
  # The upstream manager passes num_products=None, which selects `indexes`.
  # The A6000 smoke uses the official 1k index; replace this link with the
  # official full index for a full WebShop run.
  ln -sfn indexes_1k "$VERL_ROOT/agent_system/environments/env_package/webshop/webshop/search_engine/indexes"
fi

DATA_ROOT="${DATA_ROOT:-$HOME/data/verl-agent/delta-agent}"
mkdir -p "$DATA_ROOT"
"$VENV_PYTHON" "$DT_ROOT/experiments/rl/prepare_agent_data.py" \
  --output "$DATA_ROOT/train.parquet" --size "$TRAIN_SIZE" --split train
"$VENV_PYTHON" "$DT_ROOT/experiments/rl/prepare_agent_data.py" \
  --output "$DATA_ROOT/test.parquet" --size "$VAL_SIZE" --split test

cd "$VERL_ROOT"
if [[ "$ENV_NAME" == "AppWorld" && -f "$APPWORLD_ROOT/appworld_ports.ports" ]]; then
  ln -sfn "$APPWORLD_ROOT/appworld_ports.ports" "$VERL_ROOT/appworld_ports.ports"
fi
if [[ "$ENV_NAME" == "AppWorld" && ! -f "$VERL_ROOT/appworld_ports.ports" ]]; then
  echo "AppWorld needs appworld_ports.ports; start the official service first" >&2
  exit 2
fi
exec "$VENV_PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator="$ADV_ESTIMATOR" \
  data.train_files="$DATA_ROOT/train.parquet" \
  data.val_files="$DATA_ROOT/test.parquet" \
  data.train_batch_size="$TRAIN_SIZE" \
  data.val_batch_size="$VAL_SIZE" \
  data.max_prompt_length="$MAX_PROMPT" \
  data.max_response_length="$MAX_RESPONSE" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.offload_policy=True \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  env.env_name="$ENV_NAME" \
  env.seed=0 \
  env.max_steps=15 \
  env.rollout.n="$GROUP_SIZE" \
  env.sokoban.mode=rgb_array \
  env.sokoban.num_boxes=1 \
  env.sokoban.search_depth=30 \
  env.resources_per_worker.num_cpus=0.1 \
  trainer.critic_warmup=0 \
  trainer.logger="['console']" \
  trainer.project_name=delta_trace_agent \
  trainer.experiment_name="${METHOD}_${ENV_NAME}_qwen35_9b" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=1 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.val_before_train=True
