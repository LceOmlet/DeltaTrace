#!/usr/bin/env bash
set -euo pipefail

# Thin launcher around the pinned upstream verl-agent2 trainer.  It does not
# reimplement PPO/GRPO, rollout, or any environment.  Override variables for
# another machine instead of editing the command below.

METHOD="${METHOD:-grpo}"             # grpo, ppo, or dt
ENV_NAME="${ENV_NAME:-Webshop}"      # Webshop, Sokoban, or AppWorld
DT_ROOT="${DT_ROOT:-$PWD}"
MODEL_PATH="${MODEL_PATH:-/data/liangchen/models/Qwen3.5-9B}"
VERL_ROOT="${VERL_ROOT:-$DT_ROOT/third_party/verl-agent2}"
APPWORLD_ROOT="${APPWORLD_ROOT:-$DT_ROOT/third_party/appworld}"
VENV_PYTHON="${VENV_PYTHON:-$DT_ROOT/env/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TRAIN_SIZE="${TRAIN_SIZE:-4}"
VAL_SIZE="${VAL_SIZE:-4}"
GROUP_SIZE="${GROUP_SIZE:-4}"           # environment rollouts per prompt (GRPO group)
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-4}"
MAX_PROMPT="${MAX_PROMPT:-32256}"
MAX_RESPONSE="${MAX_RESPONSE:-512}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
LORA_RANK="${LORA_RANK:-1}"
LORA_ALPHA="${LORA_ALPHA:-2}"
ACTOR_STRATEGY="${ACTOR_STRATEGY:-fsdp2}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
ACTOR_CPU_OFFLOAD="${ACTOR_CPU_OFFLOAD:-False}"
ACTOR_OFFLOAD_POLICY="${ACTOR_OFFLOAD_POLICY:-False}"
FSDP_MIN_PARAMS="${FSDP_MIN_PARAMS:-0}"
FSDP_RESHARD_AFTER_FORWARD="${FSDP_RESHARD_AFTER_FORWARD:-True}"
HF_FSDP_WRAP="${HF_FSDP_WRAP:-False}"
ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-1}"
PROMPT_FILL_TOKENS="${PROMPT_FILL_TOKENS:-0}"
SOKOBAN_MODE="${SOKOBAN_MODE:-tiny_rgb_array}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
TEST_FREQ="${TEST_FREQ:--1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-15}"
CHAT_TEMPLATE_ARGS=()
if [[ -n "${ENABLE_THINKING:-}" ]]; then
  CHAT_TEMPLATE_ARGS+=("+data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING")
fi

case "$METHOD" in
  grpo) ADV_ESTIMATOR=grpo ;;
  ppo) ADV_ESTIMATOR=gae ;;
  dt) ADV_ESTIMATOR=deltatrace ;;
  *) echo "METHOD must be grpo, ppo, or dt" >&2; exit 2 ;;
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
export DT_ROOT
if [[ -z "${DT_ENVIRONMENT_JSON:-}" ]]; then
  if [[ -f "$DT_ROOT/environment.json" ]]; then
    export DT_ENVIRONMENT_JSON="$DT_ROOT/environment.json"
  else
    export DT_ENVIRONMENT_JSON="$DT_ROOT/../../environment.json"
  fi
fi
if [[ -n "${DT_OFFICIAL_ROOT:-}" ]]; then export DT_OFFICIAL_ROOT; fi
export PYTHONPATH="$DT_ROOT/experiments/rl:$DT_ROOT:$VERL_ROOT:${VERL_ROOT}/agent_system/environments/env_package/webshop/webshop:${APPWORLD_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VERL_TRIM_SHARED_PADDING=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"
# The pinned upstream tree requests FlashAttention 2 by default, but its
# optional CUDA extension may be unavailable on older-glibc hosts.  Hugging
# Face SDPA is the portable upstream attention backend; set this to
# flash_attention_2 on a host with a compatible flash-attn build.
if [[ "$METHOD" == "dt" ]]; then
  export VERL_ATTN_IMPLEMENTATION="${VERL_ATTN_IMPLEMENTATION:-sdpa}"
  export DT_TASK="$ENV_NAME" DT_MAX_STEPS="$MAX_STEPS" DT_MAX_LENGTH="$MAX_TOTAL_TOKENS"
  # Keep attribution inside the same 32k cap, including its event query and
  # target. Reserve this space at the upstream prompt-length boundary rather
  # than truncating an already-generated action or exceeding the cap later.
  DT_READOUT_TOKENS=$("$VENV_PYTHON" - "$MODEL_PATH" "$ENV_NAME" "$MAX_STEPS" <<'PY'
import sys
from transformers import AutoTokenizer
from reward_readout import RewardAlphabet
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
print(RewardAlphabet.for_task(sys.argv[2]).readout_token_budget(tokenizer, int(sys.argv[3])))
PY
)
  DT_PROMPT_LIMIT=$((MAX_TOTAL_TOKENS - MAX_RESPONSE - DT_READOUT_TOKENS))
  if (( DT_PROMPT_LIMIT < 1 )); then
    echo "DT context cap cannot fit the response and event readout" >&2; exit 2
  fi
  if (( MAX_PROMPT > DT_PROMPT_LIMIT )); then MAX_PROMPT="$DT_PROMPT_LIMIT"; fi
  echo "DT context budget: prompt=$MAX_PROMPT response=$MAX_RESPONSE readout=$DT_READOUT_TOKENS cap=$MAX_TOTAL_TOKENS"
else
  export VERL_ATTN_IMPLEMENTATION="${VERL_ATTN_IMPLEMENTATION:-sdpa}"
fi
if [[ "$ACTOR_CPU_OFFLOAD" == "True" ]]; then
  export VERL_ACTOR_CPU_OFFLOAD=1
else
  export VERL_ACTOR_CPU_OFFLOAD=0
fi
if [[ "$HF_FSDP_WRAP" == "True" ]]; then
  export VERL_ENABLE_HF_FSDP_WRAP=1
else
  export VERL_ENABLE_HF_FSDP_WRAP=0
fi

# Parallel task runs share this pinned checkout. Serialize the existing
# patcher rather than allowing simultaneous writes to installed Python files.
flock "$VERL_ROOT/.deltatrace-patch.lock" \
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
  --output "$DATA_ROOT/train.parquet" --size "$TRAIN_SIZE" --split train \
  --prompt-fill-tokens "$PROMPT_FILL_TOKENS"
"$VENV_PYTHON" "$DT_ROOT/experiments/rl/prepare_agent_data.py" \
  --output "$DATA_ROOT/test.parquet" --size "$VAL_SIZE" --split test \
  --prompt-fill-tokens "$PROMPT_FILL_TOKENS"

cd "$VERL_ROOT"
if [[ "$ENV_NAME" == "AppWorld" && -f "$APPWORLD_ROOT/appworld_ports.ports" ]]; then
  ln -sfn "$APPWORLD_ROOT/appworld_ports.ports" "$VERL_ROOT/appworld_ports.ports"
fi
if [[ "$ENV_NAME" == "AppWorld" && ! -f "$VERL_ROOT/appworld_ports.ports" ]]; then
  echo "AppWorld needs appworld_ports.ports; start the official service first" >&2
  exit 2
fi
# Use the original PPO clipped objective, without the optional dual clipping.
exec "$VENV_PYTHON" -m verl.trainer.main_ppo \
  +ray_kwargs.ray_init.num_cpus="${DT_RAY_NUM_CPUS:-null}" \
  algorithm.adv_estimator="$ADV_ESTIMATOR" \
  algorithm.gamma=1.0 \
  actor_rollout_ref.actor.clip_ratio_c=inf \
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
  actor_rollout_ref.model.lora_rank="$LORA_RANK" \
  actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_TOTAL_TOKENS" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy="$ACTOR_STRATEGY" \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.offload_policy="$ACTOR_OFFLOAD_POLICY" \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload="$PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params="$FSDP_MIN_PARAMS" \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward="$FSDP_RESHARD_AFTER_FORWARD" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  +actor_rollout_ref.rollout.micro_batch_size="$ROLLOUT_MICRO_BATCH_SIZE" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  algorithm.use_kl_in_reward=False \
  env.env_name="$ENV_NAME" \
  env.seed=0 \
  env.max_steps="$MAX_STEPS" \
  env.rollout.n="$GROUP_SIZE" \
  env.sokoban.mode="$SOKOBAN_MODE" \
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
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  "${CHAT_TEMPLATE_ARGS[@]}"
