#!/bin/bash
# Small real-loop diagnostic: no altered owner methods or synthetic rewards.
set -euo pipefail
R=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922
A=$R/receipts/upstream-alignment-20260929
export DT_ROOT=$R/releases/fc2e6c2 DT_ENVIRONMENT_JSON=$R/releases/fc2e6c2/environment.json
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export VERL_ROOT=$R/candidates/official-verl-20bd331-distributed-dt
export PYTHONPATH=$A:$DT_ROOT/experiments/rl:$DT_ROOT/clean/qwen35:$DT_ROOT:$VERL_ROOT
: "${CUDA_VISIBLE_DEVICES:?Choose two verified free GPUs}"
unset MACA_VISIBLE_DEVICES RAY_TMPDIR RAY_ADDRESS
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export METHOD=grpo ENV_NAME=Sokoban TRAIN_SIZE=2 GROUP_SIZE=4 VAL_SIZE=1 VAL_DATA_SIZE=1
export MAX_TOTAL_TOKENS=32768 MAX_RESPONSE=512 MAX_STEPS=2 TOTAL_EPOCHS=2
export MINI_BATCH_SIZE=64 ACTOR_MICRO_BATCH_SIZE=4 ACTOR_OFFLOAD_POLICY=True
export VAL_BEFORE_TRAIN=False SAVE_FREQ=-1 TEST_FREQ=-1 RESUME_MODE=disable ENABLE_THINKING=False
export DT_RAY_NUM_CPUS=8
export DATA_ROOT=$A/probability-native-trainer/data
export CHECKPOINT_DIR=$A/probability-native-trainer/checkpoints
export ROLLOUT_DATA_DIR=$A/probability-native-trainer/rollouts
mkdir -p "$A/probability-native-trainer"
bash "$A/run_verl_agent_dataloader0.sh" trainer.n_gpus_per_node=2 --cfg job --resolve > "$A/probability-native-trainer/resolved-config.log" 2>&1
exec "$VENV_PYTHON" -u "$A/observe_original_trainer.py"
