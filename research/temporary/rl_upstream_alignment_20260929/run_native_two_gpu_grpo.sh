#!/bin/bash
# Bounded native distributed connection/update check, not a formal experiment.
set -euo pipefail
R=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922
A=$R/receipts/upstream-alignment-20260929
export DT_ROOT=$A/distributed-runtime DT_ENVIRONMENT_JSON=$A/distributed-runtime/environment.json
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export VERL_ROOT=$R/candidates/official-verl-20bd331-distributed-dt
export CUDA_VISIBLE_DEVICES=0,1
unset MACA_VISIBLE_DEVICES
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export METHOD=grpo ENV_NAME=Webshop
export TRAIN_SIZE=8 GROUP_SIZE=4 VAL_SIZE=1 VAL_DATA_SIZE=1
export MINI_BATCH_SIZE=64 ACTOR_MICRO_BATCH_SIZE=4 ACTOR_OFFLOAD_POLICY=True
export MAX_STEPS=2 MAX_RESPONSE=512 MAX_TOTAL_TOKENS=32768
export VAL_BEFORE_TRAIN=False ENABLE_THINKING=False TOTAL_EPOCHS=1
export SAVE_FREQ=1 RESUME_MODE=disable
export DATA_ROOT=$A/two-gpu-grpo-data CHECKPOINT_DIR=$A/two-gpu-grpo-checkpoint
export ROLLOUT_DATA_DIR=$A/two-gpu-grpo-rollouts
unset RAY_TMPDIR RAY_ADDRESS
bash "$DT_ROOT/experiments/rl/run_verl_agent.sh" trainer.n_gpus_per_node=2 trainer.total_training_steps=1
