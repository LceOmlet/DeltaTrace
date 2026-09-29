#!/bin/bash
set -euo pipefail
R=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922
A=$R/receipts/upstream-alignment-20260929
export DT_ROOT=$R/releases/64e5876 DT_ENVIRONMENT_JSON=$R/releases/64e5876/environment.json
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export VERL_ROOT=$R/candidates/official-verl-20bd331-native-reward
export PYTHONPATH=$A/dt-candidate/experiments/rl:$DT_ROOT/experiments/rl:$DT_ROOT/clean/qwen35:$DT_ROOT:$VERL_ROOT
export CUDA_VISIBLE_DEVICES=7 MACA_VISIBLE_DEVICES=7 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29761
export DT_TASK=Sokoban DT_MAX_STEPS=15 DT_MAX_LENGTH=32768
export VERL_TRIM_SHARED_PADDING=1 VERL_TRIM_RESPONSE_HEAD=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$VENV_PYTHON" -u "$A/verify_native_reward_model.py"
