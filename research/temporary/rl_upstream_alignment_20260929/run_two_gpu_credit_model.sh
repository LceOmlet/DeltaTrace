#!/bin/bash
set -euo pipefail
R=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922
A=$R/receipts/upstream-alignment-20260929
export DT_ROOT=$A/distributed-runtime DT_ENVIRONMENT_JSON=$A/distributed-runtime/environment.json
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export VERL_ROOT=$R/candidates/official-verl-20bd331-distributed-dt
export PYTHONPATH=$DT_ROOT/experiments/rl:$DT_ROOT/clean/qwen35:$DT_ROOT:$VERL_ROOT
export CUDA_VISIBLE_DEVICES=2,3
unset MACA_VISIBLE_DEVICES RAY_TMPDIR RAY_ADDRESS
export DT_TASK=Sokoban DT_MAX_STEPS=15 DT_MAX_LENGTH=32768
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 VERL_TRIM_SHARED_PADDING=1
"$VENV_PYTHON" -u "$A/verify_two_gpu_credit_model.py"
