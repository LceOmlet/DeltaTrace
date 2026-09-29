#!/bin/bash
set -euo pipefail
R=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922
A=$R/receipts/upstream-alignment-20260929
export DT_ROOT=$R/releases/fc2e6c2 DT_ENVIRONMENT_JSON=$R/releases/fc2e6c2/environment.json
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export VERL_ROOT=$R/candidates/official-verl-20bd331-distributed-dt
export PYTHONPATH=$DT_ROOT/experiments/rl:$DT_ROOT/clean/qwen35:$DT_ROOT:$VERL_ROOT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
: "${CUDA_VISIBLE_DEVICES:?Select two verified idle GPUs}"
unset MACA_VISIBLE_DEVICES RAY_TMPDIR RAY_ADDRESS
exec "$VENV_PYTHON" -u "$A/verify_rollout_nan.py"
