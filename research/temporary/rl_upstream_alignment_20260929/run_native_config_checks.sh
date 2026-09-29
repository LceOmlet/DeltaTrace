#!/bin/bash
set -euo pipefail
R=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922
A=$R/receipts/upstream-alignment-20260929
export DT_ROOT=$R/releases/64e5876
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export DT_ENVIRONMENT_JSON=$DT_ROOT/environment.json
export DT_ROOT=$A/dt-candidate
export VERL_ROOT=$R/candidates/official-verl-20bd331-native-reward
export PYTHONPATH=$R/releases/64e5876/experiments/rl:$R/releases/64e5876:$VERL_ROOT
export CUDA_VISIBLE_DEVICES='' MACA_VISIBLE_DEVICES=''
export ROLLOUT_BACKEND=vllm
"$VENV_PYTHON" -u "$A/check_launcher_configs.py" "$@"
