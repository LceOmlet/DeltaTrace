#!/bin/bash
set -euo pipefail
R=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922
A=$R/receipts/upstream-alignment-20260929
export DT_ROOT=$R/releases/64e5876
export DT_ENVIRONMENT_JSON=$DT_ROOT/environment.json
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export CUDA_VISIBLE_DEVICES=6 MACA_VISIBLE_DEVICES=6
export PYTHONPATH=$DT_ROOT/experiments/rl:$DT_ROOT/clean/qwen35:$DT_ROOT:$VERL_ROOT
SOURCES=$R/receipts/training-setup/official-kernel-tests
"$VENV_PYTHON" -u "$A/dt-candidate/experiments/rl/verify_dt_fla_partition.py" \
 --sources "$SOURCES" --coefficient-start 64 --initial-state --native-cache-split --dtype float16 \
 --output "$A/dt-fla-original-tolerances.json"
LIBRARY=$("$VENV_PYTHON" -c 'import json,os; print(json.load(open(os.environ["DT_ENVIRONMENT_JSON"]))["qwen35"]["finite_library"])')
"$VENV_PYTHON" -u "$DT_ROOT/experiments/rl/verify_dt_fa_candidate.py" \
 --candidate "$LIBRARY" --sources "$SOURCES" --cached-native \
 --output "$A/dt-fa-original-tolerances.json"
