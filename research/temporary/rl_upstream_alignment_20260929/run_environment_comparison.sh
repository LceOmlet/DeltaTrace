#!/usr/bin/env bash
set -euo pipefail
R=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922
A=$R/receipts/upstream-alignment-20260929
export DT_ROOT=$R/releases/c9cd147
export DT_ENVIRONMENT_JSON=$DT_ROOT/environment.json
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
unset MACA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=''
export PATH=$(dirname "$VENV_PYTHON"):$PATH
export LOOP_ROOT=$R/third_party/ml-loop-f14107a976e5793990329d3193df4742076c5a1d
export PYTHONPATH=$R/environments/loop-extras-f14107a:$A:$LOOP_ROOT${PYTHONPATH:+:$PYTHONPATH}
export APPWORLD_ROOT=$R/third_party/appworld-42b5bcf3cd334fee33f0c37c02070a9f5807add5
cd "$LOOP_ROOT"
"$VENV_PYTHON" "$A/loop_owner_recipe.py" --owner-root "$LOOP_ROOT" --output "$A/loop-environment-config.json"
"$VENV_PYTHON" "$A/compare_loop_environment.py" \
  --owner-root "$LOOP_ROOT" \
  --project-root "$R/candidates/official-verl-20bd331-distributed-dt" \
  --pristine-project-root "$A/official" \
  --asset-root "$APPWORLD_ROOT" \
  --run-root "$A/environment-comparison-runtime" \
  --output "$A/loop-project-environment-comparison.json"
