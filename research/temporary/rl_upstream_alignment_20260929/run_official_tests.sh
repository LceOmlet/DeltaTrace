#!/bin/bash
set -euo pipefail
R=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922
export DT_ROOT="$R/releases/64e5876"
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export CUDA_VISIBLE_DEVICES=6 MACA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"
A="$R/receipts/upstream-alignment-20260929"
cd "$VERL_ROOT"
set +e
"$VENV_PYTHON" -m pytest -q "$A/official/tests/trainer/ppo/test_metric_utils.py" \
  "$A/official/tests/gpu_utility/test_torch_functional.py" \
  --junitxml="$A/official-tests.xml"
rc=$?
printf '%s\n' "$rc" > "$A/official-tests.exit-code"
exit "$rc"
