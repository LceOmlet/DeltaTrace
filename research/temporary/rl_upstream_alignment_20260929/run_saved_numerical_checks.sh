#!/bin/bash
set -euo pipefail
R=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922
A=$R/receipts/upstream-alignment-20260929
export DT_ROOT=$R/releases/fc2e6c2 DT_ENVIRONMENT_JSON=$R/releases/fc2e6c2/environment.json
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export VERL_ROOT=$R/candidates/official-verl-20bd331-distributed-dt
export PYTHONPATH=$DT_ROOT/experiments/rl:$DT_ROOT/clean/qwen35:$DT_ROOT:$VERL_ROOT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset MACA_VISIBLE_DEVICES RAY_TMPDIR RAY_ADDRESS
export CUDA_VISIBLE_DEVICES=${DIAGNOSTIC_GPU:?Choose a verified idle physical GPU}
case "$1" in
  fa)
    exec "$VENV_PYTHON" -u "$A/verify_saved_fa_dtypes.py" \
      --operands "$A/actual-dt-layer3-boundaries.pt" \
      --sources "$R/receipts/training-setup/official-kernel-tests" \
      --output "$A/actual-finite-consumer-fa-dtypes.json" ;;
  operator)
    export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29758
    export DT_DUPLICATE_OPERATOR_PROBE=1 DT_TASK=Sokoban DT_MAX_STEPS=15 DT_MAX_LENGTH=32768
    exec "$VENV_PYTHON" -u "$A/inspect_duplicate_native_rows.py" ;;
  boundary)
    export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=29758 DT_DUPLICATE_BOUNDARY_PROBE=1
    export DT_TASK=Sokoban DT_MAX_STEPS=15 DT_MAX_LENGTH=32768
    exec "$VENV_PYTHON" -u "$A/inspect_duplicate_native_rows.py" ;;
  dtype)
    exec "$VENV_PYTHON" -u "$A/verify_saved_fla_dtypes.py" \
      --operands "$A/actual-dt-layer2-finite-consumer.pt" \
      --sources "$R/receipts/training-setup/official-kernel-tests" \
      --output "$A/actual-finite-consumer-fla-dtypes.json" ;;
esac
