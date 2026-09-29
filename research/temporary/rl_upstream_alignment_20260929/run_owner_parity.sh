#!/bin/bash
set -euo pipefail
export DT_RUNTIME_ROOT=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922
export DT_ROOT="$DT_RUNTIME_ROOT/releases/64e5876"
export DT_ENVIRONMENT_JSON="$DT_ROOT/environment.json"
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export CUDA_VISIBLE_DEVICES=7 MACA_VISIBLE_DEVICES=7 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29759
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$DT_ROOT/experiments/rl:$DT_ROOT/clean/qwen35:$DT_ROOT:$VERL_ROOT:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 FLASH_ATTENTION_DETERMINISTIC=1 PYTHONHASHSEED=0
export DT_TASK=Sokoban DT_MAX_STEPS=15 DT_MAX_LENGTH=32768
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
AUDIT="$DT_RUNTIME_ROOT/receipts/upstream-alignment-20260929"
set +e
"$VENV_PYTHON" -u "$AUDIT/verify_short_owner_parity.py" \
  --paired-owner-source "$AUDIT/official/verl/workers/actor/dp_actor.py" \
  --fixed-credit-from "$DT_RUNTIME_ROOT/receipts/training-setup/short-parity-paired-reshard.pt" \
  --actor-microbatch 4 --trim-shared-padding --trim-response-head --no-owner-math \
  --disable-bf16-reduced-reduction \
  --output "$AUDIT/native-loss-fp32-reduction.json" --artifacts "$AUDIT/native-loss-fp32-reduction.pt"
rc=$?
printf '%s\n' "$rc" > "$AUDIT/native-loss-fp32-reduction.exit-code"
exit "$rc"
