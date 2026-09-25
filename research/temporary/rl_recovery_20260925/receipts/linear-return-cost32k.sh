#!/bin/bash
export DT_ROOT=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/releases/64e5876
export DT_ENVIRONMENT_JSON="$DT_ROOT/environment.json"
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export CUDA_VISIBLE_DEVICES=7 MACA_VISIBLE_DEVICES=7 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29758
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/rollout-major-cost:$DT_ROOT/experiments/rl:$DT_ROOT/clean/qwen35:$DT_ROOT:$VERL_ROOT:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 FLASH_ATTENTION_DETERMINISTIC=1 PYTHONHASHSEED=0
export DT_TASK=Sokoban DT_MAX_STEPS=15 DT_MAX_LENGTH=32768
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
"$VENV_PYTHON" -u /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/rollout-major-cost/benchmark_saved_credit.py --source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/rollout-major-cost/linear-return-capacity-input.json --output /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/rollout-major-cost/linear-return-cost32k.json --parameter-offload-policy
rc=$?
printf '%s\n' "$rc" > /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/rollout-major-cost/linear-return-cost32k.exitcode
exit "$rc"
