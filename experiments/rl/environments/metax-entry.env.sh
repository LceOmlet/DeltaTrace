#!/usr/bin/env bash
# Existing verified runtime plus official environment-only leaf dependencies.
# No installation, cache clearing, service startup or device selection occurs.
export DT_RUNTIME_ROOT=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922
export DT_ROOT=$DT_RUNTIME_ROOT/releases/c9cd147
export DT_ENVIRONMENT_JSON=$DT_ROOT/environment.json
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export VERL_ROOT=$DT_RUNTIME_ROOT/candidates/official-verl-20bd331-env-entry-20260930
export DT_ENTRY_ROOT=$DT_RUNTIME_ROOT/receipts/environment-only-20260930/entry
export AGENTGYM_RL_ROOT=$DT_RUNTIME_ROOT/third_party/AgentGym-RL-82402a99c62a293735a3f412fb8ac9a600673bc0
export TEXTCRAFT_DATA=$DT_RUNTIME_ROOT/third_party/textcraft-data
export LOOP_ROOT=$DT_RUNTIME_ROOT/third_party/ml-loop-f14107a976e5793990329d3193df4742076c5a1d
export TEXTCRAFT_EXTRAS=$DT_RUNTIME_ROOT/environments/textcraft-extras-20260930
export LOOP_EXTRAS=$DT_RUNTIME_ROOT/environments/loop-extras-f14107a
# Missing environment leaves are appended by their factory via site.addsitedir.
# In particular, LOOP's archived accelerate must never precede the model runtime.
export PYTHONPATH=$DT_ENTRY_ROOT:$DT_RUNTIME_ROOT/third_party/skyrl-gym-7d94cc:$DT_RUNTIME_ROOT/third_party/AgentGym-d014732d9fe39b975c368c03749bfd50950067f6/agentenv:$LOOP_ROOT:$VERL_ROOT:$DT_ROOT/experiments/rl:$DT_ROOT:$DT_RUNTIME_ROOT/receipts/upstream-alignment-20260929:${PYTHONPATH:-}
export PATH=$(dirname "$VENV_PYTHON"):$PATH
export VERL_ENTRY_BASELINE=$DT_RUNTIME_ROOT/candidates/official-verl-20bd331-distributed-dt/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
export DT_TOKENIZER_PATH=$MODEL_PATH
export DT_LOOP_ENVIRONMENT_RECEIPT=$DT_ENTRY_ROOT/loop-project-environment-comparison.json
export TOKENIZERS_PARALLELISM=false VERL_TRIM_SHARED_PADDING=1 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset MACA_VISIBLE_DEVICES RAY_TMPDIR RAY_ADDRESS
