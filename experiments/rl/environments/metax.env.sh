#!/usr/bin/env bash
# Source on the checked C550 host. Reuse existing model, Python, kernels/caches.
export DT_RUNTIME_ROOT="${DT_RUNTIME_ROOT:-/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922}"
export DT_ROOT="${DT_ROOT:-$DT_RUNTIME_ROOT/repo}"
export DT_METAX_BASE="${DT_METAX_BASE:-/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912}"
export VENV_PYTHON="${VENV_PYTHON:-$DT_METAX_BASE/env/bin/python}"
export MODEL_PATH="${MODEL_PATH:-/mnt/si0021787ci2/default/models/Qwen3.5-9B}"
export DT_ENVIRONMENT_JSON="${DT_ENVIRONMENT_JSON:-$DT_RUNTIME_ROOT/environment.json}"
export DT_OFFICIAL_ROOT="${DT_OFFICIAL_ROOT:-$DT_METAX_BASE/flashtrace_paper}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$DT_METAX_BASE/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$DT_METAX_BASE/cache/inductor}"
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export MACA_TORCH_COMPILE_CONF="${MACA_TORCH_COMPILE_CONF:-maca.disable_maca_triton_heuristics:1}"
export FLA_BOUNDED_NORM_TUNING=1
export VERL_ROOT="${VERL_ROOT:-$DT_RUNTIME_ROOT/third_party/verl-agent2-732f37acd7684b8c24d14ba3ededfe9fab1ed472}"
export APPWORLD_ROOT="${APPWORLD_ROOT:-$DT_RUNTIME_ROOT/third_party/appworld-42b5bcf3cd334fee33f0c37c02070a9f5807add5}"
export APPWORLD_BIN="${APPWORLD_BIN:-$(dirname "$VENV_PYTHON")/appworld}"
export WEBSHOP_ROOT="${WEBSHOP_ROOT:-$VERL_ROOT/agent_system/environments/env_package/webshop/webshop}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"
# Ray adds a session directory and Unix socket filename; the mounted runtime
# prefix exceeds Linux's 107-byte socket-path limit. Keep only IPC in /tmp.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/dt-rl-mx-20260922}"
# Installed MetaX FA2 backward was checked against FP32 math SDPA at the
# checkpoint's GQA/head layout. The local SDPA build lacks memory-efficient
# attention; reuse the verified installed FA2 kernel for the actor as well.
export VERL_ATTN_IMPLEMENTATION="${VERL_ATTN_IMPLEMENTATION:-flash_attention_2}"
# Reuse upstream batching and Ray resource settings. The actor minibatch is
# independently fixed at 4 by run_verl_agent.sh.
export ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-4}"
export DT_RAY_NUM_CPUS="${DT_RAY_NUM_CPUS:-8}"
# Set CUDA_VISIBLE_DEVICES and MACA_VISIBLE_DEVICES after checking mx-smi.
# Pinned sources and Python installs exist; task assets/services still require
# their separate verification receipts. This file never installs or starts them.
