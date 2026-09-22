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
# Set CUDA_VISIBLE_DEVICES and MACA_VISIBLE_DEVICES after checking mx-smi.
# VERL/task runtime paths must be verified separately; they are not assumed ready.
