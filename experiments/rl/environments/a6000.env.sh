#!/usr/bin/env bash
# Source on the A6000 host. Configuration only: no installs, builds, or jobs.
export DT_RUNTIME_ROOT="${DT_RUNTIME_ROOT:-/data/liangchen/deltatrace_resume_20260917}"
export DT_ROOT="${DT_ROOT:-$DT_RUNTIME_ROOT/repo/deltatrace}"
export VENV_PYTHON="${VENV_PYTHON:-$DT_RUNTIME_ROOT/env/bin/python}"
export MODEL_PATH="${MODEL_PATH:-/data/liangchen/models/Qwen3.5-9B}"
export VERL_ROOT="${VERL_ROOT:-$DT_RUNTIME_ROOT/third_party/verl-agent2}"
export APPWORLD_ROOT="${APPWORLD_ROOT:-$DT_RUNTIME_ROOT/third_party/appworld}"
export WEBSHOP_ROOT="${WEBSHOP_ROOT:-$DT_RUNTIME_ROOT/third_party/webshop}"
export DT_ENVIRONMENT_JSON="${DT_ENVIRONMENT_JSON:-$DT_RUNTIME_ROOT/environment.json}"
export DT_OFFICIAL_ROOT="${DT_OFFICIAL_ROOT:-$DT_RUNTIME_ROOT/flashtrace_paper}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$DT_RUNTIME_ROOT/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$DT_RUNTIME_ROOT/cache/inductor}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"
export FLA_BOUNDED_NORM_TUNING=1
export FLASHTRACE_HYBRID_ATTENTION_CPU_CACHE=1
export PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
# Choose CUDA_VISIBLE_DEVICES after checking occupancy; do not fix a GPU here.
