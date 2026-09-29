#!/usr/bin/env bash
# Environment comparison only; no LOOP trainer or model process is launched.
# Existing torch, Transformers, vLLM, model assets and caches are reused.
source "${DT_ROOT:?Set the existing published DT_ROOT}/experiments/rl/environments/metax.env.sh"
export LOOP_ROOT=$DT_RUNTIME_ROOT/third_party/ml-loop-f14107a976e5793990329d3193df4742076c5a1d
export LOOP_EXTRAS=$DT_RUNTIME_ROOT/environments/loop-extras-f14107a
export PYTHONPATH=$LOOP_EXTRAS:$DT_ROOT/experiments/rl:$DT_ROOT:$LOOP_ROOT${PYTHONPATH:+:$PYTHONPATH}
export PATH=$(dirname "$VENV_PYTHON"):$PATH
unset MACA_VISIBLE_DEVICES
