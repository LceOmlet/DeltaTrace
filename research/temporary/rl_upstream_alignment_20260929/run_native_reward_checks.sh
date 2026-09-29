#!/bin/bash
set -euo pipefail
R=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922
A=$R/receipts/upstream-alignment-20260929
export DT_ROOT=$R/releases/64e5876
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export VERL_ROOT=$R/candidates/official-verl-20bd331-native-reward
export DT_PRISTINE_VERL=$A/official
export PYTHONPATH=$A/dt-candidate/experiments/rl:$DT_ROOT/experiments/rl:$DT_ROOT:$VERL_ROOT
export CUDA_VISIBLE_DEVICES='' MACA_VISIBLE_DEVICES=''
"$VENV_PYTHON" -u "$A/dt-candidate/experiments/rl/patch_verl_agent2.py" "$VERL_ROOT"
"$VENV_PYTHON" -u "$A/dt-candidate/experiments/rl/patch_verl_agent2.py" "$VERL_ROOT"
"$VENV_PYTHON" -m pytest -q \
 "$A/dt-candidate/experiments/rl/test_native_reward_credit.py" \
 "$DT_ROOT/experiments/rl/test_counterfactual.py" \
 "$A/dt-candidate/experiments/rl/test_reward_readout.py" \
 "$A/dt-candidate/experiments/rl/test_rollout_credit.py" \
 "$A/dt-candidate/experiments/rl/test_verl_counterfactual.py" \
 --junitxml="$A/native-reward-tests.xml"
