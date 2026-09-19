#!/usr/bin/env bash
set -euo pipefail

DT_ROOT="${DT_ROOT:-$PWD}"
VERL_ROOT="${VERL_ROOT:-$DT_ROOT/third_party/verl-agent2}"
VENV_PYTHON="${VENV_PYTHON:-$DT_ROOT/env/bin/python}"

[[ -x "$VENV_PYTHON" ]] || { echo "missing VENV_PYTHON=$VENV_PYTHON" >&2; exit 2; }
[[ -d "$VERL_ROOT" ]] || { echo "clone verl-agent2 at $VERL_ROOT first" >&2; exit 2; }

# Keep the existing CUDA/PyTorch environment intact. The upstream source is
# installed editable, while task wheels are installed without dependency
# resolution so pip cannot downgrade torch, transformers, or FlashAttention.
"$VENV_PYTHON" -m pip install -e "$VERL_ROOT" --no-deps
"$VENV_PYTHON" -m pip install -r "$DT_ROOT/experiments/rl/requirements-upstream.txt" --no-deps
"$VENV_PYTHON" "$DT_ROOT/experiments/rl/patch_verl_agent2.py" "$VERL_ROOT"
"$VENV_PYTHON" - <<'PY'
import ray, torch, verl
from gym_sokoban.envs.sokoban_env import SokobanEnv
print({"torch": torch.__version__, "ray": ray.__version__, "verl": verl.__version__, "gym_sokoban": SokobanEnv.__name__})
PY
