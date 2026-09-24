#!/usr/bin/env bash
set -euo pipefail

DT_ROOT="${DT_ROOT:-$PWD}"
VERL_ROOT="${VERL_ROOT:-$DT_ROOT/third_party/verl-agent}"
APPWORLD_ROOT="${APPWORLD_ROOT:-$DT_ROOT/third_party/appworld}"
VENV_PYTHON="${VENV_PYTHON:-$DT_ROOT/env/bin/python}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"
export PYTHONPATH="$VERL_ROOT:${VERL_ROOT}/agent_system/environments/env_package/webshop/webshop:${APPWORLD_ROOT}:${PYTHONPATH:-}"

"$VENV_PYTHON" "$DT_ROOT/experiments/rl/patch_verl_agent2.py" "$VERL_ROOT"
"$VENV_PYTHON" - <<'PY'
from pathlib import Path

from web_agent_site.envs import WebAgentTextEnv
from agent_system.environments.env_package.sokoban.sokoban import SokobanEnv
from appworld import AppWorld, load_task_ids

web = WebAgentTextEnv(observation_mode="text", num_products=1000, seed=7, disable_env_checker=True)
obs, reward, done, _ = web.step("search[shoes]")
assert not done and isinstance(obs, str)
web.close()

sokoban = SokobanEnv("rgb_array", dim_room=(6, 6), num_boxes=1, max_steps=5, search_depth=30)
obs, _ = sokoban.reset(seed=7)
assert getattr(obs, "shape", None) == (96, 96, 3)
sokoban.step(1)
sokoban.close()

world = AppWorld(task_id=load_task_ids("train")[0], experiment_name="delta_trace_smoke")
assert world.task.instruction
world.execute("help()")
world.close()
print("WebShop, Sokoban, and AppWorld smoke passed")
PY
