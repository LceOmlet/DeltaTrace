#!/bin/bash
set -euo pipefail
R=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922
A=$R/receipts/upstream-alignment-20260929
export DT_ROOT=$R/releases/64e5876 DT_ENVIRONMENT_JSON=$R/releases/64e5876/environment.json
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export VERL_ROOT=$R/candidates/official-verl-20bd331-native-reward
export PYTHONPATH=$A/dt-candidate/experiments/rl:$DT_ROOT/experiments/rl:$DT_ROOT/clean/qwen35:$DT_ROOT:$VERL_ROOT
export CUDA_VISIBLE_DEVICES=7 MACA_VISIBLE_DEVICES=7 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29762
export DT_TASK=Sokoban DT_MAX_STEPS=15 DT_MAX_LENGTH=32768
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF//expandable_segments:True/expandable_segments:False}"
unset RAY_TMPDIR
"$VENV_PYTHON" - <<'PY'
from pyserini.search.lucene import LuceneSearcher
from jnius import autoclass
print('Existing Pyserini JVM:', autoclass('java.lang.System').getProperty('java.version'), flush=True)
PY
# Use the existing five recorded ports and the official CLI. Do not run the
# author's fleet launcher, which kills all AppWorld processes and starts 296.
ports=$(head -n 5 "$VERL_ROOT/appworld_ports.ports")
"$VENV_PYTHON" - $ports <<'PY'
import socket, sys
for value in sys.argv[1:]:
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', int(value))) != 0, f'port {value} already owned'
PY
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
cd "$APPWORLD_ROOT"
for port in $ports; do
  CUDA_VISIBLE_DEVICES='' MACA_VISIBLE_DEVICES='' "$APPWORLD_BIN" serve environment --port "$port" > "$A/appworld-service-$port.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$A/rollout-service-pids.txt"
"$VENV_PYTHON" - $ports <<'PY'
import socket, sys, time
deadline = time.monotonic()+60
pending = set(map(int, sys.argv[1:]))
while pending and time.monotonic()<deadline:
    for port in list(pending):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                pending.remove(port)
    if pending: time.sleep(1)
if pending: raise RuntimeError(f'official service startup incomplete: {pending}')
PY
cd "$VERL_ROOT"
CUDA_VISIBLE_DEVICES='' MACA_VISIBLE_DEVICES='' "$VENV_PYTHON" -u "$A/verify_native_environment_reset.py"
if (( $# == 0 )); then set -- Sokoban Webshop AppWorld; fi
extra=()
output="$A/native-fresh-rollout-parity-$1.json"
if [[ "${PROFILE_ONLY:-0}" == 1 ]]; then
  extra=(--implementations candidate)
  output="$A/native-task-length-cost-$1.json"
fi
"$VENV_PYTHON" -u "$A/dt-candidate/experiments/rl/verify_author_rollout.py" \
  --author-collector "$A/official/agent_system/multi_turn_rollout/rollout_loop.py" \
  --author-manager "$A/official/agent_system/environments/env_manager.py" \
  --tasks "$@" --rounds "${ROUNDS:-2}" "${extra[@]}" \
  --output "$output"
