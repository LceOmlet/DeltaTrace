#!/bin/bash
# Two complete bounded iterations through the original VERL trainer.
# Explicit pilot budget, not a paper-scale experiment or a replacement scheduler.
set -euo pipefail
R=/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922
A=$R/receipts/upstream-alignment-20260929
: "${DT_ROOT:?Set the tested runtime release}"
: "${CUDA_VISIBLE_DEVICES:?Select two currently free GPUs}"
: "${ENV_NAME:?Set official task name}"
: "${METHOD:?Set dt or grpo}"
export DT_ENVIRONMENT_JSON="$DT_ROOT/environment.json"
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export VERL_ROOT=$R/candidates/official-verl-20bd331-distributed-dt
unset MACA_VISIBLE_DEVICES RAY_TMPDIR RAY_ADDRESS
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TRAIN_SIZE=4 GROUP_SIZE=4 VAL_SIZE=1 VAL_DATA_SIZE=1
export MINI_BATCH_SIZE=64 ACTOR_MICRO_BATCH_SIZE=4 ACTOR_OFFLOAD_POLICY=True
export MAX_STEPS=15 MAX_RESPONSE=512 MAX_TOTAL_TOKENS=32768
export VAL_BEFORE_TRAIN=False ENABLE_THINKING=False TOTAL_EPOCHS=2
export SAVE_FREQ=1 TEST_FREQ=-1 RESUME_MODE=disable
export DT_RAY_NUM_CPUS=16
job="$A/two-gpu-pilots/$METHOD-$ENV_NAME"
mkdir -p "$job"
export DATA_ROOT="$job/data" CHECKPOINT_DIR="$job/checkpoints" ROLLOUT_DATA_DIR="$job/rollouts"
service_pids=()
cleanup() {
  for pid in "${service_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${service_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
if [[ "$ENV_NAME" == AppWorld ]]; then
  : "${APPWORLD_SERVER_OFFSET:?Select disjoint entries of the existing official port list}"
  export APPWORLD_PORT_FILE="$job/appworld_ports.ports"
  "$VENV_PYTHON" - "$APPWORLD_ROOT/appworld_ports.ports" "$APPWORLD_PORT_FILE" "$APPWORLD_SERVER_OFFSET" <<'PY'
from pathlib import Path
import socket, sys
ports = Path(sys.argv[1]).read_text().splitlines()[int(sys.argv[3]):int(sys.argv[3])+17]
assert len(ports) == 17
for port in ports:
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', int(port))) != 0, f'Port {port} already owned'
Path(sys.argv[2]).write_text('\n'.join(ports)+'\n')
PY
  cd "$APPWORLD_ROOT"
  while read -r port; do
    CUDA_VISIBLE_DEVICES='' MACA_VISIBLE_DEVICES='' "$APPWORLD_BIN" serve environment --port "$port" > "$job/service-$port.log" 2>&1 &
    service_pids+=("$!")
  done < "$APPWORLD_PORT_FILE"
  printf '%s\n' "${service_pids[@]}" > "$job/service-pids.txt"
  "$VENV_PYTHON" - "$APPWORLD_PORT_FILE" <<'PY'
from pathlib import Path
import socket, sys, time
pending = set(map(int, Path(sys.argv[1]).read_text().splitlines()))
deadline = time.monotonic()+60
while pending and time.monotonic()<deadline:
    for port in list(pending):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0: pending.remove(port)
    if pending: time.sleep(1)
if pending: raise RuntimeError(f'Official services did not start: {pending}')
PY
fi
set +e
bash "$DT_ROOT/experiments/rl/run_verl_agent.sh" trainer.n_gpus_per_node=2
rc=$?
printf '%s\n' "$rc" > "$job/exit-code"
exit "$rc"
