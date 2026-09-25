"""Launch the existing owner capacity verifier against the linear-readout candidate."""
import json
from pathlib import Path
import socket
import subprocess
import time

root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
rec=root/'receipts/rollout-major-cost'
candidate=root/'candidates/linear-return-20260925'
name='linear-return-capacity'
assert not (rec/(name+'-job.json')).exists()
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 29755))
config=json.loads((candidate/'environment.json').read_text())
config['qwen35']['dt_offload_replay_mixer']=False
(candidate/'environment.json').write_text(json.dumps(config,indent=2)+'\n')
script=rec/(name+'.sh')
script.write_text(f'''#!/bin/bash
export DT_ROOT={candidate}
export DT_ENVIRONMENT_JSON="$DT_ROOT/environment.json"
source "$DT_ROOT/experiments/rl/environments/metax.env.sh"
export CUDA_VISIBLE_DEVICES=7 MACA_VISIBLE_DEVICES=7 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29755
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTHONPATH="$DT_ROOT/experiments/rl:$DT_ROOT/clean/qwen35:$DT_ROOT:$VERL_ROOT:${{PYTHONPATH:-}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 FLASH_ATTENTION_DETERMINISTIC=1 PYTHONHASHSEED=0
export DT_TASK=Sokoban DT_MAX_STEPS=15 DT_MAX_LENGTH=32768
"$VENV_PYTHON" -u "$DT_ROOT/experiments/rl/verify_dt_context_capacity.py" --backend vllm --rollout-max-num-seqs 32 --no-rollout-enforce-eager --no-rollout-enable-prefix-caching --actor-microbatch 4 --activation-offload --parameter-offload-policy --response-tokens 1024 --dt-repeats 2 --verify-task-return-readouts --output "{rec}/{name}.json" --artifacts "{rec}/{name}.pt"
rc=$?
printf '%s\\n' "$rc" > "{rec}/{name}.exitcode"
exit "$rc"
''')
subprocess.run(['bash','-n',str(script)],check=True)
with (rec/(name+'.log')).open('w') as log:
    proc=subprocess.Popen(['bash',str(script)],stdout=log,stderr=subprocess.STDOUT,
                          stdin=subprocess.DEVNULL,start_new_session=True)
job=dict(pid=proc.pid,started=time.time(),gpu=7,script=str(script),
         scope='New complete-return targets, real owner DT over saved three-task fixtures, exact32768 B4 DT, two original PPO updates and native LoRA sync. Not task performance.')
(rec/(name+'-job.json')).write_text(json.dumps(job,indent=2)+'\n')
print(json.dumps(job))
