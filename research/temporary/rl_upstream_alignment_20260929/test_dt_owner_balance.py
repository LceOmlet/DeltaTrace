"""CPU-only candidate test; does not modify frozen entries or running workers."""
import subprocess
from stage_environment_entry import remote, ROOT, ENTRY, AUDIT, REPO, SSH, SCP

dest=ROOT+'/receipts/owner-b8-dispatch-20260930/dt-owner-balance-candidate'
remote(f'mkdir -p {dest}\n')
for name in ('dt_training_batch.py','test_distributed_credit.py'):
    subprocess.run(SCP+[str(REPO/'experiments/rl'/name),f'{SSH[-1]}:{dest}/{name}'],check=True)
remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
export CUDA_VISIBLE_DEVICES=''
export MACA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=@DEST@:@ROOT@/runs/official-trajectory-20260930-v8/sql-entry:@ROOT@/candidates/official-verl-20bd331-sql-formal-20260930-v8:$PYTHONPATH
cd @DEST@
"$VENV_PYTHON" -m pytest -q test_distributed_credit.py --junitxml=cpu-tests.xml
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY).replace('@DEST@',dest))
local=AUDIT/'dt-owner-balance-20261001'
local.mkdir(exist_ok=True)
subprocess.run(SCP+[f'{SSH[-1]}:{dest}/cpu-tests.xml',str(local/'cpu-tests.xml')],check=True)
for name in ('summary.json','completed-partition-comparison.json'):
    subprocess.run(SCP+[f'{SSH[-1]}:{ROOT}/receipts/owner-b8-dispatch-20260930/dt-wait-profile/{name}',str(local/name)],check=True)
