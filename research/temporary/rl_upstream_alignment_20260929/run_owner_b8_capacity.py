import subprocess
from stage_environment_entry import remote,ROOT,ENTRY,AUDIT,REPO,SCP,SSH
out=ROOT+'/receipts/owner-b8-dispatch-20260930'
remote(f'mkdir -p {out}\n')
for p in [AUDIT/'verify_owner_b8_capacity.py']+[REPO/'experiments/rl'/n for n in [
    'owner_runtime_options.py','launch_textcraft_native.py','launch_sql_native.py','launch_appworld_native.py',
    'test_owner_entry_launch.py']]:
    subprocess.run(SCP+[str(p),f'{SSH[-1]}:{out}/{p.name}'],check=True)
remote(fr'''set -e
source {ENTRY}/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import os,json,psutil,subprocess,time,shutil
root=Path('{ROOT}');out=Path('{out}')
for name in ['final-fused-temperature1.json','materialized-temperature09.json']:
 proof=json.loads((root/'receipts/owner-entropy-20260930'/name).read_text())
 assert all(a['passed'] for c in proof['cases'] for a in c['assertions'])
for p in psutil.process_iter(['cmdline']):
 assert not any(a.endswith(('/verify_owner_b8_capacity.py','/verify_owner_b4_capacity.py','/probe_pinned_fused_head_dtype.py')) for a in (p.info['cmdline'] or [])),('existing diagnostic',p.pid)
entry=root/'runs/official-trajectory-20260930-v7/textcraft-entry'
for name in ['owner_environment_configs.json','textcraft_qwen_template.json']:
 shutil.copy2(entry/name,out/name)
verl=root/'candidates/official-verl-20bd331-fused-head-b4-20260930'
env=os.environ.copy();env.pop('MACA_VISIBLE_DEVICES',None)
env.update(VERL_ROOT=str(verl),DT_ENTRY_ROOT=str(entry),CUDA_VISIBLE_DEVICES='6,7',
 AGENTGYM_RL_ROOT=str(root/'runs/official-trajectory-20260930-v4/textcraft-owner'))
env['PYTHONPATH']=':'.join([str(out),str(entry),str(verl),env['PYTHONPATH']])
argv=[env['VENV_PYTHON'],'-u',str(out/'verify_owner_b8_capacity.py')]
with (out/'capacity.log').open('wb') as f:
 p=subprocess.Popen(argv,env=env,cwd=str(out),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
job=dict(pid=p.pid,started_unix=time.time(),argv=argv,gpus=[6,7],scope='One B8 batch, native DP dispatch, B4 per GPU')
(out/'job.json').write_text(json.dumps(job,indent=2))
with (out/'resources.log').open('wb') as f:
 observer=subprocess.Popen([env['VENV_PYTHON'],'{ENTRY}/observe_entry_resources.py','--job',str(out/'job.json'),
  '--output',str(out/'physical-resources.jsonl')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(**job,observer=observer.pid)))
PY
''')
