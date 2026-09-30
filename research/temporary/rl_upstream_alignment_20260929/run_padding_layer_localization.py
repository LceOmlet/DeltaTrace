"""Launch read-only layer hooks on the two free cards, using original actor role."""
import subprocess
from stage_environment_entry import remote, ROOT, ENTRY, AUDIT, SCP, SSH

out=ROOT+'/receipts/owner-b8-dispatch-20260930/actor-response-padding/layer-localization-v2'
remote(f'set -e\ntest ! -e {out}/job.json\nmkdir -p {out}\n')
subprocess.run(SCP+[str(AUDIT/'localize_actor_padding_layers.py'),f'{SSH[-1]}:{out}/'],check=True)
remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,os,psutil,re,subprocess,time
root=Path('@ROOT@');out=Path('@OUT@');previous=out.parent/'real-model'
assert not (out/'job.json').exists(), 'Preserve existing diagnostic evidence'
source=json.loads((previous/'source.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
candidate=Path(source['candidate_owner']);base=Path(source['reference_owner'])
actor='verl/workers/actor/dp_actor.py'
assert sha(candidate/actor)==source['candidate_actor_sha256']
assert sha(base/actor)==source['reference_actor_sha256']
state=subprocess.check_output(['mx-smi'],text=True)
for device in [6,7]:assert not re.search(r'\|\s*'+str(device)+r'\s+\d+\s+\S',state),state
active=json.loads((root/'active-training.json').read_text())
job=next(j for j in active['jobs'] if j['task']=='SkyRL-SQL')
assert job['verl_root']==str(base) and job['pid']==source['formal_driver_pid']
assert psutil.Process(job['pid']).create_time()==job['observed_process_created_unix']
source.update(saved_inputs=str(previous/'same-input-readouts.pt'),
    saved_inputs_sha256=sha(previous/'same-input-readouts.pt'),
    prior_diagnostic_source_sha256=sha(previous/'source.json'),
    script_sha256=sha(out/'localize_actor_padding_layers.py'),
    role='actor',scope='Same saved input; original standalone actor; layer hooks only, no optimizer step or vLLM loading')
(out/'source.json').write_text(json.dumps(source,indent=2)+'\n')
env=os.environ.copy();env.pop('MACA_VISIBLE_DEVICES',None)
env.update(VERL_ROOT=str(candidate),DT_ENTRY_ROOT=job['entry'],CUDA_VISIBLE_DEVICES='6,7',
    PADDING_REFERENCE_ROOT=str(base),PADDING_DIAGNOSTIC_DIR=str(out))
env['PYTHONPATH']=':'.join([str(out),job['entry'],str(candidate),env['PYTHONPATH']])
argv=[env['VENV_PYTHON'],'-u',str(out/'localize_actor_padding_layers.py')]
with (out/'diagnostic.log').open('wb') as log:
    p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,started_unix=time.time(),observed_process_created_unix=psutil.Process(p.pid).create_time(),
    argv=argv,devices=[6,7],gpus=[6,7],scope=source['scope'])
(out/'job.json').write_text(json.dumps(record,indent=2)+'\n')
with (out/'resources.log').open('wb') as log:
    observer=subprocess.Popen([env['VENV_PYTHON'],'@ENTRY@/observe_entry_resources.py','--job',str(out/'job.json'),
        '--output',str(out/'physical-resources.jsonl')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'observer.json').write_text(json.dumps(dict(pid=observer.pid,created=psutil.Process(observer.pid).create_time()),indent=2)+'\n')
print(json.dumps(dict(**record,observer_pid=observer.pid)),flush=True)
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY).replace('@OUT@',out))
