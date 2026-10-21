"""Submit one prepared native controller and its original read-only observer."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rl_textcraft_v5_credit_integration_20261020_v1'))
from transport import run_python

commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
code = r'''
import hashlib,json,os,psutil,subprocess,sys,time
from pathlib import Path
A=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-v10-eight-gpu-generation-aligned-20261022-v1')
COMMIT='__COMMIT__'
sys.path.insert(0,str(A/'source'))
from run_campaign import ready
ready()
assert not (A/'controller-job.json').exists() and not (A/'campaign-state.json').exists(), 'Inspect existing controller; no duplicate submission'
row=json.loads((A/'dt-prepared.json').read_text())
assert not Path(row['run']).exists() and not Path(row['checkpoints']).exists()
row.update(local_patch_commit=COMMIT,adapter_source_commit=COMMIT,execution_source_commit=COMMIT,
    execution_source_commit_scope='Source assembly and generation control; method and parent trainer identified separately by file SHA',
    training_logic_parent_commit='4545c55807e7d002287717b6a4254f9b083dd5fc',
    eight_GPU_owner_parent_execution_commit='9bf225356b251ace356c1119e9d27c7c08550867',
    generation_control_source_commit=COMMIT)
(A/'dt-prepared.json').write_text(json.dumps(row,indent=2)+'\n')
(A/'campaign-prepared.json').write_text(json.dumps(dict(status='v10_eight_GPU_formal_configuration_resolved',order=['dt'],jobs=[row]),indent=2)+'\n')
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',RL_CAMPAIGN_COMMIT=COMMIT)
stdout,stderr=A/'controller.stdout',A/'controller.stderr'
with stdout.open('xb') as out,stderr.open('xb') as err:
    child=subprocess.Popen([row['effective_environment']['VENV_PYTHON'],str(A/'source/run_campaign.py')],
        cwd=A/'source',env=environment,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
job=dict(status='submitted_native_v10_eight_gpu_controller',pid=child.pid,birth=psutil.Process(child.pid).create_time(),
    submitted_unix=time.time(),source_commit=COMMIT,stdout=str(stdout),stderr=str(stderr),
    controller_source=str(A/'source/run_campaign.py'),controller_SHA256=hashlib.sha256((A/'source/run_campaign.py').read_bytes()).hexdigest(),
    run=row['run'],budget=row['budget'],completed_iterations=0,initialization_is_not_training_progress=True)
(A/'controller-job.json').write_text(json.dumps(job,indent=2)+'\n')
assert not (A/'initial-worker-observer-job.json').exists()
with (A/'initial-worker-observer.stdout').open('xb') as out,(A/'initial-worker-observer.stderr').open('xb') as err:
    observer=subprocess.Popen([row['effective_environment']['VENV_PYTHON'],str(A/'source/observe_initial_workers.py')],
        cwd=A/'source',env=environment,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
observed=dict(status='submitted_original_once_per_phase_read_only_observer',pid=observer.pid,
    birth=psutil.Process(observer.pid).create_time(),source=str(A/'source/observe_initial_workers.py'),
    source_SHA256=hashlib.sha256((A/'source/observe_initial_workers.py').read_bytes()).hexdigest(),submitted_unix=time.time(),
    model_calls=0,DT_calls=0,optimizer_updates=0,recorder_installations=0)
(A/'initial-worker-observer-job.json').write_text(json.dumps(observed,indent=2)+'\n')
print(json.dumps(dict(controller=job,observer=observed)))
'''.replace('__COMMIT__', commit)
record = json.loads(run_python(code, timeout=60).splitlines()[-1])
(HERE / 'submitted-native-controller-v2.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record, indent=2))
