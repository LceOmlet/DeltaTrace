"""Submit the one authorized fresh native controller after accepted SSM repair."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rl_textcraft_v5_credit_integration_20261020_v1'))
from transport import run_python


def main():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    remote = r'''
import json,os,psutil,subprocess,sys,time
from pathlib import Path
A=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-v12-attribution-then-group-20261024-v1')
COMMIT='__COMMIT__'
sys.path.insert(0,str(A/'source'))
from run_campaign import ready
from prepare_campaign import STATUS,sha
row=ready()
assert not (A/'controller-job.json').exists() and not (A/'campaign-state.json').exists(), 'No duplicate controller'
assert not Path(row['run']).exists() and not Path(row['checkpoints']).exists(), 'Fresh base0 run only'
assert not (A/'initial-worker-observer-job.json').exists()
row['source_bindings'].update(row['required_installed_source_bindings'])
row.update(local_patch_commit=COMMIT,adapter_source_commit=COMMIT,execution_source_commit=COMMIT,method_source_commit=COMMIT,
    execution_source_commit_scope='v12 raw attribution and group advantage owner extension; original DT kernels and native PPO/scheduling remain identified by file SHA and parent commits.',
    source_assembly_uncommitted=False,
    orchestration_source_bindings={str(p):sha(p) for p in (A/'source').glob('*.py')})
(A/'dt-prepared.json').write_text(json.dumps(row,indent=2)+'\n')
(A/'campaign-prepared.json').write_text(json.dumps(dict(status=STATUS,order=['dt'],jobs=[row]),indent=2)+'\n')
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',RL_CAMPAIGN_COMMIT=COMMIT)
stdout,stderr=A/'controller.stdout',A/'controller.stderr'
with stdout.open('xb') as out,stderr.open('xb') as err:
    child=subprocess.Popen([row['effective_environment']['VENV_PYTHON'],str(A/'source/run_campaign.py')],
        cwd=A/'source',env=environment,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
job=dict(status='submitted_native_fresh_v12_attribution_then_group_eight_gpu_controller',
    pid=child.pid,birth=psutil.Process(child.pid).create_time(),submitted_unix=time.time(),
    source_commit=COMMIT,controller_source=str(A/'source/run_campaign.py'),
    controller_SHA256=sha(A/'source/run_campaign.py'),stdout=str(stdout),stderr=str(stderr),
    run=row['run'],budget=row['budget'],completed_iterations=0,fresh_base_model=True,
    initialization_is_not_training_progress=True)
(A/'controller-job.json').write_text(json.dumps(job,indent=2)+'\n')
with (A/'initial-worker-observer.stdout').open('xb') as out,(A/'initial-worker-observer.stderr').open('xb') as err:
    observer=subprocess.Popen([row['effective_environment']['VENV_PYTHON'],str(A/'source/observe_initial_workers.py')],
        cwd=A/'source',env=environment,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
watch=dict(status='submitted_original_once_per_phase_read_only_observer',pid=observer.pid,
    birth=psutil.Process(observer.pid).create_time(),submitted_unix=time.time(),
    source=str(A/'source/observe_initial_workers.py'),source_SHA256=sha(A/'source/observe_initial_workers.py'),
    model_calls=0,DT_calls=0,optimizer_updates=0,recorder_installations=0)
(A/'initial-worker-observer-job.json').write_text(json.dumps(watch,indent=2)+'\n')
print(json.dumps(dict(controller=job,observer=watch)))
'''.replace('__COMMIT__', commit)
    record = json.loads(run_python(remote, timeout=60).strip().splitlines()[-1])
    (HERE / 'submitted-native-controller-v1.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
