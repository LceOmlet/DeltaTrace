"""User-authorized fresh restart, preserving all pilot data and native state."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/six-run-fresh-after-efficiency-20261017-v1'
PILOT = ROOT / 'receipts/textcraft-eight-gpu-sequential-20261017-v1'
SIX = ROOT / 'receipts/two-task-eight-gpu-sequential-20261017-v2'
IDENTITIES = [(3173187, 1792235540.72), (3335370, 1792237019.8), (3173204, 1792235541.21)]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(tmp, path)


def main():
    plan = json.loads((AUDIT / 'six-combination-manifest.json').read_bytes())
    assert plan['status'] == 'prepared_not_started' and len(plan['combinations']) == 6
    assert not (AUDIT / 'controller-job.json').exists(), 'Inspect actual existing launch; no duplicate'
    prepared = json.loads((AUDIT / 'native-resource-owner-prepared.json').read_bytes())
    assert sha(prepared['candidate']) == prepared['candidate_SHA256']
    deployed_path = AUDIT / 'native-resource-owner-deployed.json'
    if deployed_path.exists():
        existing = json.loads(deployed_path.read_bytes())
        assert existing['status'] == 'deployed_after_pilot_stopped'
        assert sha(prepared['owner_path']) == prepared['candidate_SHA256']
    else:
        assert sha(prepared['owner_path']) == prepared['original_SHA256']
    stop_path = AUDIT / 'user-replaced-tuning-jobs-stop.json'
    if not stop_path.exists():
        processes, seen = [], set()
        for pid, birth in IDENTITIES:
            p = psutil.Process(pid)
            assert p.create_time() == birth and p.status() != psutil.STATUS_ZOMBIE
            for q in [p, *p.children(recursive=True)]:
                if q.pid not in seen:
                    seen.add(q.pid)
                    processes.append(q)
        for p in psutil.process_iter(['pid', 'create_time']):
            try:
                if os.getsid(p.pid) == 3173204 and p.create_time() >= 1792235541.21 and p.pid not in seen:
                    seen.add(p.pid)
                    processes.append(p)
            except (psutil.Error, OSError):
                pass
        for name in ('formal-training.json', 'active-training.json', 'active-source.json'):
            (AUDIT / ('before-'+name)).write_bytes((ROOT / name).read_bytes())
        for name in ('campaign-state.json','dt-latest-source-snapshot.json'):
            (AUDIT / ('tuning-final-'+name)).write_bytes((PILOT / name).read_bytes())
        record = dict(status='stopping_user_replaced_tuning', started=time.time(),
            reason='Latest human instruction: after accepted efficiency fixes all six comparisons restart from base and zero',
            roots=[dict(pid=p,birth=b) for p,b in IDENTITIES],
            processes=[dict(pid=p.pid,birth=p.create_time(),name=p.name()) for p in processes],
            original_logs_and_checkpoints_preserved=True, new_checkpoints=0, SIGSTOP_calls=0)
        write(AUDIT / 'user-replaced-tuning-jobs-stopping.json', record)
        # Stop orchestration first so a completing pilot cannot launch an old queued job.
        for pid, birth in IDENTITIES[:2]:
            p = psutil.Process(pid)
            assert p.create_time() == birth
            p.terminate()
            p.wait(timeout=10)
        owner = PILOT / 'source/resume_at_native_checkpoint.py'
        assert sha(owner) == 'b5be347a4ba7deb2f02b7c892f375642b91d4992290d64f1d7b7c2659b211f0f'
        template = next(n.value for n in ast.walk(ast.parse(owner.read_text()))
            if isinstance(n,ast.Constant) and isinstance(n.value,str)
            and 'processes=[parent]+parent.children(recursive=True)' in n.value)
        block = template[template.index('for p in reversed(processes):'):template.index("record['other_jobs_unchanged']={}")]
        exec(compile(block,str(owner),'exec'), dict(processes=processes,record=record,psutil=psutil,signal=signal,time=time))
        record.update(status='stopped_by_user_replacement',stop_owner=str(owner),stop_owner_SHA256=sha(owner))
        write(stop_path, record)
    record = json.loads(stop_path.read_bytes())
    remaining = []
    for item in record['processes']:
        try:
            p=psutil.Process(item['pid'])
            if p.create_time()==item['birth'] and p.status()!=psutil.STATUS_ZOMBIE:
                remaining.append(item)
        except psutil.NoSuchProcess:
            pass
    assert not remaining, 'Native teardown still in progress: preserve receipt and check, do not launch'
    record.update(remaining_non_zombie=[], finished_unix=time.time())
    write(stop_path,record)
    physical = subprocess.check_output(['mx-smi'],text=True)
    table = physical.split('| Process:')[-1]
    for device in range(8):
        assert not re.search(r'\|\s*'+str(device)+r'\s+\d+\s+',table),'GPU still occupied'
    (AUDIT / 'physical-before-fresh-launch.txt').write_text(physical)
    pilot_state = json.loads((PILOT / 'campaign-state.json').read_bytes())
    pilot_state.update(status='user_stopped_tuning_history_excluded_from_fresh_comparison', stopped=time.time(), stop_receipt=str(stop_path))
    for row in pilot_state['jobs']:
        row.update(active=False,alive=False,status='user_stopped_tuning_history')
        write(Path(row['run'])/'source.json',row)
    write(PILOT / 'campaign-state.json',pilot_state)
    for name in ('formal-training.json','active-training.json','active-source.json'):
        value=json.loads((ROOT/name).read_bytes())
        for row in value['jobs']:
            if (row.get('pid'),row.get('birth')) == IDENTITIES[2]:
                row.update(status='user_stopped_tuning_history',active=False,alive=False,stop_receipt=str(stop_path))
        value['sequential_campaign_plan']=str(AUDIT/'six-combination-manifest.json')
        write(ROOT/name,value)
    owner = Path(prepared['owner_path'])
    if not deployed_path.exists():
        owner.write_bytes(Path(prepared['candidate']).read_bytes())
        assert sha(owner)==prepared['candidate_SHA256']
        deployed=dict(prepared,status='deployed_after_pilot_stopped',deployed=time.time(),
            pilot_stop_receipt=str(stop_path),source_commit=os.environ['RL_CAMPAIGN_COMMIT'],
            original_owner_backup=str(AUDIT/'gpu_model_runner.before.py'),
            allocator_configuration='expandable_segments:False,pinned_use_cuda_host_register:True',
            numerical_model_sampler_critic_actor_changes=0)
        write(deployed_path,deployed)
    text_prepared = json.loads(Path(plan['segments'][0]['prepared']).read_bytes())
    python=text_prepared['jobs'][0]['effective_environment']['VENV_PYTHON']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    argv=[python,str(AUDIT/'source/run_fresh_queue.py')]
    with (AUDIT/'controller.stdout').open('xb') as out,(AUDIT/'controller.stderr').open('xb') as err:
        child=subprocess.Popen(argv,cwd=AUDIT/'source',env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    job=dict(pid=child.pid,birth=psutil.Process(child.pid).create_time(),argv=argv,created=time.time(),
        status='submitted_fresh_six_run_native_queue',source_commit=os.environ['RL_CAMPAIGN_COMMIT'],
        manifest=str(AUDIT/'six-combination-manifest.json'),
        source_bindings={str(p):sha(p) for p in (AUDIT/'source').glob('*.py')})
    write(AUDIT/'controller-job.json',job)
    plan.update(status='submitted_fresh_six_run_native_queue',controller=job)
    write(AUDIT/'six-combination-manifest.json',plan)
    print(json.dumps(dict(status=job['status'],pid=job['pid'],birth=job['birth'],all_six_fresh=True,resource_SHA256=sha(owner))))


if __name__ == '__main__':
    main()
