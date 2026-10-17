"""Publish six prepared combinations and attach one successor process."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import psutil
from run_campaign import ROOT, write

AUDIT = ROOT / 'receipts/two-task-eight-gpu-sequential-20261017-v2'
TEXT = ROOT / 'receipts/textcraft-eight-gpu-sequential-20261017-v1'
APP = ROOT / 'receipts/appworld-eight-gpu-sequential-20261017-v1'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (AUDIT / 'successor-job.json').exists(), 'Inspect recorded successor; no duplicate process'
    text = json.loads((TEXT / 'campaign-prepared.json').read_bytes())
    app = json.loads((APP / 'campaign-prepared.json').read_bytes())
    state = json.loads((TEXT / 'campaign-state.json').read_bytes())
    assert app['status'] == 'three_native_CPU_configs_resolved_and_common_workload_equal'
    assert sha(TEXT / 'source/run_campaign.py') == '86d65a671e1c7167b025388ffcd07e1dbdbbf0291eba62c68ed085cb9313ff6f'
    assert state['status'] != 'needs_attention_no_failed_combination_skipped'
    predecessor = psutil.Process(state['controller_pid'])
    assert predecessor.create_time() == state['controller_birth'] and predecessor.is_running()
    jobs = text['jobs'] + app['jobs']
    assert [(j['task'], j['method']) for j in jobs] == [
        (task, method) for task in ('TextCraft', 'AppWorld') for method in ('dt', 'ppo', 'grpo')]
    assert all(j['devices'] == list(range(8)) and j['budget']['total_iterations'] == 200 for j in jobs)
    assert all(j['options']['trainer.save_freq'] == 2 for j in app['jobs'])
    plan = dict(version=AUDIT.name, created_unix=time.time(),
        source_commit=os.environ['RL_CAMPAIGN_COMMIT'],
        segments=[dict(task='TextCraft', state=str(TEXT / 'campaign-state.json'),
                       prepared=str(TEXT / 'campaign-prepared.json'),
                       controller_pid=state['controller_pid'], controller_birth=state['controller_birth']),
                  dict(task='AppWorld', state=str(APP / 'campaign-state.json'),
                       prepared=str(APP / 'campaign-prepared.json'))],
        combinations=[dict(task=j['task'], method=j['method'], version=j['version'],
                           run=j['run'], output=j['output'], checkpoints=j['checkpoints'],
                           budget=j['budget']) for j in jobs],
        source_bindings={str(p): sha(p) for p in (AUDIT / 'source').glob('*.py')},
        predecessor_source_preserved=True, predecessor_restart_calls=0,
        workload_owner='Pinned VERL and native task generation/environment/LOOP owners',
        completion_owner='Original native metrics, completion marker and full checkpoints; no failed combination skipped')
    write(AUDIT / 'six-combination-manifest.json', plan)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    argv = [app['jobs'][0]['effective_environment']['VENV_PYTHON'],
            str(AUDIT / 'source/run_appworld_successor.py')]
    with (AUDIT / 'successor.stdout').open('xb') as out, (AUDIT / 'successor.stderr').open('xb') as err:
        child = subprocess.Popen(argv, cwd=AUDIT / 'source', env=env,
            stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
    job = dict(pid=child.pid, birth=psutil.Process(child.pid).create_time(), argv=argv,
        status='submitted_successor_waiting_for_TextCraft', source_commit=os.environ['RL_CAMPAIGN_COMMIT'],
        manifest=str(AUDIT / 'six-combination-manifest.json'), created_unix=time.time(),
        source_bindings=plan['source_bindings'], training_processes_launched_by_activation=0)
    write(AUDIT / 'successor-job.json', job)
    for name in ('formal-training.json', 'active-training.json', 'active-source.json'):
        path = ROOT / name
        value = json.loads(path.read_bytes())
        value['sequential_campaign_plan'] = job['manifest']
        write(path, value)
    print(json.dumps(dict(pid=job['pid'], birth=job['birth'], status=job['status'],
        combinations=[(j['task'], j['method']) for j in jobs], iterations_per_combination=200)))


if __name__ == '__main__':
    main()
