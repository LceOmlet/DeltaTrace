"""Launch the prepared native queue after the bounded placement benchmark exits."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT/'receipts/six-run-fresh-native-dp-owner-20261018-v1'
DP_AUDIT = ROOT/'receipts/textcraft-native-dp-owner-20261018-v1'


def main():
    assert not (AUDIT/'controller-job.json').exists(), 'Inspect existing controller; no duplicate launch'
    plan = json.loads((AUDIT/'six-combination-manifest.json').read_bytes())
    assert plan['status'] == 'prepared_not_started' and plan['all_six_start_from_base']
    accepted = json.loads(Path(plan['native_placement_acceptance']).read_bytes())
    assert accepted['status'] == 'accepted_native_DP_episode_placement'
    for path, expected in accepted['candidate_source_SHA256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    benchmark = json.loads((DP_AUDIT/'native-benchmark/job.json').read_bytes())
    try:
        p = psutil.Process(benchmark['pid'])
        assert p.create_time() != benchmark['birth'] or p.status() == psutil.STATUS_ZOMBIE, 'Benchmark still active'
    except psutil.NoSuchProcess:
        pass
    physical = subprocess.check_output(['mx-smi'], text=True)
    table = physical.split('| Process:')[-1]
    for device in range(8):
        assert not re.search(r'\|\s*'+str(device)+r'\s+\d+\s+', table), 'Physical GPU still occupied'
    (AUDIT/'physical-before-fresh-launch.txt').write_text(physical)
    first = json.loads(Path(plan['segments'][0]['prepared']).read_bytes())['jobs'][0]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    argv = [first['effective_environment']['VENV_PYTHON'], str(AUDIT/'source/run_fresh_campaign.py')]
    with (AUDIT/'controller.stdout').open('xb') as out, (AUDIT/'controller.stderr').open('xb') as err:
        child = subprocess.Popen(argv, cwd=AUDIT/'source', env=env, stdin=subprocess.DEVNULL,
            stdout=out, stderr=err, start_new_session=True)
    job = dict(pid=child.pid, birth=psutil.Process(child.pid).create_time(), argv=argv,
        created=time.time(), source_commit=os.environ['RL_CAMPAIGN_COMMIT'],
        manifest=str(AUDIT/'six-combination-manifest.json'),
        status='submitted_fresh_six_run_native_queue',
        source_bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (AUDIT/'source').glob('*.py')})
    (AUDIT/'controller-job.json').write_text(json.dumps(job, indent=2)+'\n')
    plan.update(status='submitted_fresh_six_run_native_queue', controller=job)
    (AUDIT/'six-combination-manifest.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(job))


if __name__ == '__main__':
    main()
