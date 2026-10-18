"""Stop the exact replaced v6 campaign through the existing process-stop owner.

Logs and native checkpoints are retained. This neither saves a new checkpoint
nor starts training; the existing launch owner is called separately.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT/'receipts/dtpo-symmetric-probability-ratio-20261018-v7'
PRIOR = ROOT/'receipts/six-run-fresh-validated-efficiency-20261018-v1'
OLD_SEGMENT = ROOT/'receipts/textcraft-eight-gpu-sequential-20261018-v5'
EXPECTED = [(2004205, 1792263400.01), (2004329, 1792263400.35)]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(tmp, path)

def main():
    plan = json.loads((AUDIT/'six-combination-manifest.json').read_bytes())
    assert plan['status'] == 'prepared_not_started' and len(plan['combinations']) == 6
    assert not (AUDIT/'controller-job.json').exists()
    check = json.loads((AUDIT/'CPU-behavior-verification.json').read_bytes())
    assert check['status'] == 'passed'
    stop_path = AUDIT/'superseded-v6-stopped.json'
    if not stop_path.exists():
        controller = json.loads((PRIOR/'controller-job.json').read_bytes())
        assert (controller['pid'], controller['birth']) == EXPECTED[0]
        state = json.loads((OLD_SEGMENT/'campaign-state.json').read_bytes())
        assert any((r.get('pid'), r.get('birth')) == EXPECTED[1] for r in state['jobs'])
        processes, seen = [], set()
        for pid, birth in EXPECTED:
            p = psutil.Process(pid)
            assert p.create_time() == birth and p.status() != psutil.STATUS_ZOMBIE
            for child in [p, *p.children(recursive=True)]:
                if child.pid not in seen:
                    seen.add(child.pid)
                    processes.append(child)
        for p in psutil.process_iter(['pid', 'create_time']):
            try:
                if os.getsid(p.pid) == EXPECTED[1][0] and p.create_time() >= EXPECTED[1][1] and p.pid not in seen:
                    seen.add(p.pid)
                    processes.append(p)
            except (psutil.Error, OSError):
                pass
        for name in ('formal-training.json', 'active-training.json', 'active-source.json'):
            (AUDIT/('before-'+name)).write_bytes((ROOT/name).read_bytes())
        (AUDIT/'superseded-v6-final-state.json').write_bytes((OLD_SEGMENT/'campaign-state.json').read_bytes())
        record = dict(status='stopping_superseded_v6', started=time.time(), roots=EXPECTED,
            reason='Human requested symmetric-probability-ratio v7 and six fresh 200-iteration runs',
            processes=[dict(pid=p.pid,birth=p.create_time(),name=p.name()) for p in processes],
            logs_and_checkpoints_preserved=True, extra_checkpoints=0, SIGSTOP_calls=0)
        write(AUDIT/'superseded-v6-stopping.json',record)
        controller_process = psutil.Process(EXPECTED[0][0])
        assert controller_process.create_time() == EXPECTED[0][1]
        controller_process.terminate()
        controller_process.wait(timeout=10)
        owner = ROOT/'receipts/textcraft-eight-gpu-sequential-20261017-v1/source/resume_at_native_checkpoint.py'
        assert sha(owner) == 'b5be347a4ba7deb2f02b7c892f375642b91d4992290d64f1d7b7c2659b211f0f'
        template = next(n.value for n in ast.walk(ast.parse(owner.read_text()))
            if isinstance(n,ast.Constant) and isinstance(n.value,str)
            and 'processes=[parent]+parent.children(recursive=True)' in n.value)
        block = template[template.index('for p in reversed(processes):'):template.index("record['other_jobs_unchanged']={}")]
        exec(compile(block,str(owner),'exec'),dict(processes=processes,record=record,
            psutil=psutil,signal=signal,time=time))
        record.update(status='stopped_by_user_version_replacement',
            stop_owner=str(owner), stop_owner_SHA256=sha(owner))
        write(stop_path, record)
    record = json.loads(stop_path.read_bytes())
    remaining = []
    for item in record['processes']:
        try:
            p = psutil.Process(item['pid'])
            if p.create_time() == item['birth'] and p.status() != psutil.STATUS_ZOMBIE:
                remaining.append(item)
        except psutil.NoSuchProcess:
            pass
    assert not remaining, 'Teardown incomplete; inspect this receipt before launching'
    record.update(remaining_non_zombie=[],finished_unix=time.time())
    write(stop_path,record)
    for task in ('textcraft','appworld'):
        folder = ROOT/'receipts'/(task+'-eight-gpu-sequential-20261018-v5')
        path = folder/'campaign-state.json'
        if path.exists():
            value=json.loads(path.read_bytes())
            value.update(status='superseded_by_symmetric_probability_ratio_v7',stop_receipt=str(stop_path))
            for row in value.get('jobs',[]):
                if (row.get('pid'),row.get('birth')) == EXPECTED[1]:
                    row.update(active=False,alive=False,status='user_stopped_superseded_v6',stop_receipt=str(stop_path))
                    write(Path(row['run'])/'source.json',row)
            write(path,value)
    prior=json.loads((PRIOR/'six-combination-manifest.json').read_bytes())
    prior.update(status='superseded_by_symmetric_probability_ratio_v7',stop_receipt=str(stop_path))
    write(PRIOR/'six-combination-manifest.json',prior)
    for name in ('formal-training.json','active-training.json','active-source.json'):
        value=json.loads((ROOT/name).read_bytes())
        for row in value.get('jobs',[]):
            if (row.get('pid'),row.get('birth')) == EXPECTED[1]:
                row.update(active=False,alive=False,status='user_stopped_superseded_v6',stop_receipt=str(stop_path))
        value['sequential_campaign_plan']=str(AUDIT/'six-combination-manifest.json')
        write(ROOT/name,value)
    print(json.dumps(dict(status=record['status'],remaining_non_zombie=[],receipt=str(stop_path))))

if __name__=='__main__':main()
