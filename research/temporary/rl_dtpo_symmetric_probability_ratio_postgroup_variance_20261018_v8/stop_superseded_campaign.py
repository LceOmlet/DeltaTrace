"""Stop only the replaced v7 campaign through the already accepted stop owner."""
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/dtpo-symmetric-probability-ratio-postgroup-variance-20261018-v8'
PRIOR=ROOT/'receipts/dtpo-symmetric-probability-ratio-20261018-v7'
EXPECTED=[(2854542,1792312285.09),(2854659,1792312285.4)]
AUDIT.mkdir(parents=True,exist_ok=True)
def write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    stop_path=AUDIT/'superseded-v7-stopped.json'
    if not stop_path.exists():
        cj=json.loads((PRIOR/'controller-job.json').read_bytes())
        assert (cj['pid'],cj['birth'])==EXPECTED[0]
        formal=json.loads((ROOT/'formal-training.json').read_bytes())
        old,=[j for j in formal['jobs'] if j.get('task')=='TextCraft' and j.get('method')=='dt']
        assert (old['pid'],old['birth'])==EXPECTED[1]
        assert 'symmetric-probability-ratio-v7' in old['version']
        processes=[];seen=set()
        for pid,birth in EXPECTED:
            p=psutil.Process(pid);assert p.create_time()==birth and p.status()!=psutil.STATUS_ZOMBIE
            for child in [p,*p.children(recursive=True)]:
                if child.pid not in seen:seen.add(child.pid);processes.append(child)
        for p in psutil.process_iter(['pid','create_time']):
            try:
                if os.getsid(p.pid)==EXPECTED[1][0] and p.create_time()>=EXPECTED[1][1] and p.pid not in seen:
                    seen.add(p.pid);processes.append(p)
            except (psutil.Error,OSError):pass
        for name in ('formal-training.json','active-training.json','active-source.json'):
            (AUDIT/('before-'+name)).write_bytes((ROOT/name).read_bytes())
        record=dict(status='stopping_superseded_v7',started=time.time(),roots=EXPECTED,
            reason='Human approved v8 post-group std(g*u) and comparable global norm reporting after rejecting continued v7 monitoring.',
            processes=[dict(pid=p.pid,birth=p.create_time(),name=p.name()) for p in processes],
            logs_and_checkpoints_preserved=True,extra_checkpoints=0,SIGSTOP_calls=0)
        write(AUDIT/'superseded-v7-stopping.json',record)
        p=psutil.Process(EXPECTED[0][0]);assert p.create_time()==EXPECTED[0][1]
        p.terminate();p.wait(timeout=10)
        owner=ROOT/'receipts/textcraft-eight-gpu-sequential-20261017-v1/source/resume_at_native_checkpoint.py'
        assert sha(owner)=='b5be347a4ba7deb2f02b7c892f375642b91d4992290d64f1d7b7c2659b211f0f'
        template=next(n.value for n in ast.walk(ast.parse(owner.read_text()))
            if isinstance(n,ast.Constant) and isinstance(n.value,str)
            and 'processes=[parent]+parent.children(recursive=True)' in n.value)
        block=template[template.index('for p in reversed(processes):'):template.index("record['other_jobs_unchanged']={}")]
        exec(compile(block,str(owner),'exec'),dict(processes=processes,record=record,psutil=psutil,signal=signal,time=time))
        record.update(status='stopped_by_user_v8_version_replacement',stop_owner=str(owner),stop_owner_SHA256=sha(owner))
        write(stop_path,record)
    record=json.loads(stop_path.read_bytes())
    remaining=[]
    for item in record['processes']:
        try:
            p=psutil.Process(item['pid'])
            if p.create_time()==item['birth'] and p.status()!=psutil.STATUS_ZOMBIE:remaining.append(item)
        except psutil.NoSuchProcess:pass
    assert not remaining,'Inspect teardown; do not launch over remaining v7 processes'
    record.update(remaining_non_zombie=[],finished_unix=time.time());write(stop_path,record)
    for task in ('textcraft','appworld'):
        folder=ROOT/'receipts'/(task+'-eight-gpu-symmetric-probability-ratio-20261018-v7')
        path=folder/'campaign-state.json'
        if path.exists():
            state=json.loads(path.read_bytes());state.update(status='superseded_by_v8',stop_receipt=str(stop_path))
            for row in state.get('jobs',[]):
                if (row.get('pid'),row.get('birth'))==EXPECTED[1]:
                    row.update(active=False,alive=False,status='user_stopped_superseded_v7',stop_receipt=str(stop_path))
                    write(Path(row['run'])/'source.json',row)
            write(path,state)
    plan=json.loads((PRIOR/'six-combination-manifest.json').read_bytes())
    plan.update(status='superseded_by_v8',stop_receipt=str(stop_path));write(PRIOR/'six-combination-manifest.json',plan)
    for name in ('formal-training.json','active-training.json','active-source.json'):
        value=json.loads((ROOT/name).read_bytes())
        for row in value.get('jobs',[]):
            if (row.get('pid'),row.get('birth'))==EXPECTED[1]:
                row.update(active=False,alive=False,status='user_stopped_superseded_v7',stop_receipt=str(stop_path))
        write(ROOT/name,value)
    print(json.dumps(dict(status=record['status'],remaining_non_zombie=[],receipt=str(stop_path))))
if __name__=='__main__':main()
