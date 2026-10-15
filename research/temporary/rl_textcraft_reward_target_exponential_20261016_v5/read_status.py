"""Read only this version's existing phase/status/log receipts."""
from transport import entry,AUDIT
import sys

script=r'''
import json,psutil,time
from pathlib import Path
audit=Path(AUDIT)
result={'unix':time.time()}
formal_only=FORMAL_ONLY
prepared=audit/'v5-prepared/launch.json'
if prepared.is_file():
    launch=json.loads(prepared.read_bytes());run=Path(launch['run'])
    result['prepared']={'status':launch['status'],'run':str(run),'children':sorted(p.name for p in run.iterdir()) if run.exists() else None}
for name in (('DTPO-v5-job.json',) if formal_only else ('native-behavior-job.json','native-behavior-status.json','DTPO-v5-job.json')):
    path=audit/name
    if path.is_file():
        record=json.loads(path.read_bytes());result[name]=({k:record.get(k) for k in ('pid','birth','status','run','version')} if name=='DTPO-v5-job.json' else record)
        if 'pid' in record:
            try:
                p=psutil.Process(record['pid'])
                result[name]['actual_process']={'birth':p.create_time(),'status':p.status(),
                    'children':[{'pid':c.pid,'birth':c.create_time(),'name':c.name()} for c in p.children(recursive=True) if 'WorkerDict' in c.name()]}
            except psutil.NoSuchProcess:result[name]['actual_process']=None
if formal_only and (audit/'DTPO-v5-job.json').exists():
    job=json.loads((audit/'DTPO-v5-job.json').read_bytes());run=Path(job['run'])
    for name in ('driver.stdout.log','driver.stderr.log'):
        path=run/name
        if path.is_file():result[name]=[line[:700] for line in path.read_text(errors='replace').splitlines()[-12:]]
    path=run/'runtime-overrides/initial-state-and-existing-passive-records-v5.json'
    if path.exists():
        value=json.loads(path.read_bytes());result['initial_runtime_override']={k:value.get(k) for k in ('complete','status','initial_optimizer0_scheduler0_verified')}
for stage in (() if formal_only else ('native-small','native-multi-target','capacity-multi-32768')):
    for rank in (0,1):
        path=audit/f'{stage}-rank{rank}.json'
        if path.is_file():
            record=json.loads(path.read_bytes())
            result[path.name]={k:record.get(k) for k in ('status','stage','rank','pid','birth','seconds','error','finite_calls','credit_calls')}
for name in (() if formal_only else ('native-behavior.stdout.log','native-behavior.stderr.log')):
    path=audit/name
    if path.is_file():result[name]=[line[:700] for line in path.read_text(errors='replace').splitlines()[-10:]]
print(json.dumps(result,indent=2))
'''.replace('Path(AUDIT)','Path('+repr(AUDIT)+')').replace('FORMAL_ONLY',repr('--formal' in sys.argv))
entry.remote('source '+entry.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES="" "$VENV_PYTHON" - <<\'PY\'\n'+script+'\nPY\n')
