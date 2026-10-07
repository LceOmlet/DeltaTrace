"""Read existing diagnostic phase metadata; no GPU or actor calls."""
import hashlib
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('transport', AUDIT/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)

CODE = r'''
from pathlib import Path
import hashlib,json,psutil,subprocess,time
root=Path(@ROOT@)
out=root/'receipts/direct-target-native-mlp-memory-20261007-v@VERSION@'
record=dict(observed_unix=time.time(),diagnostic=str(out))
launch=out/'launch.json'
record['launch']=json.loads(launch.read_bytes()) if launch.exists() else None
record['physical']=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout
record['cgroup']={str(p):p.read_text() for p in [Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'),Path('/sys/fs/cgroup/memory/memory.stat')] if p.exists()}
record['processes']=[]
for pid in [2833207,2838967,2840776,2786671,3432934]+([] if record['launch'] is None else [record['launch']['pid']]):
 try:
  p=psutil.Process(pid);record['processes'].append(dict(pid=pid,birth=p.create_time(),status=p.status(),name=p.name(),pss=p.memory_full_info().pss))
 except psutil.NoSuchProcess:record['processes'].append(dict(pid=pid,status='NoSuchProcess'))
record['text_holds']=[]
for rank in (0,1):
 p=root/('receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt/rank'+str(rank)+'-hold.json')
 hold=json.loads(p.read_bytes());hold['release_file_exists']=Path(hold['release_file']).exists()
 record['text_holds'].append(hold)
record['files']=[]
for p in sorted(out.rglob('*')):
 if not p.is_file() or p.suffix=='.pt':continue
 b=p.read_bytes();item=dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 if p.name.endswith('phases.jsonl'):
  events=[json.loads(line) for line in b.splitlines()]
  item.update(events=len(events),last=events[-1] if events else None,
   summaries=[event for event in events if event['phase'] in ('DT_begin','DT_complete','failed','complete')])
 elif p.name=='driver.log':item['tail']=b.decode(errors='replace').splitlines()[-18:]
 elif p.suffix=='.json':item['json']=json.loads(b)
 record['files'].append(item)
print(json.dumps(record))
'''


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['3','4','5','6'],default='3')
    args=parser.parse_args()
    code = CODE.replace('@ROOT@',repr(transport.ROOT)).replace('@VERSION@',args.version)
    shell = 'set -eu\nsource '+transport.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
    result = subprocess.run(transport.SSH+['bash','-s'],input=shell.encode(),capture_output=True)
    result.check_returncode()
    record=json.loads(result.stdout)
    target=HERE/('observation-'+str(int(record['observed_unix']))+'.json')
    target.write_text(json.dumps(record,indent=2)+'\n')
    def brief_event(event):
        return {key:value for key,value in event.items() if key not in ('owners','input_receipt')}
    print(json.dumps(dict(path=str(target),processes=record['processes'],
        holds=[dict(rank=x['rank'],hold_entered=x['hold_entered'],release=x['release_file_exists']) for x in record['text_holds']],
        files=[dict(item, **({'last':brief_event(item['last']),
                             'summaries':[brief_event(event) for event in item.get('summaries',[])]}
                            if item.get('last') else {}))
               for item in record['files'] if item['path'].endswith(('phases.jsonl','comparison.json','completed.json'))])))
