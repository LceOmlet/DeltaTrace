"""Observe the exact conditional-V process; optionally fetch terminal artifacts."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--fetch',action='store_true');parser.add_argument('--factual-jacobian',action='store_true');args=parser.parse_args()
label='native-factual-jacobian' if args.factual_jacobian else 'native-conditional-v'
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
code=r'''
import hashlib,json,psutil,subprocess,time
from pathlib import Path
root=Path(@ROOT@);out=root/('receipts/direct-target-existing-pv-rule-20261008-v1-gdn-v2/'+@LABEL@)
launch=json.loads((out/'launch.json').read_bytes())
alive=psutil.pid_exists(launch['pid']) and psutil.Process(launch['pid']).create_time()==launch['birth']
result=json.loads((out/'results/result.json').read_bytes()) if (out/'results/result.json').is_file() else None
held=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
record=dict(unix=time.time(),launch=launch,alive_same_birth=alive,result=result,
    physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout,
    cgroup_memory_bytes=int(Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text()),
    textcraft_same_birth=psutil.Process(2833207).create_time()==1791370325.16,
    textcraft_releases=[(held/('rank'+str(i)+'-release-update')).exists() for i in (0,1)])
if alive:record['process_pss_bytes']=psutil.Process(launch['pid']).memory_full_info().pss
if not result or result['phase']=='failed':record['driver_log_tail']=(out/'driver.log').read_text(errors='replace')[-5000:]
if @FETCH@:
 assert not alive and result and result['phase']=='complete'
 names=['launch.json','driver.log','results/result.json','before-physical.txt']
 record['files']=[]
 for name in names:
  p=out/name;data=p.read_bytes();record['files'].append(dict(path=str(p),relative=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
print(json.dumps(record))
'''.replace('@ROOT@',repr(transport.ROOT)).replace('@FETCH@',repr(args.fetch)).replace('@LABEL@',repr(label))
command='set -eu\nsource '+transport.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
result=subprocess.run(transport.SSH+['bash','-s'],input=command.encode(),capture_output=True)
if result.returncode:print(result.stderr.decode(errors='replace'))
result.check_returncode();record=json.loads(result.stdout)
(HERE/(label+'-observation-'+str(int(record['unix']))+'.json')).write_text(json.dumps(record,indent=2)+'\n')
if args.fetch:
    dest=HERE/(label+'-results')
    for row in record['files']:
        target=dest/row['relative'];target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(transport.SCP+[transport.SSH[-1]+':'+row['path'],str(target)],capture_output=True,check=True)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
        row['local_path']=str(target)
    (dest/'transport.json').write_text(json.dumps(record['files'],indent=2)+'\n')
record.pop('physical');record.pop('files',None)
if record['result']:
    record['result']={k:v for k,v in record['result'].items() if k in ('pid','birth','phase','summary','traceback')}
print(json.dumps(record))
