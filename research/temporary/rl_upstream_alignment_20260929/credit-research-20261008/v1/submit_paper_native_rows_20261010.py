"""Launch one read-only native paper forward using the frozen runtime."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ROOT,ENTRY,SSH,SCP

OUT = HERE/'paper-native-rows-20261010-v1'
OUT.mkdir(exist_ok=True)
REMOTE = ROOT+'/receipts/paper-native-rows-20261010-v1'
script = HERE/'probe_paper_native_rows_20261010.py'
ast.parse(script.read_text(encoding='utf-8'))
digest = hashlib.sha256(script.read_bytes()).hexdigest()
subprocess.run(SSH+['bash','-s'],input=('mkdir -p '+REMOTE+'\n').encode(),check=True,timeout=25)
subprocess.run(SCP+[str(script),SSH[-1]+':'+REMOTE+'/'],check=True,timeout=35)
body = r'''
import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
root=Path(ROOT);out=Path(REMOTE)
assert not (out/'launch.json').exists(),'Never duplicate this diagnostic'
assert hashlib.sha256((out/'probe_paper_native_rows_20261010.py').read_bytes()).hexdigest()==DIGEST
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*4\s+\d+\s+\S',physical,re.M),'GPU4 occupied'
source=root/'runs/textcraft-formal-stable-20261009-v1/source.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
s=json.loads(source.read_bytes());env=dict(os.environ,**s['environment'])
env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env.update(CUDA_VISIBLE_DEVICES='4',DT_TASK='TextCraft',DT_MAX_STEPS='30',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
dt=Path(env['DT_ROOT']);q=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
observer=root/'receipts/credit-native-identity-operators-appworld-20261009-v2'
assert hashlib.sha256((observer/'inspect_native_identity_operators.py').read_bytes()).hexdigest()=='2f7790af3afac4ac136c68d9489334408bde3f7ee8b8c84fa1da913bfb140451'
env['PYTHONPATH']=':'.join([str(out),str(observer),
 str(root/'receipts/credit-author-development-collection-20261008-v1'),
 str(root/'receipts/direct-target-action-author-curve-20261007-v1'),
 str(root/'runs/textcraft-formal-stable-20261009-v1/entry'),str(dt),q['official_root'],
 str(dt/'clean/qwen35'),s['pythonpath'],q['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(out/'probe_paper_native_rows_20261010.py')]
with (out/'driver.log').open('xb') as log:
 p=subprocess.Popen(argv,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(unix=time.time(),pid=p.pid,birth=psutil.Process(p.pid).create_time(),devices=[4],argv=argv,
 script_sha256=DIGEST,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 physical_before=physical,host_available_before=psutil.virtual_memory().available,
 planned_native_forward_calls=1,DT_calls=0,optimizer_steps=0,production_changes=0)
(out/'launch.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
'''
body = 'ROOT='+repr(ROOT)+'\nREMOTE='+repr(REMOTE)+'\nDIGEST='+repr(digest)+'\n'+body
shell = 'source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
(OUT/'launch-command.sh').write_text(shell,encoding='utf-8',newline='\n')
run = subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=45)
(OUT/'launch.stderr').write_bytes(run.stderr)
run.check_returncode()
d = json.loads(run.stdout)
(OUT/'launch.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in d.items() if k!='physical_before'}))
