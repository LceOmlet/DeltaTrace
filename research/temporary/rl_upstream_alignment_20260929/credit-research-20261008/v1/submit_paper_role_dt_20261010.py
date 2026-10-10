"""Submit only the requested literal paper-input DT implementation comparison."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ROOT, ENTRY, SSH, SCP, REPO

OUT=HERE/'paper-role-implementation-check-20261010-v1'
REMOTE=ROOT+'/receipts/paper-role-implementation-check-20261010-v1'
script=HERE/'probe_paper_role_dt_20261010.py'
fixture=[];sources=[]
for name in ('overview_role_case.json','cases.json','overview_case.json'):
    p=REPO/'paper/iclr2027/figures/data'/name
    fixture.extend(c for c in json.loads(p.read_bytes())['cases'] if c['model']=='qwen35')
    sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
assert len(fixture)==4 and len({(c['dataset'],c['index']) for c in fixture})==4
inputs=OUT/'paper-inputs.json'
assert not inputs.exists(), 'Preserve original literal inputs'
inputs.write_text(json.dumps(dict(cases=fixture,sources=sources),indent=2)+'\n',encoding='utf-8')
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (script,inputs)}
subprocess.run(SSH+['bash','-s'],input=('mkdir -p '+REMOTE+'\n').encode(),check=True,timeout=25)
subprocess.run(SCP+[str(script),str(inputs),SSH[-1]+':'+REMOTE+'/'],check=True,timeout=40)
body=r'''
import ast,hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
out=Path(REMOTE);root=Path(ROOT)
assert not (out/'launch.json').exists(),'Do not duplicate this diagnostic'
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*4\s+\d+\s+\S',physical,re.M),'GPU4 is occupied'
assert psutil.Process(982372).create_time()==1791553809.84
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,value in HASHES.items():assert sha(out/name)==value
ast.parse((out/'probe_paper_role_dt_20261010.py').read_text())
p=root/'runs/textcraft-formal-stable-20261009-v1/source.json'
assert sha(p)=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
s=json.loads(p.read_bytes());env=dict(os.environ,**s['environment'])
env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env.update(CUDA_VISIBLE_DEVICES='4',DT_TASK='TextCraft',DT_MAX_STEPS='30',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
dt=Path(env['DT_ROOT']);q=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(root/'runs/textcraft-formal-stable-20261009-v1/entry'),str(dt),
 q['official_root'],str(dt/'clean/qwen35'),s['pythonpath'],q['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(out/'probe_paper_role_dt_20261010.py')]
with (out/'driver.log').open('xb') as stream:
 process=subprocess.Popen(argv,cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(unix=time.time(),pid=process.pid,birth=psutil.Process(process.pid).create_time(),
 devices=[4],script_sha256=HASHES['probe_paper_role_dt_20261010.py'],inputs_sha256=HASHES['paper-inputs.json'],
 source_sha256=sha(p),argv=argv,physical_before=physical,host_available_before=psutil.virtual_memory().available,
 planned_DT_calls=2,original_paper_cases=4,optimizer_steps=0,rollout=0,production_changes=0,
 scope='Paper-example implementation diagnosis; current official symmetric and official clean profiles on same literal inputs and base weights. No new algorithm, numerical tolerance, training restart or parameter change.')
(out/'launch.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
'''
body='REMOTE='+repr(REMOTE)+'\nROOT='+repr(ROOT)+'\nHASHES='+repr(hashes)+'\n'+body
shell='source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
(OUT/'launch-command.sh').write_text(shell,encoding='utf-8',newline='\n')
run=subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=40)
(OUT/'launch.stderr').write_bytes(run.stderr);run.check_returncode()
d=json.loads(run.stdout)
(OUT/'launch.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in d.items() if k!='physical_before'}))
