"""Read the active DT owners and the paper fixture; no model or training calls."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from stage_environment_entry import ENTRY, SSH

OUT = HERE / 'paper-role-implementation-check-20261010-v1'
body = r'''
import hashlib,json,psutil,subprocess,time
from pathlib import Path
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
f=r/'runs/textcraft-formal-stable-20261009-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((f/'source.json').read_bytes());e=s['environment']
q=json.loads(Path(e['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
dt=Path(e['DT_ROOT'])
files=[dt/'profiles/official.py',dt/'profiles/qwen35_gdn_symmetric.py',
 dt/'clean/qwen35/qwen35_clean_runner.py',
 dt/'clean/qwen35/qwen35_gdn_finite.py',
 dt/'clean/qwen35/qwen35_dense_finite_runner.py',
 f/'entry/deltatrace_rollout.py']
records=[]
for p in files:
 v=dict(path=str(p),exists=p.exists())
 if p.exists():v.update(resolved=str(p.resolve()),sha256=sha(p),text=p.read_text())
 records.append(v)
workers=[]
for pid,birth in [(982372,1791553809.84),(987808,1791553850.),(989860,1791553867.51)]:
 try:
  p=psutil.Process(pid);workers.append(dict(pid=pid,expected_birth=birth,birth=p.create_time(),name=p.name()))
 except psutil.NoSuchProcess:workers.append(dict(pid=pid,exists=False))
result=dict(unix=time.time(),source_sha256=sha(f/'source.json'),
 owners=records,workers=workers,config=q,
 environment_paths={k:v for k,v in e.items() if k in ['DT_ROOT','DT_OFFICIAL_ROOT','DT_ENVIRONMENT_JSON','VENV_PYTHON','PYTHONPATH']},
 pythonpath=s.get('pythonpath'),startup=s.get('startup_options'),
 physical=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout,
 host_available=psutil.virtual_memory().available,
 model_calls=0,DT_calls=0,optimizer_calls=0,production_changes=0)
print(json.dumps(result))
'''
shell = 'source ' + ENTRY + '/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n' + body + '\nPY\n'
OUT.mkdir(exist_ok=True)
assert not (OUT/'runtime-source.json').exists(), 'Preserve the first observation'
run=subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=50)
(OUT/'runtime-source-command.sh').write_text(shell,encoding='utf-8',newline='\n')
(OUT/'runtime-source.stderr').write_bytes(run.stderr)
run.check_returncode()
d=json.loads(run.stdout)
(OUT/'runtime-source.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(unix=d['unix'],source_sha256=d['source_sha256'],
 workers=d['workers'],files=[{k:v for k,v in p.items() if k!='text'} for p in d['owners']],
 config_keys=list(d['config']),physical=d['physical'],host_available=d['host_available'],
 output=str(OUT/'runtime-source.json')),ensure_ascii=False))
