"""Read the actual finite-seed/propagation owners; no model or GPU work."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1]/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
code = r'''
import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import psutil
root=Path(@ROOT@)
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
source=json.loads(source_path.read_bytes())
candidate=json.loads((root/'candidates/direct-target-consumed-cache-release-20261007-v1/preparation.json').read_bytes())
env=json.loads(Path(candidate['candidate_environment']).read_bytes())['qwen35']
os.environ.update(source['environment'])
os.environ['CUDA_VISIBLE_DEVICES']=''
dt=Path(candidate['candidate_dt_root'])
sys.path[:0]=[str(dt),os.environ.get('DT_OFFICIAL_ROOT') or env['official_root'],str(dt/'clean/qwen35'),*source['pythonpath'].split(':'),env['ft_extension_root']]
files=[]
for name in ('qwen35_answer_finite','qwen35_gdn_finite','profiles.qwen35_gdn_symmetric','compiled_logprob_seed','finite_fla_gpu','qwen35_dense_finite_runner','reward_readout'):
 spec=importlib.util.find_spec(name)
 assert spec is not None,name
 p=Path(spec.origin)
 files.append(dict(module=name,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),text=p.read_text()))
import torch
assert not torch.cuda.is_initialized()
held=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
print(json.dumps(dict(unix=time.time(),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),files=files,cuda_initialized=False,scope='CPU source resolution only; no model, DT, update, profile deployment or restart',textcraft_same_birth=psutil.Process(2833207).create_time()==1791370325.16,textcraft_release_present=[(held/('rank'+str(i)+'-release-update')).exists() for i in (0,1)])))
'''.replace('@ROOT@',repr(transport.ROOT))
command='set -eu\nsource '+transport.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
(HERE/'joint-finite-sources-command.sh').write_text(command,encoding='utf8',newline='\n')
result=subprocess.run(transport.SSH+['bash','-s'],input=command.encode(),capture_output=True)
(HERE/'joint-finite-sources.stderr.txt').write_bytes(result.stderr)
result.check_returncode()
record=json.loads(result.stdout)
(HERE/'joint-finite-sources.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({**{k:v for k,v in record.items() if k!='files'},'files':[{k:v for k,v in row.items() if k!='text'} for row in record['files']]},ensure_ascii=False,indent=2))
