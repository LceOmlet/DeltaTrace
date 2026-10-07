"""Read owner-preserved clean profile using the frozen actual import roots."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
code=r'''
import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
root=Path(@ROOT@)
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
source=json.loads(source_path.read_bytes())
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
candidate=json.loads((root/'candidates/direct-target-consumed-cache-release-20261007-v1/preparation.json').read_bytes())
env=json.loads(Path(candidate['candidate_environment']).read_bytes())['qwen35']
os.environ.update(source['environment'])
dt=Path(candidate['candidate_dt_root'])
sys.path[:0]=[str(dt),os.environ.get('DT_OFFICIAL_ROOT') or env['official_root'],str(dt/'clean/qwen35'),*source['pythonpath'].split(':'),env['ft_extension_root']]
files=[]
for name in ('profiles.official','qwen35_clean_runner'):
 spec=importlib.util.find_spec(name)
 if spec is None:
  files.append(dict(module=name,resolved=False))
 else:
  p=Path(spec.origin)
  files.append(dict(module=name,resolved=True,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),text=p.read_text()))
import torch
assert not torch.cuda.is_initialized()
print(json.dumps(dict(unix=time.time(),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),files=files,cuda_initialized=False,scope='CPU source resolution only, no model or profile deployment')))
'''.replace('@ROOT@',repr(transport.ROOT))
command='set -eu\nsource '+transport.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
(HERE/'clean-profile-owner-command.sh').write_text(command,encoding='utf8',newline='\n')
result=subprocess.run(transport.SSH+['bash','-s'],input=command.encode(),capture_output=True)
(HERE/'clean-profile-owner.stderr.txt').write_bytes(result.stderr)
result.check_returncode()
record=json.loads(result.stdout)
(HERE/'clean-profile-owner.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(record,ensure_ascii=False,indent=2))
