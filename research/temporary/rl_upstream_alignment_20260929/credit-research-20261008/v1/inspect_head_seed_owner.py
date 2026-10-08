"""Read the actually imported scalar seed owner with the existing CPU runtime."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
owner=json.loads((HERE/'layer-suboperations-textcraft-v2-observations/rank0.json').read_bytes())['owners']['qwen35_answer_finite']
body=r'''
import hashlib,importlib,inspect,json,sys,torch
from pathlib import Path
source=json.loads(Path(SOURCE).read_bytes())
dt=Path(source['dt_root']);q=json.loads(Path(source['environment']['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
paths=[str(Path(OWNER['path']).parent),str(dt),str(dt/'clean/qwen35'),
       source['environment'].get('DT_OFFICIAL_ROOT') or q['official_root'],source['pythonpath'],q['ft_extension_root']]
sys.path[:0]=':'.join(paths).split(':')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(OWNER['path'])==OWNER['sha256']
files=[]
for name in ('compiled_logprob_seed','compiled_finite_rules','signed_secant_rules'):
 module=importlib.import_module(name);path=inspect.getsourcefile(module)
 files.append(dict(module=name,path=path,resolved_path=str(Path(path).resolve()),sha256=sha(path),source=Path(path).read_text()))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(files=files,head_owner=OWNER,import_paths=sys.path,source_sha256=sha(SOURCE),CUDA_initialized=False,model_calls=0,DT_calls=0)))
'''
source_path=transport.ROOT+'/runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'
script='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\nOWNER='+repr(owner)+'\nSOURCE='+repr(source_path)+'\n'+body+'\nPY\n'
result=subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=60)
(HERE/'head-seed-owner.stderr.txt').write_bytes(result.stderr)
result.check_returncode();value=json.loads(result.stdout)
out=HERE/'actual-head-seed-owner.json';out.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(output=str(out),files=[{k:v for k,v in f.items() if k!='source'} for f in value['files']])))
