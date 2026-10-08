"""Read native CPU-offload synchronization and activation hooks, without CUDA."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1] / 'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
code = r'''
import ast,hashlib,inspect,json,sys,torch
from pathlib import Path
root=Path(ROOT)
source=json.loads((root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json').read_bytes())
folder=Path(torch.__file__).parent/'distributed/fsdp'
paths=[folder/'_fully_shard/_fsdp_param_group.py',folder/'_fully_shard/_fsdp_param.py',folder/'_fully_shard/_fsdp_collectives.py',
       Path(source['verl_root'])/'verl/utils/activation_offload.py',Path(source['verl_root'])/'verl/protocol.py']
files=[]
for p in paths:
 if not p.exists(): continue
 raw=p.read_bytes(); text=raw.decode(); selected=[]
 for n in ast.walk(ast.parse(text)):
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
   s=ast.get_source_segment(text,n)
   if ((p.name=='protocol.py' and n.name=='to') or
       (p.name!='protocol.py' and any(x in s for x in ('grad_offload','grad_cpu','grad_offloading','offload_event','cpu_offload','offload_to_cpu','record_event','wait_event')))):
    selected.append(dict(name=n.name,line=n.lineno,source=s))
 files.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),functions=selected,
                   source=text if p.name=='activation_offload.py' else None))
print(json.dumps(dict(torch_version=torch.__version__,torch_path=torch.__file__,cuda_initialized=torch.cuda.is_initialized(),files=files)))
'''
script='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\nROOT='+repr(transport.ROOT)+'\n'+code+'\nPY\n'
r=subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=60)
(HERE/'native-offload-source.stderr.txt').write_bytes(r.stderr)
r.check_returncode()
result=json.loads(r.stdout)
(HERE/'native-offload-sources.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(torch_version=result['torch_version'],CUDA_initialized=result['cuda_initialized'],
 files=[dict(path=f['path'],sha256=f['sha256'],functions=[n['name'] for n in f['functions']]) for f in result['files']])))
