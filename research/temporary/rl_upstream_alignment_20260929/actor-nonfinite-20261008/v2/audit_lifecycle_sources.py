"""Read exact deployed lifecycle sources without initializing a model."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1]/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
code = r'''
import ast,hashlib,json
from pathlib import Path
root=Path(ROOT)
source=json.loads((root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json').read_bytes())
env=source['environment']; dt=Path(env['DT_ROOT'])
paths=[(Path(source['verl_root'])/'verl/workers/fsdp_workers.py',
        {'compute_dt_token_advantages','update_actor','compute_log_prob','compute_ref_log_prob'}),
       (Path(source['verl_root'])/'verl/workers/actor/dp_actor.py', {'_forward_micro_batch','update_policy'}),
       (Path(source['entry_root'])/'deltatrace_rollout.py',None)] if 'entry_root' in source else [
       (Path(source['verl_root'])/'verl/workers/fsdp_workers.py',
        {'compute_dt_token_advantages','update_actor','compute_log_prob','compute_ref_log_prob'}),
       (Path(source['verl_root'])/'verl/workers/actor/dp_actor.py', {'_forward_micro_batch','update_policy'}),
       (root/'candidates/direct-target-prefix-runtime-20261007-v1/textcraft/entry/deltatrace_rollout.py',None)]
paths.append((dt/'accelerated/qwen35/native_fla_precision.py',None))
for relative in ('verl/utils/experimental/torch_functional.py',
                 'verl/models/transformers/qwen3_vl.py',
                 'verl/models/transformers/monkey_patch.py'):
 paths.append((Path(source['verl_root'])/relative,None))
result=[]
for path,names in paths:
 raw=path.read_bytes(); text=raw.decode()
 selected={}
 if names is not None:
  tree=ast.parse(text)
  for node in ast.walk(tree):
   if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:
    selected[node.name]=ast.get_source_segment(text,node)
 result.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
                    launch_expected_sha256=source['source_bindings'].get(str(path)),
                    source=text if names is None else selected))
print(json.dumps(result))
'''
code='ROOT='+repr(transport.ROOT)+'\n'+code
script='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
r=subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=60)
(HERE/'lifecycle-source-stderr.txt').write_bytes(r.stderr)
r.check_returncode()
value=json.loads(r.stdout)
(HERE/'lifecycle-sources.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps([dict(path=v['path'],sha256=v['sha256']) for v in value]))
