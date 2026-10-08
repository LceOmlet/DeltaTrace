"""Read the original environment owner's position construction; no model run."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1] / 'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
code = r'''
import ast,hashlib,json
from pathlib import Path
root=Path(ROOT)
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'
source=json.loads(source_path.read_bytes())
owner=Path(source['agentgym_root'])/'AgentGym-RL'
paths=[owner/'verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py',
       Path(source['verl_root'])/'verl/utils/torch_functional.py',
       Path(source['verl_root'])/'verl/workers/actor/dp_actor.py',
       Path(source['verl_root'])/'verl/trainer/ppo/ray_trainer.py']
files=[]
for path in paths:
 raw=path.read_bytes()
 text=raw.decode()
 tree=ast.parse(text)
 functions=[]
 for node in ast.walk(tree):
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
   segment=ast.get_source_segment(text,node)
   if 'position_ids' in segment:
    functions.append(dict(name=node.name,line=node.lineno,source=segment))
 files.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),functions=functions))
print(json.dumps(dict(source_path=str(source_path),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),files=files)))
'''
code = 'ROOT=' + repr(transport.ROOT) + '\n' + code
script = 'source ' + transport.ENTRY + '/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n'
result = subprocess.run(transport.SSH + ['bash', '-s'], input=script.encode(), capture_output=True, timeout=60)
(HERE / 'audit-original-positions.stderr.txt').write_bytes(result.stderr)
if result.returncode:
    print(result.stderr.decode(errors='replace'))
result.check_returncode()
value = json.loads(result.stdout)
(HERE / 'original-position-sources.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
for file in value['files']:
    print(json.dumps({k:v for k,v in file.items() if k!='functions'}))
    for function in file['functions']:
        if function['name'] in ('generate_sequences', '_forward_micro_batch', 'compute_position_id_with_mask'):
            print(function['source'])
