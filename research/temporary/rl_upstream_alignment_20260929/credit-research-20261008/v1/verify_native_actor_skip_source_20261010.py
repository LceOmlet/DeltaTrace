"""Compare the exact native optimizer/actor methods with the pinned owner source."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
code=r'''
import ast,hashlib,json,psutil,subprocess,time
from pathlib import Path
root=Path(ROOT);formal=root/'runs/textcraft-formal-stable-20261009-v1'
assert psutil.Process(982372).create_time()==1791553809.84
s=json.loads((formal/'source.json').read_bytes())
current=Path(s['verl_root'])/'verl/workers/actor/dp_actor.py'
original=root/'candidates/official-verl-20bd331/verl/workers/actor/dp_actor.py'
def methods(path):
 text=path.read_text();tree=ast.parse(text);values={}
 for node in ast.walk(tree):
  if isinstance(node,ast.ClassDef) and node.name=='DataParallelPPOActor':
   for function in node.body:
    if isinstance(function,ast.FunctionDef) and function.name in ['_optimizer_step','update_policy']:
     values[function.name]=dict(ast=ast.dump(function,include_attributes=False),
      text='\n'.join(text.splitlines()[function.lineno-1:function.end_lineno]))
 return values
a=methods(current);b=methods(original)
assert set(a)==set(b)=={'_optimizer_step','update_policy'}
checks={key:a[key]['ast']==b[key]['ast'] for key in a};assert all(checks.values())
record=dict(unix=time.time(),upstream_commit=s['upstream_commit'],
 current=dict(path=str(current),resolved=str(current.resolve()),sha256=hashlib.sha256(current.read_bytes()).hexdigest()),
 owner=dict(path=str(original),resolved=str(original.resolve()),sha256=hashlib.sha256(original.read_bytes()).hexdigest()),
 exact_method_AST_checks=checks,original_method_text={key:b[key]['text'] for key in b},
 physical_mx_smi=subprocess.run(['mx-smi'],capture_output=True,text=True,timeout=15).stdout,
 model_calls=0,optimizer_calls=0,production_changes=0,
 scope='Exact source behavior check for the native actor update and nonfinite skip. Not a numerical tolerance or accuracy test.')
print(json.dumps(record))
'''.replace('ROOT',repr(transport.ROOT),1)
script='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
r=subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=45)
out=HERE/'direct-credit-records-20261009-v1/native-actor-official-skip-source-20261010.json'
out.with_suffix('.stderr').write_bytes(r.stderr);r.check_returncode()
d=json.loads(r.stdout);out.write_bytes(r.stdout)
print(json.dumps({k:v for k,v in d.items() if k!='original_method_text'}))
