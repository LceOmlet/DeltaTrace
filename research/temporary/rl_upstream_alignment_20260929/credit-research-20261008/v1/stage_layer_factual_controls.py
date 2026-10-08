"""Stage one bounded passive extension of the existing layer observer.

Only published diagnostic source is copied. No owner, runtime, configuration,
training job or persistent cache is modified by staging.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('transport',AUDIT/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)

task = sys.argv[1]
assert task in ('textcraft','appworld')
out = transport.ROOT+f'/receipts/credit-layer-factual-controls-{task}-20261008-v1'
folder = HERE/f'layer-factual-controls-{task}'
folder.mkdir(exist_ok=True)
files = [HERE/'inspect_layer_collection.py',HERE/'layer-collection-inputs.json',
    AUDIT/'direct-target-textcraft-author-curve-20261008/v1/actual-results/inspect_action_curve.py',
    AUDIT/'direct-target-extreme-token-endpoint-20261007/v1/inspect_extreme_endpoint.py']
hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
archive = folder/'observer-source.tar'
with tarfile.open(archive,'w') as stream:
    for path in files:
        stream.add(path,arcname=path.name)
subprocess.run(transport.SSH+['mkdir','-p',out],capture_output=True,check=True,timeout=30)
subprocess.run(transport.SCP+[str(archive),transport.SSH[-1]+':'+out+'/observer-source.tar'],
               capture_output=True,check=True,timeout=45)
code = r'''
import ast,hashlib,json,os,subprocess,time
from pathlib import Path
out=Path(OUT)
assert not (out/'launch.json').exists(),'Do not alter a launched diagnostic'
subprocess.run(['tar','-xf',str(out/'observer-source.tar'),'-C',str(out)],check=True)
for name,digest in HASHES.items():
 path=out/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
 if path.suffix=='.py':ast.parse(path.read_text())
value=dict(unix=time.time(),task=TASK,remote=str(out),files=HASHES,
 status='prepared_only_not_launched',base_commit=COMMIT,
 extension='Original hook additionally contracts m*(captured_DT_factual-native_factual). No coefficient or credit replacement.',
 frozen_input_sha256=HASHES['layer-collection-inputs.json'],
 model_calls=0,DT_calls=0,updates=0,production_modified=False)
(out/'preparation.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''
header = ('OUT='+repr(out)+'\nTASK='+repr(task)+'\nHASHES='+repr(hashes)
          +'\nCOMMIT='+repr(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())+'\n')
script = 'source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+header+code+'\nPY\n'
result = subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=45)
(folder/'stage.stdout').write_bytes(result.stdout)
(folder/'stage.stderr').write_bytes(result.stderr)
if result.returncode:
    print(result.stderr.decode(errors='replace'))
result.check_returncode()
value = json.loads(result.stdout)
(folder/'preparation.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value))
