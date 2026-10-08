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
suboperations = '--suboperations' in sys.argv[2:]
attention_branches = '--attention-branches' in sys.argv[2:]
attention_input = '--attention-input' in sys.argv[2:]
attention_pv = '--attention-pv' in sys.argv[2:]
assert not attention_branches or suboperations
assert not attention_input or attention_branches
assert not attention_pv or attention_input
revision = next((v.split('=',1)[1] for v in sys.argv[2:] if v.startswith('--revision=')), 'v1')
assert revision in ('v1','v2')
tag = 'attention-pv' if attention_pv else ('attention-input' if attention_input else ('attention-gate' if attention_branches else ('layer-suboperations' if suboperations else 'layer-factual-controls')))
date = '20261009' if attention_branches else '20261008'
out = transport.ROOT+f'/receipts/credit-{tag}-{task}-{date}-{revision}'
folder = HERE/(f'{tag}-{task}'+('-'+revision if revision!='v1' else ''))
folder.mkdir(exist_ok=True)
files = [HERE/'inspect_layer_collection.py',HERE/'layer-collection-inputs.json',
    AUDIT/'direct-target-textcraft-author-curve-20261008/v1/actual-results/inspect_action_curve.py',
    AUDIT/'direct-target-extreme-token-endpoint-20261007/v1/inspect_extreme_endpoint.py']
if suboperations:
    files += [HERE/'passive_suboperations.py',HERE/('attention-pv-protocol.json' if attention_pv else ('attention-input-protocol.json' if attention_input else ('attention-gate-protocol.json' if attention_branches else 'suboperation-protocol.json')))]
if attention_branches:
    files += [HERE/'passive_attention_gate.py']
if attention_pv:
    files += [HERE/'passive_attention_pv.py']
names = {p: ('suboperation-protocol.json' if p.name in ('attention-gate-protocol.json','attention-input-protocol.json','attention-pv-protocol.json') else p.name) for p in files}
hashes = {names[p]:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
if attention_branches:
    protocol = json.loads(next(p for p in files if p.name.endswith('protocol.json')).read_bytes())
    assert bool(protocol.get('read_attention_input')) == attention_input
    assert bool(protocol.get('read_attention_pv')) == attention_pv
    for item in protocol['inputs']:
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
git_blobs = {}
if attention_branches:
    repo = HERE.parents[4]
    for path in files:
        relative = path.relative_to(repo).as_posix()
        blob = subprocess.check_output(['git','-c','core.longpaths=true','show',commit+':'+relative])
        assert hashlib.sha256(blob).hexdigest() == hashes[names[path]], relative
        git_blobs[names[path]] = dict(commit=commit,path=relative,sha256=hashes[names[path]])
archive = folder/'observer-source.tar'
with tarfile.open(archive,'w') as stream:
    for path in files:
        stream.add(path,arcname=names[path])
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
 extension=('Passive suboperation readout of unchanged callbacks; no coefficient or credit replacement.' if SUBOPERATIONS else
            'Original hook additionally contracts m*(captured_DT_factual-native_factual). No coefficient or credit replacement.'),
 suboperations=SUBOPERATIONS,
 attention_branches=ATTENTION_BRANCHES,attention_input=ATTENTION_INPUT,attention_pv=ATTENTION_PV,source_git_blobs=GIT_BLOBS,
 frozen_input_sha256=HASHES['layer-collection-inputs.json'],
 model_calls=0,DT_calls=0,updates=0,production_modified=False)
(out/'preparation.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''
header = ('OUT='+repr(out)+'\nTASK='+repr(task)+'\nHASHES='+repr(hashes)
          +'\nSUBOPERATIONS='+repr(suboperations)
          +'\nATTENTION_BRANCHES='+repr(attention_branches)+'\nGIT_BLOBS='+repr(git_blobs)
          +'\nATTENTION_INPUT='+repr(attention_input)
          +'\nATTENTION_PV='+repr(attention_pv)
          +'\nCOMMIT='+repr(commit)+'\n')
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
