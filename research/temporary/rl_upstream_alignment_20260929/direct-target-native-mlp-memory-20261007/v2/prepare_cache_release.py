"""Prepare an isolated DT owner lifetime patch; do not deploy or train."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('transport',AUDIT/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
original=AUDIT/'direct-target-mlp-token-chunk-20261007/v1/candidate/qwen35_dense_finite_runner.py'
source=original.read_bytes()
assert hashlib.sha256(source).hexdigest()=='628006b637516f8d62e95583a9eb51fe9038ea7931798e2c1f42c28e154cf24f'
before="            if callable(release_layer):release_layer(layer)\n            if observer is not None:observer.boundary(str(i),m.detach(),root[str(i)])"
after="""            if callable(release_layer):release_layer(layer)
            if offload_mixer and is_fa and replay_cache is not None:
                # This native cached layer has now replayed and its finite
                # consumer has finished. Reverse traversal never reads its
                # K/V again. Keep the owner's layer type and index, releasing
                # only this local replay cache's consumed tensor storage.
                replay_cache.layers[i]=type(replay_cache.layers[i])()
            if observer is not None:observer.boundary(str(i),m.detach(),root[str(i)])"""
text=source.decode();assert text.count(before)==1
changed=text.replace(before,after)
ast.parse(changed)
candidate=HERE/'cache-release-candidate/qwen35_dense_finite_runner.py'
candidate.parent.mkdir(parents=True,exist_ok=True);candidate.write_text(changed,encoding='utf-8',newline='\n')
candidate_sha=hashlib.sha256(candidate.read_bytes()).hexdigest()
remote=transport.ROOT+'/candidates/direct-target-consumed-cache-release-20261007-v1'
subprocess.run(transport.SSH+['bash','-s'],input=('set -eu\nmkdir -p '+remote+'\n').encode(),check=True)
subprocess.run(transport.SCP+[str(candidate),transport.SSH[-1]+':'+remote+'/qwen35_dense_finite_runner.py'],check=True)
code=r'''
from pathlib import Path
import hashlib,json,os,subprocess,time
root=Path(@ROOT@);out=Path(@OUT@)
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
source=json.loads(source_path.read_bytes());base=Path(source['dt_root'])
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
replacement=out/'qwen35_dense_finite_runner.py'
assert hashlib.sha256(replacement.read_bytes()).hexdigest()==@SHA@
tree=out/'deltatrace';relative=Path('clean/qwen35/qwen35_dense_finite_runner.py')
def overlay(old,new,remaining):
 new.mkdir(exist_ok=False)
 for item in old.iterdir():
  target=new/item.name
  if item.name==remaining.parts[0]:
   if len(remaining.parts)==1:target.write_bytes(replacement.read_bytes())
   else:overlay(item,target,Path(*remaining.parts[1:]))
  else:target.symlink_to(item,target_is_directory=item.is_dir())
overlay(base,tree,relative)
original_environment=Path(source['candidate_environment']['path'])
assert hashlib.sha256(original_environment.read_bytes()).hexdigest()==source['candidate_environment']['sha256']
environment=json.loads(original_environment.read_bytes())
assert environment['qwen35']['dt_offload_replay_mixer'] is False
environment['qwen35']['dt_offload_replay_mixer']=True
envpath=out/'environment.json';envpath.write_text(json.dumps(environment,indent=2)+'\n')
record=dict(prepared_unix=time.time(),status='prepared_only_unaccepted',base_source_path=str(source_path),base_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),base_dt_root=str(base),candidate_dt_root=str(tree),changed_file=str(tree/relative),changed_sha256=@SHA@,original_sha256='628006b637516f8d62e95583a9eb51fe9038ea7931798e2c1f42c28e154cf24f',candidate_environment=str(envpath),candidate_environment_sha256=hashlib.sha256(envpath.read_bytes()).hexdigest(),environment_change={'dt_offload_replay_mixer':{'before':False,'after':True}},scope='Only consumed local replay FA cache storage lifetime; HF/PEFT/FA/FLA/finite math/QVA/PPO/task configurations unchanged. Offload-disabled default unchanged. No formal restart, update, checkpoint, module replacement in another job or install.')
(out/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''.replace('@ROOT@',repr(transport.ROOT)).replace('@OUT@',repr(remote)).replace('@SHA@',repr(candidate_sha))
shell='set -eu\nsource '+transport.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
(HERE/'cache-release-candidate/prepare-command.sh').write_text(shell,encoding='utf-8',newline='\n')
result=subprocess.run(transport.SSH+['bash','-s'],input=shell.encode(),capture_output=True)
(HERE/'cache-release-candidate/prepare.stdout.txt').write_bytes(result.stdout)
(HERE/'cache-release-candidate/prepare.stderr.txt').write_bytes(result.stderr)
result.check_returncode()
record=json.loads(result.stdout)
(HERE/'cache-release-candidate/preparation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
