"""Reuse completed native-reference launch transport for the current actual token."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parents[1]
ORIGINAL=AUDIT/'direct-target-reference-interaction-20261007/v1'
spec=importlib.util.spec_from_file_location('transport',AUDIT/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
remote=transport.ROOT+'/receipts/direct-target-current-reference-context-20261008-v1'
files=[ORIGINAL/'inspect_reference_context.py',
       AUDIT/'direct-target-extreme-token-endpoint-20261007/v1/inspect_extreme_endpoint.py',
       HERE/'inspect_current_reference.py',HERE/'case.json']
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
assert hashes['inspect_extreme_endpoint.py']=='8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'
command=(ORIGINAL/'launch-command.sh').read_text()
original=json.loads((ORIGINAL/'launch.json').read_bytes())
changes={
    transport.ROOT+'/receipts/direct-target-reference-interaction-20261007-v1':remote,
    repr(original['scripts']):repr(hashes),
    "str(out/'inspect_reference_context.py')":"str(out/'inspect_current_reference.py')",
    'DT_calls=0,optimizer_steps=0':
        "code_commit="+repr(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())+
        ",case_binding="+repr(json.loads((HERE/'case.json').read_bytes()))+",DT_calls=0,optimizer_steps=0",
}
for old,new in changes.items():
    expected=2 if old==repr(original['scripts']) else 1
    assert command.count(old)==expected,(old,command.count(old))
    command=command.replace(old,new)
(HERE/'launch-command.sh').write_text(command,encoding='utf8',newline='\n')
subprocess.run(transport.SSH+['bash','-s'],input=('set -eu\nmkdir -p '+remote+'\n').encode(),capture_output=True,check=True)
for path in files:
    subprocess.run(transport.SCP+[str(path),transport.SSH[-1]+':'+remote+'/'+path.name],capture_output=True,check=True)
result=subprocess.run(transport.SSH+['bash','-s'],input=command.encode(),capture_output=True)
(HERE/'launch.stdout.txt').write_bytes(result.stdout)
(HERE/'launch.stderr.txt').write_bytes(result.stderr)
if result.returncode:
    print(result.stderr.decode(errors='replace'))
result.check_returncode()
launch=json.loads(result.stdout)
(HERE/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps(launch))
