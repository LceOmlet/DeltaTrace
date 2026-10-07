"""Reuse the existing author-curve diagnostic for the evidenced TextCraft row.

Only task/source selection and the original resolved scheduler scalar differ.
No metric, sorting, scorer, DT, training update or environment is implemented.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parents[1]
ORIGINAL=AUDIT/'direct-target-action-author-curve-20261007/v1'
spec=importlib.util.spec_from_file_location('transport',AUDIT/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
remote=transport.ROOT+'/receipts/direct-target-textcraft-author-curve-20261008-v1'
files=[ORIGINAL/'inspect_action_curve.py',
       AUDIT/'direct-target-extreme-token-endpoint-20261007/v1/inspect_extreme_endpoint.py',
       AUDIT/'direct-target-prefix-runtime-20261007/v1/textcraft-taskrunner-resolved-training-steps.json']
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
assert hashes['inspect_extreme_endpoint.py']=='8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'
compile(files[0].read_bytes(),str(files[0]),'exec')
command=(ORIGINAL/'launch-command.sh').read_text()
original=json.loads((ORIGINAL/'launch.json').read_bytes())
changes={
    transport.ROOT+'/receipts/direct-target-action-author-curve-20261007-v1':remote,
    'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json':
        'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json',
    '58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0':
        '2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52',
    repr(original['scripts']):repr(hashes),
    "str(out/'results')]":"str(out/'results'),'--case','textcraft',"
        "'--owner-total-training-steps','330','--owner-total-steps-evidence',"
        "str(out/'textcraft-taskrunner-resolved-training-steps.json')]",
    'planned_native_forward_calls_per_rank=42':
        "code_commit="+repr(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())+
        ",case='textcraft',planned_native_forward_calls_per_rank=42",
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
record=json.loads(result.stdout)
(HERE/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
