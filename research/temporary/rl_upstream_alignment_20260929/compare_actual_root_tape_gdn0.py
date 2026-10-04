"""Compare saved real native operands; no GPU, re-forward or numeric gate."""
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import ENTRY, REPO, SSH

raw_path=REPO/'experiments/rl/results_native_root_tape_gdn0_20261004.json'
raw=json.loads(raw_path.read_bytes());assert raw['completed']
script=r'''source @ENTRY@/metax-entry.env.sh
CUDA_VISIBLE_DEVICES='' "$VENV_PYTHON" - <<'PY'
import hashlib,json,pathlib,torch,xml.etree.ElementTree as ET
out=pathlib.Path(@OUT@);results=[]
for rank in (0,1):
 report=json.loads((out/f'rank{rank}.json').read_bytes())
 operands={}
 for label in ('shared_warm','root_tape_disabled_warm','root_tape_warm'):
  recorded=report['reports'][label]['actual_gdn0_observations'][0]['records'][0]
  path=pathlib.Path(recorded['path'])
  assert hashlib.sha256(path.read_bytes()).hexdigest()==recorded['sha256']
  operands[label]=torch.load(path,map_location='cpu',weights_only=True)
 comparisons=[]
 for label in ('root_tape_disabled_warm','root_tape_warm'):
  fields=[];reference=operands['shared_warm']
  for name,value in operands[label]['tensors'].items():
   expected=reference['tensors'][name]
   if value is None or expected is None:
    fields.append(dict(field=name,none_equal=value is expected));continue
   same_shape=value.shape==expected.shape
   fields.append(dict(field=name,shape=list(value.shape),dtype=str(value.dtype),
    shape_equal=same_shape,dtype_equal=value.dtype==expected.dtype,
    equal=bool(torch.equal(value,expected)),maximum_absolute_difference=(
     float((value.double()-expected.double()).abs().max()) if same_shape else None)))
  comparisons.append(dict(variant=label,reference='shared_warm',fields=fields,
   call_arguments_equal=operands[label]['calls']==reference['calls']))
 results.append(dict(rank=rank,comparisons=comparisons))
path=out/'gdn0-actual-operand-comparison.json'
value=dict(rows=results,source_scope='Actual native captured suffix from real rows40-43; equality is a transport observation, not a custom tolerance',
 cpu_tests=[n.attrib for n in ET.parse(out/'cpu-tests.xml').getroot().iter('testsuite')])
with path.open('x') as stream:json.dump(value,stream,indent=2)
print(json.dumps(dict(value=value,receipt=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))))
PY
'''.replace('@ENTRY@',ENTRY).replace('@OUT@',repr(raw['remote_root']))
run=subprocess.run(SSH+['bash','-s'],input=script.encode(),stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT,timeout=45)
if run.returncode:
    print(run.stdout.decode('utf8','replace'));raise SystemExit(run.returncode)
value=json.loads(run.stdout)
value['source_readout']=dict(path=str(raw_path),sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
path=REPO/'experiments/rl/results_native_root_tape_gdn0_operands_20261004.json'
with path.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(dict(local=str(path),cpu_tests=value['value']['cpu_tests'],
 changed=[dict(rank=row['rank'],comparisons=[dict(variant=c['variant'],call_arguments_equal=c['call_arguments_equal'],
 unequal=[f for f in c['fields'] if not f.get('equal',f.get('none_equal',False))]) for c in row['comparisons']])
 for row in value['value']['rows']]),indent=2))
