"""Reuse the completed operator launch transport for one conditional-V probe."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--factual-jacobian',action='store_true');args=parser.parse_args()
label='native-factual-jacobian' if args.factual_jacobian else 'native-conditional-v'
program='inspect_factual_jacobian.py' if args.factual_jacobian else 'inspect_conditional_v.py'
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
helper=HERE/'replay_native_fla_output.py'
compile(helper.read_bytes(),str(helper),'exec')
digest=hashlib.sha256(helper.read_bytes()).hexdigest()
remote=transport.ROOT+'/receipts/direct-target-existing-pv-rule-20261008-v1-gdn-v2/'+program
subprocess.run(transport.SCP+[str(helper),transport.SSH[-1]+':'+remote],capture_output=True,check=True)
command=(HERE/'native-memory-orders-launch-command.sh').read_text()
changes={
    "out=parent/'native-memory-orders'":"out=parent/"+repr(label),
    'observe_native_memory_orders.py':program,
    '49d17f99e51bdaa841506c8d4455446ecc8596d0eaeb6e2de1b87ff934dde16d':digest,
    '--observe-memory-orders':'--factual-jacobian' if args.factual_jacobian else '--conditional-v-only',
    '4e7faa795ccfd5fdb5b989c7b17d6828585bcee3':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'Passive original memory endpoint order observation on saved actual operands; no model/full-DT/rollout/actor/training-backward/optimizer/restore':('Original native factual-endpoint VJP on saved actual single-deletion FLA operands; fixed incoming state and original output cotangent; operator-only forward/backward, no model/full DT/credit/profile/update/rollout/restore or tolerance change' if args.factual_jacobian else 'Conditional V only on saved actual single V operands, other inputs factual; unchanged native FLA and existing forward coefficients; no model/full-DT/rollout/actor/training-backward/optimizer/restore or production rule change'),
}
for old,new in changes.items():
    assert old in command,old
    command=command.replace(old,new)
(HERE/(label+'-launch-command.sh')).write_text(command,encoding='utf8',newline='\n')
result=subprocess.run(transport.SSH+['bash','-s'],input=command.encode(),capture_output=True)
(HERE/(label+'-launch.stderr.txt')).write_bytes(result.stderr)
(HERE/(label+'-launch.stdout.txt')).write_bytes(result.stdout)
result.check_returncode()
launch=json.loads(result.stdout)
(HERE/(label+'-launch.json')).write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps(launch))
