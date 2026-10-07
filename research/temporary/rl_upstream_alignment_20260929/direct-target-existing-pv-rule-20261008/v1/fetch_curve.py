"""Fetch terminal original-author comparison artifacts with exact SHA checks."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1] / 'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group();group.add_argument('--layer',action='store_true');group.add_argument('--subops',action='store_true');args=parser.parse_args()
remote = transport.ROOT + '/receipts/direct-target-existing-pv-rule-20261008-v1-' + ('subops' if args.subops else ('layers' if args.layer else 'curves'))
code = '''from pathlib import Path
import hashlib,json,psutil
root=Path(REMOTE)
launch=json.loads((root/'launch.json').read_bytes())
assert (root/'results/completed.json').is_file()
assert not psutil.pid_exists(launch['pid']) or psutil.Process(launch['pid']).create_time()!=launch['birth']
names=['launch.json','driver.log','curve-inputs.json','curve-physical-mx-smi.jsonl','compare_existing_pv_curves.py','inspect_action_curve.py','inspect_extreme_endpoint.py','results/completed.json','results/effective-config.yaml','results/rank0.json','results/rank1.json']
if LAYER:
 names=['launch.json','driver.log','case.json','inspect_layer_effect.py','inspect_extreme_endpoint.py','results/completed.json','results/effective-config.yaml']
 for rank in (0,1):
  names+=['results/rank'+str(rank)+suffix for suffix in (('.json','-phases.jsonl') if SUBOPS else ('.json','-phases.jsonl','-single_EOS-signed.pt','-original_joint_EOS-signed.pt'))]
rows=[]
for name in names:
 p=root/name;data=p.read_bytes();rows.append(dict(path=str(p),relative=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
print(json.dumps(rows))
'''.replace('REMOTE', repr(remote)).replace('LAYER',repr(args.layer or args.subops)).replace('SUBOPS',repr(args.subops))
shell = 'set -eu\nsource ' + transport.ENTRY + '/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n'
r = subprocess.run(transport.SSH + ['bash', '-s'], input=shell.encode(), capture_output=True)
if r.returncode:
    print(r.stderr.decode(errors='replace'))
r.check_returncode()
rows = json.loads(r.stdout)
dest = HERE / ('subops-results' if args.subops else ('layer-results' if args.layer else 'curve-results'))
for row in rows:
    target = dest / row['relative']
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
        subprocess.run(transport.SCP + [transport.SSH[-1] + ':' + row['path'], str(target)], capture_output=True, check=True)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == row['sha256']
    row['local_path'] = str(target)
(dest / 'transport.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(dict(files=len(rows), bytes=sum(x['bytes'] for x in rows))))
