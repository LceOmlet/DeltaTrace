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
parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group();group.add_argument('--layer',action='store_true');group.add_argument('--subops',action='store_true');group.add_argument('--gdn',action='store_true')
group.add_argument('--memory',action='store_true');group.add_argument('--memory-curve',action='store_true')
group.add_argument('--code-fence-only',action='store_true')
parser.add_argument('--revision',type=int,default=1);parser.add_argument('--failed',action='store_true');args=parser.parse_args()
remote = transport.ROOT + '/receipts/direct-target-existing-pv-rule-20261008-v1-' + ('gdn' if args.gdn else ('subops' if args.subops else ('layers' if args.layer else 'curves')))
if args.gdn and args.revision!=1:remote+='-v'+str(args.revision)
if args.memory or args.memory_curve:remote=transport.ROOT+'/receipts/direct-target-existing-pv-rule-20261008-v1-'+('memory-curves' if args.memory_curve else 'memory')
if args.code_fence_only:remote=transport.ROOT+'/receipts/direct-target-existing-pv-rule-20261008-v1-code-fence'
code = '''from pathlib import Path
import hashlib,json,psutil
root=Path(REMOTE)
launch=json.loads((root/'launch.json').read_bytes())
if not FAILED:assert (root/'results/completed.json').is_file()
else:assert all(json.loads((root/'results'/('rank'+str(i)+'.json')).read_bytes())['phase']=='failed' for i in (0,1))
assert not psutil.pid_exists(launch['pid']) or psutil.Process(launch['pid']).create_time()!=launch['birth']
names=['launch.json','driver.log','curve-inputs.json','curve-physical-mx-smi.jsonl','compare_existing_pv_curves.py','inspect_action_curve.py','inspect_extreme_endpoint.py','results/completed.json','results/effective-config.yaml','results/rank0.json','results/rank1.json']
if MEMORY:
 names=['launch.json','driver.log','compare_existing_pv_rule.py','profile_existing_offload.py','inspect_extreme_endpoint.py','results/completed.json','results/physical-mx-smi.jsonl']
 for rank in (0,1):
  names+=['results/rank'+str(rank)+suffix for suffix in ('.json','-phases.jsonl','-original_symmetric_memory.pt','-existing_forward_memory.pt')]
if FENCE:
 names=['launch.json','driver.log','compare_existing_pv_rule.py','profile_existing_offload.py','inspect_extreme_endpoint.py','results/completed.json','results/physical-mx-smi.jsonl']
 for rank in (0,1):
  names+=['results/rank'+str(rank)+suffix for suffix in ('.json','-phases.jsonl','-original_next_code_fence_score_only.pt')]
if LAYER:
 names=['launch.json','driver.log','case.json','inspect_layer_effect.py','inspect_extreme_endpoint.py','results/completed.json','results/effective-config.yaml']
 for rank in (0,1):
  names+=['results/rank'+str(rank)+suffix for suffix in (('.json','-phases.jsonl') if SUBOPS else ('.json','-phases.jsonl','-single_EOS-signed.pt','-original_joint_EOS-signed.pt'))]
if FAILED:names.remove('results/completed.json')
rows=[]
for name in names:
 p=root/name;data=p.read_bytes();rows.append(dict(path=str(p),relative=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
print(json.dumps(rows))
'''.replace('REMOTE', repr(remote)).replace('LAYER',repr(args.layer or args.subops or args.gdn)).replace('SUBOPS',repr(args.subops or args.gdn)).replace('FAILED',repr(args.failed)).replace('MEMORY',repr(args.memory)).replace('FENCE',repr(args.code_fence_only))
shell = 'set -eu\nsource ' + transport.ENTRY + '/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n'
r = subprocess.run(transport.SSH + ['bash', '-s'], input=shell.encode(), capture_output=True)
if r.returncode:
    print(r.stderr.decode(errors='replace'))
r.check_returncode()
rows = json.loads(r.stdout)
dest = HERE / ('gdn-results' if args.gdn else ('subops-results' if args.subops else ('layer-results' if args.layer else 'curve-results')))
if args.gdn and args.revision!=1:dest=dest.with_name('gdn-v'+str(args.revision)+'-results')
if args.memory or args.memory_curve:dest=HERE/('memory-curve-results' if args.memory_curve else 'memory-results')
if args.code_fence_only:dest=HERE/'code-fence-results'
for row in rows:
    target = dest / row['relative']
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
        subprocess.run(transport.SCP + [transport.SSH[-1] + ':' + row['path'], str(target)], capture_output=True, check=True)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == row['sha256']
    row['local_path'] = str(target)
(dest / 'transport.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(dict(files=len(rows), bytes=sum(x['bytes'] for x in rows))))
