"""Read the exact original-author comparison job; no runtime mutation."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1] / 'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
parser = argparse.ArgumentParser()
group=parser.add_mutually_exclusive_group();group.add_argument('--layer', action='store_true');group.add_argument('--subops',action='store_true')
group.add_argument('--gdn',action='store_true')
parser.add_argument('--revision',type=int,default=1)
args = parser.parse_args()
code = r'''
import json,psutil,subprocess,time
from pathlib import Path
root=Path(ROOT);out=root/('receipts/direct-target-existing-pv-rule-20261008-v1-'+('gdn' if GDN else ('subops' if SUBOPS else ('layers' if LAYER else 'curves'))))
if GDN and REVISION!=1:out=out.with_name(out.name+'-v'+str(REVISION))
a=json.loads((out/'launch.json').read_bytes())
p=psutil.Process(a['pid']) if psutil.pid_exists(a['pid']) else None
r=dict(unix=time.time(),driver=dict(pid=a['pid'],birth=a['birth'],same_birth=p is not None and p.create_time()==a['birth']),completed=(out/'results/completed.json').exists(),ranks=[])
for rank in (0,1):
 f=out/'results'/('rank'+str(rank)+'.json')
 if f.exists():
  d=json.loads(f.read_bytes())
  entry={k:d[k] for k in ('rank','pid','birth','phase','active_view','active_mode','active_decoder','completed_points','last_point','traceback','native_forward_calls','decoder_subops','gdn_subops') if k in d}
  entry['views']={key:dict(author_return=value['author_return'],points=len(value['score_points']),seconds=sum(x['seconds'] for x in value['score_points'])) for key,value in d.get('views',{}).items()}
  entry['layer_modes']={key:{k:v for k,v in value.items() if k in ('candidate_signed','seconds','remaining_snapshot_bytes')} for key,value in d.get('phases',{}).items()}
  entry['layer_contractions']={key:{k:v for k,v in value.items() if k in ('joint_coefficient_times_single_deletion_delta','factual_endpoints_equal','factual_endpoint_maxabs')} for key,value in d.get('cross_boundary_contractions',{}).items()}
  if psutil.pid_exists(d['pid']):
   w=psutil.Process(d['pid'])
   if w.create_time()==d['birth']:entry['current_pss_bytes']=w.memory_full_info().pss
  r['ranks'].append(entry)
r['physical_mx_smi']=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
for name in ('memory.usage_in_bytes','memory.stat'):r[name]=(Path('/sys/fs/cgroup/memory')/name).read_text()
r['textcraft_same_birth']=psutil.Process(2833207).create_time()==1791370325.16
r['textcraft_release_present']=[(root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'/('rank'+str(i)+'-release-update')).exists() for i in (0,1)]
if not r['ranks'] or not r['driver']['same_birth']:r['driver_log_tail']=(out/'driver.log').read_text(errors='replace')[-6000:]
print(json.dumps(r))
'''.replace('ROOT', repr(transport.ROOT)).replace('LAYER', repr(args.layer)).replace('SUBOPS',repr(args.subops)).replace('GDN',repr(args.gdn)).replace('REVISION',repr(args.revision))
shell = 'set -eu\nsource ' + transport.ENTRY + '/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n'
r = subprocess.run(transport.SSH + ['bash', '-s'], input=shell.encode(), capture_output=True)
if r.returncode:
    print(r.stderr.decode(errors='replace'))
r.check_returncode()
record = json.loads(r.stdout)
prefix=('gdn-v'+str(args.revision)+'-') if args.gdn and args.revision!=1 else None
(HERE / ((prefix or ('gdn-' if args.gdn else ('subops-' if args.subops else ('layer-' if args.layer else 'curve-')))) + 'observation-' + str(int(record['unix'])) + '.json')).write_text(json.dumps(record, indent=2) + '\n')
# Exact positions remain in the saved receipt; keep the live report compact.
for entry in record['ranks']:
    if 'last_point' in entry:
        entry['last_point'] = {k:v for k,v in entry['last_point'].items() if k != 'changed_input_positions'}
physical = record.pop('physical_mx_smi')
record['physical_memory_mib'] = {}
gpu = None
for line in physical.splitlines():
    board = re.match(r'^\|\s*(\d+)\s+MetaX\s', line)
    if board: gpu = board[1]
    memory = re.search(r'(\d+)/(\d+) MiB', line)
    if memory: record['physical_memory_mib'][gpu] = int(memory[1])
record['cgroup_bytes'] = int(record.pop('memory.usage_in_bytes'))
record.pop('memory.stat')
if 'driver_log_tail' in record: record['driver_log_tail'] = record['driver_log_tail'][-1200:]
print(json.dumps(record))
