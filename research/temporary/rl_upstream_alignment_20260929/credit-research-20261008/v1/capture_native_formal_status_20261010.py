"""Read existing formal artifacts; update only the same TextCraft job's status.

No worker RPC, imports of Torch, model calls, patches, launches or optimizers.
The previous live-worker import receipt remains bound to the same PID births.
"""
import json
from pathlib import Path
import subprocess
import sys
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--iteration",type=int,required=True)
iteration=parser.parse_args().iteration
if iteration < 1:
    parser.error("iteration must be positive")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import stage_environment_entry as transport

expected = json.loads((HERE / 'stable-version-recheck-20261010-v1' /
                       'disk-and-config.json').read_bytes())['checks']
remote = r'''
import hashlib,json,psutil,re,subprocess,time
from pathlib import Path
r=Path(ROOT);f=r/'runs/textcraft-formal-stable-20261009-v1'
p=psutil.Process(982372)
assert p.create_time()==1791553809.84
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(f/'source.json')=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
checks=[]
for item in EXPECTED:
 path=Path(item['path']);value=sha(path)
 assert value==item['expected'],(str(path),value,item['expected'])
 checks.append(dict(name=item['name'],path=str(path),resolved=str(path.resolve()),sha256=value))
ray=Path('/tmp/ray/session_2026-10-09_21-50-24_958031_982372/logs')
task=next(ray.glob('*-985585.out'))
ansi=re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
lines=ansi.sub('',task.read_text(errors='replace')).splitlines()
marker='step:'+str(ITERATION)+' - '
fifth=[line[line.index(marker):] for line in lines if marker in line]
assert len(fifth)==1,len(fifth)
metrics={key:float(value) for key,value in re.findall(r'([\w/]+):(-?\d+(?:\.\d+)?)',fifth[0])}
assert metrics['step']==ITERATION and metrics['training/global_step']==ITERATION
workers=[];errors={}
bad=['Traceback (most','OutOfMemoryError','FloatingPointError','[Skip the step]',
     'Non-finite grad','grad_norm is not finite']
for pid,birth in [(987808,1791553850.00),(989860,1791553867.51)]:
 q=psutil.Process(pid);assert q.create_time()==birth
 mem=q.memory_full_info()
 workers.append(dict(pid=pid,birth=birth,phase=q.name(),PSS_bytes=mem.pss,
                     RSS_bytes=mem.rss,USS_bytes=mem.uss))
 for suffix in ['out','err']:
  path=next(ray.glob('*-'+str(pid)+'.'+suffix))
  errors[str(path)]=[x for x in path.read_text(errors='replace').splitlines() if any(s in x for s in bad)]
for suffix in ['out','err']:
 path=next(ray.glob('*-985585.'+suffix))
 errors[str(path)]=[x for x in path.read_text(errors='replace').splitlines() if any(s in x for s in bad)]
driver=(f/'train.log').read_text(errors='replace').splitlines()
observed=time.time()
record=dict(unix=observed,pid=p.pid,birth=p.create_time(),source_sha256=sha(f/'source.json'),
 numerical_version='fla-early-output-scale-20261009-v1',numerical_source_commit='26bef6c8',
 original_step_line=fifth[0],original_metrics=metrics,metrics_source=str(task),workers=workers,
 source_checks=checks,errors=errors,completed_formal_iterations=ITERATION,total_formal_iterations=330,
 progress=[x for x in driver if 'Training Progress:' in x or 'Rounds ' in x][-8:],
 host_available_bytes=psutil.virtual_memory().available,disk_free_bytes=psutil.disk_usage(str(r)).free,
 physical_mx_smi=subprocess.run(['mx-smi'],capture_output=True,text=True,timeout=15).stdout,
 cgroup_memory_usage_bytes=Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text().strip(),
 cgroup_memory_stat=Path('/sys/fs/cgroup/memory/memory.stat').read_text(),
 checkpoint_restore=False,manual_training_step_cap=None,production_changes=0,
 scope='Read existing native step metrics, same PID births and disk hashes bound to prior live import receipt. No fresh worker RPC or model/DT/optimizer calls.')
updates=[]
for name in ['formal-training.json','active-training.json','active-source.json']:
 path=r/name;data=json.loads(path.read_bytes());old_other=[x for x in data['jobs'] if x.get('pid')!=982372]
 jobs=[x for x in data['jobs'] if x.get('pid')==982372];assert len(jobs)==1
 job=jobs[0];old_status=job.get('status')
 job.update(status='formal_running_after_completed_iteration'+str(ITERATION),
  last_status_observed_unix=observed,completed_iterations_observed=ITERATION,total_iterations_observed=330,
  latest_original_step_metrics=metrics,latest_original_metrics_source=str(task),
  last_native_worker_phases=[dict(pid=x['pid'],name=x['phase']) for x in workers])
 if 'complete_iterations_observed' in job:job['complete_iterations_observed']=ITERATION
 if 'completed_iterations' in job:job['completed_iterations']=ITERATION
 assert [x for x in data['jobs'] if x.get('pid')!=982372]==old_other
 previous=sha(path);path.write_text(json.dumps(data,indent=2)+'\n')
 updates.append(dict(path=str(path),previous_sha256=previous,sha256=sha(path),
                     previous_status=old_status,status=job['status']))
record['authority_status_updates']=updates
out=r/'receipts/direct-credit-records-20261009-v1'/('formal-iteration-'+str(ITERATION)+'-complete-20261010.json')
out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''.replace('ROOT', repr(transport.ROOT), 1).replace('EXPECTED', repr(expected), 1).replace('ITERATION',str(iteration))
script = 'source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+remote+'\nPY\n'
result = subprocess.run(transport.SSH+['bash', '-s'], input=script.encode(),
                        capture_output=True, timeout=50)
(HERE/'direct-credit-records-20261009-v1'/('formal-iteration-'+str(iteration)+'-status.stderr')).write_bytes(result.stderr)
result.check_returncode()
record = json.loads(result.stdout)
out = HERE/'direct-credit-records-20261009-v1'/('formal-iteration-'+str(iteration)+'-complete-20261010.json')
out.write_bytes(result.stdout)
print(json.dumps({key:record[key] for key in ['unix','pid','birth','numerical_version',
 'completed_formal_iterations','total_formal_iterations','original_metrics','workers','errors']}))
print('SAVED',out)
