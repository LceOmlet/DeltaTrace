source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$VENV_PYTHON" - <<'PY'
from pathlib import Path
import json,hashlib,time,psutil,subprocess,os
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');f=r/'runs/textcraft-formal-stable-20261009-v1'
assert psutil.Process(982372).create_time()==1791553809.84
assert hashlib.sha256((f/'source.json').read_bytes()).hexdigest()=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
ray=Path('/tmp/ray/session_2026-10-09_21-50-24_958031_982372/logs')
workers=[];paths=set();uids=set()
for pid in [987808,989860]:
 p=next(ray.glob('*-'+str(pid)+'.out'));raw=p.read_bytes();reports=0;batch_paths=[]
 for line in raw.decode(errors='replace').splitlines():
  marker='[DT direct joint record] '
  if marker in line:batch_paths.append(json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0]['path'])
  marker='[DeltaTrace readout] '
  if marker in line:
   reports+=1
   if reports==7:
    report=json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0]
    break
   batch_paths=[]
 else:raise RuntimeError('Seventh original DT has not returned; do not label partial records complete')
 assert len(batch_paths)==report['finite_trace_calls']
 paths.update(batch_paths);uids.update(str(v['traj_uid']) for v in report['traces'])
 workers.append(dict(pid=pid,birth=psutil.Process(pid).create_time(),source=str(p),source_bytes=len(raw),source_sha256=hashlib.sha256(raw).hexdigest(),paths=batch_paths,report=report))
times=[int(Path(p).stem.removeprefix('joint-'))/1e9 for p in paths]
lo=min(times)-.001;hi=max(times)+.001
selected={str(p) for p in (f/'credit-records').glob('*/*.pt') if lo<int(p.stem.removeprefix('joint-'))/1e9<=hi}
assert selected==paths
base=r/'receipts/direct-credit-records-20261009-v1'
output=base/'complete-seventh-DT-record-audit.json';assert not output.exists()
script=base/'audit_saved_joint_records_v2.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()=='cef95774efcebb8e8abdaac7aa3c5ec71bd38c500a09834938ba3369bb8e8ca8'
a=subprocess.run([os.environ['VENV_PYTHON'],str(script),'--after-unix',str(lo),'--before-unix',str(hi),'--output',str(output)],capture_output=True)
(base/'seventh-DT-audit-owner.stdout').write_bytes(a.stdout);(base/'seventh-DT-audit-owner.stderr').write_bytes(a.stderr)
print(a.stdout.decode());print(a.stderr.decode());a.check_returncode()
audit=json.loads(output.read_bytes())
assert {row['path'] for row in audit['rows']}==paths
assert {row['traj_uid'] for row in audit['rows']}==uids
task=next(ray.glob('*-985585.out'));lines=task.read_text(errors='replace').splitlines()
work=[line for line in lines if '[DT direct target workload]' in line]
record=dict(unix=time.time(),formal_pid=982372,formal_birth=1791553809.84,
 workers=workers,actual_original_workload=work[6] if len(work)>=7 else None,
 taskrunner_source=str(task),expected_paths=sorted(paths),paths_exactly_matched=True,
 unique_UIDs=len(uids),UIDs_exactly_matched=True,original_rank_rows=len(audit['rows']),
 source_sha256=hashlib.sha256((f/'source.json').read_bytes()).hexdigest(),
 native_record_time_bounds=dict(after_unix=lo,before_unix=hi),
 minimum_raw_row=min(audit['rows'],key=lambda x:x['minimum_prior_source']['raw_advantage']),
 complete_formal_iterations=6,actor_complete=False)
(base/'seventh-DT-complete-observation.json').write_text(json.dumps(record,indent=2)+'\n')
bounds=r/'receipts/formal-credit-probability-bounds-20261010-v1'
b=subprocess.run([os.environ['VENV_PYTHON'],str(bounds/'audit_formal_credit_probability_bounds.py'),'--audit',str(output),'--output',str(bounds/'seventh-result.json')],capture_output=True)
(bounds/'seventh-owner.stdout').write_bytes(b.stdout);(bounds/'seventh-owner.stderr').write_bytes(b.stderr)
print(b.stdout.decode());print(b.stderr.decode());b.check_returncode()
print(json.dumps({k:v for k,v in record.items() if k not in ['workers','expected_paths']}))
PY
