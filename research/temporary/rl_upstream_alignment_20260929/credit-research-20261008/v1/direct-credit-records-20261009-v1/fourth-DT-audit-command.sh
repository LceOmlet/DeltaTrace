source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$VENV_PYTHON" - <<'PY'

import hashlib,json,psutil,subprocess,time
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');formal=root/'runs/textcraft-formal-stable-20261009-v1';out=root/'receipts/direct-credit-records-20261009-v1';before=time.time();after=1791562253.0762296
script=out/'audit_saved_joint_records_v2.py';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(script)=='cef95774efcebb8e8abdaac7aa3c5ec71bd38c500a09834938ba3369bb8e8ca8'
destination=out/'complete-fourth-DT-record-audit.json';assert not destination.exists()
args=['/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python',str(script),'--after-unix',str(after),'--before-unix',str(before),'--output',str(destination)]
r=subprocess.run(args,capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
audit=json.loads(destination.read_bytes());paths={row['path'] for row in audit['rows']};uids={row['traj_uid'] for row in audit['rows']}
ray=Path('/tmp/ray/session_2026-10-09_21-50-24_958031_982372/logs');reports=[];nativeuids=set();passive=[];sources=[];errors={}
for pid in [987808,989860]:
 path=next(ray.glob('*-'+str(pid)+'.out'));raw=path.read_bytes();lines=raw.decode(errors='replace').splitlines();allreports=[json.loads(line.split('[DeltaTrace readout] ',1)[1]) for line in lines if line.startswith('[DeltaTrace readout] ') and 'real_executed_action_joint' in line]
 assert len(allreports)==4,len(allreports)
 report=allreports[3];reports.append(dict(pid=pid,report=report));nativeuids.update(str(row['traj_uid']) for row in report['traces'])
 sources.append(dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
 for line in lines:
  if line.startswith('[DT direct joint record] '):
   entry=json.loads(line.split('[DT direct joint record] ',1)[1]);stamp=int(Path(entry['path']).stem.removeprefix('joint-'))/1e9
   if after<stamp<=before:passive.append(entry)
 for suffix in ['out','err']:
  p=next(ray.glob('*-'+str(pid)+'.'+suffix));errors[str(p)]=[line for line in p.read_text(errors='replace').splitlines() if any(k in line for k in ['Traceback (most','OutOfMemoryError','FloatingPointError','[Skip the step]','Non-finite grad','grad_norm is not finite'])]
assert nativeuids==uids and {p['path'] for p in passive}==paths
p=psutil.Process(982372);assert p.create_time()==1791553809.84
source_sha=sha(formal/'source.json');assert source_sha=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
tasklog=next(ray.glob('*-985585.out'));workloads=[line for line in tasklog.read_text(errors='replace').splitlines() if line.startswith('[DT direct target workload]')];assert len(workloads)==4
record=dict(unix=time.time(),pid=p.pid,birth=p.create_time(),source_sha256=source_sha,
 fourth_original_request=workloads[3],actual_request_UIDs=len(nativeuids),all_original_request_UIDs_equal_saved=True,
 all_original_passive_paths_equal_saved=True,original_worker_report_sources=sources,original_worker_reports=reports,
 passive_record_log=passive,audit_stdout=json.loads(r.stdout),audit_path=str(destination),audit_sha256=sha(destination),
 phase=[dict(pid=c.pid,name=c.name()) for c in p.children(recursive=True) if c.name().startswith('ray::WorkerDict')],
 worker_PSS_bytes={str(pid):psutil.Process(pid).memory_full_info().pss for pid in [987808,989860]},
 host_available_bytes=psutil.virtual_memory().available,physical_mx_smi=subprocess.check_output(['mx-smi'],text=True),
 errors=errors,operations=dict(model=0,DT=0,GPU=0,optimizer=0,rollout=0))
for filename,value in [('fourth-DT-complete-observation.json',record)]:
 dest=out/filename;assert not dest.exists();dest.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['unix','pid','birth','fourth_original_request','actual_request_UIDs','all_original_request_UIDs_equal_saved','all_original_passive_paths_equal_saved','audit_stdout','phase','worker_PSS_bytes','host_available_bytes','errors']}))

PY
