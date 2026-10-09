source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import json,hashlib,psutil,subprocess,time
from pathlib import Path
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');run=r/'runs/textcraft-formal-stable-20261009-v1';p=psutil.Process(982372);assert p.create_time()==1791553809.84
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest();assert sha(run/'source.json')=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
gdn=r/'candidates/appworld-row-cuts-finite-20261007-v1/production-wiring-v1/deltatrace/clean/qwen35/qwen35_gdn_finite.py';assert sha(gdn)=='7c06d5e0a4d6c00c13483dadd27d6389e25eee7868a05666d2d1faeaa2d62656'
workers=[];errors={};ray=Path('/tmp/ray/session_2026-10-09_21-50-24_958031_982372/logs');bad=['Traceback (most','OutOfMemoryError','FloatingPointError','[Skip the step]','Non-finite grad','grad_norm is not finite']
for pid,birth in [(987808,1791553850.00),(989860,1791553867.51)]:
 q=psutil.Process(pid);assert q.create_time()==birth;m=q.memory_full_info();workers.append(dict(pid=pid,birth=birth,phase=q.name(),PSS_bytes=m.pss,USS_bytes=m.uss))
for pid in [987808,989860,985585]:
 for suffix in ['out','err']:
  path=next(ray.glob('*-'+str(pid)+'.'+suffix));errors[str(path)]=[line for line in path.read_text(errors='replace').splitlines() if any(x in line for x in bad)]
lines=(run/'train.log').read_text(errors='replace').splitlines();d=dict(unix=time.time(),pid=p.pid,birth=p.create_time(),source_sha256=sha(run/'source.json'),gdn_resolved=str(gdn.resolve()),gdn_sha256=sha(gdn),workers=workers,errors=errors,host_available_bytes=psutil.virtual_memory().available,cgroup_memory_usage_bytes=int(Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text()),physical_mx_smi=subprocess.run(['mx-smi'],capture_output=True,text=True,timeout=15).stdout,progress=[x for x in lines if 'Rounds ' in x or 'Training Progress:' in x][-8:],completed_formal_iterations=6,total_formal_iterations=330,seventh_sampling_complete=True,seventh_DT_complete=False,seventh_actor_update_complete=False,production_changes=0,model_DT_optimizer_calls_added=0,scope='Read-only native seventh post-sampling phase/resources and same disk hashes tied to prior actual live imports. No restart, RPC, patch, model, DT or optimizer call.')
updates=[]
for name in ['formal-training.json','active-training.json','active-source.json']:
 path=r/name;data=json.loads(path.read_bytes());others=[j for j in data['jobs'] if j.get('pid')!=982372];job=next(j for j in data['jobs'] if j.get('pid')==982372);job.update(status='formal_running_iteration7_native_post_sampling',last_status_observed_unix=d['unix'],last_native_worker_phases=[dict(pid=x['pid'],name=x['phase']) for x in workers]);assert others==[j for j in data['jobs'] if j.get('pid')!=982372];before=sha(path);path.write_text(json.dumps(data,indent=2)+chr(10));updates.append(dict(path=str(path),previous_sha256=before,sha256=sha(path)))
d['authority_status_updates']=updates;out=r/'receipts/native-target-records-20261010-v1/seventh-post-sampling-status.json';out.write_text(json.dumps(d,indent=2)+chr(10));print(json.dumps(d))

PY
