source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'

import hashlib,json,math,os,psutil,re,subprocess,time
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');formal=root/'runs/textcraft-formal-stable-20261009-v1';ray=Path('/tmp/ray/session_2026-10-09_21-50-24_958031_982372/logs');p=psutil.Process(982372)
assert p.create_time()==1791553809.84
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(formal/'source.json')=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
log=next(ray.glob('*-985585.out'));raw=log.read_bytes();lines=raw.decode(errors='replace').splitlines();step=next(line for line in reversed(lines) if line.startswith('step:4 - '))
metrics={name:float(value) for name,value in re.findall(r'([^\s:]+):(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)',step)}
assert metrics['training/global_step']==4 and all(math.isfinite(v) for v in metrics.values())
errors={}
for pid in [987808,989860,985585]:
 for suffix in ['out','err']:
  path=next(ray.glob('*-'+str(pid)+'.'+suffix));errors[str(path)]=[line for line in path.read_text(errors='replace').splitlines() if any(k in line for k in ['Traceback (most','OutOfMemoryError','FloatingPointError','[Skip the step]','Non-finite grad','grad_norm is not finite'])]
record=dict(unix=time.time(),pid=p.pid,birth=p.create_time(),source_sha256=sha(formal/'source.json'),original_step4=step,metrics=metrics,
 taskrunner_source=dict(path=str(log),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),errors=errors,
 phase=[dict(pid=c.pid,name=c.name()) for c in p.children(recursive=True) if c.name().startswith('ray::WorkerDict')],
 physical_mx_smi=subprocess.check_output(['mx-smi'],text=True),worker_PSS_bytes={str(pid):psutil.Process(pid).memory_full_info().pss for pid in [987808,989860]},host_available_bytes=psutil.virtual_memory().available,
 completed_formal_iterations=4,total_formal_iterations=330,numerical_version='fla-early-output-scale-20261009-v1',numerical_source_commit='26bef6c8',checkpoint_restore=False,source_configuration_unchanged=True)
assert not any(errors.values())
updates=[]
for name in ['formal-training.json','active-training.json','active-source.json']:
 path=root/name;data=json.loads(path.read_bytes());job=next(j for j in data['jobs'] if j['task']=='TextCraft');assert job['pid']==p.pid
 before=dict(job);job.update(status='formal_running_iteration5_sampling_fourth_iteration_complete',completed_iterations_observed=4,total_iterations_observed=330,last_status_observed_unix=record['unix'])
 tmp=path.with_name(path.name+'.fourth-status.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)
 updates.append(dict(path=str(path),sha256=sha(path),before=before,after=job))
record['authority_updates']=updates
out=root/'receipts/direct-credit-records-20261009-v1/fourth-formal-iteration-complete-20261010.json';assert not out.exists();out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['unix','pid','birth','source_sha256','metrics','phase','errors','completed_formal_iterations','total_formal_iterations']}))

PY
