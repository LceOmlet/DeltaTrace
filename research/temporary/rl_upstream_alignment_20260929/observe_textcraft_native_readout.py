"""Read the isolated reader job's actual phases/resources and completed receipts."""
import json
from pathlib import Path
import subprocess

from stage_textcraft_native_readout import OUT, SSH, SCP


LOCAL = Path(__file__).parent / 'textcraft-degradation-20261005/readout-quality-20261006/v2'
SCRIPT = r'''/opt/conda/bin/python - <<'PY'
import json,pathlib,psutil,subprocess,time
out=pathlib.Path('__OUT__'); job=json.loads((out/'job.json').read_bytes())
same=False; processes=[]
try:
 p=psutil.Process(job['pid']); same=abs(p.create_time()-job['pid_birth'])<.05
 if same:
  for proc in [p]+p.children(recursive=True):
   try:
    mem=proc.memory_full_info()
    processes.append(dict(pid=proc.pid,birth=proc.create_time(),name=proc.name(),
     status=proc.status(),rss_bytes=mem.rss,pss_bytes=getattr(mem,'pss',None)))
   except (psutil.NoSuchProcess,psutil.AccessDenied): pass
except psutil.NoSuchProcess: pass
ranks=[]
for rank in range(2):
 path=out/f'rank{rank}-readout.json'
 if path.exists():
  row=json.loads(path.read_bytes())
  ranks.append(dict(rank=rank,phase=row['phase'],observed_unix=row['observed_unix'],
   completed_cases=len(row['cases']),native_forward_calls=row['native_forward_calls'],
   last_case=row['cases'][-1] if row['cases'] else None,
   sources=row['sources']))
log=out/'diagnostic.log'; text=log.read_text(errors='replace')
completed=(out/'completed.json').exists()
snapshot=dict(observed_unix=time.time(),job_pid=job['pid'],job_birth=job['pid_birth'],
 same_process_alive=same,completed=completed,processes=processes,ranks=ranks,
 host_available_bytes=psutil.virtual_memory().available,
 physical_mx_smi=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout,
 log_tail=text.splitlines()[-35:])
print(json.dumps(snapshot))
PY
'''


if __name__ == '__main__':
    LOCAL.mkdir(exist_ok=True, parents=True)
    run = subprocess.run(SSH + ['bash', '-s'], input=SCRIPT.replace('__OUT__', OUT).encode(),
                         capture_output=True)
    run.check_returncode()
    data = json.loads(run.stdout)
    path = LOCAL / f"observation-{int(data['observed_unix'])}.json"
    path.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), completed=data['completed'],
                         alive=data['same_process_alive'], host_available=data['host_available_bytes'],
                         ranks=[{k:row[k] for k in ['rank','phase','completed_cases','native_forward_calls']}
                                for row in data['ranks']], log_tail=data['log_tail'][-10:]), indent=2))
    if data['completed']:
        for name in ['job.json', 'prepared.json', 'completed.json', 'native-owner-inspection.json',
                     'effective-config.yaml', 'rank0-readout.json', 'rank1-readout.json', 'diagnostic.log']:
            subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/{name}', str(LOCAL / name)], check=True)
