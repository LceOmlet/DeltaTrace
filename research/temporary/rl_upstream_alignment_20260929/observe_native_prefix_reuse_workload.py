"""Read the isolated replay's actual phase and physical resource usage."""
import argparse
import json
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ENTRY, ROOT, SSH


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default=ROOT+'/receipts/owner-b8-dispatch-20260930/native-prefix-reuse-workload-20261004-712795d')
    args=parser.parse_args()
    out=args.out
    script=r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import json,pathlib,psutil,subprocess,time
out=pathlib.Path('@OUT@');job=json.loads((out/'job.json').read_bytes())
r=dict(observed_unix=time.time(),out=str(out),pid=job['pid'],pid_birth=job['pid_birth'])
try:
 p=psutil.Process(job['pid']);assert p.create_time()==job['pid_birth']
 r['process_status']=p.status();processes=[p]+p.children(recursive=True)
 r['pss_bytes']=sum(x.memory_full_info().pss for x in processes if x.is_running())
except psutil.NoSuchProcess:r['process_status']='exited'
r['ranks']=[]
for rank in (0,1):
 p=out/f'rank{rank}.json'
 if p.exists():
  a=json.loads(p.read_bytes())
  record={k:a.get(k) for k in ('rank','pid','phase','observed_unix','variant','total_wall_seconds','shared_preparation') if k in a}
  record['phase_age_seconds']=time.time()-a['observed_unix']
  if 'reports' in a:
   record['reports']={key:{k:value[k] for k in ('total_wall_seconds','original_runner_phase_seconds','original_runner_phase_counts') if k in value}
                       for key,value in a['reports'].items()}
  r['ranks'].append(record)
p=out/'probe.log'
if p.exists():
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-131072));tail=f.read().decode('utf8','replace')
 r['log_last_stages']=[line[:2200] for line in tail.splitlines() if any(k in line for k in
  ('[DT EOS minibatch]','[DT EOS plan]','Traceback','AssertionError','ValueError','Error:','Exception','diagnostic_complete','variant_complete'))][-12:]
r['result_exists']=(out/'result.json').exists()
r['physical']=subprocess.check_output(['mx-smi'],text=True)
r['memory_available_bytes']=psutil.virtual_memory().available
paths=(pathlib.Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'),pathlib.Path('/sys/fs/cgroup/memory.current'))
r['cgroup_memory_bytes']=next((int(p.read_text()) for p in paths if p.exists()),None)
path=out/f'observation-{int(time.time())}.json';path.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(remote_receipt=str(path),**r),indent=2))
PY
'''.replace('@ENTRY@',ENTRY).replace('@OUT@',out)
    r=subprocess.run(SSH+['bash','-s'],input=script.encode(),stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT,timeout=45)
    text=r.stdout.decode('utf8','replace')
    if r.returncode==0:
        result=json.loads(text)
        directory=AUDIT/'phase-observation-20261004'
        directory.mkdir(exist_ok=True)
        (directory/f'prefix-reuse-{int(result["observed_unix"])}.json').write_text(text,encoding='utf8')
        concise={k:result[k] for k in ('remote_receipt','observed_unix','out','pid','pid_birth','process_status','result_exists')}
        concise['pss_bytes']=result.get('pss_bytes')
        concise['last_stages']=result['log_last_stages'][-4:]
        concise['ranks']=[]
        for rank in result['ranks']:
            item={k:v for k,v in rank.items() if k!='reports'}
            if 'reports' in rank:
                item['times']={k:v['total_wall_seconds'] for k,v in rank['reports'].items()}
            concise['ranks'].append(item)
        print(json.dumps(concise,ensure_ascii=False,indent=2))
    else:
        print(text)
    raise SystemExit(r.returncode)
