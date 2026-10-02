"""Read current owner logs/resources without stopping or rerunning training."""
import json
from pathlib import Path
import subprocess

from stage_environment_entry import SSH, ROOT


script = r'''/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python - <<'PY'
import json,pathlib,psutil,subprocess,time
root=pathlib.Path('@ROOT@')
active=json.loads((root/'active-training.json').read_text())
out={'unix':time.time(),'jobs':[]}
for j in active['jobs']:
 item={k:j[k] for k in ['task','pid','entry','verl_root','dt_root']}
 try:
  driver=psutil.Process(j['pid']);item['created_unix']=driver.create_time()
  item['workers']=[];logs=set()
  for p in driver.children(recursive=True):
   if p.name() != 'ray::TaskRunner.run' and not p.name().startswith('ray::WorkerDict'):continue
   x={'pid':p.pid,'name':p.name(),'created_unix':p.create_time(),'pss':p.memory_full_info().pss}
   try:
    q=subprocess.run(['/opt/conda/bin/py-spy','dump','--nonblocking','--pid',str(p.pid)],capture_output=True,text=True,timeout=4)
    x['stack']=q.stdout[:4000];x['stack_returncode']=q.returncode
   except Exception as e:x['stack_unavailable']=repr(e)
   item['workers'].append(x)
   for f in p.open_files():
    if '/worker-' in f.path and f.path.endswith('.out'):logs.add(f.path)
  item['logs']=[]
  for f in sorted(logs):
   hits=[]
   for line in pathlib.Path(f).open(errors='replace'):
    if line.startswith(('step:','[loop_trajectory]','[loop_transport]','[DT EOS plan]','[DT EOS minibatch]','[native_host_cache]')):hits.append(line.rstrip())
   item['logs'].append({'path':f,'tail':hits[-7:]})
  item['checkpoint_markers']={p.name:p.read_text() for p in pathlib.Path(j['checkpoints']).glob('latest*')}
 except psutil.NoSuchProcess:item['terminal']='PID missing'
 out['jobs'].append(item)
out['memory']={line.split(':')[0]:int(line.split()[1])*1024 for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in ['MemTotal','MemAvailable','AnonPages','Shmem']}
out['gpu']=subprocess.check_output(['mx-smi'],text=True)
print(json.dumps(out))
PY
'''.replace('@ROOT@', ROOT)

if __name__ == '__main__':
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(),
                            capture_output=True, timeout=45, check=True)
    receipt = json.loads(result.stdout)
    path = Path(__file__).parent / f'prefix-reuse-inspection-{int(receipt["unix"])}.json'
    path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    for job in receipt['jobs']:
        print(job['task'], job['pid'], job.get('terminal', job.get('checkpoint_markers')))
        for worker in job.get('workers', []):
            print(worker['pid'], worker['name'], worker.get('stack', '')[:1000])
        for log in job.get('logs', []):
            print(log['path'], log['tail'][-3:])
    print(receipt['memory'])
    print(receipt['gpu'][:1800])
    print(path)
