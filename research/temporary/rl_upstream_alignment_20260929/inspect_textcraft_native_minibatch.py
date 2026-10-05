"""Read this specific diagnostic's real log/phase/resources; do not restart it."""
import argparse
import json
import subprocess
from stage_environment_entry import AUDIT, ROOT, SSH, SCP


SCRIPT = r'''/opt/conda/bin/python - <<'PY'
import json,pathlib,psutil,subprocess,time
out=pathlib.Path('__OUT__'); job=json.loads((out/'job.json').read_bytes())
result=dict(observed_unix=time.time(),out=str(out),pid=job['pid'],expected_birth=job['pid_birth'])
try:
 p=psutil.Process(job['pid']); result.update(actual_birth=p.create_time(),identity_match=abs(p.create_time()-job['pid_birth'])<.05,status=p.status(),rss_bytes=p.memory_info().rss)
except psutil.NoSuchProcess: result['alive']=False
result['phases']=[json.loads(p.read_bytes()) for p in out.glob('native-minibatch-phase-*.json')]
result['receipts']=[p.name for p in out.glob('*gradients.json')]
log=out/'diagnostic.log'
with log.open('rb') as f:
 f.seek(max(0,log.stat().st_size-7000)); result['log_tail']=f.read().decode(errors='replace')
result['host_available_bytes']=psutil.virtual_memory().available
for name in ['memory.usage_in_bytes','memory.stat']:
 p=pathlib.Path('/sys/fs/cgroup/memory')/name
 if p.exists(): result[name]=p.read_text()
result['physical_mx_smi']=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
p=out/f'current-observation-{int(time.time())}.json';p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(receipt=str(p),pid=result['pid'],alive=result.get('identity_match',False),status=result.get('status'),
 phases=result['phases'],gradient_receipts=result['receipts'],host_available_bytes=result['host_available_bytes'],
 log_tail=result['log_tail'])))
PY
'''


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--version',default='v1')
    p.add_argument('--fetch',action='store_true');args=p.parse_args()
    out=ROOT+'/receipts/textcraft-native-minibatch-20261006-'+args.version
    local=AUDIT/'textcraft-degradation-20261005'/('native-minibatch-'+args.version)
    local.mkdir(exist_ok=True)
    if args.fetch:
        for name in ['diagnostic.log','job.json','prepared-diagnostic.json','native-minibatch-inspect.json',
                     'scalar-owner-tests.xml','scalar-owner-tests.stdout.txt']:
            subprocess.run(SCP+[f'{SSH[-1]}:{out}/{name}',str(local/name)],check=True)
    script=SCRIPT.replace('__OUT__',out)
    result=subprocess.run(SSH+['bash','-s'],input=script.encode(),capture_output=True)
    (local/'inspection.stdout.txt').write_bytes(result.stdout+result.stderr)
    print(result.stdout.decode(errors='replace'));print(result.stderr.decode(errors='replace'))
    result.check_returncode()
