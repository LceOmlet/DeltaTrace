"""Read installed native recorder capability; no process group or CUDA work."""
from pathlib import Path
import json
import subprocess

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / 'direct-credit-records-20261009-v1'
PREFIX = RAW / 'native-flight-recorder-capability-20261011-v1'
REMOTE = r'''source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import hashlib,importlib.util,json,os,pickle,resource,subprocess,time
from pathlib import Path
import torch
import torch.distributed as dist
c10d=torch._C._distributed_c10d
out={'unix':time.time(),'torch_version':torch.__version__,'torch_git':torch.version.git_version,
 'CUDA_initialized_before':torch.cuda.is_initialized(),'process_group_initialized':dist.is_initialized(),
 'native_apis':{},'installed_analysis_modules':{},'environment':{k:v for k,v in os.environ.items() if k.startswith(('TORCH_NCCL','MCCL_DEBUG'))}}
for name in sorted(n for n in dir(c10d) if any(s in n.lower() for s in ('dump_nccl','flight','trace_buffer'))):
 obj=getattr(c10d,name);out['native_apis'][name]={'doc':getattr(obj,'__doc__',None)}
 if name in ('_dump_nccl_trace','_dump_nccl_trace_json'):
  try:
   value=obj(includeCollectives=True,includeStackTraces=False,onlyActive=False)
   parsed=pickle.loads(value) if isinstance(value,bytes) else json.loads(value)
   out['native_apis'][name].update({'returned_type':type(value).__name__,'returned_bytes':len(value),'keys':list(parsed),'entries':len(parsed.get('entries',[]))})
  except Exception as e:out['native_apis'][name]['read_error']=repr(e)
for name in ('torch.distributed.flight_recorder.fr_trace','torch.distributed.flight_recorder.components.builder','torch.distributed.flight_recorder.components.types'):
 try:
  spec=importlib.util.find_spec(name);p=Path(spec.origin) if spec and spec.origin else None
  out['installed_analysis_modules'][name]={'path':str(p) if p else None,'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p and p.is_file() else None}
 except Exception as e:out['installed_analysis_modules'][name]={'error':repr(e)}
out['CUDA_initialized_after']=torch.cuda.is_initialized()
out['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
r=subprocess.run(['mx-smi'],capture_output=True,text=True,timeout=15);out['physical']=r.stdout;out['physical_returncode']=r.returncode
print(json.dumps(out))
PY
'''
SSH = ['ssh','-oBatchMode=yes','-oConnectTimeout=12','-oStrictHostKeyChecking=yes',
       '-oServerAliveInterval=15','-oServerAliveCountMax=3',
       '-oHostKeyAlias=[ssh.v5000-prod-gw.nhss.zhejianglab.com]:32036',
       '-p','31146','root@ssh.v5000-prod-gw.nhss.zhejianglab.com','bash','-s']

def main():
    PREFIX.with_suffix('.command.txt').write_bytes(REMOTE.encode())
    result=subprocess.run(SSH,input=REMOTE.replace('\r\n','\n').encode(),capture_output=True,timeout=90)
    PREFIX.with_suffix('.stdout.txt').write_bytes(result.stdout)
    PREFIX.with_suffix('.stderr.txt').write_bytes(result.stderr)
    result.check_returncode()
    parsed=json.loads(result.stdout)
    PREFIX.with_suffix('.json').write_bytes((json.dumps(parsed,indent=2)+'\n').encode())
    print(json.dumps(parsed,ensure_ascii=False))

if __name__=='__main__':
    main()
