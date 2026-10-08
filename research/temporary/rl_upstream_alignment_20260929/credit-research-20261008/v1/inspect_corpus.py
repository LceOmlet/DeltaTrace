"""Read saved capture metadata and process identities; no model or CUDA use."""
from pathlib import Path
import json
import subprocess
import sys

AUDIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AUDIT))
from stage_environment_entry import SSH, ENTRY

BODY = r'''
import time,json,resource
from pathlib import Path
import psutil
import torch
torch.set_num_threads(1)
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
result={"unix":time.time(),"tasks":{},"processes":{}}
for task in ('textcraft','appworld'):
 p=next(iter(sorted((root/'receipts/direct-target-prefix-runtime-20261007-v1'/(task+'-first-dt')).glob('rank*-readout-native-batch-*.pt'))))
 d=torch.load(p,map_location='cpu',weights_only=False);r=d['rows'][0]
 result['tasks'][task]={"path":str(p),"payload_keys":list(d),"saved_keys":list(r),"row_keys":list(r['row']),"case_keys":list(r['case']),"small_row_metadata":{k:str(v)[:350] for k,v in r['row'].items() if not isinstance(v,torch.Tensor)},"files":[{"name":f.name,"bytes":f.stat().st_size} for f in p.parent.iterdir() if f.is_file() and 'native-batch' not in f.name]}
 prepared=torch.load(p.parent/'rank0-input-prepared.pt',map_location='cpu',weights_only=False)
 def describe(x):
  if isinstance(x,dict):return {str(k):describe(v) for k,v in x.items()}
  if isinstance(x,torch.Tensor):return {'type':'tensor','shape':list(x.shape),'dtype':str(x.dtype)}
  if isinstance(x,(list,tuple)):return {'type':type(x).__name__,'length':len(x),'first':describe(x[0]) if x else None}
  if hasattr(x,'non_tensor_batch'):return {'type':str(type(x)),'non_tensor_batch':describe(x.non_tensor_batch),'batch_keys':list(x.batch),'meta_info':describe(x.meta_info)}
  return str(x)[:200]
 result['tasks'][task]['prepared']=describe(prepared)
 del prepared,d
for task,pid,birth in [('textcraft',2833207,1791370325.16),('appworld',2786671,1791369896.67)]:
 try:
  p=psutil.Process(pid)
  result['processes'][task]={"pid":pid,"birth":p.create_time(),"same_birth":p.create_time()==birth,"status":p.status()}
 except psutil.NoSuchProcess:
  result['processes'][task]={"pid":pid,"missing":True}
hold=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
result['text_update_releases']=[(hold/f'rank{rank}-release-update').exists() for rank in (0,1)]
result['resources']={"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"cuda_initialized":torch.cuda.is_initialized()}
assert not result['resources']['cuda_initialized']
print(json.dumps(result,ensure_ascii=False))
'''

if __name__ == '__main__':
    script = ('source '+ENTRY+'/metax-entry.env.sh\n'
              'CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+BODY+'\nPY\n')
    run = subprocess.run(SSH+['bash','-s'], input=script.encode(), capture_output=True, timeout=90)
    (Path(__file__).parent/'inspection.stderr.txt').write_bytes(run.stderr)
    run.check_returncode()
    value = json.loads(run.stdout)
    (Path(__file__).parent/'inspection.json').write_bytes(run.stdout)
    print(json.dumps(value, ensure_ascii=False, indent=2))
