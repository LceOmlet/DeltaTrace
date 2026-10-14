"""Read original TaskRunner logs and existing passive vectors from the new run."""
import hashlib
import json
from pathlib import Path
import psutil
import time
import torch

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
job=json.loads((OUT/'DT-group-multiplier-v40-job.json').read_bytes())
run=Path(job['run']);p=psutil.Process(job['pid']);assert p.create_time()==job['birth']
runner=next(c for c in p.children(recursive=True) if c.name().startswith('ray::TaskRunner'))
lines=[]
for session in Path('/tmp/ray').glob('session_*_'+str(job['pid'])):
    for path in (session/'logs').glob('worker-*-'+str(runner.pid)+'.out'):
        for number,line in enumerate(path.read_text(errors='replace').splitlines(),1):
            if any(marker in line for marker in ('[DT direct target workload]','[DT direct joint minibatch]',
                                                  '[DT direct joint record]','step:','Traceback','NaN','OOM')):
                lines.append(dict(path=str(path),line=number,text=line[-2500:]))
vectors=[]
for directory in sorted((run/'credit-records').glob('rank*-pid*')):
    paths=sorted(directory.glob('joint-*.pt'))
    for path in dict.fromkeys([paths[0],paths[-1]] if paths else []):
        data=torch.load(path,map_location='cpu',weights_only=False)
        vectors.append(dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
            files_so_far=len(paths),rows=[dict(traj_uid=row['traj_uid'],group_multiplier=row['reward'],
                target_tokens=int(row['target_mask'].sum()),policy_tokens=int(row['policy_mask'].sum()),
                finite_DT_source_log_ratios=bool(torch.isfinite(row['source_log_ratios'][row['prior_source_mask']]).all()))
                for row in data['rows']]))
report=dict(unix=time.time(),job=job,native_log_evidence=lines[-24:],first_existing_vectors=vectors,
            negative_multiplier_rows_in_first_files=sum(row['group_multiplier']<0 for item in vectors for row in item['rows']),
            reward_field_semantics='Existing credit-record reward field is g_Y, not raw environment reward.',
            new_model_calls=0,new_DT_calls=0,optimizer_calls=0,rootcause_repair_verified=False)
assert not torch.cuda.is_initialized()
dest=OUT/'first-native-group-credit-observation-v40.json'
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
