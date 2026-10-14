"""Compare saved first-iteration DT inputs; CPU read only, no recomputation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
import torch

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
source=json.loads((OUT/'DT-historical-timing-versions-v48.json').read_bytes())
report=dict(unix=time.time(),runs={},model_calls=0,DT_calls=0,optimizer_calls=0)

def digest(row):
    h=hashlib.sha256()
    for name in ('input_ids','policy_mask','target_mask'):
        tensor=row[name].contiguous()
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    h.update(str(row['prompt_length']).encode())
    return h.hexdigest()

def stats(rows):
    lengths=[r['length'] for r in rows]
    return dict(count=len(rows),length_sum=sum(lengths),length_mean=statistics.mean(lengths) if lengths else None,
                length_min=min(lengths) if lengths else None,length_max=max(lengths) if lengths else None,
                negative_multiplier=sum(r['multiplier']<0 for r in rows),
                positive_multiplier=sum(r['multiplier']>0 for r in rows))

for label,item in source['runs'].items():
    rows=[];files=[]
    for log,v in item['worker_logs'].items():
        batches=[x for x in v['first_two_round_batches'] if x['round']==1]
        if not batches: continue
        path=Path(log);lines=path.read_text(errors='replace').splitlines()
        records=[]
        # The existing passive record is emitted immediately before its native minibatch line.
        for lineno in range(batches[0]['line']-1,batches[-1]['line']):
            line=lines[lineno]
            if '[DT direct joint record] ' in line:
                record=json.loads(line.split('[DT direct joint record] ',1)[1])
                records.append((lineno+1,Path(record['path'])))
        # Include the first record, one line before the first batch marker.
        lineno=batches[0]['line']-2
        if lineno>=0 and '[DT direct joint record] ' in lines[lineno]:
            record=json.loads(lines[lineno].split('[DT direct joint record] ',1)[1])
            records.insert(0,(lineno+1,Path(record['path'])))
        assert len(records)==len(batches),(label,log,len(records),len(batches))
        for (line,p),batch in zip(records,batches):
            data=torch.load(p,map_location='cpu',weights_only=False)
            files.append(dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),log=log,line=line,
                              batch=batch['batch'],seconds=batch['seconds'],compute_length=batch['length']))
            for row in data['rows']:
                rows.append(dict(traj_uid=row['traj_uid'],input_SHA256=digest(row),length=row['input_ids'].numel(),
                                 multiplier=row['reward'],source_file=str(p)))
    unique={}
    for row in rows:
        uid=row['traj_uid']
        if uid in unique:
            assert (unique[uid]['input_SHA256'],unique[uid]['multiplier'])==(row['input_SHA256'],row['multiplier'])
        unique[uid]=row
    rows=list(unique.values())
    report['runs'][label]=dict(rows=rows,summary=stats(rows),passive_files=files,
        duplicate_content_hashes={k:v for k,v in Counter(r['input_SHA256'] for r in rows).items() if v>1},
        native_DP_padding_rows=sum(len(torch.load(Path(f['path']),map_location='cpu',weights_only=False)['rows']) for f in files)-len(rows))
old,new=(report['runs'][k]['rows'] for k in ('old','new'))
old_hashes=Counter(r['input_SHA256'] for r in old)
new_hashes=Counter(r['input_SHA256'] for r in new)
report['content_multiset_overlap']=sum((old_hashes&new_hashes).values())
report['new_negative']=stats([r for r in new if r['multiplier']<0])
report['new_positive']=stats([r for r in new if r['multiplier']>0])
report['limits']='Content hashes compare actual IDs and masks; duplicate content identities are retained, not guessed across runs. No equal-cost or exact timing-model claim.'
assert not torch.cuda.is_initialized()
dest=OUT/'first-DT-request-comparison-v50.json'
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
