"""Compare actual native H/W/IDs/output from two already-computed forwards.

CPU-only, complete captured tensors, no model/DT/optimizer call or tolerance.
This compares diagnostic executions; it does not reconstruct the lost
original persistent runtime or declare the cause of incident37.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import torch


def summarize(a, b):
    result=dict(shape_before=list(a.shape),shape_after=list(b.shape),
                dtype_before=str(a.dtype),dtype_after=str(b.dtype))
    if a.shape!=b.shape:
        result['comparable']=False
        return result
    result.update(comparable=True,exact=torch.equal(a,b))
    if result['exact']:
        result.update(changed_elements=0,max_abs_difference=0.0)
        return result
    changed=0
    squared_difference=0.0
    squared_before=0.0
    maximum=0.0
    nonfinite=0
    aa=a.reshape(-1);bb=b.reshape(-1)
    for start in range(0,aa.numel(),1048576):
        x=aa[start:start+1048576];y=bb[start:start+1048576]
        changed+=int((x!=y).sum())
        if x.is_floating_point():
            x=x.float();y=y.float();difference=y-x
            nonfinite+=int((~torch.isfinite(difference)).sum())
            maximum=max(maximum,float(difference.abs().max()))
            squared_difference+=float(difference.double().square().sum())
            squared_before+=float(x.double().square().sum())
    result.update(changed_elements=changed,max_abs_difference=maximum,
        squared_difference=squared_difference,squared_before=squared_before,
        nonfinite_difference=nonfinite)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=dict(unix=time.time(),scope=__doc__,records=[],model_calls=0,DT_calls=0,optimizer_steps=0)
    for rank in [0,1]:
        paths=[folder/f'rank{rank}-native-head-live-microbatch1.pt' for folder in [args.before,args.after]]
        captures=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
        row=dict(rank=rank,files=[dict(path=str(path),bytes=path.stat().st_size,
             sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths],
             metadata=[x['metadata'] for x in captures],tensors={})
        for key in ['input_ids','vocab_weights','hidden_states','token_log_probs','entropy']:
            row['tensors'][key]=summarize(captures[0]['tensors'][key],captures[1]['tensors'][key])
        result['records'].append(row)
        del captures
    result.update(CUDA_initialized=torch.cuda.is_initialized(),
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=True)+'\n')
    print(json.dumps(result,indent=2,allow_nan=True))


if __name__=='__main__':
    main()
