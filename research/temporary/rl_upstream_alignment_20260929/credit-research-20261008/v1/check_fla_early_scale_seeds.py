"""CPU range census of both saved pre-cast tensors; zero model/DT work.

This applies the proposed scalar to saved coefficients. It does not claim to
reproduce the separately compiled RMS arithmetic or to validate DT quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import psutil
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    scale = 128**-.5
    result = dict(scope=__doc__,scale=scale,ranks=[],operations=dict(model=0,DT=0,GPU=0,optimizer=0))
    started = time.perf_counter()
    for rank in (0,1):
        path = args.directory/f'rank{rank}-precast-seed.pt'
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream,'sha256').hexdigest()
        data = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        mo = data['mo']
        assert mo.dtype==torch.float32 and mo.is_contiguous()
        flat = mo.view(-1)
        r = dict(rank=rank,path=str(path),sha256=digest,elements=flat.numel(),
            maxabs=0.,scaled_maxabs=0.,old_FP16_nonfinite=0,new_FP16_nonfinite=0,
            newly_zero=0,scaled_squared_norm=0.,zeroed_squared_norm=0.)
        for start in range(0,len(flat),1048576):
            part = flat[start:start+1048576]
            assert torch.isfinite(part).all()
            scaled = part*scale
            old = part.to(data['output_dtype']).to(torch.float16)
            new = scaled.to(data['output_dtype']).to(torch.float16)
            zeros = (scaled!=0)&(new==0)
            r['maxabs']=max(r['maxabs'],float(part.abs().max()))
            r['scaled_maxabs']=max(r['scaled_maxabs'],float(scaled.abs().max()))
            r['old_FP16_nonfinite']+=int((~torch.isfinite(old)).sum())
            r['new_FP16_nonfinite']+=int((~torch.isfinite(new)).sum())
            r['newly_zero']+=int(zeros.sum())
            r['scaled_squared_norm']+=float(scaled.double().square().sum())
            r['zeroed_squared_norm']+=float(scaled[zeros].double().square().sum())
        r['zeroed_L2_fraction']=(r['zeroed_squared_norm']/r['scaled_squared_norm'])**.5
        r['PSS_bytes']=psutil.Process().memory_full_info().pss
        result['ranks'].append(r)
        del data,mo,flat
    result.update(seconds=time.perf_counter()-started,cuda_initialized=torch.cuda.is_initialized())
    assert not result['cuda_initialized']
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
