"""Native-chunk tiling against the same saved official derivative quantities.

Also report, without inventing a tolerance, changes in the already saved
nonzero finite coefficients. No new native/model forwards or sample selection.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import fla.utils
from finite_fla_gpu import native_input_adjoints, mixed_coefficients
from tiled_conditional_memory import conditional_memory_tiles


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('original','limit','finite','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    original, limit, finite = [json.loads((p/'result.json').read_bytes())
                               for p in (args.original,args.limit,args.finite)]
    assert not fla.utils.FLA_CI_ENV
    torch.set_num_threads(8)
    result=dict(scope=__doc__,cases=[],model_calls=0,DT_calls=0,native_forward=0,
        optimizer=0,production_modified=False,whole_DT_repair_accepted=False,
        official_test=limit['original_test'],
        sources={n:dict(path=importlib.import_module(n).__file__,sha256=sha(importlib.import_module(n).__file__))
                 for n in ('native_conditional_queries','conditional_window_memory','tiled_conditional_memory')},
        script=dict(path=__file__,sha256=sha(__file__)))
    assert sha(limit['original_test']['path'])==limit['original_test']['sha256']
    for index, old in enumerate(original['cases']):
        tick=time.perf_counter()
        for item in (old['exact_artifact'],limit['cases'][index]['artifact'],finite['cases'][index]['artifact']):
            assert sha(item['path'])==item['sha256']
        saved=torch.load(old['exact_artifact']['path'],map_location='cpu',mmap=True,weights_only=False)
        wanted=torch.load(limit['cases'][index]['artifact']['path'],map_location='cpu',mmap=True,weights_only=False)
        nonzero=torch.load(finite['cases'][index]['artifact']['path'],map_location='cpu',mmap=True,weights_only=False)
        f={key:value.cuda().contiguous() for key,value in saved['capture'].items()}
        do,scale=saved['do'].cuda(),saved['scale']
        B,T,H,K=f['q'].shape
        assert T==199
        def unpack(v):return v.reshape(B,H,((T+63)//64)*64,K)[:,:,:T].permute(0,2,1,3).contiguous()
        def ahead(v,lag):return F.pad(v[:,lag:],(0,0,0,0,0,lag))
        with torch.no_grad():
            paired={key:value.repeat_interleave(2,0) for key,value in f.items()}
            adjoints=native_input_adjoints(paired,do,scale)
            base,detail=mixed_coefficients(paired,adjoints,scale,diagnostics=True)
            L,r0=unpack(detail['L']),unpack(detail['r0'])
            case=dict(dtype=old['dtype'],B=B,T=T,tile_size=64,derivative_checks=[],nonzero_finite_changes={})
            for mode in ('derivative','nonzero_finite'):
                c=f if mode=='derivative' else {key:value.cuda().to(f[key].dtype) for key,value in nonzero['conditional'].items()}
                values=lambda start,stop:{key:torch.stack([ahead(c[key],lag)[:,start:stop] for lag in range(4)])
                                         for key in ('q','k','v')}
                parts={};boundaries=[]
                for start,stop,coefficients,calls in conditional_memory_tiles(f,base,L,r0,adjoints,values,
                      c['raw_g'].float().exp(),c['beta'],scale,tile_size=64):
                    boundaries.append(dict(start=start,stop=stop,readout_calls=calls))
                    for name,value in coefficients.items():parts.setdefault(name,[]).append(value.cpu())
                actual={name:torch.cat(values,dim=2 if name in ('q','k','v') else 1).cuda()
                        for name,values in parts.items()}
                if mode=='derivative':
                    # Use the original FP32 derivatives already saved by the
                    # prior official-limit test, rather than rerunning it.
                    expected=dict(saved['reference'],dg=wanted['original_FP32_dg'])
                    for name,key in (('q','dq'),('k','dk'),('v','dv')):
                        for lag in range(4):
                            target=ahead(expected[key].cuda(),lag)
                            row=dict(quantity=key,lag=lag,threshold=limit['original_test']['thresholds'][key])
                            row.update(normalized_error=float(fla.utils.get_err_ratio(target,actual[name][lag])))
                            fla.utils.assert_close(key,target,actual[name][lag],row['threshold']);row['passed']=True
                            case['derivative_checks'].append(row)
                    for key,value in (('db',actual['beta']),('dg',actual['alpha']*f['raw_g'].float().exp())):
                        target=expected[key].cuda()
                        threshold=limit['original_test']['thresholds'][key]
                        fla.utils.assert_close(key,target,value,threshold)
                        case['derivative_checks'].append(dict(quantity=key,threshold=threshold,
                            normalized_error=float(fla.utils.get_err_ratio(target,value)),passed=True))
                else:
                    for name,value in actual.items():
                        target=nonzero['coefficients'][name].cuda()
                        case['nonzero_finite_changes'][name]=dict(normalized_RMS=float(fla.utils.get_err_ratio(target,value)),
                            max_abs=float((target-value).abs().max()),nonfinite=int((~torch.isfinite(value)).sum()),
                            official_finite_tolerance=None)
                case[mode+'_tiles']=boundaries
                del c,actual,parts
        torch.cuda.synchronize()
        case['seconds']=time.perf_counter()-tick
        result['cases'].append(case)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(case),flush=True)
    result.update(phase='complete',peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
