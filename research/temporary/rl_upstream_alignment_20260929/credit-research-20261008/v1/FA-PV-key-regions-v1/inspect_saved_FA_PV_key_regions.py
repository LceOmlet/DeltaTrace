"""Locate the measured PV-background residual using original public FA.

Read original frozen operator tensors only. Partition values by key position,
while leaving query/key operands and the owner's causal mask unchanged. The
partition sum is linear in exact arithmetic; its actual BF16 closure residual
is recorded, never corrected. This does not compute replacement token credit.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import psutil
import torch
from flash_attn import flash_attn_func


def ref(path):
    path=Path(path);digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def contract(output,row,start,length,upstream):
    # Original passive_attention_pv.finish_point uses these same coordinates.
    result=0.
    for first in range(0,length,128):
        last=min(first+128,length)
        delta=output[2*row+1,start+first:start+last].double()-output[2*row,start+first:start+last].double()
        result+=float((delta*upstream[row,first:last].double()).sum())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    begin=time.perf_counter();proc=psutil.Process()
    ranks=[args.directory/'results'/f'rank{r}.json' for r in (0,1)]
    reports=[json.loads(p.read_bytes()) for p in ranks]
    points={(p['traj_uid'],p['packed_slot']):p for r in reports for b in r['batches'] for p in b['points']}
    assert len(points)==165 and all(r['phase']=='complete' for r in reports)
    original_files={a['path']:a for r in reports for b in r['batches']
                    for a in b['attention_PV_readout']['original_operand_artifacts']}
    public=ref(inspect.getsourcefile(flash_attn_func))
    expected={p['final_FA_PV_ledger']['actual_owner']['sha256'] for p in points.values()}
    assert expected=={public['sha256']}
    result=dict(scope=__doc__,pid=os.getpid(),birth=proc.create_time(),unix=time.time(),phase='initialized',
        sources=[ref(__file__),*[ref(p) for p in ranks]],original_FA_owner=public,
        operations=dict(model=0,DT=0,optimizer=0,rollout=0,checkpoint_restore=0,original_public_FA=0),
        production_modified=False,candidate=False,points=[],files=[],sampled_PSS_peak_bytes=0,
        GPU_peak_allocated_bytes=0,GPU_peak_reserved_bytes=0,elapsed_seconds=0.,
        numerical_scope='Arithmetic localization of saved BF16 operands, not an official tolerance test, causal share, or accepted repair.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    phases=args.output.with_suffix('.phases.jsonl').open('x',buffering=1)

    def save(phase,**extra):
        result.update(phase=phase,elapsed_seconds=time.perf_counter()-begin,**extra)
        result['sampled_PSS_peak_bytes']=max(result['sampled_PSS_peak_bytes'],proc.memory_full_info().pss)
        if torch.cuda.is_initialized():
            result['GPU_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            result['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        phases.write(json.dumps(dict(phase=phase,elapsed_seconds=result['elapsed_seconds'],
            points=len(result['points']),FA_calls=result['operations']['original_public_FA'],**extra))+'\n')
        args.output.write_text(json.dumps(result,indent=2)+'\n')

    save('begin')
    try:
        visited=set()
        for report in reports:
            for batch in report['batches']:
                artifacts=batch['attention_PV_readout']['original_operand_artifacts']
                jointref=next(a for a in artifacts if '-joint.pt' in a['path'])
                actual=ref(jointref['path']);assert actual==jointref
                joint=torch.load(jointref['path'],map_location='cpu',mmap=True,weights_only=False)
                result['files'].append(actual)
                for native_ref in [a for a in artifacts if '-native-' in a['path']]:
                    save('read_original_native_operands',artifact=native_ref['path'])
                    actual=ref(native_ref['path']);assert actual==native_ref
                    native=torch.load(native_ref['path'],map_location='cpu',mmap=True,weights_only=False)
                    owner=native['owner'];assert owner['sha256']==public['sha256']
                    arguments=owner['arguments']
                    assert arguments==dict(dropout_p=0.,softmax_scale=.0625,causal=True,return_attn_probs=False)
                    q,k=(native['endpoints'][n].transpose(1,2).to('cuda') for n in ('query','key'))
                    width=q.shape[1]
                    VR=joint['bank']['value'][0::2,:,:width].transpose(1,2).repeat_interleave(2,0).to('cuda')
                    VD=native['endpoints']['value'][0::2].transpose(1,2).repeat_interleave(2,0).to('cuda')
                    assert q.dtype==k.dtype==VR.dtype==VD.dtype==torch.bfloat16
                    original_shape=owner['query_shape'];assert list(q.shape)==original_shape
                    time_positions=torch.arange(width,device='cuda')
                    positions=torch.tensor([x['packed_slot'] if x is not None else width
                        for x in native['queries']],device='cuda').repeat_interleave(2)
                    by_region={}
                    for name in ('all','before','at','after'):
                        if name=='all': mask=torch.ones((q.shape[0],width),device='cuda',dtype=torch.bool)
                        elif name=='before': mask=time_positions[None,:]<positions[:,None]
                        elif name=='at': mask=time_positions[None,:]==positions[:,None]
                        else: mask=time_positions[None,:]>positions[:,None]
                        terms=[]
                        for value in (VR,VD):
                            masked=value.masked_fill(~mask[:,:,None,None],0)
                            output=flash_attn_func(q,k,masked,**arguments)
                            result['operations']['original_public_FA']+=1
                            terms.append(output.cpu())
                            del masked,output
                        by_region[name]={}
                        for row,query in enumerate(native['queries']):
                            if query is None:continue
                            start=native['starts'][row];length=native['actual_context_lengths'][row]-start
                            by_region[name][row]=contract(terms[0],row,start,length,joint['bank']['mc'])-contract(
                                terms[1],row,start,length,joint['bank']['mc'])
                        del terms,mask
                    for row,query in enumerate(native['queries']):
                        if query is None:continue
                        key=(native['trajectories'][row],query['packed_slot'])
                        assert key in points and key not in visited;visited.add(key)
                        p=points[key]
                        ledger={name:by_region[name][row] for name in by_region}
                        result['points'].append(dict(traj_uid=key[0],packed_slot=key[1],token_id=query['token_id'],
                            initial_state_sha256=p['initial_state_sha256'],previously_examined=p['previously_examined'],
                            native_file=actual,PV_key_regions=ledger,
                            partition_closure_residual=sum(ledger[n] for n in ('before','at','after'))-ledger['all'],
                            prior_recorded_PV_background=p['final_FA_PV_ledger']['PV_background_residual'],
                            original_PV_replay_difference=ledger['all']-p['final_FA_PV_ledger']['PV_background_residual'],
                            interpretation='Original upstream contraction of key-partitioned values. Rounded partition residual is retained, not a corrected credit.'))
                    result['files'].append(actual)
                    del native,q,k,VR,VD,time_positions,positions,by_region
                    gc.collect();save('original_native_regions_complete',artifact=actual['path'])
                del joint
        assert visited==set(points)
        torch.cuda.synchronize();save('complete')
    except BaseException:
        import traceback
        save('failed',traceback=traceback.format_exc());raise
    finally:
        phases.close()
    print(json.dumps(dict(output=ref(args.output),points=len(result['points']),operations=result['operations'],
                         elapsed_seconds=result['elapsed_seconds'],PSS=result['sampled_PSS_peak_bytes'],
                         GPU_allocated=result['GPU_peak_allocated_bytes'],GPU_reserved=result['GPU_peak_reserved_bytes'])))


if __name__=='__main__':
    main()
