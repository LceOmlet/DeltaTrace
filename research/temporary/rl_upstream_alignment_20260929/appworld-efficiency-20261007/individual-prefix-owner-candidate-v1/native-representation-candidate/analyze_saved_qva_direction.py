"""CPU raw-A coefficient geometry and actual HF Cache dtype source only."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time


def identity(path):
    path=Path(path)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('MACA_VISIBLE_DEVICES')==''
    import torch,psutil
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    torch.set_num_threads(1)
    started=time.time()
    labels=('shared_cold','shared_warm','shared_individual_rows_cold','shared_individual_rows_warm','shared_individual_rows_observed')
    pairs=(('shared_cold','shared_warm'),('shared_individual_rows_cold','shared_individual_rows_warm'),
           ('shared_individual_rows_warm','shared_individual_rows_observed'),('shared_warm','shared_individual_rows_warm'))
    def describe(tensor):
        x=tensor.double()
        return dict(count=x.numel(),dtype=str(tensor.dtype),finite=int(torch.isfinite(x).sum()),
            min=float(x.min()),max=float(x.max()),mean=float(x.mean()),
            population_std=float(x.std(unbiased=False)),sample_std=float(x.std(unbiased=True)),
            L1=float(x.abs().sum()),L2=float(x.square().sum().sqrt()),
            positive=int((x>0).sum()),negative=int((x<0).sum()),zero=int((x==0).sum()))
    def compare(a,b):
        left,right=a.double(),b.double()
        dot=float((left*right).sum());ln=float(left.square().sum().sqrt());rn=float(right.square().sum().sqrt())
        return dict(dot=dot,left_L2=ln,right_L2=rn,right_over_left_L2=rn/ln if ln else None,
            cosine=dot/(ln*rn) if ln and rn else None,
            difference_L2=float((right-left).square().sum().sqrt()))
    prepared=json.loads((args.source/'prepared.json').read_bytes())
    ranks=[];pooled={key:[] for key in labels}
    for rank in range(2):
        rankfile=args.source/f'rank{rank}.json';record=json.loads(rankfile.read_bytes())
        vpath=Path(record['vectors']['path']);assert identity(vpath)['sha256']==record['vectors']['sha256']
        vector=torch.load(vpath,map_location='cpu',weights_only=False)
        path=args.source/f'actual-requests-rank{rank}.pt'
        assert identity(path)['sha256']==prepared['literal_request_files'][str(path)]
        payload=torch.load(path,map_location='cpu',weights_only=False)
        requests=sorted(payload['requests'],key=lambda r:r['context_tokens'])
        offset=prepared['selected_observation']['offset'];chosen=requests[offset:offset+4]
        masks=[]
        for request in chosen:
            row=payload['rows'][request['row_index']];width=row['responses'].numel()
            mask=row['attention_mask'].bool()[-width:]
            assert bool(row['active_masks']) and int(mask.sum())==request['actions'].numel()
            assert mask.nonzero().flatten().tolist()==list(range(request['actions'].numel()))
            assert torch.equal(row['responses'][:int(mask.sum())],request['actions'])
            masks.append(mask)
        mask=torch.stack(masks);values={key:vector[key]['dt_token_advantages'][mask] for key in labels}
        for key in labels:pooled[key].append(values[key])
        ranks.append(dict(rank=rank,rank_file=identity(rankfile),vectors=identity(vpath),requests=identity(path),
            action_tokens=int(mask.sum()),variants={key:describe(values[key]) for key in labels},
            comparisons=[dict(left=a,right=b,**compare(values[a],values[b])) for a,b in pairs]))
    values={key:torch.cat(value) for key,value in pooled.items()}
    cache=DynamicCache(config=Qwen3_5TextConfig(num_hidden_layers=2,layer_types=['linear_attention','full_attention']))
    sources=[]
    for target,name in ((cache,'update_conv_state'),(cache,'update_recurrent_state'),
                        (cache.layers[0],'lazy_initialization'),(cache.layers[0],'update_conv_state'),
                        (cache.layers[0],'update_recurrent_state')):
        fn=getattr(target,name);lines,line=inspect.getsourcelines(fn)
        sources.append(dict(owner_class=type(target).__name__,method=name,source_file=identity(inspect.getsourcefile(fn)),
            first_line=line,signature=str(inspect.signature(fn)),source=''.join(lines)))
    # Small original Cache calls expose the documented dtype transition only;
    # no captured state is changed, no model/FLA is run and no new rule is used.
    cache.update_conv_state(torch.zeros(4,6,4,dtype=torch.bfloat16),0)
    state=torch.ones(4,2,8,8,dtype=torch.float32)
    cache.update_recurrent_state(state,0)
    dtype_observation=dict(conv_input_dtype='torch.bfloat16',recurrent_input_dtype=str(state.dtype),
        stored_conv_dtype=str(cache.layers[0].conv_states.dtype),
        stored_recurrent_dtype=str(cache.layers[0].recurrent_states.dtype),
        scope='Small actual HF Cache dtype API observation, not a saved GDN-state numerical reference.')
    assert not torch.cuda.is_initialized()
    result=dict(scope='Actual saved raw-A coefficient vectors on original response action masks; not parameter-gradient geometry.',
        script=identity(Path(__file__)),prepared=identity(args.source/'prepared.json'),ranks=ranks,
        pooled_actual_B8=dict(action_tokens=sum(r['action_tokens'] for r in ranks),
            reduction='Concatenate original masked raw-A coefficients across both ranks; this is not a model gradient reduction.',
            variants={key:describe(values[key]) for key in labels},
            comparisons=[dict(left=a,right=b,**compare(values[a],values[b])) for a,b in pairs]),
        actual_cache_owner_sources=sources,small_actual_cache_dtype_observation=dtype_observation,
        execution=dict(pid=os.getpid(),elapsed_seconds=time.time()-started,PSS_bytes=psutil.Process().memory_full_info().pss,
            torch_version=torch.__version__,CUDA_initialized=False,CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES=''),
        limitations=['No whole-network tolerance, clipping, normalization, numerical correction, performance promise or production change.',
            'Baseline-versus-row changes capture grouping, bank states, per-row cut and native representation together; endpoint differences cannot be assigned to one kernel.',
            'HF Cache has an actual dtype conversion contract; this alone does not prove that it explains the observed endpoint or credit differences.',
            'Raw-A direction is not PG direction, parameter-update direction or a counterfactual-quality criterion.',
            'No model, FA/FLA, DT, optimizer, checkpoint or GPU operation.'])
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),sha256=identity(args.output)['sha256'],ranks=ranks,
        pooled_actual_B8=result['pooled_actual_B8'],dtype_observation=dtype_observation)))


if __name__=='__main__':
    main()
