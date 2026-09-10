"""Rerun only FT's final-answer seed control with full reasoning-hop support."""
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OFFICIAL = ROOT / 'experiments/official'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'environment', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(OFFICIAL))
    from evidence_protocol import recovery_curve
    from retrieval_views import sentence_density_order
    parent = json.loads((a.parent/'results.json').read_bytes())
    assert parent['status'] == 'complete' and parent['experiment'] == 'answer-pilot-v1'
    assert parent['stage'] == 'development' and parent['driver_sha256'] == digest(HERE/'evaluate_answer.py')
    assert parent['vectors_sha256'] == digest(a.parent/'vectors.npz')
    release = json.loads((OFFICIAL/'protocol.json').read_bytes())
    env = json.loads(a.environment.read_bytes())['qwen3']
    official = Path(env['official_root'])
    for name, value in release['official_normalized_sources'].items():
        assert hashlib.sha256((official/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value
    manifest = json.loads((ROOT/'deltatrace/clean/sources.json').read_bytes())
    for model in manifest['models'].values():
        for name, row in model['files'].items():
            assert digest(ROOT/name)==row['sha256']
    receipt = json.loads(Path(env['checkpoint_receipt']).read_bytes())
    assert digest(Path(env['checkpoint_receipt']))==env['checkpoint_receipt_sha256']==parent['weight_identity']['receipt_sha256']
    for row in receipt['files']:
        path=Path(env['checkpoint'])/row['name']
        assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256']
    os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    os.environ.setdefault('MACA_PATH','/opt/maca')
    sys.path.insert(0,str(official))
    for overlay in env.get('dependency_overlays',[]):sys.path.insert(0,overlay)
    import numpy as np
    import torch
    import ft_ifr_improve as ft
    from ft_target_control import InitialTargetFT
    from exp.exp2 import run_exp as author
    from exp.exp2 import dataset_utils
    assert textwrap.dedent(inspect.getsource(ft.LLMIFRAttributionBoth.calculate_ifr_multi_hop_both)).strip()==textwrap.dedent((HERE/'ft_both_method_source.py').read_text()).strip()
    torch.set_num_threads(4);torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
    a.output.mkdir(parents=True,exist_ok=False)
    report=copy.deepcopy(parent)
    report['status']='correcting_FT_initial_seed'
    report['FT_target_correction']={'version':'initial_seed_full_hops_v1','script_sha256':digest(Path(__file__)),
        'adapter_sha256':digest(HERE/'ft_target_control.py'),'addendum_sha256':digest(HERE/'ANSWER_HOPS_ADDENDUM.md'),
        'parent_results_sha256':digest(a.parent/'results.json'),'parent_vectors_sha256':parent['vectors_sha256'],
        'identity_checks':[],'corrected_cases':[]}
    with np.load(a.parent/'vectors.npz') as f:vectors={name:f[name].copy() for name in f.files}
    def save():
        temp=a.output/'results.partial'
        temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        temp.replace(a.output/'results.json')
    def timed(name,fn):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        item={'name':name,'status':'entered'};report['costs'].append(item)
        try:
            result=fn();torch.cuda.synchronize();item['status']='returned';return result
        finally:item.update(seconds=time.perf_counter()-started,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    save()
    try:
        model,tokenizer=timed('FT_seed_correction_model_load',lambda:author.load_model(env['checkpoint'],'cuda:0'))
        model.eval().requires_grad_(False);model.set_attn_implementation('eager')
        assert digest(Path(inspect.getfile(type(model))))==env['native_model_sha256']
        forwards={name:module.forward for name,module in model.named_modules()}
        examples={}
        for task in parent['selected_counts']:
            path=official/'exp/exp2/data'/(task+'.jsonl')
            assert digest(path)==release['tasks'][task]['cache_sha256']
            examples[task]=dataset_utils.load_cached(path)
        identity_done=set()
        for row in report['cases']:
            if row['target_mode']!='answer_conditioned':continue
            task,index=row['dataset'],row['index'];ex=examples[task][index]
            prefix=f'{task}_{index}_answer_conditioned_'
            engine=ft.LLMIFRAttributionBoth(model,tokenizer,show_progress=False)
            ids,_,prompt_len,gen_len=engine._ensure_generation(ex.prompt,ex.target)
            assert ids[0].cpu().tolist()==row['input_ids'] and prompt_len==row['prompt_length']
            encoded=tokenizer(' '+ex.prompt,add_special_tokens=False,return_offsets_mapping=True)
            offsets=encoded['offset_mapping'];keep=row['keep'];gold=row['gold']
            row['restricted_hop_diagnostic']={'metrics':{},'aggregations':{}}
            row['FT_correction_thinking_span']=list(ex.thinking_span)
            row['FT_hop_span_generation']=[0,gen_len-2]
            for hops in (1,3):
                def trace(mask):
                    actual_inputs=[];aggregations=[]
                    code=inspect.unwrap(ft.compute_ifr_sentence_aggregate).__code__
                    def hook(_module,_args,kwargs):
                        assert torch.equal(kwargs['input_ids'],ids);actual_inputs.append(row['input_sha256'])
                    def observe(frame,event,_arg):
                        if event=='call' and frame.f_code is code:
                            d=frame.f_locals;w=d.get('sink_weights')
                            aggregations.append({'start':int(d['sink_start'])-prompt_len,'end':int(d['sink_end'])-prompt_len,
                                'weights':w.detach().float().cpu().tolist() if w is not None else None})
                    handle=model.register_forward_pre_hook(hook,with_kwargs=True)
                    assert sys.getprofile() is None
                    sys.setprofile(observe)
                    try:
                        tracer=InitialTargetFT(model,tokenizer,chunk_tokens=128,sink_chunk_tokens=32,show_progress=False)
                        result=tracer.calculate_ifr_multi_hop_both(ex.prompt,target=ex.target,sink_span=tuple(ex.sink_span),
                            thinking_span=tuple(ex.thinking_span),n_hops=hops,initial_target_mask=mask)
                        seq,_,_=result.get_all_token_attrs(ex.indices_to_explain)
                        scores=seq[:,:len(row['user_positions'])].sum(0).detach().cpu().numpy()
                    finally:sys.setprofile(None);handle.remove()
                    assert len(actual_inputs)==1 and aggregations
                    assert all(x['start']==0 and x['end']==gen_len-2 for x in aggregations), 'Reasoning-hop support was reduced'
                    first=aggregations[0];actual=np.zeros(gen_len)
                    actual[:gen_len-1]=first['weights'] if first['weights'] is not None else 1
                    if mask==row['target_weights']:assert np.array_equal(actual,np.asarray(mask))
                    return scores,aggregations
                if task not in identity_done:
                    control,_=timed(f'{task}_{index}_FT_K{hops}_all_ones_seed_identity',lambda:trace([1.]*gen_len))
                    baseline=vectors[f'{task}_{index}_full_FT_K{hops}_prompt']
                    assert np.array_equal(control,baseline),'All-ones FT seed adapter changed the original vector'
                    report['FT_target_correction']['identity_checks'].append({'dataset':task,'index':index,'hops':hops,'bitwise_equal':True})
                scores,aggregations=timed(prefix+f'FT_K{hops}_full_hops_seed_correction',lambda:trace(row['target_weights']))
                method=f'FT_K{hops}'
                vectors[prefix+method+'_restricted_hops_prompt']=vectors[prefix+method+'_prompt'].copy()
                vectors[prefix+method+'_prompt']=scores
                row['restricted_hop_diagnostic']['metrics'][method]=row['metrics'][method]
                row['restricted_hop_diagnostic']['aggregations'][method]=row[f'{method}_actual_target_aggregation']
                row[f'{method}_actual_target_aggregation']=aggregations
                positive=np.maximum(scores,0);order=sentence_density_order(' '+ex.prompt,offsets,positive,keep)
                rank=np.zeros_like(positive);rank[order]=np.arange(len(keep),0,-1)
                row['metrics'][method]={'raw':recovery_curve(positive,keep,gold,[.05,.1,.2]),
                    'density':recovery_curve(rank,keep,gold,[.05,.1,.2])}
            identity_done.add(task)
            assert all(module.forward==forwards[name] for name,module in model.named_modules())
            report['FT_target_correction']['corrected_cases'].append([task,index])
            report['status']=f'corrected_{task}_{index}'
            np.savez_compressed(a.output/'vectors.npz',**vectors);save()
            print(report['status'],flush=True)
        assert len(report['FT_target_correction']['corrected_cases'])==sum(parent['selected_counts'].values())
        report['vectors_sha256']=digest(a.output/'vectors.npz');report['status']='complete'
    except Exception:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:save()


if __name__=='__main__':main()
