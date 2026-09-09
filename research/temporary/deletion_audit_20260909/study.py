"""Bounded uniform GDN allocation audit on unchanged author development cases.

Uses the pinned clean controller and original author scoring functions. No
generation, new data, FT rerun, model replacement, or metric implementation.
"""
import argparse,gc,hashlib,inspect,json,os,re,sys,time,traceback
from pathlib import Path

sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--environment',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.protocol.read_bytes())
    env=json.loads(args.environment.read_bytes())['qwen35']
    refs=[];ref_vectors={}
    official_protocol=json.loads((args.release/'experiments/official/protocol.json').read_bytes())
    manifest_path=args.release/'deltatrace/clean/sources.json'
    manifest=json.loads(manifest_path.read_bytes())
    for name,entry in manifest['models']['qwen35']['files'].items():
        assert sha((args.release/name).read_bytes())==entry['sha256']
    for name,digest in official_protocol['official_normalized_sources'].items():
        assert sha((Path(env['official_root'])/name).read_bytes().replace(b'\r\n',b'\n'))==digest
    for name,digest in plan['study_sources'].items():assert sha((Path(__file__).parent/name).read_bytes())==digest
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    os.environ.setdefault('TRITON_CACHE_DIR','/tmp/deltatrace_clean_v1_triton')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/deltatrace_clean_v1_inductor')
    sys.path.insert(0,env['official_root'])
    for overlay in env.get('dependency_overlays',[]):sys.path.insert(0,overlay)
    sys.path.insert(0,str(args.release/'deltatrace/clean/qwen35'))
    import numpy as np
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner
    from qwen35_answer_finite import PackedAnswerTargets
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    from uniform_gdn import configure_uniform_gdn
    from exp.exp2 import dataset_utils
    import ft_ifr_improve as ft
    import llm_attr_eval
    for module in (dataset_utils,ft,llm_attr_eval):
        assert Path(module.__file__).resolve().is_relative_to(Path(env['official_root']).resolve())
    for entry in plan['references']:
        source=Path(entry['path']);assert sha(source.read_bytes())==entry['sha256']
        data=json.loads(source.read_bytes());assert data['clean_sources_sha256']==sha(manifest_path.read_bytes())
        vp=source.parent/'vectors.npz';assert sha(vp.read_bytes())==entry['vectors_sha256']
        ref_vectors.update(dict(np.load(vp,allow_pickle=False)))
        refs.extend(c for c in data['cases'] if c['status']=='complete')
    selected=[next(c for c in refs if [c['dataset'],c['index']]==key) for key in plan['cases']]
    torch.set_num_threads(4);torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.cache_size_limit=max(torch._dynamo.config.cache_size_limit,128)
    torch._dynamo.config.accumulated_cache_size_limit=max(torch._dynamo.config.accumulated_cache_size_limit,512)
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'loading','scope':'temporary causal allocation audit; original development data',
        'protocol_sha256':sha(args.protocol.read_bytes()),'plan':plan,'sample_batch':1,'endpoint_batch':2,
        'clean_sources_sha256':sha(manifest_path.read_bytes()),'calls':[],'cases':[],
        'FT_reruns':0,'generation_calls':0,'native_forwards_unchanged':True}
    vectors={}
    def save():
        tmp=args.output/'results.partial';tmp.write_text(json.dumps(report,indent=2,allow_nan=False))
        tmp.replace(args.output/'results.json')
    def timed(name,fn):
        report['status']=name;save();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        row={'name':name,'status':'entered'};report['calls'].append(row);start=time.perf_counter()
        try:
            result=fn();torch.cuda.synchronize();row['status']='returned';return result
        finally:
            row.update(seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved());save()
    save()
    try:
        tokenizer=AutoTokenizer.from_pretrained(env['checkpoint'],local_files_only=True)
        tokenizer.pad_token=tokenizer.eos_token
        model=timed('model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'],
            dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True))
        model.eval().requires_grad_(False)
        assert sha(Path(inspect.getfile(type(model))).read_bytes())==env['native_model_sha256']
        original={n:(type(m).forward,m.forward) for n,m in model.named_modules()}
        def identity():
            for n,m in model.named_modules():assert (type(m).forward,m.forward)==original[n]
        verify_native_sources(env['native_stage_source_sha256'])
        finite_fa=VendorFAFiniteP1BF16D256(env['finite_library'],env['finite_library_sha256'])
        finite_fla=make_compiled_finite_pullback(reuse_scalar_products=False)
        modes=['clean']+plan['candidates']
        runners={m:configure_uniform_gdn(Qwen35DenseFiniteRunner(model,finite_fa,finite_fla),m) for m in modes}
        evaluator=llm_attr_eval.LLMAttributionEvaluator(model,tokenizer)
        cache={}
        for task in {c['dataset'] for c in selected}:
            path=Path(env['official_root'])/'exp/exp2/data'/(task+'.jsonl')
            assert sha(path.read_bytes())==official_protocol['tasks'][task]['cache_sha256']
            cache[task]=dataset_utils.load_cached(path)
        for case_number,c in enumerate(selected):
            key=f"{c['dataset']}_{c['index']}";ex=cache[c['dataset']][c['index']]
            engine=ft.LLMIFRAttributionBoth(model,tokenizer,show_progress=False)
            ids,mask,prompt_len,gen_len=engine._ensure_generation(ex.prompt,ex.target)
            positions=list(engine.user_prompt_indices);keep=ft.keep_token_indices(engine.user_prompt_tokens)
            assert sha(ids.cpu().numpy().tobytes())==c['input_sha256']
            assert positions==c['user_positions'] and keep==c['keep'] and prompt_len==c['prompt_length']
            target=tokenizer(ex.target+tokenizer.eos_token,add_special_tokens=False,return_tensors='pt').input_ids.to('cuda')
            formatted=evaluator.format_prompt(' '+ex.prompt)
            assert torch.equal(ids,torch.cat((tokenizer(formatted,add_special_tokens=False,return_tensors='pt').input_ids.to('cuda'),target),1))
            assert gen_len==target.shape[1]
            eligible=[positions[i] for i in keep]
            row={'dataset':c['dataset'],'index':c['index'],'input_sha256':c['input_sha256'],
                 'methods':{},'metrics':{},'fixed_set_comparisons':{},'gold':c['gold'],'keep':keep,'user_positions':positions}
            row['gold_digit_indices']=[i for i in c['gold'] if re.fullmatch(r'[0-9]+',tokenizer.decode([int(ids[0,positions[i]])]).strip())]
            row['all_digit_indices']=[i for i in keep if re.fullmatch(r'[0-9]+',tokenizer.decode([int(ids[0,positions[i]])]).strip())]
            report['cases'].append(row);save()
            if case_number==0:
                with torch.no_grad():init=timed('native_eager_initialization',lambda:model(input_ids=ids,attention_mask=mask,use_cache=False))
                del init
            model.set_attn_implementation('flash_attention_2')
            base=ids.clone();base[0,eligible]=tokenizer.eos_token_id;pair=torch.cat((base,ids))
            selection=PackedAnswerTargets([{'target_ids':target[0].cpu(),'prompt_length':prompt_len}],
                [list(range(gen_len))],ids.shape[1],'cuda')
            for mode in modes:
                root_calls=[]
                def observe_root(_module,_args,kwargs):
                    assert torch.equal(kwargs['input_ids'],pair)
                    assert torch.equal(kwargs['attention_mask'],torch.ones_like(pair))
                    root_calls.append(sha(pair.cpu().numpy().tobytes()))
                handle=model.register_forward_pre_hook(observe_root,with_kwargs=True)
                try:signed,detail=timed(key+'_'+mode+'_DT',lambda:runners[mode].attribute(pair,torch.ones_like(pair),selection))
                finally:handle.remove()
                identity();assert len(root_calls)==1 and bool(torch.isfinite(signed).all())
                signed=signed[0];positive=signed[positions].float().clamp_min(0)
                vkey=key+'_'+mode;vectors[vkey+'_signed_full']=signed.numpy();vectors[vkey+'_positive']=positive.numpy()
                row['methods'][mode]={'detail':detail,'root_calls':root_calls}
                if mode=='clean':
                    ref=ref_vectors[key+'_DT_signed_full']
                    row['clean_vs_frozen_relative_L2']=float(np.linalg.norm(signed.numpy()-ref)/max(np.linalg.norm(ref),1e-30))
                fixed=[]
                for step in (1,2,3,4):
                    item={'step':step,'sets':{}}
                    for name in ('DT','FT_K1'):
                        curve=c['metrics'][name];deleted=curve['deleted_user_indices'][step]
                        item['sets'][name]={'count':len(deleted),'predicted_signed':float(signed[[positions[j] for j in deleted]].sum()),
                            'predicted_positive':float(positive[deleted].sum()),'actual_logprob_drop':curve['scores'][0]-curve['scores'][step],
                            'gold_digit_count':len(set(deleted)&set(row['gold_digit_indices']))}
                    fixed.append(item)
                row['fixed_set_comparisons'][mode]=fixed
                np.savez_compressed(args.output/'vectors.npz',**vectors);save()
                del detail,signed,positive
            model.set_attn_implementation('eager')
            for mode in plan['candidates']:
                scores=torch.tensor(vectors[key+'_'+mode+'_positive'])
                curve={'actual_input_hashes':[],'deleted_user_indices':[]}
                def observe_input(_module,_args,kwargs):
                    actual=kwargs['input_ids'].detach().cpu();original_ids=ids.cpu()
                    changed=(actual[0]!=original_ids[0]).nonzero().flatten().tolist()
                    assert set(changed)<=set(eligible)
                    assert all(int(actual[0,j])==tokenizer.eos_token_id for j in changed)
                    assert torch.equal(actual[:,prompt_len:],target.cpu())
                    curve['actual_input_hashes'].append(sha(actual.numpy().tobytes()))
                    curve['deleted_user_indices'].append([positions.index(j) for j in changed])
                def observe_metric(frame,event,value):
                    if frame.f_code is ft.faithfulness_test_skip_tokens.__code__ and event=='return' and value is not None:
                        for name in ('scores','density','normalized_model_response','alignment_penalty','corrected_scores'):
                            curve[name]=np.asarray(frame.f_locals[name]).tolist()
                handle=model.register_forward_pre_hook(observe_input,with_kwargs=True)
                assert sys.getprofile() is None;sys.setprofile(observe_metric)
                try:
                    with torch.no_grad():metrics=timed(key+'_'+mode+'_original_metrics',lambda:ft.faithfulness_test_skip_tokens(
                        evaluator,scores[None],ex.prompt,ex.target,keep_prompt_token_indices=keep,user_prompt_indices=positions,k=20))
                finally:sys.setprofile(None);handle.remove()
                identity();assert len(curve['actual_input_hashes'])==21
                assert curve['actual_input_hashes'][0]==c['input_sha256']
                assert curve['actual_input_hashes'][-1]==c['metrics']['DT']['actual_input_hashes'][-1]
                curve['rise'],curve['mas'],curve['rise_plus_ap']=map(float,metrics)
                curve['needle']=float(ft.evaluate_attr_recovery_skip_tokens(scores[None],keep_prompt_token_indices=keep,
                    gold_prompt_token_indices=c['gold'],top_fraction=.1)) if c['gold'] else None
                row['metrics'][mode]=curve;save()
            row['status']='complete';save()
            print(json.dumps({'case':key,'metrics':{m:{k:v[k] for k in ('rise','mas','needle')} for m,v in row['metrics'].items()}}),flush=True)
            del engine,ids,mask,base,pair,selection,target;gc.collect()
        report['status']='complete';report['vectors_sha256']=sha((args.output/'vectors.npz').read_bytes())
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:save()

if __name__=='__main__':main()
