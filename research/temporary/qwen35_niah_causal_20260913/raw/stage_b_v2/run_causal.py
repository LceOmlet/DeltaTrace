"""Native paired interventions on preserved NIAH cases; unchanged DT operators."""
import argparse
import copy
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import numpy as np

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ids_digest(x):return hashlib.sha256(np.asarray(x,dtype=np.int64).tobytes()).hexdigest()
def save(p,obj):
    tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def auc(x):return float((np.sum(x)-x[0]/2-x[-1]/2)/(len(x)-1))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-cases',type=int)
    a=p.parse_args();plan=json.loads(a.protocol.read_bytes());root=a.run_root
    code=root/'repo_dynamic';full=root/'full_dynamic_with_ifr'
    env_path=root/'environment_dynamic.json';env=json.loads(env_path.read_bytes())['qwen35']
    identity=json.loads((full/'identity.json').read_bytes())
    assert digest(env_path)==identity['environment_sha256']
    assert digest(code/'deltatrace/clean/sources.json')==identity['clean_sources_sha256']
    for name,spec in json.loads((code/'deltatrace/clean/sources.json').read_bytes())['models']['qwen35']['files'].items():
        assert digest(code/name)==spec['sha256'],name
    for name,expected in env['official_extension_blob_sha1'].items():
        b=(Path(env['ft_extension_root'])/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==expected,name
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS='0',
        TRITON_CACHE_DIR=env['triton_cache'],TORCHINDUCTOR_CACHE_DIR=env['inductor_cache'])
    os.environ.update(env['runtime_environment'])
    os.environ['PATH']=str(root/'env/bin')+':/opt/conda/bin:/usr/bin:/bin:'+os.environ.get('PATH','')
    sys.path.insert(0,env['official_root']);sys.path.insert(0,str(code/'deltatrace/clean/qwen35'))
    sys.path.append(env['ft_extension_root'])
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from exp.exp2 import dataset_utils
    from llm_attr_eval import LLMAttributionEvaluator
    import ft_ifr_improve as ft
    from flashtrace import FlashTrace
    from flashtrace.improved import is_stop_token as ft_stop
    from qwen35_clean_runner import make_qwen35_clean_runner
    from qwen35_answer_finite import PackedAnswerTargets
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    torch.set_num_threads(4);torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
    reuse=json.loads((full/'process_reuse.json').read_bytes())
    for name in ('cache_size_limit','accumulated_cache_size_limit'):setattr(torch._dynamo.config,name,reuse[name])
    verify_native_sources(env['native_stage_source_sha256'])
    gpu=subprocess.run(['/usr/bin/mx-smi'],capture_output=True,text=True,check=True).stdout
    owners=re.findall(r'^\|\s+0\s+(\d+)\s+\S+',gpu,re.MULTILINE)
    assert not owners,f'GPU already in use: {owners}'
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'protocol.json').write_bytes(a.protocol.read_bytes())
    (a.output/'run_causal.py').write_bytes(Path(__file__).read_bytes())
    report=dict(status='loading',generation_calls=0,driver_sha256=digest(__file__),protocol_sha256=digest(a.protocol),
        environment_sha256=digest(env_path),clean_sources_sha256=identity['clean_sources_sha256'],
        pid=os.getpid(),initial_gpu=gpu,cases=[],model_calls=0,
        model_call_count_scope='Native scoring calls only; attribution and initialization forwards are separate.',
        scoring_source_sha256=digest(inspect.getfile(LLMAttributionEvaluator)),started=time.time())
    def state(label):report['status']=label;save(a.output/'status.json',report)
    state('loading')
    try:
        tokenizer=AutoTokenizer.from_pretrained(env['checkpoint'],local_files_only=True)
        tokenizer.pad_token=tokenizer.eos_token
        model=Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'],dtype=torch.bfloat16,
            attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True)
        model.eval().requires_grad_(False)
        assert digest(inspect.getfile(type(model)))==env['native_model_sha256']
        originals={n:(type(m).forward,m.forward) for n,m in model.named_modules()}
        evaluator=LLMAttributionEvaluator(model,tokenizer)
        finite=VendorFAFiniteP1BF16D256(env['finite_library'],env['finite_library_sha256'])
        options=env.get('dt_compiler_options')
        runner=make_qwen35_clean_runner(model,finite,
            make_compiled_finite_pullback(reuse_scalar_products=False,dynamic_shapes=True,compiler_options=options),
            dynamic_shapes=True,compiler_options=options)
        for field in ('norm_gate_rules','finite_fla_by_layer','attention_pv_rules','key_norm_by_layer'):assert getattr(runner,field)=={}
        tracer=FlashTrace(model,tokenizer,use_chat_template=False)
        completed=0
        for task,indices in plan['selection'].items():
            original=json.loads((full/task/'results.json').read_bytes())
            assert original['status']=='complete'
            rows={r['index']:r for r in original['cases']}
            assert digest(full/task/'vectors.npz')==original['vectors_sha256']
            examples=dataset_utils.load_cached(Path(env['official_root'])/'exp/exp2/data'/f'{task}.jsonl')
            with np.load(full/task/'vectors.npz',allow_pickle=False) as archive:
                for index in indices:
                    if a.max_cases is not None and completed>=a.max_cases:break
                    key=f'{task}_{index}';r=rows[index];ex=copy.deepcopy(examples[index])
                    state('prepare_'+key)
                    ex.indices_to_explain=ex.sink_span=ex.thinking_span=None
                    ex=dataset_utils.attach_spans_from_answer(ex,tokenizer)
                    assert ex.sink_span is not None and ex.thinking_span is not None
                    engine=ft.LLMIFRAttributionBoth(model,tokenizer,show_progress=False)
                    ids,mask,prompt_len,gen_len=engine._ensure_generation(ex.prompt,ex.target)
                    positions=list(engine.user_prompt_indices);keep=ft.keep_token_indices(engine.user_prompt_tokens)
                    assert ids_digest(ids.cpu().numpy())==r['input_sha256']
                    assert positions==r['user_positions'] and keep==r['keep']
                    if completed==0:
                        with torch.no_grad():warm=model(input_ids=ids,attention_mask=mask,use_cache=False)
                        del warm
                    enc=tokenizer(' '+ex.prompt,add_special_tokens=False,return_offsets_mapping=True)
                    assert enc['input_ids']==[int(ids[0,j]) for j in positions]
                    query_start=(' '+ex.prompt).rfind('\nWhat ')+1;assert query_start>0
                    query=sorted(j for j in keep if enc['offset_mapping'][j][1]>query_start)
                    nonquery=sorted(set(keep)-set(query));assert query and nonquery
                    targets=ids[:,prompt_len:];generation_tokens=list(engine.generation_tokens)
                    all_offsets=list(range(gen_len));content=[j for j in all_offsets if j<gen_len-1 and not ft_stop(generation_tokens[j])]
                    answer_span=list(range(ex.sink_span[0],ex.sink_span[1]+1))
                    answer_content=[j for j in answer_span if not ft_stop(generation_tokens[j])]
                    assert content and answer_content and int(targets[0,-1])==tokenizer.eos_token_id
                    case_dir=a.output/key;case_dir.mkdir()
                    case=dict(dataset=task,index=index,input_sha256=r['input_sha256'],input_ids=r['input_ids'],
                        prompt_length=prompt_len,target_length=gen_len,user_positions=positions,keep=keep,gold=r['gold'],
                        query=query,nonquery=nonquery,answer_span=answer_span,answer_content=answer_content,
                        full_content=content,thinking_span=ex.thinking_span,vectors={},curves={},evaluations={},
                        new_model_calls=0,started=time.time(),status='attribute')
                    vectors={};responses={};cache={}
                    def persist_case():
                        save(case_dir/'results.json',case)
                        save(a.output/'status.json',report)
                    def attrib(name,scope,offsets):
                        state(key+'_'+name)
                        model.set_attn_implementation('flash_attention_2')
                        base=ids.clone();base[0,[positions[j] for j in scope]]=tokenizer.eos_token_id
                        selection=PackedAnswerTargets([dict(target_ids=targets[0].cpu(),prompt_length=prompt_len)],
                            [offsets],ids.shape[1],model.device)
                        pair=torch.cat((base,ids))
                        observed=[]
                        def check(_m,_args,kw):observed.append(ids_digest(kw['input_ids'].detach().cpu().numpy()))
                        h=model.register_forward_pre_hook(check,with_kwargs=True)
                        try:value,detail=runner.attribute(pair,torch.ones_like(pair),selection,select_output_rows=True,observer=None)
                        finally:h.remove()
                        assert observed==[ids_digest(pair.cpu().numpy())]
                        v=value[0].detach().cpu().numpy().copy()
                        vectors[name]=v[positions].astype(np.float32)
                        vectors[name+'_full_sequence']=v
                        case['vectors'][name]=dict(reference_sha256=ids_digest(base.cpu().numpy()),target_offsets=offsets,
                            root_effect=detail['root_effect'],signed_sum=detail['signed_sum'],relative_residual=detail['relative_residual'],
                            max_replay_l2=max(x['replay_relative_L2'] for x in detail['layers'].values()))
                        del value,detail,pair,base,selection
                    attrib('DT_full',keep,all_offsets)
                    old=archive[key+'_DT_signed_full']
                    case['DT_archive_comparison']=dict(bitwise_equal=bool(np.array_equal(old,vectors['DT_full_full_sequence'])),
                        relative_l2=float(np.linalg.norm(old-vectors['DT_full_full_sequence'])/max(np.linalg.norm(old),1e-30)))
                    assert case['DT_archive_comparison']['relative_l2']<1e-4
                    # Use archived vectors for exact original ranking and fresh vectors for every new condition.
                    vectors['DT_original']=old[positions].astype(np.float32)
                    vectors['FT_answer_K1']=archive[key+'_FT_K1_prompt'].copy()
                    attrib('DT_full_content',keep,content)
                    attrib('DT_answer_content',keep,answer_content)
                    attrib('DT_nonquery_reference',nonquery,all_offsets)
                    state(key+'_FT_full_K1');model.set_attn_implementation('eager')
                    ft_inputs=[]
                    def check_ft(_m,_args,kw):
                        assert torch.equal(kw['input_ids'],ids)
                        ft_inputs.append(ids_digest(kw['input_ids'].detach().cpu().numpy()))
                    hook=model.register_forward_pre_hook(check_ft,with_kwargs=True)
                    try:
                        fresh=tracer.trace(prompt=evaluator.format_prompt(' '+ex.prompt),target=ex.target,
                            output_span=(0,gen_len-2),reasoning_span=tuple(ex.thinking_span),hops=1,method='flashtrace')
                    finally:hook.remove()
                    assert ft_inputs==[r['input_sha256']]
                    vectors['FT_full_K1']=np.asarray(fresh.scores,dtype=np.float32)[positions].copy();del fresh
                    for n,m in model.named_modules():assert (type(m).forward,m.forward)==originals[n]
                    model.set_attn_implementation('eager')
                    def evaluate(deleted):
                        deleted=tuple(sorted(set(deleted)))
                        if deleted in cache:return cache[deleted]
                        assert set(deleted)<=set(keep)
                        actual=ids[:,:prompt_len].clone()
                        actual[0,[positions[j] for j in deleted]]=tokenizer.eos_token_id
                        with torch.no_grad():lp=evaluator.compute_logprob_response_given_prompt(actual,targets)
                        native_full=float(lp.sum().item())
                        native_answer=float(lp[:,answer_span].sum().item())
                        native_answer_content=float(lp[:,answer_content].sum().item())
                        arr=lp[0].detach().float().cpu().numpy().copy()
                        eid=f'e{len(cache):04d}';responses[eid]=arr
                        info=dict(input_sha256=ids_digest(torch.cat((actual,targets),dim=1).cpu().numpy()),
                            deleted=list(deleted),full=native_full,answer=native_answer,answer_content=native_answer_content,
                            full_fp64=float(arr.astype(np.float64).sum()),answer_fp64=float(arr[answer_span].astype(np.float64).sum()),
                            answer_content_fp64=float(arr[answer_content].astype(np.float64).sum()))
                        case['evaluations'][eid]=info;cache[deleted]=eid
                        case['new_model_calls']+=1;report['model_calls']+=1
                        del lp,actual
                        return eid
                    def rank_path(name,scope,positive=False):
                        w=torch.tensor(vectors[name],dtype=torch.float32)
                        if positive:w=w.clamp_min(0)
                        order=torch.argsort(w[scope],descending=True).tolist();ordered=[scope[j] for j in order]
                        base,rem=divmod(len(scope),plan['steps']);path=[[]];offset=0
                        for step in range(plan['steps']):
                            offset+=base+(step<rem);path.append(ordered[:offset])
                        return path
                    def curve(label,path,scope,vector=None,official=False):
                        state(key+'_'+label)
                        eids=[evaluate(d) for d in path]
                        values=np.array([case['evaluations'][i]['full'] for i in eids])
                        gap=abs(values[0]-values[-1])
                        y=np.minimum.accumulate(np.clip((values-values[-1])/gap,0,1)) if gap>0 else None
                        item=dict(evaluations=eids,scope=scope,full_rise=auc(y) if y is not None else None,
                            normalized_full=y.tolist() if y is not None else None,
                            endpoint_gap=float(values[0]-values[-1]),official_path=official)
                        if vector is not None and y is not None:
                            w=np.maximum(vectors[vector][scope].astype(np.float32),0);total=float(torch.tensor(w).sum().item())
                            density=[1.0];removed=set()
                            for d in path[1:]:
                                added=set(d)-removed;removed=set(d)
                                dec=float(torch.tensor(np.maximum(vectors[vector][list(added)],0)).sum().item())/total if total>0 else 0
                                density.append(density[-1]-dec)
                            if total<=0:density=np.linspace(1,0,len(path)).tolist()
                            corrected=np.clip(y+np.abs(y-np.array(density)),0,1)
                            if corrected.max()==corrected.min():corrected=np.linspace(1,0,len(path))
                            else:corrected=(corrected-corrected.min())/(corrected.max()-corrected.min())
                            item['mas']=auc(corrected);item['density']=list(density)
                        case['curves'][label]=item;persist_case()
                    for method,saved_name in [('DT_original','DT_signed'),('FT_answer_K1','FT_K1')]:
                        saved=r['metrics'].get(saved_name,r['metrics']['DT_positive'])
                        path=rank_path(method,keep)
                        curve(method,path,keep,official=True)
                        old_scores=np.array(saved['scores']);fresh_scores=np.array([case['evaluations'][i]['full'] for i in case['curves'][method]['evaluations']])
                        case['curves'][method]['archive_max_score_difference']=float(np.max(np.abs(old_scores-fresh_scores)))
                        case['curves'][method]['archive_rise_difference']=case['curves'][method]['full_rise']-saved['rise']
                        case['curves'][method]['archive_actual_input_matches']=sum(case['evaluations'][eid]['input_sha256']==h for eid,h in zip(case['curves'][method]['evaluations'],saved['actual_input_hashes']))
                        case['curves'][method]['archive_signed_reuse_proof']=r['metrics']['DT'].get('signed_RISE_reuse_proof') if method=='DT_original' else None
                        # A factorial intervention at every original budget. These are causal diagnostics, not fresh equal-count rankings.
                        curve(method+'_restore_query',[[j for j in d if j not in query] for d in path],nonquery)
                        curve(method+'_only_query',[[j for j in d if j in query] for d in path],query)
                    for method in ('DT_full_content','DT_answer_content','FT_full_K1'):
                        curve(method,rank_path(method,keep),keep)
                        if method.startswith('DT'):
                            curve(method+'_positive',rank_path(method,keep,True),keep,vector=method)
                    curve('DT_fixed_query',rank_path('DT_nonquery_reference',nonquery),nonquery)
                    curve('DT_old_reference_fixed_query',rank_path('DT_original',nonquery),nonquery)
                    curve('FT_fixed_query',rank_path('FT_answer_K1',nonquery),nonquery,vector='FT_answer_K1')
                    np.savez_compressed(case_dir/'vectors.npz',**vectors)
                    np.savez_compressed(case_dir/'token_logprobs.npz',**responses)
                    case['vectors_sha256']=digest(case_dir/'vectors.npz');case['token_logprobs_sha256']=digest(case_dir/'token_logprobs.npz')
                    case['status']='complete';case['finished']=time.time();persist_case()
                    completed+=1
                    report['cases'].append(dict(dataset=task,index=index,path=key,results_sha256=digest(case_dir/'results.json'),model_calls=case['new_model_calls']))
                    state('completed_'+key)
                    print(json.dumps(dict(case=key,complete=completed,model_calls=case['new_model_calls'],rises={k:round(v['full_rise'],5) if v['full_rise'] is not None else None for k,v in case['curves'].items()})),flush=True)
                    del ids,mask,targets,engine,case,vectors,responses,cache
                    gc.collect();torch.cuda.empty_cache()
            if a.max_cases is not None and completed>=a.max_cases:break
        report['finished']=time.time();state('complete')
    except Exception:
        report['error']=traceback.format_exc();state('failed');raise

if __name__=='__main__':main()
