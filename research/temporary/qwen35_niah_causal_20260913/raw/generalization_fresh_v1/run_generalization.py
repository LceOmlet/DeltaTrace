"""Frozen broader-task check of the deployed symmetric GDN profile."""
import argparse,copy,gc,hashlib,inspect,json,os,re,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from run_causal import save,digest,ids_digest,auc

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=Path)
    a=p.parse_args();root=a.run_root;code=root/'repo_dynamic';full=root/'full_dynamic_with_ifr'
    plan=json.loads(a.protocol.read_bytes());env_path=root/'environment_dynamic.json'
    env=json.loads(env_path.read_bytes())['qwen35'];identity=json.loads((full/'identity.json').read_bytes())
    assert digest(env_path)==identity['environment_sha256']
    assert digest(code/'deltatrace/clean/sources.json')==identity['clean_sources_sha256']
    for name,spec in json.loads((code/'deltatrace/clean/sources.json').read_bytes())['models']['qwen35']['files'].items():assert digest(code/name)==spec['sha256']
    os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS='0',
        TRITON_CACHE_DIR=env['triton_cache'],TORCHINDUCTOR_CACHE_DIR=env['inductor_cache'])
    os.environ.update(env['runtime_environment']);os.environ['PATH']=str(root/'env/bin')+':/opt/conda/bin:/usr/bin:/bin:'+os.environ.get('PATH','')
    sys.path.insert(0,env['official_root']);sys.path.insert(0,str(code/'deltatrace/clean/qwen35'));sys.path.append(env['ft_extension_root'])
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from exp.exp2 import dataset_utils
    from llm_attr_eval import LLMAttributionEvaluator
    import ft_ifr_improve as ft
    from flashtrace import FlashTrace
    from qwen35_answer_finite import PackedAnswerTargets
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    from dt_variants import build_uniform_runners
    torch.set_num_threads(4);torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
    reuse=json.loads((full/'process_reuse.json').read_bytes())
    for name in ('cache_size_limit','accumulated_cache_size_limit'):setattr(torch._dynamo.config,name,reuse[name])
    verify_native_sources(env['native_stage_source_sha256'])
    gpu=subprocess.run(['/usr/bin/mx-smi'],capture_output=True,text=True,check=True).stdout
    assert not re.findall(r'^\|\s+0\s+(\d+)\s+\S+',gpu,re.MULTILINE),'GPU occupied'
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'protocol.json').write_bytes(a.protocol.read_bytes())
    for name in (Path(__file__).name,'dt_variants.py','run_causal.py','qwen35_gdn_symmetric.py'):(a.output/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(status='loading',generation_calls=0,cases=[],score_model_calls=0,pid=os.getpid(),
        protocol_sha256=digest(a.protocol),driver_sha256=digest(__file__),variants_sha256=digest(Path(__file__).with_name('dt_variants.py')),
        environment_sha256=digest(env_path),clean_sources_sha256=identity['clean_sources_sha256'],initial_gpu=gpu)
    if a.cases:
        report['cases_sha256']=digest(a.cases);(a.output/'input_cases.json').write_bytes(a.cases.read_bytes())
    def state(label):report['status']=label;save(a.output/'status.json',report)
    state('loading')
    try:
        tokenizer=AutoTokenizer.from_pretrained(env['checkpoint'],local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
        model=Qwen3_5ForConditionalGeneration.from_pretrained(env['checkpoint'],dtype=torch.bfloat16,
            attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True)
        model.eval().requires_grad_(False);assert digest(inspect.getfile(type(model)))==env['native_model_sha256']
        originals={n:(type(m).forward,m.forward) for n,m in model.named_modules()}
        evaluator=LLMAttributionEvaluator(model,tokenizer);tracer=FlashTrace(model,tokenizer,use_chat_template=False)
        options=env.get('dt_compiler_options');finite=VendorFAFiniteP1BF16D256(env['finite_library'],env['finite_library_sha256'])
        memory=make_compiled_finite_pullback(reuse_scalar_products=False,dynamic_shapes=True,compiler_options=options)
        runners=build_uniform_runners(model,finite,memory,options)
        from qwen35_gdn_symmetric import make_qwen35_gdn_symmetric_runner
        prototype=runners['DT_gdn_symmetric']
        runners['DT_gdn_symmetric']=make_qwen35_gdn_symmetric_runner(
            model,finite,memory,dynamic_shapes=True,compiler_options=options)
        report['deployed_profile_sha256']=digest(Path(__file__).with_name('qwen35_gdn_symmetric.py'))
        examples=[]
        if a.cases:
            for item in json.loads(a.cases.read_bytes())['cases']:
                ex=dataset_utils.CachedExample(prompt=item['prompt'],target=item['target'],indices_to_explain=None,
                    attr_mask_indices=None,sink_span=None,thinking_span=None,metadata=item['metadata'])
                examples.append((item['dataset'],item['index'],ex,None))
        else:
            for task,indices in plan['selection'].items():
                raw=json.loads((full/task/'results.json').read_bytes());records={x['index']:x for x in raw['cases']}
                cached=dataset_utils.load_cached(Path(env['official_root'])/'exp/exp2/data'/f'{task}.jsonl')
                for index in indices:examples.append((task,index,copy.deepcopy(cached[index]),records[index]))
        for ordinal,(task,index,ex,old) in enumerate(examples):
            key=f'{task}_{index}';state('prepare_'+key)
            ex.indices_to_explain=ex.sink_span=ex.thinking_span=None;ex=dataset_utils.attach_spans_from_answer(ex,tokenizer)
            assert ex.sink_span is not None and ex.thinking_span is not None
            engine=ft.LLMIFRAttributionBoth(model,tokenizer,show_progress=False)
            ids,mask,pl,gl=engine._ensure_generation(ex.prompt,ex.target);positions=list(engine.user_prompt_indices)
            keep=ft.keep_token_indices(engine.user_prompt_tokens);targets=ids[:,pl:]
            gold=dataset_utils.ruler_gold_prompt_token_indices(ex,tokenizer)
            assert not gold,'This check expects tasks without retrieval labels.'
            if old:
                assert ids_digest(ids.cpu().numpy())==old['input_sha256'];assert keep==old['keep'] and positions==old['user_positions']
                assert gold==old['gold']
            if ordinal==0:
                with torch.no_grad():init=model(input_ids=ids,attention_mask=mask,use_cache=False)
                del init
            base=ids.clone();base[0,[positions[j] for j in keep]]=tokenizer.eos_token_id
            pair=torch.cat((base,ids));selection=PackedAnswerTargets([dict(target_ids=targets[0].cpu(),prompt_length=pl)],
                [list(range(gl))],ids.shape[1],model.device)
            case_dir=a.output/key;case_dir.mkdir()
            case=dict(dataset=task,index=index,input_ids=ids[0].cpu().tolist(),input_sha256=ids_digest(ids.cpu().numpy()),
                reference_sha256=ids_digest(base.cpu().numpy()),prompt_length=pl,target_length=gl,user_positions=positions,
                keep=keep,gold=gold,methods={},curves={},evaluations={},status='attributing')
            vectors={};responses={};cache={}
            def persist():save(case_dir/'results.json',case);save(a.output/'status.json',report)
            def pull(name,runner,actual_pair):
                state(key+'_'+name);model.set_attn_implementation('flash_attention_2')
                observed=[]
                def check(_m,_args,kw):observed.append(ids_digest(kw['input_ids'].detach().cpu().numpy()))
                h=model.register_forward_pre_hook(check,with_kwargs=True)
                try:value,detail=runner.attribute(actual_pair,torch.ones_like(actual_pair),selection,select_output_rows=True,observer=None)
                finally:h.remove()
                assert observed==[ids_digest(actual_pair.cpu().numpy())]
                v=value[0].detach().cpu().numpy().copy();del value
                case['methods'][name]=dict(root_effect=detail['root_effect'],signed_sum=detail['signed_sum'],relative_residual=detail['relative_residual'],
                    max_replay_l2=max(x['replay_relative_L2'] for x in detail['layers'].values()),target_offsets=list(range(gl)),
                    attribution_seconds=detail['complete_attribution_seconds_with_diagnostics'],
                    peak_allocated_bytes=detail['peak_allocated'])
                assert case['methods'][name]['max_replay_l2']==0
                del detail
                return v
            for name in plan['methods']:
                if name=='DT_endpoint_symmetric':
                    reversed_pair=pair.flip(0)
                    reverse=pull('DT_reversed_endpoints',runners['DT_original'],reversed_pair)
                    v=(vectors['DT_original_full_sequence']-reverse)*0.5
                    vectors['DT_reversed_endpoints_full_sequence']=reverse
                    effect=case['methods']['DT_original']['root_effect']
                    case['methods'][name]=dict(root_effect=effect,signed_sum=float(v.sum()),relative_residual=float((effect-v.sum())/effect),
                        max_replay_l2=0,target_offsets=list(range(gl)))
                    del reversed_pair
                else:v=pull(name,runners[name],pair)
                vectors[name+'_full_sequence']=v;vectors[name]=v[positions].astype(np.float32)
                case['methods'][name]['recall']=None
                np.savez_compressed(case_dir/'vectors_partial.npz',**vectors);persist()
            if ordinal==0:
                expected=pull('DT_gdn_symmetric_prototype',prototype,pair)
                actual=vectors['DT_gdn_symmetric_full_sequence']
                case['deployment_relative_l2']=float(np.linalg.norm(expected-actual)/max(np.linalg.norm(expected),1e-30))
                case['deployment_bitwise']=bool(np.array_equal(expected,actual))
                assert case['deployment_bitwise'],case['deployment_relative_l2']
                del expected,actual
                # Both paths have already run. Warm once more, then alternate
                # five measured pairs on this exact input with the same runner.
                case['profile_cost']={'unit':'Complete attribution with existing CPU checkpoints and diagnostics; no scoring/model loading.',
                    'warmup_pairs':1,'measurement_pairs':5,'rows':[]}
                for rep in range(6):
                    order=plan['methods'] if rep%2==0 else list(reversed(plan['methods']))
                    for method in order:
                        label=method+'_timing';timed=pull(label,runners[method],pair)
                        assert np.array_equal(timed,vectors[method+'_full_sequence'])
                        if rep:
                            d=case['methods'][label]
                            case['profile_cost']['rows'].append(dict(method=method,rep=rep,
                                seconds=d['attribution_seconds'],peak_allocated_bytes=d['peak_allocated_bytes']))
                        del timed
            repeated=pull('DT_original_repeat',runners['DT_original'],pair)
            initial=vectors['DT_original_full_sequence']
            case['baseline_repeat_relative_l2']=float(np.linalg.norm(repeated-initial)/max(np.linalg.norm(initial),1e-30))
            case['baseline_repeat_bitwise']=bool(np.array_equal(repeated,initial))
            assert case['baseline_repeat_relative_l2']<1e-4,case['baseline_repeat_relative_l2']
            if ordinal==0:
                name='DT_gdn_symmetric';repeated=pull(name+'_repeat',runners[name],pair)
                initial=vectors[name+'_full_sequence']
                case['candidate_repeat_relative_l2']=float(np.linalg.norm(repeated-initial)/max(np.linalg.norm(initial),1e-30))
                case['candidate_repeat_bitwise']=bool(np.array_equal(repeated,initial))
                assert case['candidate_repeat_relative_l2']<1e-4,case['candidate_repeat_relative_l2']
            del repeated,initial
            if old:
                with np.load(full/task/'vectors.npz',allow_pickle=False) as vv:
                    expected=vv[key+'_DT_signed_full']
                    case['archive_dt_relative_l2']=float(np.linalg.norm(expected-vectors['DT_original_full_sequence'])/max(np.linalg.norm(expected),1e-30))
                    case['archive_dt_max_abs']=float(np.max(np.abs(expected-vectors['DT_original_full_sequence'])))
                    persist()
                    assert case['archive_dt_relative_l2']<1e-4,case['archive_dt_relative_l2']
                    vectors['FT_K1']=vv[key+'_FT_K1_prompt'].copy()
                case['FT_K1']=dict(rise=old['metrics']['FT_K1']['rise'],mas=old['metrics']['FT_K1']['mas'],recall=old['metrics']['FT_K1']['needle'])
                case['FT_K3_recall']=None
            else:
                model.set_attn_implementation('eager')
                for hops in (1,):
                    state(key+f'_FT_K{hops}')
                    expected_calls=[]
                    def check_ft(_m,_args,kw):assert torch.equal(kw['input_ids'],ids);expected_calls.append(True)
                    h=model.register_forward_pre_hook(check_ft,with_kwargs=True)
                    try:f=tracer.trace(prompt=evaluator.format_prompt(' '+ex.prompt),target=ex.target,output_span=tuple(ex.sink_span),
                        reasoning_span=tuple(ex.thinking_span),hops=hops,method='flashtrace')
                    finally:h.remove()
                    assert len(expected_calls)==1
                    vectors[f'FT_K{hops}']=np.asarray(f.scores,dtype=np.float32)[positions].copy();del f
                case['FT_K3_recall']=None
            for n,m in model.named_modules():assert (type(m).forward,m.forward)==originals[n]
            model.set_attn_implementation('eager')
            def evaluate(deleted):
                key=tuple(sorted(deleted))
                if key in cache:return cache[key]
                actual=ids[:,:pl].clone();actual[0,[positions[j] for j in deleted]]=tokenizer.eos_token_id
                with torch.no_grad():lp=evaluator.compute_logprob_response_given_prompt(actual,targets)
                score=float(lp.sum().item());eid=f'e{len(cache):04d}'
                responses[eid]=lp[0].detach().float().cpu().numpy().copy();del lp
                case['evaluations'][eid]=dict(deleted=sorted(deleted),input_sha256=ids_digest(torch.cat((actual,targets),dim=1).cpu().numpy()),score=score)
                cache[key]=eid;report['score_model_calls']+=1;del actual
                return eid
            def measure(name,positive):
                label=name+('_positive' if positive else '_signed');state(key+'_'+label)
                w=torch.tensor(vectors[name],dtype=torch.float32)
                if positive:w=w.clamp_min(0)
                order=torch.argsort(w[keep],descending=True).tolist();ordered=[keep[j] for j in order]
                total=float(w[keep].sum().item());density=[1.0];path=[[]]
                q,rem=divmod(len(keep),plan['steps']);offset=0
                for step in range(plan['steps']):
                    size=q+(step<rem);group=ordered[offset:offset+size];offset+=size;path.append(ordered[:offset])
                    density.append(density[-1]-float(w[group].sum().item())/total if total>0 else 0)
                if total<=0:density=np.linspace(1,0,plan['steps']+1).tolist()
                eids=[evaluate(d) for d in path];scores=np.array([case['evaluations'][i]['score'] for i in eids])
                assert scores[0]>scores[-1]
                y=np.minimum.accumulate(np.clip((scores-scores[-1])/abs(scores[0]-scores[-1]),0,1))
                entry=dict(evaluations=eids,rise=auc(y),normalized=y.tolist())
                if positive:
                    corrected=np.clip(y+np.abs(y-np.array(density)),0,1)
                    if corrected.max()==corrected.min():corrected=np.linspace(1,0,len(y))
                    else:corrected=(corrected-corrected.min())/(corrected.max()-corrected.min())
                    entry['mas']=auc(corrected);entry['density']=density
                case['curves'][label]=entry;persist()
                return entry
            for name in plan['methods']:
                signed=measure(name,False);positive=measure(name,True)
                case['methods'][name].update(rise=signed['rise'],mas=positive['mas'])
            fresh_ft=measure('FT_K1',True)
            ft_recall=None
            if old:
                case['archive_ft_rise_difference']=fresh_ft['rise']-case['FT_K1']['rise']
                case['archive_dt_rise_difference']=case['methods']['DT_original']['rise']-old['metrics']['DT']['rise']
                case['archive_dt_mas_difference']=case['methods']['DT_original']['mas']-old['metrics']['DT']['mas']
            case['FT_K1']=dict(rise=fresh_ft['rise'],mas=fresh_ft['mas'],recall=ft_recall)
            np.savez_compressed(case_dir/'vectors.npz',**vectors);np.savez_compressed(case_dir/'token_logprobs.npz',**responses)
            case['vectors_sha256']=digest(case_dir/'vectors.npz');case['token_logprobs_sha256']=digest(case_dir/'token_logprobs.npz')
            case['status']='complete';persist()
            report['cases'].append(dict(dataset=task,index=index,path=key,results_sha256=digest(case_dir/'results.json')));state('completed_'+key)
            print(json.dumps(dict(case=key,complete=len(report['cases']),methods={k:{f:v.get(f) for f in ['rise','mas','recall','relative_residual']} for k,v in case['methods'].items() if k in plan['methods']})),flush=True)
            del case,vectors,responses,cache,ids,mask,targets,base,pair,selection,engine
            gc.collect();torch.cuda.empty_cache()
        state('complete')
    except Exception:
        report['error']=traceback.format_exc();state('failed');raise

if __name__=='__main__':main()
