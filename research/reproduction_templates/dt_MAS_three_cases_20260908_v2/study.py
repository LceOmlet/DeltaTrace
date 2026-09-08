"""Reviewed recovery: restore the original native warm execution before the same check.

External orchestration only. Complete official FT is called unchanged; DT uses
the existing finite rules and genuine model FA/FLA. Original scoring and metrics
are called through the previously disclosed fixed-input formatter view.
"""
import os, sys, json, hashlib, time, signal, traceback, zipfile
from pathlib import Path

A = Path(__file__).resolve().parent
p = json.loads((A / 'protocol.json').read_bytes())
sha = lambda b: hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
    PYTHONDONTWRITEBYTECODE='1', TRITON_CACHE_DIR=p['compiler_cache'],
    TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']: sys.path.insert(0, overlay)
sys.path.insert(0, p['official_root']); sys.path.insert(0, str(A))
r = dict(status='starting', protocol=p, model_loads=0, DT_calls=0, FT_calls=0,
    scoring_forwards_entered=0, scoring_forwards_completed=0, generation_calls=0,
    cases={}, control={}, calls=[])
started = time.perf_counter(); handles=[]; vectors={}


def save():
    t=A/'results.partial'; t.write_text(json.dumps(r, ensure_ascii=False, indent=2)); t.replace(A/'results.json')


def timeout(*args): raise TimeoutError('Frozen three-case total execution budget expired.')


def sources():
    out={}
    for name, want in p['files_sha256'].items():
        out[name]=sha((A/name).read_bytes()); assert out[name]==want, name
    for name, want in p['official_source_blob_sha1'].items():
        raw=(Path(p['official_root'])/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want, name
        out['FT/'+name]=sha(raw)
    for name, want in p['runtime_source_sha256'].items():
        out['native/'+name]=sha((Path(p['isolated_site'])/name).read_bytes()); assert out['native/'+name]==want, name
    return out


def timed(kind, fn):
    torch.cuda.synchronize(); t=time.perf_counter(); value=fn(); torch.cuda.synchronize()
    r['calls'].append({'kind':kind,'seconds':time.perf_counter()-t}); return value


def prepare(dataset, index):
    record=json.loads(raw_data[dataset].decode().splitlines()[index])
    cls=helpers['CachedExample']; fields=['prompt','target','indices_to_explain','attr_mask_indices','sink_span','thinking_span','metadata']
    cached=cls(**{k:record.get(k) for k in fields})
    def remap(tok):
        item=cls(prompt=cached.prompt,target=cached.target,indices_to_explain=None,
            attr_mask_indices=cached.attr_mask_indices,sink_span=None,thinking_span=None,metadata=dict(cached.metadata))
        item=helpers['attach_spans_from_answer'](item,tok); item.indices_to_explain=list(item.sink_span); return item
    old, new=remap(old_tokenizer),remap(tokenizer)
    for k in ['sink_span','thinking_span','indices_to_explain']: assert getattr(old,k)==getattr(cached,k), k
    tokens=tokenizer(record['prompt'],add_special_tokens=False,return_offsets_mapping=True)
    labels=[record['prompt'][a:b] for a,b in tokens['offset_mapping']]
    keep=keep_token_indices(labels)
    assert keep==helpers['keep_token_indices'](labels)
    target=tokenizer(record['target']+tokenizer.eos_token,add_special_tokens=False)['input_ids']
    assert target==tokenizer(record['target'],add_special_tokens=False)['input_ids']+[tokenizer.eos_token_id]
    ids=torch.tensor(tokens['input_ids']+target,dtype=torch.long)
    assert len(ids)<=p['max_total_tokens'] and keep and len(keep)>=20
    base=ids.clone(); base[keep]=tokenizer.eos_token_id
    assert (base!=ids).nonzero().flatten().tolist()==keep
    gold=helpers['ruler_gold_prompt_token_indices'](new,tokenizer)
    existing=next((c for c in old_spans['cases'] if (c['dataset'],c['index'])==(dataset,index)),None)
    if existing:
        mapping=existing['mapping']
        assert list(new.sink_span)==mapping['new_sink_span'] and list(new.thinking_span)==mapping['new_thinking_span']
        assert gold==mapping['gold_user_token_indices'] and keep==mapping['keep_local_indices']
    key=f'{dataset}_{index}'
    info=dict(dataset=dataset,index=index,prompt_length=len(tokens['input_ids']),target_length=len(target),total_length=len(ids),
        input_sha256=sha(ids.numpy().tobytes()),baseline_sha256=sha(base.numpy().tobytes()),
        keep=keep,gold=gold,sink_span=new.sink_span,thinking_span=new.thinking_span,
        original_cached_spans_reproduced=True,source_record_sha256=sha(raw_data[dataset].decode().splitlines()[index].encode()))
    return dict(key=key,record=record,info=info,ids=ids,base=base,target_ids=torch.tensor(target),prompt_length=info['prompt_length'])


def recovery(score, case):
    info=case['info']; keep=info['keep']; gold=info['gold']
    if not gold: return None
    value=evaluate_attr_recovery_skip_tokens(score[None],keep_prompt_token_indices=keep,gold_prompt_token_indices=gold,top_fraction=0.1)
    return float(value)


try:
    signal.signal(signal.SIGALRM,timeout); signal.alarm(p['budget']['wall_time_seconds'])
    r['sources_before']=sources()
    import numpy as np
    import torch, flash_attn
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    from flashtrace import FlashTrace
    from flashtrace.improved import keep_token_indices,evaluate_attr_recovery_skip_tokens,faithfulness_test_skip_tokens
    from llm_attr_eval import LLMAttributionEvaluator
    from official_span_mapping import load_author_span_helpers
    from qwen35_answer_finite import PackedAnswerTargets
    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    from fixed_input_metric_view import FixedInputMetricView
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import causal_conv1d
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule and native.is_fast_path_available
    verify_native_sources(p['native_stage_source_sha256'])
    cp=Path(p['checkpoint'])
    for name,want in p['checkpoint_config_tokenizer_sha256'].items(): assert sha((cp/name).read_bytes())==want
    stats=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats(); assert r['weight_stats_before']==p['expected_weight_stats']
    tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True); tokenizer.pad_token=tokenizer.eos_token
    old_tokenizer=AutoTokenizer.from_pretrained(p['old_checkpoint'],local_files_only=True); old_tokenizer.pad_token=old_tokenizer.eos_token
    helpers=load_author_span_helpers(p['author_data_root'],p['span_source_sha256'])
    raw_data={name:Path(path).read_bytes() for name,path in p['cache_paths'].items()}
    for name,raw in raw_data.items(): assert sha(raw)==p['cache_hashes'][name]
    old_spans=json.loads(Path(p['spans_path']).read_bytes()); assert sha(Path(p['spans_path']).read_bytes())==p['spans_sha256']
    control=prepare('niah_mq_q2',0); cases=[prepare(*item) for item in p['selection']]
    assert control['info']['input_sha256']==p['control_input_sha256']
    r['control']['input']=control['info']; r['cases']={c['key']:dict(input=c['info'],curves={}) for c in cases}; save()
    assert torch.cuda.mem_get_info()[0]>=32*1024**3
    torch.manual_seed(73); r['status']='loading_actual_model';save()
    model, loading=timed('model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,
        attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True))
    model.eval().requires_grad_(False);r['model_loads']=1;assert not any(loading.values())
    for layer in model.model.language_model.layers:
        if layer.block_type=='linear_attention':
            assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
            assert layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
    # Restore the original NI0 native initialization/execution order exactly:
    # eager B1 diagnostic first, then the official backend setter to FA B2.
    cpairs=control['ids'][None].to('cuda')
    with torch.no_grad(): warm=timed('original_order_eager_B1_diagnostic',lambda:model(input_ids=cpairs,attention_mask=torch.ones_like(cpairs),use_cache=False))
    positions=torch.arange(control['info']['prompt_length']-1,control['info']['total_length']-1,device='cuda')
    labels=control['target_ids'].to('cuda')
    logp=warm.logits[0,positions].float().log_softmax(-1).gather(-1,labels[:,None]).squeeze(-1)
    r['control']['eager_B1_target_logprob_sum']=float(logp.double().sum())
    del warm,cpairs,positions,labels,logp
    model.set_attn_implementation('flash_attention_2')
    runner=Qwen35DenseFiniteRunner(model,VendorFAFiniteP1BF16D256(p['finite_FA_library'],p['finite_FA_library_sha256']),
        make_compiled_finite_pullback(reuse_scalar_products=False))
    # The output-head row optimization remains disabled in this quality check.
    for case in [control]+cases:
        key=case['key']; info=case['info'];r['status']='DT_'+key;save()
        pair=torch.stack((case['base'],case['ids'])).to('cuda');mask=torch.ones_like(pair)
        sel=PackedAnswerTargets([case],[list(range(info['target_length']))],info['total_length'],'cuda')
        signed, detail=runner.attribute(pair,mask,sel,select_output_rows=False);r['DT_calls']+=1
        signed=signed[0]; assert signed.shape==case['ids'].shape and torch.isfinite(signed).all()
        assert all(float(signed[j])==0 for j in range(len(signed)) if j not in set(info['keep']))
        score=signed[:info['prompt_length']].float();vectors[key+'_DT_full']=signed.numpy();vectors[key+'_DT']=score.numpy()
        np.savez_compressed(A/'vectors.npz',**vectors)
        detail['needle']=recovery(score,case)
        if case is control:
            raw=Path(p['control_vectors_path']).read_bytes();assert sha(raw)==p['control_vectors_sha256']
            prior=np.load(p['control_vectors_path'])['signed']; now=signed.numpy()
            rel=float(np.linalg.norm(now-prior)/np.linalg.norm(prior));detail['versus_frozen_relative_L2']=rel
            detail['versus_frozen_max_abs']=float(np.max(np.abs(now-prior)))
            r['control']['DT']=detail;save()
            # Frozen broad sanity check, not bitwise/default-precision policing.
            assert rel<=p['control_vector_relative_L2_ceiling']
            assert abs(detail['needle']-0.375)<=p['control_needle_absolute_drift_ceiling']
        else:r['cases'][key]['DT']=detail;save()
        del pair,mask,sel,signed,score
    # Complete official FT path, native eager/FLA exactly as the fixed baseline.
    model.set_attn_implementation('eager')
    assert all(l.self_attn.config._attn_implementation=='eager' for l in model.model.language_model.layers if l.block_type=='full_attention')
    for case in cases:
        key=case['key'];info=case['info'];outrow=r['cases'][key];r['status']='FT_'+key;save()
        native_receipts=[];native_outputs=[]
        def pre(_module,args,kw):
            ids=kw['input_ids'].detach().cpu();assert ids.shape==(1,info['total_length'])
            assert sha(ids.numpy().tobytes())==info['input_sha256'] and bool(kw['attention_mask'].eq(1).all())
            assert kw.get('output_attentions') is True and kw.get('use_cache') is False
            native_receipts.append({'shape':list(ids.shape),'sha256':sha(ids.numpy().tobytes())})
        def post(_module,args,kw,out):
            native_outputs.append({'attention_shapes':[list(v.shape) for v in out.attentions]})
        handles=[model.register_forward_pre_hook(pre,with_kwargs=True),model.register_forward_hook(post,with_kwargs=True)]
        tracer=FlashTrace(model,tokenizer);torch.cuda.reset_peak_memory_stats()
        result=timed('unchanged_official_FT_'+key,lambda:tracer.trace(prompt=case['record']['prompt'],target=case['record']['target'],
            output_span=tuple(info['sink_span']),reasoning_span=tuple(info['thinking_span']),hops=3,method='flashtrace'))
        r['FT_calls']+=1
        for h in handles:h.remove()
        handles=[];assert len(native_receipts)==len(native_outputs)==1
        assert len(native_outputs[0]['attention_shapes'])==8 and len(result.prompt_tokens)==info['prompt_length']
        assert keep_token_indices(result.prompt_tokens)==info['keep']
        obs=result.metadata['ifr']['observation_projected'];prefix=obs['base'].clone();cumulative=[prefix.clone()]
        for extra in obs['per_hop']:prefix=prefix+extra;cumulative.append(prefix.clone())
        assert len(cumulative)==4 and torch.equal(prefix,obs['sum'])
        assert np.array_equal(prefix[:info['prompt_length']].cpu().numpy(),np.asarray(result.scores,dtype=np.float32))
        score=torch.stack(cumulative)[:,:info['prompt_length']].detach().cpu()
        assert torch.isfinite(score).all()
        for i in range(4):vectors[key+f'_FT{i}']=score[i].numpy()
        np.savez_compressed(A/'vectors.npz',**vectors)
        outrow['FT']={'input_receipts':native_receipts,'output_receipts':native_outputs,
            'needle':[recovery(s,case) for s in score], 'peak_allocated':torch.cuda.max_memory_allocated(),
            'peak_reserved':torch.cuda.max_memory_reserved(), 'backend':'unchanged native eager + actual FLA, sample B1'}
        save();del result,tracer,obs,prefix,cumulative,score,extra
    model.set_attn_implementation('flash_attention_2')
    evaluator=LLMAttributionEvaluator(model,tokenizer)
    for case in cases:
        key=case['key'];info=case['info'];outrow=r['cases'][key];view=FixedInputMetricView(evaluator,case['record']['prompt'])
        assert view.compute_logprob_response_given_prompt.__func__ is LLMAttributionEvaluator.compute_logprob_response_given_prompt
        for method in p['method_order']:
            row={'input_receipts':[]};outrow['curves'][method]=row;r['status']='metric_'+key+'_'+method;save()
            def pre(_module,args,kw):
                ids=kw['input_ids'].detach().cpu();assert ids.shape==(1,info['total_length']) and bool(kw['attention_mask'].eq(1).all())
                changed=(ids[0]!=case['ids']).nonzero().flatten().tolist()
                assert set(changed)<=set(info['keep']) and all(int(ids[0,j])==tokenizer.eos_token_id for j in changed)
                if not row['input_receipts']:assert not changed
                row['input_receipts'].append({'input_sha256':sha(ids.numpy().tobytes()),'deleted_positions':changed})
                r['scoring_forwards_entered']+=1
            def post(*args):r['scoring_forwards_completed']+=1
            def observe(frame,event,arg):
                if frame.f_code is faithfulness_test_skip_tokens.__code__ and event=='return' and arg is not None:
                    local=frame.f_locals
                    for name in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores']:row[name]=np.asarray(local[name]).copy().tolist()
                    row['sorted_keep']=[int(j) for j in local['sorted_keep']];row['attr_sum']=float(local['attr_sum'])
                    row['return_metrics']=[float(v) for v in arg]
            handles=[model.register_forward_pre_hook(pre,with_kwargs=True),model.register_forward_hook(post)]
            sys.setprofile(observe)
            try:
                with torch.no_grad(): values=timed('original_metric_'+key+'_'+method,lambda:faithfulness_test_skip_tokens(view,
                    torch.as_tensor(vectors[key+'_'+method])[None],case['record']['prompt'],case['record']['target'],
                    keep_prompt_token_indices=info['keep'],user_prompt_indices=list(range(info['prompt_length'])),k=20))
            finally:
                sys.setprofile(None)
                for h in handles:h.remove()
                handles=[]
            assert len(row['input_receipts'])==21 and row['input_receipts'][-1]['deleted_positions']==info['keep']
            assert row['return_metrics']==[float(v) for v in values]
            assert all(np.isfinite(row[name]).all() for name in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores'])
            save()
    assert r['DT_calls']==4 and r['FT_calls']==3 and r['scoring_forwards_entered']==r['scoring_forwards_completed']==315
    r['sources_after']=sources();r['weight_stats_after']=stats()
    assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    r['status']='fixed_DT_FT_three_author_cases_complete'
except Exception:
    sys.setprofile(None);r['status']='failed';r['error']=traceback.format_exc()
finally:
    signal.alarm(0)
    for h in handles:h.remove()
    r['job_seconds']=time.perf_counter()-started
    if (A/'vectors.npz').exists():r['vectors_sha256']=sha((A/'vectors.npz').read_bytes())
    save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json','vectors.npz']:
            if (A/name).exists():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'seconds':r['job_seconds'],'DT_calls':r['DT_calls'],'FT_calls':r['FT_calls'],
        'scoring_forwards':r['scoring_forwards_completed'],'error':r.get('error')},ensure_ascii=False),flush=True)
