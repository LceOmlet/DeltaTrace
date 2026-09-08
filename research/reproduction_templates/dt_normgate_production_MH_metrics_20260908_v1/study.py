"""Two original MH1 curves for exact saved production vectors; no attribution.

One unchanged model load, the original NI0 eager initialization diagnostic,
then default native FA/FLA. Exactly42 original scorer forwards, never new DT.
"""
import os,sys,json,time,hashlib,traceback,zipfile,signal
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
r={'status':'starting','protocol':p,'model_loads':0,'native_eager_diagnostics':0,'DT_calls':0,'FT_calls':0,
   'finite_FA_calls':0,'finite_FLA_calls':0,'generation_calls':0,
   'scoring_forwards_entered':0,'scoring_forwards_returned':0,'cases':{},'calls':[]}
started=time.perf_counter();handles=[]


def save():
    q=A/'results.partial';q.write_text(json.dumps(r,indent=2));q.replace(A/'results.json')


def sources():
    out={}
    for name,want in p['files_sha256'].items():
        out[name]=sha((A/name).read_bytes());assert out[name]==want,name
    for name,want in p['official_source_blob_sha1'].items():
        raw=(Path(p['official_root'])/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want,name
        out['FT/'+name]=sha(raw)
    for name,want in p['runtime_source_sha256'].items():
        out['native/'+name]=sha((Path(p['isolated_site'])/name).read_bytes());assert out['native/'+name]==want,name
    return out


def timed(kind,fn):
    torch.cuda.synchronize();tick=time.perf_counter();out=fn();torch.cuda.synchronize()
    r['calls'].append({'kind':kind,'seconds':time.perf_counter()-tick});return out


def timeout(*args):raise TimeoutError('Frozen42-score production MH metric budget expired.')


try:
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_time_seconds']);r['sources_before']=sources()
    import numpy as np
    import torch,flash_attn
    torch.set_num_threads(4)
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    import flash_attn.flash_attn_interface as fa
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import causal_conv1d
    from flashtrace.improved import keep_token_indices,evaluate_attr_recovery_skip_tokens,faithfulness_test_skip_tokens
    from llm_attr_eval import LLMAttributionEvaluator
    from fixed_input_metric_view import FixedInputMetricView
    assert p['case_indices']==[['morehopqa',1]] and p['saved_production_run_numbers']=={'current':6,'candidate':7}
    assert p['production_vectors_sha256']=='8daba25f22bc1a48ae47d3700d94610897ce68aa0a76b36630cf3ebfa418e0a9'
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule and native.is_fast_path_available
    cp=Path(p['checkpoint'])
    for name,want in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==want
    stats=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats();assert r['weight_stats_before']==p['expected_weight_stats']
    tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    data={k:Path(v).read_bytes() for k,v in p['cache_paths'].items()}
    for key,raw in data.items():assert sha(raw)==p['cache_hashes'][key]
    raw=Path(p['production_results_path']).read_bytes();assert sha(raw)==p['production_results_sha256'];production=json.loads(raw)
    raw=Path(p['production_vectors_path']).read_bytes();assert sha(raw)==p['production_vectors_sha256']
    with np.load(p['production_vectors_path'],allow_pickle=False) as archive:
        saved_vectors={name:archive[name].copy() for name in archive.files}
    ref=p['paired_reference'];raw=Path(ref['results_path']).read_bytes();assert sha(raw)==ref['results_sha256'];paired=json.loads(raw)

    def prepare(dataset,index):
        rec=json.loads(data[dataset].decode().splitlines()[index]);tok=tokenizer(rec['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        keep=keep_token_indices([rec['prompt'][a:b] for a,b in tok['offset_mapping']])
        target=tokenizer(rec['target']+tokenizer.eos_token,add_special_tokens=False)['input_ids']
        ids=torch.tensor(tok['input_ids']+target,dtype=torch.long)
        info={'input_sha256':sha(ids.numpy().tobytes()),'prompt_length':len(tok['input_ids']),
            'target_length':len(target),'total_length':len(ids),'keep':keep}
        return rec,ids,torch.tensor(target),info

    control=prepare('niah_mq_q2',0);rec,ids,target,info=prepare('morehopqa',1);key='morehopqa_1'
    assert info==production['cases'][key]['input']==paired['cases'][key]['input']
    baseline=ids.clone();baseline[info['keep']]=tokenizer.eos_token_id
    assert sha(baseline.numpy().tobytes())==production['cases'][key]['baseline_sha256'];del baseline
    gold=paired['cases'][key]['gold']
    row={'input':info,'gold':gold,'curves':{},'prior_paired_metrics':{method:paired['cases'][key]['curves'][method]['return_metrics'] for method in ['current','candidate']},
        'prior_fixed_FT_metrics':paired['cases'][key]['prior_fixed_FT_metrics']}
    r['cases'][key]=row
    assert torch.cuda.mem_get_info()[0]>=32*1024**3
    torch.manual_seed(73);r['status']='loading';save()
    model,loading=timed('actual_model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,
        attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True))
    assert not any(loading.values());model.eval().requires_grad_(False);r['model_loads']=1
    for layer in model.model.language_model.layers:
        if layer.block_type=='linear_attention':
            assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
            assert layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
    with torch.no_grad():warm=timed('original_order_native_eager_B1_diagnostic',lambda:model(input_ids=control[1][None].to('cuda'),
        attention_mask=torch.ones_like(control[1][None],device='cuda'),use_cache=False))
    r['native_eager_diagnostics']=1;del warm,control;model.set_attn_implementation('flash_attention_2')
    evaluator=LLMAttributionEvaluator(model,tokenizer);view=FixedInputMetricView(evaluator,rec['prompt'])
    assert view.compute_logprob_response_given_prompt.__func__ is LLMAttributionEvaluator.compute_logprob_response_given_prompt
    for method in ['current','candidate']:
        number=p['saved_production_run_numbers'][method];run=production['runs'][number]
        assert run['case']==key and run['method']==method and run['number']==number and run['status']=='complete'
        name=key+'_'+method+'_run'+str(number)
        full=torch.from_numpy(saved_vectors[name+'_full']);score=torch.from_numpy(saved_vectors[name+'_evaluated'])
        assert full.dtype==torch.float64 and score.dtype==torch.float32 and score.shape==(info['prompt_length'],)
        assert torch.equal(full[:info['prompt_length']].float(),score) and bool(torch.isfinite(score).all())
        expected=run['integration']['deletion_audit']
        curve={'status':'starting','production_run':number,'source_vector_sha256':sha(score.numpy().tobytes()),
            'input_receipts':[],'native_forwards_returned':0,
            'needle':float(evaluate_attr_recovery_skip_tokens(score[None],keep_prompt_token_indices=info['keep'],
                gold_prompt_token_indices=gold,top_fraction=0.1)) if gold else None}
        row['curves'][method]=curve
        assert curve['source_vector_sha256']==run['integration']['evaluated_vector']['actual_sha256']
        def before_metric(_module,args,kw):
            step=len(curve['input_receipts']);assert step<21 and r['scoring_forwards_entered']<42
            r['scoring_forwards_entered']+=1;x=kw['input_ids'].detach().cpu()
            assert x.shape==(1,len(ids)) and bool(kw['attention_mask'].eq(1).all())
            changed=(x[0]!=ids).nonzero().flatten().tolist();assert set(changed)<=set(info['keep'])
            assert all(int(x[0,j])==tokenizer.eos_token_id for j in changed)
            receipt={'input_sha256':sha(x.numpy().tobytes()),'deleted_positions':changed}
            assert receipt==expected['input_receipts'][step]
            curve['input_receipts'].append(receipt)
        def after_metric(_module,args,output):
            if output is None:return
            r['scoring_forwards_returned']+=1;curve['native_forwards_returned']+=1
            assert output.logits.dtype==torch.bfloat16
            curve['native_logits_dtype']=str(output.logits.dtype);curve['native_logits_shape']=list(output.logits.shape)
        def observe_metric(frame,event,arg):
            if frame.f_code is faithfulness_test_skip_tokens.__code__ and event=='return' and arg is not None:
                loc=frame.f_locals
                for name in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores']:
                    curve[name]=np.asarray(loc[name]).copy().tolist()
                curve['sorted_keep']=[int(x) for x in loc['sorted_keep']];curve['attr_sum']=float(loc['attr_sum'])
                curve['return_metrics']=[float(x) for x in arg]
        handles=[model.register_forward_pre_hook(before_metric,with_kwargs=True),model.register_forward_hook(after_metric,always_call=True)]
        assert sys.getprofile() is None;r['status']='original_production_MH_metrics_'+method;save()
        tick=time.perf_counter();sys.setprofile(observe_metric)
        try:
            with torch.no_grad():returned=timed('original_full_curve_'+method,lambda:faithfulness_test_skip_tokens(view,
                score[None],rec['prompt'],rec['target'],keep_prompt_token_indices=info['keep'],
                user_prompt_indices=list(range(info['prompt_length'])),k=20))
        finally:
            sys.setprofile(None)
            for handle in handles:handle.remove()
            handles=[];curve['wall_seconds']=time.perf_counter()-tick
        assert curve['return_metrics']==[float(x) for x in returned]
        assert curve['input_receipts']==expected['input_receipts'] and curve['sorted_keep']==expected['sorted_keep']
        assert curve['native_forwards_returned']==21
        assert all(np.isfinite(curve[name]).all() for name in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores'])
        prior=paired['cases'][key]['curves'][method]
        curve['endpoint_score_comparison']={'actual':[curve['scores'][0],curve['scores'][-1]],'prior_paired':[prior['scores'][0],prior['scores'][-1]],
            'actual_minus_prior':[curve['scores'][0]-prior['scores'][0],curve['scores'][-1]-prior['scores'][-1]]}
        curve['return_metrics_minus_paired']=[x-y for x,y in zip(curve['return_metrics'],prior['return_metrics'])]
        curve['return_metrics_minus_fixed_FT']={name:[x-y for x,y in zip(curve['return_metrics'],v['return_metrics'])]
            for name,v in row['prior_fixed_FT_metrics'].items()}
        curve['status']='original_actual_scores_complete';save();del full,score
    assert r['scoring_forwards_entered']==r['scoring_forwards_returned']==42
    r['sources_after']=sources();r['weight_stats_after']=stats()
    assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    r['native_model_source_after_sha256']=sha(Path(native.__file__).read_bytes());assert r['native_model_source_after_sha256']==p['native_model_sha256']
    r['native_FA_interface_after_sha256']=sha(Path(fa.__file__).read_bytes());assert r['native_FA_interface_after_sha256']==p['installed_FA_interface_sha256']
    assert sha(Path(p['production_vectors_path']).read_bytes())==p['production_vectors_sha256']
    r['status']='production_MH_original_metrics_complete'
except Exception:
    r['status']='failed';r['error']=traceback.format_exc()
finally:
    sys.setprofile(None);signal.alarm(0)
    for handle in handles:handle.remove()
    r['job_seconds']=time.perf_counter()-started;save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json']:
            if (A/name).exists():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'DT_calls':r['DT_calls'],'scoring_forwards_entered':r['scoring_forwards_entered'],
        'scoring_forwards_returned':r['scoring_forwards_returned'],'seconds':r['job_seconds'],'error':r.get('error')}),flush=True)
