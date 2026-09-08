"""Four normal DT calls and84 unchanged original scorer forwards; no FT call."""
import os,sys,json,time,hashlib,traceback,zipfile,signal,gc
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
r={'status':'starting','protocol':p,'model_loads':0,'native_eager_diagnostics':0,'DT_entered':0,'DT_returned':0,
   'scorer_entered':0,'scorer_returned':0,'FT_calls':0,'generation_calls':0,'cases':{},'runs':[],'calls':[]}
started=time.perf_counter();handles=[];vectors={};active=None

def save():
    f=A/'results.partial';f.write_text(json.dumps(r,indent=2,allow_nan=False));f.replace(A/'results.json')
def timeout(*args):raise TimeoutError('Frozen4DT84score all-FA pilot budget expired.')
def timed(kind,fn):
    torch.cuda.synchronize();tick=time.perf_counter();out=fn();torch.cuda.synchronize()
    r['calls'].append({'kind':kind,'seconds':time.perf_counter()-tick});return out
def sources():
    found={}
    for name,want in p['files_sha256'].items():
        found[name]=sha((A/name).read_bytes());assert found[name]==want,name
    for name,want in p['official_source_blob_sha1'].items():
        raw=(Path(p['official_root'])/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want,name
        found['official/'+name]=sha(raw)
    for name,want in p['runtime_source_sha256'].items():
        found['native/'+name]=sha((Path(p['isolated_site'])/name).read_bytes());assert found['native/'+name]==want
    for item in p['protected_sources']:
        found[item['path']]=sha(Path(item['path']).read_bytes());assert found[item['path']]==item['sha256']
    return found

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
    from qwen35_answer_finite import PackedAnswerTargets
    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    from vendor_fa_clean_secant_runner_20260908 import AllFACleanSecantBackend
    from normal_finite_study_utils_20260908 import CountFinite,check_counts,deletion_audit
    assert p['call_schedule']==[['morehopqa_1','control'],['morehopqa_1','candidate'],['niah_mq_q2_1','candidate'],['niah_mq_q2_1','control']]
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule and native.is_fast_path_available
    verify_native_sources(p['native_stage_source_sha256'])
    cp=Path(p['checkpoint'])
    for name,want in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==want
    weights=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
    r['weight_stats_before']=weights();assert r['weight_stats_before']==p['expected_weight_stats']
    tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    data={name:Path(path).read_bytes() for name,path in p['cache_paths'].items()}
    for name,raw in data.items():assert sha(raw)==p['cache_hashes'][name]
    references={}
    for key,entry in p['paired_references'].items():
        raw=Path(entry['results_path']).read_bytes();assert sha(raw)==entry['results_sha256'];references[key]=json.loads(raw)
    def prepare(dataset,index):
        rec=json.loads(data[dataset].decode().splitlines()[index]);tok=tokenizer(rec['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        keep=keep_token_indices([rec['prompt'][a:b] for a,b in tok['offset_mapping']])
        target=tokenizer(rec['target']+tokenizer.eos_token,add_special_tokens=False)['input_ids']
        ids=torch.tensor(tok['input_ids']+target,dtype=torch.long);base=ids.clone();base[keep]=tokenizer.eos_token_id
        info={'input_sha256':sha(ids.numpy().tobytes()),'prompt_length':len(tok['input_ids']),'target_length':len(target),'total_length':len(ids),'keep':keep}
        key=f'{dataset}_{index}'
        if key in references:assert info==references[key]['cases'][key]['input']
        return {'record':rec,'ids':ids,'base':base,'target':torch.tensor(target),'input':info}
    init=prepare('niah_mq_q2',0)
    cases={f'{dataset}_{index}':prepare(dataset,index) for dataset,index in p['case_indices']}
    assert torch.cuda.mem_get_info()[0]>=32*1024**3
    torch.manual_seed(73);r['status']='loading';save()
    model,loading=timed('actual_model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,
        attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True))
    assert not any(loading.values());model.eval().requires_grad_(False);r['model_loads']=1
    layers=model.model.language_model.layers
    assert [i for i,layer in enumerate(layers) if layer.block_type=='full_attention']==p['candidate_FA_layers']
    for layer in layers:
        if layer.block_type=='linear_attention':
            assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
            assert layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
    with torch.no_grad():warm=timed('original_NI0_eager_initialization',lambda:model(input_ids=init['ids'][None].to('cuda'),
        attention_mask=torch.ones_like(init['ids'][None],device='cuda'),use_cache=False))
    r['native_eager_diagnostics']=1;del warm,init;model.set_attn_implementation('flash_attention_2')
    candidate_backend=AllFACleanSecantBackend(p['candidate_library'],p['candidate_library_sha256'])
    fas={'control':CountFinite(VendorFAFiniteP1BF16D256(p['finite_FA_library'],p['finite_FA_library_sha256'])),
         'candidate':CountFinite(candidate_backend)}
    finite_fla=CountFinite(make_compiled_finite_pullback(reuse_scalar_products=False))
    runners={method:Qwen35DenseFiniteRunner(model,backend,finite_fla,norm_gate_rules={0:'symmetric'}) for method,backend in fas.items()}
    evaluator=LLMAttributionEvaluator(model,tokenizer)
    for key,case in cases.items():
        r['cases'][key]={'input':case['input'],'gold':references[key]['cases'][key]['gold'],
            'baseline_sha256':sha(case['base'].numpy().tobytes()),'curves':{},
            'prior_fixed_FT_metrics':references[key]['cases'][key]['prior_fixed_FT_metrics']}
    for number,(key,method) in enumerate(p['call_schedule']):
        case=cases[key];info=case['input'];pairs=torch.stack((case['base'],case['ids'])).to('cuda')
        selection=PackedAnswerTargets([{'target_ids':case['target'],'prompt_length':info['prompt_length']}],
            [list(range(len(case['target'])))],len(case['ids']),'cuda')
        row={'case':key,'method':method,'number':number,'status':'entered','root_forwards':0}
        r['runs'].append(row);r['status']='attribute_'+key+'_'+method;save()
        def before_root(_module,args,kw):
            assert torch.equal(kw['input_ids'],pairs) and bool(kw['attention_mask'].eq(1).all())
            row['root_forwards']+=1
        handle=model.register_forward_pre_hook(before_root,with_kwargs=True)
        old_fa=fas[method].returned;old_fla=finite_fla.returned;old_receipts=len(candidate_backend.row_receipts)
        gc.collect();torch.cuda.synchronize();r['DT_entered']+=1;tick=time.perf_counter()
        try:signed,details=runners[method].attribute(pairs,torch.ones_like(pairs),selection,select_output_rows=True,observer=None)
        finally:handle.remove();row['outer_attribute_seconds']=time.perf_counter()-tick
        r['DT_returned']+=1;assert row['root_forwards']==1
        row['counts']=check_counts(details,fas[method].returned-old_fa,finite_fla.returned-old_fla)
        assert details['norm_gate_rules']=={'0':'symmetric'}
        assert details['select_output_rows'] is True
        signed=signed[0];score=signed[:info['prompt_length']].float();assert bool(torch.isfinite(signed).all())
        row['details']=details;row['status']='complete'
        row['candidate_row_receipts']=candidate_backend.row_receipts[old_receipts:] if method=='candidate' else []
        assert len(row['candidate_row_receipts'])==(8 if method=='candidate' else 0)
        row['deletion_audit']=deletion_audit(score,case['ids'],info['keep'],tokenizer.eos_token_id)
        vectors[key+'_'+method+'_full']=signed.numpy().copy();vectors[key+'_'+method+'_evaluated']=score.numpy().copy()
        np.savez_compressed(A/'vectors.npz',**vectors);save();del signed,score,pairs,selection,details
    assert r['DT_entered']==r['DT_returned']==4
    # Original scorer forwards, with own actual vectors and masks for every curve.
    for key,method in p['call_schedule']:
        case=cases[key];info=case['input'];score=torch.from_numpy(vectors[key+'_'+method+'_evaluated'])
        run=next(x for x in r['runs'] if x['case']==key and x['method']==method)
        view=FixedInputMetricView(evaluator,case['record']['prompt'])
        assert view.compute_logprob_response_given_prompt.__func__ is LLMAttributionEvaluator.compute_logprob_response_given_prompt
        curve={'input_receipts':[],'returned_forwards':0,'needle':float(evaluate_attr_recovery_skip_tokens(score[None],
            keep_prompt_token_indices=info['keep'],gold_prompt_token_indices=r['cases'][key]['gold'],top_fraction=.1)) if r['cases'][key]['gold'] else None}
        r['cases'][key]['curves'][method]=curve
        def before_score(_module,args,kw):
            step=len(curve['input_receipts']);assert step<21 and r['scorer_entered']<84
            x=kw['input_ids'].detach().cpu();assert x.shape==(1,len(case['ids'])) and bool(kw['attention_mask'].eq(1).all())
            changed=(x[0]!=case['ids']).nonzero().flatten().tolist()
            receipt={'input_sha256':sha(x.numpy().tobytes()),'deleted_positions':changed}
            assert receipt==run['deletion_audit']['input_receipts'][step]
            curve['input_receipts'].append(receipt);r['scorer_entered']+=1
        def after_score(_module,args,output):
            if output is not None:
                assert output.logits.dtype==torch.bfloat16;r['scorer_returned']+=1;curve['returned_forwards']+=1
        def observe(frame,event,arg):
            if frame.f_code is faithfulness_test_skip_tokens.__code__ and event=='return' and arg is not None:
                loc=frame.f_locals
                for name in ('scores','density','normalized_model_response','alignment_penalty','corrected_scores'):
                    curve[name]=np.asarray(loc[name]).copy().tolist()
                curve['sorted_keep']=[int(x) for x in loc['sorted_keep']];curve['attr_sum']=float(loc['attr_sum'])
        handles=[model.register_forward_pre_hook(before_score,with_kwargs=True),model.register_forward_hook(after_score,always_call=True)]
        assert sys.getprofile() is None;r['status']='original_curve_'+key+'_'+method;save();sys.setprofile(observe)
        try:
            with torch.no_grad():returned=timed('original_curve_'+key+'_'+method,lambda:faithfulness_test_skip_tokens(view,score[None],
                case['record']['prompt'],case['record']['target'],keep_prompt_token_indices=info['keep'],user_prompt_indices=list(range(info['prompt_length'])),k=20))
        finally:
            sys.setprofile(None)
            for handle in handles:handle.remove()
            handles=[]
        curve['return_metrics']=[float(x) for x in returned]
        assert curve['returned_forwards']==21 and curve['sorted_keep']==run['deletion_audit']['sorted_keep']
        assert all(np.isfinite(curve[name]).all() for name in ('scores','density','normalized_model_response','alignment_penalty','corrected_scores','return_metrics'))
        curve['status']='complete';save()
    assert r['scorer_entered']==r['scorer_returned']==84
    for key,case in cases.items():
        row=r['cases'][key];control=row['curves']['control'];candidate=row['curves']['candidate']
        row['clean_and_allEOS_scores_equal_between_methods']=[control['scores'][i]==candidate['scores'][i] for i in (0,20)]
        row['fixed_control_masks']={'source':'Actual newly scored control21-point curve in this same execution;0additional scorers. Raw conditional prediction ledger,not another MAS curve.', 'points':[]}
        for step,receipt in enumerate(control['input_receipts']):
            actual=control['scores'][0]-control['scores'][step];deleted=receipt['deleted_positions']
            entry={'step':step,'input_receipt':receipt,'actual_logprob_drop':actual}
            for method in ('control','candidate'):
                value=float(vectors[key+'_'+method+'_evaluated'][deleted].astype(np.float64).sum())
                entry[method]={'deleted_signed_sum':value,'prediction_minus_actual':value-actual}
            row['fixed_control_masks']['points'].append(entry)
    r['finite_counts']={method:{'entered':op.entered,'returned':op.returned} for method,op in fas.items()}
    r['finite_counts']['FLA']={'entered':finite_fla.entered,'returned':finite_fla.returned}
    assert all(fas[method].entered==fas[method].returned==16 for method in fas)
    assert finite_fla.entered==finite_fla.returned==96
    r['sources_after']=sources();r['weight_stats_after']=weights()
    assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256'] and sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    r['vectors_sha256']=sha((A/'vectors.npz').read_bytes());r['status']='all8FA_clean_secant_4DT84score_pilot_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    sys.setprofile(None);signal.alarm(0)
    for handle in handles:handle.remove()
    r['seconds']=time.perf_counter()-started
    r['cost_scope']='One attribution per method/case,normal runner plus explicitly counted candidate row diagnostics;pilot cost only,not stable speed evidence. No FT calls.'
    save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/name).is_file():archive.write(A/name,name)
    print(json.dumps({'status':r['status'],'DT_entered':r['DT_entered'],'DT_returned':r['DT_returned'],
        'scorer_entered':r['scorer_entered'],'scorer_returned':r['scorer_returned'],'seconds':r['seconds'],'error':r.get('error')}),flush=True)
