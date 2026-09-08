"""Eight frozen production DT calls; no observer, extra candidate pass or score forward.

Compare both rules with their actual paired-study vectors. Exact original
deletion bins/input hashes permit replay of the unchanged metric function
with cached original score totals and NEW own attribution density, even when
coefficient magnitudes differ. That replay is not a model.
"""
import os,sys,json,time,hashlib,traceback,zipfile,signal,gc,statistics
from pathlib import Path

A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
r={'status':'starting','protocol':p,'model_loads':0,'DT_calls_entered':0,'DT_calls':0,
   'FT_calls':0,'generation_calls':0,'scoring_forwards':0,'cached_metric_replays':0,
   'cached_original_score_reads':0,'native_root_forwards':0,'cases':{},'runs':[],'calls':[]}
started=time.perf_counter();vectors={};model_handle=None;active_root=None;finite_fa=None;finite_fla=None


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
        out['native/'+name]=sha((Path(p['isolated_site'])/name).read_bytes())
        assert out['native/'+name]==want,name
    return out


def timed(kind,fn):
    torch.cuda.synchronize();tick=time.perf_counter();out=fn();torch.cuda.synchronize()
    r['calls'].append({'kind':kind,'seconds':time.perf_counter()-tick});return out


def timeout(*args):raise TimeoutError('Frozen eight-call production budget expired.')


class CountFinite:
    """Count attribution callback entries/returns; never wrap a native operator."""
    def __init__(self,operation):self.operation=operation;self.entered=0;self.returned=0
    def __call__(self,*args,**kwargs):
        self.entered+=1;out=self.operation(*args,**kwargs);self.returned+=1;return out


def vector_comparison(actual,reference):
    assert actual.shape==reference.shape and actual.dtype==reference.dtype
    delta=actual.double()-reference.double();den=float(reference.double().norm())
    return {'shape':list(actual.shape),'dtype':str(actual.dtype),
        'bitwise_equal':bool(torch.equal(actual.contiguous().view(torch.uint8),reference.contiguous().view(torch.uint8))),
        'actual_sha256':sha(actual.contiguous().numpy().tobytes()),
        'reference_sha256':sha(reference.contiguous().numpy().tobytes()),
        'difference_L2':float(delta.norm()),'relative_L2':float(delta.norm())/den if den else None,
        'max_absolute':float(delta.abs().max()),'changed_elements':int((actual!=reference).sum())}


def deletion_audit(score,ids,keep,eos):
    """Input/order audit only; the original function computes any reported metric."""
    w=score[None].sum(0);keep=sorted(set(keep));assert len(keep)>=20
    local=torch.argsort(w[torch.tensor(keep,dtype=torch.long)],descending=True)
    order=[keep[int(i)] for i in local];size,remainder=divmod(len(keep),20)
    groups=[];receipts=[];changed=set();perturbed=ids.clone()
    receipts.append({'input_sha256':sha(perturbed[None].numpy().tobytes()),'deleted_positions':[]})
    start=0
    for step in range(20):
        group=order[start:start+size+(step<remainder)];start+=len(group);groups.append(group)
        changed.update(group);perturbed[group]=eos
        actual=(perturbed!=ids).nonzero().flatten().tolist();assert actual==sorted(changed)
        receipts.append({'input_sha256':sha(perturbed[None].numpy().tobytes()),'deleted_positions':actual})
    return {'sorted_keep':order,'groups':groups,'input_receipts':receipts}


def compare_integration(signed,case,method,prior_vectors):
    key=case['key'];info=case['input'];prior=case['reference']['cases'][key]['curves'][method]
    evaluated=signed[:info['prompt_length']].float()
    full=vector_comparison(signed,torch.from_numpy(prior_vectors[key+'_'+method+'_full']))
    short=vector_comparison(evaluated,torch.from_numpy(prior_vectors[key+'_'+method+'_evaluated']))
    order=deletion_audit(evaluated,case['ids'],info['keep'],tokenizer.eos_token_id)
    order_equal=order['sorted_keep']==prior['sorted_keep']
    bins_equal=order['input_receipts']==prior['input_receipts']
    return evaluated,{'full_vector':full,'evaluated_vector':short,'deletion_audit':order,
        'sorted_keep_equal':order_equal,'all_bin_input_receipts_equal':bins_equal,
        'cached_metric_reuse_eligible':bins_equal,
        'policy':'Vector/order drift is reported, not used as an admission threshold. Identical full deletion input hashes/bins permit original-metric recomputation with own new density and cached real scores; different order within an unchanged bin is harmless. Changed bin inputs require a separately bounded scoring followup.'}


def check_counts(details,fa_delta,fla_delta):
    kinds=[q['kind'] for q in details['calls']]
    counts={'native_root':kinds.count('native_root_with_CPU_checkpoints'),
        'native_decoder_replays':sum(q.startswith('native_replay_') for q in kinds),
        'finite_decoder_calls':sum(q.startswith('finite_decoder_') for q in kinds),
        'public_FA_auxiliary_calls':sum(q.startswith('public_FA_LSE_') for q in kinds),
        'finite_FA_calls':fa_delta,'finite_FLA_calls':fla_delta}
    assert counts=={'native_root':1,'native_decoder_replays':32,'finite_decoder_calls':32,
        'public_FA_auxiliary_calls':8,'finite_FA_calls':8,'finite_FLA_calls':24},counts
    assert len(details['layers'])==32
    for q in details['layers'].values():
        assert q['decoder_calls']=={k:1 for k in ('input_norm','post_norm','gate','up','silu','down','mlp','decoder')}
        expected=({'module':1,'interface':1,'native_varlen':0,'native_dense':1}
            if q['block_type']=='full_attention' else {'module':1,'conv':1,'FLA':1,'stage':1})
        assert q['mixer_calls']==expected
    return counts


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

    class CachedOriginalScoreView(FixedInputMetricView):
        """Explicit cached aggregate reader used ONLY by original metric .sum().

        No model forward or tokenwise logprob is synthesized. Each scalar is
        a previously measured original scorer sum bound to its full input hash.
        CPU device here affects only cached metric tokenization/indexing.
        """
        def __init__(self,evaluator,prompt,curve):
            super().__init__(evaluator,prompt);self.curve=curve;self.reads=0;self.device=torch.device('cpu')
        def compute_logprob_response_given_prompt(self,prompt_ids,response_ids):
            x=torch.cat((prompt_ids,response_ids),dim=1).cpu();index=self.reads
            assert index<len(self.curve['scores'])
            assert sha(x.numpy().tobytes())==self.curve['input_receipts'][index]['input_sha256']
            self.reads+=1;r['cached_original_score_reads']+=1
            return torch.tensor([self.curve['scores'][index]],dtype=torch.float64)

    assert p['case_indices']==[['niah_mq_q2',1],['morehopqa',1]]
    assert p['production_norm_gate_rules']=={'current':{},'candidate':{'0':'symmetric'}}
    assert p['call_schedule']==[['niah_mq_q2_1','current','initial'],['niah_mq_q2_1','candidate','initial'],
        ['niah_mq_q2_1','current','measured'],['niah_mq_q2_1','candidate','measured'],
        ['niah_mq_q2_1','candidate','measured'],['niah_mq_q2_1','current','measured'],
        ['morehopqa_1','current','integration'],['morehopqa_1','candidate','integration']]
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule and native.is_fast_path_available
    verify_native_sources(p['native_stage_source_sha256'])
    cp=Path(p['checkpoint'])
    for name,want in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==want
    stats=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats();assert r['weight_stats_before']==p['expected_weight_stats']
    tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    data={k:Path(v).read_bytes() for k,v in p['cache_paths'].items()}
    for k,raw in data.items():assert sha(raw)==p['cache_hashes'][k]
    references={};prior_vectors={}
    for key,ref in p['paired_references'].items():
        raw=Path(ref['results_path']).read_bytes();assert sha(raw)==ref['results_sha256']
        references[key]=json.loads(raw)
        raw=Path(ref['vectors_path']).read_bytes();assert sha(raw)==ref['vectors_sha256']
        with np.load(ref['vectors_path'],allow_pickle=False) as archive:
            prior_vectors[key]={name:archive[name].copy() for name in archive.files}

    def prepare(dataset,index,reference=None):
        rec=json.loads(data[dataset].decode().splitlines()[index]);tok=tokenizer(rec['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        keep=keep_token_indices([rec['prompt'][a:b] for a,b in tok['offset_mapping']])
        target=tokenizer(rec['target']+tokenizer.eos_token,add_special_tokens=False)['input_ids']
        ids=torch.tensor(tok['input_ids']+target,dtype=torch.long);base=ids.clone();base[keep]=tokenizer.eos_token_id
        key=f'{dataset}_{index}';info={'input_sha256':sha(ids.numpy().tobytes()),'prompt_length':len(tok['input_ids']),
            'target_length':len(target),'total_length':len(ids),'keep':keep}
        if reference is not None:assert info==reference['cases'][key]['input'],key
        return {'key':key,'record':rec,'ids':ids,'base':base,'target':torch.tensor(target),'input':info,'reference':reference}

    control=prepare('niah_mq_q2',0)
    cases={f'{dataset}_{index}':prepare(dataset,index,references[f'{dataset}_{index}']) for dataset,index in p['case_indices']}
    assert torch.cuda.mem_get_info()[0]>=32*1024**3
    torch.manual_seed(73);r['status']='loading';save()
    model,loading=timed('actual_model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,
        attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True))
    assert not any(loading.values());model.eval().requires_grad_(False);r['model_loads']=1
    for layer in model.model.language_model.layers:
        if layer.block_type=='linear_attention':
            assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
            assert layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
    with torch.no_grad():warm=timed('original_order_native_eager_B1_diagnostic',lambda:model(input_ids=control['ids'][None].to('cuda'),
        attention_mask=torch.ones_like(control['ids'][None],device='cuda'),use_cache=False))
    del warm,control;model.set_attn_implementation('flash_attention_2')
    finite_fa=CountFinite(VendorFAFiniteP1BF16D256(p['finite_FA_library'],p['finite_FA_library_sha256']))
    finite_fla=CountFinite(make_compiled_finite_pullback(reuse_scalar_products=False))
    runners={'current':Qwen35DenseFiniteRunner(model,finite_fa,finite_fla),
        'candidate':Qwen35DenseFiniteRunner(model,finite_fa,finite_fla,norm_gate_rules={0:'symmetric'})}
    evaluator=LLMAttributionEvaluator(model,tokenizer)

    def root_receipt(_module,args,kw):
        assert active_root is not None,'Unexpected model forward outside the eight production DT calls.'
        x=kw['input_ids'].detach().cpu();assert bool(kw['attention_mask'].eq(1).all()) and kw['use_cache'] is False
        assert sha(x.numpy().tobytes())==active_root['expected_paired_input_sha256']
        active_root['native_root_forwards']+=1;r['native_root_forwards']+=1

    model_handle=model.register_forward_pre_hook(root_receipt,with_kwargs=True)
    for key,case in cases.items():
        r['cases'][key]={'input':case['input'],'baseline_sha256':sha(case['base'].numpy().tobytes()),
            'paired_reference':p['paired_references'][key],'cached_metrics':{}}

    for number,(key,method,phase) in enumerate(p['call_schedule']):
        case=cases[key];info=case['input']
        pairs=torch.stack((case['base'],case['ids'])).to('cuda');mask=torch.ones_like(pairs)
        selection=PackedAnswerTargets([{'target_ids':case['target'],'prompt_length':info['prompt_length']}],
            [list(range(len(case['target'])))],len(case['ids']),'cuda')
        gc.collect();torch.cuda.synchronize()
        row={'number':number,'case':key,'method':method,'phase':phase,'native_root_forwards':0,'status':'attribute_entered',
            'expected_paired_input_sha256':sha(torch.stack((case['base'],case['ids'])).numpy().tobytes()),
            'GPU_allocated_before':torch.cuda.memory_allocated(),'GPU_reserved_before':torch.cuda.memory_reserved()}
        r['runs'].append(row);active_root=row;r['status']='production_'+str(number)+'_'+key+'_'+method;save()
        fa_before=finite_fa.returned;fla_before=finite_fla.returned;r['DT_calls_entered']+=1
        before_counts={name:(op.entered,op.returned) for name,op in [('finite_FA',finite_fa),('finite_FLA',finite_fla)]}
        row['native_ledger_status']='unknown_until_attribute_returns_and_counts_are_checked'
        tick=time.perf_counter()
        try:
            signed,details=runners[method].attribute(pairs,mask,selection,select_output_rows=True,observer=None)
            row['production_details']=details
            torch.cuda.synchronize();r['DT_calls']+=1;row['status']='attribute_returned'
        except Exception:
            row['status']='attribute_failed';raise
        finally:
            row['outer_attribute_seconds']=time.perf_counter()-tick
            row['finite_callback_counts']={name:{'entered':op.entered-before_counts[name][0],
                'returned':op.returned-before_counts[name][1]} for name,op in [('finite_FA',finite_fa),('finite_FLA',finite_fla)]}
            if row['status']=='attribute_failed':
                row['unfinished_native_work']={'native_decoder_replays':None,'finite_decoder_calls':None,'public_FA_auxiliary_calls':None,
                    'meaning':'Unreturned attribute has no completed native ledger; partial internal work is unknown, not zero.'}
        active_root=None;assert row['native_root_forwards']==1
        assert details['norm_gate_rules']==p['production_norm_gate_rules'][method]
        row['counts']=check_counts(details,finite_fa.returned-fa_before,finite_fla.returned-fla_before)
        row['native_ledger_status']='completed_and_checked'
        assert finite_fa.entered==finite_fa.returned and finite_fla.entered==finite_fla.returned
        signed=signed[0];assert bool(torch.isfinite(signed).all())
        evaluated,comparison=compare_integration(signed,case,method,prior_vectors[key]);row['integration']=comparison
        reference_details=case['reference']['cases'][key]['DT_with_paired_diagnostics']
        row['paired_root_output_comparison']={name:vector_comparison(torch.tensor(details[name],dtype=torch.float64),
            torch.tensor(reference_details[name],dtype=torch.float64)) for name in ['target_logp0','target_logp1']}
        row['root_effect_difference_from_paired']=details['root_effect']-reference_details['root_effect']
        vector_key=key+'_'+method+'_run'+str(number)
        vectors[vector_key+'_full']=signed.numpy().copy();vectors[vector_key+'_evaluated']=evaluated.numpy().copy()
        np.savez_compressed(A/'vectors.npz',**vectors)
        del signed,evaluated,pairs,mask,selection,details
        gc.collect();torch.cuda.synchronize();row['GPU_allocated_after_cleanup']=torch.cuda.memory_allocated();row['status']='complete'
        save()

    # Cached metric work occurs after all production timing, without a model pass.
    # The still-active model guard rejects an accidental evaluator forward.
    for key,case in cases.items():
        for method in ['current','candidate']:
            matching=[run for run in r['runs'] if run['case']==key and run['method']==method]
            curve=case['reference']['cases'][key]['curves'][method]
            summary={'production_runs':{},'prior_paired_return_metrics':curve['return_metrics'],
                'status':'all_production_runs_have_cached_original_metric_replay','native_scoring_forwards':0}
            r['cases'][key]['cached_metrics'][method]=summary
            for run in matching:
                eligible=run['integration']['cached_metric_reuse_eligible']
                out={'status':'cached_replay_pending' if eligible else 'bounded_original_metric_followup_needed',
                    'all_original_bin_input_hashes_match':eligible,'native_scoring_forwards':0}
                summary['production_runs'][str(run['number'])]=out
                if not eligible:
                    summary['status']='some_production_runs_need_bounded_original_metric_followup'
                    continue
                score=torch.from_numpy(vectors[key+'_'+method+'_run'+str(run['number'])+'_evaluated'])
                view=CachedOriginalScoreView(evaluator,case['record']['prompt'],curve)
                def observe_metric(frame,event,arg):
                    if frame.f_code is faithfulness_test_skip_tokens.__code__ and event=='return' and arg is not None:
                        loc=frame.f_locals
                        for name in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores']:
                            out[name]=np.asarray(loc[name]).copy().tolist()
                        out['sorted_keep']=[int(x) for x in loc['sorted_keep']];out['attr_sum']=float(loc['attr_sum'])
                assert sys.getprofile() is None;tick=time.perf_counter();sys.setprofile(observe_metric)
                try:
                    returned=faithfulness_test_skip_tokens(view,score[None],case['record']['prompt'],case['record']['target'],
                        keep_prompt_token_indices=case['input']['keep'],user_prompt_indices=list(range(case['input']['prompt_length'])),k=20)
                finally:sys.setprofile(None)
                out['CPU_cached_metric_wall_seconds']=time.perf_counter()-tick
                out['return_metrics']=[float(x) for x in returned];out['cached_original_score_reads']=view.reads
                assert view.reads==21 and out['sorted_keep']==run['integration']['deletion_audit']['sorted_keep']
                assert out['scores']==curve['scores']
                out['return_metrics_minus_paired']=[x-y for x,y in zip(out['return_metrics'],curve['return_metrics'])]
                out['density_minus_paired']=[x-y for x,y in zip(out['density'],curve['density'])]
                assert all(np.isfinite(out[name]).all() for name in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores','return_metrics'])
                gold=case['reference']['cases'][key]['gold']
                out['needle']=float(evaluate_attr_recovery_skip_tokens(score[None],keep_prompt_token_indices=case['input']['keep'],
                    gold_prompt_token_indices=gold,top_fraction=0.1)) if gold else None
                out['needle_minus_paired']=out['needle']-curve['needle'] if out['needle'] is not None and curve['needle'] is not None else None
                out['status']='original_function_replayed_with_new_density_and_hash_verified_cached_scores';r['cached_metric_replays']+=1
                del score,view;save()

    r['NI_measured_cost']={}
    for method in ['current','candidate']:
        runs=[q for q in r['runs'] if q['case']=='niah_mq_q2_1' and q['phase']=='measured' and q['method']==method]
        assert len(runs)==2
        r['NI_measured_cost'][method]={'samples':2,'run_numbers':[q['number'] for q in runs],
            'complete_attribute_seconds':[q['production_details']['complete_attribution_seconds_with_diagnostics'] for q in runs],
            'median_complete_attribute_seconds':statistics.median(q['production_details']['complete_attribution_seconds_with_diagnostics'] for q in runs),
            'peak_allocated_bytes':[q['production_details']['peak_allocated'] for q in runs],
            'median_peak_allocated_bytes':statistics.median(q['production_details']['peak_allocated'] for q in runs),
            'GPU_allocated_before_bytes':[q['GPU_allocated_before'] for q in runs]}
    r['NI_measured_cost']['candidate_over_current_latency_ratio']=(r['NI_measured_cost']['candidate']['median_complete_attribute_seconds']
        /r['NI_measured_cost']['current']['median_complete_attribute_seconds'])
    r['initial_call_cost']=[{'run_number':q['number'],'case':q['case'],'method':q['method'],
        'complete_attribute_seconds':q['production_details']['complete_attribution_seconds_with_diagnostics'],
        'peak_allocated_bytes':q['production_details']['peak_allocated']}
        for q in r['runs'] if q['phase']!='measured']
    r['compiler_context']='Frozen pre-existing Triton/Inductor cache paths reused. Initial means first call per rule/shape in this process, not a claim of an empty compiler cache.'
    r['cost_scope']='Complete existing native FA/FLA attribution path including native root, CPU checkpoints/capture,32 replays,32 finite decoders,8 auxiliary FA calls and input receipt hook. No extra observer/candidate propagation or L/r0 capture. Initial NI C/S and MH shape-first calls are separate. Two measured samples per method are descriptive, not a speed guarantee.'
    assert r['DT_calls_entered']==r['DT_calls']==r['native_root_forwards']==8
    assert finite_fla.returned==192 and finite_fa.returned==64
    assert sum(q['counts']['native_decoder_replays'] for q in r['runs'])==256
    assert sum(q['counts']['public_FA_auxiliary_calls'] for q in r['runs'])==64
    r['actual_call_totals']={name:sum(q['counts'][name] for q in r['runs']) for name in r['runs'][0]['counts']}
    r['sources_after']=sources();r['weight_stats_after']=stats()
    assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    r['native_model_source_after_sha256']=sha(Path(native.__file__).read_bytes())
    r['native_FA_interface_after_sha256']=sha(Path(fa.__file__).read_bytes())
    assert r['native_model_source_after_sha256']==p['native_model_sha256']
    assert r['native_FA_interface_after_sha256']==p['installed_FA_interface_sha256']
    assert r['cached_metric_replays']<=8 and r['cached_original_score_reads']<=168
    r['status']='eight_call_production_integration_cost_complete'
except Exception:
    r['status']='failed';r['error']=traceback.format_exc()
finally:
    if model_handle is not None:model_handle.remove()
    sys.setprofile(None);signal.alarm(0)
    r['finite_callback_counts']={name:{'constructed':operation is not None,
        'entered':operation.entered if operation is not None else 0,
        'returned':operation.returned if operation is not None else 0}
        for name,operation in [('finite_FA',finite_fa),('finite_FLA',finite_fla)]}
    r['count_scope']='Entered/returned attribution callbacks are recorded even on failure. Per-run native replay/auxiliary ledgers are available only for completed attribute calls; unfinished internal work is not claimed zero.'
    r['job_seconds']=time.perf_counter()-started;save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json','vectors.npz']:
            if (A/name).exists():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'DT_calls':r['DT_calls'],'scoring_forwards':r['scoring_forwards'],
        'seconds':r['job_seconds'],'error':r.get('error')}),flush=True)
