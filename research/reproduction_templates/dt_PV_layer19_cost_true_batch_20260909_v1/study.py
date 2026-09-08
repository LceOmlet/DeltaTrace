"""Same-shape C/P0 warm costs and actual NI1+46 batch wiring;12DT300FLA,0scorers."""
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
fas={};finite_fla=None;candidate_backend=None

def save():
    f=A/'results.partial';f.write_text(json.dumps(r,indent=2,allow_nan=False));f.replace(A/'results.json')
def timeout(*args):raise TimeoutError('Frozen12DT300FLA C/P0 cost and true batch budget expired.')
def timed(kind,fn):
    torch.cuda.synchronize();tick=time.perf_counter();row={'kind':kind,'status':'entered'};r['calls'].append(row)
    try:
        out=fn();torch.cuda.synchronize();row['status']='returned';return out
    finally:row['seconds']=time.perf_counter()-tick
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
    for name,want in p['span_source_sha256'].items():
        found['author_spans/'+name]=sha((Path(p['author_data_root'])/name).read_bytes().replace(b'\r\n',b'\n'));assert found['author_spans/'+name]==want
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
    from flashtrace.improved import keep_token_indices,evaluate_attr_recovery_skip_tokens
    from official_span_mapping import load_author_span_helpers
    from qwen35_answer_finite import PackedAnswerTargets
    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    from layer0_fla_endpoint_average_20260908 import Layer0FLAEndpointAverage
    from normal_finite_study_utils_20260908 import CountFinite,check_counts,deletion_audit
    class BudgetedFinite(CountFinite):
        def __init__(self,operation,limit):super().__init__(operation);self.limit=limit
        def __call__(self,*args,**kwargs):
            assert self.entered<self.limit,'Frozen finite-backend budget exhausted.'
            return super().__call__(*args,**kwargs)
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule and native.is_fast_path_available
    verify_native_sources(p['native_stage_source_sha256'])
    cp=Path(p['checkpoint'])
    for name,want in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==want
    weights=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
    r['weight_stats_before']=weights();assert r['weight_stats_before']==p['expected_weight_stats']
    tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    old_tokenizer=AutoTokenizer.from_pretrained(p['old_checkpoint'],local_files_only=True);old_tokenizer.pad_token=old_tokenizer.eos_token
    helpers=load_author_span_helpers(p['author_data_root'],p['span_source_sha256'])
    data={name:Path(path).read_bytes() for name,path in p['cache_paths'].items()}
    for name,raw in data.items():assert sha(raw)==p['cache_hashes'][name]
    def prepare(dataset,index):
        line=data[dataset].decode().splitlines()[index];rec=json.loads(line);key=f'{dataset}_{index}'
        assert sha(line.encode())==p['fixed_records'][key]['source_record_sha256']
        cls=helpers['CachedExample'];fields=['prompt','target','indices_to_explain','attr_mask_indices','sink_span','thinking_span','metadata']
        cached=cls(**{name:rec.get(name) for name in fields})
        def remap(tok):
            fresh=cls(prompt=cached.prompt,target=cached.target,indices_to_explain=None,attr_mask_indices=cached.attr_mask_indices,
                sink_span=None,thinking_span=None,metadata=dict(cached.metadata))
            fresh=helpers['attach_spans_from_answer'](fresh,tok);fresh.indices_to_explain=list(fresh.sink_span);return fresh
        old,new=remap(old_tokenizer),remap(tokenizer)
        for name in ('sink_span','thinking_span','indices_to_explain'):assert getattr(old,name)==getattr(cached,name),name
        tok=tokenizer(rec['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        labels=[rec['prompt'][a:b] for a,b in tok['offset_mapping']]
        keep=keep_token_indices(labels);assert keep==helpers['keep_token_indices'](labels)
        target=tokenizer(rec['target']+tokenizer.eos_token,add_special_tokens=False)['input_ids']
        assert target==tokenizer(rec['target'],add_special_tokens=False)['input_ids']+[tokenizer.eos_token_id]
        ids=torch.tensor(tok['input_ids']+target,dtype=torch.long);base=ids.clone();base[keep]=tokenizer.eos_token_id
        assert len(ids)<=p['max_total_tokens'] and len(keep)>=20 and (base!=ids).nonzero().flatten().tolist()==keep
        gold=helpers['ruler_gold_prompt_token_indices'](new,tokenizer)
        info={'input_sha256':sha(ids.numpy().tobytes()),'prompt_length':len(tok['input_ids']),'target_length':len(target),'total_length':len(ids),'keep':keep}
        ref=p['fixed_records'][key]
        if ref.get('expected_input') is not None:assert info==ref['expected_input']
        if ref.get('expected_lengths') is not None:assert {k:info[k] for k in ['total_length','prompt_length','target_length']}==ref['expected_lengths']
        if ref.get('expected_gold') is not None:assert gold==ref['expected_gold']
        mapping={'source_record_sha256':sha(line.encode()),'original_cached_spans_reproduced':True,
            'original_sink_span':old.sink_span,'new_sink_span':new.sink_span,'new_thinking_span':new.thinking_span,
            'gold':gold,'gold_function':'unchanged author ruler_gold_prompt_token_indices',
            'input_scope':'Original author prompt plus fixed original target and EOS;no additional chat template;same wholepilot pipeline.',
            'historical_input_hash_available':ref.get('expected_input') is not None}
        return {'record':rec,'ids':ids,'base':base,'target':torch.tensor(target),'input':info,'gold':gold,'mapping':mapping}
    cases={f'{dataset}_{index}':prepare(dataset,index) for dataset,index in p['case_indices']}
    init=cases['niah_mq_q2_0']
    r['input_freeze_before_model_load']={key:dict(input=case['input'],mapping=case['mapping'],
        input_ids=case['ids'].tolist(),target_ids=case['target'].tolist(),baseline_sha256=sha(case['base'].numpy().tobytes())) for key,case in cases.items()}
    save()
    assert torch.cuda.mem_get_info()[0]>=32*1024**3
    torch.manual_seed(73);r['status']='loading';save()
    model,loading=timed('actual_model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,
        attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True))
    assert not any(loading.values());model.eval().requires_grad_(False);r['model_loads']=1
    layers=model.model.language_model.layers
    assert [i for i,layer in enumerate(layers) if layer.block_type=='full_attention']==p['expected_FA_layers']
    for layer in layers:
        if layer.block_type=='linear_attention':
            assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
            assert layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
    with torch.no_grad():warm=timed('original_NI0_eager_initialization',lambda:model(input_ids=init['ids'][None].to('cuda'),
        attention_mask=torch.ones_like(init['ids'][None],device='cuda'),use_cache=False))
    r['native_eager_diagnostics']=1;del warm,init;model.set_attn_implementation('flash_attention_2')
    original_fa=VendorFAFiniteP1BF16D256(p['finite_FA_library'],p['finite_FA_library_sha256'])
    fas={method:BudgetedFinite(original_fa,48) for method in ('control','candidate')}
    finite_fla=BudgetedFinite(make_compiled_finite_pullback(reuse_scalar_products=False),300)
    candidate_backend=Layer0FLAEndpointAverage(finite_fla)
    runners={method:Qwen35DenseFiniteRunner(model,fas[method],finite_fla,norm_gate_rules={0:'symmetric'},
        finite_fla_by_layer={0:candidate_backend},attention_pv_rules={} if method=='control' else {19:'content0'})
        for method in fas}
    for key,case in cases.items():
        r['cases'][key]={'input':case['input'],'gold':case['gold'],'mapping':case['mapping'],
            'baseline_sha256':sha(case['base'].numpy().tobytes()),'curves':{},
            'role':'initialization_only' if key=='niah_mq_q2_0' else 'fixed_distinct_batch_case'}
    keys=['niah_mq_q2_1','niah_mq_q2_46']
    assert p['batch_cases']==keys and not torch.equal(cases[keys[0]]['ids'],cases[keys[1]]['ids'])
    assert {cases[k]['input']['total_length'] for k in keys}=={1201}
    expected_schedule=[[mode,method,phase] for mode in ('single_NI1','batch2_NI1_46')
        for phase,methods in [('warm',('control','candidate')),('measured',('control','candidate','candidate','control'))]
        for method in methods]
    assert p['run_schedule']==expected_schedule
    r['groups']=[]

    def invoke(selected,group_number,mode,phase,method):
        number=len(r['runs']);chosen=[cases[k] for k in selected];b=len(chosen);length=len(chosen[0]['ids'])
        assert len(set(selected))==b and all(len(c['ids'])==length for c in chosen)
        gc.collect();torch.cuda.synchronize();resident=torch.cuda.memory_allocated()
        pairs=torch.stack([x for case in chosen for x in (case['base'],case['ids'])]).to('cuda')
        selection=PackedAnswerTargets([{'target_ids':c['target'],'prompt_length':c['input']['prompt_length']} for c in chosen],
            [list(range(len(c['target']))) for c in chosen],length,'cuda')
        assert selection.batch==b and pairs.shape==(2*b,length)
        row={'number':number,'group':group_number,'mode':mode,'phase':phase,'method':method,'samples':selected,
            'example_batch_size':b,'physical_endpoint_batch_size':2*b,'status':'entered','root_forwards':0,
            'GPU_allocated_before_pair':resident,'sample_records':[]}
        r['runs'].append(row);r['status']='attribute_'+str(number);save()
        def before_root(_module,args,kw):
            assert torch.equal(kw['input_ids'],pairs) and bool(kw['attention_mask'].eq(1).all())
            assert kw.get('use_cache') is False
            row['root_forwards']+=1
            row['root_endpoint_input_sha256']=[sha(x.detach().cpu().numpy().tobytes()) for x in kw['input_ids']]
        handle=model.register_forward_pre_hook(before_root,with_kwargs=True)
        oldfa=(fas[method].entered,fas[method].returned);oldfla=(finite_fla.entered,finite_fla.returned)
        oldwrapper=len(candidate_backend.calls)
        gc.collect();torch.cuda.synchronize();row['GPU_allocated_before_attribute']=torch.cuda.memory_allocated()
        row['GPU_reserved_before_attribute']=torch.cuda.memory_reserved()
        assert r['DT_entered']<12;r['DT_entered']+=1;tick=time.perf_counter();details=None
        try:
            signed,details=runners[method].attribute(pairs,torch.ones_like(pairs),selection,select_output_rows=True,observer=None)
            r['DT_returned']+=1;row['details']=details
        finally:
            handle.remove()
            try:torch.cuda.synchronize()
            except Exception:row['final_sync_error']=traceback.format_exc()
            row['outer_attribute_seconds']=time.perf_counter()-tick
            row['actual_GPU_peak_allocated_after_attribute']=torch.cuda.max_memory_allocated()
            row['actual_GPU_peak_reserved_after_attribute']=torch.cuda.max_memory_reserved()
            row['actual_GPU_allocated_after_attribute']=torch.cuda.memory_allocated()
            row['actual_GPU_reserved_after_attribute']=torch.cuda.memory_reserved()
            row['finite_callback_counts']={'FA_entered':fas[method].entered-oldfa[0],
                'FA_returned':fas[method].returned-oldfa[1],
                'FLA_entered':finite_fla.entered-oldfla[0],'FLA_returned':finite_fla.returned-oldfla[1]}
            row['wrapper_receipts']=candidate_backend.calls[oldwrapper:]
            row['native_stage_accounting']={'returned_backend_times_two':2*(finite_fla.returned-oldfla[1]),
                'nonreturned_backend_stage_count':'unknown' if finite_fla.entered-oldfla[0]!=finite_fla.returned-oldfla[1] else 0}
            if details is None:row['status']='failed_attribute';row['native_runner_ledger']='unknown: no returned details'
            save()
        assert row['root_forwards']==1 and signed.shape==(b,length) and bool(torch.isfinite(signed).all())
        row['counts']=check_counts(details,fas[method].returned-oldfa[1],finite_fla.returned-oldfla[1]-1)
        row['counts']['finite_FLA_backend_calls']=finite_fla.returned-oldfla[1]
        assert row['counts']['finite_FLA_backend_calls']==25
        assert details['norm_gate_rules']=={'0':'symmetric'} and details['finite_fla_by_layer']==[0]
        assert details['attention_pv_rules']==({} if method=='control' else {'19':'content0'})
        assert len(row['wrapper_receipts'])==1
        wrap=row['wrapper_receipts'][0]
        assert wrap['status']=='returned' and wrap['endpoint_permutation']==[i^1 for i in range(2*b)]
        assert [c['orientation'] for c in wrap['backend_calls']]==['original','swapped']
        assert all(c['status']=='returned' for c in wrap['backend_calls'])
        assert details['actual_head_input_shapes']==[[2*b,len(set(selection.positions.cpu().tolist())),model.lm_head.in_features]]
        position=0
        for i,(key,case) in enumerate(zip(selected,chosen)):
            info=case['input'];count=len(case['target']);full=signed[i];score=full[:info['prompt_length']].float()
            assert row['root_endpoint_input_sha256'][2*i:2*i+2]==[sha(case['base'].numpy().tobytes()),info['input_sha256']]
            assert selection.counts[i]==count
            assert selection.labels[position:position+count].cpu().tolist()==case['target'].tolist()
            assert selection.samples[position:position+count].eq(i).all()
            l0=np.asarray(details['target_logp0'][position:position+count],dtype=np.float64)
            l1=np.asarray(details['target_logp1'][position:position+count],dtype=np.float64)
            assert len(l0)==len(l1)==count;position+=count
            vk='run'+str(number)+'_'+key
            vectors[vk+'_full']=full.numpy().copy();vectors[vk+'_evaluated']=score.numpy().copy()
            row['sample_records'].append({'case':key,'vector_key':vk,'root_logp0':l0.tolist(),'root_logp1':l1.tolist(),
                'root_effect':float((l1-l0).sum()),'signed_sum':float(full.double().sum()),
                'positive':float(full[full>0].double().sum()),'negative':float(full[full<0].double().sum()),
                'needle':float(evaluate_attr_recovery_skip_tokens(score[None],keep_prompt_token_indices=info['keep'],
                    gold_prompt_token_indices=case['gold'],top_fraction=.1)) if case['gold'] else None,
                'deletion_audit':deletion_audit(score,case['ids'],info['keep'],tokenizer.eos_token_id)})
        assert position==len(details['target_logp0'])==len(details['target_logp1'])
        assert abs(sum(x['root_effect'] for x in row['sample_records'])-details['root_effect'])<1e-7
        row['memory_cost']={'peak_allocated':details['peak_allocated'],'peak_reserved':details['peak_reserved'],
            'peak_minus_before_pair':details['peak_allocated']-resident,
            'peak_minus_before_attribute':details['peak_allocated']-row['GPU_allocated_before_attribute']}
        row['status']='complete';np.savez_compressed(A/'vectors.npz',**vectors)
        del signed,pairs,selection,details,full,score
        gc.collect();torch.cuda.synchronize();row['GPU_allocated_after_cleanup']=torch.cuda.memory_allocated()
        row['GPU_reserved_after_cleanup']=torch.cuda.memory_reserved();save()
        return number

    for group_number,(mode,method,phase) in enumerate(p['run_schedule']):
        selected=keys[:1] if mode=='single_NI1' else keys
        group={'number':group_number,'mode':mode,'phase':phase,'method':method,'samples':selected,
            'status':'entered','invocations':[]}
        r['groups'].append(group);group_tick=time.perf_counter()
        try:
            group['invocations'].append(invoke(selected,group_number,mode,phase,method))
            group['status']='complete'
        finally:group['wall_seconds_including_audit_and_setup']=time.perf_counter()-group_tick;save()
    assert r['DT_entered']==r['DT_returned']==12 and sum(x['example_batch_size'] for x in r['runs'])==18
    assert all(op.entered==op.returned==48 for op in fas.values())
    assert finite_fla.entered==finite_fla.returned==300 and candidate_backend.entered==candidate_backend.returned==12

    def drift(now,reference):
        a=np.asarray(now,dtype=np.float64);ref=np.asarray(reference,dtype=np.float64);d=a-ref;den=float(np.linalg.norm(ref))
        return {'bitwise_equal':bool(np.array_equal(a,ref)),'relative_L2':float(np.linalg.norm(d)/den) if den else None,
            'max_absolute':float(np.abs(d).max(initial=0)),'signed_sum_difference':float(d.sum())}
    def comparison(row,record,reference,scope):
        case=cases[record['case']];keep=case['input']['keep'];w=vectors[record['vector_key']+'_evaluated']
        wref=vectors[reference['vector_key']+'_evaluated'];changed=np.sign(w[keep])!=np.sign(wref[keep])
        order=record['deletion_audit']['sorted_keep'];oldorder=reference['deletion_audit']['sorted_keep']
        rank={token:i for i,token in enumerate(order)};oldrank={token:i for i,token in enumerate(oldorder)}
        return {'run':row['number'],'mode':row['mode'],'phase':row['phase'],'method':row['method'],'case':record['case'],
            'reference_vector_key':reference['vector_key'],'reference_scope':scope,
            'full':drift(vectors[record['vector_key']+'_full'],vectors[reference['vector_key']+'_full']),
            'evaluated':drift(w,wref),'root_logp0':drift(record['root_logp0'],reference['root_logp0']),
            'root_logp1':drift(record['root_logp1'],reference['root_logp1']),
            'root_effect_difference':record['root_effect']-reference['root_effect'],
            'needle':record['needle'],'needle_difference':None if record['needle'] is None else record['needle']-reference['needle'],
            'eligible_sign_changes':int(changed.sum()),'reference_absolute_mass_on_sign_changes':float(np.abs(wref[keep][changed].astype(np.float64)).sum()),
            'sorted_keep_equal':order==oldorder,'maximum_rank_displacement':max(abs(rank[k]-oldrank[k]) for k in keep),
            'same_deletion_input_count':sum(a==b for a,b in zip(record['deletion_audit']['input_receipts'],reference['deletion_audit']['input_receipts']))}
    r['repeat_comparisons']=[];r['method_comparisons']=[];r['NI1_cross_shape_comparisons']=[]
    for row in r['runs']:
        for record in row['sample_records']:
            ref=next(x for run in r['runs'] if run['mode']==row['mode'] and run['method']==row['method'] and run['phase']=='warm'
                for x in run['sample_records'] if x['case']==record['case'])
            r['repeat_comparisons'].append(comparison(row,record,ref,'Same-shape same-method same-sample warm reference; repeat drift only.'))
            if row['method']=='candidate':
                ref=next(x for run in r['runs'] if run['mode']==row['mode'] and run['method']=='control' and run['phase']=='warm'
                    for x in run['sample_records'] if x['case']==record['case'])
                r['method_comparisons'].append(comparison(row,record,ref,'Same-shape same-sample control warm reference; method plus execution drift, not a new scored quality metric.'))
            if row['mode']=='batch2_NI1_46' and record['case']=='niah_mq_q2_1':
                ref=next(x for run in r['runs'] if run['mode']=='single_NI1' and run['method']==row['method'] and run['phase']=='warm'
                    for x in run['sample_records'] if x['case']==record['case'])
                r['NI1_cross_shape_comparisons'].append(comparison(row,record,ref,'Same-process NI1 singleton versus NI1 batch row for the same method. No NI46 singleton is run in this budget.'))
    r['cost_summary']={}
    for mode in ('single_NI1','batch2_NI1_46'):
        summary={}
        for method in ('control','candidate'):
            rows=[row for row in r['runs'] if row['mode']==mode and row['method']==method and row['phase']=='measured']
            assert len(rows)==2
            times=[row['outer_attribute_seconds'] for row in rows]
            summary[method]={'runs':[row['number'] for row in rows],'seconds':times,
                'mean_seconds':float(np.mean(times)),'median_seconds':float(np.median(times)),
                'runner_seconds':[row['details']['complete_attribution_seconds_with_diagnostics'] for row in rows],
                'peak_allocated':[row['details']['peak_allocated'] for row in rows],
                'peak_reserved':[row['details']['peak_reserved'] for row in rows],
                'resident_before_attribute':[row['GPU_allocated_before_attribute'] for row in rows],
                'incremental_peak_from_before_attribute':[row['memory_cost']['peak_minus_before_attribute'] for row in rows],
                'resident_after_cleanup':[row['GPU_allocated_after_cleanup'] for row in rows]}
        summary['candidate_over_control_median_seconds']=summary['candidate']['median_seconds']/summary['control']['median_seconds']
        summary['candidate_minus_control_max_peak_allocated']=max(summary['candidate']['peak_allocated'])-max(summary['control']['peak_allocated'])
        summary['scope']='Only C versus P0 at this identical shape and sample set; two measured observations per method, warm excluded. No speed guarantee.'
        r['cost_summary'][mode]=summary
    assert r['scorer_entered']==r['scorer_returned']==r['FT_calls']==r['generation_calls']==0
    r['sources_after']=sources();r['weight_stats_after']=weights()
    assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    r['vectors_sha256']=sha((A/'vectors.npz').read_bytes());r['status']='PV19_same_shape_cost_true_NI1_46_batch_12DT300FLA_no_scores_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    sys.setprofile(None);signal.alarm(0)
    for handle in handles:handle.remove()
    r['seconds']=time.perf_counter()-started
    r['finite_counts']={name:{'entered':op.entered,'returned':op.returned} for name,op in fas.items()}
    if finite_fla is not None:r['finite_counts']['FLA_backend']={'entered':finite_fla.entered,'returned':finite_fla.returned,
        'native_stages_from_returned_calls':2*finite_fla.returned,
        'inflight_native_stages':'unknown' if finite_fla.entered!=finite_fla.returned else 0}
    if candidate_backend is not None:r['finite_counts']['endpoint_average_wrapper']={
        'entered':candidate_backend.entered,'returned':candidate_backend.returned,'calls':candidate_backend.calls}
    r['cost_scope']='Two independent shapes: NI1 singleton (2 endpoint rows) and NI1+NI46 distinct batch (4 rows). Each has C/P0 warm then C/P0/P0/C measured. Only same-shape same-sample method costs are compared; no NI1x2 proxy for serial NI1+46, and no direct B2/B1 batching speed claim. Full outerattribute time includes native head/checkpoints,endpoint copies,allFA/FLA/conv and existingstage synchronizations; packing/CPU audit/persistence are outside it and retained in group/total wall. All warm, failed, initialization, callback and memory costs remain in receipts. Two observations per method/shape; no general speed guarantee.'
    r['batch_validation_scope']='Actual root endpoint ordering, distinct source IDs, per-example PackedAnswerTargets labels/sample indices and root target logprobs are checked. NI1 has same-process singleton/batch comparisons for each method; NI46 has repeat and method comparisons within batch only. These do not prove arbitrary-batch isolation or new RISE/MAS correctness: zero scorer calls, no NI46 singleton, no swapped-batch experiment, no padded or unequal-length samples. Earlier batch score curves do not substitute for P0 batch scoring.'
    save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/name).is_file():archive.write(A/name,name)
    print(json.dumps({'status':r['status'],'DT_entered':r['DT_entered'],'DT_returned':r['DT_returned'],
        'seconds':r['seconds'],'error':r.get('error')}),flush=True)
