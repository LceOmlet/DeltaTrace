"""Actual MH2 conditional decoder boundaries;one fixed DT and four original scores."""
import os,sys,json,time,hashlib,traceback,zipfile,signal,gc
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
r={'status':'starting','protocol':p,'model_loads':0,'native_eager_diagnostics':0,'DT_entered':0,'DT_returned':0,
   'scorer_entered':0,'scorer_returned':0,'FT_calls':0,'FT_entered':0,'generation_calls':0,'cases':{},'runs':[],'calls':[]}
started=time.perf_counter();handles=[];vectors={};active=None
fas={};finite_fla=None;candidate_backend=None

def save():
    f=A/'results.partial';f.write_text(json.dumps(r,indent=2,allow_nan=False));f.replace(A/'results.json')
def timeout(*args):raise TimeoutError('Frozen1DT4score MH2 actual-boundary budget expired.')
def timed(kind,fn):
    torch.cuda.synchronize();tick=time.perf_counter();row={'kind':kind,'status':'entered'};r['calls'].append(row)
    try:
        out=fn();torch.cuda.synchronize();row['status']='returned';return out
    finally:row['seconds']=time.perf_counter()-tick
def sources():
    found={}
    assert {str(f.relative_to(Path(p['official_root']))) for f in (Path(p['official_root'])/'flashtrace').rglob('*.py')}==set(p['official_package_blob_sha1'])
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

def tensor_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(x) for x in value.values())
    return 0

class BoundaryObserver:
    def __init__(self):
        self.coeff={};self.paired={};self.endpoint_effect={};self.activations={};self.points={};self.counts={};self.logp={};self.current=None
    def boundary(self,name,m,x):
        assert name not in self.coeff and m.shape[0]==1 and x.shape[0]==2
        self.coeff[name]=m.detach().to('cpu',copy=True)
        self.paired[name]=x.detach().to('cpu',copy=True)
        self.endpoint_effect[name]=float((self.coeff[name].double()*(self.paired[name][1::2].double()-self.paired[name][0::2].double())).sum())
    def activation(self,name,x):
        assert self.current in p['capture_steps'] and name in self.coeff
        value=x.detach().to('cpu',copy=True);assert value.shape==self.coeff[name].shape
        key=str(self.current);self.counts.setdefault(key,{})[name]=self.counts.setdefault(key,{}).get(name,0)+1
        self.activations.setdefault(key,{})[name]=value
        if self.current:
            delta=self.activations['0'][name].double()-value.double()
            self.points.setdefault(key,{})[name]=(self.coeff[name].double()*delta).sum(-1).squeeze(0)
    def original_logits(self,logits,selection):
        z=logits[selection.samples,selection.positions].float()
        self.logp[str(self.current)]=float(z.log_softmax(-1).gather(-1,selection.labels[:,None]).double().sum())

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
    from llm_attr_eval import LLMAttributionEvaluator
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
    assert p['call_schedule']==[['morehopqa_2','DT']] and p['capture_steps']==[0,3,10,20]
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
    prior_raw=Path(p['source_standalone']['results_path']).read_bytes()
    assert sha(prior_raw)==p['source_standalone']['results_sha256'];prior=json.loads(prior_raw)
    source_vector_raw=Path(p['source_standalone']['vectors_path']).read_bytes();assert sha(source_vector_raw)==p['source_standalone']['vectors_sha256']
    source_vectors=np.load(p['source_standalone']['vectors_path'],allow_pickle=False)
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
        if ref.get('expected_gold') is not None:assert gold==ref['expected_gold']
        mapping={'source_record_sha256':sha(line.encode()),'original_cached_spans_reproduced':True,
            'original_sink_span':old.sink_span,'new_sink_span':new.sink_span,'new_thinking_span':new.thinking_span,
            'gold':gold,'gold_function':'unchanged author ruler_gold_prompt_token_indices',
            'input_scope':'Original author prompt plus fixed original target and EOS;no additional chat template;same wholepilot pipeline.',
            'historical_input_hash_available':ref.get('expected_input') is not None,
            'internal_target_semantics':'Existing current DT full fixed-response logprob including EOS. Four unchanged original evaluator calls use exactly these target tokens; no FT in this diagnostic.'}
        return {'record':rec,'ids':ids,'base':base,'target':torch.tensor(target),'input':info,'gold':gold,'mapping':mapping}
    cases={f'{dataset}_{index}':prepare(dataset,index) for dataset,index in p['case_indices']}
    init=cases['niah_mq_q2_0']
    r['input_freeze_before_model_load']={key:dict(input=case['input'],mapping=case['mapping'],
        input_ids=case['ids'].tolist(),target_ids=case['target'].tolist(),baseline_sha256=sha(case['base'].numpy().tobytes())) for key,case in cases.items()}
    source_case=prior['cases']['morehopqa_2'];case=cases['morehopqa_2']
    assert source_case['input']==case['input']
    curve=source_case['curves'][p['source_method']];assert curve['status']=='complete'
    order=curve['sorted_keep'];assert sorted(order)==case['input']['keep']
    source_score=torch.from_numpy(source_vectors['morehopqa_2_'+p['source_method']+'_evaluated'])
    assert source_score.dtype==torch.float32 and source_score.shape==(case['input']['prompt_length'],)
    assert bool(torch.isfinite(source_score).all()) and bool((source_score[order][1:]<=source_score[order][:-1]).all())
    assert torch.equal(torch.from_numpy(source_vectors['morehopqa_2_'+p['source_method']+'_full'])[:len(source_score)].float(),source_score)
    assert len(curve['scores'])==len(curve['input_receipts'])==21
    groups=[];offset=0;size,extra=divmod(len(order),20);deleted=set();full=case['ids'].clone()
    receipts=[{'input_sha256':sha(full[None].numpy().tobytes()),'deleted_positions':[]}]
    for step in range(20):
        group=order[offset:offset+size+(step<extra)];offset+=len(group);groups.append(group)
        full[group]=tokenizer.eos_token_id;deleted.update(group)
        receipts.append({'input_sha256':sha(full[None].numpy().tobytes()),'deleted_positions':sorted(deleted)})
    assert receipts==curve['input_receipts'] and receipts[-1]['deleted_positions']==case['input']['keep']
    assert p['frozen_capture_receipts']=={str(step):receipts[step] for step in p['capture_steps']}
    r['source_mask_closure']={'source_results_sha256':p['source_standalone']['results_sha256'],
        'source_vectors_sha256':p['source_standalone']['vectors_sha256'],
        'all_21_receipts_reconstructed':True,'source_vector_descending_on_frozen_order':True,
        'sorted_keep':order,'groups':groups,'capture_receipts':p['frozen_capture_receipts'],
        'new_vector_cannot_change_masks':True}
    del source_score,full
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
    fas={method:BudgetedFinite(original_fa,8) for method in ('DT',)}
    finite_fla=BudgetedFinite(make_compiled_finite_pullback(reuse_scalar_products=False),25)
    candidate_backend=Layer0FLAEndpointAverage(finite_fla)
    runners={method:Qwen35DenseFiniteRunner(model,backend,finite_fla,norm_gate_rules={0:'symmetric'},
        finite_fla_by_layer={0:candidate_backend}) for method,backend in fas.items()}
    assert runners['DT'].finite_fla_by_layer=={0:candidate_backend}
    assert all(runner.finite_fla is finite_fla for runner in runners.values())
    evaluator=LLMAttributionEvaluator(model,tokenizer)
    for key,case in cases.items():
        r['cases'][key]={'input':case['input'],'gold':case['gold'],'mapping':case['mapping'],
            'baseline_sha256':sha(case['base'].numpy().tobytes()),'curves':{},
            'role':'initialization_only' if key=='niah_mq_q2_0' else 'fixed_source_masks_conditional_boundaries'}
    for number,(key,method) in enumerate(p['call_schedule']):
        phase='conditional_diagnostic'
        observer=BoundaryObserver()
        gc.collect();torch.cuda.synchronize();resident_before_pair=torch.cuda.memory_allocated();reserved_before_pair=torch.cuda.memory_reserved()
        case=cases[key];info=case['input'];pairs=torch.stack((case['base'],case['ids'])).to('cuda')
        selection=PackedAnswerTargets([{'target_ids':case['target'],'prompt_length':info['prompt_length']}],
            [list(range(len(case['target'])))],len(case['ids']),'cuda')
        row={'case':key,'method':method,'phase':phase,'number':number,'status':'entered','root_forwards':0,
            'GPU_allocated_before_pair':resident_before_pair,'GPU_reserved_before_pair':reserved_before_pair}
        r['runs'].append(row);r['status']='attribute_'+key+'_'+method;save()
        def before_root(_module,args,kw):
            assert torch.equal(kw['input_ids'],pairs) and bool(kw['attention_mask'].eq(1).all())
            row['root_forwards']+=1
        handle=model.register_forward_pre_hook(before_root,with_kwargs=True)
        old_fa=fas[method].returned;old_fla=finite_fla.returned;old_receipts=len(candidate_backend.calls)
        old_fa_entered=fas[method].entered;old_fla_entered=finite_fla.entered
        gc.collect();torch.cuda.synchronize();row['GPU_allocated_before_attribute']=torch.cuda.memory_allocated();row['GPU_reserved_before_attribute']=torch.cuda.memory_reserved();assert r['DT_entered']<1;r['DT_entered']+=1;tick=time.perf_counter();details=None
        try:
            signed,details=runners[method].attribute(pairs,torch.ones_like(pairs),selection,select_output_rows=True,observer=observer)
            r['DT_returned']+=1;row['details']=details
        finally:
            handle.remove()
            try:torch.cuda.synchronize()
            except Exception:row['final_sync_error']=traceback.format_exc()
            row['outer_attribute_seconds']=time.perf_counter()-tick
            row['finite_callback_counts']={'FA_entered':fas[method].entered-old_fa_entered,'FA_returned':fas[method].returned-old_fa,
                'FLA_backend_entered':finite_fla.entered-old_fla_entered,'FLA_backend_returned':finite_fla.returned-old_fla}
            row['candidate_layer0_receipts']=candidate_backend.calls[old_receipts:]
            row['native_stage_accounting']={'returned_FLA_backend_calls_times_two':2*(finite_fla.returned-old_fla),
                'stages_inside_nonreturned_backend':'unknown' if finite_fla.entered-old_fla_entered!=finite_fla.returned-old_fla else 0}
            if details is None:row['status']='failed_attribute';row['native_runner_ledger']='unknown: attribution did not return details'
            save()
        assert row['root_forwards']==1
        extra=1
        # Original checker counts the24 GDN callback sites. Backend executions
        # are separately25 for candidate because only layer0 calls twice.
        row['counts']=check_counts(details,fas[method].returned-old_fa,finite_fla.returned-old_fla-extra)
        row['counts']['finite_FLA_backend_calls']=finite_fla.returned-old_fla
        row['counts']['finite_FLA_calls_scope']='24 original GDN callback sites; see finite_FLA_backend_calls for actual executions'
        assert row['counts']['finite_FLA_backend_calls']==24+extra
        assert details['norm_gate_rules']=={'0':'symmetric'} and details['finite_fla_by_layer']==([0] if extra else [])
        assert details['select_output_rows'] is True
        kinds=[item['kind'] for item in details['calls']]
        assert [x for x in kinds if x.startswith('native_replay_')]==['native_replay_'+str(i) for i in reversed(range(32))]
        assert [x for x in kinds if x.startswith('finite_decoder_')]==['finite_decoder_'+str(i) for i in reversed(range(32))]
        assert list(details['layers'])==[str(i) for i in reversed(range(32))]
        assert [int(i) for i,x in details['layers'].items() if x['block_type']=='linear_attention']==p['expected_GDN_reverse_order']
        assert len(row['candidate_layer0_receipts'])==extra
        assert all(x['status']=='returned' and [c['orientation'] for c in x['backend_calls']]==['original','swapped']
            and all(c['status']=='returned' for c in x['backend_calls']) for x in row['candidate_layer0_receipts'])
        signed=signed[0];score=signed[:info['prompt_length']].float();assert bool(torch.isfinite(signed).all())
        source_case=prior['cases'][key];assert source_case['input']==info
        reference=torch.from_numpy(source_vectors[key+'_'+p['source_method']+'_full'])
        row['source_vector_drift_report_only']={'relative_L2':float((signed.double()-reference.double()).norm()/reference.double().norm().clamp_min(1e-30)),
            'max_absolute':float((signed.double()-reference.double()).abs().max()),'bitwise_equal':bool(torch.equal(signed,reference))}
        row['source_B2_root_effect']=prior['runs'][p['source_run_index']]['details']['root_effect']
        row['current_minus_source_B2_root_effect']=details['root_effect']-row['source_B2_root_effect']
        row['boundary_coefficient_shapes']={name:list(value.shape) for name,value in observer.coeff.items()}
        row['B2_endpoint_boundary_contractions']=observer.endpoint_effect
        assert set(observer.coeff)==set([str(i) for i in range(33)]+['norm'])
        row['coefficient_CPU_bytes']=tensor_bytes(observer.coeff)
        del reference
        row['status']='complete'
        row['deletion_audit']=deletion_audit(score,case['ids'],info['keep'],tokenizer.eos_token_id)
        vector_key=key+'_'+method
        row['vector_key']=vector_key
        vectors[vector_key+'_full']=signed.numpy().copy();vectors[vector_key+'_evaluated']=score.numpy().copy()
        row['signed_summary']={'net':float(signed.double().sum()),'positive':float(signed[signed>0].double().sum()),
            'negative':float(signed[signed<0].double().sum()),'needle':float(evaluate_attr_recovery_skip_tokens(score[None],
                keep_prompt_token_indices=info['keep'],gold_prompt_token_indices=case['gold'],top_fraction=.1)) if case['gold'] else None}
        row['memory_cost']={'peak_allocated_full_model_resident':row['details']['peak_allocated'],
            'peak_reserved_full_model_resident':row['details']['peak_reserved'],
            'peak_allocated_minus_before_pair':row['details']['peak_allocated']-resident_before_pair,
            'peak_allocated_minus_before_attribute':row['details']['peak_allocated']-row['GPU_allocated_before_attribute']}
        np.savez_compressed(A/'vectors.npz',**vectors);del signed,score,pairs,details
        gc.collect();torch.cuda.synchronize();row['GPU_allocated_after_cleanup']=torch.cuda.memory_allocated();row['GPU_reserved_after_cleanup']=torch.cuda.memory_reserved();save()
    assert r['DT_entered']==r['DT_returned']==1
    # Four already executed source-DT masks, not the new vector's ordering.
    key='morehopqa_2';case=cases[key];info=case['input'];row=r['cases'][key]
    curve=prior['cases'][key]['curves'][p['source_method']];row['source_curve_scope']='Current-DT control masks from the completed fixed-eight-case PV study, frozen before this diagnostic; no new sorting or original MAS curve.'
    row['points']={};row['decomposition']={}
    current_score=torch.from_numpy(vectors[key+'_DT_evaluated'])
    source_score=torch.from_numpy(source_vectors[key+'_'+p['source_method']+'_evaluated'])
    def before_score(_module,args,kw):
        assert r['scorer_entered']<4;step=observer.current
        frozen=curve['input_receipts'][step];x=kw['input_ids'].detach().cpu()
        assert x.shape==(1,len(case['ids'])) and bool(kw['attention_mask'].eq(1).all())
        deleted=(x[0]!=case['ids']).nonzero().flatten().tolist()
        assert sha(x.numpy().tobytes())==frozen['input_sha256'] and deleted==frozen['deleted_positions']
        assert all(int(x[0,j])==tokenizer.eos_token_id for j in deleted)
        r['scorer_entered']+=1
        row['points'][str(step)]={'status':'entered','input_receipt':frozen,'source_original_native_score':curve['scores'][step]}
    def after_score(_module,args,output):
        if output is None:return
        assert output.logits.dtype==torch.bfloat16
        r['scorer_returned']+=1
        observer.original_logits(output.logits,selection)
    handles=[model.register_forward_pre_hook(before_score,with_kwargs=True),model.register_forward_hook(after_score,always_call=True)]
    for i,layer in enumerate(layers):
        def activation_pre(_module,args,kw,i=i):observer.activation(str(i),args[0] if args else kw['hidden_states'])
        handles.append(layer.register_forward_pre_hook(activation_pre,with_kwargs=True))
    def norm_hook(_module,args,output):
        observer.activation('32',args[0]);observer.activation('norm',output)
    handles.append(model.model.language_model.norm.register_forward_hook(norm_hook))
    assert sys.getprofile() is None
    try:
        for step in p['capture_steps']:
            observer.current=step;r['status']='original_B1_boundaries_step'+str(step);save()
            receipt=curve['input_receipts'][step]
            prompt=case['ids'][:info['prompt_length']].clone();prompt[receipt['deleted_positions']]=tokenizer.eos_token_id
            with torch.no_grad():lp=timed('original_scorer_with_passive_boundaries_step'+str(step),lambda:evaluator.compute_logprob_response_given_prompt(
                prompt[None].to('cuda'),case['target'][None].to('cuda')))
            point=row['points'][str(step)];point['original_native_score']=float(lp.sum().cpu())
            point['current_minus_source_native_score']=point['original_native_score']-point['source_original_native_score']
            point['FP32_same_native_logits_diagnostic']=observer.logp[str(step)]
            point['boundary_capture_counts']=observer.counts[str(step)]
            assert observer.counts[str(step)]=={name:1 for name in observer.coeff}
            point['status']='complete';save();del lp,prompt
    finally:
        for handle in handles:handle.remove()
        handles=[]
    assert r['scorer_entered']==r['scorer_returned']==4
    assert r['FT_entered']==r['FT_calls']==r['generation_calls']==0
    row['sign_convention']='actual minus predicted; negative decoder jump means excess DT predicted drop'
    row['B1_boundary_contractions']={step:{name:float(value.sum()) for name,value in entries.items()} for step,entries in observer.points.items()}
    for step in [3,10,20]:
        a=row['B1_boundary_contractions'][str(step)];deleted=curve['input_receipts'][step]['deleted_positions']
        predicted=float(current_score[deleted].double().sum());source_predicted=float(source_score[deleted].double().sum())
        actual=row['points']['0']['original_native_score']-row['points'][str(step)]['original_native_score']
        actual32=observer.logp['0']-observer.logp[str(step)]
        jumps={str(i):a[str(i+1)]-a[str(i)] for i in range(32)}
        terms={'native_BF16_score_minus_same_logits_FP32':actual-actual32,
            'head_and_logprob_seed':actual32-a['norm'],'final_norm':a['norm']-a['32'],
            'decoders':sum(jumps.values()),'input_map':a['0']-predicted}
        old_actual=curve['scores'][0]-curve['scores'][step]
        entry={'actual_native_logprob_drop':actual,'same_native_logits_FP32_drop':actual32,
            'current_DT_predicted_drop':predicted,'actual_minus_predicted':actual-predicted,
            'source_DT_predicted_drop':source_predicted,'source_actual_logprob_drop':old_actual,
            'source_actual_minus_predicted':old_actual-source_predicted,
            'conditional_error_drift_from_new_score':actual-old_actual,
            'conditional_error_drift_from_new_vector':-(predicted-source_predicted),
            'terms':terms,'decoder_errors':jumps,
            'decoder_errors_ranked_absolute':sorted(jumps.items(),key=lambda x:abs(x[1]),reverse=True),
            'most_overpredicting_decoder':min(jumps,key=jumps.get),
            'FA_decoder_sum':sum(jumps[str(i)] for i in p['expected_FA_layers']),
            'GDN_decoder_sum':sum(jumps[str(i)] for i in p['expected_GDN_reverse_order']),
            'telescoping_error':sum(terms.values())-(actual-predicted)}
        assert abs(entry['telescoping_error'])<1e-7
        assert abs(entry['actual_minus_predicted']-entry['source_actual_minus_predicted']-
            entry['conditional_error_drift_from_new_score']-entry['conditional_error_drift_from_new_vector'])<1e-7
        row['decomposition'][str(step)]=entry
        for name,value in observer.points[str(step)].items():vectors['step'+str(step)+'_boundary_'+name]=value.numpy().copy()
        for i in range(32):vectors['step'+str(step)+'_decoder_'+str(i)+'_actual_minus_predicted']=(observer.points[str(step)][str(i+1)]-observer.points[str(step)][str(i)]).numpy().copy()
    row['B1_allEOS_minus_B2_boundary_effect']={name:row['B1_boundary_contractions']['20'][name]-observer.endpoint_effect[name] for name in observer.coeff}
    # Private coefficients and native boundary activations enable later bounded
    # localization; they are not put in the review archive or a public manifest.
    private={'coefficients':observer.coeff,'paired_native_boundaries':observer.paired,
        'B1_native_boundaries':observer.activations,'source_protocol_sha256':sha((A/'protocol.json').read_bytes())}
    path=A/'MH2_actual_boundary_private.pt';timed('save_actual_private_boundary_tensors',lambda:torch.save(private,path))
    row['private_artifact']={'file':path.name,'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size,
        'tensor_CPU_bytes':tensor_bytes(private),'scope':'Actual tensors only; no weights. Private .pt is excluded from review ZIP.'}
    np.savez_compressed(A/'vectors.npz',**vectors)
    del private,observer,selection,current_score,source_score;gc.collect();save()
    r['finite_counts']={method:{'entered':op.entered,'returned':op.returned} for method,op in fas.items()}
    assert all(fas[method].entered==fas[method].returned==8 for method in fas)
    assert finite_fla.entered==finite_fla.returned==25
    assert candidate_backend.entered==candidate_backend.returned==1
    r['sources_after']=sources();r['weight_stats_after']=weights()
    assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256'] and sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    r['vectors_sha256']=sha((A/'vectors.npz').read_bytes());r['status']='MH2_current_conditional_boundaries_1DT4score_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    sys.setprofile(None);signal.alarm(0)
    for handle in handles:handle.remove()
    r['seconds']=time.perf_counter()-started
    r['finite_counts']={method:{'entered':op.entered,'returned':op.returned} for method,op in fas.items()}
    if finite_fla is not None:
        r['finite_counts']['FLA_backend']={'entered':finite_fla.entered,'returned':finite_fla.returned,
            'native_adjoint_stages_from_returned_calls':2*finite_fla.returned,
            'native_stages_inside_nonreturned_calls':'unknown' if finite_fla.entered!=finite_fla.returned else 0}
    if candidate_backend is not None:r['finite_counts']['layer0_average_wrapper']={
        'entered':candidate_backend.entered,'returned':candidate_backend.returned,'calls':candidate_backend.calls}
    r['cost_scope']='One current DT with passive34boundary coefficient copies;four original B1 scorer calls on predeclared standalone masks0/3/10/20. Existing1model-load/1NI0eager initialization,32native decoder replays,8FAfinite/8FAaux,25FLAbackends/50nativeadjoints. Boundary CPU double contractions,FP32diagnostics of the same actual B1 logits and private tensor save cost are all measured. No new MAS curve,FT,generation,newcandidate,extraGDN/conv or precision repair. All entered/returned and failed work retained;diagnostic cost is not production timing.'
    save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/name).is_file():archive.write(A/name,name)
    print(json.dumps({'status':r['status'],'DT_entered':r['DT_entered'],'DT_returned':r['DT_returned'],
        'scorer_entered':r['scorer_entered'],'scorer_returned':r['scorer_returned'],'seconds':r['seconds'],'error':r.get('error')}),flush=True)
