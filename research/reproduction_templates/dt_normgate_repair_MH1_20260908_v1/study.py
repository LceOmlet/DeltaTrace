"""Same frozen paired layer0 norm-gate repair on original MH1.

One actual DT with shared endpoints/upstream and one extra symmetric layer0 GDN propagation.
Four frozen original scores plus two unchanged original 20-step curves; no FT/model replacement.
"""
import os,sys,json,time,hashlib,traceback,zipfile,signal,gc
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
r={'status':'starting','protocol':p,'model_loads':0,'DT_calls':0,'FT_calls':0,'generation_calls':0,
   'scoring_forwards':0,'extra_conv_preactivation_calls':0,'cases':{},'calls':[]}
started=time.perf_counter();handles=[];vectors={};capture=None

def save():
    f=A/'results.partial';f.write_text(json.dumps(r,indent=2));f.replace(A/'results.json')

def sources():
    out={}
    for name,want in p['files_sha256'].items():out[name]=sha((A/name).read_bytes());assert out[name]==want,name
    for name,want in p['official_source_blob_sha1'].items():
        raw=(Path(p['official_root'])/name).read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want,name
        out['FT/'+name]=sha(raw)
    for name,want in p['runtime_source_sha256'].items():
        out['native/'+name]=sha((Path(p['isolated_site'])/name).read_bytes());assert out['native/'+name]==want,name
    return out

def timed(kind,fn):
    torch.cuda.synchronize();tick=time.perf_counter();out=fn();torch.cuda.synchronize()
    r['calls'].append({'kind':kind,'seconds':time.perf_counter()-tick});return out

def timeout(*args):raise TimeoutError('Frozen layer0 diagnostic budget expired.')

def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().to('cpu',copy=True)
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    return value

def features(d,c,e):
    # These are actual retained operands, with no alternate operator evaluation.
    return cpu({'d':{k:d[k] for k in ['input_norm_input','input_norm_output','post_norm_input',
                                     'post_norm_output','mlp_output','output']},
                'c':{k:c[k] for k in ['input','projected_qkv','conv_output','raw_q','raw_k','a','b','z','norm_output','output']},
                'e':{k:e[k] for k in ['q','k','v','beta','raw_g','o','g','h','v_new']}})

def difference(left,right):
    return {group:{k:left[group][k].double()-right[group][k].double() for k in left[group]} for group in left}

def endpoints(x,index):
    return {group:{k:v[index::2] for k,v in values.items()} for group,values in x.items()}

def tensor_bytes(x):
    if isinstance(x,torch.Tensor):return x.numel()*x.element_size()
    return sum(tensor_bytes(v) for v in x.values()) if isinstance(x,dict) else 0

class Layer0Observer:
    def __init__(self):self.coeff=None;self.paired=None;self.boundary_coeff={};self.current=None;self.clean=None;self.points={};self.captures={}
    def boundary(self,name,m,x):
        if name in ('0','1'):self.boundary_coeff[name]=cpu(m)
    def wants_decoder(self,index):return index==0
    def decoder(self,index,d,c,e,upstream,new,terms):
        assert index==0 and self.coeff is None
        g=terms['mixer']
        self.coeff=cpu({'upstream':upstream,'input':new,'m_mlp_norm_output':terms['m_mlp_norm_output'],
            'm_mixer_output':terms['m_mixer_output'],'m_mixer_input':terms['m_mixer_input'],
            **{k:g[k] for k in ['mnorm','mo_before_cast','mo_native','mz','coeff','mq','mk','mb','ma','mconv']},
            'mprojected':g['mprojected'][1::2]})
        self.paired=features(d,c,e)
        self.B2=self.decompose(difference(endpoints(self.paired,1),endpoints(self.paired,0)))
    def decompose(self,delta):
        m=self.coeff;d,c,e=delta['d'],delta['c'],delta['e'];f=m['coeff']
        def dot(coefficient,value):
            assert coefficient.shape==value.shape,(tuple(coefficient.shape),tuple(value.shape))
            return float((coefficient.double()*value).sum())
        u=m['upstream'];mm=m['m_mixer_output'];mi=m['m_mixer_input'];ml=m['m_mlp_norm_output']
        repeat=c['raw_q'].shape[2]//m['mq'].shape[2]
        assert repeat>=1 and c['raw_k'].shape[2]==c['raw_q'].shape[2]
        qgroups=c['raw_q'].reshape(*c['raw_q'].shape[:2],m['mq'].shape[2],repeat,c['raw_q'].shape[-1])
        kgroups=c['raw_k'].reshape(*c['raw_k'].shape[:2],m['mk'].shape[2],repeat,c['raw_k'].shape[-1])
        assert bool(qgroups.eq(qgroups[:,:,:,:1,:]).all()) and bool(kgroups.eq(kgroups[:,:,:,:1,:]).all())
        qraw=dot(m['mq'],qgroups[:,:,:,0,:]);kraw=dot(m['mk'],kgroups[:,:,:,0,:])
        qnorm=dot(f['q'],e['q']);knorm=dot(f['k'],e['k']);v=dot(f['v'],e['v'])
        beta=dot(f['beta'],e['beta']);g=dot(f['g'],e['raw_g'])
        mb=dot(m['mb'],c['b']);ma=dot(m['ma'],c['a']);mz=dot(m['mz'],c['z'])
        conv=dot(m['mconv'],c['conv_output']);projected=dot(m['mprojected'],c['projected_qkv'])
        normgate=dot(m['mnorm'],c['norm_output']);mo=dot(m['mo_before_cast'],e['o']);monative=dot(m['mo_native'],e['o'])
        terms={
            'decoder_output_residual_rounding':dot(u,d['output'])-dot(u,d['post_norm_input'])-dot(u,d['mlp_output']),
            'MLP_combined':dot(u,d['mlp_output'])-dot(ml,d['post_norm_output']),
            'post_RMSNorm':dot(ml,d['post_norm_output'])-dot(mm.double()-u.double(),d['post_norm_input']),
            'mixer_residual_rounding':dot(mm,d['post_norm_input'])-dot(mm,d['input_norm_input'])-dot(mm,c['output']),
            'GDN_output_projection':dot(mm,c['output'])-normgate,
            'GDN_fused_norm_gate':normgate-mo-mz,
            'GDN_mo_BF16_cast':mo-monative,
            'GDN_FLA_including_raw_g_exp':monative-qnorm-knorm-v-beta-g,
            'GDN_QK_L2_and_head_fold':qnorm+knorm-qraw-kraw,
            'GDN_beta_sigmoid':beta-mb,
            'GDN_raw_g_parameter_map':g-ma,
            'GDN_conv_output_split':qraw+kraw+v-conv,
            'GDN_conv_linear_and_SiLU':conv-projected,
            'GDN_input_projections':projected+mz+mb+ma-dot(mi,c['input']),
            'mixer_input_alias':dot(mi,c['input'])-dot(mi,d['input_norm_output']),
            'input_RMSNorm':dot(mi,d['input_norm_output'])-dot(m['input'].double()-mm.double(),d['input_norm_input'])}
        actual=dot(u,d['output']);pred=dot(m['input'],d['input_norm_input'])
        result={'output_contraction':actual,'input_contraction':pred,'actual_minus_predicted':actual-pred,
            'terms':terms,'telescoping_error':sum(terms.values())-(actual-pred),
            'FLA_branch_contractions':{'q':qnorm,'k':knorm,'v':v,'beta':beta,'raw_g':g},
            'norm_gate_branch_contractions':{'normalized_memory_before_cast':mo,'gate':mz}}
        assert abs(result['telescoping_error'])<1e-7,result
        return result
    def consume(self,dc,mc):
        assert dc.calls=={k:1 for k in ('input_norm','post_norm','gate','up','silu','down','mlp','decoder')},dc.calls
        assert mc.calls=={'module':1,'conv':1,'FLA':1,'stage':1},mc.calls
        value=features(dc.values,mc.values,mc.endpoints);self.last_features=value
        self.captures[str(self.current)]={'decoder_calls':dict(dc.calls),'mixer_calls':dict(mc.calls),
            'retained_feature_CPU_bytes':tensor_bytes(value),'initial_cache':mc.initial_cache_receipt}
        if self.current==0:self.clean=value
        else:self.points[str(self.current)]=self.decompose(difference(self.clean,value))


class PairedFLA:
    def __init__(self):
        self.regular=make_compiled_finite_pullback(reuse_scalar_products=False)
        self.capture=make_capture_backend();self.calls=[];self.saved={};self.last_scale=None;self.last_e=None
    def __call__(self,e,do,scale,kind='current'):
        capture=(len(self.calls)==23 or kind=='candidate')
        self.last_scale=float(scale);self.last_e=id(e)
        item={'kind':kind,'captured_layer0':capture,'scale':float(scale),'do_shape':list(do.shape),'status':'entered'}
        self.calls.append(item)
        if capture:
            coeff,diag=self.capture(e,do,scale);self.saved[kind]=cpu(diag)
            item['packed_shape']=list(diag['L'].shape);item['packed_CPU_bytes']=tensor_bytes(self.saved[kind])
        else:coeff=self.regular(e,do,scale)
        item['status']='returned';return coeff

class RepairObserver(Layer0Observer):
    def __init__(self,layer,fla,boundaries,info):
        super().__init__();self.layer=layer;self.fla=fla;self.boundaries=boundaries;self.info=info
        self.other=Layer0Observer();self.control_points={}
    def decoder(self,index,d,c,e,upstream,new,terms):
        super().decoder(index,d,c,e,upstream,new,terms)
        assert self.fla.last_e==id(e) and len(self.fla.calls)==24
        scale=self.fla.last_scale
        candidate,candidate_terms,self.metadata=compute_layer0_symmetric_normgate(self.layer,d,c,e,upstream,new,terms,
            lambda ep,mo,sc:self.fla(ep,mo,sc,kind='candidate'),self.boundaries,scale)
        # Reuse the same existing CPU coefficient mapper; no candidate model forward.
        self.other.decoder(index,d,c,e,upstream,candidate,candidate_terms)
        self.other.paired=self.paired
        self.candidate=(candidate.double()*(d['input_norm_input'][1::2].double()-d['input_norm_input'][0::2].double())).sum(-1)[0].cpu()
        self.scale=scale
        row['paired_repair']=self.metadata
        r['extra_conv_preactivation_calls']+=self.metadata['additional_work']['public_paired_conv_preactivation_calls']
        assert len(self.fla.calls)==25
    def consume(self,dc,mc):
        super().consume(dc,mc);value=self.last_features
        self.control_points[str(self.current)]=value['e']
        if self.current==0:self.other.clean=self.clean
        else:self.other.points[str(self.current)]=self.other.decompose(difference(self.clean,value))

class ScoringLayer0Capture:
    def __init__(self,layer,observer):
        self.layer=layer;self.observer=observer;self.scope=None;self.post_handle=None
        self.pre_handle=layer.register_forward_pre_hook(self.before,with_kwargs=True)
    def before(self,module,args,kwargs):
        assert self.scope is None and sys.getprofile() is None
        dc=NativeDecoderCapture(module,destination='cpu');mc=NativeFreshCacheGDNCapture(module.linear_attn,device='cpu')
        dc.__enter__()
        try:mc.__enter__()
        except Exception:dc.__exit__(None,None,None);raise
        self.scope=(dc,mc)
        # Register AFTER dc's decoder-output hook, so all actual captures exist.
        self.post_handle=module.register_forward_hook(self.after,always_call=True)
    def after(self,module,args,output):
        dc,mc=self.scope;mc.__exit__(None,None,None);dc.__exit__(None,None,None)
        self.post_handle.remove();self.post_handle=None;self.scope=None
        if output is None:raise RuntimeError('Native layer0 scoring forward failed.')
        self.observer.consume(dc,mc)
    def close(self):
        if self.scope is not None:
            dc,mc=self.scope;mc.__exit__(None,None,None);dc.__exit__(None,None,None);self.scope=None
        if self.post_handle is not None:self.post_handle.remove();self.post_handle=None
        self.pre_handle.remove()

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
    from qwen35_answer_finite import PackedAnswerTargets
    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner
    from qwen35_decoder_finite import NativeDecoderCapture
    from qwen35_gdn_finite import NativeGDNCapture
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    from fixed_input_metric_view import FixedInputMetricView
    from finite_fla_coefficient_capture_20260908 import make_capture_backend
    from layer0_normgate_paired_repair_20260908 import compute_layer0_symmetric_normgate
    class NativeFreshCacheGDNCapture(NativeGDNCapture):
        """Observe the original scorer's initial empty cache without altering it.

        Reuse all original capture events except the module-entry cache guard.
        The inherited FLA stage guard still requires actual initial_state=None.
        No native callable, forward argument, frame local or config is changed.
        """
        def event(self,frame,kind,value):
            label=self.codes.get(frame.f_code);f=frame.f_locals
            if label=='module' and kind=='call' and f['self'] is self.module:
                assert not self.active and not self.values
                assert not f.get('kwargs',{}).get('cu_seq_lens_q')
                cache=f.get('cache_params')
                previous=False if cache is None else bool(cache.has_previous_state(self.module.layer_idx))
                assert not previous,'Expected a fresh original-scoring cache, never a continued state.'
                self.initial_cache_receipt={'provided':cache is not None,'class':type(cache).__name__,
                                            'has_previous_state':previous,'layer_index':self.module.layer_idx}
                self.active=True;self.values['input']=self.copy(f['hidden_states'])
                self.values['mask']=self.copy(f.get('attention_mask'))
                self.calls['module']=self.calls.get('module',0)+1
                return
            return super().event(frame,kind,value)
    assert p['capture_steps']==[0,1,10,20]
    assert p['case_indices']==[['morehopqa',1]]
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule and native.is_fast_path_available
    verify_native_sources(p['native_stage_source_sha256'])
    cp=Path(p['checkpoint'])
    for name,want in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==want
    stats=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats();assert r['weight_stats_before']==p['expected_weight_stats']
    tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    prior_raw=Path(p['parent_results_path']).read_bytes();assert sha(prior_raw)==p['parent_results_sha256'];prior=json.loads(prior_raw)
    vector_raw=Path(p['parent_vectors_path']).read_bytes();assert sha(vector_raw)==p['parent_vectors_sha256']
    parent_vectors=np.load(p['parent_vectors_path'],allow_pickle=False)
    data={k:Path(v).read_bytes() for k,v in p['cache_paths'].items()}
    for k,raw in data.items():assert sha(raw)==p['cache_hashes'][k]
    def prepare(dataset,index):
        rec=json.loads(data[dataset].decode().splitlines()[index]);tok=tokenizer(rec['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        keep=keep_token_indices([rec['prompt'][a:b] for a,b in tok['offset_mapping']])
        target=tokenizer(rec['target']+tokenizer.eos_token,add_special_tokens=False)['input_ids']
        ids=torch.tensor(tok['input_ids']+target,dtype=torch.long);base=ids.clone();base[keep]=tokenizer.eos_token_id
        key=f'{dataset}_{index}';info={'input_sha256':sha(ids.numpy().tobytes()),'prompt_length':len(tok['input_ids']),
            'target_length':len(target),'total_length':len(ids),'keep':keep}
        if key in p['cases']:assert info==prior['cases'][key]['input'],key
        return key,rec,ids,base,torch.tensor(target),info
    control=prepare('niah_mq_q2',0);cases=[prepare(*x) for x in p['case_indices']]
    assert torch.cuda.mem_get_info()[0]>=32*1024**3
    torch.manual_seed(73);r['status']='loading';save()
    model,loading=timed('actual_model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,
        attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True))
    assert not any(loading.values());model.eval().requires_grad_(False);r['model_loads']=1
    for layer in model.model.language_model.layers:
        if layer.block_type=='linear_attention':
            assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
            assert layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
    with torch.no_grad():warm=timed('original_order_native_eager_B1_diagnostic',lambda:model(input_ids=control[2][None].to('cuda'),
        attention_mask=torch.ones_like(control[2][None],device='cuda'),use_cache=False))
    del warm,control;model.set_attn_implementation('flash_attention_2')
    paired_fla=PairedFLA()
    runner=Qwen35DenseFiniteRunner(model,VendorFAFiniteP1BF16D256(p['finite_FA_library'],p['finite_FA_library_sha256']),
        paired_fla)
    evaluator=LLMAttributionEvaluator(model,tokenizer);layer0=model.model.language_model.layers[0]
    assert layer0.block_type=='linear_attention'
    quality_raw=Path(p['quality_parent_path']).read_bytes();assert sha(quality_raw)==p['quality_parent_sha256']
    quality=json.loads(quality_raw)
    for key,rec,ids,base,target,info in cases:
        old_info=quality['cases'][key]['input']
        for name,value in info.items():assert old_info[name]==value,(key,name)
        assert sha(base.numpy().tobytes())==old_info['baseline_sha256']
        gold=old_info['gold'];row={'input':info,'gold':gold,'scoring_points':{},'curves':{}};r['cases'][key]=row
        row['prior_fixed_FT_metrics']={k:v for k,v in quality['cases'][key]['curves'].items() if k.startswith('FT')}
        observer=RepairObserver(layer0,paired_fla,runner.boundaries,info)
        pairs=torch.stack((base,ids)).to('cuda');mask=torch.ones_like(pairs)
        selection=PackedAnswerTargets([{'target_ids':target,'prompt_length':info['prompt_length']}],
            [list(range(len(target)))],len(ids),'cuda')
        r['status']='same_actual_forward_paired_normgate_repair';save()
        signed,details=runner.attribute(pairs,mask,selection,select_output_rows=True,observer=observer);r['DT_calls']+=1
        signed=signed[0];candidate=observer.candidate
        for name,value in [('current',signed),('candidate',candidate)]:
            vectors[key+'_'+name+'_full']=value.numpy();vectors[key+'_'+name+'_evaluated']=value[:info['prompt_length']].float().numpy()
        np.savez_compressed(A/'vectors.npz',**vectors)
        row['DT_with_paired_diagnostics']=details;row['finite_FLA_calls']=paired_fla.calls
        reference=torch.from_numpy(parent_vectors[key+'_DT_full'])
        row['historical_vector_relative_L2_report_only']=float((signed-reference).norm()/reference.norm())
        row['historical_guard_policy']='Prior failed 2% experiment retained. This new repair uses shared actual endpoints/upstream, not old-vector admission. Historical drift is reported and not called method effect.'
        row['B2_current']=observer.B2;row['B2_candidate']=observer.other.B2
        assert bool(torch.isfinite(candidate).all()) and bool(torch.isfinite(signed).all())
        assert torch.equal(observer.boundary_coeff['0'],observer.coeff['input'])
        assert torch.equal(observer.boundary_coeff['1'],observer.coeff['upstream'])
        del pairs,mask,selection,reference
        capture=ScoringLayer0Capture(layer0,observer)
        def before_frozen(_module,args,kw):
            r['scoring_forwards']+=1;x=kw['input_ids'].detach().cpu();step=observer.current
            frozen=prior['cases'][key]['curve']['input_receipts'][step]
            assert x.shape==(1,len(ids)) and bool(kw['attention_mask'].eq(1).all())
            assert sha(x.numpy().tobytes())==frozen['input_sha256']
            assert (x[0]!=ids).nonzero().flatten().tolist()==frozen['deleted_positions']
            row['scoring_points'][str(step)]={'input_receipt':frozen}
        handles=[model.register_forward_pre_hook(before_frozen,with_kwargs=True)]
        r['status']='four_frozen_original_scores_and_real_control_states';save()
        try:
            for step in p['capture_steps']:
                observer.current=step;frozen=prior['cases'][key]['curve']['input_receipts'][step]
                prompt=ids[:info['prompt_length']].clone();prompt[frozen['deleted_positions']]=tokenizer.eos_token_id
                with torch.no_grad():lp=timed('original_frozen_scorer_'+str(step),lambda:evaluator.compute_logprob_response_given_prompt(
                    prompt[None].to('cuda'),target[None].to('cuda')))
                point=row['scoring_points'][str(step)];point['original_native_score']=float(lp.sum().cpu())
                point['prior_original_native_score']=prior['cases'][key]['curve']['scores'][step]
                point['capture']=observer.captures[str(step)]
                if step:
                    point['current_layer0_decomposition']=observer.points[str(step)]
                    point['candidate_layer0_decomposition']=observer.other.points[str(step)]
                    actual=row['scoring_points']['0']['original_native_score']-point['original_native_score']
                    deleted=frozen['deleted_positions'];point['complete_input']={'actual_model_effect':actual}
                    for name in ['current','candidate']:
                        score=torch.from_numpy(vectors[key+'_'+name+'_evaluated'])
                        prediction=float(score[deleted].double().sum())
                        point['complete_input'][name]={'prediction':prediction,'prediction_minus_actual':prediction-actual}
                save();del lp
        finally:
            capture.close();capture=None
            for h in handles:h.remove()
            handles=[]
        assert r['scoring_forwards']==4
        artifact=A/(key+'_control_content_inputs.pt')
        torch.save({'paired':observer.paired['e'],'points':observer.control_points,
            'current':paired_fla.saved['current'],'candidate':paired_fla.saved['candidate'],'scale':observer.scale},artifact)
        row['control_content_artifact']={'file':artifact.name,'sha256':sha(artifact.read_bytes()),'bytes':artifact.stat().st_size}
        coeff_path=A/(key+'_paired_coefficients.pt');torch.save({'current':observer.coeff,'candidate':observer.other.coeff},coeff_path)
        row['coefficient_artifact']={'file':coeff_path.name,'sha256':sha(coeff_path.read_bytes()),'bytes':coeff_path.stat().st_size}
        save();del observer;gc.collect()
        view=FixedInputMetricView(evaluator,rec['prompt'])
        assert view.compute_logprob_response_given_prompt.__func__ is LLMAttributionEvaluator.compute_logprob_response_given_prompt
        for method in ['current','candidate']:
            score=torch.from_numpy(vectors[key+'_'+method+'_evaluated'])
            curve={'input_receipts':[]};row['curves'][method]=curve
            curve['needle']=float(evaluate_attr_recovery_skip_tokens(score[None],keep_prompt_token_indices=info['keep'],
                gold_prompt_token_indices=gold,top_fraction=0.1)) if gold else None
            def before_metric(_module,args,kw):
                x=kw['input_ids'].detach().cpu();assert x.shape==(1,len(ids)) and bool(kw['attention_mask'].eq(1).all())
                changed=(x[0]!=ids).nonzero().flatten().tolist();assert set(changed)<=set(info['keep'])
                assert all(int(x[0,j])==tokenizer.eos_token_id for j in changed)
                if not curve['input_receipts']:assert not changed
                curve['input_receipts'].append({'input_sha256':sha(x.numpy().tobytes()),'deleted_positions':changed})
                r['scoring_forwards']+=1
            def observe_metric(frame,event,arg):
                if frame.f_code is faithfulness_test_skip_tokens.__code__ and event=='return' and arg is not None:
                    loc=frame.f_locals
                    for name in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores']:
                        curve[name]=np.asarray(loc[name]).copy().tolist()
                    curve['sorted_keep']=[int(x) for x in loc['sorted_keep']];curve['attr_sum']=float(loc['attr_sum'])
                    curve['return_metrics']=[float(x) for x in arg]
            handles=[model.register_forward_pre_hook(before_metric,with_kwargs=True)]
            r['status']='original_needle_RISE_MAS_'+method;save();sys.setprofile(observe_metric)
            try:
                with torch.no_grad():returned=timed('original_full_curve_'+method,lambda:faithfulness_test_skip_tokens(view,
                    score[None],rec['prompt'],rec['target'],keep_prompt_token_indices=info['keep'],
                    user_prompt_indices=list(range(info['prompt_length'])),k=20))
            finally:
                sys.setprofile(None)
                for h in handles:h.remove()
                handles=[]
            assert curve['return_metrics']==[float(x) for x in returned]
            assert len(curve['input_receipts'])==21 and curve['input_receipts'][-1]['deleted_positions']==info['keep']
            assert all(np.isfinite(curve[name]).all() for name in ['scores','density','normalized_model_response','alignment_penalty','corrected_scores'])
            save()
        row['acceptance_scope']='MH1 existing-case confirmation only, not an independent holdout. Original new-order curves measured. No independent confirmation or production speed claim; no automatic promotion or multi-rule sweep.'
        del signed,candidate;gc.collect()
    assert r['DT_calls']==1 and r['scoring_forwards']==46
    assert len(paired_fla.calls)==25 and sum(x['kind']=='candidate' for x in paired_fla.calls)==1
    r['sources_after']=sources();r['weight_stats_after']=stats()
    assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    r['status']='MH1_paired_normgate_repair_original_metrics_complete'
except Exception as error:
    r['status']='failed';r['error']=traceback.format_exc()
    if hasattr(error,'metadata'):r['failed_diagnostic_metadata']=error.metadata
    if 'paired_fla' in globals():r['finite_FLA_calls_at_failure']=paired_fla.calls
finally:
    if capture is not None:capture.close()
    sys.setprofile(None);signal.alarm(0)
    for h in handles:h.remove()
    r['job_seconds']=time.perf_counter()-started;save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json','vectors.npz']:
            if (A/name).exists():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'DT_calls':r['DT_calls'],'scoring_forwards':r['scoring_forwards'],
        'seconds':r['job_seconds'],'error':r.get('error')}),flush=True)
