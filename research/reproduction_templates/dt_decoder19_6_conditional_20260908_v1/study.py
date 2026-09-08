"""Actual layer19 FA/layer6 GDN conditional ledgers on frozen production MH bins.

One normal symmetric-layer0 DT, four original scorer calls, passive captures.
No new finite rule, extra operator, alternate forward, or metric curve.
"""
import os,sys,json,time,hashlib,traceback,zipfile,signal,gc
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
    TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
r={'status':'starting','protocol':p,'model_loads':0,'native_eager_diagnostics':0,'DT_calls_entered':0,'DT_calls':0,
   'scoring_forwards_entered':0,'scoring_forwards_returned':0,'FT_calls':0,'generation_calls':0,
   'extra_operator_calls':0,'calls':[],'points':{}}
started=time.perf_counter();handles=[];capture=None;observer=None;finite_fa=None;finite_fla=None
BOUNDARIES=('0','1','6','7','19','20','32','norm')


def save():
    q=A/'results.partial';q.write_text(json.dumps(r,indent=2));q.replace(A/'results.json')


def sources():
    out={}
    for name,want in p['files_sha256'].items():out[name]=sha((A/name).read_bytes());assert out[name]==want,name
    for name,want in p['official_source_blob_sha1'].items():
        raw=(Path(p['official_root'])/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want,name
        out['FT/'+name]=sha(raw)
    for name,want in p['runtime_source_sha256'].items():
        out['native/'+name]=sha((Path(p['isolated_site'])/name).read_bytes());assert out['native/'+name]==want,name
    return out


def timed(kind,fn):
    torch.cuda.synchronize();tick=time.perf_counter();item={'kind':kind,'status':'entered'};r['calls'].append(item)
    try:
        out=fn();torch.cuda.synchronize();item['status']='returned';return out
    finally:item['seconds']=time.perf_counter()-tick


def cpu(x):
    if isinstance(x,torch.Tensor):return x.detach().to('cpu',copy=True)
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return type(x)(cpu(v) for v in x)
    return x


def nbytes(x):
    if isinstance(x,torch.Tensor):return x.numel()*x.element_size()
    if isinstance(x,dict):return sum(nbytes(v) for v in x.values())
    if isinstance(x,(tuple,list)):return sum(nbytes(v) for v in x)
    return 0


def delta(left,right):return {g:{k:left[g][k].double()-right[g][k].double() for k in left[g]} for g in left}
def endpoint(x,i):return {g:{k:v[i::2] for k,v in vs.items()} for g,vs in x.items()}
def dot(m,x):
    assert m.shape==x.shape,(m.shape,x.shape)
    return float((m.double()*x.double()).sum())


def features(index,d,c,e):
    if index==19:out=fa_diag.features(d,c,e)
    else:
        out=cpu({'d':{k:d[k] for k in ['input_norm_input','input_norm_output','post_norm_input','post_norm_output','mlp_output','output']},
            'c':{k:c[k] for k in ['input','projected_qkv','conv_output','raw_q','raw_k','a','b','z','norm_output','output']},'e':e})
    out['d'].update({k:cpu(d[k]) for k in ['gate_output','up_output','silu_output','down_input']})
    return out


def pack_gdn(upstream,new,terms):
    g=terms['mixer']
    return cpu({'upstream':upstream,'input':new,'m_mlp_norm_output':terms['m_mlp_norm_output'],
        'm_mixer_output':terms['m_mixer_output'],'m_mixer_input':terms['m_mixer_input'],
        **{k:g[k] for k in ['mnorm','mo_before_cast','mo_native','mz','coeff','mq','mk','mb','ma','mconv']},
        'mprojected':g['mprojected'][1::2]})


def gdn_decompose(m,change):
    """Existing16term ledger; explicitly convert EVERY term to prediction-actual."""
    d,c,e=change['d'],change['c'],change['e'];f=m['coeff'];u=m['upstream'];mm=m['m_mixer_output'];mi=m['m_mixer_input'];ml=m['m_mlp_norm_output']
    repeat=c['raw_q'].shape[2]//m['mq'].shape[2];assert repeat>=1
    qgroups=c['raw_q'].reshape(*c['raw_q'].shape[:2],m['mq'].shape[2],repeat,c['raw_q'].shape[-1])
    kgroups=c['raw_k'].reshape(*c['raw_k'].shape[:2],m['mk'].shape[2],repeat,c['raw_k'].shape[-1])
    assert bool(qgroups.eq(qgroups[:,:,:,:1,:]).all()) and bool(kgroups.eq(kgroups[:,:,:,:1,:]).all())
    qraw=dot(m['mq'],qgroups[:,:,:,0,:]);kraw=dot(m['mk'],kgroups[:,:,:,0,:]);qnorm=dot(f['q'],e['q']);knorm=dot(f['k'],e['k'])
    v=dot(f['v'],e['v']);beta=dot(f['beta'],e['beta']);g=dot(f['g'],e['raw_g']);mb=dot(m['mb'],c['b']);ma=dot(m['ma'],c['a']);mz=dot(m['mz'],c['z'])
    conv=dot(m['mconv'],c['conv_output']);projected=dot(m['mprojected'],c['projected_qkv']);normgate=dot(m['mnorm'],c['norm_output'])
    mo=dot(m['mo_before_cast'],e['o']);monative=dot(m['mo_native'],e['o'])
    old={
        'decoder_output_residual_rounding':dot(u,d['output'])-dot(u,d['post_norm_input'])-dot(u,d['mlp_output']),
        'MLP_combined':dot(u,d['mlp_output'])-dot(ml,d['post_norm_output']),
        'post_RMSNorm':dot(ml,d['post_norm_output'])-dot(mm.double()-u.double(),d['post_norm_input']),
        'mixer_residual_rounding':dot(mm,d['post_norm_input'])-dot(mm,d['input_norm_input'])-dot(mm,c['output']),
        'GDN_output_projection':dot(mm,c['output'])-normgate,'GDN_fused_norm_gate':normgate-mo-mz,
        'GDN_mo_BF16_cast':mo-monative,'GDN_FLA_including_raw_g_exp':monative-qnorm-knorm-v-beta-g,
        'GDN_QK_L2_and_head_fold':qnorm+knorm-qraw-kraw,'GDN_beta_sigmoid':beta-mb,'GDN_raw_g_parameter_map':g-ma,
        'GDN_conv_output_split':qraw+kraw+v-conv,'GDN_conv_linear_and_SiLU':conv-projected,
        'GDN_input_projections':projected+mz+mb+ma-dot(mi,c['input']),'mixer_input_alias':dot(mi,c['input'])-dot(mi,d['input_norm_output']),
        'input_RMSNorm':dot(mi,d['input_norm_output'])-dot(m['input'].double()-mm.double(),d['input_norm_input'])}
    terms={k:-value for k,value in old.items()};actual=dot(u,d['output']);pred=dot(m['input'],d['input_norm_input'])
    result={'sign_convention':'prediction_minus_actual','output_contraction':actual,'input_contraction':pred,
        'prediction_minus_actual':pred-actual,'actual_minus_predicted':actual-pred,'terms':terms,
        'telescoping_error':sum(terms.values())-(pred-actual),'FLA_branch_contractions':{'q':qnorm,'k':knorm,'v':v,'beta':beta,'raw_g':g},
        'norm_gate_branch_contractions':{'normalized_memory_before_cast':mo,'gate':mz}}
    assert abs(result['telescoping_error'])<1e-7,result
    return result


class CountFinite:
    def __init__(self,op):self.op=op;self.entered=0;self.returned=0
    def __call__(self,*args,**kw):
        self.entered+=1;out=self.op(*args,**kw);self.returned+=1;return out


class Observer:
    def __init__(self):
        self.boundaries={};self.layer={};self.scoring={};self.coarse_scoring={};self.receipts={};self.current=None;self.parameters={}
    def boundary(self,name,m,x):
        if name in BOUNDARIES:self.boundaries[name]={'m':cpu(m),'paired':cpu(x)}
    def wants_decoder(self,i):return i in (6,19)
    def decoder(self,i,d,c,e,upstream,new,terms):
        assert str(i) not in self.layer
        feature=features(i,d,c,e);coeff=fa_diag.pack_coeff(upstream,new,terms) if i==19 else pack_gdn(upstream,new,terms)
        self.layer[str(i)]={'paired':feature,'coeff':coeff}
        change=delta(endpoint(feature,1),endpoint(feature,0))
        self.layer[str(i)]['B2_decomposition']=fa_diag.decompose(coeff,change) if i==19 else gdn_decompose(coeff,change)
    def consume(self,index,dc,mc):
        assert dc.calls=={k:1 for k in ('input_norm','post_norm','gate','up','silu','down','mlp','decoder')}
        expected=({'module':1,'interface':1,'native_varlen':0,'native_dense':1} if index==19 else {'module':1,'conv':1,'FLA':1,'stage':1})
        assert mc.calls==expected,mc.calls
        f=features(index,dc.values,mc.values,getattr(mc,'endpoints',{}))
        if index==19:
            for name in ['query','key','value']:assert f['c'][name].shape[0]==1 and f['c'][name].shape[2]==info['total_length']
        else:assert f['e']['q'].shape[:2]==(1,info['total_length'])
        step=str(self.current);self.scoring.setdefault(step,{})[str(index)]=f
        self.receipts.setdefault(step,{})[str(index)]={'decoder_calls':dc.calls,'mixer_calls':mc.calls,
            'CPU_feature_bytes':nbytes(f),'initial_cache':getattr(mc,'initial_cache_receipt',None)}


class ScoringCapture:
    """Coarse module hooks plus exactly one selected-layer profiler at a time."""
    def __init__(self,model,observer):
        self.observer=observer;self.scope=None;self.post=None;self.handles=[]
        layers=model.model.language_model.layers;norm=model.model.language_model.norm
        for name in BOUNDARIES[:-2]:
            def boundary(module,args,kw,name=name):
                x=args[0] if args else kw['hidden_states']
                self.observer.coarse_scoring.setdefault(str(self.observer.current),{})[name]=cpu(x)
            self.handles.append(layers[int(name)].register_forward_pre_hook(boundary,with_kwargs=True))
        def finalnorm(module,args,out):
            dst=self.observer.coarse_scoring.setdefault(str(self.observer.current),{});dst['32']=cpu(args[0]);dst['norm']=cpu(out)
        self.handles.append(norm.register_forward_hook(finalnorm))
        for index in (6,19):
            def before(module,args,kw,index=index):
                assert self.scope is None and sys.getprofile() is None
                dc=NativeDecoderCapture(module,destination='cpu')
                mc=(NativeDenseAttentionCapture(module.self_attn,flash_attention_forward,flash_attn_varlen_func,flash_attn_func,destination='cpu')
                    if index==19 else NativeFreshCacheGDNCapture(module.linear_attn,device='cpu'))
                dc.__enter__()
                try:mc.__enter__()
                except Exception:mc.__exit__(None,None,None);dc.__exit__(None,None,None);raise
                self.scope=(index,dc,mc)
                self.post=module.register_forward_hook(self.after,always_call=True)
            self.handles.append(layers[index].register_forward_pre_hook(before,with_kwargs=True))
    def after(self,module,args,out):
        index,dc,mc=self.scope;mc.__exit__(None,None,None);dc.__exit__(None,None,None)
        self.post.remove();self.post=None;self.scope=None
        if out is None:raise RuntimeError('Selected actual native decoder failed.')
        self.observer.consume(index,dc,mc)
    def close(self):
        if self.scope is not None:
            _,dc,mc=self.scope;mc.__exit__(None,None,None);dc.__exit__(None,None,None);self.scope=None
        if self.post is not None:self.post.remove();self.post=None
        for h in self.handles:h.remove()
        self.handles=[]


try:
    def timeout(*args):raise TimeoutError('Frozen layer19/6 observation budget expired.')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_time_seconds']);r['sources_before']=sources()
    import numpy as np
    import torch,flash_attn
    torch.set_num_threads(4)
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    import flash_attn.flash_attn_interface as fa
    from flash_attn import flash_attn_func,flash_attn_varlen_func
    from transformers.integrations.flash_attention import flash_attention_forward
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import causal_conv1d
    from flashtrace.improved import keep_token_indices
    from llm_attr_eval import LLMAttributionEvaluator
    from qwen35_answer_finite import PackedAnswerTargets
    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner
    from qwen35_decoder_finite import NativeDecoderCapture
    from qwen35_gdn_finite import NativeGDNCapture
    from native_dense_attention_capture import NativeDenseAttentionCapture
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    import decoder19_conditional_decomposition_20260908 as fa_diag
    class NativeFreshCacheGDNCapture(NativeGDNCapture):
        def event(self,frame,kind,value):
            label=self.codes.get(frame.f_code);f=frame.f_locals
            if label=='module' and kind=='call' and f['self'] is self.module:
                assert not self.active and not self.values and not f.get('kwargs',{}).get('cu_seq_lens_q')
                cache=f.get('cache_params');previous=False if cache is None else bool(cache.has_previous_state(self.module.layer_idx));assert not previous
                self.initial_cache_receipt={'provided':cache is not None,'class':type(cache).__name__,'has_previous_state':previous,'layer_index':self.module.layer_idx}
                self.active=True;self.values['input']=self.copy(f['hidden_states']);self.values['mask']=self.copy(f.get('attention_mask'))
                self.calls['module']=self.calls.get('module',0)+1;return
            return super().event(frame,kind,value)
    assert p['capture_steps']==[0,1,10,20] and p['selected_layers']==[6,19]
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256'];assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule and native.is_fast_path_available;verify_native_sources(p['native_stage_source_sha256'])
    cp=Path(p['checkpoint'])
    for name,want in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==want
    stats=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats();assert r['weight_stats_before']==p['expected_weight_stats']
    tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    data={k:Path(v).read_bytes() for k,v in p['cache_paths'].items()}
    for k,raw in data.items():assert sha(raw)==p['cache_hashes'][k]
    raw=Path(p['frozen_metrics_path']).read_bytes();assert sha(raw)==p['frozen_metrics_sha256'];prior=json.loads(raw)
    assert prior['status']=='production_MH_original_metrics_complete'
    frozen=prior['cases']['morehopqa_1']['curves']['candidate']
    raw=Path(p['production_results_path']).read_bytes();assert sha(raw)==p['production_results_sha256'];production=json.loads(raw)
    raw=Path(p['production_vectors_path']).read_bytes();assert sha(raw)==p['production_vectors_sha256']
    with np.load(p['production_vectors_path'],allow_pickle=False) as archive:reference=archive['morehopqa_1_candidate_run7_full'].copy()
    def prepare(dataset,index):
        rec=json.loads(data[dataset].decode().splitlines()[index]);tok=tokenizer(rec['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        keep=keep_token_indices([rec['prompt'][a:b] for a,b in tok['offset_mapping']]);target=tokenizer(rec['target']+tokenizer.eos_token,add_special_tokens=False)['input_ids']
        ids=torch.tensor(tok['input_ids']+target,dtype=torch.long);base=ids.clone();base[keep]=tokenizer.eos_token_id
        inf={'input_sha256':sha(ids.numpy().tobytes()),'prompt_length':len(tok['input_ids']),'target_length':len(target),'total_length':len(ids),'keep':keep}
        return rec,ids,base,torch.tensor(target),inf
    control=prepare('niah_mq_q2',0);rec,ids,base,target,info=prepare('morehopqa',1)
    assert info==prior['cases']['morehopqa_1']['input']==production['cases']['morehopqa_1']['input'];r['input']=info
    assert sha(base.numpy().tobytes())==production['cases']['morehopqa_1']['baseline_sha256']
    assert torch.cuda.mem_get_info()[0]>=32*1024**3
    torch.manual_seed(73);r['status']='loading';save()
    model,loading=timed('actual_model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,
        attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True))
    assert not any(loading.values());model.eval().requires_grad_(False);r['model_loads']=1
    layers=model.model.language_model.layers;assert layers[19].block_type=='full_attention' and layers[6].block_type=='linear_attention'
    for layer in layers:
        if layer.block_type=='linear_attention':
            assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule and layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
    with torch.no_grad():warm=timed('original_order_native_eager_B1_diagnostic',lambda:model(input_ids=control[1][None].to('cuda'),
        attention_mask=torch.ones_like(control[1][None],device='cuda'),use_cache=False))
    r['native_eager_diagnostics']=1;del warm,control;model.set_attn_implementation('flash_attention_2')
    finite_fa=CountFinite(VendorFAFiniteP1BF16D256(p['finite_FA_library'],p['finite_FA_library_sha256']))
    finite_fla=CountFinite(make_compiled_finite_pullback(reuse_scalar_products=False))
    runner=Qwen35DenseFiniteRunner(model,finite_fa,finite_fla,norm_gate_rules={0:'symmetric'});observer=Observer()
    for i in (6,19):
        norm_parameters=lambda module:{'weight':cpu(module.weight),'eps':float(module.eps)}
        small={'input_norm':norm_parameters(layers[i].input_layernorm),'post_norm':norm_parameters(layers[i].post_attention_layernorm),
            'input_post_weight_semantics':'Original raw native weight; finite Qwen RMS uses1+weight.'}
        if i==19:
            small.update(q_norm=norm_parameters(layers[i].self_attn.q_norm),k_norm=norm_parameters(layers[i].self_attn.k_norm),qk_weight_semantics='Original raw native weights; finite Qwen RMS uses1+weight.')
        else:small.update(fused_norm_gate=norm_parameters(layers[i].linear_attn.norm),fused_weight_semantics='Original direct multiplicative weight,not1+weight.')
        observer.parameters[str(i)]=small
    r['additional_private_norm_parameter_CPU_bytes']=nbytes(observer.parameters)
    pairs=torch.stack((base,ids)).to('cuda');mask=torch.ones_like(pairs)
    selection=PackedAnswerTargets([{'target_ids':target,'prompt_length':info['prompt_length']}],[list(range(len(target)))],len(ids),'cuda')
    r['status']='one_normal_DT_with_two_existing_diagnostic_boundaries';r['DT_calls_entered']+=1;save()
    signed,details=timed('actual_DT_with_passive_boundary_diagnostics',lambda:runner.attribute(pairs,mask,selection,select_output_rows=True,observer=observer))
    r['DT_calls']+=1;r['DT_details']=details;assert details['norm_gate_rules']=={'0':'symmetric'}
    assert finite_fa.entered==finite_fa.returned==8 and finite_fla.entered==finite_fla.returned==24
    kinds=[q['kind'] for q in details['calls']]
    assert sum(k.startswith('native_replay_') for k in kinds)==32 and sum(k.startswith('finite_decoder_') for k in kinds)==32
    assert sum(k.startswith('public_FA_LSE_') for k in kinds)==8
    assert set(observer.boundaries)==set(BOUNDARIES) and set(observer.layer)=={'6','19'}
    signed=signed[0];evaluated=signed[:info['prompt_length']].float();assert bool(torch.isfinite(signed).all())
    r['production_vector_drift_report_only']={'relative_L2':float((signed-torch.from_numpy(reference)).norm()/torch.from_numpy(reference).norm()),
        'max_absolute':float((signed-torch.from_numpy(reference)).abs().max()),'bitwise_equal':bool(torch.equal(signed,torch.from_numpy(reference))),
        'policy':'No historical2percent admission guard; current actual coefficients explain the frozen most-recent production candidate deletion sets.'}
    np.savez_compressed(A/'vectors.npz',current_full=signed.numpy(),current_evaluated=evaluated.numpy())
    r['B2_decompositions']={i:x['B2_decomposition'] for i,x in observer.layer.items()}
    r['B2_boundary_contractions']={name:dot(value['m'],value['paired'][1::2].double()-value['paired'][0::2].double()) for name,value in observer.boundaries.items()}
    r['B2_actual_root_effect']=details['root_effect'];r['prior_production_root_effect']=production['runs'][7]['production_details']['root_effect']
    del pairs,mask,selection;gc.collect();evaluator=LLMAttributionEvaluator(model,tokenizer);capture=ScoringCapture(model,observer)
    def before_score(module,args,kw):
        assert r['scoring_forwards_entered']<4;r['scoring_forwards_entered']+=1
        x=kw['input_ids'].detach().cpu();receipt=frozen['input_receipts'][observer.current]
        assert x.shape==(1,len(ids)) and bool(kw['attention_mask'].eq(1).all())
        assert sha(x.numpy().tobytes())==receipt['input_sha256'] and (x[0]!=ids).nonzero().flatten().tolist()==receipt['deleted_positions']
    handles=[model.register_forward_pre_hook(before_score,with_kwargs=True)]
    for step in p['capture_steps']:
        observer.current=step;receipt=frozen['input_receipts'][step];prompt=ids[:info['prompt_length']].clone();prompt[receipt['deleted_positions']]=tokenizer.eos_token_id
        point={'input_receipt':receipt};r['points'][str(step)]=point;r['status']='four_frozen_original_scores_step'+str(step);save()
        with torch.no_grad():lp=timed('original_frozen_scorer_'+str(step),lambda:evaluator.compute_logprob_response_given_prompt(prompt[None].to('cuda'),target[None].to('cuda')))
        r['scoring_forwards_returned']+=1;point['actual_native_score']=float(lp.sum().cpu());point['actual_native_logprob_dtype']=str(lp.dtype)
        point['prior_actual_native_score']=frozen['scores'][step];point['native_score_minus_prior']=point['actual_native_score']-point['prior_actual_native_score']
        assert set(observer.scoring[str(step)])=={'6','19'} and set(observer.coarse_scoring[str(step)])==set(BOUNDARIES)
        point['capture_receipts']=observer.receipts[str(step)]
        if step:
            effect=r['points']['0']['actual_native_score']-point['actual_native_score'];deleted=receipt['deleted_positions']
            pred= float(evaluated[deleted].double().sum());point['complete_input']={'actual_effect':effect,'evaluated_prediction':pred,
                'full_signed_prediction':float(signed[deleted].sum()),'prediction_minus_actual':pred-effect}
            point['layer_decompositions']={}
            for index in (6,19):
                key=str(index);change=delta(observer.scoring['0'][key],observer.scoring[str(step)][key]);coeff=observer.layer[key]['coeff']
                point['layer_decompositions'][key]=fa_diag.decompose(coeff,change) if index==19 else gdn_decompose(coeff,change)
            contractions={name:dot(observer.boundaries[name]['m'],observer.coarse_scoring['0'][name].double()-observer.coarse_scoring[str(step)][name].double()) for name in BOUNDARIES}
            regions={left+'_to_'+right:contractions[left]-contractions[right] for left,right in zip(BOUNDARIES,BOUNDARIES[1:])}
            regions['norm_to_actual_score']=contractions['norm']-effect
            point['coarse']={'sign_convention':'prediction_minus_actual','boundary_contractions':contractions,'regions':regions,
                'prediction_minus_actual':contractions['0']-effect,'telescoping_error':sum(regions.values())-(contractions['0']-effect),
                'evaluated_input_cast_difference':pred-contractions['0']}
            assert abs(point['coarse']['telescoping_error'])<1e-7
        else:
            point['B1_clean_vs_B2_input_boundary_relative_L2']={name:float((observer.coarse_scoring['0'][name].double()-observer.boundaries[name]['paired'][1::2].double()).norm()/observer.boundaries[name]['paired'][1::2].double().norm().clamp_min(1e-30)) for name in BOUNDARIES}
        del lp;save()
    capture.close();capture=None
    for h in handles:h.remove()
    handles=[];assert r['DT_calls']==1 and r['scoring_forwards_entered']==r['scoring_forwards_returned']==4
    r['sources_after']=sources();r['weight_stats_after']=stats();assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256'] and sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    r['status']='decoder19_6_actual_conditional_observation_complete'
except Exception:
    r['status']='failed';r['error']=traceback.format_exc()
finally:
    if capture is not None:capture.close()
    sys.setprofile(None);signal.alarm(0)
    for h in handles:h.remove()
    r['finite_callback_counts']={name:{'entered':op.entered if op is not None else 0,'returned':op.returned if op is not None else 0} for name,op in [('finite_FA',finite_fa),('finite_FLA',finite_fla)]}
    r['partial_native_count_policy']='Unreturned DT has no completed replay ledger; partial native work is unknown,not zero. The entered/returned attribution callbacks and scorer counts remain recorded.'
    if observer is not None:
        artifact=A/'MH1_decoder19_6_actual_private.pt'
        try:
            stored={'layers':observer.layer,'boundaries':observer.boundaries,'scoring':observer.scoring,'coarse_scoring':observer.coarse_scoring,'small_native_norm_parameters':observer.parameters}
            torch.save(stored,artifact);digest=hashlib.sha256()
            with artifact.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
            r['private_artifact']={'file':artifact.name,'sha256':digest.hexdigest(),'bytes':artifact.stat().st_size,'tensor_CPU_bytes':nbytes(stored)}
        except Exception:r['private_artifact_error']=traceback.format_exc();r['status']='failed'
    r['job_seconds']=time.perf_counter()-started;save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json','vectors.npz']:
            if (A/name).exists():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'DT_calls':r['DT_calls'],'scoring_forwards':r['scoring_forwards_returned'],'seconds':r['job_seconds'],'error':r.get('error')}),flush=True)
