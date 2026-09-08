"""Focused passive layer0 localization on frozen original scoring inputs.

Two DT calls and eight original scorer calls; no method, FT or model replacement.
Conv preactivation and SiLU remain combined: no extra preactivation calls.
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
                'e':{k:e[k] for k in ['q','k','v','beta','raw_g','o']}})

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
        value=features(dc.values,mc.values,mc.endpoints)
        self.captures[str(self.current)]={'decoder_calls':dict(dc.calls),'mixer_calls':dict(mc.calls),
            'retained_feature_CPU_bytes':tensor_bytes(value),'initial_cache':mc.initial_cache_receipt}
        if self.current==0:self.clean=value
        else:self.points[str(self.current)]=self.decompose(difference(self.clean,value))

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
    from flashtrace.improved import keep_token_indices
    from llm_attr_eval import LLMAttributionEvaluator
    from qwen35_answer_finite import PackedAnswerTargets
    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner
    from qwen35_decoder_finite import NativeDecoderCapture
    from qwen35_gdn_finite import NativeGDNCapture
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
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
    assert p['capture_steps']==[0,1,10,20] and p['observer_vector_relative_L2_ceiling']==0.02
    assert p['case_indices']==[['niah_mq_q2',1],['morehopqa',1]]
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
    runner=Qwen35DenseFiniteRunner(model,VendorFAFiniteP1BF16D256(p['finite_FA_library'],p['finite_FA_library_sha256']),
        make_compiled_finite_pullback(reuse_scalar_products=False))
    evaluator=LLMAttributionEvaluator(model,tokenizer);layer0=model.model.language_model.layers[0]
    assert layer0.block_type=='linear_attention'
    for key,rec,ids,base,target,info in cases:
        row={'input':info,'scoring_points':{}};r['cases'][key]=row;observer=Layer0Observer()
        pairs=torch.stack((base,ids)).to('cuda');mask=torch.ones_like(pairs)
        selection=PackedAnswerTargets([{'target_ids':target,'prompt_length':info['prompt_length']}],
            [list(range(len(target)))],len(ids),'cuda')
        r['status']='focused_actual_coefficients_'+key;save()
        signed,details=runner.attribute(pairs,mask,selection,select_output_rows=True,observer=observer);r['DT_calls']+=1
        signed=signed[0];vectors[key+'_DT_full']=signed.numpy();np.savez_compressed(A/'vectors.npz',**vectors)
        reference=torch.from_numpy(parent_vectors[key+'_DT_full'])
        relative=float((signed-reference).norm()/reference.norm());row['parent_vector_relative_L2']=relative
        row['parent_vectors_equal']=bool(torch.equal(signed,reference));row['DT']=details
        row['B2_endpoint_decomposition']=observer.B2;row['coefficient_CPU_bytes']=tensor_bytes(observer.coeff)
        row['paired_feature_CPU_bytes']=tensor_bytes(observer.paired);save()
        assert relative<=p['observer_vector_relative_L2_ceiling'],'Frozen 2% vector guard failed; saved before stopping.'
        assert torch.equal(observer.boundary_coeff['0'],observer.coeff['input'])
        assert torch.equal(observer.boundary_coeff['1'],observer.coeff['upstream'])
        coeff_path=A/(key+'_layer0_coefficients.pt');torch.save(observer.coeff,coeff_path)
        row['coefficient_artifact']={'file':coeff_path.name,'sha256':sha(coeff_path.read_bytes()),'bytes':coeff_path.stat().st_size}
        del pairs,mask,selection,reference;capture=ScoringLayer0Capture(layer0,observer)
        def before_model(_module,args,kw):
            r['scoring_forwards']+=1;x=kw['input_ids'].detach().cpu();step=observer.current
            frozen=prior['cases'][key]['curve']['input_receipts'][step]
            assert x.shape==(1,len(ids)) and bool(kw['attention_mask'].eq(1).all())
            assert sha(x.numpy().tobytes())==frozen['input_sha256']
            deleted=(x[0]!=ids).nonzero().flatten().tolist();assert deleted==frozen['deleted_positions']
            row['scoring_points'][str(step)]={'input_receipt':frozen}
        handles=[model.register_forward_pre_hook(before_model,with_kwargs=True)]
        r['status']='eight_total_frozen_scoring_points_'+key;save()
        try:
            for step in p['capture_steps']:
                observer.current=step;frozen=prior['cases'][key]['curve']['input_receipts'][step]
                prompt=ids[:info['prompt_length']].clone();prompt[frozen['deleted_positions']]=tokenizer.eos_token_id
                with torch.no_grad():lp=timed('original_scorer_'+key+'_step'+str(step),lambda:evaluator.compute_logprob_response_given_prompt(
                    prompt[None].to('cuda'),target[None].to('cuda')))
                row['scoring_points'][str(step)]['original_native_score']=float(lp.sum().cpu())
                row['scoring_points'][str(step)]['prior_original_native_score']=prior['cases'][key]['curve']['scores'][step]
                row['scoring_points'][str(step)]['capture']=observer.captures[str(step)]
                if step:
                    row['scoring_points'][str(step)]['layer0_decomposition']=observer.points[str(step)]
                    row['scoring_points'][str(step)]['prior_layer0_error']=prior['cases'][key]['decomposition'][str(step)]['decoder_errors']['0']
                save();del lp
        finally:
            capture.close();capture=None
            for h in handles:h.remove()
            handles=[]
        assert set(observer.points)=={'1','10','20'}
        row['B1_allEOS_endpoint_decomposition']=observer.points['20']
        row['B1_allEOS_minus_B2_endpoint_terms']={k:observer.points['20']['terms'][k]-observer.B2['terms'][k] for k in observer.B2['terms']}
        row['limitation']='Conv preactivation/SiLU combined; no extra model or conv calls. FLA includes raw-g exp. Terms are signed conditional prediction errors, not causal token signs.'
        save();del observer,signed;gc.collect()
    assert r['DT_calls']==2 and r['scoring_forwards']==8 and r['extra_conv_preactivation_calls']==0
    r['sources_after']=sources();r['weight_stats_after']=stats()
    assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    r['status']='layer0_conditional_complete'
except Exception:
    r['status']='failed';r['error']=traceback.format_exc()
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
