"""MH0 saved-boundary FA19 localization with one official kwargs-capture forward."""
import os,sys,json,time,hashlib,traceback,zipfile,signal,gc
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
    TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
r={'status':'starting','protocol':p,'model_loads':0,'whole_model_entered':0,'whole_model_returned':0,
    'layer_replays_entered':0,'layer_replays_returned':0,'finite_layer_entered':0,'finite_layer_returned':0,
    'auxiliary_FA_entered':0,'auxiliary_FA_returned':0,'DT_calls':0,'scorer_calls':0,'FT_calls':0,'generation_calls':0,
    'calls':[],'points':{}}
started=time.perf_counter();stored={};vectors={};handles=[];finite_fa=None

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save():
    q=A/'results.partial';q.write_text(json.dumps(r,indent=2,allow_nan=False));q.replace(A/'results.json')
def sources():
    out={}
    for name,want in p['files_sha256'].items():out[name]=digest(A/name);assert out[name]==want,name
    for name,want in p['official_source_blob_sha1'].items():
        raw=(Path(p['official_root'])/name).read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want,name
        out['official/'+name]=sha(raw)
    for name,want in p['runtime_source_sha256'].items():
        out['native/'+name]=digest(Path(p['isolated_site'])/name);assert out['native/'+name]==want,name
    for item in p['protected_sources']:
        out[item['path']]=digest(item['path']);assert out[item['path']]==item['sha256']
    return out
def timed(kind,fn):
    torch.cuda.synchronize();tick=time.perf_counter();row={'kind':kind,'status':'entered'};r['calls'].append(row)
    try:
        value=fn();torch.cuda.synchronize();row['status']='returned';return value
    finally:row['seconds']=time.perf_counter()-tick

def copy_to(x,device):
    if isinstance(x,torch.Tensor):return x.detach().to(device,copy=True)
    if isinstance(x,dict):return {k:copy_to(v,device) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return type(x)(copy_to(v,device) for v in x)
    assert x is None or isinstance(x,(str,int,float,bool)),type(x)
    return x
cpu=lambda x:copy_to(x,'cpu')
def nbytes(x):
    if isinstance(x,torch.Tensor):return x.numel()*x.element_size()
    if isinstance(x,dict):return sum(nbytes(v) for v in x.values())
    if isinstance(x,(tuple,list)):return sum(nbytes(v) for v in x)
    return 0
def drift(x,y):
    assert x.shape==y.shape and bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
    dx=x.double()-y.double();den=y.double().norm()
    return {'bitwise_equal':bool(torch.equal(x,y)),'max_absolute':float(dx.abs().max()),'relative_L2':float(dx.norm()/den.clamp_min(1e-30))}
def contract(m,x):
    assert m.shape==x.shape and m.ndim==3
    return (m.double()*x.double()).sum(-1)
def stat(x):
    assert bool(torch.isfinite(x).all())
    return {'net':float(x.sum()),'positive':float(x.clamp_min(0).sum()),'negative':float(x.clamp_max(0).sum())}
class CountFinite:
    def __init__(self,op):self.op=op;self.entered=0;self.returned=0
    def __call__(self,*args,**kw):
        assert self.entered<1;self.entered+=1;out=self.op(*args,**kw);self.returned+=1;return out

try:
    def timeout(*args):raise TimeoutError('Frozen MH0 FA19 internal diagnostic expired.')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_time_seconds']);r['sources_before']=sources()
    import numpy as np
    import torch,flash_attn
    torch.set_num_threads(4)
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    from transformers.integrations.flash_attention import flash_attention_forward
    from flash_attn import flash_attn_func,flash_attn_varlen_func
    import flash_attn.flash_attn_interface as fa
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import causal_conv1d
    from qwen35_decoder_finite import NativeDecoderCapture,FiniteBoundaryOps,decoder_finite_pullback,attention_finite_pullback
    from native_dense_attention_capture import NativeDenseAttentionCapture
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256,RightPaddedLengths
    from finite_fla_gpu import verify_native_sources
    import decoder19_conditional_decomposition_20260908 as ledger
    assert p['layer']==19 and p['capture_steps']==['B2','0','3','10','20']
    assert digest(native.__file__)==p['native_model_sha256'] and digest(fa.__file__)==p['installed_FA_interface_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule and native.is_fast_path_available
    verify_native_sources(p['native_stage_source_sha256'])
    source=p['source_boundary'];assert digest(source['results_path'])==source['results_sha256']
    prior=json.loads(Path(source['results_path']).read_bytes());assert prior['status']=='MH0_current_conditional_boundaries_1DT4score_complete'
    case=prior['cases']['morehopqa_0'];r['input']=case['input'];assert r['input']==p['input']
    private=Path(source['private_path']);assert digest(private)==source['private_sha256'] and private.stat().st_size==source['private_bytes']
    original=torch.load(private,map_location='cpu',weights_only=True,mmap=True)
    assert original['source_protocol_sha256']==source['protocol_sha256']
    saved={'coefficients':{b:original['coefficients'][b] for b in ['19','20']},
        'paired':{b:original['paired_native_boundaries'][b] for b in ['19','20']},
        'B1':{s:{b:original['B1_native_boundaries'][s][b] for b in ['19','20']} for s in ['0','3','10','20']}}
    stored['source_boundaries']=saved;del original;gc.collect()
    freeze=prior['input_freeze_before_model_load']['morehopqa_0'];ids=torch.tensor(freeze['input_ids'],dtype=torch.long)
    assert sha(ids.numpy().tobytes())==r['input']['input_sha256'];base=ids.clone();base[r['input']['keep']]=ids[-1]
    assert sha(base.numpy().tobytes())==freeze['baseline_sha256'];pairs=torch.stack((base,ids)).to('cuda')
    cp=Path(p['checkpoint'])
    for name,want in p['checkpoint_config_tokenizer_sha256'].items():assert digest(cp/name)==want
    weights=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
    r['weight_stats_before']=weights();assert r['weight_stats_before']==p['expected_weight_stats']
    assert torch.cuda.mem_get_info()[0]>=32*1024**3;torch.manual_seed(73)
    model,loading=timed('official_full_model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,
        attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True))
    assert not any(loading.values());model.eval().requires_grad_(False);r['model_loads']=1;model.set_attn_implementation('flash_attention_2')
    language=model.model.language_model;layer=language.layers[19];assert layer.block_type=='full_attention' and language.config.use_cache is True
    for ll in language.layers:
        if ll.block_type=='linear_attention':
            assert ll.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule and ll.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
    capture={}
    def before_root(module,args,kw):
        assert r['whole_model_entered']<1 and torch.equal(kw['input_ids'],pairs) and bool(kw['attention_mask'].eq(1).all())
        assert kw['use_cache'] is False and kw['logits_to_keep']==1;r['whole_model_entered']+=1
    def after_root(module,args,out):
        if out is not None:r['whole_model_returned']+=1
    def before_layer(module,args,kw):
        assert not capture;capture['kwargs']=cpu({k:v for k,v in kw.items() if k!='hidden_states'})
        capture['input']=cpu(args[0] if args else kw['hidden_states'])
    def after_layer(module,args,out):capture['output']=cpu(out)
    handles=[model.register_forward_pre_hook(before_root,with_kwargs=True),model.register_forward_hook(after_root,always_call=True),
        layer.register_forward_pre_hook(before_layer,with_kwargs=True),layer.register_forward_hook(after_layer)]
    r['status']='one_official_B2_forward_capture_FA19_kwargs';save()
    try:
        with torch.no_grad():out=timed('official_B2_forward_for_actual_layer19_kwargs',lambda:model(input_ids=pairs,
            attention_mask=torch.ones_like(pairs),use_cache=False,logits_to_keep=1))
        assert out.logits.shape[:2]==(2,1)
    finally:
        for handle in handles:handle.remove()
        handles=[]
    assert r['whole_model_entered']==r['whole_model_returned']==1
    kw2=capture['kwargs'];assert kw2['past_key_values'] is None and kw2['use_cache'] is False and kw2['attention_mask'] is None
    assert set(p['required_decoder_kwargs']).issubset(kw2),list(kw2)
    assert all(value is None or isinstance(value,(str,int,float,bool)) for key,value in kw2.items() if key not in p['required_decoder_kwargs'])
    pos=kw2['position_ids'];cos,sin=kw2['position_embeddings'];T=r['input']['total_length']
    assert pos.shape==(2,T) and cos.shape==sin.shape and cos.shape[:2]==(2,T)
    assert torch.equal(pos[0],pos[1]) and torch.equal(cos[0],cos[1]) and torch.equal(sin[0],sin[1])
    kw1=dict(kw2,position_ids=pos[1:2].clone(),position_embeddings=(cos[1:2].clone(),sin[1:2].clone()),use_cache=True)
    r['kwargs_capture']={'whole_native_forward_input_sha256':sha(pairs.detach().cpu().numpy().tobytes()),
        'actual_keys':list(kw2),'position_ids_shape':list(pos.shape),'cos_sin_shape':list(cos.shape),'actual_FA_mask':None,
        'paired_position_rows_equal':True,'B1_mapping':'Take actual clean row of identical native B2 positions/cos/sin, preserving None FA mask. Fresh original DynamicCache per B1 replay; no continued state, position arange, rotary or mask implementation.',
        'native_cache_class':native.DynamicCache.__module__+'.'+native.DynamicCache.__name__,
        'captured_native_B2_layer19_input_drift':drift(capture['input'],saved['paired']['19']),
        'captured_native_B2_layer19_output_drift':drift(capture['output'],saved['paired']['20'])}
    stored['official_B2_kwargs']=kw2;stored['B1_kwargs_without_cache']=kw1
    del out,pairs,capture;gc.collect()
    def forbid_full(*args,**kw):raise RuntimeError('Only one actual kwargs-capture whole forward authorized.')
    handles=[model.register_forward_pre_hook(forbid_full)]
    boundaries=FiniteBoundaryOps(True);finite_fa=CountFinite(VendorFAFiniteP1BF16D256(p['finite_FA_library'],p['finite_FA_library_sha256']))
    upstream=saved['coefficients']['20'].to('cuda').contiguous();stored['native']={}
    for name in p['capture_steps']:
        xcpu=saved['paired']['19'] if name=='B2' else saved['B1'][name]['19']
        expected=saved['paired']['20'] if name=='B2' else saved['B1'][name]['20']
        x=xcpu.to('cuda').contiguous();kw=copy_to(kw2 if name=='B2' else kw1,'cuda')
        cache=native.DynamicCache(config=language.config) if name!='B2' else None;kw['past_key_values']=cache
        if cache is not None:assert cache.get_seq_length(19)==0
        assert x.dtype==torch.bfloat16 and x.shape[1]==T
        dc=NativeDecoderCapture(layer,destination='cuda');mc=NativeDenseAttentionCapture(layer.self_attn,flash_attention_forward,flash_attn_varlen_func,flash_attn_func,destination='cuda')
        def replay():
            with torch.no_grad(),dc,mc:return layer(x,**kw)
        assert r['layer_replays_entered']<5;r['layer_replays_entered']+=1;r['status']='native_FA19_'+name;save()
        y=timed('native_layer19_'+name,replay);r['layer_replays_returned']+=1
        assert dc.calls=={k:1 for k in ['input_norm','post_norm','gate','up','silu','down','mlp','decoder']}
        assert mc.calls=={'module':1,'interface':1,'native_varlen':0,'native_dense':1}
        d,c=dc.values,mc.values;feature=ledger.features(d,c)
        feature['d'].update({k:cpu(d[k]) for k in ['gate_output','up_output','silu_output','down_input']})
        stored['native'][name]=feature
        r['points'][name]={'output_replay_drift':drift(cpu(y),expected),'decoder_calls':dc.calls,'mixer_calls':mc.calls,
            'native_dense_arguments':mc.dense_arguments,'native_interface_arguments':mc.interface_arguments,
            'initial_cache':{'provided':cache is not None,'selected_layer_initial_length':0,'selected_layer_final_length':None if cache is None else cache.get_seq_length(19)},
            'native_feature_CPU_bytes':nbytes(feature)}
        if cache is not None:assert cache.get_seq_length(19)==T
        if name=='B2':
            args={k:v for k,v in mc.dense_arguments.items() if k!='return_attn_probs'}
            assert args['dropout_p']==0 and args['causal']
            assert r['auxiliary_FA_entered']==0;r['auxiliary_FA_entered']+=1
            with torch.no_grad():aux,lse,unused=timed('public_FA_LSE_layer19',lambda:flash_attn_func(c['dense_q'],c['dense_k'],c['dense_v'],return_attn_probs=True,**args))
            r['auxiliary_FA_returned']+=1;assert unused is None or not unused.numel()
            r['B2_public_FA_auxiliary_output_drift']=drift(cpu(aux),feature['c']['attention_output']);del aux,unused
            layout=RightPaddedLengths([T],T,'cuda');cos_gpu,sin_gpu=kw['position_embeddings']
            def mixer(m):return attention_finite_pullback(layer.self_attn,c,lse,cos_gpu,sin_gpu,m,finite_fa,layout,boundaries,True)
            assert r['finite_layer_entered']==0;r['finite_layer_entered']+=1;r['status']='one_current_finite_FA19';save()
            with torch.no_grad():new,terms=timed('current_finite_decoder19_existing_diagnostics',lambda:decoder_finite_pullback(layer,d,upstream,mixer,boundaries,True))
            r['finite_layer_returned']+=1;assert finite_fa.entered==finite_fa.returned==1 and bool(torch.isfinite(new).all())
            stored['coeff']=ledger.pack_coeff(upstream,new,terms);stored['m19_new']=cpu(new)
            r['m19_replay_drift_report_only']=drift(stored['m19_new'],saved['coefficients']['19'])
            r['FA_activity']=terms['mixer']['finite_FA_activity'];del new,terms,lse
        del x,y,kw,cache,d,c,dc,mc;gc.collect();save()
    assert r['layer_replays_entered']==r['layer_replays_returned']==5 and r['finite_layer_entered']==r['finite_layer_returned']==1
    assert r['auxiliary_FA_entered']==r['auxiliary_FA_returned']==1
    for step in ['3','10','20','B2']:
        if step=='B2':
            f=stored['native']['B2'];change=ledger.difference(ledger.endpoints(f,1),ledger.endpoints(f,0))
            dx=saved['paired']['19'][1:2].double()-saved['paired']['19'][0:1].double()
            dy=saved['paired']['20'][1:2].double()-saved['paired']['20'][0:1].double();deleted=set(r['input']['keep'])
        else:
            change=ledger.difference(stored['native']['0'],stored['native'][step])
            dx=saved['B1']['0']['19'].double()-saved['B1'][step]['19'].double()
            dy=saved['B1']['0']['20'].double()-saved['B1'][step]['20'].double();deleted=set(case['points'][step]['input_receipt']['deleted_positions'])
        # Reuse the frozen helper's public grouping contract for singleton token
        # contractions; no duplicated FA/GQA algebra or extra operator evaluation.
        dec=ledger.decompose(stored['coeff'],change,groups={str(i):[i] for i in range(T)})
        token_groups=dec.pop('token_groups');tokens={k:torch.tensor([[token_groups[str(i)]['terms'][k] for i in range(T)]],dtype=torch.float64) for k in dec['terms']}
        original_error=contract(saved['coefficients']['19'],dx)-contract(saved['coefficients']['20'],dy)
        transfer={'saved_minus_replayed_m19':contract(saved['coefficients']['19'].double()-stored['m19_new'].double(),dx),
            'input_replay_transfer':contract(stored['m19_new'],dx-change['d']['input_norm_input']),
            'output_replay_transfer':contract(saved['coefficients']['20'],change['d']['output']-dy)}
        all_tokens={**tokens,**transfer,'saved_boundary_error':original_error};full=sum(tokens.values())+sum(transfer.values())
        assert float((full-original_error).abs().max())<1e-7
        target=-(prior['runs'][0]['B2_endpoint_boundary_contractions']['20']-prior['runs'][0]['B2_endpoint_boundary_contractions']['19']) if step=='B2' else -case['decomposition'][step]['decoder_errors']['19']
        assert abs(float(original_error.sum())-target)<1e-7
        P=r['input']['prompt_length'];keep=set(r['input']['keep'])
        groups={'deleted':sorted(deleted),'kept':sorted(keep-deleted),'other_prompt':sorted(set(range(P))-keep),'response':list(range(P,T))}
        report={'replayed_9term_ledger':dec,'saved_boundary_error':stat(original_error),
            'transfer_terms':{k:stat(z) for k,z in transfer.items()},'closure_max_absolute':float((full-original_error).abs().max()),'coordinate_groups':{}}
        for key,value in all_tokens.items():
            vectors[step+'_'+key]=value.numpy();total=stat(value);gs={k:{'count':len(ix),**stat(value[:,ix])} for k,ix in groups.items()}
            assert abs(sum(z['net'] for z in gs.values())-total['net'])<1e-7;report['coordinate_groups'][key]={'total':total,'groups':gs}
        r.setdefault('conditional_ledgers',{})[step]=report
    np.savez_compressed(A/'vectors.npz',**vectors)
    assert digest(native.__file__)==p['native_model_sha256'] and digest(fa.__file__)==p['installed_FA_interface_sha256']
    r['sources_after']=sources();r['weight_stats_after']=weights()
    assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    assert r['DT_calls']==r['scorer_calls']==r['FT_calls']==r['generation_calls']==0
    r['status']='MH0_current_FA19_1native_kwargs5replay1finite_internal_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    sys.setprofile(None);signal.alarm(0)
    for handle in handles:handle.remove()
    r['finite_FA_counts']={'entered':finite_fa.entered if finite_fa else 0,'returned':finite_fa.returned if finite_fa else 0}
    r['native_call_accounting']={'FA_from_returned_whole_forwards':8*r['whole_model_returned'],
        'GDN_FLA_and_conv_each_from_returned_whole_forwards':24*r['whole_model_returned'],
        'FA_from_returned_single_layer_replays':r['layer_replays_returned'],
        'nonreturned_whole_or_layer_internal_calls':'unknown' if r['whole_model_entered']!=r['whole_model_returned'] or r['layer_replays_entered']!=r['layer_replays_returned'] else 0,
        'nonreturned_finite_native_stages':'unknown' if r['finite_FA_counts']['entered']!=r['finite_FA_counts']['returned'] else 0}
    if 'torch' in globals():
        r['GPU_peak_allocated_full_job']=torch.cuda.max_memory_allocated();r['GPU_peak_reserved_full_job']=torch.cuda.max_memory_reserved()
    if stored:
        try:
            artifact=A/'MH0_current_FA19_internal_private.pt';tick=time.perf_counter();torch.save(stored,artifact)
            r['private_artifact']={'file':artifact.name,'sha256':digest(artifact),'bytes':artifact.stat().st_size,'tensor_CPU_bytes':nbytes(stored),'save_and_hash_seconds':time.perf_counter()-tick}
        except Exception:r['private_artifact_error']=traceback.format_exc();r['status']='failed'
    r['scope']='Existing MH0 coefficients/current finite FA19 rule only. Replayed 9term internal ledger plus explicit coefficient/input/output transfer equals saved actual whole-pass boundary error. Core includes seed cast;output projection+sigmoid gate and QKnorm/RoPE/GQA/projections remain grouped. No precise routing/PV subcause or candidate-benefit claim yet.'
    r['cost_scope']='One official full BF16 model load, one original FA B2 whole forward solely for native layer kwargs, five native decoder19 replays,one existing finite decoder19 pullback,one publicFA LSE aux. No whole DT/scorer/FT/generation or extra finiteFLA/conv. Passive captures and CPU singleton group contractions/private storage included;not production speed.'
    r['seconds']=time.perf_counter()-started;save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/name).is_file():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'seconds':r['seconds'],'whole':r['whole_model_returned'],'replays':r['layer_replays_returned'],'finite':r['finite_layer_returned'],'error':r.get('error')}),flush=True)
