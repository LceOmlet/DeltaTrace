"""Five actual layer0 replays and one current finite pullback; no full forward."""
import os,sys,json,time,hashlib,traceback,zipfile,signal,gc,inspect
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
    TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
r={'status':'starting','model_loads':0,'layer_replays_entered':0,'layer_replays_returned':0,'finite_layer_entered':0,'finite_layer_returned':0,
   'whole_model_forwards':0,'DT_calls':0,'scorer_calls':0,'FA_calls':0,'FT_calls':0,'generation_calls':0,'calls':[],'points':{},'protocol':p}
started=time.perf_counter();stored={};vectors={};finite_fla=None;full_guard=None

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def save():
    f=A/'results.partial';f.write_text(json.dumps(r,indent=2,allow_nan=False));f.replace(A/'results.json')

def sources():
    out={}
    for name,want in p['files_sha256'].items():out[name]=digest(A/name);assert out[name]==want,name
    for name,want in p['official_source_blob_sha1'].items():
        raw=(Path(p['official_root'])/name).read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want,name
        out['FT/'+name]=sha(raw)
    for name,want in p['runtime_source_sha256'].items():
        out['native/'+name]=digest(Path(p['isolated_site'])/name);assert out['native/'+name]==want,name
    return out

def timed(kind,fn):
    torch.cuda.synchronize();t=time.perf_counter();row={'kind':kind,'status':'entered'};r['calls'].append(row)
    try:
        out=fn();torch.cuda.synchronize();row['status']='returned';return out
    finally:row['seconds']=time.perf_counter()-t

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

def contraction(m,x):
    assert m.shape==x.shape and m.ndim==3
    return (m.double()*x.double()).sum(-1)

def drift(x,y):
    assert x.shape==y.shape
    assert bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
    d=x.double()-y.double();den=y.double().norm()
    return {'bitwise_equal':bool(torch.equal(x,y)),'max_absolute':float(d.abs().max()),'relative_L2':float(d.norm()/den) if den else None}

def stat(x):
    assert bool(torch.isfinite(x).all())
    return {'net':float(x.sum()),'positive':float(x.clamp_min(0).sum()),'negative':float(x.clamp_max(0).sum())}

class CountFinite:
    def __init__(self,op):self.op=op;self.entered=0;self.returned=0
    def __call__(self,*args,**kw):
        self.entered+=1;out=self.op(*args,**kw);self.returned+=1;return out

try:
    def timeout(*args):raise TimeoutError('Frozen single-layer current diagnostic budget expired.')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(p['budget']['wall_time_seconds']);r['sources_before']=sources()
    import numpy as np
    import torch,flash_attn
    torch.set_num_threads(4)
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    import flash_attn.flash_attn_interface as fa
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import causal_conv1d
    from qwen35_decoder_finite import NativeDecoderCapture,FiniteBoundaryOps,decoder_finite_pullback
    from qwen35_gdn_finite import NativeGDNCapture,gdn_finite_pullback
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    import NI_layer0_boundary_ledger_20260908 as ledger
    assert digest(native.__file__)==p['native_model_sha256'] and digest(fa.__file__)==p['installed_FA_interface_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule and native.is_fast_path_available;verify_native_sources(p['native_stage_source_sha256'])
    assert digest(p['source_results_path'])==p['source_results_sha256']
    prior=json.loads(Path(p['source_results_path']).read_bytes());assert prior['status']=='NI_current_34boundaries_1DT4score_observation_complete'
    private=Path(p['source_private_path']);assert digest(private)==p['source_private_sha256'] and private.stat().st_size==p['source_private_bytes']
    original=torch.load(private,map_location='cpu',weights_only=True)
    saved={'boundaries':{k:original['boundaries'][k] for k in ('0','1')},
           'scoring':{s:{k:original['scoring'][s][k] for k in ('0','1')} for s in ('0','1','10','20')}}
    del original;gc.collect();r['input']=prior['input'];stored['source_boundary0_1']=saved
    cp=Path(p['checkpoint'])
    for name,want in p['checkpoint_config_tokenizer_sha256'].items():assert digest(cp/name)==want
    weight_stats=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
    r['weight_stats_before']=weight_stats();assert r['weight_stats_before']==p['expected_weight_stats']
    assert torch.cuda.mem_get_info()[0]>=32*1024**3
    torch.manual_seed(73)
    model,loading=timed('official_full_model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,
        attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True))
    assert not any(loading.values());model.eval().requires_grad_(False);r['model_loads']=1;model.set_attn_implementation('flash_attention_2')
    def forbid_full(*args,**kw):raise RuntimeError('No whole-model forward authorized in this layer-only study.')
    full_guard=model.register_forward_pre_hook(forbid_full)
    language=model.model.language_model;layer=language.layers[0];assert layer.block_type=='linear_attention'
    assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule and layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
    assert language.config.use_cache is True
    r['kwargs_source']={'model_source_sha256':digest(native.__file__),
        'recurrent_mask_source_sha256':sha(inspect.getsource(native.create_recurrent_attention_mask).encode()),
        'cache_class':native.DynamicCache.__module__+'.'+native.DynamicCache.__name__,
        'cache_class_source_sha256':digest(inspect.getfile(native.DynamicCache)),
        'derivation':'Original runner root use_cache=False gives B2 cache=None. Original evaluator omits use_cache;native text config=True creates fresh DynamicCache for B1. Native recurrent mask helper receives allones,arange text positions and corresponding cache. Decoder linear branch never reads position_embeddings/position_ids;position_embeddings=None deliberately skips unused rotary work.'}
    class FreshCacheCapture(NativeGDNCapture):
        def event(self,frame,kind,value):
            label=self.codes.get(frame.f_code);f=frame.f_locals
            if label=='module' and kind=='call' and f['self'] is self.module:
                assert not self.active and not self.values and not f.get('kwargs',{}).get('cu_seq_lens_q')
                cache=f.get('cache_params');previous=False if cache is None else bool(cache.has_previous_state(self.module.layer_idx));assert not previous
                self.initial_cache_receipt={'provided':cache is not None,'class':type(cache).__name__,'has_previous_state':previous}
                self.active=True;self.values['input']=self.copy(f['hidden_states']);self.values['mask']=self.copy(f.get('attention_mask'))
                self.calls['module']=self.calls.get('module',0)+1;return
            return super().event(frame,kind,value)
    finite_fla=CountFinite(make_compiled_finite_pullback(reuse_scalar_products=False));boundaries=FiniteBoundaryOps(True)
    stored['native']={};upstream=saved['boundaries']['1']['m'].to('cuda').contiguous()
    for name in ('B2','0','1','10','20'):
        xcpu=saved['boundaries']['0']['paired'] if name=='B2' else saved['scoring'][name]['0']
        expected=saved['boundaries']['1']['paired'] if name=='B2' else saved['scoring'][name]['1']
        x=xcpu.to('cuda').contiguous();B,T,H=x.shape;assert x.dtype==torch.bfloat16 and T==r['input']['total_length']
        use_cache=name!='B2';cache=native.DynamicCache(config=language.config) if use_cache else None
        position_ids=torch.arange(T,device=x.device).view(1,T).expand(B,T)
        attention_mask=native.create_recurrent_attention_mask(config=language.config,inputs_embeds=x,
            attention_mask=torch.ones((B,T),device=x.device,dtype=torch.long),past_key_values=cache,position_ids=position_ids)
        kw={'position_embeddings':None,'position_ids':position_ids,'attention_mask':attention_mask,'past_key_values':cache,'use_cache':use_cache}
        dc=NativeDecoderCapture(layer,destination='cuda');mc=FreshCacheCapture(layer.linear_attn,device='cuda')
        def replay():
            with torch.no_grad(),dc,mc:return layer(x,**kw)
        r['layer_replays_entered']+=1;r['status']='native_layer0_'+name;save();y=timed('native_layer0_'+name,replay);r['layer_replays_returned']+=1
        assert dc.calls=={k:1 for k in ('input_norm','post_norm','gate','up','silu','down','mlp','decoder')}
        assert mc.calls=={'module':1,'conv':1,'FLA':1,'stage':1}
        assert mc.initial_cache_receipt['provided']==use_cache and mc.scale is not None
        d,c,e=dc.values,mc.values,mc.endpoints
        stored['native'][name]=ledger.features(d,c,e)
        r['points'][name]={'output_replay_drift':drift(cpu(y),expected),'decoder_calls':dc.calls,'mixer_calls':mc.calls,
            'actual_scale':mc.scale,'initial_cache':mc.initial_cache_receipt,'mask':None if attention_mask is None else {'shape':list(attention_mask.shape),'all_ones':bool(attention_mask.eq(1).all())},
            'native_feature_CPU_bytes':nbytes(stored['native'][name])}
        if name=='B2':
            def mixer(m):return gdn_finite_pullback(layer.linear_attn,c,e,m,mc.scale,finite_fla,True,norm_gate_rule='symmetric')
            r['finite_layer_entered']+=1;r['status']='one_current_layer0_finite';save()
            with torch.no_grad():new,terms=timed('current_layer0_finite_with_existing_diagnostics',lambda:decoder_finite_pullback(layer,d,upstream,mixer,boundaries,True))
            r['finite_layer_returned']+=1;assert finite_fla.entered==finite_fla.returned==1 and bool(torch.isfinite(new).all())
            stored['coeff']=ledger.pack_gdn(upstream,new,terms);r['m0_replay_drift_report_only']=drift(cpu(new),saved['boundaries']['0']['m'])
            stored['m0_new']=cpu(new);r['current_norm_gate_rule']='symmetric';del new,terms
        del x,y,cache,kw,d,c,e,dc,mc;gc.collect();save()
    assert r['layer_replays_entered']==r['layer_replays_returned']==5 and r['finite_layer_entered']==r['finite_layer_returned']==1
    for step in ('1','10','20','B2'):
        if step=='B2':
            f=stored['native']['B2'];change=ledger.delta(ledger.endpoint(f,1),ledger.endpoint(f,0))
            dx=saved['boundaries']['0']['paired'][1:2].double()-saved['boundaries']['0']['paired'][0:1].double()
            dy=saved['boundaries']['1']['paired'][1:2].double()-saved['boundaries']['1']['paired'][0:1].double();deleted=set(r['input']['keep'])
        else:
            change=ledger.delta(stored['native']['0'],stored['native'][step])
            dx=saved['scoring']['0']['0'].double()-saved['scoring'][step]['0'].double()
            dy=saved['scoring']['0']['1'].double()-saved['scoring'][step]['1'].double();deleted=set(prior['points'][step]['input_receipt']['deleted_positions'])
        dec=ledger.decompose(stored['coeff'],change);tokens=ledger.decompose_tokens(stored['coeff'],change)
        original_error=contraction(saved['boundaries']['0']['m'],dx)-contraction(saved['boundaries']['1']['m'],dy)
        transfer={
            'saved_minus_replayed_m0':contraction(saved['boundaries']['0']['m'].double()-stored['m0_new'].double(),dx),
            'input_replay_transfer':contraction(stored['m0_new'],dx-change['d']['input_norm_input']),
            'output_replay_transfer':contraction(saved['boundaries']['1']['m'],change['d']['output']-dy)}
        fields={**tokens['terms'],**transfer,'saved_boundary_error':original_error}
        full=sum(tokens['terms'].values())+sum(transfer.values())
        assert float((full-original_error).abs().max())<1e-7
        if step!='B2':assert abs(float(original_error.sum())-prior['points'][step]['coarse']['regions']['0_to_1'])<1e-7
        P=r['input']['prompt_length'];L=r['input']['total_length'];keep=set(r['input']['keep'])
        groups={'deleted':sorted(deleted),'kept':sorted(keep-deleted),'other_prompt':sorted(set(range(P))-keep),'response':list(range(P,L))}
        report={'replayed_16term_ledger':dec,'saved_boundary_error':stat(original_error),'transfer_terms':{k:stat(v) for k,v in transfer.items()},
            'closure_max_absolute':float((full-original_error).abs().max()),'coordinate_groups':{}}
        for key,value in fields.items():
            vectors[step+'_'+key]=value.numpy();total=stat(value);gs={g:{'count':len(ix),**stat(value[:,ix])} for g,ix in groups.items()}
            assert abs(sum(q['net'] for q in gs.values())-total['net'])<1e-7;report['coordinate_groups'][key]={'total':total,'groups':gs}
        r.setdefault('conditional_ledgers',{})[step]=report
    np.savez_compressed(A/'vectors.npz',**vectors)
    r['sources_after']=sources();r['weight_stats_after']=weight_stats();assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    r['status']='NI_current_layer0_5replay1finite_internal_measurement_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    sys.setprofile(None);signal.alarm(0)
    if full_guard is not None:full_guard.remove()
    r['finite_FLA_counts']={'entered':finite_fla.entered if finite_fla else 0,'returned':finite_fla.returned if finite_fla else 0}
    if stored:
        try:
            artifact=A/'NI1_current_layer0_internal_private.pt';torch.save(stored,artifact)
            r['private_artifact']={'file':artifact.name,'sha256':digest(artifact),'bytes':artifact.stat().st_size,'tensor_CPU_bytes':nbytes(stored)}
        except Exception:r['private_artifact_error']=traceback.format_exc();r['status']='failed'
    r['scope']='Same current layer0 finite rule. Five actual native replays use B2no-cache versus B1fresh-cache conventions. 16localterms refer to replayed operator;explicit input/output/coefficient transfers connect them to saved actual whole-pass boundary error. No newmodelcounterfactual,scoring,metric or method improvement claim.'
    r['cost_scope']='One original full model load;5native layer0replays;1finite decoder0 includes1existing finiteFLA callback with2native input adjoints,existing compiledMLP/norm rules,and1public linear causal-conv preactivation forward+1autograd backward(including discardedweightgrad). All timing includes passively captured tensors;not production speed.'
    r['seconds']=time.perf_counter()-started;save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/name).exists():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'seconds':r['seconds'],'replays':r['layer_replays_returned'],'finite':r['finite_layer_returned'],'error':r.get('error')}),flush=True)
