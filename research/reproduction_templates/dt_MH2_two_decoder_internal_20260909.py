"""Actual MH2 FA19/GDN1 replay ledger; saved upstream, no full DT or scorer."""
import os,sys,json,time,hashlib,traceback,zipfile,signal,gc
from pathlib import Path
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
    TRITON_CACHE_DIR=p['compiler_cache'],TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root']);sys.path.insert(0,str(A))
start=time.perf_counter();handles=[];stored={};vectors={};finite_fa=finite_fla=None
r={'status':'starting','protocol':p,'model_loads':0,'whole_entered':0,'whole_returned':0,'replay_entered':0,'replay_returned':0,
   'finite_decoder_entered':0,'finite_decoder_returned':0,'aux_FA_entered':0,'aux_FA_returned':0,'DT_calls':0,'scorer_calls':0,'FT_calls':0,'generation_calls':0,'calls':[],'layers':{}}
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save():
    q=A/'results.partial';q.write_text(json.dumps(r,indent=2,allow_nan=False));q.replace(A/'results.json')
def timed(kind,fn):
    torch.cuda.synchronize();t=time.perf_counter();row={'kind':kind,'status':'entered'};r['calls'].append(row)
    try:value=fn();torch.cuda.synchronize();row['status']='returned';return value
    finally:row['seconds']=time.perf_counter()-t
def sources():
    out={}
    for n,want in p['files_sha256'].items():out[n]=digest(A/n);assert out[n]==want
    for n,want in p['runtime_source_sha256'].items():out['native/'+n]=digest(Path(p['isolated_site'])/n);assert out['native/'+n]==want
    for n,want in p['official_source_blob_sha1'].items():
        b=(Path(p['official_root'])/n).read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==want
        out['official/'+n]=hashlib.sha256(b).hexdigest()
    for item in p['protected_sources']:out[item['path']]=digest(item['path']);assert out[item['path']]==item['sha256']
    return out
def copy_to(x,device):
    if isinstance(x,torch.Tensor):return x.detach().to(device,copy=True)
    if isinstance(x,dict):return {k:copy_to(v,device) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return type(x)(copy_to(v,device) for v in x)
    assert x is None or isinstance(x,(str,int,float,bool)),type(x)
    return x
cpu=lambda x:copy_to(x,'cpu')
def drift(x,y):
    assert x.shape==y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
    return {'equal':bool(torch.equal(x,y)),'max_absolute':float((x.double()-y.double()).abs().max()),
        'relative_L2':float((x.double()-y.double()).norm()/y.double().norm().clamp_min(1e-30))}
def dot(m,x):assert m.shape==x.shape;return float((m.double()*x.double()).sum())
def token_dot(m,x):assert m.shape==x.shape;return (m.double()*x.double()).flatten(2).sum(-1)
class CountOne:
    def __init__(self,op):self.op=op;self.entered=self.returned=0
    def __call__(self,*args,**kw):
        assert self.entered<1;self.entered+=1;v=self.op(*args,**kw);self.returned+=1;return v
try:
    def timeout(*args):raise TimeoutError('Frozen two-decoder replay budget expired.')
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
    from qwen35_gdn_finite import gdn_finite_pullback
    from native_dense_attention_capture import NativeDenseAttentionCapture
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256,RightPaddedLengths
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    import decoder19_conditional_decomposition_20260908 as fa_diag
    import existing_two_decoder_ledger_20260909 as gd
    assert p['selected_layers']==[19,1] and p['capture_steps']==['B2','0','3','10','20']
    assert digest(native.__file__)==p['native_model_sha256'] and digest(fa.__file__)==p['installed_FA_interface_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule and native.is_fast_path_available
    verify_native_sources(p['native_stage_source_sha256'])
    src=p['source_boundary'];prior=json.loads(Path(src['results_path']).read_bytes());assert digest(src['results_path'])==src['results_sha256']
    assert prior['status']=='MH2_current_conditional_boundaries_1DT4score_complete'
    case=prior['cases']['morehopqa_2'];assert case['input']==p['input'];r['input']=case['input'];T=r['input']['total_length']
    assert digest(src['private_path'])==src['private_sha256'] and Path(src['private_path']).stat().st_size==src['private_bytes']
    original=torch.load(src['private_path'],map_location='cpu',mmap=True,weights_only=True)
    assert original['source_protocol_sha256']==src['protocol_sha256']
    names=['1','2','19','20']
    saved={'coeff':{b:original['coefficients'][b] for b in names},'paired':{b:original['paired_native_boundaries'][b] for b in names},
        'B1':{s:{b:original['B1_native_boundaries'][s][b] for b in names} for s in ['0','3','10','20']}}
    stored['source_boundaries']=saved;del original;gc.collect()
    ids=torch.tensor(prior['input_freeze_before_model_load']['morehopqa_2']['input_ids'],dtype=torch.long)
    assert hashlib.sha256(ids.numpy().tobytes()).hexdigest()==r['input']['input_sha256']
    base=ids.clone();base[r['input']['keep']]=ids[-1];pairs=torch.stack((base,ids)).to('cuda')
    cp=Path(p['checkpoint'])
    for n,want in p['checkpoint_config_tokenizer_sha256'].items():assert digest(cp/n)==want
    weights=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
    r['weight_stats_before']=weights();assert r['weight_stats_before']==p['expected_weight_stats']
    assert torch.cuda.mem_get_info()[0]>=32*1024**3;torch.manual_seed(73)
    model,loading=timed('official_model_load',lambda:Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True))
    assert not any(loading.values());model.eval().requires_grad_(False);model.set_attn_implementation('flash_attention_2');r['model_loads']=1
    language=model.model.language_model;layers=language.layers;assert layers[19].block_type=='full_attention' and layers[1].block_type=='linear_attention'
    captured={};root_counts={}
    def before_root(mod,args,kw):
        assert r['whole_entered']<1 and torch.equal(kw['input_ids'],pairs) and torch.all(kw['attention_mask']==1) and kw['use_cache'] is False and kw['logits_to_keep']==1;r['whole_entered']+=1
    handles.append(model.register_forward_pre_hook(before_root,with_kwargs=True))
    for i,l in enumerate(layers):
        if l.block_type=='linear_attention':assert l.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule and l.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
        def pre(mod,args,kw,i=i):
            root_counts[str(i)]=root_counts.get(str(i),0)+1
            if i in p['selected_layers']:captured[str(i)]={'kwargs':cpu({k:v for k,v in kw.items() if k!='hidden_states'}),'input':cpu(args[0] if args else kw['hidden_states'])}
        def post(mod,args,out,i=i):
            if i in p['selected_layers']:captured[str(i)]['output']=cpu(out)
        handles.extend([l.register_forward_pre_hook(pre,with_kwargs=True),l.register_forward_hook(post)])
    with torch.no_grad():out=timed('one_official_B2_for_native_kwargs',lambda:model(input_ids=pairs,attention_mask=torch.ones_like(pairs),use_cache=False,logits_to_keep=1))
    assert out.logits.shape[:2]==(2,1);r['whole_returned']+=1
    for h in handles:h.remove()
    handles=[];assert root_counts=={str(i):1 for i in range(32)};r['root_layer_counts']=root_counts
    del out,pairs;gc.collect()
    def forbid(*a,**kw):raise RuntimeError('No additional whole forward permitted.')
    handles=[model.register_forward_pre_hook(forbid)]
    boundaries=FiniteBoundaryOps(True)
    finite_fa=CountOne(VendorFAFiniteP1BF16D256(p['finite_FA_library'],p['finite_FA_library_sha256']))
    finite_fla=CountOne(make_compiled_finite_pullback(reuse_scalar_products=False))
    for index in p['selected_layers']:
        key=str(index);nxt=str(index+1);layer=layers[index];is_fa=index==19;cap=captured[key];kw2=cap['kwargs']
        assert kw2['past_key_values'] is None and kw2['use_cache'] is False
        pos=kw2['position_ids'];cos,sin=kw2['position_embeddings'];assert pos.shape==(2,T) and cos.shape[:2]==sin.shape[:2]==(2,T)
        assert torch.equal(pos[0],pos[1]) and torch.equal(cos[0],cos[1]) and torch.equal(sin[0],sin[1])
        mask=kw2['attention_mask'];assert mask is None or mask.shape==(2,T) and torch.all(mask==1)
        kw1=dict(kw2,position_ids=pos[1:2],position_embeddings=(cos[1:2],sin[1:2]),attention_mask=None if mask is None else mask[1:2],use_cache=True)
        lr={'kind':layer.block_type,'root_capture_input_drift':drift(cap['input'],saved['paired'][key]),'root_capture_output_drift':drift(cap['output'],saved['paired'][nxt]),
            'native_kwargs_keys':list(kw2),'actual_mask_shape':None if mask is None else list(mask.shape),'points':{},'ledgers':{}};r['layers'][key]=lr
        st={'official_B2_kwargs':kw2,'B1_kwargs_without_cache':kw1,'native':{}};stored[key]=st;upstream=saved['coeff'][nxt].to('cuda')
        for name in p['capture_steps']:
            xcpu=saved['paired'][key] if name=='B2' else saved['B1'][name][key]
            expected=saved['paired'][nxt] if name=='B2' else saved['B1'][name][nxt]
            kw=copy_to(kw2 if name=='B2' else kw1,'cuda');cache=None if name=='B2' else native.DynamicCache(config=language.config);kw['past_key_values']=cache
            x=xcpu.to('cuda');dc=NativeDecoderCapture(layer,destination='cuda')
            mc=NativeDenseAttentionCapture(layer.self_attn,flash_attention_forward,flash_attn_varlen_func,flash_attn_func,destination='cuda') if is_fa else gd.NativeFreshCacheGDNCapture(layer.linear_attn,device='cuda')
            def replay():
                with torch.no_grad(),dc,mc:return layer(x,**kw)
            assert r['replay_entered']<10;r['replay_entered']+=1;r['status']=f'replay_{index}_{name}';save()
            y=timed(f'native_layer{index}_{name}',replay);r['replay_returned']+=1
            assert dc.calls=={k:1 for k in ['input_norm','post_norm','gate','up','silu','down','mlp','decoder']}
            assert mc.calls==({'module':1,'interface':1,'native_varlen':0,'native_dense':1} if is_fa else {'module':1,'conv':1,'FLA':1,'stage':1})
            d,c=dc.values,mc.values;e={} if is_fa else mc.endpoints
            feature=gd.features(index,d,c,e);st['native'][name]=feature
            lr['points'][name]={'output_replay_drift':drift(cpu(y),expected),'decoder_calls':dc.calls,'mixer_calls':mc.calls,
                'initial_cache':getattr(mc,'initial_cache_receipt',{'provided':cache is not None}), 'feature_bytes':gd.nbytes(feature)}
            if name=='B2':
                if is_fa:
                    args={k:v for k,v in mc.dense_arguments.items() if k!='return_attn_probs'};assert args['dropout_p']==0 and args['causal']
                    r['aux_FA_entered']+=1
                    with torch.no_grad():aux,lse,unused=timed('public_FA_LSE19',lambda:flash_attn_func(c['dense_q'],c['dense_k'],c['dense_v'],return_attn_probs=True,**args))
                    r['aux_FA_returned']+=1;assert unused is None or not unused.numel();lr['FA_auxiliary_drift']=drift(cpu(aux),feature['c']['attention_output']);del aux,unused
                    layout=RightPaddedLengths([T],T,'cuda');cg,sg=kw['position_embeddings']
                    def mixer(m):return attention_finite_pullback(layer.self_attn,c,lse,cg,sg,m,finite_fa,layout,boundaries,True)
                else:
                    def mixer(m):return gdn_finite_pullback(layer.linear_attn,c,e,m,mc.scale,finite_fla,True)
                assert r['finite_decoder_entered']<2;r['finite_decoder_entered']+=1
                with torch.no_grad():new,terms=timed(f'finite_decoder{index}',lambda:decoder_finite_pullback(layer,d,upstream,mixer,boundaries,True))
                r['finite_decoder_returned']+=1;assert torch.isfinite(new).all()
                st['coeff']=fa_diag.pack_coeff(upstream,new,terms) if is_fa else gd.pack_gdn(upstream,new,terms)
                st['new_input_coeff']=cpu(new);lr['saved_input_coeff_drift']=drift(st['new_input_coeff'],saved['coeff'][key])
                if is_fa:lr['FA_activity']=terms['mixer']['finite_FA_activity'];del lse
                del new,terms
            del x,y,kw,cache,d,c,e,dc,mc;gc.collect();save()
        for step in ['3','10','20','B2']:
            paired=step=='B2';change=gd.delta(gd.endpoint(st['native']['B2'],1),gd.endpoint(st['native']['B2'],0)) if paired else gd.delta(st['native']['0'],st['native'][step])
            dx=saved['paired'][key][1:2].double()-saved['paired'][key][0:1].double() if paired else saved['B1']['0'][key].double()-saved['B1'][step][key].double()
            dy=saved['paired'][nxt][1:2].double()-saved['paired'][nxt][0:1].double() if paired else saved['B1']['0'][nxt].double()-saved['B1'][step][nxt].double()
            dec=fa_diag.decompose(st['coeff'],change) if is_fa else gd.gdn_decompose(st['coeff'],change)
            transfer={'saved_minus_replayed_coeff':token_dot(saved['coeff'][key].double()-st['new_input_coeff'].double(),dx),
                'input_replay_transfer':token_dot(st['new_input_coeff'],dx-change['d']['input_norm_input']),
                'output_replay_transfer':token_dot(saved['coeff'][nxt],change['d']['output']-dy)}
            original=token_dot(saved['coeff'][key],dx)-token_dot(saved['coeff'][nxt],dy)
            closure=sum(dec['terms'].values())+sum(float(x.sum()) for x in transfer.values())-float(original.sum());assert abs(closure)<1e-7
            target=-(prior['runs'][0]['B2_endpoint_boundary_contractions'][nxt]-prior['runs'][0]['B2_endpoint_boundary_contractions'][key]) if paired else -case['decomposition'][step]['decoder_errors'][key]
            assert abs(float(original.sum())-target)<1e-7
            groups={'deleted':list(r['input']['keep']) if paired else case['points'][step]['input_receipt']['deleted_positions'],
                'response':list(range(r['input']['prompt_length'],T))}
            groups['other_prompt']=sorted(set(range(r['input']['prompt_length']))-set(groups['deleted']))
            group_terms={}
            if is_fa:group_terms=fa_diag.decompose(st['coeff'],change,groups=groups)['token_groups']
            else:
                # Reuse scalar GDN ledger on disjoint token slices; no operator calls.
                used_e=('q','k','v','beta','raw_g','o')
                def sliced(x,ix,path=()):
                    if isinstance(x,torch.Tensor):
                        if path in [('mconv',),('mprojected',),('c','conv_output'),('c','projected_qkv')]:
                            assert x.ndim==3 and x.shape[0]==1 and x.shape[2]==T
                            return x[:,:,ix]
                        assert x.shape[:2]==(1,T),(path,x.shape)
                        return x[:,ix]
                    return {k:sliced(v,ix,path+(k,)) for k,v in x.items()}
                for label,ix in groups.items():
                    small={'d':change['d'],'c':change['c'],'e':{k:change['e'][k] for k in used_e}}
                    group_terms[label]=gd.gdn_decompose(sliced(st['coeff'],ix),sliced(small,ix))
            for term,value in dec['terms'].items():assert abs(sum(g['terms'][term] for g in group_terms.values())-value)<1e-7
            lr['ledgers'][step]={'internal':dec,'saved_boundary_error':float(original.sum()),'transfers':{k:float(v.sum()) for k,v in transfer.items()},'closure':closure,'groups':group_terms}
            for n,v in {**transfer,'saved_boundary_error':original}.items():vectors[f'{index}_{step}_{n}']=v.numpy()
        del upstream;gc.collect()
    assert r['replay_returned']==10 and r['finite_decoder_returned']==2 and r['aux_FA_returned']==1
    assert finite_fa.returned==finite_fla.returned==1
    np.savez_compressed(A/'vectors.npz',**vectors);r['vectors_sha256']=digest(A/'vectors.npz')
    r['sources_after']=sources();r['weight_stats_after']=weights();assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    r['status']='MH2_FA19_GDN1_1native10replay2finite_internal_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    signal.alarm(0)
    for h in handles:h.remove()
    r['finite_counts']={name:{'entered':op.entered if op else 0,'returned':op.returned if op else 0} for name,op in [('FA',finite_fa),('FLA',finite_fla)]}
    r['native_counts_from_returned_calls']={'root_FA':8*r['whole_returned'],'root_FLA':24*r['whole_returned'],'root_conv':24*r['whole_returned'],
        'selected_FA_replays':len(r['layers'].get('19',{}).get('points',{})),'selected_FLA_and_conv_replays':len(r['layers'].get('1',{}).get('points',{})),
        'finite_FLA_adjoint_stages':2*(finite_fla.returned if finite_fla else 0),'finite_GDN_conv_preactivation_and_autograd_each':1 if r['layers'].get('1',{}).get('saved_input_coeff_drift') is not None else 0}
    if stored:
        try:
            f=A/'MH2_two_decoder_internal_private.pt';t=time.perf_counter();torch.save(stored,f);r['private_artifact']={'file':f.name,'sha256':digest(f),'bytes':f.stat().st_size,'save_seconds':time.perf_counter()-t}
        except Exception:r['status']='failed';r['private_save_error']=traceback.format_exc()
    r['seconds']=time.perf_counter()-start;save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/n).exists():z.write(A/n,n)
    print(json.dumps({'status':r['status'],'seconds':r['seconds'],'error':r.get('error'),'finite_counts':r['finite_counts']}),flush=True)
