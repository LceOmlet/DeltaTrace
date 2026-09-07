"""One original decoder replay; public FA LSE/backward; bounded finite-FA checks.

No new full-model pass, generation, quality query, model patch or private FA ABI.
"""
import os
os.environ.update(MACA_PATH='/opt/maca', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1')
import ast, hashlib, io, json, sys, time, traceback, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;p=json.loads((HERE/'protocol.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ['TRITON_CACHE_DIR']=p['compiler_cache']
r={'status':'running','protocol':p,'full_model_loads':0,'full_model_forwards':0,'generation_calls':0,
   'quality_queries':0,'whole_model_attributions':0,'meta_model_constructions':0,'decoder_loads':0,
   'decoder_forward_attempts':0,'public_FA_auxiliary_forward_attempts':0,'public_FA_backward_attempts':0,
   'finite_attempts':0,'calls':[],'artifacts':[]}
start=time.perf_counter()
def save():
    f=HERE/'results.partial';f.write_text(json.dumps(r,indent=2));f.replace(HERE/'results.json')
def archive(name,obj):
    f=HERE/(name+'.pt');torch.save(obj,f)
    r['artifacts'].append({'file':f.name,'sha256':sha(f.read_bytes()),'bytes':f.stat().st_size,'download':False});save()
def persist(name,obj):
    f=HERE/(name+'.npz');np.savez_compressed(f,**{k:v.detach().float().cpu().numpy() for k,v in obj.items()})
    r['artifacts'].append({'file':f.name,'sha256':sha(f.read_bytes()),'bytes':f.stat().st_size,'download':True});save()
def relative(reference,actual):
    if hasattr(reference,'detach'):reference=reference.detach().float().cpu().numpy()
    if hasattr(actual,'detach'):actual=actual.detach().float().cpu().numpy()
    a=np.asarray(reference,dtype=np.float64).ravel();b=np.asarray(actual,dtype=np.float64).ravel()
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    return {'relative_L2':float(np.linalg.norm(b-a))/na if na else None,'max_abs':float(np.abs(b-a).max()),
            'reference_norm':na,'cosine':float(a@b)/(na*nb) if na*nb else None}
def timed(label,fn):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
    e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True)
    t=time.perf_counter();e0.record();out=fn();e1.record();torch.cuda.synchronize()
    r['calls'].append({'kind':label,'seconds':time.perf_counter()-t,'device_elapsed_ms':e0.elapsed_time(e1),
        'before_bytes':before,'peak_bytes':torch.cuda.max_memory_allocated()});save();return out
try:
    for name,digest in p['files_sha256'].items():assert sha((HERE/name).read_bytes())==digest
    parent=Path(p['parent_directory']);raw=(parent/'results.json').read_bytes();assert sha(raw)==p['parent_sha256']
    previous=json.loads(raw)
    fn=next(n for n in ast.parse((parent/'study.py').read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='sources')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<unchanged-native-source-audit>','exec'))
    r['sources_before']=sources();assert r['sources_before']['sha256']==p['source_tree_sha256']
    build=Path(p['build_directory']);raw=(build/'results.json').read_bytes();assert sha(raw)==p['build_sha256']
    built=json.loads(raw);assert built['status']=='finite_extension_compiled_not_executed'
    assert built['vendor_sources_before']==built['vendor_sources_after']
    cp=Path(p['checkpoint'])
    for name,digest in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==digest
    stats=lambda:{x.name:[x.stat().st_size,x.stat().st_mtime_ns] for x in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats();assert r['weight_stats_before']==previous['weight_stats_before']
    f=parent/'native_root_checkpoints.pt';assert sha(f.read_bytes())==p['root_checkpoints_sha256']
    import numpy as np
    import torch
    import transformers
    import flash_attn
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.utils import ContextManagers
    from transformers.core_model_loading import _materialize_copy
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    from transformers.integrations.flash_attention import flash_attention_forward
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input
    from native_attention_capture import NativeAttentionCapture
    from vendor_fa_finite_bf16_d256 import RightPaddedLengths,VendorFAFiniteP1BF16D256
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
    r['versions']={'torch':torch.__version__,'transformers':transformers.__version__,'flash_attn':flash_attn.__version__,
                   'device':torch.cuda.get_device_name()}
    cfg=AutoConfig.from_pretrained(cp,local_files_only=True,trust_remote_code=False)
    cfg._attn_implementation='flash_attention_2';cls=native.Qwen3_5ForConditionalGeneration
    r['meta_model_constructions']+=1;save()
    with ContextManagers(cls.get_init_context(torch.bfloat16,False,False,False)):container=cls(cfg)
    assert container._get_dtype_plan(torch.bfloat16)=={}
    tc=cfg.text_config;assert tc.layer_types[3]=='full_attention' and tc.layer_types[:3]==['linear_attention']*3
    layer=container.model.language_model.layers[3];attn=layer.self_attn
    assert attn.head_dim==256 and attn.num_key_value_groups==4 and attn.scaling==0.0625
    assert tc._attn_implementation=='flash_attention_2'
    desired={k:v.dtype for k,v in layer.state_dict().items()};assert set(desired.values())=={torch.bfloat16}
    rotary_dtype=container.model.language_model.rotary_emb.inv_freq.dtype
    r['official_loading_policy']={'context':'Original get_init_context, metadata only; original _materialize_copy.',
        'dtype_plan':{},'parameter_dtypes':{k:str(v) for k,v in desired.items()},'rotary_buffer_dtype':str(rotary_dtype)}
    index=json.loads((cp/'model.safetensors.index.json').read_bytes())['weight_map'];prefix='model.language_model.layers.3.'
    names={k:v for k,v in index.items() if k.startswith(prefix)};assert {k.removeprefix(prefix) for k in names}==set(desired)
    state={};r['weight_tensor_receipts']={};r['decoder_loads']+=1;save()
    for shard in sorted(set(names.values())):
        with safe_open(cp/shard,framework='pt',device='cpu') as sf:
            for name in names:
                if names[name]!=shard:continue
                raw_value=sf.get_tensor(name);local=name.removeprefix(prefix)
                value=_materialize_copy(sf.get_slice(name),device='cpu',dtype=desired[local]);state[local]=value
                r['weight_tensor_receipts'][name]={'shard':shard,'shape':list(value.shape),'loaded_dtype':str(value.dtype),
                    'checkpoint_dtype':str(raw_value.dtype),'loaded_sha256':sha(value.view(torch.uint8).numpy().tobytes()),
                    'checkpoint_sha256':sha(raw_value.view(torch.uint8).numpy().tobytes())}
    layer.load_state_dict(state,strict=True,assign=True);layer.eval().requires_grad_(False).to('cuda');del state,container
    original_forward=type(attn).forward;assert 'forward' not in attn.__dict__
    saved=torch.load(f,map_location='cpu',weights_only=True);assert len(saved)==34
    x=saved['3'].to('cuda');root_output=saved['4'];del saved
    lengths=[605,605,368,368];B,T,C=x.shape;assert (B,T,C)==(4,605,4096)
    mask=(torch.arange(T,device=x.device)[None,:]<torch.tensor(lengths,device=x.device)[:,None]).long()
    # Exact pure-text/no-cache positional construction from the original TextModel.
    pos=torch.arange(T,device=x.device).view(1,1,T).expand(4,B,T)
    rotary=native.Qwen3_5TextRotaryEmbedding(tc,device=x.device)
    assert rotary.inv_freq.dtype==rotary_dtype
    position_embeddings=rotary(x,pos[1:])
    full_mask=native.create_causal_mask(config=tc,inputs_embeds=x,attention_mask=mask,past_key_values=None,position_ids=pos[0])
    assert torch.equal(full_mask,mask) and position_embeddings[0].shape==(B,T,64)
    interface=native.ALL_ATTENTION_FUNCTIONS.get_interface(tc._attn_implementation,native.eager_attention_forward)
    assert interface is flash_attention_forward
    capture=NativeAttentionCapture(attn,interface,flash_attn_varlen_func)
    def replay():
        with torch.no_grad(),capture:
            return layer(x,position_embeddings=position_embeddings,attention_mask=full_mask,
                         position_ids=pos[0],past_key_values=None,use_cache=False)
    r['decoder_forward_attempts']+=1;save();y=timed('original_decoder3_forward_with_passive_capture',replay)
    r['decoder_replay_vs_root']=relative(root_output,y)
    r['decoder_replay_valid_vs_root']=relative(root_output[mask.cpu().bool()],y[mask.bool()])
    r['capture_calls']=capture.calls;r['actual_interface_arguments']=capture.interface_arguments
    r['actual_public_FA_arguments']=capture.packed_arguments;save()
    assert capture.calls=={'module':1,'interface':1,'native_varlen':1}
    archive('native_decoder3_attention_capture',{'values':capture.values,'decoder_input':x.cpu(),
        'decoder_output':y.cpu(),'mask':mask.cpu(),'cos':position_embeddings[0].cpu(),'sin':position_embeddings[1].cpu()})
    del y,x,root_output,layer,attn,rotary,position_embeddings
    c={k:v.to('cuda') for k,v in capture.values.items()};del capture
    q,k,v=(c['packed_'+name].detach().requires_grad_(True) for name in ('q','k','v'))
    cu=c['packed_cu_seqlens_q'];assert torch.equal(cu,c['packed_cu_seqlens_k'])
    assert cu.tolist()==[0,605,1210,1578,1946]
    indices=mask.flatten().nonzero().flatten()
    for name,value in [('query',q),('key',k),('value',v)]:
        assert torch.equal(value.detach(),c[name].transpose(1,2)[mask.bool()])
    r['public_FA_auxiliary_forward_attempts']+=1;save()
    result=timed('public_FA_varlen_auxiliary_forward_with_LSE',lambda:flash_attn_varlen_func(q,k,v,cu,cu,T,T,
        dropout_p=0.0,softmax_scale=0.0625,causal=True,return_attn_probs=True))
    assert len(result)==3
    output,lse,unused=result;assert unused.numel()==0
    r['public_auxiliary_shapes']={'output':list(output.shape),'LSE':list(lse.shape),'unused_probability_buffer_elements':unused.numel()}
    padded_output=pad_input(output,indices,B,T)
    r['public_FA_output_vs_default_model_FA']=relative(c['attention_output'],padded_output)
    if tuple(lse.shape)==(16,int(cu[-1])):
        padded_lse=pad_input(lse.T,indices,B,T).transpose(1,2).contiguous()
        r['public_LSE_layout']='head,total_valid_tokens; public padding helper restores batch/head/position'
    else:
        assert lse.ndim==3 and tuple(lse.shape[:2])==(B,16) and lse.shape[2]>=T
        padded_lse=lse[:,:,:T].contiguous();r['public_LSE_layout']='batch,head,padded_position'
    archive('native_FA_public_auxiliary',{'LSE':padded_lse.detach().cpu(),'output':padded_output.detach().cpu()})
    seed=c['attention_output'][1::2].transpose(1,2).float().contiguous()
    seed=seed*mask[1::2,None,:,None];effective=seed.to(torch.bfloat16).float()
    ops={name+cidx:c[key][endpoint::2].contiguous() for name,key in [('q','query'),('k','key')]
         for endpoint,cidx in [(0,'0'),(1,'1')]}
    ops.update(v0=c['value'][0::2].contiguous(),u=seed,lse0=padded_lse[0::2].detach().contiguous(),
               lse1=padded_lse[1::2].detach().contiguous())
    layout=RightPaddedLengths([605,368],T,'cuda')
    finite=VendorFAFiniteP1BF16D256(build/'libdeltatrace_fa_finite_bf16_d256.so',built['library']['sha256'])
    def finite_call(label,operands):
        assert r['finite_attempts']<3;r['finite_attempts']+=1;save();activity={}
        out=timed(label,lambda:finite(operands,0.0625,layout,activity));r['calls'][-1]['activity']=activity
        for key,value in out.items():
            assert torch.isfinite(value).all()
            assert torch.count_nonzero(value[1,:,368:])==0
        save();return out
    actual=finite_call('actual_finite_cold',ops)
    warm=finite_call('actual_finite_warm',ops)
    r['cold_warm_vector_differences']={key:relative(actual[key],warm[key]) for key in actual};del warm
    persist('finite_actual',actual)
    groups=4
    summed=lambda out:{'dq':out['dq'].float(),'dk':out['dk'].float().view(2,4,groups,T,256).sum(2),
                       'dv':out['dv'].float().view(2,4,groups,T,256).sum(2)}
    def effects(out):
        delta_q=ops['q1'].double()-ops['q0'].double()
        delta_k=(ops['k1'].double()-ops['k0'].double()).repeat_interleave(groups,dim=1)
        delta_v=(c['value'][1::2].double()-c['value'][0::2].double()).repeat_interleave(groups,dim=1)
        terms={key:(out[key].double()*delta).sum((-1,-2)) for key,delta in [('dq',delta_q),('dk',delta_k),('dv',delta_v)]}
        direct=((c['attention_output'][1::2].double()-c['attention_output'][0::2].double()).transpose(1,2)*effective.double()).sum((-1,-2))
        allocated=sum(terms.values())
        r['actual_finite_effect']={'native_output_per_head':direct.tolist(),'allocated_per_head':allocated.tolist(),
            'terms_per_head':{key:value.tolist() for key,value in terms.items()},
            'per_sample_heads':[relative(direct[i],allocated[i]) for i in range(2)],
            'sample_totals':relative(direct.sum(1),allocated.sum(1))}
    effects(actual);save()
    # Genuine native FA backward on the same actual-input endpoints and U rounding.
    dy=torch.zeros_like(padded_output);dy[1::2]=effective.transpose(1,2).to(dy.dtype)
    packed_dy=dy[mask.bool()].contiguous();r['public_FA_backward_attempts']+=1;save()
    gradients=timed('public_native_FA_backward_actual_endpoints',lambda:torch.autograd.grad(output,(q,k,v),packed_dy))
    native_gradient={name:pad_input(value,indices,B,T)[1::2].transpose(1,2).contiguous()
                     for name,value in zip(('dq','dk','dv'),gradients)}
    equal_ops={**ops,'q0':ops['q1'],'k0':ops['k1'],'v0':c['value'][1::2].contiguous(),'lse0':ops['lse1']}
    equal=finite_call('equal_endpoint_finite_limit',equal_ops);reduced=summed(equal)
    r['equal_endpoint_vs_native']={name:relative(native_gradient[name],reduced[name]) for name in reduced}
    r['padding_max_abs']={'actual':max(float(v[1,:,368:].abs().max()) for v in actual.values()),
        'equal':max(float(v[1,:,368:].abs().max()) for v in equal.values()),
        'native':max(float(v[1,:,368:].abs().max()) for v in native_gradient.values())}
    persist('finite_equal',equal);persist('native_gradients',native_gradient)
    persist('actual_operands',{**ops,'v1':c['value'][1::2],
        'output0':c['attention_output'][0::2].transpose(1,2),'output1':c['attention_output'][1::2].transpose(1,2)})
    r['sources_after']=sources();assert r['sources_after']==r['sources_before']
    r['weight_stats_after']=stats();assert r['weight_stats_after']==r['weight_stats_before']
    assert native.Qwen3_5Attention.forward is original_forward
    r['status']='BF16_D256_right_padded_finite_FA_executed_on_original_decoder_operands'
except Exception:
    sys.setprofile(None);r['status']='failed';r['error']=traceback.format_exc()
finally:
    r['job_seconds']=time.perf_counter()-start;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json']+[a['file'] for a in r['artifacts'] if a['download']]:
            if (HERE/name).exists():z.write(HERE/name,name)
    print(json.dumps({'status':r['status'],'finite_attempts':r['finite_attempts'],'error':r.get('error')}),flush=True)
