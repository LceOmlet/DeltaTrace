"""Existing content_P1 finite propagation on the exact official FT NI0 input.

One resident original model; one B1 eager diagnostic, one B2 EOS/input FA root,
32 original decoder replays and unchanged finite propagation. No FT modification,
old-input activation reuse, target selection, or score calibration.
"""
import os,sys,json,time,hashlib,traceback,zipfile,signal
from pathlib import Path
from collections import Counter
A=Path(__file__).resolve().parent;p=json.loads((A/'protocol.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                  PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=p['compiler_cache'],
                  TORCHINDUCTOR_CACHE_DIR=p['boundary_compiler_cache'])
sys.dont_write_bytecode=True
for overlay in p['dependency_overlays']:sys.path.insert(0,overlay)
sys.path.insert(0,p['official_root'])
sys.path.insert(0,str(A))
r={'status':'running','protocol':p,'model_loads':0,'root_calls':[],
   'decoder_replays':0,'finite_decoder_calls':0,'auxiliary_FA_calls':0,
   'finite_seed_calls':0,'original_needle_calls':0,'generation_calls':0,
   'calls':[],'layers':{},'artifacts':{}}
start=time.perf_counter();handles=[];root={};root_kwargs={}

def save():
    f=A/'results.partial';f.write_text(json.dumps(r,indent=2));f.replace(A/'results.json')

def timed(label,fn):
    torch.cuda.synchronize();tick=time.perf_counter()
    value=fn();torch.cuda.synchronize()
    r['calls'].append({'kind':label,'seconds':time.perf_counter()-tick,
                      'allocated_after':torch.cuda.memory_allocated()});save()
    return value

def copies(value,device):
    if isinstance(value,torch.Tensor):return value.detach().to(device,copy=True)
    if isinstance(value,tuple):return tuple(copies(v,device) for v in value)
    if isinstance(value,list):return [copies(v,device) for v in value]
    if isinstance(value,dict):return {k:copies(v,device) for k,v in value.items()}
    assert value is None or isinstance(value,(str,int,float,bool)),type(value)
    return value

def archive(name,value):
    f=A/(name+'.pt');torch.save(value,f)
    r['artifacts'][f.name]={'sha256':sha(f.read_bytes()),'bytes':f.stat().st_size};save()

def effect(m,x):
    return float((m.double()*(x[1::2].double()-x[0::2].double())).sum())

def differences(a,b):
    a=a.float();b=b.float();delta=b-a
    return {'relative_L2':float(delta.norm()/a.norm().clamp_min(1e-30)),
            'max_abs':float(delta.abs().max())}

def source_receipt():
    result={}
    for name,digest in p['files_sha256'].items():
        actual=sha((A/name).read_bytes());assert actual==digest,name;result[name]=actual
    for name,digest in p['runtime_source_sha256'].items():
        actual=sha((Path(p['isolated_site'])/name).read_bytes());assert actual==digest,name;result[name]=actual
    for name,blob in p['official_package_blob_sha1'].items():
        raw=(Path(p['official_root'])/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==blob,name
        result[name]=sha(raw)
    return result

def time_limit(signum,frame):
    raise TimeoutError('Frozen single-NI0 execution budget expired.')

try:
    signal.signal(signal.SIGALRM,time_limit);signal.alarm(p['budget']['wall_time_seconds'])
    r['sources_before']=source_receipt()
    import numpy as np
    import torch
    import transformers,flash_attn
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    from transformers.integrations.flash_attention import flash_attention_forward
    from flash_attn import flash_attn_func,flash_attn_varlen_func
    from flashtrace.improved import keep_token_indices,evaluate_attr_recovery_skip_tokens
    from native_dense_attention_capture import NativeDenseAttentionCapture
    from qwen35_gdn_finite import NativeGDNCapture,gdn_finite_pullback
    from qwen35_decoder_finite import NativeDecoderCapture,FiniteBoundaryOps,attention_finite_pullback,decoder_finite_pullback
    from qwen35_answer_finite import PackedAnswerTargets,FiniteAnswerOps
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    from vendor_fa_finite_bf16_d256 import RightPaddedLengths,VendorFAFiniteP1BF16D256
    import flash_attn.flash_attn_interface as fa
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import causal_conv1d
    for name in p['files_sha256']:
        if name!='study.py' and name[:-3] in sys.modules:
            assert Path(sys.modules[name[:-3]].__file__).resolve()==(A/name).resolve(),name
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
    assert sha(Path(fa.__file__).read_bytes())==p['installed_FA_interface_sha256']
    verify_native_sources(p['native_stage_source_sha256'])
    cp=Path(p['checkpoint'])
    for name,digest in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==digest
    stats=lambda:{f.name:[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats();assert {k:v[0] for k,v in r['weight_stats_before'].items()}==p['verified_shard_sizes']
    input_raw=Path(p['input_contract_path']).read_bytes();assert sha(input_raw)==p['input_contract_sha256']
    contract=json.loads(input_raw);expected=contract['modes']['current_exp2_default']
    raw=Path(p['official_FT_results_path']).read_bytes();assert sha(raw)==p['official_FT_results_sha256']
    ft=json.loads(raw);assert ft['status']=='unchanged_official_FT_NI0_trace_completed'
    assert ft['actual_input']['sha256']==expected['input_ids_sha256']
    raw=Path(p['cache_path']).read_bytes();assert sha(raw)==p['cache_sha256']
    record=json.loads(raw.decode().splitlines()[0])
    tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True,trust_remote_code=False)
    tokenizer.pad_token=tokenizer.eos_token
    # Exact current official exp2 raw-prompt contract, already measured in FT.
    prompt=tokenizer(record['prompt'],add_special_tokens=False,return_offsets_mapping=True)
    target=tokenizer(record['target'],add_special_tokens=False)['input_ids']+[tokenizer.eos_token_id]
    ids=torch.tensor(prompt['input_ids']+target,dtype=torch.long)
    assert sha(ids.numpy().tobytes())==expected['input_ids_sha256'] and len(ids)==588
    labels=[record['prompt'][a:b] for a,b in prompt['offset_mapping']]
    keep=keep_token_indices(labels);assert keep==contract['current_keep'] and len(keep)==310
    baseline=ids.clone();baseline[keep]=tokenizer.eos_token_id
    assert (baseline!=ids).nonzero().flatten().tolist()==keep
    pairs=torch.stack((baseline,ids)).to('cuda');mask=torch.ones_like(pairs)
    case={'target_ids':torch.tensor(target),'prompt_length':len(prompt['input_ids'])}
    selection=PackedAnswerTargets([case],[list(range(len(target)))],len(ids),'cuda')
    assert selection.counts==[248]
    r['input']={'clean_sha256':sha(ids.numpy().tobytes()),'baseline_sha256':sha(baseline.numpy().tobytes()),
                'prompt_length':case['prompt_length'],'target_length':len(target),'paired_shape':list(pairs.shape),
                'attribution_batch':1,'endpoint_batch':2,'target_scope':'whole fixed response including EOS',
                'baseline':'EOS at original eligible user positions only','padding_tokens':0}
    r['gpu_free_total_before_load']=list(torch.cuda.mem_get_info())
    assert r['gpu_free_total_before_load'][0]>=32*1024**3
    r['status']='loading_actual_model';save();torch.manual_seed(73)
    def load_model():
        return Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,
            attn_implementation='eager',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True)
    model,info=timed('complete_model_load',load_model);model.eval().requires_grad_(False);r['model_loads']=1
    r['nonempty_loading_info']={k:v for k,v in info.items() if v};assert not r['nonempty_loading_info']
    layers=model.model.language_model.layers;head=model.lm_head;norm=model.model.language_model.norm
    assert len(layers)==32
    for layer in layers:
        if layer.block_type=='linear_attention':
            assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
            assert layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
    r['versions']={'torch':torch.__version__,'transformers':transformers.__version__,'flash_attn':flash_attn.__version__,
                   'device':torch.cuda.get_device_name()}
    # Diagnostic matches the actual official FT model's clean B1 eager input.
    def eager_reference():
        with torch.no_grad():out=model(input_ids=pairs[1:2],attention_mask=mask[1:2],use_cache=False)
        z=out.logits[0,selection.positions]
        logp=z.float().log_softmax(-1).gather(-1,selection.labels[:,None]).squeeze(-1).cpu()
        del out,z;return logp
    r['root_calls'].append({'purpose':'common_native_eager_clean_diagnostic','batch':1})
    eager_logp=timed('native_eager_B1_reference',eager_reference)
    model.set_attn_implementation('flash_attention_2')
    assert all(l.self_attn.config._attn_implementation=='flash_attention_2' for l in layers if l.block_type=='full_attention')
    r['status']='executing_new_native_FA_pair';save()
    for i,layer in enumerate(layers):
        def capture(module,args,kwargs,i=i):
            assert str(i) not in root
            x=args[0] if args else kwargs['hidden_states']
            root[str(i)]=copies(x,'cpu')
            root_kwargs[str(i)]=copies({k:v for k,v in kwargs.items() if k!='hidden_states'},'cpu')
        handles.append(layer.register_forward_pre_hook(capture,with_kwargs=True))
    def final_norm(module,args,output):
        root['final_norm_input']=copies(args[0],'cpu');root['final_norm_output']=copies(output,'cpu')
    handles.append(norm.register_forward_hook(final_norm))
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();attribution_start=time.perf_counter()
    r['root_calls'].append({'purpose':'actual_EOS_and_clean_FA_endpoints','batch':2})
    def actual_pair():
        with torch.no_grad():return model(input_ids=pairs,attention_mask=mask,use_cache=False)
    out=timed('native_FA_B2_root_with_CPU_checkpoints',actual_pair)
    for handle in handles:handle.remove()
    handles.clear()
    z=selection.pack_hidden(out.logits);del out
    assert root['0'].shape==(2,588,4096) and len(root)==34
    root_logp=z.float().log_softmax(-1).gather(-1,selection.labels.repeat_interleave(2)[:,None]).squeeze(-1)
    root_delta=float((root_logp[1::2].double()-root_logp[0::2].double()).sum())
    r['root_effect']=root_delta
    r['backend_diagnostic']={'FA_B2_clean_vs_eager_B1_logp':differences(eager_logp,root_logp[1::2].cpu()),
        'eager_clean_sum':float(eager_logp.double().sum()),'FA_clean_sum':float(root_logp[1::2].double().sum()),
        'includes_batch_shape_difference':True,'same_backend_claim':False}
    archive('actual_native_root',{'states':root,'layer_kwargs':root_kwargs,'input_ids':pairs.cpu(),
        'FA_target_logprobs':root_logp.cpu(),'eager_target_logprobs':eager_logp})
    archive('actual_packed_target_logits',z.cpu())
    boundaries=FiniteBoundaryOps(True);answer_ops=FiniteAnswerOps(True)
    r['finite_seed_calls']=1
    with torch.no_grad():mnorm,seed=timed('existing_finite_logprob_seed',lambda:answer_ops(z,head,selection))
    xnorm=root['final_norm_input'].to('cuda')
    with torch.no_grad():m=timed('existing_finite_final_norm',lambda:boundaries.norm_residual(xnorm[0::2],xnorm[1::2],norm.weight,mnorm,torch.zeros_like(mnorm),norm.eps))
    r['seed_effect']=effect(m,xnorm);del mnorm,xnorm,z,root_logp,seed
    layout=RightPaddedLengths([588],588,'cuda')
    finite_fa=VendorFAFiniteP1BF16D256(p['finite_FA_library'],p['finite_FA_library_sha256'])
    finite_fla=make_compiled_finite_pullback(reuse_scalar_products=False)
    for i in reversed(range(32)):
        layer=layers[i];is_fa=layer.block_type=='full_attention'
        x=root[str(i)].to('cuda');kw=copies(root_kwargs[str(i)],'cuda')
        dc=NativeDecoderCapture(layer,destination='cuda')
        mc=(NativeDenseAttentionCapture(layer.self_attn,flash_attention_forward,flash_attn_varlen_func,flash_attn_func,destination='cuda')
            if is_fa else NativeGDNCapture(layer.linear_attn,device='cuda'))
        r['decoder_replays']+=1
        def replay():
            with torch.no_grad(),dc,mc:return layer(x,**kw)
        y=timed('actual_decoder'+str(i)+'_replay',replay)
        assert dc.calls=={k:1 for k in ('input_norm','post_norm','gate','up','silu','down','mlp','decoder')}
        if is_fa:assert mc.calls=={'module':1,'interface':1,'native_varlen':0,'native_dense':1}
        else:assert mc.calls=={'module':1,'conv':1,'FLA':1,'stage':1}
        d,c,e=dc.values,mc.values,getattr(mc,'endpoints',{})
        scale=getattr(mc,'scale',0.0625)
        expected=root['final_norm_input' if i==31 else str(i+1)].to('cuda')
        row={'block_type':layer.block_type,'decoder_calls':dc.calls,'mixer_calls':mc.calls,
             'replay_vs_root':differences(expected,y),'root_output_effect':effect(m,expected),
             'replay_output_effect':effect(m,y)}
        r['layers'][str(i)]=row;del expected,y,x
        lse=None
        if is_fa:
            kwargs={k:v for k,v in mc.dense_arguments.items() if k!='return_attn_probs'}
            assert kwargs['dropout_p']==0 and kwargs['causal'] is True
            r['auxiliary_FA_calls']+=1
            with torch.no_grad():aux,lse,unused=timed('public_dense_FA'+str(i)+'_LSE',
                lambda:flash_attn_func(c['dense_q'],c['dense_k'],c['dense_v'],return_attn_probs=True,**kwargs))
            assert unused is None or unused.numel()==0
            assert lse.shape==(2,16,588)
            row['auxiliary_vs_actual_core']=differences(c['attention_output'],aux)
            del aux,unused
        del dc,mc
        def mixer(upstream):
            if is_fa:
                cos,sin=kw['position_embeddings']
                return attention_finite_pullback(layer.self_attn,c,lse,cos,sin,upstream,finite_fa,layout,boundaries,False)
            return gdn_finite_pullback(layer.linear_attn,c,e,upstream,scale,finite_fla,False)
        r['finite_decoder_calls']+=1
        with torch.no_grad():new_m,_=timed('existing_finite_decoder'+str(i),lambda:decoder_finite_pullback(layer,d,m,mixer,boundaries,False))
        assert torch.isfinite(new_m).all()
        row['input_effect']=effect(new_m,d['input_norm_input'])
        m=new_m;del new_m,d,c,e,lse,kw;save()
    embed=root['0'].to('cuda')
    signed=(m.double()*(embed[1::2].double()-embed[0::2].double())).sum(-1)[0]
    assert torch.isfinite(signed).all()
    assert (embed[1]!=embed[0]).any(-1).nonzero().flatten().tolist()==keep
    assert all(float(signed[j])==0 for j in range(588) if j not in set(keep))
    r['signed_sum']=float(signed.sum());r['unassigned_effect']=root_delta-r['signed_sum']
    r['relative_residual']=r['unassigned_effect']/root_delta if root_delta else None
    torch.cuda.synchronize();r['complete_attribution_seconds_with_diagnostics']=time.perf_counter()-attribution_start
    r['peak_allocated']=torch.cuda.max_memory_allocated();r['peak_reserved']=torch.cuda.max_memory_reserved()
    archive('final_finite_coefficients',m.cpu())
    np.savez_compressed(A/'signed_result.npz',signed=signed.cpu().numpy(),FA_input0=root['0'][0].float().numpy(),
                        FA_input1=root['0'][1].float().numpy(),finite=m[0].cpu().numpy())
    r['artifacts']['signed_result.npz']={'sha256':sha((A/'signed_result.npz').read_bytes()),'bytes':(A/'signed_result.npz').stat().st_size}
    r['status']='signed_vector_saved';save()
    prompt_score=signed[:340].float().cpu();r['original_needle_calls']=1
    recovery=evaluate_attr_recovery_skip_tokens(prompt_score[None],keep_prompt_token_indices=keep,
        gold_prompt_token_indices=contract['current_gold'],top_fraction=0.1)
    selected=[keep[j] for j in torch.topk(prompt_score[keep].clamp(min=0),31).indices.tolist()]
    hits=sorted(set(selected)&set(contract['current_gold']));assert recovery==len(hits)/40
    r['needle']={'recovery':recovery,'selected':selected,'hits':hits,
                 'negative_eligible_count':int((prompt_score[keep]<0).sum()),
                 'positive_eligible_count':int((prompt_score[keep]>0).sum())}
    r['sources_after']=source_receipt();assert r['sources_before']==r['sources_after']
    r['weight_stats_after']=stats();assert r['weight_stats_after']==r['weight_stats_before']
    assert r['decoder_replays']==r['finite_decoder_calls']==32 and r['auxiliary_FA_calls']==8
    r['status']='DT_content_P1_same_official_NI0_input_completed'
except Exception:
    sys.setprofile(None);r['status']='failed';r['error']=traceback.format_exc()
finally:
    signal.alarm(0)
    for handle in handles:handle.remove()
    r['job_seconds']=time.perf_counter()-start;save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in list(p['files_sha256'])+['protocol.json','results.json','signed_result.npz']:
            if (A/name).exists():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'needle':r.get('needle'),'seconds':r['job_seconds'],
                      'finite_decoders':r['finite_decoder_calls'],'error':r.get('error')}),flush=True)
