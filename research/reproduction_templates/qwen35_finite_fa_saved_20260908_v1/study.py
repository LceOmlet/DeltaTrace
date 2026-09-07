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
    # No new decoder construction, weight load, forward, or endpoint capture.
    previous_dir=Path(p['saved_capture_directory']);raw=(previous_dir/'results.json').read_bytes()
    assert sha(raw)==p['saved_capture_result_sha256'];record=json.loads(raw)
    assert record['status']=='failed' and record['finite_attempts']==0
    source=previous_dir/'native_decoder3_attention_capture.pt'
    assert sha(source.read_bytes())==p['saved_capture_sha256']
    saved=torch.load(source,map_location='cpu',weights_only=True)
    B,T,C=saved['decoder_input'].shape;assert (B,T,C)==(4,605,4096)
    mask=saved['mask'].to('cuda');assert mask.sum(1).tolist()==[605,605,368,368]
    c={k:v.to('cuda') for k,v in saved['values'].items()};del saved
    for key in ('decoder_replay_vs_root','decoder_replay_valid_vs_root','capture_calls',
                'actual_interface_arguments','actual_public_FA_arguments','weight_tensor_receipts','official_loading_policy'):
        r[key]=record[key]
    r['inherited_decoder_evidence']={'directory':str(previous_dir),'result_sha256':p['saved_capture_result_sha256'],
        'capture_sha256':p['saved_capture_sha256'],'new_decoder_calls':0}
    original_forward=native.Qwen3_5Attention.forward
    save()
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
    output,lse,unused=result;assert unused is None or unused.numel()==0
    r['public_auxiliary_shapes']={'output':list(output.shape),'LSE':list(lse.shape),'unused_probability_buffer_elements':0 if unused is None else unused.numel(),'unused_probability_buffer_is_None':unused is None}
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
