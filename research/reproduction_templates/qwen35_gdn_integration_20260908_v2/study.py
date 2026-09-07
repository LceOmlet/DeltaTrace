"""One real B4 capture and complete finite GDN propagation; default model unchanged."""
import os
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
import hashlib,inspect,io,json,sys,time,traceback,zipfile
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;p=json.loads((HERE/'protocol.json').read_bytes())
os.environ['TRITON_CACHE_DIR']=p['compiler_cache']
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(HERE/'inductor_cache')
os.environ['OPENBLAS_NUM_THREADS']='1'
sha=lambda b:hashlib.sha256(b).hexdigest()
r={'status':'running','protocol':p,'model_load_attempts':0,'model_loads':0,
   'root_forward_attempts':0,'root_forwards_completed':0,'generation_calls':0,
   'quality_queries':0,'whole_model_attributions':0,'artifacts':[]}
start=time.perf_counter()
def save():
    tmp=HERE/'results.partial';tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2));tmp.replace(HERE/'results.json')
def sources():
    site=Path(p['isolated_site']);count=0;changed=[];h=hashlib.sha256()
    expected={'fla/utils.py':p['fla_mapped_utils_sha256'],
              'fla/ops/common/chunk_delta_h.py':p['fla_scheduled_chunk_sha256'],
              'fla/ops/gated_delta_rule/wy_fast.py':p['wy_backported_sha256']}
    for path,digest in p['wheels'].items():
        raw=Path(path).read_bytes();assert sha(raw)==digest
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for name in z.namelist():
                if not(name.endswith('.py') and name.startswith(('fla/','transformers/'))):continue
                actual=(site/name).read_bytes()
                if actual!=z.read(name):
                    assert name in expected and sha(actual)==expected[name];changed.append(name)
                h.update(name.encode()+bytes.fromhex(sha(actual)));count+=1
    assert count==2853 and set(changed)==set(expected)
    return {'count':count,'changes':changed,'sha256':h.hexdigest()}
try:
    for name,digest in p['files_sha256'].items():assert sha((HERE/name).read_bytes())==digest
    r['sources_before']=sources();assert r['sources_before']['sha256']==p['source_tree_sha256']
    root=Path(p['author_root']);cp=Path(p['checkpoint'])
    for name,digest in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==digest
    stats=lambda:{x.name:[x.stat().st_size,x.stat().st_mtime_ns] for x in cp.glob('*.safetensors')}
    r['weight_stats_before']=stats();assert {k:v[0] for k,v in r['weight_stats_before'].items()}==p['verified_shard_sizes']
    raw=(Path(p['spans_directory'])/'results.json').read_bytes();assert sha(raw)==p['spans_sha256']
    spans=json.loads(raw);assert spans['status']=='official_spans_remapped'
    import numpy as np
    import torch
    import transformers
    import triton
    import importlib
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule,fused_recurrent_gated_delta_rule
    import flash_attn.flash_attn_interface as fa
    import causal_conv1d
    chunk=importlib.import_module('fla.ops.gated_delta_rule.chunk')
    from official_fixed_text_inputs import load_author_preparer,prepare_fixed_text,right_pad_batch
    assert sha(Path(native.__file__).read_bytes())==p['native_model_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule
    assert native.fused_recurrent_gated_delta_rule is fused_recurrent_gated_delta_rule
    assert native.is_fast_path_available
    r['versions']={'torch':torch.__version__,'transformers':transformers.__version__,'triton':triton.__version__,
                   'device':torch.cuda.get_device_name(),'backend':triton.runtime.driver.active.get_current_target().backend}
    assert r['versions']['backend']=='maca'
    r['model_load_attempts']+=1;save();tick=time.perf_counter()
    model,info=Qwen3_5ForConditionalGeneration.from_pretrained(cp,dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True)
    model.eval().requires_grad_(False);r['model_loads']+=1;r['loading_seconds']=time.perf_counter()-tick
    assert not {k:v for k,v in info.items() if v}
    tokenizer=AutoTokenizer.from_pretrained(cp,local_files_only=True,trust_remote_code=False);tokenizer.pad_token=tokenizer.eos_token
    assert tokenizer.eos_token_id==248046
    preparer=load_author_preparer(root,p['author_source_sha256'])
    cases=[];r['input_metadata']=[]
    for c in spans['cases']:
        dataset,index=c['dataset'],c['index']
        raw=(root/'exp/exp2/data'/f'{dataset}.jsonl').read_bytes();assert sha(raw)==p['cache_sha256'][dataset]
        ex=json.loads(raw.decode().splitlines()[index]);engine=preparer(model,tokenizer)
        clean=prepare_fixed_text(engine,ex['prompt'],ex['target']);del engine
        assert sha(clean['input_ids'].numpy().tobytes())==c['input_metadata']['input_ids_sha256']
        base={**clean,'input_ids':clean['input_ids'].clone()}
        eligible=c['mapping']['eligible_input_positions']
        base['input_ids'][eligible]=tokenizer.eos_token_id
        assert torch.equal(base['input_ids'][clean['prompt_length']:],clean['input_ids'][clean['prompt_length']:])
        changed=(base['input_ids']!=clean['input_ids']).nonzero().flatten().tolist()
        assert changed==eligible  # These fixed cases contain no existing EOS in eligible user text.
        cases.extend([base,clean]);r['input_metadata'].append({'dataset':dataset,'index':index,
            'clean':c['input_metadata'],'mapping':c['mapping'],'baseline_input_ids_sha256':sha(base['input_ids'].numpy().tobytes()),
            'changed_input_positions':changed})
    assert len(cases)==4
    layers=model.model.language_model.layers;assert len(layers)==32
    from qwen35_gdn_finite import NativeGDNCapture,gdn_finite_pullback
    from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources
    import fla.modules.fused_norm_gate as fused_norm
    import causal_conv1d.causal_conv1d_interface as conv_interface
    r.update(finite_attempts=0,finite_completed=0,native_GDN_forward_attempts=0,native_GDN_backward_attempts=0,
        finite_calls=[],native_GDN_calls=[],boundary_effects={})
    for name,module in {'gated_norm':fused_norm,'causal_conv':conv_interface}.items():
        assert sha(Path(module.__file__).read_bytes())==p['boundary_source_sha256'][name]
    verify_native_sources(p['native_stage_source_sha256'])
    first=layers[0].linear_attn
    assert type(first.norm) is fused_norm.FusedRMSNormGated and first.norm.activation=='silu'
    original={n:type(m).forward for n,m in model.named_modules()}
    def method_audit():
        for n,m in model.named_modules():
            assert 'forward' not in m.__dict__ and type(m).forward is original[n]
            assert not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks
        for layer in layers:
            if layer.block_type=='linear_attention':
                assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
                assert layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
            else:assert layer.self_attn.config._attn_implementation=='flash_attention_2'
    method_audit()
    native_codes={inspect.unwrap(fn).__code__:label for label,fn in [
        ('FLA',chunk_gated_delta_rule),('FLA_recurrent',fused_recurrent_gated_delta_rule),
        ('FA_dense',fa.flash_attn_func),('FA_varlen',fa.flash_attn_varlen_func),
        ('conv',causal_conv1d.causal_conv1d_fn),('conv_backward',conv_interface.CausalConv1dFn.backward),
        ('fallback_chunk',native.torch_chunk_gated_delta_rule),('fallback_recurrent',native.torch_recurrent_gated_delta_rule)]}
    capture=NativeGDNCapture(first,device='cpu');counts=Counter()
    checkpoints={};handles=[]
    def checkpoint_hook(index):
        def hook(_module,args,kwargs):checkpoints[str(index)]=args[0].detach().cpu()
        return hook
    for index,layer in enumerate(layers):handles.append(layer.register_forward_pre_hook(checkpoint_hook(index),with_kwargs=True))
    def norm_hook(_module,args,output):
        checkpoints['final_norm_input']=args[0].detach().cpu();checkpoints['final_norm_output']=output.detach().cpu()
    handles.append(model.model.language_model.norm.register_forward_hook(norm_hook))
    def event(frame,kind,value):
        if kind=='call' and frame.f_code in native_codes:counts[native_codes[frame.f_code]]+=1
        capture.event(frame,kind,value)
    inputs=right_pad_batch(cases,tokenizer.pad_token_id,model.device)
    r['batch_shape']=list(inputs['input_ids'].shape);r['valid_lengths']=inputs['attention_mask'].sum(-1).tolist()
    r['root_forward_attempts']+=1;save();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
    assert sys.getprofile() is None;sys.setprofile(event)
    try:
        with torch.no_grad():out=model(**inputs,use_cache=False,logits_to_keep=0,return_dict=True)
        r['root_forwards_completed']+=1
    finally:
        sys.setprofile(None)
        for handle in handles:handle.remove()
        handles.clear();torch.cuda.synchronize()
        r['root_diagnostic_seconds']=time.perf_counter()-tick;r['root_diagnostic_peak_bytes']=torch.cuda.max_memory_allocated()
        r['root_dispatch']=dict(counts);save()
    method_audit()
    assert counts['FLA']==counts['conv']==24 and counts['FA_dense']+counts['FA_varlen']==8
    assert not any(counts[k] for k in ['FLA_recurrent','fallback_chunk','fallback_recurrent'])
    assert capture.calls=={'module':1,'conv':1,'FLA':1,'stage':1}
    assert out.logits.shape[-1]==248320 and len(checkpoints)==34
    r['endpoint_target_logprobs']=[]
    for i,case in enumerate(cases):
        target=case['target_ids'].to(model.device);plen=case['prompt_length']
        lp=out.logits[i,plen-1:plen-1+len(target)].float().log_softmax(-1).gather(-1,target[:,None]).squeeze(-1)
        r['endpoint_target_logprobs'].append(lp.detach().cpu().tolist())
    del out,lp,inputs,target;torch.cuda.empty_cache()
    def archive(name,value):
        path=HERE/(name+'.pt');torch.save(value,path)
        r['artifacts'].append({'file':path.name,'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size,'location':'remote_only'});save()
    archive('native_root_checkpoints',checkpoints);del checkpoints
    archive('native_first_GDN_endpoints',{'values':capture.values,'endpoints':capture.endpoints,'scale':capture.scale})
    c={k:(None if v is None else v.to('cuda')) for k,v in capture.values.items()}
    ep={k:v.to('cuda') for k,v in capture.endpoints.items()};scale=capture.scale
    assert c['mask'] is not None and c['mask'].shape==(4,605)
    assert c['mask'].sum(1).tolist()==[605,605,368,368]
    r['GDN_scope']={'input_shape':list(c['input'].shape),'raw_q_shape':list(c['raw_q'].shape),
        'native_q_shape':list(ep['q'].shape),'state_shape':list(ep['h'].shape),'scale':scale,
        'norm_class':str(type(first.norm)),'norm_eps':first.norm.eps,'norm_activation':first.norm.activation}
    seed=c['output'][1::2].contiguous()*c['mask'][1::2,:,None]
    r['cotangent']='Actual input-endpoint GDN output, fixed and zero on padding; local energy pairing, not the answer target.'
    finite_fla=make_compiled_finite_pullback(reuse_scalar_products=False)
    def run_finite(label,cc,ee,diagnostics):
        assert r['finite_attempts']<p['maximum_finite_GDN_calls'];r['finite_attempts']+=1;save()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated();t=time.perf_counter()
        with torch.no_grad():result,terms=gdn_finite_pullback(first,cc,ee,seed,scale,finite_fla,diagnostics=diagnostics)
        torch.cuda.synchronize();elapsed=time.perf_counter()-t
        assert torch.isfinite(result).all()
        r['finite_calls'].append({'phase':label,'seconds':elapsed,'before_bytes':before,'peak_bytes':torch.cuda.max_memory_allocated(),
            'after_bytes':torch.cuda.memory_allocated(),'diagnostics_retained':diagnostics})
        r['finite_completed']+=1;save();return result,terms
    actual,terms=run_finite('cold_actual_endpoints',c,ep,True)
    def array(x):return x.detach().float().cpu().numpy().astype(np.float64)
    def effect(m,x):return (array(m)*array(x)).reshape(m.shape[0],-1).sum(-1)
    def compare(a,b):
        a=np.asarray(a);b=np.asarray(b);den=float(np.linalg.norm(a.ravel()))
        return {'reference':a.tolist(),'allocated':b.tolist(),'residual':(b-a).tolist(),
            'relative_L2':float(np.linalg.norm((b-a).ravel()))/den if den else None}
    def relative(a,b):
        a=array(a);b=array(b);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        return float(np.linalg.norm((b-a).ravel())/np.linalg.norm(a.ravel()))
    diff=lambda x:x[1::2].float()-x[0::2].float()
    record={}
    record['output_projection']=compare(effect(seed,diff(c['output'])),effect(terms['mnorm'],diff(c['norm_output'])))
    record['gated_norm']=compare(effect(terms['mnorm'],diff(c['norm_output'])),
        effect(terms['mo_before_cast'],diff(ep['o']))+effect(terms['mz'],diff(c['z'])))
    record['FLA_upstream_cast']=compare(effect(terms['mo_before_cast'],diff(ep['o'])),effect(terms['mo_native'],diff(ep['o'])))
    for b in range(2):
        obs=(array(terms['mo_native'])[b]*array(diff(ep['o']))[b]).sum((0,2));parts=np.zeros(32)
        for name in ['q','k','v','beta','g']:
            operand='raw_g' if name=='g' else name
            part=array(terms['coeff'][name])[b]*array(diff(ep[operand]))[b]
            parts+=part.sum((0,2) if part.ndim==3 else 0)
        record['FLA_heads_'+str(b)]=compare(obs,parts)
    qraw=c['raw_q'].reshape(4,605,16,2,128)[:,:,:,0]
    kraw=c['raw_k'].reshape(4,605,16,2,128)[:,:,:,0]
    record['L2_q']=compare(effect(terms['coeff']['q'],diff(ep['q'])),effect(terms['mq'],diff(qraw)))
    record['L2_k']=compare(effect(terms['coeff']['k'],diff(ep['k'])),effect(terms['mk'],diff(kraw)))
    record['beta']=compare(effect(terms['coeff']['beta'],diff(ep['beta'])),effect(terms['mb'],diff(c['b'])))
    record['log_decay']=compare(effect(terms['coeff']['g'],diff(ep['raw_g'])),effect(terms['ma'],diff(c['a'])))
    record['conv_silu']=compare(effect(terms['mconv'],diff(c['conv_output'])),effect(terms['mpre'],diff(terms['pre'])))
    record['conv_linear']=compare(effect(terms['mpre'],diff(terms['pre'])),
        effect(terms['mprojected'][1::2],diff(c['projected_qkv'])))
    record['preactivation_equal_but_fused_output_diff_count']=terms['pre_collision'].sum().item()
    record['preactivation_collision_effect']=effect(terms['mconv']*terms['pre_collision'],diff(c['conv_output'])).tolist()
    record['GDN_total']=compare(effect(seed,diff(c['output'])),effect(actual,diff(c['input'])))
    record['padding_coefficient_max_abs']=actual[~c['mask'][1::2].bool()].abs().max().item()
    r['boundary_effects']=record;save();del terms
    actual_cpu=actual.cpu();del actual
    warm,_=run_finite('one_warm_actual_endpoints',c,ep,False)
    r['cold_warm_coeff_relative_L2']=relative(actual_cpu,warm)
    del warm
    # One ordinary native GDN forward/backward on the same actual B4 layer input.
    # Capture this replay's own states for the equal-endpoint degeneration test.
    replay=NativeGDNCapture(first,device='cpu');counts.clear()
    def replay_event(frame,kind,value):
        if kind=='call' and frame.f_code in native_codes:counts[native_codes[frame.f_code]]+=1
        replay.event(frame,kind,value)
    x=c['input'].detach().requires_grad_(True);r['native_GDN_forward_attempts']+=1;save()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();sys.setprofile(replay_event)
    try:
        y=first(x,attention_mask=c['mask'])
        r['native_GDN_backward_attempts']+=1;save()
        dy=torch.zeros_like(y);dy[1::2]=seed
        native_grad,=torch.autograd.grad(y,x,dy)
        torch.cuda.synchronize()
    finally:
        sys.setprofile(None);r['native_GDN_calls'].append({'seconds':time.perf_counter()-t,'peak_bytes':torch.cuda.max_memory_allocated(),
            'scope':'one ordinary GDN forward+backward, passive CPU capture included; not matched to finite-only timing','dispatch':dict(counts)});save()
    assert counts['FLA']==counts['conv']==counts['conv_backward']==1
    assert replay.calls=={'module':1,'conv':1,'FLA':1,'stage':1}
    native_cpu=native_grad[1::2].detach().cpu();del native_grad,x,y,dy
    r['native_replay_output_relative_L2']=relative(c['output'],replay.values['output'])
    duplicate=lambda x:None if x is None else x[1::2].repeat_interleave(2,dim=0).to('cuda')
    equal_c={k:duplicate(v) for k,v in replay.values.items()};equal_ep={k:duplicate(v) for k,v in replay.endpoints.items()}
    equal,_=run_finite('equal_endpoints_native_gradient_limit',equal_c,equal_ep,False)
    def persist(name,values):
        path=HERE/(name+'.npz')
        np.savez_compressed(path,**{k:v.detach().float().cpu().numpy() for k,v in values.items()})
        r['artifacts'].append({'file':path.name,'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size,'location':'review_bundle'});save()
    persist('complete_GDN_coefficients',{'finite':actual_cpu,'equal_finite':equal,'native_gradient':native_cpu,
        'input0':c['input'][0::2],'input1':c['input'][1::2],'output0':c['output'][0::2],'output1':c['output'][1::2],
        'seed':seed,'mask':c['mask'][1::2]})
    r['equal_endpoints_vs_native_relative_L2']=relative(native_cpu,equal)
    r['compiler_counters']={k:dict(v) for k,v in torch._dynamo.utils.counters.items()}
    r['compiler_generated_sources']=[]
    for path in sorted((HERE/'inductor_cache').rglob('*.py')):
        r['compiler_generated_sources'].append({'file':str(path.relative_to(HERE)),'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size})
    method_audit();r['sources_after']=sources();assert r['sources_before']==r['sources_after']
    r['weight_stats_after']=stats();assert r['weight_stats_before']==r['weight_stats_after']
    assert r['finite_attempts']==r['finite_completed']==3
    r['status']='complete_local_GDN_finite_propagation_executed_whole_model_pending'
except Exception:
    sys.setprofile(None)
    for handle in globals().get('handles',[]):handle.remove()
    r['status']='failed';r['error']=traceback.format_exc()
finally:
    r['job_seconds']=time.perf_counter()-start;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        names=list(p['files_sha256'])+['protocol.json','results.json']
        names += [x['file'] for x in r['artifacts'] if x.get('location')=='review_bundle']
        names += [x['file'] for x in r.get('compiler_generated_sources',[])]
        for name in names:z.write(HERE/name,name)
    print(json.dumps({'status':r['status'],'root_forwards':r['root_forwards_completed'],'finite_calls':r.get('finite_completed'),
        'native_GDN_backward_attempts':r.get('native_GDN_backward_attempts'),'error':r.get('error')}),flush=True)
