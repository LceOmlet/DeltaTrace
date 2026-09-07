"""One real B4 EOS/input forward and native adjoint stages; no shadow model."""
import os
os.environ.update(MACA_PATH='/opt/maca',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
import hashlib,inspect,io,json,sys,time,traceback,zipfile
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;p=json.loads((HERE/'protocol.json').read_bytes())
os.environ['TRITON_CACHE_DIR']=p['compiler_cache']
sha=lambda b:hashlib.sha256(b).hexdigest()
r={'status':'running','protocol':p,'model_load_attempts':0,'model_loads':0,
   'root_forward_attempts':0,'root_forwards_completed':0,'generation_calls':0,
   'quality_queries':0,'whole_model_attributions':0,'adjoint_stage_attempts':0,
   'adjoint_stages_completed':0,'artifacts':[]}
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
    codes={inspect.unwrap(fn).__code__:label for label,fn in [
        ('FLA',chunk_gated_delta_rule),('FLA_recurrent',fused_recurrent_gated_delta_rule),
        ('FA_dense',fa.flash_attn_func),('FA_varlen',fa.flash_attn_varlen_func),('conv',causal_conv1d.causal_conv1d_fn),
        ('fallback_chunk',native.torch_chunk_gated_delta_rule),('fallback_recurrent',native.torch_recurrent_gated_delta_rule),
        ('FLA_forward_stage',chunk.chunk_gated_delta_rule_fwd)]}
    captured={};counts=Counter();prefix=p['prefix_length']
    def event(frame,kind,value):
        label=codes.get(frame.f_code)
        if kind=='call' and label:counts[label]+=1
        if kind=='call' and label=='FLA' and counts[label]==1:
            captured['raw_g']=frame.f_locals['g'][:,:prefix].detach().cpu().clone()
        if kind=='return' and label=='FLA_forward_stage' and counts[label]==1 and value is not None:
            f=frame.f_locals
            assert f['initial_state'] is None and f['cu_seqlens'] is None
            r['native_scale']=float(f['scale'])
            for name in ['q','k','v','g','beta','A','w','v_new','o']:
                captured[name]=f[name][:,:prefix].detach().cpu().clone()
            captured['h']=f['h'][:,:(prefix+63)//64].detach().cpu().clone()
    assert sys.getprofile() is None
    inputs=right_pad_batch(cases,tokenizer.pad_token_id,model.device)
    r['batch_shape']=list(inputs['input_ids'].shape);r['valid_lengths']=inputs['attention_mask'].sum(-1).tolist()
    r['root_forward_attempts']+=1;save();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
    sys.setprofile(event)
    try:
        with torch.no_grad():out=model(**inputs,use_cache=False,logits_to_keep=0,return_dict=True)
        r['root_forwards_completed']+=1
    finally:
        sys.setprofile(None);torch.cuda.synchronize()
        r['forward_diagnostic_seconds']=time.perf_counter()-tick
        r['forward_diagnostic_peak_bytes']=torch.cuda.max_memory_allocated();r['native_dispatch_counts']=dict(counts);save()
    assert counts['FLA']==counts['conv']==counts['FLA_forward_stage']==24
    assert counts['FA_dense']+counts['FA_varlen']==8
    assert not any(counts[k] for k in ['FLA_recurrent','fallback_chunk','fallback_recurrent'])
    assert out.past_key_values is None and out.logits.shape[-1]==248320
    r['endpoint_target_logprobs']=[]
    for i,case in enumerate(cases):
        target=case['target_ids'].to(model.device);plen=case['prompt_length']
        logp=out.logits[i,plen-1:plen-1+len(target)].float().log_softmax(-1).gather(-1,target[:,None]).squeeze(-1)
        assert torch.isfinite(logp).all();r['endpoint_target_logprobs'].append(logp.cpu().tolist())
        del logp,target
    method_audit();del out,inputs,model,layers,original;torch.cuda.empty_cache()
    def persist(name,values):
        file=HERE/(name+'.npz')
        with file.open('wb') as f:np.savez_compressed(f,**{k:v.detach().float().cpu().numpy() for k,v in values.items()})
        r['artifacts'].append({'file':file.name,'sha256':sha(file.read_bytes()),
                              'dtypes':{k:str(v.dtype) for k,v in values.items()},'shapes':{k:list(v.shape) for k,v in values.items()}});save()
    assert captured['q'].shape==(4,prefix,32,128)
    persist('real_paired_FLA_prefix',captured)
    # Real input endpoint1, two independent examples in one native adjoint batch.
    gpu={k:captured[k][[1,3]].to('cuda').contiguous() for k in ['q','k','w','g']}
    do=captured['o'][[1,3]].to('cuda').contiguous()
    r['cotangent']='Captured endpoint1 FLA output, held fixed; local J=<O,Z>, not whole-model answer attribution.'
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
    profiler=None
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
            with torch.no_grad():
                r['adjoint_stage_attempts']+=1;save()
                dvlocal=chunk.chunk_bwd_dv_local(q=gpu['q'],k=gpu['k'],g=gpu['g'],do=do,scale=r['native_scale'],cu_seqlens=None)
                r['adjoint_stages_completed']+=1;r['adjoint_stage_attempts']+=1;save()
                dh,dh0,dU=chunk.chunk_gated_delta_rule_bwd_dhu(**gpu,h0=None,dht=None,do=do,dv=dvlocal,
                    scale=r['native_scale'],cu_seqlens=None)
                r['adjoint_stages_completed']+=1
            torch.cuda.synchronize()
        assert dh0 is None
        persist('native_input_adjoints',{'do':do,'dh_end':dh,'dU_WY':dU})
    finally:
        r['adjoint_diagnostic_seconds']=time.perf_counter()-tick
        r['adjoint_diagnostic_peak_bytes']=torch.cuda.max_memory_allocated()
        if profiler is not None:
            path=HERE/'native_adjoint_profile.json';profiler.export_chrome_trace(str(path))
            raw=path.read_bytes();events=json.loads(raw)['traceEvents']
            kernels=Counter(e['name'] for e in events if e.get('cat')=='kernel')
            r['native_adjoint_profile']={'file':path.name,'sha256':sha(raw),'kernel_counts':dict(kernels)}
        save()
    r['sources_after']=sources();assert r['sources_before']==r['sources_after']
    r['weight_stats_after']=stats();assert r['weight_stats_before']==r['weight_stats_after']
    r['status']='real_paired_endpoints_and_native_adjoints_captured'
except Exception:
    sys.setprofile(None);r['status']='failed';r['error']=traceback.format_exc()
finally:
    r['job_seconds']=time.perf_counter()-start;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','official_fixed_text_inputs.py','results.json']+[x.name for x in HERE.glob('*.npz')]:z.write(HERE/name,name)
    print(json.dumps({'status':r['status'],'root_forwards':r['root_forwards_completed'],'adjoints':r['adjoint_stages_completed'],'error':r.get('error')}),flush=True)
