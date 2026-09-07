"""Two native forwards and bounded operator diagnostics; no attribution or quality sweep."""
import os
os.environ['MACA_PATH'] = '/opt/maca'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import contextlib
import hashlib
import inspect
import json
import signal
import sys
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent
p = json.loads((HERE / 'protocol.json').read_text())
os.environ['TRITON_CACHE_DIR'] = p.get('compiler_cache', str(HERE / 'triton_cache'))
sha = lambda b: hashlib.sha256(b).hexdigest()
assert sha((HERE / 'study.py').read_bytes()) == p['study_sha256']
assert sha((HERE / 'official_fixed_text_inputs.py').read_bytes()) == p['input_runtime_sha256']
sys.path.insert(0, p['author_root'])
r = {'status': 'running', 'protocol': p, 'model_load_attempts': 0, 'model_loads': 0,
     'root_forward_attempts': 0, 'root_forwards_entered': 0, 'root_forwards_completed': 0,
     'quality_queries': 0, 'generation_calls': 0, 'attributions': 0, 'calls': []}
start = time.time()


def save():
    temp = HERE / 'results.partial'
    temp.write_text(json.dumps(r, ensure_ascii=False, separators=(',', ':')))
    temp.replace(HERE / 'results.json')


def source_receipt(scheduled=False):
    site = Path(p['isolated_site'])
    count = 0; changed = []; h = hashlib.sha256()
    for path, digest in p['wheels'].items():
        raw = Path(path).read_bytes(); assert sha(raw) == digest
        with zipfile.ZipFile(__import__('io').BytesIO(raw)) as z:
            for name in z.namelist():
                if name.endswith('.py') and name.startswith(('fla/', 'transformers/')):
                    actual = (site / name).read_bytes(); original = z.read(name)
                    if actual != original:
                        expected = {'fla/utils.py': p['fla_mapped_utils_sha256']}
                        if scheduled: expected['fla/ops/common/chunk_delta_h.py'] = p['fla_scheduled_chunk_sha256']
                        assert name in expected and sha(actual) == expected[name]
                        changed.append(name)
                    h.update(name.encode() + bytes.fromhex(sha(actual))); count += 1
    assert count == 2853 and set(changed) == ({'fla/utils.py', 'fla/ops/common/chunk_delta_h.py'} if scheduled else {'fla/utils.py'})
    return {'files_compared': count, 'explicit_changes': changed, 'source_tree_sha256': h.hexdigest()}


try:
    r['sources_before'] = source_receipt(scheduled=True)
    r['sources_during'] = r['sources_before']
    root = Path(p['author_root'])
    for name, digest in p['author_source_sha256'].items():
        assert sha((root / name).read_bytes().replace(b'\r\n', b'\n')) == digest
    checkpoint = Path(p['checkpoint'])
    r['checkpoint_stats_before'] = {x.name: {'bytes': x.stat().st_size, 'mtime_ns': x.stat().st_mtime_ns}
        for x in checkpoint.glob('*.safetensors')}
    assert {k: v['bytes'] for k, v in r['checkpoint_stats_before'].items()} == p['verified_shard_sizes']
    for name, digest in p['checkpoint_config_tokenizer_sha256'].items():
        assert sha((checkpoint / name).read_bytes()) == digest
    import torch
    import triton
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from fla import utils as fla_utils
    import flash_attn.flash_attn_interface as fa
    import causal_conv1d
    from official_fixed_text_inputs import prepare_fixed_text, right_pad_batch, load_author_preparer
    AuthorTextPreparer = load_author_preparer(p['author_root'], p['author_source_sha256'])
    r['text_preparation_source'] = 'Unchanged hash-verified author LLMAttribution class and two constants extracted from original ASTs; unrelated top-level imports omitted. No model implementation copied or replaced.'
    assert sha(Path(native.__file__).read_bytes()) == p['native_model_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule
    assert native.fused_recurrent_gated_delta_rule is fused_recurrent_gated_delta_rule
    assert native.is_fast_path_available and fla_utils.device_platform == 'maca'
    assert not fla_utils.IS_NVIDIA and not hasattr(torch, 'maca')
    r['versions'] = {'torch': torch.__version__, 'triton': triton.__version__, 'transformers': transformers.__version__,
        'device': torch.cuda.get_device_name(), 'real_triton_backend': triton.runtime.driver.active.get_current_target().backend}
    torch.manual_seed(73)
    r['model_load_attempts'] += 1; save()
    tick = time.perf_counter()
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.bfloat16, attn_implementation='flash_attention_2',
        device_map={'': 'cuda:0'}, local_files_only=True, output_loading_info=True)
    model.eval().requires_grad_(False)
    r['model_loads'] += 1
    r['loading'] = {k: v for k, v in loading.items() if v}
    assert not r['loading'], r['loading']
    r['loading_seconds'] = time.perf_counter() - tick
    assert type(model) is Qwen3_5ForConditionalGeneration
    layers = model.model.language_model.layers
    assert len(layers) == 32
    for layer in layers:
        if layer.block_type == 'linear_attention':
            assert layer.linear_attn.chunk_gated_delta_rule is chunk_gated_delta_rule
            assert layer.linear_attn.causal_conv1d_fn is causal_conv1d.causal_conv1d_fn
        else:
            assert layer.self_attn.config._attn_implementation == 'flash_attention_2'
    original_methods = {n: type(m).forward for n, m in model.named_modules()}

    def method_audit():
        for name, module in model.named_modules():
            assert 'forward' not in module.__dict__, name
            assert type(module).forward is original_methods[name]
            assert getattr(module.forward, '__func__', None) is original_methods[name]
            assert not module._forward_hooks and not module._forward_pre_hooks and not module._backward_hooks

    method_audit()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token  # Exact author run_exp.py policy.
    assert tokenizer.eos_token_id == tokenizer.pad_token_id == 248046
    cases = []
    for dataset, index in p['selection']:
        raw = (root / 'exp/exp2/data' / (dataset + '.jsonl')).read_bytes()
        assert sha(raw) == p['cache_sha256'][dataset]
        ex = json.loads(raw.decode().splitlines()[index])
        engine = AuthorTextPreparer(model, tokenizer)
        case = prepare_fixed_text(engine, ex['prompt'], ex['target'])
        case['metadata'].update(dataset=dataset, index=index)
        case['metadata']['input_ids_sha256'] = sha(case['input_ids'].numpy().tobytes())
        cases.append(case)
        del engine
    r['input_metadata'] = [case['metadata'] for case in cases]
    save()
    assert sha((HERE/'native_batch_diagnostics.py').read_bytes()) == p['diagnostic_runtime_sha256']
    from native_batch_diagnostics import NativeBatchDiagnostics, metrics
    import numpy as np
    parent_raw=(Path(p['required_parent_directory'])/'results.json').read_bytes()
    assert sha(parent_raw)==p['required_parent_raw_sha256']
    prior=json.loads(parent_raw)
    assert r['input_metadata']==prior['input_metadata']
    torch.set_num_threads(4)
    r.update(diagnostics={}, auxiliary_FA_attempts=0, auxiliary_FA_completed=0, CPU_reference_attempts=0, CPU_reference_completed=0)
    diagnostic=NativeBatchDiagnostics(HERE,len(cases[0]['input_ids']),r['diagnostics'])
    code_names={}
    for label,func in [('FLA_chunk',chunk_gated_delta_rule),('FLA_recurrent',fused_recurrent_gated_delta_rule),
        ('FA_dense',fa.flash_attn_func),('FA_varlen',fa.flash_attn_varlen_func),
        ('causal_conv',causal_conv1d.causal_conv1d_fn),('torch_chunk_fallback',native.torch_chunk_gated_delta_rule),
        ('torch_recurrent_fallback',native.torch_recurrent_gated_delta_rule)]:
        code_names[inspect.unwrap(func).__code__]=label
    modules=[('embedding',model.model.language_model.embed_tokens,True)]
    for i,layer in enumerate(layers): modules.append(('layer'+str(i),layer,False))
    first=layers[0]
    for name,module in [('layer0_input_norm',first.input_layernorm),('layer0_qkv_projection',first.linear_attn.in_proj_qkv),
        ('layer0_z_projection',first.linear_attn.in_proj_z),('layer0_a_projection',first.linear_attn.in_proj_a),
        ('layer0_b_projection',first.linear_attn.in_proj_b),('layer0_output_projection',first.linear_attn.out_proj),
        ('layer0_post_attention_norm',first.post_attention_layernorm),('final_norm',model.model.language_model.norm)]:
        modules.append((name,module,True))
    assert len({id(m) for _,m,_ in modules})==len(modules)
    for call_index,selected in enumerate(p['call_selections']):
        assert r['root_forward_attempts']<2
        batch_cases=[cases[i] for i in selected]
        row={'selection':selected,'profiled':False,'status':'attempted','python_dispatch':{},'layer_calls':[]}
        r['calls'].append(row);r['root_forward_attempts']+=1;save()
        counts=Counter();handles=[]
        def python_event(frame,event,arg):
            label=code_names.get(frame.f_code)
            if event=='call' and label:counts[label]+=1
            if event=='return' and arg is not None:
                if label=='FLA_chunk':diagnostic.capture_fla(frame.f_locals,arg,call_index)
                elif label=='FA_dense':diagnostic.capture_fa(frame.f_locals,arg,call_index)
        def root_enter(module,args):r['root_forwards_entered']+=1
        handles.append(model.register_forward_pre_hook(root_enter))
        for i,layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(lambda m,a,_i=i:row['layer_calls'].append(_i)))
        for name,module,persist in modules:
            handles.append(module.register_forward_hook(lambda m,a,o,_n=name,_p=persist:diagnostic.observe(_n,o,call_index,_p)))
        previous=sys.getprofile();assert previous is None
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        try:
            inputs=right_pad_batch(batch_cases,tokenizer.pad_token_id,model.device)
            row['input_shape']=list(inputs['input_ids'].shape)
            row['valid_lengths']=inputs['attention_mask'].sum(-1).tolist()
            sys.setprofile(python_event)
            try:
                with torch.no_grad():outputs=model(**inputs,use_cache=False,logits_to_keep=0,return_dict=True)
                r['root_forwards_completed']+=1
            finally:sys.setprofile(previous)
            assert outputs.past_key_values is None and outputs.logits.dtype==torch.bfloat16
            assert outputs.logits.shape[-1]==248320
            row['target_logprobs']=[]
            for j,case in enumerate(batch_cases):
                plen=case['prompt_length']; target_ids=case['target_ids'].to(model.device)
                logits=outputs.logits[j,plen-1:plen-1+len(target_ids)].float()
                logprob=logits.log_softmax(-1).gather(-1,target_ids[:,None]).squeeze(-1)
                assert bool(torch.isfinite(logprob).all())
                row['target_logprobs'].append(logprob.cpu().tolist())
                del logits,logprob,target_ids
            del outputs,inputs
            torch.cuda.synchronize();row['status']='complete'
        except Exception:
            row['status']='failed';row['error']=traceback.format_exc();raise
        finally:
            sys.setprofile(previous)
            row['diagnostic_seconds']=time.perf_counter()-tick
            row['diagnostic_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            row['python_dispatch']=dict(counts)
            for handle in handles:handle.remove()
            diagnostic.flush_probes(call_index);save()
        assert row['layer_calls']==list(range(32))
        assert counts['FLA_chunk']==counts['causal_conv']==24
        assert counts['FA_dense']+counts['FA_varlen']==8
        assert counts['FLA_recurrent']==counts['torch_chunk_fallback']==counts['torch_recurrent_fallback']==0
        method_audit()
        print('DIAGNOSTIC_FORWARD',call_index,row['input_shape'],row['python_dispatch'],flush=True)
    assert not diagnostic.baseline
    assert diagnostic.fa is not None and all(diagnostic.prefixes)
    r['batch_target_comparison']=metrics(r['calls'][0]['target_logprobs'][0],r['calls'][1]['target_logprobs'][0])
    r['FLA_batch_prefix']={k:metrics(diagnostic.numpy(diagnostic.prefixes[0][k]),diagnostic.numpy(diagnostic.prefixes[1][k]))
                          for k in ['q','k','v','g','beta','output']}
    from fla.ops.common import chunk_delta_h
    schedule=chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    r['selected_state_schedule']=None
    seen=set()
    while id(schedule) not in seen:
        seen.add(id(schedule))
        if hasattr(schedule,'best_config'):
            cfg=schedule.best_config
            r['selected_state_schedule']={'kwargs':cfg.kwargs,'num_warps':cfg.num_warps,'num_stages':cfg.num_stages}
            assert cfg.num_stages==1
            break
        if not hasattr(schedule,'fn'):break
        schedule=schedule.fn
    save()
    # One native varlen call with exactly the captured dense-call Q/K/V values.
    r['auxiliary_FA_attempts']+=1;save()
    operands={k:diagnostic.fa[k].to(model.device) for k in ['q','k','v']}
    q,k,v=[operands[n] for n in ['q','k','v']]
    assert q.shape[0]==k.shape[0]==v.shape[0]==1
    length=q.shape[1];cu=torch.tensor([0,length],dtype=torch.int32,device=model.device)
    torch.cuda.synchronize();tick=time.perf_counter()
    with torch.profiler.profile(activities=list(torch.profiler.supported_activities())) as profiler:
        with torch.no_grad():
            alternate=fa.flash_attn_varlen_func(q[0],k[0],v[0],cu,cu,length,length,
                dropout_p=0.0,softmax_scale=diagnostic.fa['softmax_scale'],causal=True)
        alternate=alternate.detach().cpu().unsqueeze(0)
        torch.cuda.synchronize()
    r['auxiliary_FA_completed']+=1
    r['same_QKV_FA_comparison']={**metrics(diagnostic.numpy(diagnostic.fa['output']),diagnostic.numpy(alternate)),
        'scope':'One actual first full-attention layer NI0 Q/K/V. Original dense output captured in model; one extra native varlen call. No full-model intervention.',
        'diagnostic_seconds_including_profile':time.perf_counter()-tick}
    diagnostic.persist('FA_varlen_same_QKV',{'output':diagnostic.numpy(alternate)})
    trace=HERE/'auxiliary_FA_profile.json';profiler.export_chrome_trace(str(trace))
    r['auxiliary_FA_profile']={'file':trace.name,'sha256':sha(trace.read_bytes())}
    del q,k,v,operands,cu,alternate,profiler
    save()
    # Reference is deliberately run before judging the batch diagnostic gate.
    r['CPU_reference_attempts']+=1;save();tick=time.perf_counter()
    prefix=diagnostic.prefixes[0]
    with torch.no_grad():
        reference,_=native.torch_recurrent_gated_delta_rule(prefix['q'],prefix['k'],prefix['v'],prefix['g'],prefix['beta'],
            initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=True)
    r['CPU_reference_completed']+=1
    r['CPU_reference']={**metrics(diagnostic.numpy(prefix['output']),diagnostic.numpy(reference)),
        'scope':'Official Transformers reference on actual NI0 layer0 first129 tokens, spanning two64-token boundaries. CPU diagnostic only.',
        'relative_L2_denominator':'Captured native FLA output norm; both arrays are saved for alternate normalizations.',
        'seconds':time.perf_counter()-tick}
    diagnostic.persist('FLA_CPU_reference',{'output':diagnostic.numpy(reference)})
    r['sources_after']=source_receipt(scheduled=True);assert r['sources_before']==r['sources_after']
    r['checkpoint_stats_after']={x.name:{'bytes':x.stat().st_size,'mtime_ns':x.stat().st_mtime_ns} for x in checkpoint.glob('*.safetensors')}
    assert r['checkpoint_stats_before']==r['checkpoint_stats_after']
    r['original_batch_screen']={'relative_limit':p['relative_L2_limit'],'absolute_limit':p['max_abs_logprob_limit'],
        'relative_passed':r['batch_target_comparison']['relative_L2']<=p['relative_L2_limit'],
        'absolute_passed':r['batch_target_comparison']['max_abs']<=p['max_abs_logprob_limit']}
    assert r['root_forwards_completed']==r['root_forward_attempts']==2
    assert r['auxiliary_FA_completed']==r['CPU_reference_completed']==1
    method_audit()
    r['status']='bounded_diagnostics_complete_original_gate_reported_separately'
except Exception:
    r['status']='failed';r['error']=traceback.format_exc();raise
finally:
    r['job_seconds']=time.time()-start;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','official_fixed_text_inputs.py','native_batch_diagnostics.py','results.json']+[x.name for x in HERE.glob('*_profile.json')]+[x.name for x in HERE.glob('*.npz')]+[x.name for x in HERE.glob('diagnostic_artifacts.json')]:
            z.write(HERE/name,name)
